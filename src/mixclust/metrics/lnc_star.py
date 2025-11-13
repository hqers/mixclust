# src/mixclust/metrics/lnc_star.py
from __future__ import annotations
import numpy as np
from typing import Optional, Sequence
from sklearn.preprocessing import LabelEncoder

from mixclust.gower import rerank_gower_from_candidates
from mixclust.knn_index import KNNIndex


def _default_k(n: int) -> int:
    """
    Pilihan default k yang stabil untuk LNC*.
    """
    if n <= 50:
        return max(5, n // 3)
    if n <= 200:
        return 20
    return min(50, int(np.sqrt(n)))


def lnc_star(
    X_unit: np.ndarray,                 # (n, d_unit), fitur unit-norm utk ANN cosine
    labels: Sequence,                   # (n,), label klaster (bisa str / int)
    L_idx: Sequence[int],               # indeks landmark global (|L|)
    knn_index: KNNIndex,                # index ANN cosine → kandidat tetangga
    k: Optional[int] = None,            # k tetangga final (di bawah metrik Gower)
    alpha: float = 0.7,                 # bobot konsistensi tetangga vs kontras jarak
    *,
    # Komponen Gower:
    X_num: Optional[np.ndarray] = None,     # (n, p_num) float32
    X_cat: Optional[np.ndarray] = None,     # (n, p_cat) int32
    num_min: Optional[np.ndarray] = None,
    num_max: Optional[np.ndarray] = None,
    feature_mask_num: Optional[np.ndarray] = None,
    feature_mask_cat: Optional[np.ndarray] = None,
    inv_rng: Optional[np.ndarray] = None,
    # Kandidat dari ANN sebelum re-rank Gower:
    M_candidates: int = 300,
    # Agregasi akhir (opsi eksperimen):
    use_weighted_mean: bool = True,         # bobot oleh ukuran klaster
) -> float:
    """
    LNC* (Local Neighbour Consistency with distance Contrast):
    - Ambil kandidat tetangga via ANN cosine (cepat) → ukuran M_candidates.
    - Re-rank kandidat dengan Gower → ambil k tetangga final (akurat).
    - Skor tiap landmark = alpha * NC + (1 - alpha) * Delta,
        * NC: 1 - (H_MillerMadow / log(K))   (konsistensi label tetangga)
        * Delta: (mean(d_inter) - mean(d_intra)) / IQR(nn_d)   (kontras jarak)
      dibatasi [0,1].
    - Skor global = rata-rata (atau mean berbobot) atas semua landmark.

    Catatan:
    - Tidak pernah menghitung matriks jarak penuh; re-use re-ranking per landmark.
    - Robust utk cluster kecil (handle kasus intra/inter kosong).
    """
    labels = np.asarray(labels)
    n = labels.size
    if n == 0 or len(L_idx) == 0:
        return np.nan

    # Normalisasi label ke [0..K-1]
    if labels.dtype.kind in "iu":
        lab_num = labels.astype(int, copy=False)
    else:
        lab_num = LabelEncoder().fit_transform(labels)
    K = int(lab_num.max()) + 1 if lab_num.size else 1

    # Default k
    k = int(_default_k(n) if k is None else k)
    k = max(5, min(k, max(5, n - 1)))  # jaga rentang

    # Kandidat minimum harus >= k
    M = int(max(M_candidates, k))
    # upper-bound kandidat agar tak berlebihan
    M = min(M, max(50, int(0.05 * n)))  # ≤ 5% n, tapi ≥ 50

    # Bobot by cluster size
    if use_weighted_mean:
        # hitung ukuran klaster
        _, counts = np.unique(lab_num, return_counts=True)
        # map cepat label -> size
        size_map = np.zeros(K, dtype=float)
        size_map[:len(counts)] = counts
    else:
        size_map = None

    vals = []
    weights = []

    L = np.asarray(L_idx, dtype=int)
    for i in L:
        # === 1) Kandidat via ANN cosine ===
        # (KNNIndex mengembalikan 1 baris utk single query, tapi disederhanakan oleh wrapper)
        cand_idx, _ = knn_index.kneighbors_idx_dist(int(i), int(M))
        # Pastikan 1D array (beberapa backend memberi shape (M,))
        cand_idx = np.asarray(cand_idx).reshape(-1)
        # Buang duplikat & diri sendiri (kalau ada)
        cand_idx = cand_idx[cand_idx != int(i)]
        if cand_idx.size == 0:
            continue

        # === 2) Re-rank kandidat dgn Gower → ambil kNN final ===
        nn_idx, nn_d = rerank_gower_from_candidates(
            int(i), cand_idx, int(k),
            X_num, X_cat, num_min, num_max,
            feature_mask_num=feature_mask_num,
            feature_mask_cat=feature_mask_cat,
            inv_rng=inv_rng,
        )
        if nn_idx.size == 0:
            continue

        # === 3) Komponen NC (entropy Miller–Madow) ===
        nlab = lab_num[nn_idx]
        counts_k = np.bincount(nlab, minlength=K).astype(float)  # length K
        p = counts_k / max(1.0, counts_k.sum())
        nz = p > 0
        H = float(-np.sum(p[nz] * np.log(p[nz] + 1e-12)))
        m = int(nz.sum())
        # Miller–Madow correction
        H += (m - 1) / (2.0 * max(1, nn_idx.size))
        NC = 1.0 if K <= 1 else float(1.0 - H / np.log(K))
        NC = float(np.clip(NC, 0.0, 1.0))

        # === 4) Komponen Delta (kontras jarak, ternormalisasi IQR) ===
        li = lab_num[i]
        intra = nn_d[nlab == li]
        inter = nn_d[nlab != li]

        if intra.size == 0:
            # tidak ada tetangga intra → berikan baseline kecil
            d_intra = float(np.mean(nn_d))
        else:
            d_intra = float(np.mean(intra))

        if inter.size == 0:
            d_inter = d_intra
        else:
            d_inter = float(np.mean(inter))

        # normalisasi dengan IQR 5–95 persentil agar stabil
        q5, q95 = np.percentile(nn_d, [5, 95])
        iqr = max(1e-9, float(q95 - q5))
        Delta = float(np.clip((d_inter - d_intra) / iqr, 0.0, 1.0))

        # === 5) Skor landmark ===
        v = alpha * NC + (1.0 - alpha) * Delta
        v = float(np.clip(v, 0.0, 1.0))
        vals.append(v)

        if use_weighted_mean:
            weights.append(size_map[li] if li < size_map.size else 1.0)

    if len(vals) == 0:
        return np.nan

    V = np.asarray(vals, dtype=float)
    if use_weighted_mean and len(weights) == len(vals):
        W = np.asarray(weights, dtype=float)
        return float((W * V).sum() / max(1e-12, W.sum()))
    else:
        return float(V.mean())
