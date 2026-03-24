# mixclust/core/gower.py
import numpy as np

def _ensure_bool_mask(mask, length):
    """
    Pastikan mask bertipe bool dan panjangnya sesuai 'length'.
    Jika None → kembalikan vektor True (semua fitur aktif).
    """
    if length is None or length == 0:
        return np.zeros((0,), dtype=bool)
    if mask is None:
        return np.ones(int(length), dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1 or mask.size != int(length):
        raise ValueError("feature_mask memiliki ukuran tidak sesuai.")
    return mask


def gower_to_one_mixed(
    X_num: np.ndarray,
    X_cat: np.ndarray,
    num_min: np.ndarray,
    num_max: np.ndarray,
    i: int,
    cand_idx: np.ndarray,
    *,
    feature_mask_num: np.ndarray | None = None,
    feature_mask_cat: np.ndarray | None = None,
    inv_rng: np.ndarray | None = None
) -> np.ndarray:
    """
    Hitung jarak Gower dari titik i ke sekumpulan kandidat cand_idx pada data mixed-type.
    Rumus: mean across-features dari jarak per fitur (numerik terskala [0,1], kategorik 0/1).
    - Skala numerik: |x - y| / range. Jika inv_rng disediakan, gunakan itu (cache).
    - feature_mask_*: memilih subset fitur yang aktif.
    Return: np.ndarray[m] (float32), m = len(cand_idx)
    """
    cand_idx = np.asarray(cand_idx, dtype=int)
    m = int(cand_idx.size)
    if m == 0:
        return np.zeros(0, dtype=np.float32)

    out_sum = np.zeros(m, dtype=np.float32)
    total_feats = 0

    # ---- Numerik ----
    if X_num is not None and X_num.shape[1] > 0:
        p = int(X_num.shape[1])
        num_mask = _ensure_bool_mask(feature_mask_num, p)
        p_num = int(np.sum(num_mask))
        if p_num > 0:
            if inv_rng is None:
                rng = (np.asarray(num_max, dtype=np.float32) - np.asarray(num_min, dtype=np.float32))[num_mask]
                rng[rng == 0.0] = 1.0
                scale = (1.0 / rng).astype(np.float32)
            else:
                inv_rng = np.asarray(inv_rng, dtype=np.float32)
                scale = inv_rng[num_mask].astype(np.float32)

            xi = X_num[int(i), num_mask].astype(np.float32)
            Xc = X_num[cand_idx][:, num_mask].astype(np.float32)
            # sum_k |x_ik - x_jk| * scale_k
            out_sum += np.sum(np.abs(Xc - xi) * scale, axis=1, dtype=np.float32)
            total_feats += p_num

    # ---- Kategorik ----
    if X_cat is not None and X_cat.shape[1] > 0:
        q = int(X_cat.shape[1])
        cat_mask = _ensure_bool_mask(feature_mask_cat, q)
        p_cat = int(np.sum(cat_mask))
        if p_cat > 0:
            xi = X_cat[int(i), cat_mask]
            Xc = X_cat[cand_idx][:, cat_mask]
            mism = (Xc != xi).astype(np.float32)
            out_sum += np.sum(mism, axis=1, dtype=np.float32)
            total_feats += p_cat

    if total_feats == 0:
        # Tidak ada fitur aktif → kembalikan vektor ones (jarak netral)
        return np.ones(m, dtype=np.float32)

    return (out_sum / float(total_feats)).astype(np.float32)


def rerank_gower_from_candidates(
    i: int,
    cand_idx: np.ndarray,
    k: int,
    X_num: np.ndarray,
    X_cat: np.ndarray,
    num_min: np.ndarray,
    num_max: np.ndarray,
    *,
    feature_mask_num: np.ndarray | None = None,
    feature_mask_cat: np.ndarray | None = None,
    inv_rng: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Ambil k tetangga terdekat (berdasar Gower) dari kandidat cand_idx untuk query i.
    Guard:
      - cand_idx kosong → ([],[])
      - k <= 0 → ([],[])
      - k > len(cand_idx) → dipotong otomatis
    Return:
      nn_idx: np.ndarray[k] (int)
      nn_dists: np.ndarray[k] (float32)
    """
    cand_idx = np.asarray(cand_idx, dtype=int)
    m = int(cand_idx.size)
    k = int(k)

    if m == 0 or k <= 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=np.float32)

    d = gower_to_one_mixed(
        X_num, X_cat, num_min, num_max, int(i), cand_idx,
        feature_mask_num=feature_mask_num, feature_mask_cat=feature_mask_cat,
        inv_rng=inv_rng
    )

    k_eff = int(min(k, m))
    # argpartition aman untuk k_eff >= 1
    part = np.argpartition(d, k_eff - 1)[:k_eff]
    order = part[np.argsort(d[part])]
    return cand_idx[order].astype(int, copy=False), d[order].astype(np.float32, copy=False)


def gower_to_one_mixed_scaled(
    X_num_scaled: np.ndarray,
    X_cat: np.ndarray,
    i: int,
    cand_idx: np.ndarray,
    *,
    feature_mask_num: np.ndarray | None = None,
    feature_mask_cat: np.ndarray | None = None
) -> np.ndarray:
    """
    Varian jika numerik SUDAH terskala ke [0,1] atau skala yang ekuivalen.
    Tidak butuh num_min/num_max/inv_rng.
    Return: np.ndarray[m] float32
    """
    cand_idx = np.asarray(cand_idx, dtype=int)
    m = int(cand_idx.size)
    if m == 0:
        return np.zeros(0, dtype=np.float32)

    out_sum = np.zeros(m, dtype=np.float32)
    total_feats = 0

    # Numerik (scaled)
    if X_num_scaled is not None and X_num_scaled.shape[1] > 0:
        p = int(X_num_scaled.shape[1])
        num_mask = _ensure_bool_mask(feature_mask_num, p)
        p_num = int(np.sum(num_mask))
        if p_num > 0:
            xi = X_num_scaled[int(i), num_mask].astype(np.float32)
            Xc = X_num_scaled[cand_idx][:, num_mask].astype(np.float32)
            out_sum += np.sum(np.abs(Xc - xi), axis=1, dtype=np.float32)
            total_feats += p_num

    # Kategorik
    if X_cat is not None and X_cat.shape[1] > 0:
        q = int(X_cat.shape[1])
        cat_mask = _ensure_bool_mask(feature_mask_cat, q)
        p_cat = int(np.sum(cat_mask))
        if p_cat > 0:
            xi = X_cat[int(i), cat_mask]
            Xc = X_cat[cand_idx][:, cat_mask]
            mism = (Xc != xi).astype(np.float32)
            out_sum += np.sum(mism, axis=1, dtype=np.float32)
            total_feats += p_cat

    if total_feats == 0:
        return np.ones(m, dtype=np.float32)

    return (out_sum / float(total_feats)).astype(np.float32)
