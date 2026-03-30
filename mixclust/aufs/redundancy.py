# mixclust/aufs/redundancy.py
#
# CHANGELOG v2.2 — Redesign total untuk performa dan keandalan
#
# MASALAH SEBELUMNYA (v2.1):
#   - joblib Parallel dengan backend='loky' gagal di Jupyter karena
#     pd.Series tidak bisa di-serialize dengan benar → fallback ke sequential
#     tapi sequential masih pakai loop per-observasi O(n) Python → lambat
#   - Warning "A worker stopped while some jobs were given to the executor"
#
# SOLUSI v2.2:
#   1. HAPUS joblib sepenuhnya — sequential sudah cukup cepat
#   2. Precompute premap untuk SEMUA fitur sekali (build_all_premaps):
#      O(n*d) total vs O(n*d²) sebelumnya
#   3. kmsnc_from_premaps: O(U1*U2*k + n) per pasang — tidak ada loop Python per obs
#   4. Estimasi waktu untuk 31×31, n=50k (row_subsample): ~1.1s total
#      vs ~5.8s sebelumnya (tanpa precompute) dan ~39s (loop per obs)
#
# BACKWARD COMPATIBLE: semua parameter lama tetap ada.
# Parameter use_parallel, n_jobs, backend, batch_size masih diterima
# tapi diabaikan (sequential sudah optimal).
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd
import pickle
from typing import Dict, Optional
from scipy.stats import entropy


# ─────────────────────────────────────────────────────────────────
# Core: precompute premaps untuk semua fitur (dilakukan sekali)
# ─────────────────────────────────────────────────────────────────

def _build_all_premaps(df_work: pd.DataFrame, feats: list, k: int) -> dict:
    """
    Precompute premap untuk semua fitur sekaligus — O(n*d) total.

    Premap[f] = {
        'idx'     : np.ndarray (n,) — inverse index dari np.unique
        'n_unique': int — jumlah nilai unik
        'pre'     : list of np.ndarray — k indeks terkecil per nilai unik (sorted)
        'cnt'     : np.ndarray — frekuensi per nilai unik
    }

    Keuntungan precompute:
    - argsort O(n log n) hanya dilakukan d kali, bukan d² kali
    - Total penghematan: 465 pasang × (skip argsort) = ~0.7s
    """
    premaps = {}
    for f in feats:
        arr = df_work[f].values
        if arr.dtype.kind == 'f':
            # Fitur numerik kontinyu: bin menjadi kategorik (10 bin)
            # agar kMSNC* tetap bermakna
            arr = pd.cut(pd.Series(arr), bins=10, labels=False).fillna(0).values.astype(int)
        u, idx = np.unique(arr, return_inverse=True)
        # Temukan k indeks terkecil per nilai unik via argsort sekali
        sort_order = np.argsort(idx, kind='stable')
        cnt = np.bincount(idx, minlength=len(u))
        splits = np.split(sort_order, np.cumsum(cnt[:-1]))
        premaps[f] = {
            'idx':      idx,
            'n_unique': int(len(u)),
            'pre':      [np.sort(s[:k]) for s in splits],  # sorted untuk intersect1d
            'cnt':      cnt,
        }
    return premaps


def _kmsnc_from_premaps(pm1: dict, pm2: dict, k: int, n: int) -> float:
    """
    Hitung kMSNC* dari precomputed premap — O(U1*U2*k + n).

    Tidak ada loop per observasi (selain lookup numpy O(n)).
    """
    nu1, nu2   = pm1['n_unique'], pm2['n_unique']
    pre1, pre2 = pm1['pre'],      pm2['pre']
    idx1, idx2 = pm1['idx'],      pm2['idx']

    # Intersection matrix (nu1 × nu2) via np.intersect1d
    # Kompleksitas: O(U1 * U2 * k log k)
    # Untuk U1=U2=5, k=5: hanya 25 set intersections × 5 elemen = trivial
    inter = np.zeros((nu1, nu2), dtype=np.float32)
    for v1i in range(nu1):
        p1 = pre1[v1i]
        for v2i in range(nu2):
            inter[v1i, v2i] = float(len(
                np.intersect1d(p1, pre2[v2i], assume_unique=True)
            ))

    # Lookup per obs — O(n) numpy indexing
    overlap_sum = float(np.sum(inter[idx1, idx2]))
    overlap = (overlap_sum / n) / k

    # Shannon entropy
    H1 = entropy(pm1['cnt'] / n) if n > 0 else 0.0
    H2 = entropy(pm2['cnt'] / n) if n > 0 else 0.0

    return float(overlap * (H1 + H2) / 2.0)


# ─────────────────────────────────────────────────────────────────
# API lama (dipertahankan untuk backward compat)
# ─────────────────────────────────────────────────────────────────

def _overlap_k_neighbors(s1: pd.Series, s2: pd.Series, k: int,
                         pre1=None, pre2=None) -> float:
    """Implementasi asli — O(n) Python loop. Hanya untuk n kecil (<5000)."""
    if len(s1) != len(s2) or k <= 0 or len(s1) == 0:
        return 0.0
    acc = 0.0
    for i in range(len(s1)):
        v1, v2 = s1.iloc[i], s2.iloc[i]
        n1 = pre1.get(v1, pd.Index([])) if pre1 is not None \
             else s1[s1 == v1].index[:k]
        n2 = pre2.get(v2, pd.Index([])) if pre2 is not None \
             else s2[s2 == v2].index[:k]
        acc += len(set(n1).intersection(set(n2))) / k
    return float(acc / len(s1))


def kmsnc_star_pair(s1: pd.Series, s2: pd.Series, k: int,
                    pre1=None, pre2=None) -> float:
    """
    kMSNC* untuk satu pasang fitur.
    Dipakai oleh build_redundancy_matrix secara internal (via premap baru).
    API ini tetap tersedia untuk backward compat.
    """
    H1 = entropy(s1.value_counts(normalize=True, sort=False)) if len(s1) else 0.0
    H2 = entropy(s2.value_counts(normalize=True, sort=False)) if len(s2) else 0.0
    return _overlap_k_neighbors(s1, s2, k, pre1, pre2) * ((H1 + H2) / 2.0)


# ─────────────────────────────────────────────────────────────────
# Cache fingerprint
# ─────────────────────────────────────────────────────────────────

def _make_fingerprint(df: pd.DataFrame, k: int) -> str:
    cols_str  = ",".join(sorted(df.columns.tolist()))
    shape_str = f"{df.shape[0]}x{df.shape[1]}"
    key       = f"{cols_str}|{shape_str}|k={k}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _load_cache(cache_path: str, fingerprint: str, verbose: bool = True):
    try:
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        if not isinstance(cached, dict) or "_meta" not in cached:
            if verbose:
                print("[redundancy] Cache format lama — dihitung ulang.")
            return None
        meta = cached["_meta"]
        if meta.get("fingerprint") != fingerprint:
            if verbose:
                print("[redundancy] Cache tidak valid (fingerprint mismatch) — dihitung ulang.")
            return None
        if verbose:
            print(f"[redundancy] Cache hit ✓  "
                  f"(cols={meta.get('n_cols')}, n={meta.get('n_rows')}, k={meta.get('k')})")
        return cached["matrix"]
    except FileNotFoundError:
        return None
    except Exception as e:
        if verbose:
            print(f"[redundancy] Cache rusak ({e}) — dihitung ulang.")
        return None


def _save_cache(cache_path: str, matrix: dict, fingerprint: str,
                df: pd.DataFrame, k: int) -> None:
    import os
    try:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        payload = {
            "_meta": {
                "fingerprint": fingerprint,
                "n_cols":      df.shape[1],
                "n_rows":      df.shape[0],
                "k":           k,
                "columns":     sorted(df.columns.tolist()),
            },
            "matrix": matrix,
        }
        with open(cache_path, "wb") as f:
            pickle.dump(payload, f)
    except Exception as e:
        print(f"[redundancy] Gagal simpan cache: {e}")


# ─────────────────────────────────────────────────────────────────
# Main function
# ─────────────────────────────────────────────────────────────────

def build_redundancy_matrix(
    df: pd.DataFrame,
    k: int = 5,
    cache_path: Optional[str] = None,
    # Parameter lama — tetap diterima tapi diabaikan (backward compat)
    precompute: bool = True,
    use_parallel: bool = False,   # ← default False (joblib dihapus)
    n_jobs: int = 1,
    row_subsample: Optional[int] = 50_000,
    backend: str = "sequential",  # ← tidak dipakai
    batch_size: int = 32,         # ← tidak dipakai
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Hitung kMSNC* redundancy matrix antar semua pasangan fitur.

    v2.2: Sequential vectorized dengan precompute premap.
    Estimasi waktu: 31×31, row_subsample=50k → ~1-2s (vs ~40s v2.0).

    Parameter use_parallel, n_jobs, backend, batch_size masih diterima
    untuk backward compat tapi tidak digunakan.
    """
    if cache_path:
        fingerprint = _make_fingerprint(df, k)
        cached = _load_cache(cache_path, fingerprint, verbose=verbose)
        if cached is not None:
            return cached

    # Subsample baris jika perlu
    df_work = df
    if row_subsample is not None and len(df) > row_subsample:
        df_work = df.sample(row_subsample, random_state=42).reset_index(drop=True)

    feats  = df_work.columns.tolist()
    d      = len(feats)
    n_work = len(df_work)

    if verbose:
        print(f"[redundancy] Menghitung matrix {d}×{d} "
              f"(n={n_work:,}, k={k}) ...")

    # ── Precompute premaps untuk semua fitur sekali ──
    premaps = _build_all_premaps(df_work, feats, k)

    # ── Hitung semua pasang secara sequential ──
    M = np.zeros((d, d), dtype=float)
    pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]

    for i, j in pairs:
        v = _kmsnc_from_premaps(premaps[feats[i]], premaps[feats[j]], k, n_work)
        M[i, j] = M[j, i] = v

    mat = {feats[i]: {feats[j]: M[i, j] for j in range(d)} for i in range(d)}

    if cache_path:
        _save_cache(cache_path, mat, fingerprint, df, k)
        if verbose:
            print(f"[redundancy] Cache disimpan → {cache_path}")

    return mat


# ─────────────────────────────────────────────────────────────────
# Fungsi bantu
# ─────────────────────────────────────────────────────────────────

def init_by_least_redundant(red_matrix: Dict[str, Dict[str, float]], k: int):
    if not red_matrix or k <= 0:
        return []
    mean_r = {}
    for f, row in red_matrix.items():
        vals = [v for g, v in row.items() if g != f and isinstance(v, (int, float))]
        mean_r[f] = float(np.mean(vals)) if vals else np.inf
    ok = {f: v for f, v in mean_r.items() if np.isfinite(v)}
    if not ok:
        return list(mean_r.keys())[:k]
    return sorted(ok, key=lambda x: ok[x])[:k]


def make_mab_reward_from_matrix(red_mat):
    def _r(df_subset: pd.DataFrame) -> float:
        cols = list(df_subset.columns)
        m = len(cols)
        if m <= 1:
            return 1.0
        s = 0.0; c = 0
        for i in range(m):
            ri = red_mat.get(cols[i], {})
            for j in range(i + 1, m):
                s += float(ri.get(cols[j], 0.0)); c += 1
        return 1.0 - max(0.0, min(1.0, s / max(1, c)))
    return _r


def redundancy_penalty(cols, red_mat, mode: str = "mean_invert") -> float:
    if not cols:
        return 0.0
    s = 0.0; c = 0
    for i in range(len(cols)):
        ri = red_mat.get(cols[i], {})
        for j in range(i + 1, len(cols)):
            s += float(ri.get(cols[j], 0.0)); c += 1
    if c == 0:
        return 1.0 if mode == "mean_invert" else 0.0
    mean_red = max(0.0, min(1.0, s / c))
    return 1.0 - mean_red if mode == "mean_invert" else mean_red
