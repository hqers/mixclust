# mixclust/aufs/redundancy.py
from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd
import pickle
from typing import Dict, Any, Optional
from scipy.stats import entropy


def _overlap_k_neighbors(s1: pd.Series, s2: pd.Series, k: int,
                         pre1=None, pre2=None) -> float:
    if len(s1) != len(s2) or k <= 0 or len(s1) == 0:
        return 0.0
    acc = 0.0
    for i in range(len(s1)):
        v1, v2 = s1.iloc[i], s2.iloc[i]
        n1 = pre1.get(v1, pd.Index([])) if pre1 is not None else s1[s1 == v1].index[:k]
        n2 = pre2.get(v2, pd.Index([])) if pre2 is not None else s2[s2 == v2].index[:k]
        acc += len(set(n1).intersection(set(n2))) / k
    return float(acc / len(s1))


def kmsnc_star_pair(s1: pd.Series, s2: pd.Series, k: int,
                    pre1=None, pre2=None) -> float:
    H1 = entropy(s1.value_counts(normalize=True, sort=False)) if len(s1) else 0.0
    H2 = entropy(s2.value_counts(normalize=True, sort=False)) if len(s2) else 0.0
    return _overlap_k_neighbors(s1, s2, k, pre1, pre2) * ((H1 + H2) / 2.0)


# ─────────────────────────────────────────────────────────────────────────────
# Cache fingerprint
# ─────────────────────────────────────────────────────────────────────────────

def _make_fingerprint(df: pd.DataFrame, k: int) -> str:
    """
    Buat fingerprint unik berdasarkan:
    - Nama kolom (sorted) → dataset/fitur yang berbeda → hash berbeda
    - Shape (n_rows, n_cols)
    - Parameter k

    Tidak pakai hash isi data (terlalu lambat untuk 334K baris).
    Shape + kolom sudah cukup untuk mendeteksi:
    - Ganti dataset (kolom/shape berbeda)
    - Subset fitur berbeda (kolom berbeda)
    - Parameter k berbeda
    """
    cols_str  = ",".join(sorted(df.columns.tolist()))
    shape_str = f"{df.shape[0]}x{df.shape[1]}"
    key       = f"{cols_str}|{shape_str}|k={k}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _load_cache(cache_path: str, fingerprint: str, verbose: bool = True):
    """
    Load cache jika ada DAN fingerprint cocok.
    Return: matrix dict atau None jika cache tidak valid.
    """
    try:
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)

        # Format lama (dict langsung, tanpa metadata) → otomatis invalid
        if not isinstance(cached, dict) or "_meta" not in cached:
            if verbose:
                print("[redundancy] Cache format lama terdeteksi — akan dihitung ulang.")
            return None

        meta = cached["_meta"]
        if meta.get("fingerprint") != fingerprint:
            if verbose:
                reason = []
                if meta.get("n_cols") != fingerprint.split("|")[0] if "|" in fingerprint else "?":
                    reason.append("dataset/kolom berbeda")
                print(f"[redundancy] Cache tidak valid (fingerprint mismatch: "
                      f"dataset atau parameter k berubah) — dihitung ulang.")
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
    """Simpan matrix + metadata ke cache."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Main function
# ─────────────────────────────────────────────────────────────────────────────

def build_redundancy_matrix(
    df: pd.DataFrame,
    k: int = 5,
    cache_path: Optional[str] = None,
    precompute: bool = True,
    use_parallel: bool = True,
    n_jobs: int = 2,
    row_subsample: Optional[int] = 50_000,
    backend: str = "loky",
    batch_size: int = 8,
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Hitung kMSNC* redundancy matrix antar semua pasangan fitur.

    Cache otomatis ter-invalidasi jika:
    - Dataset berbeda (nama kolom atau shape berubah)
    - Parameter k berubah

    Parameters
    ----------
    cache_path : str, optional
        Path file cache (.pkl). None = tidak pakai cache.
    verbose : bool
        Print info cache hit/miss dan progress.
    """
    # ── Cache load dengan validasi fingerprint ────────────────────────────────
    if cache_path:
        fingerprint = _make_fingerprint(df, k)
        cached = _load_cache(cache_path, fingerprint, verbose=verbose)
        if cached is not None:
            # Intersect dengan kolom df — safety net jika ada kolom
            # yang tidak matching (misal cache format lama)
            df_cols = set(df.columns.tolist())
            filtered = {
                c: {k2: v for k2, v in row.items() if k2 in df_cols}
                for c, row in cached.items()
                if c in df_cols
            }
            if len(filtered) != df.shape[1] and verbose:
                missing = df_cols - set(filtered.keys())
                extra   = set(filtered.keys()) - df_cols
                if missing:
                    print(f"[redundancy] Cache miss kolom: {missing} — akan dihitung ulang.")
                    # kolom ada di df tapi tidak di cache → cache tidak valid
                    pass
                else:
                    return filtered
            elif len(filtered) == df.shape[1]:
                return filtered
            # jika ada kolom df yang tidak di cache → hitung ulang
            if verbose:
                print("[redundancy] Cache tidak lengkap untuk dataset ini — dihitung ulang.")

    # ── Hitung dari scratch ───────────────────────────────────────────────────
    df_work = df
    if row_subsample is not None and len(df) > row_subsample:
        df_work = df.sample(row_subsample, random_state=42).reset_index(drop=True)

    feats = df_work.columns.tolist()
    d = len(feats)

    if verbose:
        print(f"[redundancy] Menghitung matrix {d}×{d} "
              f"(n={len(df_work):,}, k={k})...")

    precompute_eff = precompute and not (row_subsample is not None and row_subsample < 30_000)
    premap = None
    if precompute_eff:
        premap = {
            f: {val: df_work[f][df_work[f] == val].index[:k] for val in df_work[f].unique()}
            for f in feats
        }

    M = np.zeros((d, d), dtype=float)

    if use_parallel:
        from joblib import Parallel, delayed
        pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
        vals = Parallel(n_jobs=n_jobs, backend=backend, batch_size=batch_size)(
            delayed(kmsnc_star_pair)(
                df_work[feats[i]], df_work[feats[j]], k,
                premap.get(feats[i]) if premap else None,
                premap.get(feats[j]) if premap else None,
            )
            for (i, j) in pairs
        )
        for p, (i, j) in enumerate(pairs):
            M[i, j] = M[j, i] = vals[p]
    else:
        for i in range(d):
            for j in range(i + 1, d):
                M[i, j] = M[j, i] = kmsnc_star_pair(
                    df_work[feats[i]], df_work[feats[j]], k,
                    premap.get(feats[i]) if premap else None,
                    premap.get(feats[j]) if premap else None,
                )

    mat = {feats[i]: {feats[j]: M[i, j] for j in range(d)} for i in range(d)}

    # ── Cache save ─────────────────────────────────────────────────────────────
    if cache_path:
        _save_cache(cache_path, mat, fingerprint, df, k)
        if verbose:
            print(f"[redundancy] Cache disimpan → {cache_path}")

    return mat


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
    """
    Reward MAB = 1 - mean(kMSNC*) atas pasangan fitur di subset.
    Maksimasi → subset makin tidak redundan.
    """
    def _r(df_subset: pd.DataFrame) -> float:
        cols = list(df_subset.columns)
        m = len(cols)
        if m <= 1:
            return 1.0
        s = 0.0
        c = 0
        for i in range(m):
            ri = red_mat.get(cols[i], {})
            for j in range(i + 1, m):
                s += float(ri.get(cols[j], 0.0))
                c += 1
        mean_red = max(0.0, min(1.0, s / max(1, c)))
        return 1.0 - mean_red
    return _r


def redundancy_penalty(cols, red_mat, mode: str = "mean_invert") -> float:
    """
    Hitung penalti redundansi berbasis kMSNC* matrix.
    mode "mean_invert": 1 - mean(redundancy)
    """
    if not cols:
        return 0.0
    s = 0.0
    c = 0
    for i in range(len(cols)):
        ri = red_mat.get(cols[i], {})
        for j in range(i + 1, len(cols)):
            s += float(ri.get(cols[j], 0.0))
            c += 1
    if c == 0:
        return 1.0 if mode == "mean_invert" else 0.0
    mean_red = max(0.0, min(1.0, s / c))
    return 1.0 - mean_red if mode == "mean_invert" else mean_red
