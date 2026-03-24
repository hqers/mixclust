# mixclust/aufs/redundancy.py
from __future__ import annotations
import numpy as np, pandas as pd, pickle
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

def build_redundancy_matrix(
    df: pd.DataFrame,
    k: int = 5,
    cache_path: Optional[str] = None,
    precompute: bool = True,
    use_parallel: bool = True,
    n_jobs: int = 2,                     # proses-berbasis, aman RAM
    row_subsample: Optional[int] = 50_000, # subsample baris khusus redundansi
    backend: str = "loky",               # proses (bukan threading)
    batch_size: int = 8                  # tugas kecil biar lancar
) -> Dict[str, Dict[str, float]]:
    # --- cache load ---
    if cache_path:
        try:
            with open(cache_path, "rb") as f: return pickle.load(f)
        except Exception:
            pass

    # --- baris kerja (hemat O(p^2 * n)) ---
    df_work = df
    if row_subsample is not None and len(df) > row_subsample:
        df_work = df.sample(row_subsample, random_state=42).reset_index(drop=True)

    feats = df_work.columns.tolist()
    d = len(feats)

    # --- premap kandidat tetangga per nilai (boleh dinonaktifkan jika n' kecil) ---
    if row_subsample is not None and row_subsample < 30_000:
        precompute_eff = False
    else:
        precompute_eff = precompute

    premap = None
    if precompute_eff:
        premap = {
            f: {val: df_work[f][df_work[f] == val].index[:k] for val in df_work[f].unique()}
            for f in feats
        }

    M = np.zeros((d, d), dtype=float)

    if use_parallel:
        from joblib import Parallel, delayed
        pairs = [(i, j) for i in range(d) for j in range(i+1, d)]
        vals = Parallel(n_jobs=n_jobs, backend=backend, batch_size=batch_size)(
            delayed(kmsnc_star_pair)(
                df_work[feats[i]], df_work[feats[j]], k,
                premap.get(feats[i]) if premap else None,
                premap.get(feats[j]) if premap else None
            )
            for (i, j) in pairs
        )
        for p, (i, j) in enumerate(pairs):
            M[i, j] = M[j, i] = vals[p]
    else:
        for i in range(d):
            for j in range(i+1, d):
                M[i, j] = M[j, i] = kmsnc_star_pair(
                    df_work[feats[i]], df_work[feats[j]], k,
                    premap.get(feats[i]) if premap else None,
                    premap.get(feats[j]) if premap else None
                )

    mat = {feats[i]: {feats[j]: M[i, j] for j in range(d)} for i in range(d)}

    # --- cache save ---
    if cache_path:
        try:
            with open(cache_path, "wb") as f: pickle.dump(mat, f)
        except Exception:
            pass
    return mat


def init_by_least_redundant(red_matrix: Dict[str, Dict[str, float]], k: int):
    if not red_matrix or k <= 0: return []
    mean_r = {}
    for f, row in red_matrix.items():
        vals = [v for g, v in row.items() if g != f and isinstance(v, (int, float))]
        mean_r[f] = float(np.mean(vals)) if vals else np.inf
    ok = {f:v for f,v in mean_r.items() if np.isfinite(v)}
    if not ok: return list(mean_r.keys())[:k]
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
            return 1.0  # tidak ada pasangan → dianggap tidak redundan
        s = 0.0
        c = 0
        for i in range(m):
            ri = red_mat.get(cols[i], {})
            for j in range(i+1, m):
                s += float(ri.get(cols[j], 0.0))
                c += 1
        mean_red = s / max(1, c)
        # clamp supaya aman
        mean_red = max(0.0, min(1.0, mean_red))
        return 1.0 - mean_red
    return _r

# --- NEW: penalti redundansi untuk reward AUFS/SA ---
def redundancy_penalty(cols, red_mat, mode: str = "mean_invert") -> float:
    """
    Hitung penalti redundansi berbasis kMSNC* matrix.
    - mode "mean_invert": 1 - mean(redundancy)  (makin besar -> makin baik)
    """
    if not cols:
        return 0.0
    s = 0.0; c = 0
    for i in range(len(cols)):
        ri = red_mat.get(cols[i], {})
        for j in range(i+1, len(cols)):
            s += float(ri.get(cols[j], 0.0))
            c += 1
    if c == 0:
        return 1.0 if mode == "mean_invert" else 0.0
    mean_red = max(0.0, min(1.0, s / c))
    return 1.0 - mean_red if mode == "mean_invert" else mean_red

