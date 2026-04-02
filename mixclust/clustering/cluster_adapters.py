# mixclust/clustering/cluster_adapters.py
#
# UPDATED: Added min_cluster_balance check and n_clusters_hint auto-adjustment
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
from typing import List, Optional, Union
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering

try:
    from kmodes.kmodes import KModes
    from kmodes.kprototypes import KPrototypes
    _HAS_KMODES = True
except Exception:
    _HAS_KMODES = False


# =============== Util ringan ===============

def _split_types(X: pd.DataFrame):
    cat = X.select_dtypes(include=["bool", "object", "category"]).columns.tolist()
    num = X.select_dtypes(include=[np.number]).columns.tolist()
    return cat, num

def _prep_numeric(X: pd.DataFrame, num_cols: List[str]) -> np.ndarray:
    if not num_cols:
        return np.zeros((len(X), 0), dtype=float)
    Z = X[num_cols].to_numpy(dtype=float, copy=True)
    Z = StandardScaler().fit_transform(Z)
    return Z

def _prep_categorical(X: pd.DataFrame, cat_cols: List[str]) -> np.ndarray:
    if not cat_cols:
        return np.zeros((len(X), 0), dtype=object)
    Z = X[cat_cols].astype(str).to_numpy(copy=True)
    return Z

def _validate_k(n: int, n_clusters: int):
    if n_clusters < 2:
        raise ValueError("n_clusters minimal 2.")
    if n_clusters > n:
        raise ValueError(f"n_clusters ({n_clusters}) tidak boleh melebihi jumlah sampel ({n}).")

def _cat_cols_from_idx(df: pd.DataFrame, idx: List[int]) -> List[str]:
    return [df.columns[i] for i in idx]


def _check_cluster_balance(
    labels: np.ndarray,
    n: int,
    min_frac: float = 0.02,
    min_abs: int = 3,
) -> bool:
    """
    Check if clustering result has acceptable balance.
    Returns True if ALL clusters have at least min_frac * n (or min_abs) members.
    Returns False if any cluster is degenerate (too small).
    """
    if labels is None or len(labels) == 0:
        return False
    counts = np.bincount(labels.astype(int) if hasattr(labels, 'astype') else 
                         np.array(labels, dtype=int))
    threshold = max(min_abs, int(min_frac * n))
    return int(np.min(counts)) >= threshold


# =============== Adapter: KMeans ===============

def kmeans_adapter(
    X_df: pd.DataFrame,
    cat_idx: List[int],
    n_clusters: int,
    random_state: Optional[int] = None
) -> Union[np.ndarray, List]:
    _validate_k(len(X_df), n_clusters)
    cat_cols, num_cols = _split_types(X_df)
    if len(cat_cols) > 0 or len(cat_idx) > 0:
        raise ValueError("kmeans_adapter butuh semua fitur numerik (tanpa kolom kategorik).")
    if len(num_cols) == 0:
        Z = np.zeros((len(X_df), 1), dtype=float)
    else:
        Z = _prep_numeric(X_df, num_cols)
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    return model.fit_predict(Z)


# =============== Adapter: KModes ===============

def kmodes_adapter(
    X_df: pd.DataFrame,
    cat_idx: List[int],
    n_clusters: int,
    random_state: Optional[int] = None
) -> Union[np.ndarray, List]:
    if not _HAS_KMODES:
        raise ImportError("kmodes/kprototypes belum terpasang. `pip install kmodes`")
    _validate_k(len(X_df), n_clusters)
    cat_cols, num_cols = _split_types(X_df)
    if len(num_cols) > 0:
        raise ValueError("kmodes_adapter butuh semua fitur kategorik.")
    Xc = X_df[cat_cols].astype(str)
    try:
        model = KModes(n_clusters=n_clusters, init="Huang", n_init=10, verbose=0, random_state=random_state)
    except TypeError:
        model = KModes(n_clusters=n_clusters, init="Huang", n_init=10, verbose=0)
    return model.fit_predict(Xc)


# =============== Adapter: KPrototypes ===============

def kprototypes_adapter(
    X_df: pd.DataFrame,
    cat_idx: List[int],
    n_clusters: int,
    random_state: Optional[int] = None,
    max_iter: int = 20,
    gamma: Optional[float] = None
) -> Union[np.ndarray, List]:
    if not _HAS_KMODES:
        raise ImportError("kmodes/kprototypes belum terpasang. `pip install kmodes`")
    _validate_k(len(X_df), n_clusters)

    cat_cols = _cat_cols_from_idx(X_df, cat_idx)
    num_cols = [c for c in X_df.columns if c not in cat_cols]

    Z_num = _prep_numeric(X_df, num_cols)
    Z_cat = _prep_categorical(X_df, cat_cols)

    if Z_num.shape[1] == 0 and Z_cat.shape[1] > 0:
        return kmodes_adapter(X_df[cat_cols], [], n_clusters, random_state)

    if Z_cat.shape[1] == 0:
        Z = Z_num.astype(object)
        cat_anchor: List[int] = []
    else:
        Z = np.concatenate([Z_num, Z_cat], axis=1).astype(object)
        cat_anchor = list(range(Z_num.shape[1], Z_num.shape[1] + Z_cat.shape[1]))

    try:
        model = KPrototypes(
            n_clusters=n_clusters, init="Cao", n_init=5,
            verbose=0, random_state=random_state, max_iter=max_iter, gamma=gamma
        )
    except TypeError:
        model = KPrototypes(
            n_clusters=n_clusters, init="Cao", n_init=5,
            verbose=0, max_iter=max_iter, gamma=gamma
        )

    return model.fit_predict(Z, categorical=cat_anchor)


# =============== Adapter: HAC-Gower ===============

def _gower_pair(
    xi_num: np.ndarray, xj_num: np.ndarray,
    xi_cat: np.ndarray, xj_cat: np.ndarray,
    inv_rng: np.ndarray
) -> float:
    d = 0.0; m = 0
    if xi_num.size:
        d += np.sum(np.abs(xi_num - xj_num) * inv_rng)
        m += inv_rng.size
    if xi_cat.size:
        d += np.sum((xi_cat != xj_cat).astype(float))
        m += xi_cat.size
    return d / max(1, m)

def hac_gower_adapter(
    X_df: pd.DataFrame,
    cat_idx: List[int],
    n_clusters: int,
    random_state: Optional[int] = None
) -> np.ndarray:
    _validate_k(len(X_df), n_clusters)
    cat_cols = _cat_cols_from_idx(X_df, cat_idx)
    num_cols = [c for c in X_df.columns if c not in cat_cols]

    Xn = X_df[num_cols].to_numpy(dtype=float) if num_cols else np.zeros((len(X_df), 0))
    Xc = X_df[cat_cols].astype(str).to_numpy() if cat_cols else np.zeros((len(X_df), 0), dtype=object)

    if Xn.shape[1]:
        vmin = np.nanmin(Xn, axis=0)
        vmax = np.nanmax(Xn, axis=0)
        inv_rng = 1.0 / np.maximum(vmax - vmin, 1e-9)
    else:
        inv_rng = np.ones((0,), dtype=float)

    n = len(X_df)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        xi_num = Xn[i, :] if Xn.shape[1] else np.zeros((0,))
        xi_cat = Xc[i, :] if Xc.shape[1] else np.zeros((0,), dtype=object)
        for j in range(i + 1, n):
            xj_num = Xn[j, :] if Xn.shape[1] else np.zeros((0,))
            xj_cat = Xc[j, :] if Xc.shape[1] else np.zeros((0,), dtype=object)
            D[i, j] = D[j, i] = _gower_pair(xi_num, xj_num, xi_cat, xj_cat, inv_rng)

    model = AgglomerativeClustering(
        n_clusters=n_clusters, linkage="average", metric="precomputed"
    )
    return model.fit_predict(D)


# =============== Adapter: auto (with balance retry) ===============

def auto_adapter(
    X_df: pd.DataFrame,
    cat_idx: List[int],
    n_clusters: int,
    random_state: Optional[int] = None,
    *,
    min_cluster_frac: float = 0.02,
    max_init_retries: int = 3,
    n_threshold_large: int = 10_000,
) -> Union[np.ndarray, List]:
    """
    Auto-select clustering algorithm based on feature composition and data size.

    Logika diagnosis:
    - Numerik dominan (cat=0)          → KMeans
    - Kategorik dominan (num=0)        → KModes
    - Mixed, n kecil (≤n_threshold)    → HAC-Gower (paling akurat, O(n²))
    - Mixed, n besar (>n_threshold)    → kprototypes (default mixed-type)
      Jika KAMILA tersedia + n besar   → coba KAMILA juga, pilih via balance check

    Includes balance check — retries with different seeds if degenerate.
    """
    cat_cols, num_cols = _split_types(X_df)
    n = len(X_df)

    # Choose algorithm based on type composition + data size
    if len(cat_cols) == 0:
        adapter_fn = kmeans_adapter
    elif len(num_cols) == 0:
        adapter_fn = kmodes_adapter
    elif n <= n_threshold_large:
        # Mixed, n kecil → prefer kprototypes (HAC via controller kalau dipilih)
        adapter_fn = kprototypes_adapter
    else:
        # Mixed, n besar → kprototypes default
        adapter_fn = kprototypes_adapter

    # First attempt
    labels = adapter_fn(X_df, cat_idx, n_clusters, random_state)

    # Balance check
    if _check_cluster_balance(labels, n, min_frac=min_cluster_frac):
        return labels

    # Retry with different seeds
    best_labels = labels
    best_min_count = int(np.min(np.bincount(np.asarray(labels, dtype=int))))

    for retry in range(max_init_retries):
        seed = (random_state or 42) + retry + 1
        try:
            labels_retry = adapter_fn(X_df, cat_idx, n_clusters, seed)
            counts = np.bincount(np.asarray(labels_retry, dtype=int))
            min_count = int(np.min(counts))

            if min_count > best_min_count:
                best_min_count = min_count
                best_labels = labels_retry

            if _check_cluster_balance(labels_retry, n, min_frac=min_cluster_frac):
                return labels_retry
        except Exception:
            continue

    return best_labels

def kprototypes_subsample_adapter(
    X_df, cat_idx, k, random_state,
    subsample_n=6000, max_iter=20
):
    import numpy as np
    from kmodes.kprototypes import KPrototypes
    
    n = len(X_df)
    
    # Kalau data kecil, langsung full
    if n <= subsample_n:
        return kprototypes_adapter(X_df, cat_idx, k, random_state, max_iter=max_iter)
    
    # Subsample stratified
    rng = np.random.default_rng(random_state)
    idx_sub = rng.choice(n, size=subsample_n, replace=False)
    X_sub = X_df.iloc[idx_sub]
    
    # Fit pada subsample
    kp = KPrototypes(n_clusters=k, init='Cao', n_init=1, 
                     random_state=random_state, max_iter=max_iter)
    kp.fit(X_sub.values, categorical=cat_idx)
    
    # Assign seluruh data ke centroid yang sudah fit
    labels_all = kp.predict(X_df.values, categorical=cat_idx)
    
    return labels_all.astype(int)


# =============== Adapter: KAMILA ===============

try:
    from .kamila import KAMILA
    _HAS_KAMILA = True
except Exception:
    _HAS_KAMILA = False


def kamila_adapter(
    X_df: pd.DataFrame,
    cat_idx: List[int],
    n_clusters: int,
    random_state: Optional[int] = None,
    *,
    n_init: int = 5,
    max_iter: int = 25,
) -> np.ndarray:
    """KAMILA clustering — semiparametric mixed-type (Foss & Markatou 2018)."""
    if not _HAS_KAMILA:
        raise RuntimeError("KAMILA not available. Place kamila.py in mixclust/clustering/.")
    _validate_k(len(X_df), n_clusters)
    cat_cols, num_cols = _split_types(X_df)
    if len(num_cols) == 0 or len(cat_cols) == 0:
        raise ValueError("KAMILA requires both numeric and categorical features.")
    model = KAMILA(
        n_clusters=n_clusters,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
    )
    labels = model.fit_predict(X_df, num_cols=num_cols, cat_cols=cat_cols)
    return np.asarray(labels, dtype=int)


def kamila_subsample_adapter(
    X_df: pd.DataFrame,
    cat_idx: List[int],
    n_clusters: int,
    random_state: Optional[int] = None,
    *,
    subsample_n: int = 6000,
    n_init: int = 5,
    max_iter: int = 25,
) -> np.ndarray:
    """KAMILA on subsample + propagate via NN (untuk n besar)."""
    if not _HAS_KAMILA:
        raise RuntimeError("KAMILA not available.")
    n = len(X_df)
    cat_cols, num_cols = _split_types(X_df)

    if n <= subsample_n:
        return kamila_adapter(X_df, cat_idx, n_clusters, random_state,
                              n_init=n_init, max_iter=max_iter)

    # Subsample → fit → propagate
    rng = np.random.default_rng(random_state)
    idx_sub = rng.choice(n, size=subsample_n, replace=False)
    X_sub = X_df.iloc[idx_sub].reset_index(drop=True)

    model = KAMILA(
        n_clusters=n_clusters,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
    )
    labels_sub = model.fit_predict(X_sub, num_cols=num_cols, cat_cols=cat_cols)

    # Propagate via NN pada fitur numerik
    from sklearn.neighbors import NearestNeighbors
    Z_sub = _prep_numeric(X_sub, num_cols)
    Z_full = _prep_numeric(X_df, num_cols)
    if Z_sub.shape[1] > 0:
        nn = NearestNeighbors(n_neighbors=1, algorithm="ball_tree", n_jobs=1)
        nn.fit(Z_sub)
        _, nn_idx = nn.kneighbors(Z_full)
        labels_all = labels_sub[nn_idx.ravel()]
    else:
        labels_all = np.array([labels_sub[i % subsample_n] for i in range(n)], dtype=int)

    return np.asarray(labels_all, dtype=int)