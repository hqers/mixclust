# dynamic_clustering/src/mixclust/utils/cluster_adapters.py

from __future__ import annotations
from typing import List, Optional, Union
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering

# Catatan: kmodes/kprototypes opsional — pastikan terpasang bila mau pakai
try:
    from kmodes.kmodes import KModes
    from kmodes.kprototypes import KPrototypes
    _HAS_KMODES = True
except Exception:
    _HAS_KMODES = False


# =============== Util ringan ===============

def _split_types(X: pd.DataFrame):
    """Pisahkan kolom kategorik & numerik berdasarkan dtype DataFrame."""
    cat = X.select_dtypes(include=["bool", "object", "category"]).columns.tolist()
    num = X.select_dtypes(include=[np.number]).columns.tolist()
    return cat, num

def _prep_numeric(X: pd.DataFrame, num_cols: List[str]) -> np.ndarray:
    """Standardize numerik agar stabil untuk KMeans / KPrototypes komponen numerik."""
    if not num_cols:
        return np.zeros((len(X), 0), dtype=float)
    Z = X[num_cols].to_numpy(dtype=float, copy=True)
    Z = StandardScaler().fit_transform(Z)
    return Z

def _prep_categorical(X: pd.DataFrame, cat_cols: List[str]) -> np.ndarray:
    """Ubah kategorik ke array string (tanpa encoding) untuk KModes/KPrototypes."""
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


# =============== Adapter: KMeans (numerik penuh) ===============

def kmeans_adapter(
    X_df: pd.DataFrame,
    cat_idx: List[int],
    n_clusters: int,
    random_state: Optional[int] = None
) -> Union[np.ndarray, List]:
    """
    KMeans pada fitur numerik murni.
    Akan memaksa semua fitur numerik (cat_idx harus kosong).
    """
    _validate_k(len(X_df), n_clusters)
    cat_cols, num_cols = _split_types(X_df)
    if len(cat_cols) > 0 or len(cat_idx) > 0:
        raise ValueError("kmeans_adapter butuh semua fitur numerik (tanpa kolom kategorik).")
    if len(num_cols) == 0:
        # fallback aman: semua nol → 1 fitur dummy untuk hindari error KMeans
        Z = np.zeros((len(X_df), 1), dtype=float)
    else:
        Z = _prep_numeric(X_df, num_cols)
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    return model.fit_predict(Z)


# =============== Adapter: KModes (kategorik penuh) ===============

def kmodes_adapter(
    X_df: pd.DataFrame,
    cat_idx: List[int],
    n_clusters: int,
    random_state: Optional[int] = None
) -> Union[np.ndarray, List]:
    """
    KModes pada fitur kategorik murni.
    """
    if not _HAS_KMODES:
        raise ImportError("kmodes/kprototypes belum terpasang. `pip install kmodes`")
    _validate_k(len(X_df), n_clusters)

    cat_cols, num_cols = _split_types(X_df)
    if len(num_cols) > 0:
        raise ValueError("kmodes_adapter butuh semua fitur kategorik.")

    # Gunakan kolom kategorik dari DataFrame apa adanya
    Xc = X_df[cat_cols].astype(str)

    # KModes beberapa versi tidak punya arg random_state — jaga kompatibilitas
    try:
        model = KModes(n_clusters=n_clusters, init="Huang", n_init=10, verbose=0, random_state=random_state)
    except TypeError:
        model = KModes(n_clusters=n_clusters, init="Huang", n_init=10, verbose=0)

    return model.fit_predict(Xc)


# =============== Adapter: KPrototypes (campuran) ===============

def kprototypes_adapter(
    X_df: pd.DataFrame,
    cat_idx: List[int],
    n_clusters: int,
    random_state: Optional[int] = None,
    max_iter: int = 20,
    gamma: Optional[float] = None
) -> Union[np.ndarray, List]:
    """
    KPrototypes untuk fitur campuran (numerik + kategorik).
    - Numerik distandardisasi (StandardScaler)
    - Kategorik di-cast ke str
    - gamma=None → biarkan library menaksir otomatis
    """
    if not _HAS_KMODES:
        raise ImportError("kmodes/kprototypes belum terpasang. `pip install kmodes`")
    _validate_k(len(X_df), n_clusters)

    cat_cols = _cat_cols_from_idx(X_df, cat_idx)
    num_cols = [c for c in X_df.columns if c not in cat_cols]

    Z_num = _prep_numeric(X_df, num_cols)
    Z_cat = _prep_categorical(X_df, cat_cols)

    # ✅ Fallback: kalau tidak ada fitur numerik, pakai KModes saja
    if Z_num.shape[1] == 0 and Z_cat.shape[1] > 0:
        return kmodes_adapter(X_df[cat_cols], [], n_clusters, random_state)

    # Jika semua numerik (tanpa kategorik), tetap jalankan KPrototypes
    # dengan categorical index kosong agar antar-adapter uniform.
    if Z_cat.shape[1] == 0:
        Z = Z_num.astype(object)
        cat_anchor: List[int] = []
    else:
        Z = np.concatenate([Z_num, Z_cat], axis=1).astype(object)
        cat_anchor = list(range(Z_num.shape[1], Z_num.shape[1] + Z_cat.shape[1]))

    # Beberapa versi KPrototypes tidak menerima random_state; jaga kompatibilitas
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


# =============== Adapter: HAC-Gower (opsional) ===============

def _gower_pair(
    xi_num: np.ndarray, xj_num: np.ndarray,
    xi_cat: np.ndarray, xj_cat: np.ndarray,
    inv_rng: np.ndarray
) -> float:
    """Jarak Gower sederhana (rata-rata per fitur)."""
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
    """
    AgglomerativeClustering (average linkage) pada matriks jarak Gower (O(n^2)).
    Cocok untuk N kecil-menengah.
    """
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


# =============== Adapter: otomatis pilih algoritma ===============

def auto_adapter(
    X_df: pd.DataFrame,
    cat_idx: List[int],
    n_clusters: int,
    random_state: Optional[int] = None
) -> Union[np.ndarray, List]:
    """
    Pilih otomatis:
      - semua numerik → KMeans
      - semua kategorik → KModes
      - campuran → KPrototypes
    """
    cat_cols, num_cols = _split_types(X_df)
    if len(cat_cols) == 0:
        return kmeans_adapter(X_df, cat_idx, n_clusters, random_state)
    if len(num_cols) == 0:
        return kmodes_adapter(X_df, cat_idx, n_clusters, random_state)
    # campuran
    return kprototypes_adapter(
        X_df, cat_idx, n_clusters, random_state=random_state, max_iter=20, gamma=None
    )
