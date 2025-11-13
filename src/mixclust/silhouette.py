# src/mixclust/silhouette.py
from __future__ import annotations
import numpy as np
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.preprocessing import LabelEncoder
from mixclust.gower import gower_to_one_mixed

def full_silhouette_cosine( 
        X_unit: np.ndarray,
        labels,
        max_n: int = 2000,   # set ke None atau <=0 kalau mau selalu full
        seed: int = 42
    ): 
    """
    Return: (score, mode, n_eval)
      - mode = "full"  atau  f"sub({max_n})"
      - n_eval = jumlah sampel yang betul-betul dihitung
    """
    labels_arr = np.asarray(labels)
    y_num = LabelEncoder().fit_transform(labels_arr)
    n = len(X_unit)

    if (max_n is None) or (max_n <= 0) or (n <= max_n):
        D = pairwise_distances(X_unit, metric='cosine')  # O(n^2)
        ss = float(silhouette_score(D, y_num, metric='precomputed'))
        return ss, "full", n

    rng = np.random.RandomState(seed)
    idx = rng.choice(n, size=max_n, replace=False)
    D = pairwise_distances(X_unit[idx], metric='cosine')
    ss = float(silhouette_score(D, y_num[idx], metric='precomputed'))
    return ss, f"sub({max_n})", int(max_n)

def full_silhouette_gower(X_num, X_cat, num_min, num_max, labels,
                          feature_mask_num=None, feature_mask_cat=None, inv_rng=None):
    """
    Versi full (tanpa subsampling) dari silhouette score Gower
    """
    y = np.asarray(labels)
    n = len(y)
    idx = np.arange(n, dtype=int)

    D = np.zeros((n, n), dtype=np.float32)
    for r, i in enumerate(idx):
        D[r, :] = gower_to_one_mixed(
            X_num, X_cat, num_min, num_max, int(i), idx,
            feature_mask_num=feature_mask_num,
            feature_mask_cat=feature_mask_cat,
            inv_rng=inv_rng
        )

    ss = float(silhouette_score(D, LabelEncoder().fit_transform(y), metric="precomputed"))
    return ss, "full", n


def full_silhouette_gower_subsample(X_num, X_cat, num_min, num_max, labels,
                                    max_n=2000, seed=42,
                                    feature_mask_num=None, feature_mask_cat=None, inv_rng=None):
    y = np.asarray(labels); n = len(y)
    if (max_n is None) or (max_n <= 0) or (n <= max_n):
        idx = np.arange(n, dtype=int); mode = "full"
    else:
        rng = np.random.RandomState(seed)
        idx = np.sort(rng.choice(n, size=int(max_n), replace=False).astype(int))
        mode = f"sub({len(idx)})"

    s = len(idx)
    D = np.zeros((s, s), dtype=np.float32)
    for r, i in enumerate(idx):
        D[r, :] = gower_to_one_mixed(
            X_num, X_cat, num_min, num_max, int(i), idx,
            feature_mask_num=feature_mask_num, feature_mask_cat=feature_mask_cat,
            inv_rng=inv_rng
        )
    ss = float(silhouette_score(D, LabelEncoder().fit_transform(y[idx]), metric="precomputed"))
    return ss, mode, s
