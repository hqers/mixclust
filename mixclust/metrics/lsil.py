# mixclust/metrics/lsil.py — v1.2.0
# paper-aligned: c*sqrt(n) landmark count + vectorized distance cache O(n*|L|*p)
from __future__ import annotations
import numpy as np
from typing import Optional, Sequence

from ..core.gower import gower_to_one_mixed, gower_distances_to_landmarks


def adaptive_landmark_count(
    n: int, K: int = 3, c: float = 3.0,
    per_cluster_min: int = 3, cap_frac: float = 0.2,
) -> int:
    """
    |L| = max(K*per_cluster_min, min(c*sqrt(n), cap_frac*n, n))
    Theorem 1 (JDSA paper): O(c*p*n^{3/2}) total complexity.
    """
    return max(K * per_cluster_min, min(int(c * np.sqrt(n)), int(cap_frac * n), n))


def _aggregate(dists: np.ndarray, mode: str, topk: int) -> float:
    if len(dists) == 0:
        return np.nan
    if mode == "min":
        return float(np.min(dists))
    if mode == "topk":
        k = min(topk, len(dists))
        return float(np.mean(np.partition(dists, k - 1)[:k]))
    return float(np.mean(dists))


def compute_lsil_from_D(
    D: np.ndarray,
    labels: np.ndarray,
    landmark_labels: np.ndarray,
    *,
    agg_mode: str = "topk",
    topk: int = 3,
    weighted: bool = True,
) -> tuple[float, np.ndarray]:
    """
    Hitung L-Sil dari distance matrix D (n x |L|).
    a_L(x) = Agg{D[x,l] | lab(l)=c(x)}
    b_L(x) = min_{c'!=c} Agg{D[x,l] | lab(l)=c'}
    s_L(x) = (b-a)/max(a,b)
    """
    n = D.shape[0]
    unique_labels = np.unique(labels)
    lm_masks = {k: np.where(landmark_labels == k)[0] for k in unique_labels}

    size_map = None
    if weighted:
        _, counts = np.unique(labels, return_counts=True)
        size_map = dict(zip(np.unique(labels), counts.astype(float)))

    per_sample = np.zeros(n, dtype=np.float64)
    for i in range(n):
        c = labels[i]
        dists = D[i]
        own_d = dists[lm_masks.get(c, np.array([], dtype=int))]
        own_d = own_d[own_d > 1e-12] if len(own_d) > 0 else own_d
        if len(own_d) == 0:
            continue
        a = _aggregate(own_d, agg_mode, topk)
        b = np.inf
        for k in unique_labels:
            if k == c:
                continue
            oi = lm_masks.get(k, np.array([], dtype=int))
            if len(oi) == 0:
                continue
            b = min(b, _aggregate(dists[oi], agg_mode, topk))
        if b < np.inf:
            denom = max(a, b)
            per_sample[i] = (b - a) / denom if denom > 1e-12 else 0.0

    per_sample = np.clip(per_sample, -1.0, 1.0)
    if weighted and size_map:
        w = np.array([size_map.get(labels[i], 1.0) for i in range(n)])
        score = float(np.sum(w * per_sample) / max(np.sum(w), 1e-12))
    else:
        score = float(np.mean(per_sample))
    return score, per_sample


def lsil_using_landmarks(
    labels: Sequence,
    landmark_idx: np.ndarray,
    X_num: np.ndarray,
    X_cat: np.ndarray,
    num_min: np.ndarray,
    num_max: np.ndarray,
    *,
    feature_mask_num: Optional[np.ndarray] = None,
    feature_mask_cat: Optional[np.ndarray] = None,
    inv_rng: Optional[np.ndarray] = None,
    agg_mode: str = "topk",
    topk: int = 3,
    weighted: bool = True,
    return_D: bool = False,
):
    """
    Compute L-Sil — vectorized O(n*|L|*p), paper-aligned.
    Landmark sebagai referensi klaster, bukan prototype.
    return_D=True: kembalikan (score, D) agar D bisa reuse untuk LNC*.
    """
    labels = np.asarray(labels)
    landmark_idx = np.asarray(landmark_idx, dtype=int)
    if len(landmark_idx) == 0 or len(labels) == 0:
        return (np.nan, None) if return_D else np.nan

    D = gower_distances_to_landmarks(
        X_num, X_cat, num_min, num_max, landmark_idx,
        feature_mask_num=feature_mask_num,
        feature_mask_cat=feature_mask_cat,
        inv_rng=inv_rng,
    )
    landmark_labels = labels[landmark_idx]
    score, _ = compute_lsil_from_D(D, labels, landmark_labels,
                                    agg_mode=agg_mode, topk=topk, weighted=weighted)
    return (score, D) if return_D else score


# backward-compat — semua caller lama masih bisa pakai ini
def lsil_using_prototypes_gower(
    labels: Sequence,
    landmark_idx: np.ndarray,
    X_num: np.ndarray,
    X_cat: np.ndarray,
    num_min: np.ndarray,
    num_max: np.ndarray,
    *,
    feature_mask_num: Optional[np.ndarray] = None,
    feature_mask_cat: Optional[np.ndarray] = None,
    inv_rng: Optional[np.ndarray] = None,
    proto_sample_cap: Optional[int] = None,     # diabaikan
    per_cluster_proto: Optional[int] = None,    # diabaikan
    select_landmarks_fn=None,                   # diabaikan
    agg_mode: str = "topk",
    topk: int = 3,
    weighted: bool = True,
) -> float:
    """Backward-compat wrapper → lsil_using_landmarks(). Proto params diabaikan."""
    return lsil_using_landmarks(
        labels, landmark_idx, X_num, X_cat, num_min, num_max,
        feature_mask_num=feature_mask_num,
        feature_mask_cat=feature_mask_cat,
        inv_rng=inv_rng,
        agg_mode=agg_mode, topk=topk, weighted=weighted,
    )


def lsil_fast_mean_only(
    labels: Sequence,
    landmark_idx: np.ndarray,
    X_num: np.ndarray,
    X_cat: np.ndarray,
    num_min: np.ndarray,
    num_max: np.ndarray,
    *,
    feature_mask_num: Optional[np.ndarray] = None,
    feature_mask_cat: Optional[np.ndarray] = None,
    inv_rng: Optional[np.ndarray] = None,
    weighted: bool = True,
) -> float:
    """Fast variant (agg_mode=mean) untuk screening K."""
    return lsil_using_landmarks(
        labels, landmark_idx, X_num, X_cat, num_min, num_max,
        feature_mask_num=feature_mask_num,
        feature_mask_cat=feature_mask_cat,
        inv_rng=inv_rng,
        agg_mode="mean", topk=3, weighted=weighted,
    )
