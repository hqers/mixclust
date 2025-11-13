# src/mixclust/metrics/lsil.py
from __future__ import annotations
import numpy as np
from mixclust.gower import gower_to_one_mixed


def lsil_using_prototypes_gower(
    labels,
    landmark_idx,
    prototypes,
    X_num,
    X_cat,
    num_min,
    num_max,
    *,
    feature_mask_num=None,
    feature_mask_cat=None,
    inv_rng=None,
    agg_mode: str = "topk",   # "mean" | "min" | "topk"
    topk: int = 1,
    verbose: bool = False
) -> float:
    """
    Compute Landmark-based Silhouette (L-Sil) using Gower distance
    with given cluster prototypes.

    Parameters
    ----------
    labels : array-like of shape (n,)
        Cluster labels.
    landmark_idx : list[int]
        Global row indices of landmarks (subset of points).
    prototypes : dict[int, list[int]]
        Dict mapping cluster -> list of prototype indices (global).
    X_num, X_cat, num_min, num_max : arrays
        Components for Gower computation.
    feature_mask_num, feature_mask_cat : np.ndarray[bool]
        Masks for valid numeric/categorical features.
    inv_rng : np.ndarray
        Cached inverse ranges for numeric part.
    agg_mode : str, default="topk"
        Aggregation of Gower distances to prototypes:
          "mean" = mean to all,
          "min"  = nearest,
          "topk" = mean of k nearest.
    topk : int
        Used only if agg_mode="topk".
    verbose : bool
        Print progress logs.

    Returns
    -------
    float : overall landmark-based silhouette score ∈ [-1, 1]
    """
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    n = len(labels)

    if len(landmark_idx) == 0 or n == 0:
        return np.nan

    # --- Distance from each landmark to prototypes ---
    proto_per_cluster = {c: prototypes.get(c, []) for c in uniq}
    D_intra = np.zeros(len(landmark_idx), dtype=np.float32)
    D_neigh = np.zeros(len(landmark_idx), dtype=np.float32)

    for i, lid in enumerate(landmark_idx):
        c_i = labels[lid]

        # --- distance to own cluster prototypes ---
        set_in = proto_per_cluster.get(c_i, [])
        if len(set_in) > 0:
            d_in = gower_to_one_mixed(
                X_num, X_cat, num_min, num_max, lid, set_in,
                feature_mask_num=feature_mask_num,
                feature_mask_cat=feature_mask_cat,
                inv_rng=inv_rng
            )
            if agg_mode == "min":
                D_intra[i] = np.min(d_in)
            elif agg_mode == "topk":
                k = min(topk, len(d_in))
                D_intra[i] = np.mean(np.partition(d_in, k-1)[:k])
            else:
                D_intra[i] = np.mean(d_in)
        else:
            D_intra[i] = np.nan

        # --- distance to nearest other cluster prototypes ---
        d_out_min = np.inf
        for c2, set_out in proto_per_cluster.items():
            if c2 == c_i or len(set_out) == 0:
                continue
            d_out = gower_to_one_mixed(
                X_num, X_cat, num_min, num_max, lid, set_out,
                feature_mask_num=feature_mask_num,
                feature_mask_cat=feature_mask_cat,
                inv_rng=inv_rng
            )
            d_val = (
                np.min(d_out)
                if agg_mode in ("min", "topk")
                else np.mean(d_out)
            )
            d_out_min = min(d_out_min, d_val)
        D_neigh[i] = d_out_min if np.isfinite(d_out_min) else np.nan

    # --- compute silhouette components ---
    valid_mask = np.isfinite(D_intra) & np.isfinite(D_neigh)
    if not np.any(valid_mask):
        return np.nan

    a = D_intra[valid_mask]
    b = D_neigh[valid_mask]
    s = (b - a) / np.maximum(np.maximum(a, b), 1e-12)
    s = np.clip(s, -1.0, 1.0)

    score = float(np.mean(s))
    if verbose:
        print(f"[L-Sil] valid={np.sum(valid_mask)}/{len(D_intra)}, mean={score:.6f}")
    return score


def lsil_fast_mean_only(
    labels,
    landmark_idx,
    X_num,
    X_cat,
    num_min,
    num_max,
    *,
    feature_mask_num=None,
    feature_mask_cat=None,
    inv_rng=None
) -> float:
    """
    Simplified L-Sil (no prototypes): use mean distance within vs between clusters.
    Faster but less accurate; used mainly for ablation.
    """
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    if len(landmark_idx) == 0:
        return np.nan

    intra, inter = [], []
    for i in landmark_idx:
        c_i = labels[i]
        others = np.where(labels != c_i)[0]
        same = np.where(labels == c_i)[0]
        if len(same) > 1:
            d_in = np.mean(gower_to_one_mixed(
                X_num, X_cat, num_min, num_max, i, same,
                feature_mask_num=feature_mask_num,
                feature_mask_cat=feature_mask_cat,
                inv_rng=inv_rng
            ))
            intra.append(d_in)
        if len(others) > 0:
            d_out = np.mean(gower_to_one_mixed(
                X_num, X_cat, num_min, num_max, i, others,
                feature_mask_num=feature_mask_num,
                feature_mask_cat=feature_mask_cat,
                inv_rng=inv_rng
            ))
            inter.append(d_out)
    if len(intra) == 0 or len(inter) == 0:
        return np.nan

    a, b = np.mean(intra), np.mean(inter)
    s = (b - a) / max(a, b)
    return float(np.clip(s, -1, 1))
