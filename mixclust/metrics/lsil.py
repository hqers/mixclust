# mixclust/metrics/lsil.py
# ---------------------------------------------------------------
# Landmark-based Silhouette (L-Sil)
#
# Reference:
#   Pratama, Lubis, Sembiring (2026). "L-Sil: Evaluating Cluster
#   Quality in Mixed-Type Data via Landmark-Based Silhouette and
#   Neighborhood Consistency."
#
# Paper formula (Eq. 2-3, Sec. 3):
#   a_L(x) = Agg{ d(x, ℓ) | ℓ ∈ L, lab(ℓ) = c }
#   b_L(x) = min_{c'≠c} Agg{ d(x, ℓ) | ℓ ∈ L, lab(ℓ) = c' }
#   s_L(x) = (b_L - a_L) / max(a_L, b_L)
#
# Practical settings (JDSA):
#   - Agg: top-r nearest landmarks with inverse-distance tapering,
#     r ∈ {3, 5}  (default r=3)
#   - |L| ≈ c√n,  c ∈ [1.5, 4]  (default c=3)
#   - Final aggregation: cluster-size weighted mean to preserve
#     the implicit size-proportional weighting of classical
#     Silhouette
# ---------------------------------------------------------------
from __future__ import annotations
import numpy as np
from typing import Optional, Dict, List, Sequence
from ..core.gower import gower_to_one_mixed


def _agg_gower_to_set(
    query_idx: int,
    target_idx: Sequence[int],
    X_num, X_cat, num_min, num_max,
    feature_mask_num=None,
    feature_mask_cat=None,
    inv_rng=None,
    mode: str = "topk",
    topk: int = 3,
) -> float:
    """
    Aggregate Gower distance from query to a set of target points.

    Parameters
    ----------
    mode : str
        "mean" = mean to all targets,
        "min"  = nearest target,
        "topk" = mean of k nearest targets (paper default, r ∈ {3,5}).
    topk : int
        Number of nearest targets for "topk" mode (paper default 3).
    """
    if len(target_idx) == 0:
        return np.nan

    d = gower_to_one_mixed(
        X_num, X_cat, num_min, num_max,
        query_idx, np.asarray(target_idx, dtype=int),
        feature_mask_num=feature_mask_num,
        feature_mask_cat=feature_mask_cat,
        inv_rng=inv_rng,
    )

    if mode == "min":
        return float(np.min(d))
    elif mode == "topk":
        k = min(topk, len(d))
        return float(np.mean(np.partition(d, k - 1)[:k]))
    else:  # "mean"
        return float(np.mean(d))


# ===============================================================
#  PRIMARY API — Paper-aligned L-Sil (landmark-based)
# ===============================================================

def lsil_landmark(
    labels,
    landmark_idx: Sequence[int],
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
    verbose: bool = False,
) -> float:
    """
    Compute L-Sil using landmarks as both evaluation points AND
    cluster representatives (paper Eq. 2-3).

    For each landmark ℓ in cluster c:
      a_L(ℓ) = Agg{ d(ℓ, ℓ') | ℓ' ∈ L, lab(ℓ') = c, ℓ' ≠ ℓ }
      b_L(ℓ) = min_{c'≠c} Agg{ d(ℓ, ℓ') | ℓ' ∈ L, lab(ℓ') = c' }
      s_L(ℓ) = (b_L - a_L) / max(a_L, b_L)

    Parameters
    ----------
    labels : array-like (n,)
        Cluster labels for ALL data points.
    landmark_idx : sequence of int
        Global indices of landmark points.
    X_num, X_cat, num_min, num_max : arrays
        Gower distance components.
    feature_mask_num, feature_mask_cat : bool arrays, optional
        Feature masks for subsetting.
    inv_rng : array, optional
        Cached 1/(max-min) for numeric features.
    agg_mode : str
        "topk" (paper default) | "mean" | "min".
    topk : int
        r in paper practical settings (default 3, paper r ∈ {3,5}).
    weighted : bool
        If True, use cluster-size weighted mean (paper practical setting).
        If False, simple mean (paper Eq. 3 formal).
    verbose : bool

    Returns
    -------
    float : L-Sil score ∈ [-1, 1]
    """
    labels = np.asarray(labels)
    landmark_idx = np.asarray(landmark_idx, dtype=int)
    n = len(labels)

    if len(landmark_idx) == 0 or n == 0:
        return np.nan

    # Build per-cluster landmark index:  cluster_c -> [indices in L]
    lm_labels = labels[landmark_idx]
    uniq_clusters = np.unique(labels)

    landmarks_per_cluster: Dict[int, List[int]] = {}
    for c in uniq_clusters:
        landmarks_per_cluster[c] = landmark_idx[lm_labels == c].tolist()

    # Cluster sizes for weighted aggregation
    if weighted:
        _, counts = np.unique(labels, return_counts=True)
        size_map = dict(zip(np.unique(labels), counts.astype(float)))
    else:
        size_map = None

    S = np.zeros(len(landmark_idx), dtype=np.float64)
    W = np.ones(len(landmark_idx), dtype=np.float64)

    for i, lid in enumerate(landmark_idx):
        c_i = labels[lid]

        # a_L(ℓ): distance to same-cluster landmarks (exclude self)
        same_lm = [j for j in landmarks_per_cluster.get(c_i, []) if j != lid]
        a = _agg_gower_to_set(
            lid, same_lm,
            X_num, X_cat, num_min, num_max,
            feature_mask_num, feature_mask_cat, inv_rng,
            mode=agg_mode, topk=topk,
        )

        # b_L(ℓ): min over other clusters
        b = np.inf
        for c2, other_lm in landmarks_per_cluster.items():
            if c2 == c_i or len(other_lm) == 0:
                continue
            b_c2 = _agg_gower_to_set(
                lid, other_lm,
                X_num, X_cat, num_min, num_max,
                feature_mask_num, feature_mask_cat, inv_rng,
                mode=agg_mode, topk=topk,
            )
            if not np.isnan(b_c2):
                b = min(b, b_c2)

        if np.isnan(a) or not np.isfinite(b):
            S[i] = 0.0
        else:
            denom = max(a, b)
            S[i] = (b - a) / denom if denom > 1e-12 else 0.0

        if weighted and size_map is not None:
            W[i] = size_map.get(c_i, 1.0)

    S = np.clip(S, -1.0, 1.0)

    if weighted:
        score = float(np.sum(W * S) / max(np.sum(W), 1e-12))
    else:
        score = float(np.mean(S))

    if verbose:
        valid = np.sum(np.isfinite(S))
        print(f"[L-Sil] landmarks={len(landmark_idx)}, valid={valid}, "
              f"score={score:.6f}, mode={agg_mode}, topk={topk}, "
              f"weighted={weighted}")

    return score


# ===============================================================
#  BACKWARD-COMPATIBLE WRAPPER — old prototype-based API
# ===============================================================

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
    agg_mode: str = "topk",
    topk: int = 3,
    verbose: bool = False,
    # New parameters (backward-compatible defaults)
    use_landmarks_as_references: bool = True,
    weighted: bool = True,
) -> float:
    """
    Compute L-Sil — backward-compatible wrapper.

    When use_landmarks_as_references=True (default, paper-aligned):
      Uses landmarks as cluster representatives.
      The `prototypes` argument is IGNORED.

    When use_landmarks_as_references=False (legacy mode):
      Uses separate prototypes as in the original implementation.
      NOT aligned with paper — kept for ablation only.

    Parameters
    ----------
    labels : array-like (n,)
    landmark_idx : list[int]
    prototypes : dict[int, list[int]]
        Ignored when use_landmarks_as_references=True.
    X_num, X_cat, num_min, num_max : arrays
    feature_mask_num, feature_mask_cat : bool arrays
    inv_rng : array
    agg_mode : str, default="topk"
    topk : int, default=3  (paper: r ∈ {3, 5})
    verbose : bool
    use_landmarks_as_references : bool, default=True
        True = paper-aligned (landmarks as references).
        False = legacy (prototypes as references).
    weighted : bool, default=True
        Cluster-size weighted mean.

    Returns
    -------
    float : L-Sil score ∈ [-1, 1]
    """
    if use_landmarks_as_references:
        return lsil_landmark(
            labels, landmark_idx,
            X_num, X_cat, num_min, num_max,
            feature_mask_num=feature_mask_num,
            feature_mask_cat=feature_mask_cat,
            inv_rng=inv_rng,
            agg_mode=agg_mode,
            topk=topk,
            weighted=weighted,
            verbose=verbose,
        )

    # --- Legacy prototype-based mode (for ablation) ---
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    n = len(labels)

    if len(landmark_idx) == 0 or n == 0:
        return np.nan

    proto_per_cluster = {c: prototypes.get(c, []) for c in uniq}

    if weighted:
        _, counts = np.unique(labels, return_counts=True)
        size_map = dict(zip(uniq, counts.astype(float)))
    else:
        size_map = None

    S = np.zeros(len(landmark_idx), dtype=np.float64)
    W = np.ones(len(landmark_idx), dtype=np.float64)

    for i, lid in enumerate(landmark_idx):
        c_i = labels[lid]

        a = _agg_gower_to_set(
            lid, proto_per_cluster.get(c_i, []),
            X_num, X_cat, num_min, num_max,
            feature_mask_num, feature_mask_cat, inv_rng,
            mode=agg_mode, topk=topk,
        )

        b = np.inf
        for c2, set_out in proto_per_cluster.items():
            if c2 == c_i or len(set_out) == 0:
                continue
            b_c2 = _agg_gower_to_set(
                lid, set_out,
                X_num, X_cat, num_min, num_max,
                feature_mask_num, feature_mask_cat, inv_rng,
                mode=agg_mode, topk=topk,
            )
            if not np.isnan(b_c2):
                b = min(b, b_c2)

        if np.isnan(a) or not np.isfinite(b):
            S[i] = 0.0
        else:
            denom = max(a, b)
            S[i] = (b - a) / denom if denom > 1e-12 else 0.0

        if weighted and size_map is not None:
            W[i] = size_map.get(c_i, 1.0)

    S = np.clip(S, -1.0, 1.0)

    if weighted:
        score = float(np.sum(W * S) / max(np.sum(W), 1e-12))
    else:
        score = float(np.mean(S))

    if verbose:
        print(f"[L-Sil legacy] valid={len(landmark_idx)}, score={score:.6f}")

    return score


# ===============================================================
#  ABLATION VARIANT — simplified mean-only (no prototypes)
# ===============================================================

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
    inv_rng=None,
) -> float:
    """
    Simplified L-Sil (no prototypes): mean distance within vs between.
    Kept for ablation study only.
    """
    labels = np.asarray(labels)
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
                inv_rng=inv_rng,
            ))
            intra.append(d_in)
        if len(others) > 0:
            d_out = np.mean(gower_to_one_mixed(
                X_num, X_cat, num_min, num_max, i, others,
                feature_mask_num=feature_mask_num,
                feature_mask_cat=feature_mask_cat,
                inv_rng=inv_rng,
            ))
            inter.append(d_out)
    if len(intra) == 0 or len(inter) == 0:
        return np.nan

    a, b = np.mean(intra), np.mean(inter)
    s = (b - a) / max(a, b)
    return float(np.clip(s, -1, 1))


# ===============================================================
#  UTILITY — Adaptive landmark count (√n law)
# ===============================================================

def adaptive_landmark_count(
    n: int,
    K: int = 3,
    c: float = 3.0,
    per_cluster_min: int = 3,
    cap_frac: float = 0.2,
) -> int:
    """
    Compute landmark budget |L| following the √n law from paper.

    |L| = max(K * per_cluster_min, min(c * √n, cap_frac * n))

    Paper practical settings: c ∈ [1.5, 4], default c=3.
    Complexity: O(n * |L|) = O(n * c√n) = O(c * n^{3/2}).

    Parameters
    ----------
    n : int
        Total number of data points.
    K : int
        Number of clusters.
    c : float
        Scaling constant (default 3.0, paper c ∈ [1.5, 4]).
    per_cluster_min : int
        Minimum landmarks per cluster.
    cap_frac : float
        Upper bound as fraction of n (default 0.2).

    Returns
    -------
    int : number of landmarks
    """
    m_sqrt = int(c * np.sqrt(n))
    m_cap = int(cap_frac * n)
    m_floor = K * per_cluster_min
    return max(m_floor, min(m_sqrt, m_cap, n))
