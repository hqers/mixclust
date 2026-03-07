# dynamic_clustering/src/mixclust/utils/structural_control.py
#
# Structural Control: post-clustering refinement using LNC* and L-Sil
# Operations: Retain / Split / Merge
# ─────────────────────────────────────────────────────────────────────

from __future__ import annotations
import numpy as np
import pandas as pd
from time import perf_counter
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

# Evaluation metrics
try:
    from mixclust.metrics.lnc_star import lnc_star
except Exception:
    lnc_star = None

try:
    from mixclust.metrics.lsil import lsil_using_prototypes_gower
except Exception:
    lsil_using_prototypes_gower = None

try:
    from mixclust.knn_index import KNNIndex
except Exception:
    KNNIndex = None

from mixclust.aufs_samba.preprocess import prepare_mixed_arrays_no_label
from mixclust.prototypes import build_prototypes_by_cluster_gower
from mixclust.landmarks import select_landmarks_cluster_aware, select_landmarks_kcenter

try:
    from mixclust.features import build_features
except Exception:
    build_features = None


# ================================================================
# Data classes
# ================================================================

@dataclass
class ClusterDiagnostic:
    """Per-cluster diagnostic from structural control."""
    cluster_id: int = -1
    size: int = 0
    fraction: float = 0.0
    lnc_mean: float = np.nan         # mean LNC* of landmarks in this cluster
    sil_contrib: float = np.nan      # mean silhouette contribution of this cluster
    action: str = "retain"           # "retain" | "split" | "merge"
    merge_with: Optional[int] = None # target cluster for merge
    reason: str = ""


@dataclass
class StructuralControlResult:
    """Full output from structural control."""
    # Global
    lnc_global: float = np.nan
    lsil_global: float = np.nan
    passed: bool = True
    n_clusters_before: int = 0
    n_clusters_after: int = 0
    labels_refined: Optional[List[int]] = None

    # Per-cluster diagnostics
    diagnostics: List[Dict[str, Any]] = field(default_factory=list)

    # Actions taken
    n_retained: int = 0
    n_split: int = 0
    n_merged: int = 0
    actions_log: List[str] = field(default_factory=list)

    # Meta
    timing_s: float = 0.0
    message: str = ""


# ================================================================
# Per-cluster LNC* computation
# ================================================================

def _compute_per_cluster_lnc(
    X_unit: np.ndarray,
    labels: np.ndarray,
    L_idx: List[int],
    knn_index,
    *,
    X_num: np.ndarray,
    X_cat: np.ndarray,
    num_min: np.ndarray,
    num_max: np.ndarray,
    mask_num, mask_cat, inv_rng,
    lnc_k: int = 50,
    lnc_alpha: float = 0.7,
    M_candidates: int = 300,
) -> Dict[int, float]:
    """
    Compute mean LNC* per cluster by averaging landmark scores
    within each cluster.
    """
    if lnc_star is None:
        return {}

    labels = np.asarray(labels)
    unique_clusters = np.unique(labels)
    L_arr = np.asarray(L_idx, dtype=int)

    # We compute LNC* globally but then aggregate per cluster
    # by mapping each landmark to its cluster
    from sklearn.preprocessing import LabelEncoder
    from scipy.stats import entropy as _entropy

    lab_num = labels.astype(int) if labels.dtype.kind in "iu" else LabelEncoder().fit_transform(labels)
    K = int(lab_num.max()) + 1

    k = max(5, min(lnc_k, max(5, len(labels) - 1)))
    M = max(M_candidates, 3 * k)
    M = min(M, max(50, int(0.05 * len(labels))))

    # Collect per-landmark scores grouped by cluster
    cluster_scores: Dict[int, List[float]] = {int(c): [] for c in unique_clusters}

    for i in L_arr:
        li = int(lab_num[i])

        # 1) ANN candidates
        try:
            cand_idx, _ = knn_index.kneighbors_idx_dist(int(i), int(M))
            cand_idx = np.asarray(cand_idx).reshape(-1)
            cand_idx = cand_idx[cand_idx != int(i)]
            if cand_idx.size == 0:
                continue
        except Exception:
            continue

        # 2) Gower re-rank
        try:
            from mixclust.gower import rerank_gower_from_candidates
            nn_idx, nn_d = rerank_gower_from_candidates(
                int(i), cand_idx, int(k),
                X_num, X_cat, num_min, num_max,
                feature_mask_num=mask_num,
                feature_mask_cat=mask_cat,
                inv_rng=inv_rng,
            )
            if nn_idx.size == 0:
                continue
        except Exception:
            continue

        # 3) NC component
        nlab = lab_num[nn_idx]
        counts_k = np.bincount(nlab, minlength=K).astype(float)
        p = counts_k / max(1.0, counts_k.sum())
        nz = p > 0
        H = float(-np.sum(p[nz] * np.log(p[nz] + 1e-12)))
        m_nz = int(nz.sum())
        H += (m_nz - 1) / (2.0 * max(1, nn_idx.size))
        NC = 1.0 if K <= 1 else float(1.0 - H / np.log(K))
        NC = float(np.clip(NC, 0.0, 1.0))

        # 4) Delta component
        intra = nn_d[nlab == li]
        inter = nn_d[nlab != li]
        d_intra = float(np.mean(intra)) if intra.size > 0 else float(np.mean(nn_d))
        d_inter = float(np.mean(inter)) if inter.size > 0 else d_intra
        q5, q95 = np.percentile(nn_d, [5, 95])
        iqr = max(1e-9, float(q95 - q5))
        Delta = float(np.clip((d_inter - d_intra) / iqr, 0.0, 1.0))

        v = lnc_alpha * NC + (1.0 - lnc_alpha) * Delta
        v = float(np.clip(v, 0.0, 1.0))

        cluster_scores[li].append(v)

    # Average per cluster
    result = {}
    for c in unique_clusters:
        scores = cluster_scores.get(int(c), [])
        result[int(c)] = float(np.mean(scores)) if scores else np.nan

    return result


# ================================================================
# Per-cluster silhouette contribution
# ================================================================

def _compute_per_cluster_silhouette(
    X_df: pd.DataFrame,
    labels: np.ndarray,
    L_idx: List[int],
    prototypes: Dict[int, List[int]],
    *,
    X_num, X_cat, num_min, num_max,
    mask_num, mask_cat, inv_rng,
) -> Dict[int, float]:
    """
    Compute mean silhouette contribution per cluster using L-Sil landmarks.
    """
    if lsil_using_prototypes_gower is None:
        return {}

    from mixclust.gower import gower_to_one_mixed

    labels = np.asarray(labels)
    unique_clusters = np.unique(labels)
    proto_per_cluster = {c: prototypes.get(c, []) for c in unique_clusters}

    cluster_sil: Dict[int, List[float]] = {int(c): [] for c in unique_clusters}

    for lid in L_idx:
        c_i = int(labels[lid])

        # a(lid) — distance to own cluster prototypes
        set_in = proto_per_cluster.get(c_i, [])
        if len(set_in) > 0:
            d_in = gower_to_one_mixed(
                X_num, X_cat, num_min, num_max, lid, set_in,
                feature_mask_num=mask_num, feature_mask_cat=mask_cat, inv_rng=inv_rng
            )
            a_val = float(np.mean(d_in))
        else:
            a_val = np.nan

        # b(lid) — distance to nearest other cluster
        b_val = np.inf
        for c2, set_out in proto_per_cluster.items():
            if c2 == c_i or len(set_out) == 0:
                continue
            d_out = gower_to_one_mixed(
                X_num, X_cat, num_min, num_max, lid, set_out,
                feature_mask_num=mask_num, feature_mask_cat=mask_cat, inv_rng=inv_rng
            )
            b_val = min(b_val, float(np.mean(d_out)))

        if np.isfinite(a_val) and np.isfinite(b_val):
            s = (b_val - a_val) / max(max(a_val, b_val), 1e-12)
            cluster_sil[c_i].append(float(np.clip(s, -1.0, 1.0)))

    result = {}
    for c in unique_clusters:
        vals = cluster_sil.get(int(c), [])
        result[int(c)] = float(np.mean(vals)) if vals else np.nan

    return result


# ================================================================
# Split: re-cluster large, high-variance cluster locally
# ================================================================

def _split_cluster(
    X_df: pd.DataFrame,
    labels: np.ndarray,
    cluster_id: int,
    cat_cols: List[str],
    random_state: int = 42,
) -> np.ndarray:
    """
    Split one cluster into two by running local clustering (K=2)
    on its members only. Returns updated global labels.
    """
    from mixclust.utils.cluster_adapters import auto_adapter

    labels = np.asarray(labels).copy()
    mask = labels == cluster_id
    idx_members = np.where(mask)[0]

    if len(idx_members) < 4:
        return labels  # too small to split

    df_local = X_df.iloc[idx_members].reset_index(drop=True)
    cat_idx_local = [df_local.columns.get_loc(c) for c in df_local.columns if c in cat_cols]

    try:
        local_labels = auto_adapter(df_local, cat_idx_local, 2, random_state)
    except Exception:
        return labels  # split failed, keep original

    # Assign new labels: keep cluster_id for sub-cluster 0,
    # create new ID for sub-cluster 1
    new_id = int(labels.max()) + 1
    for i, member_idx in enumerate(idx_members):
        if local_labels[i] == 1:
            labels[member_idx] = new_id

    return labels


# ================================================================
# Merge: combine two small/weak clusters
# ================================================================

def _merge_clusters(
    labels: np.ndarray,
    cluster_a: int,
    cluster_b: int,
) -> np.ndarray:
    """
    Merge cluster_b into cluster_a by relabeling.
    """
    labels = np.asarray(labels).copy()
    labels[labels == cluster_b] = cluster_a
    return labels


# ================================================================
# Main: Structural Control
# ================================================================

def structural_control(
    X_df: pd.DataFrame,
    labels: np.ndarray,
    cat_cols: List[str],
    *,
    # Thresholds
    lnc_retain_threshold: float = 0.45,
    sil_retain_threshold: float = -0.05,
    split_size_factor: float = 2.0,
    split_lnc_threshold: float = 0.35,
    merge_min_size_frac: float = 0.02,
    merge_max_inter_dist: float = 0.3,
    merge_lnc_threshold: float = 0.35,

    # LNC* params
    lnc_k: int = 50,
    lnc_alpha: float = 0.7,
    M_candidates: int = 300,

    # Landmark params
    landmark_mode: str = "cluster_aware",
    lm_max_frac: float = 0.2,
    lm_per_cluster_min: int = 3,
    central_frac: float = 0.8,
    boundary_frac: float = 0.2,

    # General
    try_hnsw: bool = True,
    random_state: int = 42,
    max_iterations: int = 2,
    verbose: bool = True,
) -> StructuralControlResult:
    """
    Post-clustering structural refinement using LNC* and L-Sil signals.

    Three operations:
    - RETAIN: keep cluster if LNC* >= threshold AND sil_contrib >= threshold
    - SPLIT:  split cluster if size >> mean AND LNC* low (hidden sub-structure)
    - MERGE:  merge two clusters if both small AND inter-distance low AND LNC* low

    Parameters
    ----------
    X_df : pd.DataFrame
        Feature subset DataFrame.
    labels : np.ndarray
        Initial cluster labels from Auto-K.
    cat_cols : list[str]
        Categorical columns.
    lnc_retain_threshold : float
        Min LNC* to retain a cluster without question.
    sil_retain_threshold : float
        Min silhouette contribution to retain.
    split_size_factor : float
        Cluster is "large" if size > split_size_factor * mean_size.
    split_lnc_threshold : float
        LNC* below this in a large cluster triggers split.
    merge_min_size_frac : float
        Cluster is "small" if size < merge_min_size_frac * n.
    merge_max_inter_dist : float
        Max inter-cluster distance to allow merge (placeholder).
    merge_lnc_threshold : float
        LNC* below this makes cluster eligible for merge.
    max_iterations : int
        Max passes of retain/split/merge (usually 1-2 is enough).

    Returns
    -------
    StructuralControlResult
    """
    t0 = perf_counter()
    labels = np.asarray(labels).copy()
    n = len(labels)

    result = StructuralControlResult(
        n_clusters_before=len(np.unique(labels)),
        labels_refined=labels.tolist(),
    )

    # Guard: required modules
    if build_features is None or KNNIndex is None:
        result.message = "Required modules not available; structural control skipped."
        result.timing_s = perf_counter() - t0
        return result

    if n == 0 or len(np.unique(labels)) < 2:
        result.message = "Insufficient data or clusters."
        result.timing_s = perf_counter() - t0
        return result

    for iteration in range(max_iterations):
        if verbose:
            print(f"\n[STRUCTURAL CONTROL] Iteration {iteration + 1}/{max_iterations}")

        unique_clusters = np.unique(labels)
        K = len(unique_clusters)
        if K < 2:
            break

        # ── Build evaluation components ──
        try:
            X_unit, _, _ = build_features(X_df, label_col=None, scaler_type="standard", unit_norm=True)
        except Exception as e:
            result.message = f"build_features failed: {e}"
            break

        X_num, X_cat, num_min, num_max, mask_num, mask_cat, inv_rng = \
            prepare_mixed_arrays_no_label(X_df)

        # Landmarks
        m = min(int(lm_max_frac * n), max(20, n // 10))
        if landmark_mode == "cluster_aware":
            L = select_landmarks_cluster_aware(
                X_unit, labels, m,
                central_frac=central_frac, boundary_frac=boundary_frac,
                per_cluster_min=lm_per_cluster_min, seed=random_state
            )
        else:
            L = select_landmarks_kcenter(X_unit, m=m, seed=random_state)

        # KNN index
        try:
            knn_index = KNNIndex(X_unit, try_hnsw=try_hnsw, verbose=False)
        except Exception:
            result.message = "KNNIndex failed."
            break

        # Prototypes
        prototypes = build_prototypes_by_cluster_gower(
            labels, X_num, X_cat, num_min, num_max,
            per_cluster=1, sample_cap=200, seed=random_state,
            feature_mask_num=mask_num, feature_mask_cat=mask_cat, inv_rng=inv_rng
        )

        # ── Per-cluster LNC* ──
        per_cluster_lnc = _compute_per_cluster_lnc(
            X_unit, labels, L, knn_index,
            X_num=X_num, X_cat=X_cat,
            num_min=num_min, num_max=num_max,
            mask_num=mask_num, mask_cat=mask_cat, inv_rng=inv_rng,
            lnc_k=lnc_k, lnc_alpha=lnc_alpha, M_candidates=M_candidates,
        )

        # ── Per-cluster silhouette contribution ──
        per_cluster_sil = _compute_per_cluster_silhouette(
            X_df, labels, L, prototypes,
            X_num=X_num, X_cat=X_cat,
            num_min=num_min, num_max=num_max,
            mask_num=mask_num, mask_cat=mask_cat, inv_rng=inv_rng,
        )

        # ── Global scores ──
        lnc_vals = [v for v in per_cluster_lnc.values() if np.isfinite(v)]
        sil_vals = [v for v in per_cluster_sil.values() if np.isfinite(v)]
        result.lnc_global = float(np.mean(lnc_vals)) if lnc_vals else np.nan
        result.lsil_global = float(np.mean(sil_vals)) if sil_vals else np.nan

        # ── Cluster sizes ──
        cluster_sizes = {int(c): int(np.sum(labels == c)) for c in unique_clusters}
        mean_size = float(np.mean(list(cluster_sizes.values())))

        # ── Diagnose each cluster ──
        diagnostics: List[ClusterDiagnostic] = []
        actions_this_round: List[Tuple[str, int, Optional[int]]] = []

        for c in sorted(unique_clusters):
            c = int(c)
            size = cluster_sizes[c]
            frac = size / n
            lnc_c = per_cluster_lnc.get(c, np.nan)
            sil_c = per_cluster_sil.get(c, np.nan)

            diag = ClusterDiagnostic(
                cluster_id=c, size=size, fraction=frac,
                lnc_mean=lnc_c, sil_contrib=sil_c,
            )

            # ── Decision logic ──

            # SPLIT check: large cluster with low LNC* (hidden sub-structure)
            if (size > split_size_factor * mean_size
                    and np.isfinite(lnc_c)
                    and lnc_c < split_lnc_threshold):
                diag.action = "split"
                diag.reason = (f"Size={size} > {split_size_factor:.1f}×mean={mean_size:.0f}, "
                               f"LNC*={lnc_c:.3f} < {split_lnc_threshold:.2f}")
                actions_this_round.append(("split", c, None))

            # MERGE check: small cluster with low LNC*
            elif (frac < merge_min_size_frac
                  and np.isfinite(lnc_c)
                  and lnc_c < merge_lnc_threshold):
                diag.action = "merge"
                diag.reason = (f"Frac={frac:.3f} < {merge_min_size_frac:.3f}, "
                               f"LNC*={lnc_c:.3f} < {merge_lnc_threshold:.2f}")
                # Find nearest cluster to merge with (by silhouette proximity)
                # Simple heuristic: merge with the cluster that has highest sil_contrib
                candidates = [
                    (cc, per_cluster_sil.get(cc, -np.inf))
                    for cc in unique_clusters if cc != c
                ]
                if candidates:
                    best_target = max(candidates, key=lambda x: x[1])[0]
                    diag.merge_with = int(best_target)
                    actions_this_round.append(("merge", c, int(best_target)))

            # RETAIN check
            elif (np.isfinite(lnc_c) and lnc_c >= lnc_retain_threshold
                  and (np.isnan(sil_c) or sil_c >= sil_retain_threshold)):
                diag.action = "retain"
                diag.reason = (f"LNC*={lnc_c:.3f} >= {lnc_retain_threshold:.2f}, "
                               f"Sil={sil_c:.3f}")

            else:
                # Default: retain with note
                diag.action = "retain"
                diag.reason = f"No action triggered (LNC*={lnc_c:.3f}, Sil={sil_c:.3f})"

            diagnostics.append(diag)

            if verbose:
                icon = {"retain": "✅", "split": "🔀", "merge": "🔗"}.get(diag.action, "❓")
                print(f"  {icon} Cluster {c}: size={size} ({frac:.1%}) | "
                      f"LNC*={lnc_c:.3f} | Sil={sil_c:.3f} → {diag.action.upper()} "
                      f"{'→ '+str(diag.merge_with) if diag.merge_with is not None else ''}")

        # ── Execute actions ──
        any_change = False

        # Execute splits first (they increase K)
        for action, cid, _ in actions_this_round:
            if action == "split":
                labels_new = _split_cluster(X_df, labels, cid, cat_cols, random_state)
                if not np.array_equal(labels_new, labels):
                    labels = labels_new
                    any_change = True
                    result.n_split += 1
                    result.actions_log.append(f"SPLIT cluster {cid}")
                    if verbose:
                        print(f"    → Split cluster {cid} into 2 sub-clusters")

        # Execute merges (they decrease K)
        merged_already = set()
        for action, cid, target in actions_this_round:
            if action == "merge" and target is not None:
                if cid in merged_already or target in merged_already:
                    continue
                labels = _merge_clusters(labels, target, cid)
                any_change = True
                merged_already.add(cid)
                result.n_merged += 1
                result.actions_log.append(f"MERGE cluster {cid} → {target}")
                if verbose:
                    print(f"    → Merged cluster {cid} into cluster {target}")

        # Count retained
        result.n_retained = len([d for d in diagnostics if d.action == "retain"])
        result.diagnostics = [asdict(d) for d in diagnostics]

        # Relabel to consecutive integers
        unique_new = np.unique(labels)
        if len(unique_new) > 0:
            label_map = {old: new for new, old in enumerate(sorted(unique_new))}
            labels = np.array([label_map[l] for l in labels], dtype=int)

        if not any_change:
            if verbose:
                print(f"  [CONVERGED] No changes in iteration {iteration + 1}")
            break

    # ── Finalize ──
    result.n_clusters_after = len(np.unique(labels))
    result.labels_refined = labels.tolist()
    result.passed = (result.n_split == 0 and result.n_merged == 0)
    result.timing_s = perf_counter() - t0

    if result.passed:
        result.message = (f"All {result.n_clusters_before} clusters retained. "
                          f"LNC*_global={result.lnc_global:.4f}")
    else:
        result.message = (f"Refined: {result.n_clusters_before} → {result.n_clusters_after} clusters "
                          f"({result.n_split} splits, {result.n_merged} merges, "
                          f"{result.n_retained} retained). LNC*_global={result.lnc_global:.4f}")

    if verbose:
        print(f"\n[STRUCTURAL CONTROL DONE] {result.message} ({result.timing_s:.2f}s)")

    return result
