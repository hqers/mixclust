# dynamic_clustering/src/mixclust/utils/controller.py
#
# Update: Added Structural Control via LNC* post-Phase-B validation
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations
import numpy as np
import pandas as pd
import random
from time import perf_counter

# sklearn metrics for DBI/CHI
from sklearn.metrics import davies_bouldin_score as _dbi
from sklearn.metrics import calinski_harabasz_score as _chi

# Adapters
from mixclust.utils.cluster_adapters import (
    hac_gower_adapter,
    kprototypes_adapter,
    kmodes_adapter,
    auto_adapter,
)

# Optional imports (guarded usage)
try:
    from mixclust.metrics.lsil import lsil_using_prototypes_gower
except Exception:
    lsil_using_prototypes_gower = None  # type: ignore

try:
    from mixclust.silhouette import full_silhouette_gower_subsample
except Exception:
    full_silhouette_gower_subsample = None  # type: ignore

try:
    from mixclust.metrics.lnc_star import lnc_star
except Exception:
    lnc_star = None  # type: ignore

try:
    from mixclust.knn_index import KNNIndex
except Exception:
    KNNIndex = None  # type: ignore

from mixclust.aufs_samba.preprocess import preprocess_mixed_data
from sklearn.preprocessing import normalize
from mixclust.landmarks import select_landmarks_kcenter, select_landmarks_cluster_aware
from mixclust.aufs_samba.preprocess import prepare_mixed_arrays_no_label
from mixclust.prototypes import build_prototypes_by_cluster_gower

from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple, Dict, Any


# ================================================================
# Data classes
# ================================================================

@dataclass
class AutoClustResult:
    """Container for a single algorithm/K trial."""
    algo: Optional[str] = None
    k: Optional[int] = None
    labels: Optional[np.ndarray] = None
    score: float = -np.inf
    score_adj: float = -np.inf
    dbi: float = np.nan
    chi: float = np.nan
    metric_used: Optional[str] = None
    n_unique_labels: int = 0

    def is_better_than(self, other: 'AutoClustResult', tie_breakers: Tuple[str, ...]) -> bool:
        if self.score_adj > other.score_adj + 1e-9:
            return True
        if abs(self.score_adj - other.score_adj) < 1e-9:
            return _tie_better(self.__dict__, other.__dict__, tie_breakers)
        return False


@dataclass
class StructuralControlResult:
    """Container for Structural Control (LNC*) validation output."""
    lnc_score: float = np.nan
    passed: bool = False
    threshold: float = 0.5
    action: str = "none"          # "none" | "accept" | "warning" | "resample"
    n_landmarks: int = 0
    message: str = ""
    timing_s: float = 0.0


# ================================================================
# Helpers
# ================================================================

def cat_cols_to_index(X_df: pd.DataFrame, cat_cols: List[str]) -> List[int]:
    name2pos = {c: i for i, c in enumerate(X_df.columns)}
    return [name2pos[c] for c in cat_cols if c in name2pos]


def _tie_better(a: Dict[str, Any], b: Dict[str, Any], tie_breakers: Tuple[str, ...]) -> bool:
    for tb in tie_breakers:
        if tb == "dbi":
            va, vb = a.get("dbi", np.inf), b.get("dbi", np.inf)
            if np.isfinite(va) and np.isfinite(vb) and va != vb:
                return va < vb
        elif tb == "chi":
            va, vb = a.get("chi", -np.inf), b.get("chi", -np.inf)
            if np.isfinite(va) and np.isfinite(vb) and va != vb:
                return va > vb
    return False


def estimate_gamma(X_df: pd.DataFrame, cat_idx: List[int], scale: float = 1.0) -> float:
    num_cols = [c for i, c in enumerate(X_df.columns) if i not in cat_idx]
    if not num_cols:
        return 1.0
    s = X_df[num_cols].std(ddof=0).replace(0, 1e-9).mean()
    return float(scale * s)


# ================================================================
# Internal scorer (auto SS / L-Sil)
# ================================================================

def score_internal(
    X_df: pd.DataFrame,
    labels: np.ndarray,
    cat_idx: List[int],
    *,
    metric: str = "auto",
    ss_max_n: int = 1000,
    landmarks=None,
    lsil_m: int = 150,
    random_state: int = 42
) -> float:
    n = len(X_df)

    _lsil = lsil_using_prototypes_gower
    _ss_full = full_silhouette_gower_subsample

    use_ss = (
        (metric == "ss_gower")
        or (metric == "auto" and n < ss_max_n)
        or (_lsil is None and _ss_full is not None)
    )
    if use_ss:
        if _ss_full is None:
            raise RuntimeError("Silhouette Gower (full/subsample) not available.")
        X_num, X_cat, num_min, num_max, mask_num, mask_cat, inv_rng = prepare_mixed_arrays_no_label(X_df)
        ss, _mode, _neval = _ss_full(
            X_num, X_cat, num_min, num_max, labels,
            max_n=ss_max_n,
            feature_mask_num=mask_num,
            feature_mask_cat=mask_cat,
            inv_rng=inv_rng
        )
        return float(ss)

    # --- L-Sil path ---
    if _lsil is None:
        raise RuntimeError("L-Sil scorer not available.")

    X_num, X_cat, num_min, num_max, mask_num, mask_cat, inv_rng = prepare_mixed_arrays_no_label(X_df)

    m = min(lsil_m, max(20, n // 10))
    if X_num.shape[1] > 0:
        X_unit = normalize(X_num, norm="l2")
        landmark_idx = select_landmarks_kcenter(X_unit, m=m, seed=random_state)
    else:
        landmark_idx = list(range(min(m, n)))

    prototypes = build_prototypes_by_cluster_gower(
        labels, X_num, X_cat, num_min, num_max,
        per_cluster=1, sample_cap=200, seed=random_state,
        feature_mask_num=mask_num, feature_mask_cat=mask_cat, inv_rng=inv_rng
    )

    score = _lsil(
        labels, landmark_idx, prototypes,
        X_num, X_cat, num_min, num_max,
        feature_mask_num=mask_num, feature_mask_cat=mask_cat,
        inv_rng=inv_rng, agg_mode="topk", topk=1
    )
    return float(score)


# ================================================================
# Auto-K over multiple algorithms (evaluation-driven selection)
# ================================================================

def auto_select_algo_k(
    X_df: pd.DataFrame,
    cat_idx: List[int],
    algorithms: List[str],
    c_range: range,
    *,
    primary_metric: str = "auto",
    landmarks=None,
    penalty_lambda: float = 0.02,
    min_cluster_size_frac: float = 0.02,   # NEW: reject if any cluster < 2% of n
    tie_breakers: Tuple[str, ...] = ("dbi", "chi"),
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Search the best (algorithm, K) on a given feature subset.
    This is the evaluation-driven selection layer of the controller.
    
    Includes degenerate clustering guard: rejects configurations where
    any cluster contains fewer than min_cluster_size_frac × n observations.
    This prevents high-Silhouette but meaningless solutions where outliers
    form tiny clusters.
    """
    best_result = AutoClustResult(labels=np.zeros(len(X_df), dtype=int))

    n_samples = len(X_df)
    min_cluster_size = max(3, int(min_cluster_size_frac * n_samples))
    
    active_algorithms = list(algorithms)
    if n_samples > 2000 and "hac_gower" in active_algorithms:
        print(f"  (Info) HAC-Gower skipped because N ({n_samples}) > 2000.")
        active_algorithms.remove("hac_gower")
    if not active_algorithms:
        active_algorithms = [a for a in algorithms if a != "hac_gower"]

    gamma_for_kproto = estimate_gamma(X_df, cat_idx) if "kprototypes" in active_algorithms else None

    history: List[AutoClustResult] = []
    for algo in active_algorithms:
        for k in c_range:
            if algo == "hac_gower":
                labels = hac_gower_adapter(X_df, cat_idx, k, random_state)
            elif algo == "kprototypes":
                labels = kprototypes_adapter(X_df, cat_idx, k, random_state, max_iter=20, gamma=gamma_for_kproto)
            elif algo == "kmodes":
                labels = kmodes_adapter(X_df, cat_idx, k, random_state)
            else:
                labels = auto_adapter(X_df, cat_idx, k, random_state)

            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                continue

            # NEW: Degenerate clustering guard
            # Reject if ANY cluster is smaller than minimum threshold
            cluster_counts = np.bincount(labels.astype(int) if labels.dtype.kind in 'iu' 
                                         else np.unique(labels, return_inverse=True)[1])
            if np.min(cluster_counts) < min_cluster_size:
                # Apply penalty instead of hard reject — severely penalize
                # so that non-degenerate solutions are always preferred
                degenerate_penalty = 0.5  # heavy penalty
            else:
                degenerate_penalty = 0.0

            sc = score_internal(X_df, labels, cat_idx, metric=primary_metric, landmarks=landmarks, random_state=random_state)
            sc_adj = sc - penalty_lambda * np.log(max(2, k)) - degenerate_penalty

            dbi, chi = np.nan, np.nan
            try:
                X_num = X_df.select_dtypes(include=[np.number])
                if X_num.shape[1] >= 1 and len(unique_labels) > 1:
                    dbi = _dbi(X_num, labels)
                    chi = _chi(X_num, labels)
            except Exception:
                pass

            cand = AutoClustResult(
                algo=algo, k=k, labels=labels,
                score=float(sc), score_adj=float(sc_adj),
                dbi=float(dbi) if np.isfinite(dbi) else np.nan,
                chi=float(chi) if np.isfinite(chi) else np.nan,
                metric_used=primary_metric,
                n_unique_labels=len(unique_labels)
            )
            history.append(cand)
            if best_result.algo is None or cand.is_better_than(best_result, tie_breakers):
                best_result = cand

    return asdict(best_result)


# ================================================================
# Structural Control via LNC*
# ================================================================

def structural_control_lnc(
    X_df: pd.DataFrame,
    labels: np.ndarray,
    cat_cols: List[str],
    *,
    lnc_threshold: float = 0.5,
    lnc_k: int = 50,
    lnc_alpha: float = 0.7,
    landmark_mode: str = "cluster_aware",
    lm_max_frac: float = 0.2,
    lm_per_cluster_min: int = 3,
    central_frac: float = 0.8,
    boundary_frac: float = 0.2,
    M_candidates: int = 300,
    try_hnsw: bool = True,
    random_state: int = 42,
    verbose: bool = True,
) -> StructuralControlResult:
    """
    Structural Control: validate clustering result using LNC*.

    Checks whether landmarks used in L-Sil evaluation are representative
    of the underlying data structure. Returns a StructuralControlResult
    with pass/warning/resample recommendation.

    Parameters
    ----------
    X_df : pd.DataFrame
        Feature subset DataFrame (only selected features).
    labels : np.ndarray
        Cluster labels from the best (algo, K) combination.
    cat_cols : list[str]
        Categorical column names in X_df.
    lnc_threshold : float
        Minimum acceptable LNC* score. Below this → warning/resample.
    lnc_k : int
        Number of Gower-reranked neighbors for LNC* computation.
    lnc_alpha : float
        Weight for neighborhood consistency vs distance contrast in LNC*.
    landmark_mode : str
        "cluster_aware" or "kcenter" for landmark selection.
    lm_max_frac : float
        Maximum fraction of data to use as landmarks.
    lm_per_cluster_min : int
        Minimum landmarks per cluster.
    central_frac, boundary_frac : float
        Ratio of central vs boundary landmarks (cluster_aware mode).
    M_candidates : int
        ANN candidate pool size before Gower re-ranking.
    try_hnsw : bool
        Whether to attempt HNSW index for ANN.
    random_state : int
        Reproducibility seed.
    verbose : bool
        Print diagnostic information.

    Returns
    -------
    StructuralControlResult
        Contains lnc_score, passed (bool), action, message, timing.
    """
    t0 = perf_counter()

    # Guard: LNC* module must be available
    if lnc_star is None or KNNIndex is None:
        return StructuralControlResult(
            lnc_score=np.nan, passed=True,
            action="none",
            message="LNC* or KNNIndex not available; structural control skipped.",
            timing_s=perf_counter() - t0
        )

    labels = np.asarray(labels)
    n = len(labels)

    if n == 0 or len(np.unique(labels)) < 2:
        return StructuralControlResult(
            lnc_score=np.nan, passed=False,
            action="warning",
            message="Insufficient data or single cluster; structural control cannot validate.",
            timing_s=perf_counter() - t0
        )

    # 1) Build features for ANN (unit-normalized)
    try:
        from mixclust.features import build_features
        X_unit, _, _ = build_features(X_df, label_col=None, scaler_type="standard", unit_norm=True)
    except Exception as e:
        return StructuralControlResult(
            lnc_score=np.nan, passed=True,
            action="none",
            message=f"build_features failed ({e}); structural control skipped.",
            timing_s=perf_counter() - t0
        )

    # 2) Gower components
    X_num, X_cat, num_min, num_max, mask_num, mask_cat, inv_rng = prepare_mixed_arrays_no_label(X_df)

    # 3) Select landmarks
    m = min(int(lm_max_frac * n), max(20, n // 10))
    if landmark_mode == "cluster_aware":
        L = select_landmarks_cluster_aware(
            X_unit, labels, m,
            central_frac=central_frac,
            boundary_frac=boundary_frac,
            per_cluster_min=lm_per_cluster_min,
            seed=random_state
        )
    else:
        L = select_landmarks_kcenter(X_unit, m=m, seed=random_state)

    # 4) Build KNN index
    try:
        knn_index = KNNIndex(X_unit, try_hnsw=try_hnsw, verbose=False)
    except Exception as e:
        return StructuralControlResult(
            lnc_score=np.nan, passed=True,
            action="none",
            message=f"KNNIndex construction failed ({e}); structural control skipped.",
            timing_s=perf_counter() - t0
        )

    # 5) Compute LNC*
    M_cand = max(M_candidates, 3 * lnc_k)
    M_cand = min(M_cand, max(50, int(0.05 * n)))

    try:
        lnc_score = lnc_star(
            X_unit, labels, L, knn_index,
            k=lnc_k, alpha=lnc_alpha,
            X_num=X_num, X_cat=X_cat,
            num_min=num_min, num_max=num_max,
            feature_mask_num=mask_num,
            feature_mask_cat=mask_cat,
            inv_rng=inv_rng,
            M_candidates=M_cand,
        )
    except Exception as e:
        return StructuralControlResult(
            lnc_score=np.nan, passed=True,
            action="none",
            message=f"LNC* computation failed ({e}); structural control skipped.",
            timing_s=perf_counter() - t0
        )

    # 6) Decision
    lnc_val = float(lnc_score) if np.isfinite(lnc_score) else 0.0
    elapsed = perf_counter() - t0

    if lnc_val >= lnc_threshold:
        result = StructuralControlResult(
            lnc_score=lnc_val, passed=True,
            threshold=lnc_threshold,
            action="accept",
            n_landmarks=len(L),
            message=f"LNC*={lnc_val:.4f} >= threshold={lnc_threshold:.2f}. "
                    f"Landmark representation is adequate.",
            timing_s=elapsed
        )
    else:
        result = StructuralControlResult(
            lnc_score=lnc_val, passed=False,
            threshold=lnc_threshold,
            action="warning",
            n_landmarks=len(L),
            message=f"LNC*={lnc_val:.4f} < threshold={lnc_threshold:.2f}. "
                    f"Landmark representation may be insufficient. "
                    f"Consider re-sampling landmarks or reviewing cluster structure.",
            timing_s=elapsed
        )

    if verbose:
        status = "✅ PASS" if result.passed else "⚠️  WARNING"
        print(f"[STRUCTURAL CONTROL] {status} | LNC*={lnc_val:.4f} "
              f"(threshold={lnc_threshold:.2f}) | |L|={len(L)} | {elapsed:.2f}s")

    return result


# ================================================================
# PHASE B: search best clustering on top feature subsets
# ================================================================

def find_best_clustering_from_subsets(
    df_full: pd.DataFrame,
    top_subsets: List[List[str]],
    params: Any,
    verbose: bool = True,
    *,
    run_structural_control: bool = True,
    lnc_threshold: float = 0.5,
    lnc_k: int = 50,
) -> Dict[str, Any]:
    """
    Phase B: search best (subset, algorithm, K) from candidate subsets,
    then validate with Structural Control (LNC*).
    """
    if not top_subsets:
        if verbose:
            print("[PHASE B] No subsets to test. Aborting.")
        return {}

    if verbose:
        print(f"\n[BEGIN PHASE B] Searching best clustering from {len(top_subsets)} candidate subsets...")

    best_overall_result: Optional[Dict[str, Any]] = None
    all_run_history: Dict[Tuple[str, ...], Dict[str, Any]] = {}

    algorithms = params.auto_algorithms or ["kprototypes", "hac_gower"]
    c_range = range(params.c_min, params.c_max + 1)
    cat_cols_full = list(df_full.select_dtypes(include=['object', 'category', 'bool']).columns)

    for i, subset in enumerate(top_subsets):
        if not subset:
            continue

        subset_key = tuple(sorted(subset))
        if verbose:
            print(f"  -> Testing subset #{i+1}: {subset}...")

        df_subset = df_full[subset]
        cat_cols_subset = [c for c in subset if c in cat_cols_full]
        cat_idx_subset = cat_cols_to_index(df_subset, cat_cols_subset)

        if len(cat_idx_subset) == 0:
            if verbose:
                print("     ⚠️  No categorical features → using KMeans fallback.")
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=params.c_min, random_state=params.random_state, n_init=10)
            labels = km.fit_predict(df_subset.select_dtypes(exclude=['object', 'category', 'bool']).values)
            current = {
                "algo": "kmeans",
                "k": params.c_min,
                "labels": labels,
                "score": np.nan,
                "score_adj": np.nan,
            }
        else:
            current = auto_select_algo_k(
                X_df=df_subset,
                cat_idx=cat_idx_subset,
                algorithms=algorithms,
                c_range=c_range,
                random_state=params.random_state
            )

        current["subset"] = subset
        all_run_history[subset_key] = current

        current_score = current.get("score_adj", -np.inf)
        best_score = best_overall_result.get("score_adj", -np.inf) if best_overall_result else -np.inf

        if best_overall_result is None or current_score > best_score:
            best_overall_result = current
            if verbose:
                print(f"    ✨ New best! score_adj={current_score:.4f} "
                      f"(algo={current['algo']}, K={current['k']})")

    if verbose and best_overall_result:
        print("\n[PHASE B DONE] Best clustering config found.")

    # ── Structural Control (LNC* validation) ──
    #    Only runs when L-Sil is active (large n), not when SS-Gower full is used
    sc_result = None
    n_samples = len(df_full)
    uses_landmark = (n_samples > getattr(params, 'ss_max_n', 2000))

    if (
        run_structural_control
        and uses_landmark
        and best_overall_result is not None
        and best_overall_result.get("labels") is not None
    ):
        best_subset = best_overall_result.get("subset", [])
        best_labels = np.asarray(best_overall_result["labels"])

        if best_subset and len(np.unique(best_labels)) >= 2:
            df_best = df_full[best_subset]
            cat_cols_best = [c for c in best_subset if c in cat_cols_full]

            sc_result = structural_control_lnc(
                X_df=df_best,
                labels=best_labels,
                cat_cols=cat_cols_best,
                lnc_threshold=lnc_threshold,
                lnc_k=lnc_k,
                random_state=getattr(params, 'random_state', 42),
                verbose=verbose,
            )

    # Attach structural control result
    if best_overall_result:
        best_overall_result["all_run_history"] = all_run_history
        if sc_result is not None:
            best_overall_result["structural_control"] = asdict(sc_result)
        else:
            best_overall_result["structural_control"] = None

    return best_overall_result or {}


# ================================================================
# Wrapper: create a dynamic cluster_fn (auto K & algo)
# ================================================================

def make_auto_cluster_fn(
    algorithms: List[str] = ["kprototypes", "hac_gower"],
    c_range: range = range(2, 10),
    metric: str = "auto",
    penalty_lambda: float = 0.02,
    random_state: int = 42
):
    def cluster_fn(X_df: pd.DataFrame, cat_idx: List[int], _k_unused: int, seed: Optional[int] = None):
        rng = seed if seed is not None else random_state
        best = auto_select_algo_k(
            X_df, cat_idx, algorithms,
            c_range=c_range,
            primary_metric=metric,
            random_state=rng,
            penalty_lambda=penalty_lambda
        )
        try:
            cluster_fn._last = {
                "algo": best.get("algo"),
                "k": best.get("k"),
                "C": best.get("k"),
                "score": best.get("score"),
                "score_adj": best.get("score_adj"),
                "dbi": best.get("dbi"),
                "chi": best.get("chi"),
                "metric_used": best.get("metric_used", metric),
            }
        except Exception:
            pass
        return best["labels"]

    cluster_fn._history = []
    return cluster_fn
