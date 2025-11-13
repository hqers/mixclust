# dynamic_clustering/src/mixclust/utils/controller.py

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

from mixclust.aufs_samba.preprocess import preprocess_mixed_data
from sklearn.preprocessing import normalize
from mixclust.landmarks import select_landmarks_kcenter
from mixclust.aufs_samba.preprocess import prepare_mixed_arrays_no_label
from mixclust.prototypes import build_prototypes_by_cluster_gower

from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple, Dict, Any


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


def cat_cols_to_index(X_df: pd.DataFrame, cat_cols: List[str]) -> List[int]:
    name2pos = {c: i for i, c in enumerate(X_df.columns)}
    return [name2pos[c] for c in cat_cols if c in name2pos]


# --- Internal scorer (auto SS/L-Sil) ---
def score_internal(
    X_df: pd.DataFrame,
    labels: np.ndarray,
    cat_idx: List[int],
    *,
    metric: str = "auto",        # "auto" | "ss_gower" | "l_sil"
    ss_max_n: int = 1000,
    landmarks=None,              # kept for API compatibility
    lsil_m: int = 150,
    random_state: int = 42
) -> float:
    n = len(X_df)

    _lsil = lsil_using_prototypes_gower
    _ss_full = full_silhouette_gower_subsample

    use_ss = (metric == "ss_gower") or (metric == "auto" and n < ss_max_n) or (_lsil is None and _ss_full is not None)
    if use_ss:
        if _ss_full is None:
            raise RuntimeError("Silhouette Gower (full/subsample) not available.")
        # Prepare mixed arrays from the DataFrame (no label column)
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

    # 1) Mixed arrays for Gower components
    X_num, X_cat, num_min, num_max, mask_num, mask_cat, inv_rng = prepare_mixed_arrays_no_label(X_df)

    # 2) Landmark indices (k-center on numeric unit-norm); fallback if no numeric columns
    m = min(lsil_m, max(20, n // 10))
    if X_num.shape[1] > 0:
        X_unit = normalize(X_num, norm="l2")
        landmark_idx = select_landmarks_kcenter(X_unit, m=m, seed=random_state)
    else:
        landmark_idx = list(range(min(m, n)))  # simple fallback when no numeric feature

    # 3) Cluster prototypes (Gower medoids)
    prototypes = build_prototypes_by_cluster_gower(
        labels, X_num, X_cat, num_min, num_max,
        per_cluster=1,
        sample_cap=200,
        seed=random_state,
        feature_mask_num=mask_num,
        feature_mask_cat=mask_cat,
        inv_rng=inv_rng
    )

    # 4) Compute L-Sil (landmark-based)
    score = _lsil(
        labels, landmark_idx, prototypes,
        X_num, X_cat, num_min, num_max,
        feature_mask_num=mask_num,
        feature_mask_cat=mask_cat,
        inv_rng=inv_rng,
        agg_mode="topk", topk=1
    )
    return float(score)


# --- Auto-K over multiple algorithms ---
def auto_select_algo_k(
    X_df: pd.DataFrame,
    cat_idx: List[int],
    algorithms: List[str],
    c_range: range,
    *,
    primary_metric: str = "auto",
    landmarks=None,  # forwarded to score_internal (kept for compat)
    penalty_lambda: float = 0.02,
    tie_breakers: Tuple[str, ...] = ("dbi", "chi"),
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Search the best (algorithm, K) on a given feature subset.
    """
    best_result = AutoClustResult(labels=np.zeros(len(X_df), dtype=int))

    # Filter algorithms (e.g., HAC-Gower for small N)
    n_samples = len(X_df)
    active_algorithms = list(algorithms)
    if n_samples > 2000 and "hac_gower" in active_algorithms:
        print(f"  (Info) HAC-Gower skipped because N ({n_samples}) > 2000.")
        active_algorithms.remove("hac_gower")
    if not active_algorithms:
        active_algorithms = [a for a in algorithms if a != "hac_gower"]

    # Precompute gamma once for k-prototypes
    gamma_for_kproto = estimate_gamma(X_df, cat_idx) if "kprototypes" in active_algorithms else None

    history: List[AutoClustResult] = []
    for algo in active_algorithms:
        for k in c_range:
            # 1) Predict labels
            if algo == "hac_gower":
                labels = hac_gower_adapter(X_df, cat_idx, k, random_state)
            elif algo == "kprototypes":
                labels = kprototypes_adapter(X_df, cat_idx, k, random_state, max_iter=20, gamma=gamma_for_kproto)
            elif algo == "kmodes":
                labels = kmodes_adapter(X_df, cat_idx, k, random_state)
            else:  # "auto" fallback
                labels = auto_adapter(X_df, cat_idx, k, random_state)

            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                continue

            # 2) Primary score (+ complexity penalty)
            sc = score_internal(X_df, labels, cat_idx, metric=primary_metric, landmarks=landmarks, random_state=random_state)
            sc_adj = sc - penalty_lambda * np.log(max(2, k))

            # 3) Tie-breakers (DBI lower is better, CHI higher is better)
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


def _tie_better(a: Dict[str, Any], b: Dict[str, Any], tie_breakers: Tuple[str, ...]) -> bool:
    # prefer DBI smaller, CHI larger
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


# --- PHASE B: search best clustering on top feature subsets ---

def find_best_clustering_from_subsets(
    df_full: pd.DataFrame,
    top_subsets: List[List[str]],
    params: Any,
    verbose: bool = True
) -> Dict[str, Any]:
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
    cat_cols_full = list(df_full.select_dtypes(include=['object','category','bool']).columns)

    for i, subset in enumerate(top_subsets):
        if not subset:
            continue

        subset_key = tuple(sorted(subset))
        if verbose:
            print(f"  -> Testing subset #{i+1}: {subset}...")

        df_subset = df_full[subset]
        cat_cols_subset = [c for c in subset if c in cat_cols_full]
        cat_idx_subset = cat_cols_to_index(df_subset, cat_cols_subset)

        # 🛡️ NEW: jika subset tidak punya kategorik, jangan coba k-prototypes
        if len(cat_idx_subset) == 0:
            if verbose:
                print("     ⚠️  No categorical features → using KMeans fallback.")
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=params.c_min, random_state=params.random_state, n_init=10)
            labels = km.fit_predict(df_subset.select_dtypes(exclude=['object','category','bool']).values)
            current = {
                "algo": "kmeans",
                "k": params.c_min,
                "labels": labels,
                "score": np.nan,
                "score_adj": np.nan,
            }
        else:
            # Jalur normal (campuran / ada kategorik)
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
                print(f"    ✨ New best! score_adj={current_score:.4f} (algo={current['algo']}, K={current['k']})")

    if verbose and best_overall_result:
        print("\n[PHASE B DONE] Best clustering config found.")

    if best_overall_result:
        best_overall_result["all_run_history"] = all_run_history

    return best_overall_result or {}



# --- Wrapper: create a dynamic cluster_fn (auto K & algo) ---
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
        # Stash meta for downstream access
        try:
            cluster_fn._last = {
                "algo": best.get("algo"),
                "k": best.get("k"),
                "C": best.get("k"),   # alias for consistency
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
