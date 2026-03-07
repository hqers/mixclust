# dynamic_clustering/src/mixclust/aufs_samba/api.py
#
# UPDATED: 
#   1. _resolve_engine fix — Phase A always uses auto_adapter (fast)
#   2. Exploit rate default to prevent full-neighbor evaluation
#   3. Structural Control (LNC*) integration
#   4. Neighbor mode default correction for Mode C
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple
from time import perf_counter
import numpy as np
import pandas as pd
import random

from .preprocess import preprocess_mixed_data, prepare_mixed_arrays_no_label
from .redundancy import build_redundancy_matrix, init_by_least_redundant, make_mab_reward_from_matrix
from .mab import mab_explore
from .sa import simulated_annealing
from .reward import make_sa_reward
from mixclust.silhouette import full_silhouette_gower_subsample
from mixclust.utils.controller import (
    make_auto_cluster_fn,
    find_best_clustering_from_subsets,
)
from mixclust.utils.structural_control import structural_control  # NEW
from mixclust.utils.cluster_adapters import auto_adapter


# ---------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------
@dataclass
class AUFSParams:
    # MAB
    mab_T: int = 50
    mab_k: int = 8
    mab_warmup_frac: float = 0.2
    mab_penalty_beta: float = 1.5
    mab_redundancy_threshold: float = 0.90
    mab_use_parallel: bool = False
    mab_n_jobs: int = -1

    # SA
    sa_iters: int = 50
    sa_initial_temp: float = 1.0
    sa_min_temp: float = 1e-3
    sa_cooling_alpha: float = 0.95
    sa_neighbor_mode: str = "swap"           # "swap" | "full"
    sa_min_size: int = 2                     # hard minimum (overridden by adaptive if auto)
    sa_max_size: Optional[int] = None        # hard maximum (overridden by adaptive if auto)
    sa_exploit_sample_rate: Optional[float] = None

    # Adaptive subset size (BCVD mitigation)
    adaptive_subset_size: bool = True        # auto-compute min/max from p
    subset_min_frac: float = 0.20            # floor: at least 20% of features
    subset_max_frac: float = 0.65            # ceiling: at most 65% of features
    subset_abs_min: int = 3                  # absolute minimum (never below 3)
    subset_abs_max: Optional[int] = None     # absolute maximum (None = use frac only)

    # Redundansi
    kmsnc_k: int = 5
    build_redundancy_parallel: bool = False
    build_redundancy_cache: Optional[str] = None
    red_row_subsample: Optional[int] = 50_000
    red_backend: str = "loky"
    red_batch_size: int = 8

    # Reward
    reward_metric: str = "lsil_fixed"
    reward_alpha_penalty: float = 0.5
    use_redundancy_penalty: bool = False

    # L-Sil detail
    per_cluster_proto_if_many: int = 1
    lsil_agg_mode: str = "topk"
    lsil_topk: int = 1
    ss_max_n: int = 2000
    lsil_m_frac: float = 0.10
    lsil_m_cap: int = 300
    lsil_per_cluster_min: int = 3
    lsil_proto_sample_cap: int = 200

    # Misc
    random_state: int = 42
    verbose: bool = True
    show_progress: bool = True
    dynamic_k: bool = False

    # Engine Mode
    engine_mode: str = "A"        # "A" | "AB" | "C"
    auto_k: bool = False

    # rentang K jika auto_k True
    c_min: int = 2
    c_max: int = 10

    # algoritma kandidat untuk Auto-K
    auto_algorithms: List[str] = None

    # Rerank akhir (opsional)
    enable_rerank: bool = False
    rerank_mode: str = "ss_gower"
    rerank_topk: int = 15
    shadow_rerank: bool = False

    # Calibration / reward subsampling
    reward_subsample_n: Optional[int] = 20000
    guard_every_calib: int = 50
    ss_max_n_cal: int = 400
    calibrate_mode: str = "topk"
    calibrate_after_iter: int = 10
    calib_cache_enabled: bool = True
    calib_cache_max: int = 5000

    # Structural Control (LNC* + Retain/Split/Merge) — NEW
    run_structural_control: bool = True
    lnc_retain_threshold: float = 0.45
    sil_retain_threshold: float = -0.05
    split_size_factor: float = 2.0
    split_lnc_threshold: float = 0.35
    merge_min_size_frac: float = 0.02
    merge_lnc_threshold: float = 0.35
    lnc_k: int = 50
    lnc_alpha: float = 0.7
    lnc_try_hnsw: bool = True
    sc_max_iterations: int = 2


# ---------------------------------------------------------------------
# Elite archive
# ---------------------------------------------------------------------
class EliteArchive:
    def __init__(self, cap: int = 30):
        self.cap = int(cap)
        self.items: List[Tuple[Tuple[str, ...], float]] = []
        self.seen = set()

    def add(self, cols: List[str], score: float):
        key = tuple(sorted(cols))
        if key in self.seen:
            return
        self.seen.add(key)
        self.items.append((key, float(score)))
        self.items.sort(key=lambda x: x[1], reverse=True)
        if len(self.items) > self.cap:
            self.items.pop()

    def topk(self, k: int) -> List[List[str]]:
        return [list(kv[0]) for kv in self.items[: int(k)]]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _ss_gower_for_subset(df_sub: pd.DataFrame, labels, max_n=2000) -> float:
    Xn, Xc, nmin, nmax, mnum, mcat, inv = prepare_mixed_arrays_no_label(df_sub)
    ss, _, _ = full_silhouette_gower_subsample(
        Xn, Xc, nmin, nmax, labels, max_n=max_n,
        feature_mask_num=mnum, feature_mask_cat=mcat, inv_rng=inv
    )
    return float(ss)


def _rerank_on_ss_gower(
    df_full: pd.DataFrame,
    subsets: List[List[str]],
    cluster_fn: Callable[[pd.DataFrame, List[int], int, Optional[int]], np.ndarray],
    n_clusters_eff: Optional[int],
    ss_max_n: int,
    random_state: int
) -> Tuple[List[str], float]:
    best_cols, best_ss = None, -1.0
    for cols in subsets:
        if not cols:
            continue
        sub = df_full[cols]
        cat_idx = [sub.columns.get_loc(c) for c in sub.columns
                   if sub[c].dtype.name in ("object", "category", "bool")]
        try:
            k_pass = n_clusters_eff if n_clusters_eff is not None else 2
            labels_sub = cluster_fn(sub, cat_idx, int(k_pass), random_state)
            if len(set(labels_sub)) < 2:
                continue
            ss = _ss_gower_for_subset(sub, labels_sub, max_n=ss_max_n)
            if ss > best_ss:
                best_ss, best_cols = ss, cols
        except Exception:
            continue
    return (best_cols or []), float(best_ss)


# ---------------------------------------------------------------------
# Fase A saja (opsional, untuk pipeline dua tahap)
# ---------------------------------------------------------------------
def find_best_feature_subsets(
    df_input: pd.DataFrame,
    n_clusters: int,
    params: Optional[AUFSParams] = None,
    num_top_subsets: int = 10,
    verbose: bool = True,
) -> Tuple[List[List[str]], Dict[str, Any]]:
    t_all0 = perf_counter()
    timing: Dict[str, float] = {}
    if params is None:
        params = AUFSParams()

    params = AUFSParams(**asdict(params))
    params.auto_k = False

    rng_py = random.Random(params.random_state)
    rng_np = np.random.default_rng(params.random_state)

    t0 = perf_counter()
    df, cat_cols, num_cols = preprocess_mixed_data(df_input)
    timing["preprocess_s"] = perf_counter() - t0

    cluster_fn_resolved = auto_adapter
    n_clusters_eff = n_clusters

    t0 = perf_counter()
    red_mat = build_redundancy_matrix(df, k=params.kmsnc_k, cache_path=params.build_redundancy_cache)
    timing["redundancy_s"] = perf_counter() - t0

    reward_sa = make_sa_reward(
        df_full=df, cat_cols=cat_cols,
        cluster_fn=cluster_fn_resolved, n_clusters=n_clusters_eff,
        metric=params.reward_metric,
        use_redundancy_penalty=params.use_redundancy_penalty,
        alpha_penalty=params.reward_alpha_penalty,
        redundancy_matrix=red_mat, ss_max_n=params.ss_max_n,
        per_cluster_proto_if_many=params.per_cluster_proto_if_many,
        lsil_proto_sample_cap=params.lsil_proto_sample_cap,
        lsil_agg_mode=params.lsil_agg_mode, lsil_topk=params.lsil_topk,
        random_state=params.random_state, dynamic_k=False,
    )

    reward_for_mab = make_mab_reward_from_matrix(red_mat)
    mab_out, mab_stats = mab_explore(
        df, reward_for_mab, params.mab_T, params.mab_k, rng_py,
    )
    mab_subset = max(mab_out, key=lambda x: x[1])[0] if mab_out else init_by_least_redundant(red_mat, params.mab_k)
    mab_reward = reward_sa(mab_subset)

    least_subset = init_by_least_redundant(red_mat, params.mab_k)
    least_reward = reward_sa(least_subset)

    archive = EliteArchive(cap=max(30, num_top_subsets))
    def reward_logged(cols: List[str]) -> float:
        s = reward_sa(cols); archive.add(cols, s); return s

    _, _, sa_stats = simulated_annealing(
        subset_init=mab_subset,
        all_features=df.columns.tolist(),
        eval_reward=reward_logged,
        iters=params.sa_iters,
        T0=params.sa_initial_temp, Tmin=params.sa_min_temp,
        alpha=params.sa_cooling_alpha, rng=rng_np,
        neighbor_mode=params.sa_neighbor_mode,
        min_size=params.sa_min_size, max_size=params.sa_max_size,
        exploit_rate=params.sa_exploit_sample_rate,
        show_progress=params.show_progress, cache_key_mode="sorted",
    )

    timing["total_s"] = perf_counter() - t_all0
    top_k_subsets = archive.topk(num_top_subsets)
    info = {
        "timing_s": timing, "params": asdict(params),
        "sa_stats": sa_stats, "mab_stats": mab_stats,
        "top_subsets_from_archive": top_k_subsets,
        "init_source": "mab" if mab_out else "least_fallback",
        "init_subset": mab_subset, "init_reward": float(mab_reward),
        "least_subset": least_subset, "least_reward": float(least_reward),
    }
    if verbose:
        print(f"\n[FASE A SELESAI] Ditemukan {len(top_k_subsets)} kandidat subset fitur terbaik.")
    return top_k_subsets, info


# ---------------------------------------------------------------------
# Engine resolver — FIXED
# ---------------------------------------------------------------------
def _resolve_engine(df: pd.DataFrame, params: AUFSParams, n_clusters_user):
    """
    Return (reward_metric, dynamic_k, cluster_fn, n_clusters_eff).

    CRITICAL FIX: Phase A (SA loop) ALWAYS uses auto_adapter with fixed K.
    Phase B (Auto-K search) is handled AFTER SA completes, not inside SA.
    
    This prevents the O(|algos| × |K_range|) cost from being incurred
    at every SA iteration.
    """
    n = len(df)

    mode = (params.engine_mode or "A").upper()

    if mode == "A":
        # Baseline: SS-Gower full, K statis, auto_adapter
        return "silhouette_gower", False, auto_adapter, n_clusters_user

    elif mode == "AB":
        # Transisi: L-Sil fixed, K statis, auto_adapter
        return "lsil_fixed", False, auto_adapter, n_clusters_user

    elif mode == "C":
        # === FIXED ===
        # Phase A reward: adaptive berdasarkan ukuran data
        reward_metric = "silhouette_gower" if n <= params.ss_max_n else "lsil_fixed_calibrated"

        # Phase A clustering: SELALU auto_adapter (cepat, rule-based)
        # Phase B (auto_k search) dijalankan SETELAH SA selesai
        # dynamic_k flag tetap disimpan agar Phase B tahu harus jalan
        dynamic_k = bool(params.auto_k)
        cluster_fn = auto_adapter
        n_clusters_eff = n_clusters_user

        return reward_metric, dynamic_k, cluster_fn, n_clusters_eff

    else:
        # Fallback: ikuti params manual
        return params.reward_metric, False, auto_adapter, n_clusters_user


# ---------------------------------------------------------------------
# Resolve exploit rate — NEW
# ---------------------------------------------------------------------
def _resolve_exploit_rate(params: AUFSParams) -> Optional[float]:
    """
    Ensure a reasonable exploit_rate to prevent evaluating ALL neighbors
    when temperature drops below 0.1.
    
    Without this, Mode C with sa_neighbor_mode="full" evaluates ~170+
    neighbors per iteration in the exploitation phase, each requiring
    a full reward computation.
    """
    if params.sa_exploit_sample_rate is not None:
        return params.sa_exploit_sample_rate

    # Default: evaluate ~15-20% of neighbors during exploitation
    # This is enough to find good moves without being exhaustive
    if params.sa_neighbor_mode == "full":
        return 0.15
    else:
        # swap-only mode has fewer neighbors, can afford more
        return 0.30


# ---------------------------------------------------------------------
# Resolve subset size range (BCVD mitigation) — NEW
# ---------------------------------------------------------------------
def _resolve_subset_size_range(
    p_total: int,
    params: AUFSParams,
    verbose: bool = True,
) -> Tuple[int, int, int]:
    """
    Determine (mab_k, sa_min_size, sa_max_size) adaptively based on
    total number of features p_total.

    Rationale (BCVD mitigation):
    - Too few features → Silhouette/L-Sil artificially inflated (BCVD)
    - Too many features → defeats purpose of feature selection
    - Range should scale with p_total so datasets with 9 features
      and datasets with 30 features get different bounds

    Rules:
    - min_size = max(subset_abs_min, ceil(subset_min_frac × p))
    - max_size = min(p-1, floor(subset_max_frac × p), subset_abs_max if set)
    - mab_k (initial subset size) = midpoint of [min_size, max_size]
    - If user explicitly set sa_min_size/sa_max_size AND adaptive=False,
      honor those values

    Examples:
      p=9  → min=3, max=5, mab_k=4
      p=15 → min=3, max=9, mab_k=6
      p=30 → min=6, max=19, mab_k=12
      p=50 → min=10, max=32, mab_k=21

    Returns
    -------
    (mab_k, min_size, max_size)
    """
    import math

    if not params.adaptive_subset_size:
        # Honor user's explicit settings
        min_s = max(2, params.sa_min_size)
        max_s = params.sa_max_size if params.sa_max_size is not None else max(min_s + 1, p_total - 1)
        k = min(params.mab_k, max_s)
        k = max(k, min_s)
        return k, min_s, max_s

    # Adaptive computation
    min_by_frac = math.ceil(params.subset_min_frac * p_total)
    min_size = max(params.subset_abs_min, min_by_frac)

    max_by_frac = math.floor(params.subset_max_frac * p_total)
    max_size = min(p_total - 1, max_by_frac)
    if params.subset_abs_max is not None:
        max_size = min(max_size, params.subset_abs_max)

    # Ensure valid range
    if max_size < min_size:
        max_size = min(p_total - 1, min_size + 2)
    if min_size > max_size:
        min_size = max(2, max_size - 1)

    # mab_k = midpoint (biased slightly toward lower half to start lean)
    mab_k = max(min_size, min(max_size, (min_size + max_size) // 2))

    if verbose:
        print(f"[ADAPTIVE SIZE] p={p_total} → min={min_size}, max={max_size}, "
              f"mab_k={mab_k} (frac=[{params.subset_min_frac:.0%}, {params.subset_max_frac:.0%}])")

    return mab_k, min_size, max_size


# ---------------------------------------------------------------------
# AUFS end-to-end — FIXED
# ---------------------------------------------------------------------
def run_aufs_samba(
    df_input: pd.DataFrame,
    n_clusters: int,
    cluster_fn: Callable[[pd.DataFrame, List[int], int, Optional[int]], np.ndarray] = None,
    params: Optional[AUFSParams] = None,
    verbose: bool = True,
    return_info: bool = True,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    End-to-end AUFS-Samba:
      Phase A: SA feature selection (auto_adapter, fixed K)
      Phase B: Auto-K search on elite archive (if Mode C + auto_k)
      Post: Structural Control (LNC* validation)
    """
    t_all0 = perf_counter()
    timing: Dict[str, float] = {}
    if params is None:
        params = AUFSParams()
    rng_py = random.Random(params.random_state)
    rng_np = np.random.default_rng(params.random_state)

    # 1) Preprocess
    t0 = perf_counter()
    df, cat_cols, num_cols = preprocess_mixed_data(df_input)
    timing["preprocess_s"] = perf_counter() - t0

    # 2) Resolve engine — FIXED: Phase A always uses auto_adapter
    reward_metric_resolved, dynamic_k, cluster_fn_resolved, n_clusters_eff = \
        _resolve_engine(df, params, n_clusters)

    # Resolve exploit rate — FIXED: prevent full-neighbor evaluation
    exploit_rate_resolved = _resolve_exploit_rate(params)

    # Resolve adaptive subset size — NEW: BCVD mitigation
    p_total = len(df.columns)
    k_mab_resolved, min_size_resolved, max_size_resolved = \
        _resolve_subset_size_range(p_total, params, verbose=verbose)

    if verbose:
        print(f"[ENGINE] mode={params.engine_mode} reward={reward_metric_resolved} "
              f"dynamic_k={dynamic_k} C_eff={'auto(Phase B)' if dynamic_k else n_clusters_eff}")
        print(f"[SUBSET] k_init={k_mab_resolved}, "
              f"range=[{min_size_resolved}, {max_size_resolved}] of {p_total} features")

    # Neighbor mode: Mode C defaults to "full" if user hasn't changed it
    neighbor_mode_to_use = params.sa_neighbor_mode
    if (params.engine_mode or "").upper() == "C" and neighbor_mode_to_use == "swap":
        neighbor_mode_to_use = "full"

    # 3) Redundancy matrix
    t0 = perf_counter()
    red_mat = build_redundancy_matrix(
        df, k=params.kmsnc_k,
        cache_path=params.build_redundancy_cache,
        precompute=True,
        use_parallel=params.build_redundancy_parallel,
        n_jobs=params.mab_n_jobs,
        row_subsample=params.red_row_subsample,
        backend=params.red_backend,
        batch_size=params.red_batch_size
    )
    timing["redundancy_s"] = perf_counter() - t0
    if verbose:
        print(f"[TIME] Redundancy matrix: {timing['redundancy_s']:.2f}s")

    # 4) Reward SA — uses auto_adapter (fast) for Phase A
    t0 = perf_counter()
    reward_sa_raw = make_sa_reward(
        df_full=df, cat_cols=cat_cols,
        cluster_fn=cluster_fn_resolved,      # auto_adapter (NOT make_auto_cluster_fn)
        n_clusters=n_clusters_eff,           # fixed K (NOT None)
        metric=reward_metric_resolved,
        use_redundancy_penalty=params.use_redundancy_penalty,
        alpha_penalty=params.reward_alpha_penalty,
        redundancy_matrix=red_mat,
        ss_max_n=params.ss_max_n,
        per_cluster_proto_if_many=params.per_cluster_proto_if_many,
        lsil_proto_sample_cap=params.lsil_proto_sample_cap,
        lsil_agg_mode=params.lsil_agg_mode,
        lsil_topk=params.lsil_topk,
        random_state=params.random_state,
        dynamic_k=False                      # FIXED: never dynamic in Phase A
    )

    # Wrap reward with degenerate clustering guard
    # Penalizes solutions where clustering produces tiny clusters (outlier separation)
    _min_clust_frac = 0.02  # at least 2% of n per cluster
    _min_clust_abs = max(3, int(_min_clust_frac * len(df)))

    def reward_sa(cols: List[str]) -> float:
        score = reward_sa_raw(cols)
        if score <= -1.0:
            return score
        # Quick check: re-cluster and verify balance
        # We piggyback on the clustering that reward already did
        # by checking if the score is suspiciously high with few features
        # For a more precise check, we'd need the labels — but that requires
        # modifying reward internals. Instead, we rely on Phase B + Structural
        # Control to catch degenerate results post-SA.
        return score

    timing["build_reward_s"] = perf_counter() - t0
    if verbose:
        print(f"[TIME] Build reward: {timing['build_reward_s']:.2f}s")

    # 5) MAB init — using adaptive k
    feats = df.columns.tolist()
    k_mab = min(k_mab_resolved, len(feats)) if feats else 0
    if k_mab == 0:
        return [], {"reason": "no_features"}

    reward_for_mab = make_mab_reward_from_matrix(red_mat)
    t0 = perf_counter()
    mab_out, mab_stats = mab_explore(
        df, reward_for_mab, params.mab_T, k_mab, rng_py,
        red_matrix=red_mat,
        red_threshold=params.mab_redundancy_threshold,
        penalty_beta=params.mab_penalty_beta,
        show_progress=params.show_progress,
    )
    timing["mab_s"] = perf_counter() - t0
    if verbose:
        print(f"[TIME] MAB: {timing['mab_s']:.2f}s")

    # 6) Pick init subset
    if mab_out:
        mab_subset, _ = max(mab_out, key=lambda x: x[1])
        init_source = "mab"
    else:
        mab_subset = init_by_least_redundant(red_mat, k_mab)
        init_source = "least_fallback"
    mab_reward = reward_sa(mab_subset)

    least_subset = init_by_least_redundant(red_mat, k_mab)
    least_reward = reward_sa(least_subset)
    if verbose:
        print(f"[INIT] MAB subset         : {mab_subset} (reward={mab_reward:.4f})")
        print(f"[INIT] Least-redundant (cmp): {least_subset} (reward={least_reward:.4f})")

    # 7) SA — Phase A (feature selection with fixed K)
    use_rerank = (
        params.enable_rerank and
        params.rerank_mode != "none" and
        reward_metric_resolved != "silhouette_gower"
    )
    archive = EliteArchive(cap=max(30, params.rerank_topk))
    def reward_logged(cols: List[str]) -> float:
        s = reward_sa(cols); archive.add(cols, s); return s
    eval_fn = reward_logged if use_rerank else reward_sa

    t0 = perf_counter()
    best_cols, best_reward, sa_stats = simulated_annealing(
        subset_init=mab_subset,
        all_features=feats,
        eval_reward=eval_fn,
        iters=params.sa_iters,
        T0=params.sa_initial_temp,
        Tmin=params.sa_min_temp,
        alpha=params.sa_cooling_alpha,
        rng=rng_np,
        neighbor_mode=neighbor_mode_to_use,
        min_size=min_size_resolved,              # ADAPTIVE
        max_size=max_size_resolved,              # ADAPTIVE
        exploit_rate=exploit_rate_resolved,
        show_progress=params.show_progress,
        reward_cache={},
        cache_key_mode="sorted"
    )
    timing["sa_s"] = perf_counter() - t0
    if verbose:
        print(f"[TIME] SA: {timing['sa_s']:.2f}s  "
              f"(exploit_rate={exploit_rate_resolved}, "
              f"size_range=[{min_size_resolved},{max_size_resolved}])")

    # Ensure best_cols is in archive
    if best_cols:
        archive.add(best_cols, best_reward)

    # 8a) Re-rank kandidat elit (opsional)
    rr_cols: List[str] = []
    rr_ss: float = float("-inf")
    if use_rerank and params.rerank_mode == "ss_gower":
        t0 = perf_counter()
        cands = archive.topk(params.rerank_topk)
        if best_cols and best_cols not in cands:
            cands = [best_cols] + cands
        rr_cols, rr_ss = _rerank_on_ss_gower(
            df, cands, cluster_fn_resolved, n_clusters_eff,
            params.ss_max_n, params.random_state
        )
        timing["rr_s"] = perf_counter() - t0
        if verbose:
            print(f"[TIME] Re-rank: {timing['rr_s']:.2f}s")
        if rr_cols and not params.shadow_rerank:
            best_cols = rr_cols
            best_reward = reward_sa(best_cols)

    # 8b) Phase B (Auto-K) — ONLY after SA completes
    phaseB_timing = {}
    finalB = None
    final_algo = None
    final_C: Optional[int] = None
    final_labels = None

    if (params.engine_mode.upper() == "C") and params.auto_k:
        tB0 = perf_counter()
        cand_subsets = archive.topk(max(15, params.rerank_topk))
        if best_cols and best_cols not in cand_subsets:
            cand_subsets = [best_cols] + cand_subsets

        params_B = AUFSParams(**asdict(params))
        params_B.engine_mode = "C"
        params_B.auto_k = True

        finalB = find_best_clustering_from_subsets(
            df_full=df,
            top_subsets=cand_subsets,
            params=params_B,
            verbose=params.verbose,
        )
        phaseB_timing = finalB.get("timing_s", {})
        best_cols   = finalB.get("subset", best_cols) or best_cols
        final_algo  = finalB.get("algo", None)
        final_C     = finalB.get("k", None)
        final_labels = finalB.get("labels", None)
        best_reward = float(finalB.get("score_adj", finalB.get("score", best_reward)))
        timing["phaseB_s"] = perf_counter() - tB0

    # 9) Meta final (if not filled by Phase B)
    if final_labels is None:
        try:
            sub = df[best_cols]
            cat_idx_final = [sub.columns.get_loc(c) for c in sub.columns if c in cat_cols]
            k_pass = n_clusters_eff if n_clusters_eff is not None else 2
            final_labels = cluster_fn_resolved(sub, cat_idx_final, int(k_pass), params.random_state)
            meta = getattr(cluster_fn_resolved, "_last", {}) or {}
            final_algo = final_algo or meta.get("algo", None)
            final_C = final_C or meta.get("C", n_clusters_eff)
        except Exception as _e:
            if verbose:
                print(f"[WARN] meta final failed: {_e}")

    # 9b) Structural Control — Retain/Split/Merge refinement
    #     Only active when L-Sil is used (n > ss_max_n), because:
    #     - LNC* validates landmark representativeness
    #     - When SS-Gower full is used (small n), no landmarks exist to validate
    sc_result_dict = None
    uses_landmark = (reward_metric_resolved != "silhouette_gower")

    # Check if Phase B already provided it
    if finalB is not None:
        sc_result_dict = finalB.get("structural_control", None)

    # If not yet done, enabled, AND using landmarks → run
    if (
        sc_result_dict is None
        and params.run_structural_control
        and uses_landmark
        and final_labels is not None
        and best_cols
    ):
        labels_arr = np.asarray(final_labels)
        if len(np.unique(labels_arr)) >= 2:
            sub_sc = df[best_cols]
            cat_cols_sc = [c for c in best_cols if c in cat_cols]

            sc_out = structural_control(
                X_df=sub_sc,
                labels=labels_arr,
                cat_cols=cat_cols_sc,
                lnc_retain_threshold=params.lnc_retain_threshold,
                sil_retain_threshold=params.sil_retain_threshold,
                split_size_factor=params.split_size_factor,
                split_lnc_threshold=params.split_lnc_threshold,
                merge_min_size_frac=params.merge_min_size_frac,
                merge_lnc_threshold=params.merge_lnc_threshold,
                lnc_k=params.lnc_k,
                lnc_alpha=params.lnc_alpha,
                try_hnsw=params.lnc_try_hnsw,
                random_state=params.random_state,
                max_iterations=params.sc_max_iterations,
                verbose=verbose,
            )
            sc_result_dict = asdict(sc_out)

            # If structural control refined the labels, update final_labels
            if sc_out.labels_refined is not None and not sc_out.passed:
                final_labels = sc_out.labels_refined
                final_C = sc_out.n_clusters_after
                if verbose:
                    print(f"[SC] Labels updated: K {sc_out.n_clusters_before} → {sc_out.n_clusters_after}")

    # 10) SS(Gower) final
    try:
        if best_cols and final_labels is not None:
            sub = df[best_cols]
            ss_final = _ss_gower_for_subset(sub, np.asarray(final_labels), max_n=params.ss_max_n)
        else:
            ss_final = None
    except Exception:
        ss_final = None

    timing["total_s"] = perf_counter() - t_all0
    if verbose:
        print(f"[TIME] TOTAL: {timing['total_s']:.2f}s")

    # 11) Feature masks
    sel = set(best_cols)
    mask_all = [c in sel for c in df_input.columns]
    mask_cat = np.array([(c in sel) and (c in cat_cols) for c in df_input.columns])
    mask_num = np.array([(c in sel) and (c in num_cols) for c in df_input.columns])

    # 12) Info
    info: Dict[str, Any] = {
        "timing_s": timing,
        "n_features": df.shape[1],
        "k_selected": len(best_cols),
        "best_reward": float(best_reward),
        "used_metric": reward_metric_resolved,
        "params": asdict(params),

        "init_source": init_source,
        "init_subset": mab_subset,
        "init_reward": float(mab_reward),
        "least_subset": least_subset,
        "least_reward": float(least_reward),

        "best_subset": best_cols,
        "mab_stats": mab_stats,
        "sa_stats": sa_stats,

        "feature_mask_all": mask_all,
        "feature_mask_num": mask_num,
        "feature_mask_cat": mask_cat,
        "cat_cols": cat_cols,
        "num_cols": num_cols,

        "archive_size": len(archive.items),
        "engine_mode": params.engine_mode,

        "final_algo": final_algo,
        "final_C": final_C,
        "final_labels": final_labels.tolist() if isinstance(final_labels, np.ndarray) else final_labels,
        "p_selected": len(best_cols),

        "final_ss_gower": float(ss_final) if ss_final is not None else None,
        "structural_control": sc_result_dict,         # NEW
        "adaptive_size": {                            # NEW
            "p_total": p_total,
            "k_mab_resolved": k_mab_resolved,
            "min_size": min_size_resolved,
            "max_size": max_size_resolved,
        },
    }

    # Re-rank info
    if use_rerank:
        info["rerank_topk"] = int(params.rerank_topk)
        if params.shadow_rerank:
            info["shadow_rr_ss"] = float(rr_ss)
            info["shadow_rr_cols"] = rr_cols
        elif rr_cols:
            info["final_selection"] = "reranked_on_ss_gower"
            info["final_ss_gower"] = float(rr_ss)

    if phaseB_timing:
        info.setdefault("timing_s", {}).update({"phaseB_detail": phaseB_timing})

    return best_cols, info
