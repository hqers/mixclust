# dynamic_clustering/src/mixclust/aufs_samba/api.py
#
# ARSITEKTUR FIX: Phase B menggunakan kembali cache Phase A
#
# Aliran baru (dibanding versi sebelumnya):
#
#   Phase A:
#     make_sa_reward(metric="lsil_fixed_calibrated")
#     → reward.__phase_a_cache__ berisi:
#         X_num_full, X_cat_full, num_min_full, num_max_full
#         inv_rng_full, num_pos, cat_pos
#         L_fixed, labels0, protos0, n_samples
#
#   Setelah SA selesai (sebelum Phase B):
#     cache = _extract_phase_a_cache(reward_sa)
#     → PhaseACache.available = True (jika metric L-Sil)
#
#   Phase B:
#     find_best_clustering_from_subsets(..., phase_a_cache=cache)
#     → setiap trial: gunakan L_fixed + mask, tidak re-Gower
#     → O(n·|L|) per trial, bukan O(n²)
#
# Estimasi speedup:
#   ~0.5s/trial × 150 trial = ~75s (vs 4513s sebelumnya)
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
from mixclust.utils.cluster_adapters import auto_adapter

# ── BARU: PhaseACache infrastructure ──
from mixclust.utils.phase_a_cache import PhaseACache, _extract_phase_a_cache


# ─────────────────────────────────────────────────────────────────
# Helper internal: SS-Gower untuk subset
# ─────────────────────────────────────────────────────────────────
def _ss_gower_for_subset(df_sub, labels, max_n=1000):
    X_num, X_cat, num_min, num_max, mask_num, mask_cat, inv_rng = \
        prepare_mixed_arrays_no_label(df_sub)
    ss, _, _ = full_silhouette_gower_subsample(
        X_num, X_cat, num_min, num_max, labels,
        max_n=max_n,
        feature_mask_num=mask_num, feature_mask_cat=mask_cat,
        inv_rng=inv_rng
    )
    return float(ss)


def _rerank_on_ss_gower(df, cands, cluster_fn, n_clusters, ss_max_n, random_state):
    best_ss, best_cols = float("-inf"), []
    cat_cols = list(df.select_dtypes(include=['object', 'category', 'bool']).columns)
    for cols in cands:
        if not cols:
            continue
        sub = df[cols]
        cat_idx = [sub.columns.get_loc(c) for c in cols if c in cat_cols]
        try:
            k_pass = n_clusters if n_clusters is not None else 2
            labels_sub = cluster_fn(sub, cat_idx, int(k_pass), random_state)
            if len(set(labels_sub)) < 2:
                continue
            ss = _ss_gower_for_subset(sub, labels_sub, max_n=ss_max_n)
            if ss > best_ss:
                best_ss, best_cols = ss, cols
        except Exception:
            continue
    return (best_cols or []), float(best_ss)


# ─────────────────────────────────────────────────────────────────
# EliteArchive
# ─────────────────────────────────────────────────────────────────
class EliteArchive:
    def __init__(self, cap: int = 30):
        self.cap = cap
        self.items: Dict[tuple, float] = {}

    def add(self, cols: List[str], score: float):
        key = tuple(sorted(cols))
        existing = self.items.get(key, float("-inf"))
        if score > existing:
            self.items[key] = score
        if len(self.items) > self.cap:
            worst = min(self.items, key=self.items.get)
            del self.items[worst]

    def topk(self, k: int) -> List[List[str]]:
        sorted_items = sorted(self.items.items(), key=lambda x: x[1], reverse=True)
        return [list(cols) for cols, _ in sorted_items[:k]]


# ─────────────────────────────────────────────────────────────────
# AUFSParams
# ─────────────────────────────────────────────────────────────────
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
    sa_neighbor_mode: str = "swap"
    sa_min_size: int = 2
    sa_max_size: Optional[int] = None
    sa_exploit_sample_rate: Optional[float] = None

    # Adaptive subset size (BCVD mitigation)
    adaptive_subset_size: bool = True
    subset_min_frac: float = 0.20
    subset_max_frac: float = 0.75

    # Reward
    reward_metric: str = "lsil_fixed_calibrated"
    use_redundancy_penalty: bool = True
    reward_alpha_penalty: float = 0.3
    ss_max_n: int = 2000
    force_lsil_proxy: bool = False  # True = paksa lsil_fixed_calibrated meski n < ss_max_n
    per_cluster_proto_if_many: int = 1
    lsil_proto_sample_cap: int = 200
    lsil_agg_mode: str = "topk"
    lsil_topk: int = 1

    # Clustering
    n_clusters: int = 5
    engine_mode: str = "C"
    auto_k: bool = True
    auto_algorithms: Optional[List[str]] = None
    c_min: int = 2
    c_max: int = 8

    # Redundancy
    kmsnc_k: int = 5
    build_redundancy_cache: Optional[str] = None
    build_redundancy_parallel: bool = False
    red_row_subsample: Optional[int] = None
    red_backend: str = "loky"
    red_batch_size: int = 500

    # Re-rank
    use_rerank: bool = False
    enable_rerank: bool = False        # alias lama untuk use_rerank
    rerank_mode: str = "ss_gower"
    rerank_topk: int = 15
    shadow_rerank: bool = False

    # Reward calibration (field lama — tetap diterima, diabaikan jika tidak dipakai)
    guard_every_calib: int = 50        # alias lama untuk guard_every di reward
    ss_max_n_cal: int = 200            # alias lama untuk ss_max_n_cal di reward
    reward_subsample_n: int = 20000    # alias lama untuk reward_subsample_n di reward
    calibrate_mode: str = "topk"       # alias lama
    calib_cache_enabled: bool = True   # alias lama

    # Structural Control
    run_structural_control: bool = True
    sc_lnc_threshold: float = 0.5
    sc_lnc_k: int = 50
    sc_max_iterations: int = 3

    # Phase B tuning (FIX v2+)
    hac_mode: str = "hybrid"
    cluster_adapter_lambda: float = 0.6
    enable_screening: bool = True
    screening_k_values: tuple = (2, 3, 4)
    screening_prune_threshold: float = 0.15

    # Misc
    random_state: int = 42
    verbose: bool = True
    show_progress: bool = True
    
    # DAV — Domain Anchor Variable (opsional, default None = nonaktif)
    dav_anchor_cols: Optional[List[str]] = None
    dav_lnc_global_threshold: float = 0.50
    dav_lnc_anchor_threshold: float = 0.40
    dav_lm_frac: float = 0.20


# ─────────────────────────────────────────────────────────────────
# Engine resolver
# ─────────────────────────────────────────────────────────────────
def _resolve_engine(df: pd.DataFrame, params: AUFSParams, n_clusters_user):
    n = len(df)
    mode = (params.engine_mode or "A").upper()

    if mode == "A":
        return "silhouette_gower", False, auto_adapter, n_clusters_user
    elif mode == "AB":
        return "lsil_fixed", False, auto_adapter, n_clusters_user
    elif mode == "C":
        if params.force_lsil_proxy:
            reward_metric = "lsil_fixed_calibrated"  # force — skip SS Gower exact
        else:
            reward_metric = "silhouette_gower" if n <= params.ss_max_n else "lsil_fixed_calibrated"
        dynamic_k = bool(params.auto_k)
        return reward_metric, dynamic_k, auto_adapter, n_clusters_user
    else:
        return params.reward_metric, False, auto_adapter, n_clusters_user


def _resolve_exploit_rate(params: AUFSParams) -> Optional[float]:
    if params.sa_exploit_sample_rate is not None:
        return params.sa_exploit_sample_rate
    if params.sa_neighbor_mode == "full":
        return 0.15
    return 0.30


def _resolve_subset_size_range(
    p_total: int, params: AUFSParams, verbose: bool = True
) -> Tuple[int, int, int]:
    if not params.adaptive_subset_size:
        min_size = params.sa_min_size
        max_size = params.sa_max_size or p_total
        mab_k = params.mab_k
        return mab_k, min_size, max_size

    min_size = max(2, int(np.ceil(params.subset_min_frac * p_total)))
    max_size = max(min_size + 1, int(np.floor(params.subset_max_frac * p_total)))
    max_size = min(p_total - 1, min_size + 2) if max_size >= p_total else max_size
    if min_size > max_size:
        min_size = max(2, max_size - 1)
    mab_k = max(min_size, min(max_size, (min_size + max_size) // 2))

    if verbose:
        print(f"[ADAPTIVE SIZE] p={p_total} → min={min_size}, max={max_size}, "
              f"mab_k={mab_k} (frac=[{params.subset_min_frac:.0%}, {params.subset_max_frac:.0%}])")
    return mab_k, min_size, max_size


# ─────────────────────────────────────────────────────────────────
# run_aufs_samba — MAIN FUNCTION
# ─────────────────────────────────────────────────────────────────
def run_aufs_samba(
    df_input: pd.DataFrame,
    n_clusters: int,
    cluster_fn: Callable = None,
    params: Optional[AUFSParams] = None,
    verbose: bool = True,
    return_info: bool = True,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    End-to-end AUFS-Samba:

      Phase A : SA feature selection (auto_adapter, fixed K)
                → reward.__phase_a_cache__ di-inject oleh make_sa_reward
      Cache   : _extract_phase_a_cache(reward_sa) → PhaseACache
      Phase B : Auto-K + Cluster Adapter
                → find_best_clustering_from_subsets(phase_a_cache=cache)
                → O(n·|L|) per trial (bukan O(n²))
      Post    : Structural Control (LNC* validation)
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

    # 2) Resolve engine
    reward_metric_resolved, dynamic_k, cluster_fn_resolved, n_clusters_eff = \
        _resolve_engine(df, params, n_clusters)
    exploit_rate_resolved = _resolve_exploit_rate(params)
    p_total = len(df.columns)
    k_mab_resolved, min_size_resolved, max_size_resolved = \
        _resolve_subset_size_range(p_total, params, verbose=verbose)

    if verbose:
        print(f"[ENGINE] mode={params.engine_mode} reward={reward_metric_resolved} "
              f"dynamic_k={dynamic_k} C_eff={'auto(Phase B)' if dynamic_k else n_clusters_eff}")
        print(f"[SUBSET] k_init={k_mab_resolved}, "
              f"range=[{min_size_resolved}, {max_size_resolved}] of {p_total} features")

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

    # 4) Build reward (Phase A)
    t0 = perf_counter()
    reward_sa = make_sa_reward(
        df_full=df, cat_cols=cat_cols,
        cluster_fn=cluster_fn_resolved,
        n_clusters=n_clusters_eff,
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
        dynamic_k=False,
        # field kalibrasi dari params (alias lama tetap kompatibel)
        guard_every=params.guard_every_calib,
        ss_max_n_cal=params.ss_max_n_cal,
        reward_subsample_n=params.reward_subsample_n,
        calibrate_mode=params.calibrate_mode,
        use_calib_cache=params.calib_cache_enabled,
    )
    timing["build_reward_s"] = perf_counter() - t0
    if verbose:
        print(f"[TIME] Build reward: {timing['build_reward_s']:.2f}s")

    # 5) MAB explore
    t0 = perf_counter()
    reward_for_mab = make_mab_reward_from_matrix(red_mat)
    mab_out, mab_stats = mab_explore(
        df, reward_for_mab, params.mab_T, k_mab_resolved, rng_py,
    )
    mab_subset = (
        max(mab_out, key=lambda x: x[1])[0]
        if mab_out else init_by_least_redundant(red_mat, k_mab_resolved)
    )
    mab_reward = reward_sa(mab_subset)
    least_subset = init_by_least_redundant(red_mat, k_mab_resolved)
    least_reward = reward_sa(least_subset)
    timing["mab_s"] = perf_counter() - t0
    if verbose:
        print(f"[TIME] MAB: {timing['mab_s']:.2f}s")

    # 6) SA (Phase A)
    archive = EliteArchive(cap=max(30, params.rerank_topk))

    def reward_logged(cols: List[str]) -> float:
        s = reward_sa(cols)
        archive.add(cols, s)
        return s

    t0 = perf_counter()
    best_cols_sa, best_reward, sa_stats = simulated_annealing(
        subset_init=mab_subset,
        all_features=df.columns.tolist(),
        eval_reward=reward_logged,
        iters=params.sa_iters,
        T0=params.sa_initial_temp,
        Tmin=params.sa_min_temp,
        alpha=params.sa_cooling_alpha,
        rng=rng_np,
        neighbor_mode=neighbor_mode_to_use,
        min_size=min_size_resolved,
        max_size=max_size_resolved,
        exploit_rate=exploit_rate_resolved,
        show_progress=params.show_progress,
        reward_cache={},
        cache_key_mode="sorted"
    )
    timing["sa_s"] = perf_counter() - t0
    if verbose:
        print(f"[TIME] SA: {timing['sa_s']:.2f}s")

    best_cols = best_cols_sa
    if best_cols:
        archive.add(best_cols, best_reward)

    # 7) Optional re-rank
    # support alias lama enable_rerank
    use_rerank = params.use_rerank or params.enable_rerank
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

    # ────────────────────────────────────────────────────────────────────
    # 8) EKSTRAK CACHE PHASE A
    #
    # reward_sa dibuat oleh make_sa_reward() dengan metric=
    # "lsil_fixed_calibrated". Fungsi tersebut meng-inject
    # __phase_a_cache__ ke dalam closure reward function.
    #
    # Kita ekstrak cache ini SETELAH SA selesai (Phase A done),
    # lalu teruskan ke Phase B.
    # ────────────────────────────────────────────────────────────────────
    phase_a_cache = _extract_phase_a_cache(reward_sa, df)

    if verbose:
        if phase_a_cache.available:
            print(
                f"[CACHE] Phase A cache tersedia: "
                f"|L|={phase_a_cache.n_landmarks}, "
                f"n={phase_a_cache.n_samples}, "
                f"X_num={phase_a_cache.X_num_full.shape if phase_a_cache.X_num_full is not None else None}, "
                f"X_cat={phase_a_cache.X_cat_full.shape if phase_a_cache.X_cat_full is not None else None}"
            )
        else:
            print("[CACHE] Phase A cache tidak tersedia — Phase B akan fallback ke scratch.")

    # ────────────────────────────────────────────────────────────────────
    # 9) PHASE B: Auto-K + Cluster Adapter
    #
    # Teruskan phase_a_cache ke find_best_clustering_from_subsets.
    # Di dalam fungsi itu, setiap (subset, algo, K) trial akan:
    #   - cluster ulang (diperlukan untuk auto-K)
    #   - evaluasi L-Sil via L_fixed + mask dari cache → O(n·|L|)
    #   - evaluasi LNC* via L_fixed + mask dari cache → O(n·|L|)
    # Total: ~0.5s/trial × 150 trial ≈ 75s (vs 4513s sebelumnya)
    # ────────────────────────────────────────────────────────────────────
    phaseB_timing = {}
    finalB = None
    final_algo = None
    final_C: Optional[int] = None
    final_labels = None

    if (params.engine_mode.upper() == "C") and params.auto_k:
        tB0 = perf_counter()
        cand_subsets = archive.topk(params.rerank_topk)
        if best_cols and best_cols not in cand_subsets:
            cand_subsets = [best_cols] + cand_subsets

        params_B = AUFSParams(**asdict(params))
        params_B.engine_mode = "C"
        params_B.auto_k = True

        # DAV: aktif jika dav_anchor_cols diisi
        _dav_Va = getattr(params, 'dav_anchor_cols', None)
        if _dav_Va:
            from mixclust.utils.dav import find_best_clustering_dav
            finalB = find_best_clustering_dav(
                df_full=df,
                top_subsets=cand_subsets,
                params=params_B,
                Va=_dav_Va,
                phase_a_cache=phase_a_cache,
                lnc_global_threshold=getattr(params, 'dav_lnc_global_threshold', 0.50),
                lnc_anchor_threshold=getattr(params, 'dav_lnc_anchor_threshold', 0.40),
                lm_frac=getattr(params, 'dav_lm_frac', 0.20),
                verbose=params.verbose,
            )
        else:
            finalB = find_best_clustering_from_subsets(
                df_full=df,
                top_subsets=cand_subsets,
                params=params_B,
                verbose=params.verbose,
                phase_a_cache=phase_a_cache,
            )
        phaseB_timing = finalB.get("timing_s", {})
        best_cols   = finalB.get("subset", best_cols) or best_cols
        final_algo  = finalB.get("algo", None)
        final_C     = finalB.get("k", None)
        final_labels = finalB.get("labels", None)
        best_reward = float(finalB.get("score_adj", finalB.get("score", best_reward)))
        timing["phaseB_s"] = perf_counter() - tB0
        if verbose:
            print(
                f"[TIME] Phase B: {timing['phaseB_s']:.2f}s "
                f"(cache={'HIT' if phase_a_cache.available else 'MISS'})"
            )

    # 10) Meta final (if not filled by Phase B)
    if final_labels is None:
        try:
            sub = df[best_cols]
            cat_idx_final = [
                sub.columns.get_loc(c) for c in sub.columns if c in cat_cols
            ]
            k_pass = n_clusters_eff if n_clusters_eff is not None else 2
            final_labels = cluster_fn_resolved(
                sub, cat_idx_final, int(k_pass), params.random_state
            )
            meta = getattr(cluster_fn_resolved, "_last", {}) or {}
            final_algo = final_algo or meta.get("algo", None)
            final_C = final_C or meta.get("C", n_clusters_eff)
        except Exception as _e:
            if verbose:
                print(f"[WARN] meta final failed: {_e}")

    # 11) Structural Control
    sc_result_dict = None
    uses_landmark = (reward_metric_resolved != "silhouette_gower")

    if finalB is not None:
        sc_result_dict = finalB.get("structural_control", None)

    if (
        sc_result_dict is None
        and params.run_structural_control
        and uses_landmark
        and final_labels is not None
    ):
        try:
            from mixclust.utils.controller import structural_control_lnc
            sub_sc = df[best_cols]
            cat_cols_sc = [c for c in best_cols if c in cat_cols]
            sc_obj = structural_control_lnc(
                X_df=sub_sc,
                labels=np.asarray(final_labels),
                cat_cols=cat_cols_sc,
                lnc_threshold=params.sc_lnc_threshold,
                lnc_k=params.sc_lnc_k,
                random_state=params.random_state,
                verbose=verbose,
            )
            from dataclasses import asdict as _asdict
            sc_result_dict = _asdict(sc_obj)
        except Exception as e:
            if verbose:
                print(f"[WARN] Structural Control gagal: {e}")

    # 12) Final SS-Gower (opsional, untuk laporan)
    ss_final = None
    try:
        sub_final = df[best_cols]
        cat_idx_f = [
            sub_final.columns.get_loc(c)
            for c in best_cols if c in cat_cols
        ]
        if final_labels is not None and len(set(final_labels)) > 1:
            ss_final = _ss_gower_for_subset(
                sub_final, np.asarray(final_labels),
                max_n=params.ss_max_n
            )
    except Exception:
        pass

    # 13) Metadata
    timing["total_s"] = perf_counter() - t_all0
    mask_all = np.array([(c in best_cols) for c in df_input.columns])
    mask_cat = np.array([(c in best_cols) and (c in cat_cols) for c in df_input.columns])
    mask_num = np.array([(c in best_cols) and (c in num_cols) for c in df_input.columns])

    init_source = "mab" if mab_out else "least_fallback"

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
        "final_labels": (
            final_labels.tolist()
            if isinstance(final_labels, np.ndarray) else final_labels
        ),
        "p_selected": len(best_cols),
        "final_ss_gower": float(ss_final) if ss_final is not None else None,
        "structural_control": sc_result_dict,

        "adaptive_size": {
            "p_total": p_total,
            "k_mab_resolved": k_mab_resolved,
            "min_size": min_size_resolved,
            "max_size": max_size_resolved,
        },

        # Info cache Phase A → Phase B
        "phase_b_config": {
            "phaseB_s": timing.get("phaseB_s"),
            "hac_mode_used": params.hac_mode,
            "composite_lambda": params.cluster_adapter_lambda,
            "screening_enabled": params.enable_screening,
            "cache_hit": phase_a_cache.available,       # ← BARU
            "n_landmarks_reused": phase_a_cache.n_landmarks,  # ← BARU
        },
    }

    if use_rerank:
        info["rerank_topk"] = int(params.rerank_topk)
        if params.shadow_rerank:
            info["shadow_rr_ss"] = float(rr_ss)
            info["shadow_rr_cols"] = rr_cols
        elif rr_cols:
            info["final_selection"] = "reranked_on_ss_gower"
            info["final_ss_gower"] = float(rr_ss)

    if phaseB_timing:
        info["timing_s"]["phaseB_detail"] = phaseB_timing

    return best_cols, info


# ─────────────────────────────────────────────────────────────────
# find_best_feature_subsets — Phase A only (untuk pipeline dua tahap)
# ─────────────────────────────────────────────────────────────────
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
        lsil_agg_mode=params.lsil_agg_mode,
        lsil_topk=params.lsil_topk,
        random_state=params.random_state,
    )

    reward_for_mab = make_mab_reward_from_matrix(red_mat)
    mab_out, mab_stats = mab_explore(
        df, reward_for_mab, params.mab_T, params.mab_k, rng_py,
    )
    mab_subset = (
        max(mab_out, key=lambda x: x[1])[0]
        if mab_out else init_by_least_redundant(red_mat, params.mab_k)
    )
    mab_reward = reward_sa(mab_subset)
    least_subset = init_by_least_redundant(red_mat, params.mab_k)
    least_reward = reward_sa(least_subset)

    archive = EliteArchive(cap=max(30, num_top_subsets))

    def reward_logged(cols: List[str]) -> float:
        s = reward_sa(cols)
        archive.add(cols, s)
        return s

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