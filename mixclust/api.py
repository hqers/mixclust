# mixclust/api.py
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

from .core.preprocess import preprocess_mixed_data, prepare_mixed_arrays_no_label
from .aufs.redundancy import build_redundancy_matrix, init_by_least_redundant, make_mab_reward_from_matrix
from .aufs.mab import mab_explore
from .aufs.sa import simulated_annealing
from .aufs.reward import make_sa_reward
from .metrics.silhouette import full_silhouette_gower_subsample
from .clustering.controller import (
    make_auto_cluster_fn,
    find_best_clustering_from_subsets,
)
from .clustering.cluster_adapters import auto_adapter

# ── BARU: PhaseACache infrastructure ──
from .aufs.phase_a_cache import PhaseACache, _extract_phase_a_cache


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
    sa_min_size: int = 3               # validated default
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
    auto_reward: bool = True        # False = pakai reward_metric dari params langsung
    lsil_agg_mode: str = "topk"
    lsil_topk: int = 3              # paper default r=3
    lsil_c: float = 3.0            # |L|=c*sqrt(n), Theorem 1 JDSA
    lsil_cap_frac: float = 0.2     # batas atas landmark fraction

    # Reward v2.2 — percepatan build_reward & SA reward
    lsil_eval_n: int = 20_000           # [A] n untuk evaluasi per reward call
    lsil_c_reward: Optional[float] = None  # [A] c untuk landmark eval
    subsample_n_cluster: int = 6_000    # [B] n untuk initial clustering

    # Clustering
    n_clusters: int = 5
    engine_mode: str = "C"
    auto_k: bool = True
    auto_algorithms: Optional[List[str]] = None
    c_min: int = 2
    c_max: int = 8

    # Phase B tuning
    phase_b_eval_n: int = 30_000       # subsample L-Sil di Phase B
    phase_b_skip_lnc: bool = True      # skip LNC* per trial (default True — hemat ~30s/trial)

    # Redundancy
    kmsnc_k: int = 5
    build_redundancy_cache: Optional[str] = None
    build_redundancy_parallel: bool = True    # parallel build (default True)
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
    screening_prune_threshold: float = 0.20  # validated default

    # Misc
    random_state: int = 42
    verbose: bool = True
    show_progress: bool = True

    # v1.1.12 — landmark strategy for Phase B cache
    # "cluster_aware" (default): cluster-aware landmarks, paper JDSA default.
    #   80% central + 20% boundary per cluster — optimal BCVD mitigation
    #   for known K, but biased toward K_hint when evaluating other K values.
    # "kcenter": K-agnostic k-center greedy landmarks.
    #   Maximally spread in feature space — fair evaluation for all K values,
    #   slightly higher BCVD risk but eliminates auto-K bias toward K_hint.
    # Recommendation: use "kcenter" when auto_k=True and c_min < c_max,
    #   use "cluster_aware" when K is fixed or known a priori.
    landmark_mode: str = "cluster_aware"
    
    # DAV — Domain Anchor Variable (opsional, default None = nonaktif)
    dav_anchor_cols: Optional[List[str]] = None
    dav_lnc_global_threshold: float = 0.50
    dav_lnc_anchor_threshold: float = 0.40
    dav_lm_c: float = 3.0                    # v1.1.10: landmark count = lm_c * sqrt(n_sub)
    dav_anchor_subsample_n: int = 10_000     # v1.1.10: subsample size for AnchorContext
    dav_lm_frac: float = 0.20               # deprecated — kept for backward compat, not used


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
        if params.auto_reward:
            reward_metric = "silhouette_gower" if n <= params.ss_max_n else "lsil_fixed_calibrated"
        else:
            reward_metric = params.reward_metric  # pakai pilihan user langsung
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
        lsil_agg_mode=params.lsil_agg_mode,
        lsil_topk=params.lsil_topk,
        lsil_c=params.lsil_c,
        lsil_cap_frac=params.lsil_cap_frac,
        random_state=params.random_state,
        dynamic_k=False,
        guard_every=params.guard_every_calib,
        ss_max_n_cal=params.ss_max_n_cal,
        reward_subsample_n=params.reward_subsample_n,
        calibrate_mode=params.calibrate_mode,
        use_calib_cache=params.calib_cache_enabled,
        lsil_eval_n=params.lsil_eval_n,
        lsil_c_reward=params.lsil_c_reward,
        subsample_n_cluster=params.subsample_n_cluster,
        landmark_mode=getattr(params, 'landmark_mode', 'cluster_aware'),
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
    phase_a_cache = _extract_phase_a_cache(
        reward_sa, df,
        phase_b_eval_n=getattr(params, 'phase_b_eval_n', 30_000),
    )

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

        # v1.1.13: dedup — EliteArchive stores sorted key but topk returns
        # original order, so ['a','b'] and ['b','a'] can both appear.
        # Remove duplicates while preserving rank order.
        _seen = set()
        _deduped = []
        for _sub in cand_subsets:
            _key = tuple(sorted(_sub))
            if _key not in _seen:
                _seen.add(_key)
                _deduped.append(_sub)
        if len(_deduped) < len(cand_subsets) and verbose:
            print(f"  [Phase B] Dedup: {len(cand_subsets)} → {len(_deduped)} subsets")
        cand_subsets = _deduped

        params_B = AUFSParams(**asdict(params))
        params_B.engine_mode = "C"
        params_B.auto_k = True

        # DAV: aktif jika dav_anchor_cols diisi
        _dav_Va = getattr(params, 'dav_anchor_cols', None)
        if _dav_Va:
            from .utils.dav import find_best_clustering_dav
            finalB = find_best_clustering_dav(
                df_full=df,
                top_subsets=cand_subsets,
                params=params_B,
                Va=_dav_Va,
                phase_a_cache=phase_a_cache,
                lnc_global_threshold=getattr(params, 'dav_lnc_global_threshold', 0.50),
                lnc_anchor_threshold=getattr(params, 'dav_lnc_anchor_threshold', 0.25),
                lm_c=getattr(params, 'dav_lm_c', 3.0),
                anchor_subsample_n=getattr(params, 'dav_anchor_subsample_n', 10_000),
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
            from .clustering.controller import structural_control_lnc
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
        lsil_agg_mode=params.lsil_agg_mode,
        lsil_topk=params.lsil_topk,
        lsil_c=params.lsil_c,
        lsil_cap_frac=params.lsil_cap_frac,
        random_state=params.random_state,
        lsil_eval_n=params.lsil_eval_n,
        lsil_c_reward=params.lsil_c_reward,
        subsample_n_cluster=params.subsample_n_cluster,
        landmark_mode=getattr(params, 'landmark_mode', 'cluster_aware'),
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

# ─────────────────────────────────────────────────────────────────
# v1.1.12 — auto_params: self-configuring AUFSParams from data
# ─────────────────────────────────────────────────────────────────

def _profile_data(df: pd.DataFrame) -> dict:
    """
    Extract data signals for adaptive parameter selection.
    O(n*p) — runs once before any clustering.
    All signals derivable without labels or domain knowledge.
    """
    n, p = df.shape
    cat_cols    = df.select_dtypes(include=['object','category','bool']).columns
    binary_cols = [c for c in df.columns if df[c].nunique() == 2]
    spike_cols  = [c for c in df.columns
                   if df[c].value_counts(normalize=True).iloc[0] > 0.5]
    missing     = float(df.isna().mean().mean())
    cat_ratio    = len(cat_cols) / p
    binary_ratio = len(binary_cols) / p
    spike_ratio  = len(spike_cols) / p
    # entropy proxy: spike = low entropy, no-spike = high entropy
    entropy_ratio = max(0.0, 1.0 - spike_ratio)
    return {
        'n': n, 'p': p,
        'cat_ratio':     cat_ratio,
        'binary_ratio':  binary_ratio,
        'spike_ratio':   spike_ratio,
        'entropy_ratio': entropy_ratio,
        'missing_ratio': missing,
        'n_ratio':       n / 10_000,
    }


def auto_params(
    df: pd.DataFrame,
    **overrides,
) -> 'AUFSParams':
    """
    Derive ALL sensible AUFSParams from data characteristics.
    Target: zero manual parameter tuning for standard use cases.

    Signals extracted from df (O(n*p), runs once):
      n_ratio      = n / 10_000
      cat_ratio    = p_cat / p
      binary_ratio = p_binary / p       (geometric dominance risk)
      spike_ratio  = p_spike / p        (cols where top value >50%)
      entropy_ratio= 1 - spike_ratio    (proxy for data diversity)
      missing_ratio= mean(isna fraction)

    Auto-computed parameters (13 total):
      lsil_c, c_max, screening_k_values  — landmark & K range
      landmark_mode                       — kcenter vs cluster_aware
      cluster_adapter_lambda              — L-Sil vs LNC* balance
      auto_algorithms                     — algorithm candidates
      mab_T, mab_k, sa_iters             — MAB/SA exploration
      lsil_topk                           — topk aggregation
      subsample_n_cluster                 — Phase A cluster subsample
      phase_b_eval_n                      — Phase B eval subsample
      lsil_eval_n                         — SA reward eval subsample
      sc_lnc_threshold                    — structural control threshold

    Fixed regardless of data (controlled by engine_mode user arg):
      engine_mode     : "C" (full MixClust) unless user passes "A" (AUFS only)
      sa_neighbor_mode: "full" for engine C, "swap" for engine A
      hac_mode        : "hybrid" always (production mode)
      phase_b_skip_lnc: True always (speed optimization)

    User-controlled (not auto):
      random_state, verbose, show_progress
      build_redundancy_cache, dav_anchor_cols

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to cluster (features only, no label column).
    **overrides
        Any AUFSParams field — takes precedence over auto values.
        Pass engine_mode="A" for AUFS-Samba standalone.

    Example
    -------
    >>> params = auto_params(df, random_state=42)
    >>> result = run_generic_end2end(df, outdir='out/', params=params)

    >>> # AUFS-Samba only (no controller, no auto-K):
    >>> params = auto_params(df, engine_mode="A", random_state=42)
    """
    import math

    prof = _profile_data(df)
    n           = prof['n']
    p           = prof['p']
    cat_ratio   = prof['cat_ratio']
    binary_ratio= prof['binary_ratio']
    spike_ratio = prof['spike_ratio']
    entropy_r   = prof['entropy_ratio']
    n_ratio     = prof['n_ratio']

    c_min = overrides.get('c_min', 2)

    # ── 1. lsil_c — log-proportional, floor 3.0 (Theorem 1) ─────
    c_lsil = max(3.0, 3.0 * math.log10(max(n, 10)) / math.log10(1000))
    c_lsil = round(c_lsil, 1)

    # ── 2. c_max ─────────────────────────────────────────────────
    # Cap 10: K > 10 hampir tidak pernah interpretatif untuk data survei
    # maupun UCI benchmark. User bisa override via overrides={'c_max': N}.
    # Dataset kecil (n < 1500) tidak terpengaruh karena formula log2/sqrt
    # sudah menghasilkan ≤ 10 secara alami.
    c_max = min(int(math.log2(max(n, 4))), int(math.sqrt(n / 2)), 10)

    # ── 3. screening_k_values — evenly-spaced from c_range ───────
    c_range = list(range(c_min, c_max + 1))
    n_pts   = min(4, len(c_range))
    idx     = np.linspace(0, len(c_range) - 1, n_pts, dtype=int)
    screening_k = tuple(c_range[i] for i in idx)

    # ── 4. landmark_mode ─────────────────────────────────────────
    # kcenter if: large n (K range matters more), or binary/spike risk
    # cluster_aware if: small n, low geometric dominance risk
    geo_dom_risk = (n_ratio > 1) or (binary_ratio > 0.3) or (spike_ratio > 0.4)
    landmark_mode = "kcenter" if geo_dom_risk else "cluster_aware"

    # ── 5. cluster_adapter_lambda ────────────────────────────────
    # cat-heavy → lower lambda (trust LNC* structure more)
    # num-heavy → higher lambda (L-Sil more reliable)
    if cat_ratio > 0.6:
        lam = 0.4
    elif cat_ratio < 0.3:
        lam = 0.7
    else:
        lam = 0.6

    # ── 6. auto_algorithms ───────────────────────────────────────
    # kamila = model-based EM, not feasible for n > 10K
    if cat_ratio == 0.0:
        algos = ["kmeans"]
    elif n > 10_000:
        algos = ["kprototypes", "hac_gower"]
    else:
        algos = ["kprototypes", "hac_gower"]

    # ── 7. MAB/SA — scale with data diversity ────────────────────
    mab_T    = max(8,  min(20,  int(12 * (1 + 0.3 * entropy_r))))
    mab_k    = max(3,  min(10,  p // 4))
    sa_iters = max(30, min(100, int(50 * (1 + 0.3 * entropy_r))))

    # ── 8. lsil_topk — scale with K range ────────────────────────
    lsil_topk = max(2, min(5, c_max - c_min))

    # ── 9. Subsample sizes — proportional to n ───────────────────
    # S6: high missing_ratio → reduce subsample (missing rows less useful)
    # missing > 0.3 → scale down by 30%; missing > 0.5 → scale down 50%
    missing_scale = max(0.5, 1.0 - prof['missing_ratio'])
    sub_n    = min(n, max(2000,  int(0.02 * n * missing_scale)))  # Phase A cluster
    pb_eval  = min(n, max(10000, int(0.10 * n)))                  # Phase B L-Sil eval

    # v1.1.13: cap lsil_eval_n lebih agresif untuk n besar
    # SA reward menggunakan lsil_eval_n — terlalu besar → SA lambat
    # SA hanya butuh sinyal arah, bukan presisi tinggi → 3% n cukup
    lsil_eval = min(n, max(5000, int(0.03 * n)))   # SA reward eval (3%, down from 6%)

    # ── 9b. lsil_c_reward — c khusus untuk SA reward evaluation ──
    # v1.1.13: pisahkan c untuk SA (cepat) vs Phase B (akurat)
    # SA reward hanya butuh ranking relatif antar subset, bukan nilai absolut
    # → c_reward kecil (|L| kecil) → tiap reward call jauh lebih cepat
    # → lsil_c tetap besar untuk Phase B evaluation (akurasi tinggi)
    # Cap di 2.0 agar |L|_reward ≈ 2√n — cukup untuk ranking tapi ringan
    c_reward = round(min(2.0, c_lsil), 1)

    # ── 10. sc_lnc_threshold — relax for large n ─────────────────
    # Large n → more heterogeneous → LNC* naturally lower
    if n < 50_000:
        sc_thr = 0.5
    else:
        sc_thr = round(max(0.4, 0.5 - 0.1 * math.log10(n / 50_000)), 2)

    # ── 11. engine_mode → sa_neighbor_mode ───────────────────────
    engine = overrides.get('engine_mode', 'C')
    sa_neighbor = "swap" if engine == "A" else "full"

    auto = dict(
        # landmark & K range
        lsil_c             = c_lsil,
        lsil_c_reward      = c_reward,   # v1.1.13: SA pakai c kecil, Phase B c besar
        c_min              = c_min,
        c_max              = c_max,
        screening_k_values = screening_k,
        landmark_mode      = landmark_mode,
        # clustering strategy
        cluster_adapter_lambda = lam,
        auto_algorithms    = algos,
        # MAB / SA
        mab_T              = mab_T,
        mab_k              = mab_k,
        sa_iters           = sa_iters,
        # eval
        lsil_topk          = lsil_topk,
        subsample_n_cluster= sub_n,
        phase_b_eval_n     = pb_eval,
        lsil_eval_n        = lsil_eval,
        # structural control
        sc_lnc_threshold   = sc_thr,
        # fixed/engine-driven
        engine_mode        = engine,
        sa_neighbor_mode   = sa_neighbor,
        hac_mode           = "hybrid",
        phase_b_skip_lnc   = True,
    )
    # user overrides take precedence
    auto.update(overrides)

    if auto.get('verbose', False):
        lm_tag = "kcenter (K-agnostic)" if auto['landmark_mode']=="kcenter" \
                 else "cluster_aware (JDSA default)"
        print(f"[auto_params] n={n:,}  p={p}")
        print(f"  Signals: cat={cat_ratio:.0%}  binary={binary_ratio:.0%}  "
              f"spike={spike_ratio:.0%}  missing={prof['missing_ratio']:.0%}")
        print(f"  lsil_c={auto['lsil_c']}  |L|_phaseB={int(auto['lsil_c']*n**0.5):,}  "
              f"lsil_c_reward={auto['lsil_c_reward']}  |L|_SA={int(auto['lsil_c_reward']*n**0.5):,}  "
              f"c_max={auto['c_max']}")
        print(f"  landmark_mode  = {lm_tag}")
        print(f"  lambda         = {auto['cluster_adapter_lambda']}  "
              f"algos={auto['auto_algorithms']}")
        print(f"  mab_T={auto['mab_T']}  mab_k={auto['mab_k']}  "
              f"sa_iters={auto['sa_iters']}  lsil_topk={auto['lsil_topk']}")
        print(f"  sub_n={auto['subsample_n_cluster']:,}  "
              f"pb_eval={auto['phase_b_eval_n']:,}  "
              f"lsil_eval={auto['lsil_eval_n']:,}")
        print(f"  sc_lnc_threshold={auto['sc_lnc_threshold']}  "
              f"engine={auto['engine_mode']}  "
              f"sa_neighbor={auto['sa_neighbor_mode']}")
        if geo_dom_risk:
            print(f"  ⚠  Geometric dominance risk detected "
                  f"(n_ratio={n_ratio:.1f}, binary={binary_ratio:.0%}, "
                  f"spike={spike_ratio:.0%})")
            if spike_ratio > 0.4 and binary_ratio > 0.2:
                print(f"  ⚠  Consider DAV if domain anchor variables are available")

    return AUFSParams(**{k: v for k, v in auto.items()
                         if k in AUFSParams.__dataclass_fields__})
