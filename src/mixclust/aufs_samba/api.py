# dynamic_clustering/src/mixclust/aufs_samba/api.py
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
    sa_min_size: int = 2                     # cegah subset 1 fitur
    sa_max_size: Optional[int] = None
    sa_exploit_sample_rate: Optional[float] = None

    # Redundansi
    kmsnc_k: int = 5
    build_redundancy_parallel: bool = False
    build_redundancy_cache: Optional[str] = None
    # NEW (khusus data besar):
    red_row_subsample: Optional[int] = 50_000   # ← subsample baris saat bangun matriks redundansi
    red_backend: str = "loky"                   # ← pakai proses-based (lebih cepat dari threading)
    red_batch_size: int = 8                     # ← kurangi overhead task kecil

    # Reward
    reward_metric: str = "lsil_fixed"        # "lsil" | "lsil_fixed" | "lsil_fixed_calibrated" | "silhouette_gower"
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
    auto_k: bool = False          # aktifkan Auto-C hanya untuk Mode C

    # rentang C jika auto_k True
    c_min: int = 2
    c_max: int = 10

    # algoritma kandidat untuk Auto-C
    auto_algorithms: List[str] = None  # default nanti diisi ["kprototypes","hac_gower"]

    # Rerank akhir (opsional)
    enable_rerank: bool = False
    rerank_mode: str = "ss_gower"      # "ss_gower" | "none"
    rerank_topk: int = 15
    shadow_rerank: bool = False


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
    """
    Re-rank kandidat subset berdasarkan SS(Gower).
    Aman untuk dynamic_k: bila n_clusters_eff None, pass placeholder (2).
    """
    best_cols, best_ss = None, -1.0
    for cols in subsets:
        if not cols:
            continue
        sub = df_full[cols]
        cat_idx = [sub.columns.get_loc(c) for c in sub.columns
                   if sub[c].dtype.name in ("object", "category", "bool")]
        try:
            k_pass = n_clusters_eff if n_clusters_eff is not None else 2  # placeholder, diabaikan oleh auto_fn
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

    # Fase A: K statis
    params = AUFSParams(**asdict(params))
    params.auto_k = False

    rng_py = random.Random(params.random_state)
    rng_np = np.random.default_rng(params.random_state)

    # 1) Preprocess
    t0 = perf_counter()
    df, cat_cols, num_cols = preprocess_mixed_data(df_input)
    timing["preprocess_s"] = perf_counter() - t0

    cluster_fn_resolved = auto_adapter
    n_clusters_eff = n_clusters

    # 2) Redundansi
    t0 = perf_counter()
    red_mat = build_redundancy_matrix(df, k=params.kmsnc_k, cache_path=params.build_redundancy_cache)
    timing["redundancy_s"] = perf_counter() - t0

    # 3) Reward SA
    reward_sa = make_sa_reward(
        df_full=df,
        cat_cols=cat_cols,
        cluster_fn=cluster_fn_resolved,
        n_clusters=n_clusters_eff,
        metric=params.reward_metric,
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
    )

    # 4) MAB init
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

    # 5) SA
    _, _, sa_stats = simulated_annealing(
        subset_init=mab_subset,
        all_features=df.columns.tolist(),
        eval_reward=reward_logged,
        iters=params.sa_iters,
        T0=params.sa_initial_temp,
        Tmin=params.sa_min_temp,
        alpha=params.sa_cooling_alpha,
        rng=rng_np,
        neighbor_mode=params.sa_neighbor_mode,
        min_size=params.sa_min_size,
        max_size=params.sa_max_size,
        exploit_rate=params.sa_exploit_sample_rate,
        show_progress=params.show_progress,
        cache_key_mode="sorted",
    )

    timing["total_s"] = perf_counter() - t_all0

    # 6) keluarkan top-K
    top_k_subsets = archive.topk(num_top_subsets)
    info = {
        "timing_s": timing,
        "params": asdict(params),
        "sa_stats": sa_stats,
        "mab_stats": mab_stats,
        "top_subsets_from_archive": top_k_subsets,
        "init_source": "mab" if mab_out else "least_fallback",
        "init_subset": mab_subset,
        "init_reward": float(mab_reward),
        "least_subset": least_subset,
        "least_reward": float(least_reward),
    }
    if verbose:
        print(f"\n[FASE A SELESAI] Ditemukan {len(top_k_subsets)} kandidat subset fitur terbaik.")
    return top_k_subsets, info


# ---------------------------------------------------------------------
# Engine resolver
# ---------------------------------------------------------------------
def _resolve_engine(df: pd.DataFrame, params: AUFSParams, n_clusters_user):
    """Return (reward_metric, dynamic_k, cluster_fn, n_clusters_eff)."""
    n = len(df)
    algos = params.auto_algorithms or ["kprototypes", "hac_gower"]

    reward_metric = params.reward_metric
    dynamic_k = False
    cluster_fn = auto_adapter
    n_clusters_eff: Optional[int] = n_clusters_user

    mode = (params.engine_mode or "A").upper()
    if mode == "A":
        reward_metric = "silhouette_gower"
        dynamic_k = False
        cluster_fn = auto_adapter
        n_clusters_eff = n_clusters_user

    elif mode == "AB":
        reward_metric = "lsil_fixed"
        dynamic_k = False
        cluster_fn = auto_adapter
        n_clusters_eff = n_clusters_user

    elif mode == "C":
        reward_metric = "silhouette_gower" if n <= params.ss_max_n else "lsil_fixed_calibrated"
        dynamic_k = bool(params.auto_k)
        if dynamic_k:
            cluster_fn = make_auto_cluster_fn(
                algorithms=algos,
                c_range=range(params.c_min, params.c_max + 1),
                metric="auto",
                random_state=params.random_state,
                penalty_lambda=0.02
            )
            n_clusters_eff = None
        else:
            cluster_fn = auto_adapter
            n_clusters_eff = n_clusters_user
    else:
        dynamic_k = getattr(params, "dynamic_k", False)
        if dynamic_k:
            cluster_fn = make_auto_cluster_fn(
                algorithms=algos,
                c_range=range(params.c_min, params.c_max + 1),
                metric="auto",
                random_state=params.random_state
            )
            n_clusters_eff = None
        else:
            cluster_fn = auto_adapter
            n_clusters_eff = n_clusters_user

    return reward_metric, dynamic_k, cluster_fn, n_clusters_eff


# ---------------------------------------------------------------------
# AUFS end-to-end
# ---------------------------------------------------------------------
def run_aufs_samba(
    df_input: pd.DataFrame,
    n_clusters: int,
    cluster_fn: Callable[[pd.DataFrame, List[int], int, Optional[int]], np.ndarray],
    params: Optional[AUFSParams] = None,
    verbose: bool = True,
    return_info: bool = True,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    End-to-end AUFS-Samba (Fase A + optional re-rank + optional Phase B Auto-K).
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

    # Resolve engine
    reward_metric_resolved, dynamic_k, cluster_fn_resolved, n_clusters_eff = _resolve_engine(df, params, n_clusters)
    if verbose:
        print(f"[ENGINE] mode={params.engine_mode} reward={reward_metric_resolved} "
              f"dynamic_k={dynamic_k} C_eff={'auto' if n_clusters_eff is None else n_clusters_eff}")

    # Engine C → default neighbor jadi "full" bila user belum set
    neighbor_mode_to_use = params.sa_neighbor_mode
    if (params.engine_mode or "").upper() == "C" and neighbor_mode_to_use == "swap":
        neighbor_mode_to_use = "full"

    # 2) Redundancy matrix
    t0 = perf_counter()
    red_mat = build_redundancy_matrix(
        df,
        k=params.kmsnc_k,
        cache_path=params.build_redundancy_cache,
        precompute=True,
        use_parallel=params.build_redundancy_parallel,
        n_jobs=params.mab_n_jobs,
        # NEW:
        row_subsample=params.red_row_subsample,
        backend=params.red_backend,
        batch_size=params.red_batch_size
    )
    timing["redundancy_s"] = perf_counter() - t0
    if verbose:
        print(f"[TIME] Redundancy matrix: {timing['redundancy_s']:.2f}s")

    # 3) Reward SA
    t0 = perf_counter()
    reward_sa = make_sa_reward(
        df_full=df,
        cat_cols=cat_cols,
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
        dynamic_k=dynamic_k
    )
    timing["build_reward_s"] = perf_counter() - t0
    if verbose:
        print(f"[TIME] Build reward: {timing['build_reward_s']:.2f}s")

    # 4) MAB init (pakai reward kMSNC* utk eksplorasi cepat)
    feats = df.columns.tolist()
    k_mab = min(params.mab_k, len(feats)) if feats else 0
    if k_mab == 0:
        return [], {"reason": "no_features"}

    reward_for_mab = make_mab_reward_from_matrix(red_mat)
    t0 = perf_counter()
    mab_out, mab_stats = mab_explore(
        df,
        reward_for_mab,
        params.mab_T,
        k_mab,
        rng_py,
        red_matrix=red_mat,
        red_threshold=params.mab_redundancy_threshold,
        penalty_beta=params.mab_penalty_beta,
        show_progress=params.show_progress,
    )
    timing["mab_s"] = perf_counter() - t0
    if verbose:
        print(f"[TIME] MAB: {timing['mab_s']:.2f}s")

    # 5) pilih init subset
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

    # 6) SA (mulai dari MAB)
    reward_cache: Dict[Tuple[str, ...], float] = {}
    use_rerank = (
        params.enable_rerank and
        params.rerank_mode != "none" and
        reward_metric_resolved != "silhouette_gower"   # kalau reward sudah SS, re-rank SS tidak perlu
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
        min_size=params.sa_min_size,
        max_size=params.sa_max_size,
        exploit_rate=params.sa_exploit_sample_rate,
        show_progress=params.show_progress,
        reward_cache=reward_cache,
        cache_key_mode="sorted"
    )
    timing["sa_s"] = perf_counter() - t0
    if verbose:
        print(f"[TIME] SA   : {timing['sa_s']:.2f}s")

    # 7) Re-rank kandidat elit di SS(Gower) (opsional)
    rr_cols: List[str] = []
    rr_ss: float = float("-inf")
    if use_rerank and params.rerank_mode == "ss_gower":
        t0 = perf_counter()
        cands = archive.topk(params.rerank_topk)
        if best_cols and best_cols not in cands:
            cands = [best_cols] + cands
        rr_cols, rr_ss = _rerank_on_ss_gower(
            df, cands, cluster_fn_resolved, n_clusters_eff, params.ss_max_n, params.random_state
        )
        timing["rr_s"] = perf_counter() - t0
        if verbose:
            print(f"[TIME] Re-rank : {timing['rr_s']:.2f}s")
        if rr_cols and not params.shadow_rerank:
            best_cols = rr_cols
            best_reward = reward_sa(best_cols)
            if verbose:
                print(f"[RERANK] pilih {len(best_cols)} fitur (SS={rr_ss:.4f})")

    # 8) Phase B (Auto-K) hanya saat Engine C + auto_k
    phaseB_timing = {}
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
            verbose=params.verbose
        )
        phaseB_timing = finalB.get("timing_s", {})
        best_cols   = finalB.get("subset", best_cols) or best_cols
        final_algo  = finalB.get("algo", None)
        final_C     = finalB.get("k", None)
        final_labels = finalB.get("labels", None)
        best_reward = float(finalB.get("score_adj", finalB.get("score", best_reward)))
        timing["phaseB_s"] = perf_counter() - tB0

    # 9) Meta final (satu kali, kalau belum terisi oleh Phase B)
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
                print(f"[WARN] meta final gagal: {_e}")

    # --- SS(Gower) final utk subset terpilih (tanpa rerank mahal)
    try:
        if best_cols and final_labels is not None:
            sub = df[best_cols]
            ss_final = _ss_gower_for_subset(sub, np.asarray(final_labels), max_n=params.ss_max_n)
        else:
            ss_final = None
    except Exception:
        ss_final = None
    
    info["final_ss_gower"] = float(ss_final) if ss_final is not None else None

    
    timing["total_s"] = perf_counter() - t_all0
    if verbose:
        print(f"[TIME] TOTAL: {timing['total_s']:.2f}s")

    # 10) Mask untuk downstream
    sel = set(best_cols)
    mask_all = [c in sel for c in df_input.columns]
    mask_cat = np.array([(c in sel) and (c in cat_cols) for c in df_input.columns])
    mask_num = np.array([(c in sel) and (c in num_cols) for c in df_input.columns])

    # 11) Info
    info: Dict[str, Any] = {
        "timing_s": timing,
        "n_features": df.shape[1],
        "k_selected": len(best_cols),
        "best_reward": float(best_reward),
        "used_metric": reward_metric_resolved,  # ← metric resolved yang benar
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
    }

    # info tambahan khusus re-rank
    if use_rerank:
        info["rerank_topk"] = int(params.rerank_topk)
        if params.shadow_rerank:
            info["shadow_rr_ss"] = float(rr_ss)
            info["shadow_rr_cols"] = rr_cols
        elif rr_cols:
            info["final_selection"] = "reranked_on_ss_gower"
            info["final_ss_gower"] = float(rr_ss)

    # gabungkan timing Phase B jika ada
    if phaseB_timing:
        info.setdefault("timing_s", {}).update({"phaseB_s": phaseB_timing.get("phaseB_s", None)})

    return best_cols, info
