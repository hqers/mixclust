# mixclust/aufs/reward.py
#
# CHANGELOG v2.2 — Percepatan build_reward & SA reward
#
# TIGA MASALAH PERFORMA YANG DIPERBAIKI:
#
# [A] gower_distances_to_landmarks dipanggil dengan X_num_full (n=334k)
#     → O(n*m*p) dengan m=c*sqrt(334k)=1734, ~20s per reward call
#     FIX: evaluasi L-Sil hanya pada subsample eval (lsil_eval_n≤20k).
#          n_eval=20k → m=424 → ~0.3s per reward (67x lebih cepat).
#
# [B] Initial clustering via subsample_and_propagate_labels (20k rows)
#     → kprototypes 20k: ~30–120s per build_reward
#     FIX: _fast_cluster_subsample pakai subsample_n_cluster=6_000.
#          kprototypes 6k: ~3–8s (10–20x lebih cepat).
#
# [C] lsil_using_prototypes_gower dipanggil dengan protos0 di posisi X_num
#     setelah refactor (bug v2.0) → semua reward = -1.0
#     FIX (v2.1, tetap): gunakan lsil_using_landmarks langsung.
#
# ESTIMASI SPEEDUP vs versi asli:
#   build_reward: ~120s → ~8s  ([B] cluster 6k + [A] gower 20k)
#   per SA reward call: ~20s  → ~0.3s  ([A] eval subsample)
#   total SA 50 iter: ~1000s → ~15s
#
# BACKWARD COMPATIBLE: semua parameter lama tetap diterima.
# Parameter baru (opsional): lsil_eval_n, lsil_c_reward, subsample_n_cluster.
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations
from typing import Callable, List, Optional, Dict, Tuple
import numpy as np, pandas as pd
from time import perf_counter

from ..core.prototypes import build_prototypes_by_cluster_gower
from ..metrics.lsil import lsil_using_landmarks
from ..metrics.lsil import lsil_using_prototypes_gower   # backward-compat
from ..metrics.silhouette import full_silhouette_gower_subsample
from ..core.adaptive import adaptive_landmark_count
from ..core.landmarks import (
    subsample_and_propagate_labels,
    select_landmarks_kcenter,
    select_landmarks_cluster_aware,
    cluster_aware_landmarks_on_subsample,
    stratified_landmarks,
)
from ..core.features import build_features
from ..core.preprocess import prepare_mixed_arrays_no_label
from ..metrics.silhouette import full_silhouette_gower

try:
    from .redundancy import redundancy_penalty
except Exception:
    def redundancy_penalty(cols, red_mat, mode="mean_invert"):
        return 0.0


# ================================================================
# [B] Fast initial clustering — subsample kecil
# ================================================================

def _fast_cluster_subsample(
    df_full: pd.DataFrame,
    cat_cols_full: List[str],
    cluster_fn: Callable,
    n_clusters: int,
    random_state: int,
    subsample_n_cluster: int = 6_000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cluster pada subsample kecil (subsample_n_cluster rows), lalu
    propagasi label ke seluruh data via NN numerik.

    Keunggulan vs subsample_and_propagate_labels (20k):
      kprototypes 6k rows ~3–8s vs ~30–120s pada 20k.

    Returns: (labels_full, idx_sub, labels_sub)
    """
    from sklearn.neighbors import NearestNeighbors

    n = len(df_full)
    rng = np.random.default_rng(random_state)

    sub_n = min(subsample_n_cluster, n)
    idx_sub = rng.choice(n, size=sub_n, replace=False)
    df_sub  = df_full.iloc[idx_sub].reset_index(drop=True)
    cat_idx_sub = [
        df_sub.columns.get_loc(c)
        for c in df_sub.columns if c in cat_cols_full
    ]

    try:
        labels_sub = np.asarray(
            cluster_fn(df_sub, cat_idx_sub, n_clusters, random_state), dtype=int
        )
        if len(np.unique(labels_sub)) < 2:
            raise ValueError(
                f"hanya {len(np.unique(labels_sub))} klaster pada subsample {sub_n}"
            )
    except Exception as e:
        print(f"⚠️  [reward] cluster gagal (sub_n={sub_n}): {e} — pakai fallback round-robin")
        labels_sub = np.array([i % n_clusters for i in range(sub_n)], dtype=int)

    # Propagasi via NN numerik
    Xn_full, _, _, _, _, _, _ = prepare_mixed_arrays_no_label(df_full)
    Xn_sub,  _, _, _, _, _, _ = prepare_mixed_arrays_no_label(df_sub)

    if Xn_sub.shape[1] > 0 and Xn_full.shape[1] > 0:
        nn = NearestNeighbors(n_neighbors=1, algorithm="ball_tree", n_jobs=1)
        nn.fit(Xn_sub)
        _, nn_idx = nn.kneighbors(Xn_full)
        labels_full = labels_sub[nn_idx.ravel()]
    else:
        # Fallback tanpa numerik
        labels_full = np.array([labels_sub[i % sub_n] for i in range(n)], dtype=int)

    return np.asarray(labels_full, dtype=int), idx_sub, labels_sub


# ================================================================
# [A] Subsample evaluasi reward — subsample dari full data
# ================================================================

def _prepare_eval_subsample(
    df_full: pd.DataFrame,
    labels_full: np.ndarray,
    eval_n: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray]:
    """
    Ambil subsample stratified dari df_full untuk evaluasi reward.
    Reward dievaluasi hanya pada eval_n rows → ~eval_n/n_full kali lebih cepat.

    Returns: (X_num_e, X_cat_e, num_min_e, num_max_e, inv_rng_e, labels_e, idx_eval)
    """
    n = len(df_full)
    rng = np.random.default_rng(random_state + 7777)

    if n <= eval_n:
        idx_eval = np.arange(n, dtype=int)
    else:
        # Stratified per klaster
        idx_list = []
        uniq, counts = np.unique(labels_full, return_counts=True)
        for c, cnt in zip(uniq, counts):
            take = max(3, int(round(eval_n * cnt / n)))
            pool = np.where(labels_full == c)[0]
            take = min(take, len(pool))
            idx_list.append(rng.choice(pool, size=take, replace=False))
        idx_eval = np.unique(np.concatenate(idx_list))
        if len(idx_eval) > eval_n:
            idx_eval = rng.choice(idx_eval, size=eval_n, replace=False)
        idx_eval = np.sort(idx_eval)

    df_eval = df_full.iloc[idx_eval].reset_index(drop=True)
    X_num_e, X_cat_e, num_min_e, num_max_e, _, _, inv_rng_e = \
        prepare_mixed_arrays_no_label(df_eval)
    labels_e = labels_full[idx_eval]

    return X_num_e, X_cat_e, num_min_e, num_max_e, inv_rng_e, labels_e, idx_eval


def _stratified_landmarks_local(
    y: np.ndarray, m_target: int, per_cluster_min: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n = len(y)
    if m_target <= 0 or m_target >= n:
        return np.arange(n, dtype=int)
    L = []
    vals, counts = np.unique(y, return_counts=True)
    q = {
        c: max(per_cluster_min, int(round(m_target * counts[i] / n)))
        for i, c in enumerate(vals)
    }
    for c in vals:
        pool = np.where(y == c)[0]
        take = min(len(pool), q[c])
        if take > 0:
            L.extend(rng.choice(pool, size=take, replace=False).tolist())
    if not L:
        return np.arange(n, dtype=int)
    if len(L) > m_target:
        L = L[:m_target]
    return np.array(sorted(L), dtype=int)


# ================================================================
# make_sa_reward
# ================================================================

def make_sa_reward(
    df_full,
    cat_cols,
    cluster_fn,
    n_clusters,
    metric="silhouette_gower",
    use_redundancy_penalty=False,
    alpha_penalty=0.3,
    redundancy_matrix=None,
    ss_max_n=2000,
    lsil_c: float = 3.0,
    lsil_cap_frac: float = 0.2,
    lsil_agg_mode="mean",
    lsil_topk=5,
    random_state=42,
    dynamic_k: bool = False,
    guard_every: int = 50,
    ss_max_n_cal: int = 200,
    reward_subsample_n: int = 20000,
    calibrate_mode: str = "topk",
    use_calib_cache: bool = True,
    # ── Parameter baru v2.2 ──────────────────────────────────────
    lsil_eval_n: int = 20_000,
    lsil_c_reward: Optional[float] = None,
    subsample_n_cluster: int = 6_000,
    # ── Parameter baru v1.1.12 ──────────────────────────────────
    # "cluster_aware" (default, paper JDSA): cluster-aware landmarks
    #   80% central + 20% boundary — optimal BCVD mitigation for known K,
    #   biased toward K_hint when auto_k evaluates other K values.
    # "kcenter": K-agnostic k-center greedy landmarks.
    #   Maximally spread — fair for all K values, slightly higher BCVD risk.
    #   Recommended when auto_k=True and c_min < c_max.
    landmark_mode: str = "cluster_aware",
):
    n_full = len(df_full)

    # ────────────────────────────────────────────────────────────
    # CABANG 1: silhouette_gower
    # ────────────────────────────────────────────────────────────
    if metric == "silhouette_gower":

        def reward(cols: List[str]) -> float:
            if not cols:
                return -1.0
            df = df_full[cols]
            cat_in = [c for c in cols if c in cat_cols]
            try:
                cat_idx = [df.columns.get_loc(c) for c in cat_in]
                labels  = cluster_fn(df, cat_idx, n_clusters, random_state)
                if len(set(labels)) < 2:
                    return -1.0
            except Exception:
                return -1.0
            try:
                X_num, X_cat, num_min, num_max, mn, mc, inv = \
                    prepare_mixed_arrays_no_label(df)
                score, _, _ = full_silhouette_gower(
                    X_num, X_cat, num_min, num_max, labels,
                    feature_mask_num=mn, feature_mask_cat=mc, inv_rng=inv
                )
            except Exception:
                return -1.0
            if use_redundancy_penalty and redundancy_matrix is not None:
                score = (1 - alpha_penalty) * score + \
                        alpha_penalty * redundancy_penalty(cols, redundancy_matrix)
            return float(score)

        return reward

    # ────────────────────────────────────────────────────────────
    # CABANG 2: lsil (online)
    # ────────────────────────────────────────────────────────────
    elif metric == "lsil":
        X_unit_full, _, _ = build_features(
            df_full, label_col=None, scaler_type="standard", unit_norm=True
        )
        labels_full = cluster_fn(
            df_full,
            [df_full.columns.get_loc(c) for c in df_full.columns if c in cat_cols],
            n_clusters, random_state
        )
        m = adaptive_landmark_count(len(df_full), K=n_clusters,
                                    c=lsil_c, cap_frac=lsil_cap_frac)
        L_fixed = select_landmarks_cluster_aware(
            X_unit_full, labels_full, m,
            central_frac=0.8, boundary_frac=0.2,
            per_cluster_min=3, seed=random_state
        )

        def reward(cols: List[str]) -> float:
            if not cols:
                return -1.0
            df = df_full[cols]
            cat_idx_sub = [df.columns.get_loc(c) for c in cols if c in cat_cols]
            try:
                labels_sub = cluster_fn(df, cat_idx_sub, n_clusters, random_state)
                if len(set(labels_sub)) < 2:
                    return -1.0
            except Exception:
                return -1.0
            X_num, X_cat, num_min, num_max, mn, mc, inv = \
                prepare_mixed_arrays_no_label(df)
            score = lsil_using_landmarks(
                labels_sub, L_fixed, X_num, X_cat, num_min, num_max,
                feature_mask_num=mn, feature_mask_cat=mc, inv_rng=inv,
                agg_mode=lsil_agg_mode, topk=lsil_topk,
            )
            if use_redundancy_penalty and redundancy_matrix is not None:
                score = (1 - alpha_penalty) * score + \
                        alpha_penalty * redundancy_penalty(cols, redundancy_matrix)
            return float(score)

        return reward

    # ────────────────────────────────────────────────────────────
    # CABANG 3: lsil_fixed
    # ────────────────────────────────────────────────────────────
    elif metric == "lsil_fixed":

        # Array full — untuk Phase B cache
        X_num_full, X_cat_full, num_min_full, num_max_full, _, _, inv_rng_full = \
            prepare_mixed_arrays_no_label(df_full)
        num_cols_full = df_full.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols_full = df_full.select_dtypes(
            include=['object', 'category', 'bool']
        ).columns.tolist()
        num_pos = {c: i for i, c in enumerate(num_cols_full)}
        cat_pos = {c: i for i, c in enumerate(cat_cols_full)}

        # [B] Cluster cepat
        t0 = perf_counter()
        labels0, idx_sub_cl, labels_sub_cl = _fast_cluster_subsample(
            df_full, cat_cols_full, cluster_fn, n_clusters,
            random_state, subsample_n_cluster=subsample_n_cluster
        )
        protos0 = {}
        print(f"[reward:lsil_fixed] cluster: K={len(np.unique(labels0))}, "
              f"{perf_counter()-t0:.1f}s")

        # [A] Subsample evaluasi
        eval_n = min(n_full, lsil_eval_n)
        X_num_e, X_cat_e, num_min_e, num_max_e, inv_rng_e, labels_e, idx_eval = \
            _prepare_eval_subsample(df_full, labels0, eval_n, random_state)
        c_rew = lsil_c_reward if lsil_c_reward is not None else lsil_c
        m_eval = adaptive_landmark_count(eval_n, K=n_clusters,
                                         c=c_rew, cap_frac=lsil_cap_frac)
        L_eval = _stratified_landmarks_local(
            labels_e, m_eval, per_cluster_min=3,
            rng=np.random.default_rng(random_state)
        )
        print(f"[reward:lsil_fixed] eval_n={eval_n:,}, |L_eval|={len(L_eval)}")

        # Landmark full untuk Phase B
        m_full = adaptive_landmark_count(n_full, K=n_clusters,
                                         c=lsil_c, cap_frac=lsil_cap_frac)
        L_fixed_full = cluster_aware_landmarks_on_subsample(
            df_full=df_full, idx_sub=idx_sub_cl,
            labels_sub=labels_sub_cl, labels_full=labels0,
            m_cap=m_full, per_cluster_min=3,
            random_state=random_state,
            select_landmarks_fn=select_landmarks_cluster_aware,
        )

        def make_masks_for_subset(cols):
            mnum = np.zeros(X_num_e.shape[1], dtype=bool) \
                if X_num_e.shape[1] else None
            mcat = np.zeros(X_cat_e.shape[1], dtype=bool) \
                if X_cat_e.shape[1] else None
            if mnum is not None:
                idxs = [num_pos[c] for c in cols if c in num_pos]
                if idxs:
                    mnum[np.array(idxs)] = True
            if mcat is not None:
                idxs = [cat_pos[c] for c in cols if c in cat_pos]
                if idxs:
                    mcat[np.array(idxs)] = True
            return mnum, mcat

        def reward(cols: List[str]) -> float:
            if not cols:
                return -1.0
            mask_num, mask_cat = make_masks_for_subset(cols)
            try:
                score = lsil_using_landmarks(
                    labels_e, L_eval,
                    X_num_e, X_cat_e, num_min_e, num_max_e,
                    feature_mask_num=mask_num, feature_mask_cat=mask_cat,
                    inv_rng=inv_rng_e,
                    agg_mode=lsil_agg_mode, topk=lsil_topk,
                )
            except Exception as e:
                print(f"❌ lsil_fixed: {e}")
                return -1.0
            if use_redundancy_penalty and redundancy_matrix is not None:
                score = (1 - alpha_penalty) * score + \
                        alpha_penalty * redundancy_penalty(cols, redundancy_matrix)
            return float(score)

        reward.__phase_a_cache__ = {
            'X_num_full': X_num_full, 'X_cat_full': X_cat_full,
            'num_min_full': num_min_full, 'num_max_full': num_max_full,
            'mask_num_full': None, 'mask_cat_full': None,
            'inv_rng_full': inv_rng_full,
            'num_pos': num_pos, 'cat_pos': cat_pos,
            'L_fixed': L_fixed_full, 'labels0': labels0,
            'protos0': protos0, 'n_samples': n_full,
            'random_state': random_state,
        }
        return reward

    # ────────────────────────────────────────────────────────────
    # CABANG 4: lsil_fixed_calibrated  ← UTAMA Mode C
    # ────────────────────────────────────────────────────────────
    elif metric == "lsil_fixed_calibrated":
        guard_every_eff     = guard_every
        ss_max_n_cal_eff    = ss_max_n_cal
        calibrate_mode_eff  = calibrate_mode
        use_cache           = use_calib_cache

        # Array full — untuk Phase B cache
        X_num_full, X_cat_full, num_min_full, num_max_full, _, _, inv_rng_full = \
            prepare_mixed_arrays_no_label(df_full)
        num_cols_full = df_full.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols_full = df_full.select_dtypes(
            include=['object', 'category', 'bool']
        ).columns.tolist()
        num_pos = {c: i for i, c in enumerate(num_cols_full)}
        cat_pos = {c: i for i, c in enumerate(cat_cols_full)}

        # ── [B] Initial clustering: subsample kecil ──────────────
        t0 = perf_counter()
        labels0, idx_sub_cl, labels_sub_cl = _fast_cluster_subsample(
            df_full, cat_cols_full, cluster_fn, n_clusters,
            random_state, subsample_n_cluster=subsample_n_cluster
        )
        protos0 = {}
        K_unique = len(np.unique(labels0))
        print(f"[reward] cluster: K={K_unique}, {perf_counter()-t0:.1f}s "
              f"(sub={subsample_n_cluster:,})")
        if K_unique < 2:
            print(f"⚠️  [reward] labels0 hanya {K_unique} klaster — semua reward=-1.0")

        # ── [A] Subsample evaluasi reward ────────────────────────
        eval_n = min(n_full, lsil_eval_n)
        X_num_e, X_cat_e, num_min_e, num_max_e, inv_rng_e, labels_e, idx_eval = \
            _prepare_eval_subsample(df_full, labels0, eval_n, random_state)

        c_rew   = lsil_c_reward if lsil_c_reward is not None else min(lsil_c, 2.0)
        m_eval  = adaptive_landmark_count(eval_n, K=n_clusters,
                                          c=c_rew, cap_frac=lsil_cap_frac)
        L_eval  = _stratified_landmarks_local(
            labels_e, m_eval, per_cluster_min=3,
            rng=np.random.default_rng(random_state)
        )

        # Landmark full untuk Phase B — v1.1.12: dua strategi
        m_full = adaptive_landmark_count(n_full, K=n_clusters,
                                         c=lsil_c, cap_frac=lsil_cap_frac)

        if landmark_mode == "kcenter":
            # K-agnostic: k-center greedy — fair untuk semua K di Phase B
            # Tidak bergantung pada labels0/K_hint → tidak ada bias auto-K
            # Trade-off: sedikit lebih rentan BCVD karena tidak ada
            # cluster-aware placement, tapi di-kompensasi oleh topk aggregation
            from ..core.features import build_features as _bf
            try:
                X_unit_lm, _, _ = _bf(
                    df_full, label_col=None,
                    scaler_type="standard", unit_norm=True
                )
                L_fixed_full = select_landmarks_kcenter(
                    X_unit_lm, m=m_full, seed=random_state
                )
                print(f"[reward] landmark_mode=kcenter |L_full|={len(L_fixed_full)} "
                      f"(K-agnostic, fair auto-K)")
            except Exception as e:
                print(f"[reward] kcenter gagal ({e}), fallback ke cluster_aware")
                L_fixed_full = cluster_aware_landmarks_on_subsample(
                    df_full=df_full, idx_sub=idx_sub_cl,
                    labels_sub=labels_sub_cl, labels_full=labels0,
                    m_cap=m_full, per_cluster_min=3,
                    random_state=random_state,
                    select_landmarks_fn=select_landmarks_cluster_aware,
                )
        else:
            # cluster_aware (default, paper JDSA):
            # 80% central + 20% boundary — optimal BCVD mitigation
            # Biased toward K_hint tapi akurat untuk K dekat K_hint
            L_fixed_full = cluster_aware_landmarks_on_subsample(
                df_full=df_full, idx_sub=idx_sub_cl,
                labels_sub=labels_sub_cl, labels_full=labels0,
                m_cap=m_full, per_cluster_min=3,
                random_state=random_state,
                select_landmarks_fn=select_landmarks_cluster_aware,
            )
            print(f"[reward] landmark_mode=cluster_aware |L_full|={len(L_fixed_full)} "
                  f"(JDSA default)")

        def make_masks_for_subset(cols):
            mnum = np.zeros(X_num_e.shape[1], dtype=bool) \
                if X_num_e.shape[1] else None
            mcat = np.zeros(X_cat_e.shape[1], dtype=bool) \
                if X_cat_e.shape[1] else None
            if mnum is not None:
                idxs = [num_pos[c] for c in cols if c in num_pos]
                if idxs:
                    mnum[np.array(idxs)] = True
            if mcat is not None:
                idxs = [cat_pos[c] for c in cols if c in cat_pos]
                if idxs:
                    mcat[np.array(idxs)] = True
            return mnum, mcat

        _calib_cache: Dict        = {}
        lsil_hist_global: List    = []
        ss_hist_global:   List    = []
        call_count                = 0

        def reward(cols):
            nonlocal call_count, _calib_cache, lsil_hist_global, ss_hist_global
            if not cols:
                return -1.0
            if K_unique < 2:
                return -1.0
            call_count += 1
            mask_num, mask_cat = make_masks_for_subset(cols)
            try:
                # [A+C] L-Sil hanya pada eval subsample — FIX BUG + SPEEDUP
                L = lsil_using_landmarks(
                    labels_e, L_eval,
                    X_num_e, X_cat_e, num_min_e, num_max_e,
                    feature_mask_num=mask_num, feature_mask_cat=mask_cat,
                    inv_rng=inv_rng_e,
                    agg_mode=lsil_agg_mode, topk=lsil_topk,
                )
            except Exception as e:
                if call_count <= 3:
                    print(f"❌ lsil_fixed_calibrated (call={call_count}): "
                          f"{type(e).__name__}: {e}")
                return -1.0

            if not np.isfinite(L):
                if call_count <= 3:
                    print(f"⚠️  L-Sil={L} call={call_count}, cols={cols[:3]}...")
                return -1.0

            # Kalibrasi opsional
            do_calib = (
                (calibrate_mode_eff == "always"
                 and (call_count % guard_every_eff) == 0)
                or (calibrate_mode_eff == "on_demand"
                    and call_count > 10
                    and (call_count % guard_every_eff) == 0)
            )

            key = tuple(sorted(cols))
            if use_cache and key in _calib_cache:
                A, B  = _calib_cache[key]
                score = A * float(L) + B
            elif do_calib:
                try:
                    sub_df  = df_full[cols]
                    cat_idx_sub = [
                        sub_df.columns.get_loc(c)
                        for c in sub_df.columns if c in cat_cols_full
                    ]
                    labels_sub_new = cluster_fn(
                        sub_df, cat_idx_sub, n_clusters, random_state
                    )
                    Xn_s, Xc_s, nm_s, nx_s, mn_s, mc_s, inv_s = \
                        prepare_mixed_arrays_no_label(sub_df)
                    S, _, _ = full_silhouette_gower_subsample(
                        Xn_s, Xc_s, nm_s, nx_s, labels_sub_new,
                        max_n=ss_max_n_cal_eff,
                        feature_mask_num=mn_s, feature_mask_cat=mc_s,
                        inv_rng=inv_s
                    )
                    lsil_hist_global.append(float(L))
                    ss_hist_global.append(float(S))
                    if len(lsil_hist_global) >= 5:
                        X_fit = np.vstack([
                            np.array(lsil_hist_global),
                            np.ones(len(lsil_hist_global))
                        ]).T
                        A_, B_ = np.linalg.lstsq(
                            X_fit, np.array(ss_hist_global), rcond=None
                        )[0]
                        A, B = float(A_), float(B_)
                        if use_cache:
                            _calib_cache[key] = (A, B)
                    else:
                        A, B = 1.0, 0.0
                    score = A * float(L) + B
                except Exception:
                    score = float(L)
            else:
                score = float(L)

            if use_redundancy_penalty and redundancy_matrix is not None:
                score = (1 - alpha_penalty) * float(score) + \
                        alpha_penalty * redundancy_penalty(cols, redundancy_matrix)

            return float(score)

        reward.__guard_every__         = guard_every_eff
        reward.__ss_max_n_cal__        = ss_max_n_cal_eff
        reward.__reward_subsample_n__  = min(n_full, reward_subsample_n)
        reward.__calibrate_mode__      = calibrate_mode_eff
        reward.__calib_cache_enabled__ = use_cache

        reward.__phase_a_cache__ = {
            'X_num_full':    X_num_full,
            'X_cat_full':    X_cat_full,
            'num_min_full':  num_min_full,
            'num_max_full':  num_max_full,
            'mask_num_full': None,
            'mask_cat_full': None,
            'inv_rng_full':  inv_rng_full,
            'num_pos':       num_pos,
            'cat_pos':       cat_pos,
            'L_fixed':       L_fixed_full,
            'labels0':       labels0,
            'protos0':       protos0,
            'n_samples':     n_full,
            'random_state':  random_state,
        }

        return reward

    else:
        raise ValueError(f"Unknown reward metric: {metric!r}")
