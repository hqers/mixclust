# mixclust/aufs/reward.py
#
# CHANGELOG v2.1 (Fix pasca refactor lsil_using_landmarks):
#
#   BUG FIX KRITIS:
#     Setelah lsil_using_prototypes_gower direfactor menjadi wrapper
#     lsil_using_landmarks, signature positional args berubah:
#
#     LAMA (prototype-based):
#       lsil_using_prototypes_gower(labels, protos, landmark_idx, X_num, ...)
#
#     BARU (landmark-based):
#       lsil_using_prototypes_gower(labels, landmark_idx, X_num, X_cat, ...)
#
#     Akibatnya pemanggilan di cabang lsil_fixed dan lsil_fixed_calibrated
#     mengirim protos0 ke posisi X_num → TypeError → exception → return -1.0
#     → SA reward selalu -1.0.
#
#   FIX:
#     1. Hapus protos0 dari pemanggilan lsil_using_prototypes_gower
#        di cabang lsil_fixed dan lsil_fixed_calibrated.
#     2. Ganti dengan lsil_using_landmarks secara langsung (lebih bersih).
#     3. Hapus build_prototypes_by_cluster_gower dari cabang
#        lsil_fixed_calibrated (tidak diperlukan lagi — protos0 tidak dipakai
#        oleh lsil_using_landmarks).
#     4. Tetap simpan protos0 di phase_a_cache untuk kompatibilitas Phase B
#        (controller.py masih mungkin memakainya sebagai fallback).
#
#   EFEK SAMPING POSITIF:
#     - Build reward lebih cepat karena skip build_prototypes untuk n besar
#     - Tidak ada perubahan API eksternal
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations
from typing import Callable, List, Optional, Dict, Tuple
import numpy as np, pandas as pd
from time import perf_counter

from ..core.prototypes import build_prototypes_by_cluster_gower
from ..metrics.lsil import lsil_using_landmarks          # ← import langsung
from ..metrics.lsil import lsil_using_prototypes_gower   # ← tetap untuk backward-compat
from ..metrics.silhouette import full_silhouette_gower_subsample
from ..core.adaptive import adaptive_landmark_count
from ..core.landmarks import (
    subsample_and_propagate_labels,
    select_landmarks_cluster_aware,
    cluster_aware_landmarks_on_subsample
)
from ..core.features import build_features
from ..core.preprocess import prepare_mixed_arrays_no_label
from ..metrics.silhouette import full_silhouette_gower

try:
    from .redundancy import redundancy_penalty
except Exception:
    def redundancy_penalty(cols, red_mat):
        return 0.0


def _stratified_landmarks(
    y: np.ndarray, m_target: int, per_cluster_min: int,
    rng: np.random.Generator
) -> np.ndarray:
    n = len(y)
    if m_target <= 0 or m_target >= n:
        return np.arange(n, dtype=int)
    idxs = np.arange(n, dtype=int)
    L = []
    vals, counts = np.unique(y, return_counts=True)
    q = {
        c: max(per_cluster_min, int(round(m_target * (counts[i] / n))))
        for i, c in enumerate(vals)
    }
    for c in vals:
        pool = idxs[y == c]
        take = min(len(pool), q[c])
        if take > 0:
            L.extend(rng.choice(pool, size=take, replace=False).tolist())
    if len(L) == 0:
        return np.arange(n, dtype=int)
    if len(L) > m_target:
        L = L[:m_target]
    return np.array(sorted(L), dtype=int)


# ================================================================
# make_sa_reward — fungsi utama
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
):
    # ────────────────────────────────────────────────────────────
    # CABANG 1: silhouette_gower
    # ────────────────────────────────────────────────────────────
    if metric == "silhouette_gower":

        def reward(cols: List[str]) -> float:
            if not cols:
                return -1.0
            df = df_full[cols]
            cat_cols_in_subset = [c for c in cols if c in cat_cols]
            try:
                cat_idx = [df.columns.get_loc(c) for c in cat_cols_in_subset]
                labels = cluster_fn(df, cat_idx, n_clusters, random_state)
                if len(set(labels)) < 2:
                    return -1.0
            except Exception:
                return -1.0
            try:
                X_num, X_cat, num_min, num_max, mask_num, mask_cat, inv_rng = \
                    prepare_mixed_arrays_no_label(df)
                score, _, _ = full_silhouette_gower(
                    X_num, X_cat, num_min, num_max, labels,
                    feature_mask_num=mask_num,
                    feature_mask_cat=mask_cat,
                    inv_rng=inv_rng
                )
            except Exception:
                return -1.0
            if use_redundancy_penalty and redundancy_matrix is not None:
                red_score = redundancy_penalty(cols, redundancy_matrix)
                score = (1 - alpha_penalty) * score + alpha_penalty * red_score
            return float(score)

        return reward

    # ────────────────────────────────────────────────────────────
    # CABANG 2: lsil (online — landmark diperbarui tiap iterasi)
    # ────────────────────────────────────────────────────────────
    elif metric == "lsil":
        from ..core.features import build_features, prepare_mixed_arrays

        X_unit_full, _, _ = build_features(
            df_full, label_col=None, scaler_type="standard", unit_norm=True
        )
        labels_full = cluster_fn(
            df_full,
            [df_full.columns.get_loc(c) for c in df_full.columns if c in cat_cols],
            n_clusters, random_state
        )
        m = adaptive_landmark_count(len(df_full), K=n_clusters, c=lsil_c,
                                    cap_frac=lsil_cap_frac)
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
            X_num, X_cat, num_min, num_max, mask_num, mask_cat, inv_rng = \
                prepare_mixed_arrays_no_label(df)
            # ── FIX: gunakan lsil_using_landmarks langsung ──
            score = lsil_using_landmarks(
                labels_sub, L_fixed,
                X_num, X_cat, num_min, num_max,
                feature_mask_num=mask_num, feature_mask_cat=mask_cat,
                inv_rng=inv_rng,
                agg_mode=lsil_agg_mode, topk=lsil_topk,
            )
            if use_redundancy_penalty and redundancy_matrix is not None:
                red_score = redundancy_penalty(cols, redundancy_matrix)
                score = (1 - alpha_penalty) * score + alpha_penalty * red_score
            return float(score)

        return reward

    # ────────────────────────────────────────────────────────────
    # CABANG 3: lsil_fixed
    # ────────────────────────────────────────────────────────────
    elif metric == "lsil_fixed":

        X_num_full, X_cat_full, num_min_full, num_max_full, _, _, inv_rng_full = \
            prepare_mixed_arrays_no_label(df_full)

        num_cols_full = df_full.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols_full = df_full.select_dtypes(
            include=['object', 'category', 'bool']
        ).columns.tolist()
        num_pos = {c: i for i, c in enumerate(num_cols_full)}
        cat_pos = {c: i for i, c in enumerate(cat_cols_full)}

        reward_subsample_n_eff = min(len(df_full), 20000)
        use_cache = True

        labels0, protos0, idx_sub, labels_sub = subsample_and_propagate_labels(
            df_full=df_full,
            cat_cols_full=cat_cols_full,
            cluster_fn=cluster_fn,
            n_clusters=n_clusters,
            random_state=random_state,
            subsample_n=reward_subsample_n_eff,
            proto_sample_cap=None,
            per_cluster_proto=1
        )

        m = adaptive_landmark_count(len(df_full), K=n_clusters, c=lsil_c,
                                    cap_frac=lsil_cap_frac)
        L_fixed = cluster_aware_landmarks_on_subsample(
            df_full=df_full,
            idx_sub=idx_sub,
            labels_sub=labels_sub,
            labels_full=labels0,
            m_cap=m,
            per_cluster_min=3,
            random_state=random_state,
            select_landmarks_fn=select_landmarks_cluster_aware
            if 'select_landmarks_cluster_aware' in globals() else None
        )

        def make_masks_for_subset(cols):
            if X_num_full.shape[1]:
                mnum = np.zeros(X_num_full.shape[1], dtype=bool)
                for c in cols:
                    if c in num_pos:
                        mnum[num_pos[c]] = True
            else:
                mnum = None
            if X_cat_full.shape[1]:
                mcat = np.zeros(X_cat_full.shape[1], dtype=bool)
                for c in cols:
                    if c in cat_pos:
                        mcat[cat_pos[c]] = True
            else:
                mcat = None
            return mnum, mcat

        def reward(cols: List[str]) -> float:
            if not cols:
                return -1.0
            mask_num, mask_cat = make_masks_for_subset(cols)
            try:
                # ── FIX: hapus protos0, pakai lsil_using_landmarks langsung ──
                score = lsil_using_landmarks(
                    labels0, L_fixed,
                    X_num_full, X_cat_full, num_min_full, num_max_full,
                    feature_mask_num=mask_num, feature_mask_cat=mask_cat,
                    inv_rng=inv_rng_full,
                    agg_mode=lsil_agg_mode, topk=lsil_topk,
                )
            except Exception as e:
                print(f"❌ lsil_fixed gagal utk {cols} → {e}")
                return -1.0
            if use_redundancy_penalty and redundancy_matrix is not None:
                red_score = redundancy_penalty(cols, redundancy_matrix)
                score = (1 - alpha_penalty) * score + alpha_penalty * red_score
            return float(score)

        reward.__phase_a_cache__ = {
            'X_num_full': X_num_full,
            'X_cat_full': X_cat_full,
            'num_min_full': num_min_full,
            'num_max_full': num_max_full,
            'mask_num_full': None,
            'mask_cat_full': None,
            'inv_rng_full': inv_rng_full,
            'num_pos': num_pos,
            'cat_pos': cat_pos,
            'L_fixed': L_fixed,
            'labels0': labels0,
            'protos0': protos0,   # tetap disimpan untuk Phase B fallback
            'n_samples': len(df_full),
        }

        return reward

    # ────────────────────────────────────────────────────────────
    # CABANG 4: lsil_fixed_calibrated  ← UTAMA untuk Mode C
    # ────────────────────────────────────────────────────────────
    elif metric == "lsil_fixed_calibrated":
        guard_every_eff = guard_every
        ss_max_n_cal_eff = ss_max_n_cal
        reward_subsample_n_eff = min(len(df_full), reward_subsample_n)
        calibrate_mode_eff = calibrate_mode
        use_cache = use_calib_cache

        X_num_full, X_cat_full, num_min_full, num_max_full, _, _, inv_rng_full = \
            prepare_mixed_arrays_no_label(df_full)

        num_cols_full = df_full.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols_full = df_full.select_dtypes(
            include=['object', 'category', 'bool']
        ).columns.tolist()
        num_pos = {c: i for i, c in enumerate(num_cols_full)}
        cat_pos = {c: i for i, c in enumerate(cat_cols_full)}

        # ── FIX: subsample_and_propagate_labels tetap dipanggil untuk
        #    labels0 dan idx_sub (dibutuhkan oleh cluster_aware_landmarks).
        #    protos0 sudah tidak dipakai di reward, tapi disimpan di cache
        #    untuk Phase B fallback.
        labels0, protos0, idx_sub, labels_sub = subsample_and_propagate_labels(
            df_full=df_full,
            cat_cols_full=cat_cols_full,
            cluster_fn=cluster_fn,
            n_clusters=n_clusters,
            random_state=random_state,
            subsample_n=reward_subsample_n,
            proto_sample_cap=None,
            per_cluster_proto=1
        )

        m = adaptive_landmark_count(len(df_full), K=n_clusters, c=lsil_c,
                                    cap_frac=lsil_cap_frac)
        L_fixed = cluster_aware_landmarks_on_subsample(
            df_full=df_full,
            idx_sub=idx_sub,
            labels_sub=labels_sub,
            labels_full=labels0,
            m_cap=m,
            per_cluster_min=3,
            random_state=random_state,
            select_landmarks_fn=select_landmarks_cluster_aware
            if 'select_landmarks_cluster_aware' in globals() else None
        )

        def make_masks_for_subset(cols):
            mnum = np.zeros(X_num_full.shape[1], dtype=bool) \
                if X_num_full.shape[1] else None
            mcat = np.zeros(X_cat_full.shape[1], dtype=bool) \
                if X_cat_full.shape[1] else None
            if mnum is not None:
                idxs = [num_pos[c] for c in cols if c in num_pos]
                if idxs:
                    mnum[np.array(idxs)] = True
            if mcat is not None:
                idxs = [cat_pos[c] for c in cols if c in cat_pos]
                if idxs:
                    mcat[np.array(idxs)] = True
            return mnum, mcat

        _calib_cache: Dict = {}
        lsil_hist_global: List = []
        ss_hist_global: List = []
        call_count = 0

        def reward(cols):
            nonlocal call_count, _calib_cache, lsil_hist_global, ss_hist_global
            if not cols:
                return -1.0
            call_count += 1
            mask_num, mask_cat = make_masks_for_subset(cols)
            try:
                # ── FIX KRITIS: hapus protos0, pakai lsil_using_landmarks
                #    langsung dengan signature yang benar ──
                L = lsil_using_landmarks(
                    labels0, L_fixed,
                    X_num_full, X_cat_full, num_min_full, num_max_full,
                    feature_mask_num=mask_num, feature_mask_cat=mask_cat,
                    inv_rng=inv_rng_full,
                    agg_mode=lsil_agg_mode, topk=lsil_topk,
                )
            except Exception as e:
                # Debug: print error pertama kali saja
                if call_count <= 3:
                    print(f"❌ lsil_fixed_calibrated gagal (call={call_count}): {e}")
                return -1.0

            if np.isnan(L) or not np.isfinite(L):
                return -1.0

            if (calibrate_mode_eff == "always" and (call_count % guard_every_eff) == 0) or \
               (calibrate_mode_eff == "on_demand" and call_count > 10
                    and (call_count % guard_every_eff) == 0):
                do_calib = True
            elif calibrate_mode_eff == "topk":
                do_calib = False
            else:
                do_calib = False

            key = tuple(sorted(cols))
            if use_cache and key in _calib_cache:
                A, B = _calib_cache[key]
                score = A * float(L) + B
            elif do_calib:
                try:
                    sub_df = df_full[cols]
                    cat_idx_sub = [
                        sub_df.columns.get_loc(c)
                        for c in sub_df.columns if c in cat_cols_full
                    ]
                    labels_sub_new = cluster_fn(sub_df, cat_idx_sub, n_clusters, random_state)
                    Xn_sub, Xc_sub, nmin_sub, nmax_sub, mnum_sub, mcat_sub, inv_sub = \
                        prepare_mixed_arrays_no_label(sub_df)
                    S, _, _ = full_silhouette_gower_subsample(
                        Xn_sub, Xc_sub, nmin_sub, nmax_sub, labels_sub_new,
                        max_n=ss_max_n_cal_eff,
                        feature_mask_num=mnum_sub, feature_mask_cat=mcat_sub,
                        inv_rng=inv_sub
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
                red_score = redundancy_penalty(cols, redundancy_matrix)
                score = (1 - alpha_penalty) * float(score) + alpha_penalty * red_score

            return float(score)

        reward.__guard_every__ = guard_every_eff
        reward.__ss_max_n_cal__ = ss_max_n_cal_eff
        reward.__reward_subsample_n__ = reward_subsample_n_eff
        reward.__calibrate_mode__ = calibrate_mode_eff
        reward.__calib_cache_enabled__ = use_cache

        reward.__phase_a_cache__ = {
            'X_num_full': X_num_full,
            'X_cat_full': X_cat_full,
            'num_min_full': num_min_full,
            'num_max_full': num_max_full,
            'mask_num_full': None,
            'mask_cat_full': None,
            'inv_rng_full': inv_rng_full,
            'num_pos': num_pos,
            'cat_pos': cat_pos,
            'L_fixed': L_fixed,
            'labels0': labels0,
            'protos0': protos0,   # tetap disimpan untuk Phase B fallback
            'n_samples': len(df_full),
        }

        return reward

    else:
        raise ValueError(f"Unknown reward metric: {metric}")
