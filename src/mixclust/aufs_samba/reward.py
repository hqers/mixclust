# src/mixclust/aufs_samba/reward.py
from __future__ import annotations
from typing import Callable, List, Optional, Dict, Tuple
import numpy as np, pandas as pd
from time import perf_counter

# ---- dependensi dari paket utama ----
from mixclust.prototypes import build_prototypes_by_cluster_gower
from mixclust.metrics.lsil import lsil_using_prototypes_gower
from mixclust.silhouette import full_silhouette_gower_subsample
from mixclust.landmarks import (
    subsample_and_propagate_labels,
    select_landmarks_cluster_aware,
    cluster_aware_landmarks_on_subsample
)
from mixclust.features import build_features
from mixclust.aufs_samba.preprocess import prepare_mixed_arrays_no_label

# FIX: import full_silhouette_gower di level modul (bukan di dalam cabang if)
from mixclust.silhouette import full_silhouette_gower


try:
    from mixclust.aufs_samba.redundancy import redundancy_penalty
except Exception:
    def redundancy_penalty(cols, red_mat):  # noop fallback
        return 0.0


# ---- helper: siapkan komponen Gower (fallback jika modul preprocess tidak tersedia) ----

        
def _stratified_landmarks(y: np.ndarray, m_target: int, per_cluster_min: int, rng: np.random.Generator) -> np.ndarray:
    n = len(y)
    if m_target <= 0 or m_target >= n:
        return np.arange(n, dtype=int)
    idxs = np.arange(n, dtype=int)
    L = []
    vals, counts = np.unique(y, return_counts=True)
    q = {c: max(per_cluster_min, int(round(m_target * (counts[i] / n)))) for i, c in enumerate(vals)}
    for c in vals:
        pool = idxs[y == c]
        take = min(len(pool), q[c])
        if take > 0:
            L.extend(rng.choice(pool, size=take, replace=False).tolist())
    if len(L) == 0:
        return np.arange(n, dtype=int)
    # rapikan ukuran pas:
    if len(L) > m_target: L = L[:m_target]
    return np.array(sorted(L), dtype=int)
    
# ---- fungsi utama: pembuat reward untuk SA ----
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
    per_cluster_proto_if_many: int = 1,
    lsil_proto_sample_cap: int = 200,
    lsil_agg_mode="mean",
    lsil_topk=5,
    random_state=42,
    dynamic_k: bool = False 
):
    if metric == "silhouette_gower":
        # FIX: hapus re-import lokal di sini — sudah diimport di level modul atas
        # (re-import lokal menyebabkan Python menganggap prepare_mixed_arrays_no_label
        #  sebagai local variable di seluruh fungsi make_sa_reward, sehingga
        #  cabang lain (lsil_fixed_calibrated dll) yang memanggilnya sebelum
        #  blok ini dieksekusi akan raise UnboundLocalError)

        def reward(cols: List[str]) -> float:
            if not cols:
                print("⚠️ Empty subset, skipping.")
                return -1.0
        
            df = df_full[cols]
            cat_cols_in_subset = [c for c in cols if c in cat_cols]
        
            try:
                cat_idx = [df.columns.get_loc(c) for c in cat_cols_in_subset]
                labels = cluster_fn(df, cat_idx, n_clusters, random_state)

                if len(set(labels)) < 2:
                    print(f"⚠️ Hanya 1 cluster untuk subset: {cols}")
                    return -1.0
            except Exception as e:
                print(f"❌ cluster_fn gagal untuk {cols} → {e}")
                return -1.0
        
            try:
                X_num, X_cat, num_min, num_max, mask_num, mask_cat, inv_rng = prepare_mixed_arrays_no_label(df)
                score, _, _ = full_silhouette_gower(
                    X_num, X_cat, num_min, num_max, labels,
                    feature_mask_num=mask_num,
                    feature_mask_cat=mask_cat,
                    inv_rng=inv_rng
                )
            except Exception as e:
                print(f"❌ silhouette gagal untuk {cols} → {e}")
                return -1.0


            if use_redundancy_penalty and redundancy_matrix is not None:
                red_score = redundancy_penalty(cols, redundancy_matrix)
                score = (1 - alpha_penalty) * score + alpha_penalty * red_score

            return float(score)

        return reward



    elif metric == "lsil":
        # FIX: hapus semua re-import lokal di sini — sudah diimport di level modul atas
        from mixclust.features import build_features, prepare_mixed_arrays  # prepare_mixed_arrays mungkin belum diimport di atas

        # --- di make_sa_reward (metric == "lsil") ---
        # (1) Siapkan LANDMARK sekali dari representasi full (hemat biaya)
        X_unit_full, _, _ = build_features(df_full, label_col=None, scaler_type="standard", unit_norm=True)
        labels_full = cluster_fn(df_full,  # opsional: hanya utk landmark 'cluster_aware'
                                 [df_full.columns.get_loc(c) for c in [c for c in df_full.columns if c in cat_cols]],
                                 n_clusters, random_state)
        m = min(int(0.1 * len(df_full)), 300)
        L_fixed = select_landmarks_cluster_aware(
            X_unit_full, labels_full, m,
            central_frac=0.8, boundary_frac=0.2,
            per_cluster_min=3, seed=random_state
        )
        
        def reward(cols: List[str]) -> float:
            if not cols:
                return -1.0
            df = df_full[cols]
            # >>> RECLUSTER per-subset <<< (konsisten dg cabang silhouette_gower)
            cat_idx_sub = [df.columns.get_loc(c) for c in cols if c in cat_cols]
            try:
                labels_sub = cluster_fn(df, cat_idx_sub, n_clusters, random_state)
                if len(set(labels_sub)) < 2:
                    return -1.0
            except Exception:
                return -1.0
        
            # komponen Gower untuk subset
            X_num, X_cat, num_min, num_max, mask_num, mask_cat, inv_rng = prepare_mixed_arrays_no_label(df)
        
            # prototipe berbasis subset + labels_sub
            protos = build_prototypes_by_cluster_gower(
                labels_sub, X_num, X_cat, num_min, num_max,
                per_cluster=per_cluster_proto_if_many,
                sample_cap=300, seed=random_state,
                feature_mask_num=mask_num, feature_mask_cat=mask_cat, inv_rng=inv_rng
            )
        
            # L-Sil pada LANDMARK FIXED (hemat biaya, indeks baris tetap berlaku)
            score = lsil_using_prototypes_gower(
                labels_sub, L_fixed, protos,
                X_num, X_cat, num_min, num_max,
                feature_mask_num=mask_num, feature_mask_cat=mask_cat,
                inv_rng=inv_rng,
                agg_mode=lsil_agg_mode, topk=lsil_topk
            )
        
            if use_redundancy_penalty and redundancy_matrix is not None:
                red_score = redundancy_penalty(cols, redundancy_matrix)
                score = (1 - alpha_penalty) * score + alpha_penalty * red_score
            return float(score)


        return reward

    # di make_sa_reward(...):
    elif metric == "lsil_fixed":
        # FIX: hapus re-import lokal di sini — sudah diimport di level modul atas
    
        # 0) PRECOMPUTE sekali untuk FULL DF  ------------------------------------
        # 0a) arrays Gower utk FULL DF
        X_num_full, X_cat_full, num_min_full, num_max_full, _, _, inv_rng_full = \
            prepare_mixed_arrays_no_label(df_full)
    
        # 0b) peta indeks kolom → posisi fitur numerik/kategorik di arrays full
        #     agar nanti subset → mask boolean (sangat murah)
        num_cols_full = df_full.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols_full = df_full.select_dtypes(include=['object','category','bool']).columns.tolist()

    
        num_pos = {c: i for i, c in enumerate(num_cols_full)}
        cat_pos = {c: i for i, c in enumerate(cat_cols_full)}
    
        # 0c) label awal (sekali) di FULL DF
        cat_idx_full = [df_full.columns.get_loc(c) for c in cat_cols_full]
        #labels0 = cluster_fn(df_full, cat_idx_full, n_clusters, random_state)
        # (baru) subsample + propagate + cluster-aware landmarks on subsample
        cat_cols_full = df_full.select_dtypes(include=['object','category','bool']).columns.tolist()
        # --- Defaults untuk subsample / kalibrasi (bisa dipindahkan nanti ke AUFSParams) ---
        # reward_subsample_n: jumlah baris untuk subsample awal (cap). kalau kecil dataset pakai n.
        reward_subsample_n = min(len(df_full), 20000)
        
        # lsil_m_cap: upper cap untuk jumlah landmark (m)
        lsil_m_cap = 300
        
        # Kalibrasi: frekuensi/strategi default
        guard_every = 50           # hitung SS subsample tiap 50 panggilan (kalibrasi opsional)
        ss_max_n_cal = 400         # ukuran subsample SS saat kalibrasi
        calibrate_mode = "topk"    # "always" | "on_demand" | "topk"
        calibrate_after_iter = 10  # mulai kalibrasi setelah iterasi ini (untuk on_demand)
        
        # caching dan kontrol
        use_cache = True           # cache koef A,B per-subset bila kalibrasi dipakai

        labels0, protos0, idx_sub, labels_sub = subsample_and_propagate_labels(
            df_full=df_full,
            cat_cols_full=cat_cols_full,
            cluster_fn=cluster_fn,
            n_clusters=n_clusters,
            random_state=random_state,
            subsample_n=reward_subsample_n,         # pass from params or hardcode 20000
            proto_sample_cap=lsil_proto_sample_cap,
            per_cluster_proto=per_cluster_proto_if_many
        )
        
        m = min(int(0.1 * len(df_full)), lsil_m_cap)   # lsil_m_cap from params
        L_fixed = cluster_aware_landmarks_on_subsample(
            df_full=df_full,
            idx_sub=idx_sub,
            labels_sub=labels_sub,
            labels_full=labels0,
            m_cap=m,
            per_cluster_min=3,
            random_state=random_state,
            select_landmarks_fn=select_landmarks_cluster_aware if 'select_landmarks_cluster_aware' in globals() else None
        )


        
        # 0d) X_unit utk landmark & landmark cluster-aware (sekali)
        X_unit_full, _, _ = build_features(
            df_full, label_col=None, scaler_type="standard", unit_norm=True
        )
    
        # 0e) Prototipe (sekali) di FULL DF
       # protos0 = build_prototypes_by_cluster_gower(
       #     labels0, X_num_full, X_cat_full, num_min_full, num_max_full,
        #    per_cluster=per_cluster_proto_if_many, sample_cap=lsil_proto_sample_cap,
       #    seed=random_state, feature_mask_num=None, feature_mask_cat=None, inv_rng=inv_rng_full
       # )
    
        # util: buat mask fitur utk subset kolom (sangat cepat)
        def make_masks_for_subset(cols):
            # boolean mask untuk dimensi X_num_full dan X_cat_full
            if X_num_full.shape[1]:
                mnum = np.zeros(X_num_full.shape[1], dtype=bool)
                for c in cols:
                    if c in num_pos: mnum[num_pos[c]] = True
            else:
                mnum = None
    
            if X_cat_full.shape[1]:
                mcat = np.zeros(X_cat_full.shape[1], dtype=bool)
                for c in cols:
                    if c in cat_pos: mcat[cat_pos[c]] = True
            else:
                mcat = None
            return mnum, mcat
    
        def reward(cols: List[str]) -> float:
            if not cols:
                return -1.0
    
            # NB: TIDAK re-cluster, TIDAK re-landmark, TIDAK re-prototype
            mask_num, mask_cat = make_masks_for_subset(cols)
    
            try:
                score = lsil_using_prototypes_gower(
                    labels0, L_fixed, protos0,
                    X_num_full, X_cat_full, num_min_full, num_max_full,
                    feature_mask_num=mask_num, feature_mask_cat=mask_cat,
                    inv_rng=inv_rng_full,
                    agg_mode=lsil_agg_mode, topk=lsil_topk
                )
            except Exception as e:
                print(f"❌ lsil_fixed gagal utk {cols} → {e}")
                return -1.0
    
            if use_redundancy_penalty and redundancy_matrix is not None:
                red_score = redundancy_penalty(cols, redundancy_matrix)
                score = (1 - alpha_penalty) * score + alpha_penalty * red_score
            return float(score)
    
        return reward

    elif metric == "lsil_fixed_calibrated":
        guard_every = 50
        ss_max_n_cal = 400
        reward_subsample_n = 20000
        calibrate_mode = "topk"
        calibrate_after_iter = 10
        use_cache = True

        # --- PRECOMPUTE once (cheap) using subsample & propagate ---
        # FIX: hapus re-import lokal di sini — sudah diimport di level modul atas
        X_num_full, X_cat_full, num_min_full, num_max_full, _, _, inv_rng_full = \
            prepare_mixed_arrays_no_label(df_full)

        num_cols_full = df_full.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols_full = df_full.select_dtypes(include=['object','category','bool']).columns.tolist()
        num_pos = {c: i for i, c in enumerate(num_cols_full)}
        cat_pos = {c: i for i, c in enumerate(cat_cols_full)}

        # use subsample_and_propagate_labels to get initial labels0 and protos0 cheaply
        labels0, protos0, idx_sub, labels_sub = subsample_and_propagate_labels(
            df_full=df_full,
            cat_cols_full=cat_cols_full,
            cluster_fn=cluster_fn,
            n_clusters=n_clusters,
            random_state=random_state,
            subsample_n=reward_subsample_n,
            proto_sample_cap=lsil_proto_sample_cap,
            per_cluster_proto=per_cluster_proto_if_many
        )

        lsil_m_cap = 300
        m = min(int(0.1 * len(df_full)), lsil_m_cap)
        L_fixed = cluster_aware_landmarks_on_subsample(
            df_full=df_full,
            idx_sub=idx_sub,
            labels_sub=labels_sub,
            labels_full=labels0,
            m_cap=m,
            per_cluster_min=3,
            random_state=random_state,
            select_landmarks_fn=select_landmarks_cluster_aware if 'select_landmarks_cluster_aware' in globals() else None
        )

        # protos0 already built by subsample_and_propagate_labels (if implemented). Fallback:
        if protos0 is None:
            protos0 = build_prototypes_by_cluster_gower(
                labels0, X_num_full, X_cat_full, num_min_full, num_max_full,
                per_cluster=per_cluster_proto_if_many, sample_cap=lsil_proto_sample_cap,
                seed=random_state, feature_mask_num=None, feature_mask_cat=None, inv_rng=inv_rng_full
            )

        # masks util (vectorized)
        def make_masks_for_subset(cols):
            mnum = np.zeros(X_num_full.shape[1], dtype=bool) if X_num_full.shape[1] else None
            mcat = np.zeros(X_cat_full.shape[1], dtype=bool) if X_cat_full.shape[1] else None
            if mnum is not None:
                idxs = [num_pos[c] for c in cols if c in num_pos]
                if idxs: mnum[np.array(idxs)] = True
            if mcat is not None:
                idxs = [cat_pos[c] for c in cols if c in cat_pos]
                if idxs: mcat[np.array(idxs)] = True
            return mnum, mcat

        # calibration cache: key -> (A,B, last_updated_iter)
        _calib_cache = {}
        lsil_hist_global = []
        ss_hist_global = []
        call_count = 0

        def reward(cols):
            nonlocal call_count, _calib_cache, lsil_hist_global, ss_hist_global

            if not cols:
                return -1.0

            call_count += 1
            mask_num, mask_cat = make_masks_for_subset(cols)

            # 1) Fast L-Sil (fixed) evaluation using precomputed labels0/protos0 & masked features
            try:
                L = lsil_using_prototypes_gower(
                    labels0, L_fixed, protos0,
                    X_num_full, X_cat_full, num_min_full, num_max_full,
                    feature_mask_num=mask_num, feature_mask_cat=mask_cat,
                    inv_rng=inv_rng_full, agg_mode=lsil_agg_mode, topk=lsil_topk
                )
            except Exception as e:
                return -1.0

            if (calibrate_mode == "always" and (call_count % guard_every) == 0) or (calibrate_mode == "on_demand" and call_count > calibrate_after_iter and (call_count % guard_every) == 0):
                do_calib = True
            elif calibrate_mode == "topk":
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
                    cat_idx_sub = [sub_df.columns.get_loc(c) for c in sub_df.columns if c in cat_cols_full]
                    labels_sub_new = cluster_fn(sub_df, cat_idx_sub, n_clusters, random_state)

                    Xn_sub, Xc_sub, nmin_sub, nmax_sub, mnum_sub, mcat_sub, inv_sub = prepare_mixed_arrays_no_label(sub_df)
                    S, _, _ = full_silhouette_gower_subsample(
                        Xn_sub, Xc_sub, nmin_sub, nmax_sub, labels_sub_new, max_n=ss_max_n_cal,
                        feature_mask_num=mnum_sub, feature_mask_cat=mcat_sub, inv_rng=inv_sub
                    )
                    lsil_hist_global.append(float(L)); ss_hist_global.append(float(S))
                    if len(lsil_hist_global) >= 5:
                        X = np.vstack([np.array(lsil_hist_global), np.ones(len(lsil_hist_global))]).T
                        A_, B_ = np.linalg.lstsq(X, np.array(ss_hist_global), rcond=None)[0]
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

        reward.__guard_every__ = guard_every
        reward.__ss_max_n_cal__ = ss_max_n_cal
        reward.__reward_subsample_n__ = reward_subsample_n
        reward.__calibrate_mode__ = calibrate_mode
        reward.__calibrate_after_iter__ = calibrate_after_iter
        reward.__calib_cache_enabled__ = use_cache

        return reward
    
    else:
        raise ValueError(f"Unknown reward metric: {metric}")
