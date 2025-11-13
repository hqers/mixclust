# src/mixclust/aufs_samba/reward.py
from __future__ import annotations
from typing import Callable, List, Optional, Dict, Tuple
import numpy as np, pandas as pd
from time import perf_counter

# ---- dependensi dari paket utama ----
from mixclust.prototypes import build_prototypes_by_cluster_gower
from mixclust.metrics.lsil import lsil_using_prototypes_gower
from mixclust.silhouette import full_silhouette_gower_subsample

try:
    from mixclust.aufs_samba.redundancy import redundancy_penalty
except Exception:
    def redundancy_penalty(cols, red_mat):  # noop fallback
        return 0.0


# ---- helper: siapkan komponen Gower (fallback jika modul preprocess tidak tersedia) ----
try:
    # kalau kamu sudah punya fungsi ini di modul preprocess, pakai saja
    from mixclust.preprocess import prepare_mixed_arrays_no_label  # prefer di root
except Exception:
    try:
        from .preprocess import prepare_mixed_arrays_no_label       # atau di submodul aufs_samba
    except Exception:
        # fallback minimal agar file ini tetap mandiri
        def prepare_mixed_arrays_no_label(df: pd.DataFrame):
            df2 = df.copy()
            cat_cols = df2.select_dtypes(include=['object','category','bool']).columns.tolist()
            num_cols = df2.select_dtypes(include=[np.number]).columns.tolist()

            if len(num_cols) > 0:
                X_num = df2[num_cols].values.astype(np.float32)
                vmin = X_num.min(axis=0).astype(np.float32)
                vmax = X_num.max(axis=0).astype(np.float32)
                inv  = (1.0 / np.maximum(vmax - vmin, 1e-9)).astype(np.float32)
                mnum = np.ones(X_num.shape[1], dtype=bool)
            else:
                X_num = np.zeros((len(df2), 0), dtype=np.float32)
                vmin = np.zeros((0,), dtype=np.float32)
                vmax = np.ones((0,), dtype=np.float32)
                inv  = np.ones((0,), dtype=np.float32)
                mnum = None

            if len(cat_cols) > 0:
                X_cat_list = []
                for c in cat_cols:
                    vals, _ = pd.factorize(df2[c].astype(str), sort=True)
                    X_cat_list.append(vals.astype(np.int32))
                X_cat = np.vstack(X_cat_list).T
                mcat  = np.ones(X_cat.shape[1], dtype=bool)
            else:
                X_cat = np.zeros((len(df2), 0), dtype=np.int32)
                mcat  = None

            return X_num, X_cat, vmin, vmax, mnum, mcat, inv
        
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
        from mixclust.silhouette import full_silhouette_gower
        from mixclust.aufs_samba.preprocess import prepare_mixed_arrays_no_label

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
        from mixclust.features import build_features, prepare_mixed_arrays
        from mixclust.landmarks import select_landmarks_cluster_aware
        from mixclust.metrics.lsil import lsil_using_prototypes_gower
        from mixclust.prototypes import build_prototypes_by_cluster_gower
        from mixclust.aufs_samba.preprocess import prepare_mixed_arrays_no_label

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
        from mixclust.features import build_features
        from mixclust.landmarks import select_landmarks_cluster_aware
        from mixclust.metrics.lsil import lsil_using_prototypes_gower
        from mixclust.prototypes import build_prototypes_by_cluster_gower
        from mixclust.aufs_samba.preprocess import prepare_mixed_arrays_no_label
    
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
        labels0 = cluster_fn(df_full, cat_idx_full, n_clusters, random_state)
    
        # 0d) X_unit utk landmark & landmark cluster-aware (sekali)
        X_unit_full, _, _ = build_features(
            df_full, label_col=None, scaler_type="standard", unit_norm=True
        )
        m = min(int(0.1 * len(df_full)), 300)
        L_fixed = select_landmarks_cluster_aware(
            X_unit_full, labels0, m,
            central_frac=0.8, boundary_frac=0.2,
            per_cluster_min=3, seed=random_state
        )
    
        # 0e) Prototipe (sekali) di FULL DF
        protos0 = build_prototypes_by_cluster_gower(
            labels0, X_num_full, X_cat_full, num_min_full, num_max_full,
            per_cluster=per_cluster_proto_if_many, sample_cap=lsil_proto_sample_cap,
            seed=random_state, feature_mask_num=None, feature_mask_cat=None, inv_rng=inv_rng_full
        )
    
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

    # src/mixclust/aufs_samba/reward.py (tambahkan cabang baru)
    elif metric == "lsil_fixed_calibrated":
        from mixclust.features import build_features
        from mixclust.landmarks import select_landmarks_cluster_aware
        from mixclust.metrics.lsil import lsil_using_prototypes_gower
        from mixclust.prototypes import build_prototypes_by_cluster_gower
        from mixclust.silhouette import full_silhouette_gower_subsample
        from mixclust.aufs_samba.preprocess import prepare_mixed_arrays_no_label
        #from mixclust.metrics.lsil import lsil_using_prototypes_gower as _lsil_using_proto
        #import numpy as np
    
        # --- PRECOMPUTE sekali (sama dgn lsil_fixed) ---
        X_num_full, X_cat_full, num_min_full, num_max_full, _, _, inv_rng_full = \
            prepare_mixed_arrays_no_label(df_full)
    
        num_cols_full = df_full.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols_full = df_full.select_dtypes(include=['object','category','bool']).columns.tolist()
        num_pos = {c: i for i, c in enumerate(num_cols_full)}
        cat_pos = {c: i for i, c in enumerate(cat_cols_full)}
    
        cat_idx_full = [df_full.columns.get_loc(c) for c in cat_cols_full]
        labels0 = cluster_fn(df_full, cat_idx_full, n_clusters, random_state)
    
        X_unit_full, _, _ = build_features(df_full, label_col=None, scaler_type="standard", unit_norm=True)
        m = min(int(0.1 * len(df_full)), 300)
        L_fixed = select_landmarks_cluster_aware(
            X_unit_full, labels0, m, central_frac=0.8, boundary_frac=0.2,
            per_cluster_min=3, seed=random_state
        )
    
        protos0 = build_prototypes_by_cluster_gower(
            labels0, X_num_full, X_cat_full, num_min_full, num_max_full,
            per_cluster=per_cluster_proto_if_many, sample_cap=lsil_proto_sample_cap,
            seed=random_state, feature_mask_num=None, feature_mask_cat=None, inv_rng=inv_rng_full
        )
    
        def make_masks_for_subset(cols):
            mnum = np.zeros(X_num_full.shape[1], dtype=bool) if X_num_full.shape[1] else None
            mcat = np.zeros(X_cat_full.shape[1], dtype=bool) if X_cat_full.shape[1] else None
            for c in cols:
                if mnum is not None and c in num_pos: mnum[num_pos[c]] = True
                if mcat is not None and c in cat_pos: mcat[cat_pos[c]] = True
            return mnum, mcat
    
        # --- Kalibrasi linear S ≈ aL + b ---
        A, B = 1.0, 0.0   # koefisien awal (identitas)
        lsil_hist, ss_hist = [], []
        call_count = 0
        guard_every = 10      # hitung SS(subsample) tiap 15 panggilan
        ss_max_n_cal = 400    # subsample kecil biar cepat
    
        def reward(cols):
            nonlocal A, B, call_count, lsil_hist, ss_hist
            if not cols: return -1.0
            mask_num, mask_cat = make_masks_for_subset(cols)
    
            # 1) L-Sil (fixed)
            
            if lsil_agg_mode=="topk" and int(lsil_topk)==1:
                L = _lsil_fast(labels0, L_fixed, protos0,
                               X_num_full, X_cat_full, num_min_full, num_max_full,
                               feature_mask_num=mask_num, feature_mask_cat=mask_cat, inv_rng=inv_rng_full)
            else:
                L = lsil_using_prototypes_gower(
                    labels0, L_fixed, protos0,
                    X_num_full, X_cat_full, num_min_full, num_max_full,
                    feature_mask_num=mask_num, feature_mask_cat=mask_cat,
                    inv_rng=inv_rng_full, agg_mode=lsil_agg_mode, topk=lsil_topk
                )
    
            call_count += 1
            # 2) Sesekali ambil titik kalibrasi: SS Gower subsample atas KLASTER TERBARU (recluster)
            if (call_count % guard_every) == 0:
                try:
                    # re-cluster dgn subset (biar SS representatif), ringan saja:
                    sub_df = df_full[cols]
                    cat_idx_sub = [sub_df.columns.get_loc(c) for c in sub_df.columns if c in cat_cols]
                    labels_sub = cluster_fn(sub_df, cat_idx_sub, n_clusters, random_state)
    
                    Xn, Xc, nmin, nmax, mnum_sub, mcat_sub, inv_sub = prepare_mixed_arrays_no_label(sub_df)
                    S, _, _ = full_silhouette_gower_subsample(
                        Xn, Xc, nmin, nmax, labels_sub, max_n=ss_max_n_cal,
                        feature_mask_num=mnum_sub, feature_mask_cat=mcat_sub, inv_rng=inv_sub
                    )
                    lsil_hist.append(float(L)); ss_hist.append(float(S))
    
                    # jika sudah ≥5 titik → fit linear least squares
                    if len(lsil_hist) >= 5:
                        X = np.vstack([np.array(lsil_hist), np.ones(len(lsil_hist))]).T
                        A_, B_ = np.linalg.lstsq(X, np.array(ss_hist), rcond=None)[0]
                        A, B = float(A_), float(B_)
                        if 'verbose' in dir(np) or True:
                            print(f"[CAL] update A={A:.3f}, B={B:.3f}, n={len(lsil_hist)}")
                except Exception as e:
                    # aman-aman saja kalau kalibrasi gagal; biarkan A,B lama
                    pass
    
            # 3) kembalikan nilai TERKALIBRASI agar lebih “sejalan” dg SS
            score = A * float(L) + B
    
            if use_redundancy_penalty and redundancy_matrix is not None:
                red_score = redundancy_penalty(cols, redundancy_matrix)
                score = (1 - alpha_penalty) * score + alpha_penalty * red_score
            return float(score)
    
        return reward


    else:
        raise ValueError(f"Unknown reward metric: {metric}")


