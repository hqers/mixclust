# mixclust/utils/landmark_eval.py
from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from ..core.features import build_features, prepare_mixed_arrays
from ..core.landmarks import select_landmarks_kcenter, select_landmarks_cluster_aware
from ..core.knn_index import KNNIndex
from ..core.prototypes import build_prototypes_by_cluster_gower
from ..metrics.silhouette import full_silhouette_gower_subsample
from ..core.adaptive import adaptive_landmark_count

# Metrics
from ..metrics.lsil import lsil_using_prototypes_gower
from ..metrics.lnc_star import lnc_star

# Optional cluster adapters (untuk mode dengan cluster_fn custom)
from ..clustering.cluster_adapters import auto_adapter

SEED = 42


# ---------------------------------------------------------------------
# Helper kecil
# ---------------------------------------------------------------------
def _cat_indices(df_feat: pd.DataFrame) -> List[int]:
    """Kembalikan posisi kolom kategorik (object/category/bool)."""
    return [
        df_feat.columns.get_loc(c)
        for c in df_feat.columns
        if df_feat[c].dtype.name in ("object", "category", "bool")
    ]


def _ensure_labels(
    df_feat: pd.DataFrame,
    X_unit: np.ndarray,
    label_text: Optional[np.ndarray],
    use_gt_labels: bool,
    cluster_fn: Optional[
        Callable[[pd.DataFrame, List[int], int, Optional[int]], np.ndarray]
    ] = None,
    n_clusters: Optional[int] = None,
    random_state: int = SEED,
) -> Tuple[np.ndarray, int]:
    """
    Tentukan labels untuk evaluasi:
    - Jika use_gt_labels True dan label_text ada → pakai itu (string/teks).
    - Jika cluster_fn disediakan → pakai cluster_fn(df_feat, cat_idx, n_clusters).
    - Else fallback KMeans pada X_unit (k = #labels unik di label_text atau min 2).
    """
    if use_gt_labels and label_text is not None:
        labels = np.asarray(label_text)
        nC = max(2, len(np.unique(labels)))
        return labels, nC

    if cluster_fn is not None:
        cat_idx = _cat_indices(df_feat)
        # tebakan k jika tidak diberi: berdasarkan variasi label_text (jika ada)
        k_eff = n_clusters if n_clusters is not None else max(
            2, len(np.unique(label_text)) if label_text is not None else 3
        )
        labels = np.asarray(cluster_fn(df_feat, cat_idx, k_eff, random_state))
        nC = len(np.unique(labels))
        return labels, nC

    # Fallback KMeans internal
    k_guess = max(2, len(np.unique(label_text)) if label_text is not None else 3)
    km = KMeans(n_clusters=k_guess, random_state=random_state, n_init=10)
    labels = km.fit_predict(X_unit).astype(str)
    return labels, k_guess


def _calc_m_candidates(n: int, k_intra: int, k_inter: int) -> int:
    """
    Kandidat awal ANN sebelum re-rank Gower untuk LNC*.
    Dijaga stabil: minimal 200, maksimal 1% dari n, dan >= (k_intra+k_inter).
    """
    base = max(200, 3 * (k_intra + k_inter))
    cap = max(base, int(0.01 * n))  # 1% dari n biasanya cukup
    cap = max(cap, k_intra + k_inter)
    return int(cap)


# ---------------------------------------------------------------------
# API: Evaluasi berbasis path CSV (legacy)
# ---------------------------------------------------------------------
def evaluate_dataset(
    path: str,
    label_col: str,
    use_gt_labels: bool = False,      # default: internal murni (KMeans)
    scaler_type: str = "standard",
    unit_norm: bool = True,
    landmark_mode: str = "cluster_aware",    # "cluster_aware" | "kcenter"
    lm_max_frac: float = 0.2,
    lm_per_cluster: int = 5,
    central_frac: float = 0.8,
    boundary_frac: float = 0.2,
    k_intra: int = 150,
    k_inter: int = 150,
    oversample: int = 5,
    knn_k_lnc: int = 50,
    ss_max_n: int = 2000,
    try_hnsw: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Muat CSV → build fitur → landmark → prototipe → L-Sil & LNC* → SS(Gower).
    """
    t0_all = time.time()
    if not os.path.exists(path):
        return {"Dataset": os.path.basename(path), "Status": "File not found"}

    df = pd.read_csv(path).dropna()
    n = len(df)
    if n == 0:
        return {"Dataset": os.path.basename(path), "Status": "Empty data"}

    if verbose:
        print(f"\n▶️  {os.path.basename(path)}  (n={n})")

    # 1) Build features (untuk index cosine ANN)
    t1 = time.time()
    X_unit, label_text, _ = build_features(
        df, label_col, scaler_type=scaler_type, unit_norm=unit_norm
    )
    if verbose:
        print(f"  • Feature build: {time.time()-t1:.2f}s  (dim={X_unit.shape[1]})")

    # 1b) Komponen Gower (raw numerik & kategorik int-encoded)
    (
        X_num_raw,
        X_cat_int,
        num_min,
        num_max,
        mask_num,
        mask_cat,
        inv_rng,
    ) = prepare_mixed_arrays(df, label_col)

    # 2) Labels untuk evaluasi internal
    labels, n_clusters_eff = _ensure_labels(
        df.drop(columns=[label_col]) if label_col in df.columns else df,
        X_unit,
        label_text,
        use_gt_labels,
        cluster_fn=None,
        n_clusters=None,
        random_state=SEED,
    )

    # 3) Hitung m adaptif
    m, H, n_clusters = adaptive_landmark_count(
        labels, n, lm_max_frac=lm_max_frac, lm_per_cluster=lm_per_cluster
    )
    if verbose:
        print(f"  • Landmarks target m={m}  (clusters={n_clusters}, H={H:.3f})")

    # 4) Pilih landmark
    t2 = time.time()
    if landmark_mode == "cluster_aware":
        L = select_landmarks_cluster_aware(
            X_unit,
            labels,
            m,
            central_frac=central_frac,
            boundary_frac=boundary_frac,
            per_cluster_min=3,
            seed=SEED,
        )
    else:
        L = select_landmarks_kcenter(X_unit, m=m, seed=SEED, verbose=verbose)
    if verbose:
        print(f"  • Landmark select: {time.time()-t2:.2f}s  (|L|={len(L)})")

    # 5) kNN index (cosine ANN untuk kandidat)
    knn_index = KNNIndex(X_unit, try_hnsw=try_hnsw, verbose=verbose)

    # 5b) Prototipe medoid Gower
    per_proto = 1 if n_clusters >= 8 else 2
    protos = build_prototypes_by_cluster_gower(
        labels,
        X_num_raw,
        X_cat_int,
        num_min,
        num_max,
        per_cluster=per_proto,
        sample_cap=400,
        seed=SEED,
        feature_mask_num=mask_num,
        feature_mask_cat=mask_cat,
        inv_rng=inv_rng,
    )

    # 6) L-Sil (prototipe Gower) & LNC*
    t4 = time.time()
    lsil = lsil_using_prototypes_gower(
        labels,
        L,
        protos,
        X_num_raw,
        X_cat_int,
        num_min,
        num_max,
        feature_mask_num=mask_num,
        feature_mask_cat=mask_cat,
        inv_rng=inv_rng,
        agg_mode="topk",
        topk=1,
    )

    M_cand = _calc_m_candidates(n, k_intra, k_inter)
    lnc = lnc_star(
        X_unit,
        labels,
        L,
        knn_index,
        k=int(knn_k_lnc),
        alpha=0.7,
        X_num=X_num_raw,
        X_cat=X_cat_int,
        num_min=num_min,
        num_max=num_max,
        feature_mask_num=mask_num,
        feature_mask_cat=mask_cat,
        M_candidates=int(M_cand),
        inv_rng=inv_rng,
    )
    if verbose:
        print(
            f"  • Metrics: {time.time()-t4:.2f}s  →  L-Sil_proto={lsil:.6f}, LNC*={lnc:.6f}"
        )

    # 7) SS(Gower) (subsample bila n besar)
    t5 = time.time()
    ss_full, ss_mode, ss_n = full_silhouette_gower_subsample(
        X_num_raw,
        X_cat_int,
        num_min,
        num_max,
        labels,
        max_n=ss_max_n,
        seed=SEED,
        feature_mask_num=mask_num,
        feature_mask_cat=mask_cat,
        inv_rng=inv_rng,
    )
    if verbose:
        print(
            f"  • SS_Gower_{ss_mode}: {time.time()-t5:.2f}s  →  SS={ss_full:.6f}"
        )

    out = {
        "Dataset": os.path.basename(path),
        "N": n,
        "Clusters": n_clusters,
        "H": round(H, 6),
        "m": len(L),
        "k_intra": k_intra,
        "k_inter": k_inter,
        "oversample": oversample,
        "L-Sil_proto": round(lsil, 6),
        "LNC*": round(lnc, 6),
        "SS_Gower": round(ss_full, 6),
        "SS_mode": ss_mode,
        "SS_n_eval": int(ss_n),
        "diff(L-Sil-SS)": round(lsil - ss_full, 6),
        "MAE": round(abs(lsil - ss_full), 6),
        "Runtime(s)": round(time.time() - t0_all, 2),
        "Status": "OK",
    }
    return out


# ---------------------------------------------------------------------
# API: Evaluasi dari DataFrame (lebih fleksibel)
# ---------------------------------------------------------------------
def evaluate_lsil_only(
    df: pd.DataFrame,
    label_col: Optional[str] = None,
    selected_cols: Optional[List[str]] = None,
    *,
    scaler_type: str = "standard",
    unit_norm: bool = True,
    landmark_mode: str = "cluster_aware",
    lm_max_frac: float = 0.2,
    lm_per_cluster: int = 5,
    central_frac: float = 0.8,
    boundary_frac: float = 0.2,
    try_hnsw: bool = True,
    verbose: bool = True,
    use_gt_labels: bool = False
) -> Dict[str, Any]:
    """
    Uji hanya L-Sil (fase B parsial).
    """
    t0 = time.time()
    df = df.dropna().copy()
    n = len(df)

    # Pisah fitur vs label
    if label_col is not None and label_col in df.columns:
        df_feat = df.drop(columns=[label_col])
    else:
        df_feat = df.copy()

    if selected_cols is not None:
        df_feat = df_feat[selected_cols]

    # 1) fitur & komponen Gower
    X_unit, label_text, _ = build_features(
        df_feat, label_col=None, scaler_type=scaler_type, unit_norm=unit_norm
    )
    X_num, X_cat, num_min, num_max, mask_num, mask_cat, inv_rng = prepare_mixed_arrays(
        df_feat, label_col=None
    )

    # 2) labels
    labels, n_clusters_eff = _ensure_labels(
        df_feat, X_unit, label_text, use_gt_labels, cluster_fn=None, n_clusters=None
    )

    # 3) m adaptif & landmark
    m, H, n_clusters = adaptive_landmark_count(
        labels, n, lm_max_frac=lm_max_frac, lm_per_cluster=lm_per_cluster
    )
    if landmark_mode == "cluster_aware":
        L = select_landmarks_cluster_aware(
            X_unit,
            labels,
            m,
            central_frac=central_frac,
            boundary_frac=boundary_frac,
            per_cluster_min=3,
            seed=SEED,
        )
    else:
        L = select_landmarks_kcenter(X_unit, m=m, seed=SEED, verbose=verbose)

    # 4) prototipe
    per_proto = 1 if n_clusters >= 8 else 2
    protos = build_prototypes_by_cluster_gower(
        labels,
        X_num,
        X_cat,
        num_min,
        num_max,
        per_cluster=per_proto,
        sample_cap=400,
        seed=SEED,
        feature_mask_num=mask_num,
        feature_mask_cat=mask_cat,
        inv_rng=inv_rng,
    )

    # 5) L-Sil
    lsil = lsil_using_prototypes_gower(
        labels,
        L,
        protos,
        X_num,
        X_cat,
        num_min,
        num_max,
        feature_mask_num=mask_num,
        feature_mask_cat=mask_cat,
        inv_rng=inv_rng,
        agg_mode="topk",
        topk=1,
    )

    return {
        "N": n,
        "Clusters": n_clusters,
        "H": float(H),
        "m": len(L),
        "L-Sil_proto": float(lsil),
        "Runtime(s)": round(time.time() - t0, 2),
        "Status": "OK",
    }


def evaluate_lnc_only(
    df: pd.DataFrame,
    label_col: Optional[str] = None,
    selected_cols: Optional[List[str]] = None,
    *,
    scaler_type: str = "standard",
    unit_norm: bool = True,
    landmark_mode: str = "cluster_aware",
    lm_max_frac: float = 0.2,
    lm_per_cluster: int = 5,
    central_frac: float = 0.8,
    boundary_frac: float = 0.2,
    knn_k_lnc: int = 50,
    try_hnsw: bool = True,
    verbose: bool = True,
    use_gt_labels: bool = False
) -> Dict[str, Any]:
    """
    Uji hanya LNC* (fase B parsial).
    """
    t0 = time.time()
    df = df.dropna().copy()
    n = len(df)

    # Pisah fitur vs label
    if label_col is not None and label_col in df.columns:
        df_feat = df.drop(columns=[label_col])
    else:
        df_feat = df.copy()

    if selected_cols is not None:
        df_feat = df_feat[selected_cols]

    # 1) fitur & komponen Gower
    X_unit, label_text, _ = build_features(
        df_feat, label_col=None, scaler_type=scaler_type, unit_norm=unit_norm
    )
    X_num, X_cat, num_min, num_max, mask_num, mask_cat, inv_rng = prepare_mixed_arrays(
        df_feat, label_col=None
    )

    # 2) labels
    labels, n_clusters_eff = _ensure_labels(
        df_feat, X_unit, label_text, use_gt_labels, cluster_fn=None, n_clusters=None
    )

    # 3) m adaptif & landmark
    m, H, n_clusters = adaptive_landmark_count(
        labels, n, lm_max_frac=lm_max_frac, lm_per_cluster=lm_per_cluster
    )
    if landmark_mode == "cluster_aware":
        L = select_landmarks_cluster_aware(
            X_unit,
            labels,
            m,
            central_frac=central_frac,
            boundary_frac=boundary_frac,
            per_cluster_min=3,
            seed=SEED,
        )
    else:
        L = select_landmarks_kcenter(X_unit, m=m, seed=SEED, verbose=verbose)

    # 4) kNN index
    knn_index = KNNIndex(X_unit, try_hnsw=try_hnsw, verbose=verbose)

    # 5) kandidat adaptif untuk LNC*
    M_cand = _calc_m_candidates(n, k_intra=knn_k_lnc, k_inter=knn_k_lnc)

    # 6) LNC*
    lnc = lnc_star(
        X_unit,
        labels,
        L,
        knn_index,
        k=int(knn_k_lnc),
        alpha=0.7,
        X_num=X_num,
        X_cat=X_cat,
        num_min=num_min,
        num_max=num_max,
        feature_mask_num=mask_num,
        feature_mask_cat=mask_cat,
        M_candidates=int(M_cand),
        inv_rng=inv_rng,
    )

    return {
        "N": n,
        "Clusters": n_clusters,
        "H": float(H),
        "m": len(L),
        "k_lnc": int(knn_k_lnc),
        "M_candidates": int(M_cand),
        "LNC*": float(lnc),
        "Runtime(s)": round(time.time() - t0, 2),
        "Status": "OK",
    }


def evaluate_dataframe_phaseB(
    df: pd.DataFrame,
    label_col: Optional[str] = None,
    selected_cols: Optional[List[str]] = None,
    *,
    use_gt_labels: bool = False,
    scaler_type: str = "standard",
    unit_norm: bool = True,
    landmark_mode: str = "cluster_aware",
    lm_max_frac: float = 0.2,
    lm_per_cluster: int = 5,
    central_frac: float = 0.8,
    boundary_frac: float = 0.2,
    knn_k_lnc: int = 50,
    ss_max_n: int = 2000,
    try_hnsw: bool = True,
    cluster_fn: Optional[
        Callable[[pd.DataFrame, List[int], int, Optional[int]], np.ndarray]
    ] = None,
    n_clusters: Optional[int] = None,
    random_state: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluasi Phase B lengkap dari DataFrame (tanpa path file).
    - Mendukung cluster_fn custom (mis. adapter auto/AUFS).
    - Menghasilkan L-Sil, LNC*, dan SS(Gower).
    """
    t0 = time.time()
    df = df.dropna().copy()
    n = len(df)

    # Pisah fitur vs label
    if label_col is not None and label_col in df.columns:
        df_feat = df.drop(columns=[label_col])
    else:
        df_feat = df.copy()

    if selected_cols is not None:
        df_feat = df_feat[selected_cols]

    # 1) fitur & komponen Gower
    X_unit, label_text, _ = build_features(
        df_feat, label_col=None, scaler_type=scaler_type, unit_norm=unit_norm
    )
    X_num, X_cat, num_min, num_max, mask_num, mask_cat, inv_rng = prepare_mixed_arrays(
        df_feat, label_col=None
    )

    # 2) labels (GT / cluster_fn / KMeans)
    labels, n_clusters_eff = _ensure_labels(
        df_feat,
        X_unit,
        label_text,
        use_gt_labels,
        cluster_fn=cluster_fn,
        n_clusters=n_clusters,
        random_state=random_state,
    )

    # 3) m adaptif & landmark
    m, H, n_clusters_adapt = adaptive_landmark_count(
        labels, n, lm_max_frac=lm_max_frac, lm_per_cluster=lm_per_cluster
    )
    if landmark_mode == "cluster_aware":
        L = select_landmarks_cluster_aware(
            X_unit,
            labels,
            m,
            central_frac=central_frac,
            boundary_frac=boundary_frac,
            per_cluster_min=3,
            seed=SEED,
        )
    else:
        L = select_landmarks_kcenter(X_unit, m=m, seed=SEED, verbose=verbose)

    # 4) index kNN & prototipe
    knn_index = KNNIndex(X_unit, try_hnsw=try_hnsw, verbose=verbose)
    per_proto = 1 if n_clusters_adapt >= 8 else 2
    protos = build_prototypes_by_cluster_gower(
        labels,
        X_num,
        X_cat,
        num_min,
        num_max,
        per_cluster=per_proto,
        sample_cap=400,
        seed=SEED,
        feature_mask_num=mask_num,
        feature_mask_cat=mask_cat,
        inv_rng=inv_rng,
    )

    # 5) L-Sil & LNC*
    lsil = lsil_using_prototypes_gower(
        labels,
        L,
        protos,
        X_num,
        X_cat,
        num_min,
        num_max,
        feature_mask_num=mask_num,
        feature_mask_cat=mask_cat,
        inv_rng=inv_rng,
        agg_mode="topk",
        topk=1,
    )
    M_cand = _calc_m_candidates(n, k_intra=knn_k_lnc, k_inter=knn_k_lnc)
    lnc = lnc_star(
        X_unit,
        labels,
        L,
        knn_index,
        k=int(knn_k_lnc),
        alpha=0.7,
        X_num=X_num,
        X_cat=X_cat,
        num_min=num_min,
        num_max=num_max,
        feature_mask_num=mask_num,
        feature_mask_cat=mask_cat,
        M_candidates=int(M_cand),
        inv_rng=inv_rng,
    )

    # 6) SS(Gower)
    ss_full, ss_mode, ss_n = full_silhouette_gower_subsample(
        X_num, X_cat, num_min, num_max, labels, max_n=ss_max_n, seed=SEED
    )

    return {
        "N": n,
        "Clusters": int(n_clusters_adapt),
        "H": float(H),
        "m": len(L),
        "k_lnc": int(knn_k_lnc),
        "M_candidates": int(M_cand),
        "L-Sil_proto": float(lsil),
        "LNC*": float(lnc),
        "SS_Gower": float(ss_full),
        "SS_mode": ss_mode,
        "SS_n_eval": int(ss_n),
        "diff(L-Sil-SS)": float(lsil - ss_full),
        "MAE": float(abs(lsil - ss_full)),
        "Runtime(s)": round(time.time() - t0, 2),
        "Status": "OK",
    }
