# mixclust/utils/dav.py
#
# Domain Anchor Variable (DAV) — Phase B extension untuk MixClust
#
# POSISI FILE:  mixclust/utils/dav.py
# (sama folder dengan controller.py, structural_control.py, cluster_adapters.py)
#
# ────────────────────────────────────────────────────────────────
#  Konsep:
#    Tanpa DAV:  K* = argmax  LNC*(C, L, S*)
#    Dengan DAV: K* = argmax  LNC*_a(C, L, Va)
#                s.t. LNC*(S*) >= lnc_global_threshold
#
#  Va  = anchor subset (domain knowledge user)
#        contoh Susenas : ['DDS12_noTobPrep_norm', 'DDS13_noTob_norm']
#        contoh Obesity : ['Weight', 'BMI_category']
#        contoh Adult   : ['hours_per_week', 'education_num']
#
#  Perubahan dari pipeline standar:
#    HANYA satu baris di Auto-K loop yang berbeda —
#    lnc_star() dipanggil di ruang Va, bukan S*.
#    Semua komponen lain (AUFS-Samba, Cluster Adapter, HAC-Gower,
#    landmark, profiling) TIDAK berubah sama sekali.
# ────────────────────────────────────────────────────────────────

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import asdict
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ── Import komponen MixClust yang sudah ada ──────────────────────
from ..metrics.lnc_star import lnc_star
from ..core.preprocess import prepare_mixed_arrays_no_label
from ..core.features import build_features
from ..core.knn_index import KNNIndex
from ..core.landmarks import select_landmarks_cluster_aware

from ..clustering.controller import (
    auto_select_algo_k,
    find_best_clustering_from_subsets,
    structural_control_lnc,
    cat_cols_to_index,
    hac_landmark_hybrid_adapter,
)
from ..aufs.phase_a_cache import PhaseACache


# ================================================================
# 1.  lnc_star_anchored()
#     Evaluasi LNC* di ruang Va (anchor subset), bukan S* penuh.
# ================================================================

def lnc_star_anchored(
    X_df: pd.DataFrame,         # subset S* — hanya kolom fitur terpilih
    labels: Sequence,           # label klaster hasil clustering
    Va: List[str],              # anchor subset ⊆ kolom X_df
    *,
    lm_frac: float = 0.20,     # fraksi landmark (default paper: 20%)
    central_frac: float = 0.80,
    boundary_frac: float = 0.20,
    lnc_k: int = 50,
    lnc_alpha: float = 0.70,
    M_candidates: int = 300,
    seed: int = 42,
    verbose: bool = False,
) -> float:
    """
    Hitung LNC*_a: LNC* yang dievaluasi di ruang fitur Va.

    Langkah:
      1. Subset X_df ke Va saja.
      2. Bangun KNN index + landmark di ruang Va.
      3. Panggil lnc_star() standar — semua representasi dari Va.

    Return : float LNC*_a ∈ [0,1], atau nan jika Va tidak valid.
    """
    # Validasi Va
    Va_valid = [c for c in Va if c in X_df.columns]
    if not Va_valid:
        if verbose:
            print(f"[DAV] Kolom anchor tidak ditemukan di X_df: {Va}")
        return np.nan
    if len(Va_valid) < len(Va) and verbose:
        print(f"[DAV] Kolom anchor hilang (diabaikan): "
              f"{[c for c in Va if c not in X_df.columns]}")

    X_anchor = X_df[Va_valid].copy()
    n = len(X_anchor)

    # Array Gower di ruang Va
    X_num_a, X_cat_a, num_min_a, num_max_a, \
        mask_num_a, mask_cat_a, inv_rng_a = \
        prepare_mixed_arrays_no_label(X_anchor)

    # Unit-norm features untuk ANN
    try:
        X_unit_a, _, _ = build_features(
            X_anchor, label_col=None, scaler_type="standard", unit_norm=True
        )
    except Exception as e:
        if verbose:
            print(f"[DAV] build_features gagal di ruang Va: {e}")
        return np.nan

    # Landmark di ruang Va  (√n ≤ |L| ≤ 20%×n — sesuai paper)
    m = max(int(np.sqrt(n)), min(int(lm_frac * n), n - 1))
    labels_arr = np.asarray(labels)
    try:
        L_idx = select_landmarks_cluster_aware(
            X_unit_a, labels_arr, m,
            central_frac=central_frac,
            boundary_frac=boundary_frac,
            per_cluster_min=3,
            seed=seed,
        )
    except Exception as e:
        if verbose:
            print(f"[DAV] select_landmarks gagal: {e}")
        return np.nan

    # KNN index di ruang Va
    try:
        knn_a = KNNIndex(X_unit_a, try_hnsw=True, verbose=False)
    except Exception as e:
        if verbose:
            print(f"[DAV] KNNIndex gagal: {e}")
        return np.nan

    # Hitung LNC*_a
    try:
        val = lnc_star(
            X_unit_a, labels_arr, L_idx, knn_a,
            k=lnc_k, alpha=lnc_alpha,
            X_num=X_num_a, X_cat=X_cat_a,
            num_min=num_min_a, num_max=num_max_a,
            feature_mask_num=mask_num_a,
            feature_mask_cat=mask_cat_a,
            inv_rng=inv_rng_a,
            M_candidates=M_candidates,
        )
    except Exception as e:
        if verbose:
            print(f"[DAV] lnc_star gagal: {e}")
        return np.nan

    if verbose:
        K = len(np.unique(labels_arr))
        print(f"[DAV] LNC*_a({Va_valid}) = "
              f"{val:.4f} | n={n}, |L|={len(L_idx)}, K={K}")
    return float(val) if np.isfinite(val) else np.nan


# ================================================================
# 2.  auto_select_algo_k_dav()
#     Auto-K yang memaksimalkan LNC*_a(Va) sebagai objective,
#     dengan LNC*(S*) >= threshold sebagai guardrail.
# ================================================================

def auto_select_algo_k_dav(
    X_df: pd.DataFrame,
    cat_idx: List[int],
    algorithms: List[str],
    c_range: range,
    Va: List[str],
    *,
    phase_a_cache: Optional[PhaseACache] = None,
    lnc_global_threshold: float = 0.50,
    lnc_anchor_threshold: float = 0.40,
    lm_frac: float = 0.20,
    lnc_k: int = 50,
    lnc_alpha: float = 0.70,
    hac_mode: str = "hybrid",
    lambda_weight: float = 0.60,
    random_state: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Auto-K dengan DAV.

    K* = argmax  LNC*_a(C, L, Va)
         s.t.    LNC*(S*) >= lnc_global_threshold

    Return dict kompatibel dengan auto_select_algo_k():
        algo, k, labels, score, score_adj,
        lnc_score  (= LNC*_a — objective DAV),
        lnc_global (= LNC*(S*) — guardrail),
        dav_applied, anchor_cols, dav_history
    """
    best_k: Optional[int] = None
    best_algo: Optional[str] = None
    best_labels: Optional[np.ndarray] = None
    best_lnc_a: float = -np.inf
    best_lnc_global: float = np.nan
    history: List[Dict] = []

    cat_names = [X_df.columns[i] for i in cat_idx if i < len(X_df.columns)]

    for algo in algorithms:
        for k in c_range:

            # ── Clustering ──
            try:
                if algo in ("hac_gower", "hac_landmark"):
                    labels_k = hac_landmark_hybrid_adapter(
                        X_df, cat_idx, k, random_state, mode=hac_mode
                    )
                elif algo == "kprototypes":
                    from ..clustering.cluster_adapters import kprototypes_adapter
                    labels_k = kprototypes_adapter(X_df, cat_idx, k, random_state)
                else:
                    from ..clustering.cluster_adapters import auto_adapter
                    labels_k = auto_adapter(X_df, cat_idx, k, random_state)
            except Exception as e:
                if verbose:
                    print(f"  [DAV] {algo} K={k} gagal: {e}")
                continue

            if len(np.unique(labels_k)) < 2:
                continue

            # ── Guardrail: LNC*(S*) global ──
            try:
                sc = structural_control_lnc(
                    X_df=X_df, labels=labels_k,
                    cat_cols=cat_names,
                    lnc_threshold=lnc_global_threshold,
                    lnc_k=lnc_k,
                    random_state=random_state,
                    verbose=False,
                )
                lnc_global = sc.lnc_score
                global_ok = sc.passed
            except Exception:
                lnc_global = np.nan
                global_ok = True    # gagal hitung → tidak blokir

            # ── Objective: LNC*_a(Va) ──
            lnc_a = lnc_star_anchored(
                X_df, labels_k, Va,
                lm_frac=lm_frac, lnc_k=lnc_k, lnc_alpha=lnc_alpha,
                seed=random_state, verbose=verbose,
            )

            entry = dict(algo=algo, k=k, lnc_a=lnc_a,
                         lnc_global=lnc_global, global_ok=global_ok)
            history.append(entry)

            if verbose:
                g = "✓" if global_ok else "✗"
                a = f"{lnc_a:.4f}" if np.isfinite(lnc_a) else "nan"
                g_s = f"{lnc_global:.4f}" if np.isfinite(lnc_global) else "nan"
                print(f"  [DAV] {algo} K={k}: "
                      f"LNC*_a={a}  LNC*_global={g_s} [{g}]")

            # ── Update best ──
            if (global_ok
                    and np.isfinite(lnc_a)
                    and lnc_a >= lnc_anchor_threshold
                    and lnc_a > best_lnc_a):
                best_k, best_algo = k, algo
                best_labels = labels_k.copy()
                best_lnc_a = lnc_a
                best_lnc_global = lnc_global

    # ── Fallback ke Auto-K standar jika tidak ada yang lulus ──
    if best_labels is None:
        if verbose:
            print("[DAV] Tidak ada K yang lulus. "
                  "Fallback ke auto_select_algo_k() standar.")
        fallback = auto_select_algo_k(
            X_df=X_df, cat_idx=cat_idx,
            algorithms=algorithms, c_range=c_range,
            phase_a_cache=phase_a_cache,
            hac_mode=hac_mode,
            lambda_weight=lambda_weight,
            random_state=random_state,
        )
        fallback.update(dav_applied=False, anchor_cols=Va, dav_history=history)
        return fallback

    return {
        "algo": best_algo,
        "k": best_k,
        "labels": best_labels,
        "score": best_lnc_a,
        "score_adj": best_lnc_a,
        "lsil_score": np.nan,
        "lnc_score": best_lnc_a,       # LNC*_a  (objective DAV)
        "lnc_global": best_lnc_global,  # LNC*(S*) (guardrail)
        "dav_applied": True,
        "anchor_cols": Va,
        "dav_history": history,
        "n_unique_labels": int(len(np.unique(best_labels))),
    }


# ================================================================
# 3.  find_best_clustering_dav()
#     Phase B entry point — drop-in replacement untuk
#     find_best_clustering_from_subsets() saat DAV aktif.
# ================================================================

def find_best_clustering_dav(
    df_full: pd.DataFrame,
    top_subsets: List[List[str]],
    params: Any,                        # AUFSParams
    Va: List[str],
    *,
    phase_a_cache: Optional[PhaseACache] = None,
    lnc_global_threshold: float = 0.50,
    lnc_anchor_threshold: float = 0.40,
    lm_frac: float = 0.20,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Phase B dengan DAV aktif.

    Identik dengan find_best_clustering_from_subsets() kecuali
    satu baris: memanggil auto_select_algo_k_dav() bukan
    auto_select_algo_k().

    Parameter
    ---------
    Va : list[str]
        Kolom anchor — fitur yang mendefinisikan makna klaster
        dari perspektif domain.
        Susenas : ['DDS12_noTobPrep_norm', 'DDS13_noTob_norm']
        Obesity : ['Weight', 'BMI_category']
        Adult   : ['hours_per_week', 'education_num']
    """
    if not top_subsets:
        return {}

    t0 = perf_counter()
    cat_cols_full = list(
        df_full.select_dtypes(include=["object", "category", "bool"]).columns
    )
    algorithms = getattr(params, "auto_algorithms", None) or ["kprototypes", "hac_gower"]
    c_range = range(params.c_min, params.c_max + 1)
    hac_mode = getattr(params, "hac_mode", "hybrid")
    lambda_weight = getattr(params, "cluster_adapter_lambda", 0.6)

    if verbose:
        print(f"\n[DAV Phase B] {len(top_subsets)} subset | Va={Va} | "
              f"c_range={list(c_range)} | "
              f"guardrail LNC*(S*)>={lnc_global_threshold}")

    best_overall: Optional[Dict[str, Any]] = None
    all_history: Dict[Tuple, Dict] = {}

    for i, subset in enumerate(top_subsets):
        if not subset:
            continue
        subset_key = tuple(sorted(subset))
        df_sub = df_full[subset]
        cat_sub = [c for c in subset if c in cat_cols_full]
        cat_idx = cat_cols_to_index(df_sub, cat_sub)

        if verbose:
            print(f"\n  Subset #{i+1}/{len(top_subsets)}: {subset}")

        # ── SATU-SATUNYA BARIS YANG BERBEDA dari versi standar ──
        current = auto_select_algo_k_dav(
            X_df=df_sub,
            cat_idx=cat_idx,
            algorithms=algorithms,
            c_range=c_range,
            Va=Va,
            phase_a_cache=phase_a_cache,
            lnc_global_threshold=lnc_global_threshold,
            lnc_anchor_threshold=lnc_anchor_threshold,
            lm_frac=lm_frac,
            lnc_k=getattr(params, "lnc_k", 50),
            lnc_alpha=getattr(params, "lnc_alpha", 0.7),
            hac_mode=hac_mode,
            lambda_weight=lambda_weight,
            random_state=params.random_state,
            verbose=verbose,
        )
        # ────────────────────────────────────────────────────────

        current["subset"] = subset
        all_history[subset_key] = current

        cur_s = current.get("score_adj", -np.inf)
        best_s = best_overall.get("score_adj", -np.inf) if best_overall else -np.inf

        if best_overall is None or cur_s > best_s:
            best_overall = current
            if verbose:
                da = current.get("dav_applied", False)
                print(f"  ★ Best K={current.get('k')} "
                      f"algo={current.get('algo')} "
                      f"LNC*_a={current.get('lnc_score', np.nan):.4f} "
                      f"LNC*_global={current.get('lnc_global', np.nan):.4f} "
                      f"{'[DAV ✓]' if da else '[fallback]'}")

    if verbose:
        print(f"\n[DAV Phase B DONE] K*={best_overall.get('k') if best_overall else None}"
              f" ({perf_counter()-t0:.1f}s)")

    if best_overall:
        best_overall["all_run_history"] = {str(k): v for k, v in all_history.items()}

    return best_overall or {}
