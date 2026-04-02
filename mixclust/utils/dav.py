# mixclust/utils/dav.py
#
# Domain Anchor Variable (DAV) — Phase B extension untuk MixClust
#
# v1.1.8 FIX:
#   1. Clustering pakai subsample adapter (bukan full 334k)
#   2. LNC* global reuse phase_a_cache.knn_index (bukan bangun ulang)
#   3. Anchor KNN+landmark di-cache per subset (bukan per trial)
# ────────────────────────────────────────────────────────────────

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import asdict
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
# 1. Prebuilt anchor context — dibangun SEKALI per subset
# ================================================================

class _AnchorContext:
    """Cache KNN index + landmarks + arrays di ruang Va.
    Dibangun sekali per subset, dipakai ulang per trial (algo, K)."""
    __slots__ = ('X_unit_a', 'knn_a', 'L_idx',
                 'X_num_a', 'X_cat_a', 'num_min_a', 'num_max_a',
                 'mask_num_a', 'mask_cat_a', 'inv_rng_a',
                 'Va_valid', 'ok')

    def __init__(self, X_df: pd.DataFrame, Va: List[str], labels_init: np.ndarray,
                 lm_frac: float = 0.20, seed: int = 42, verbose: bool = False):
        self.ok = False
        self.Va_valid = [c for c in Va if c in X_df.columns]
        if not self.Va_valid:
            if verbose:
                print(f"[DAV] Anchor cols tidak ditemukan: {Va}")
            return

        X_anchor = X_df[self.Va_valid]
        n = len(X_anchor)

        # Gower arrays di ruang Va
        self.X_num_a, self.X_cat_a, self.num_min_a, self.num_max_a, \
            self.mask_num_a, self.mask_cat_a, self.inv_rng_a = \
            prepare_mixed_arrays_no_label(X_anchor)

        # Unit-norm untuk ANN
        try:
            self.X_unit_a, _, _ = build_features(
                X_anchor, label_col=None, scaler_type="standard", unit_norm=True
            )
        except Exception as e:
            if verbose:
                print(f"[DAV] build_features gagal: {e}")
            return

        # Landmark di ruang Va
        m = max(int(np.sqrt(n)), min(int(lm_frac * n), n - 1))
        try:
            self.L_idx = select_landmarks_cluster_aware(
                self.X_unit_a, labels_init, m,
                central_frac=0.80, boundary_frac=0.20,
                per_cluster_min=3, seed=seed,
            )
        except Exception as e:
            if verbose:
                print(f"[DAV] select_landmarks gagal: {e}")
            return

        # KNN index di ruang Va
        try:
            self.knn_a = KNNIndex(self.X_unit_a, try_hnsw=True, verbose=False)
        except Exception as e:
            if verbose:
                print(f"[DAV] KNNIndex gagal: {e}")
            return

        self.ok = True
        if verbose:
            print(f"[DAV] Anchor context: Va={self.Va_valid}, "
                  f"|L|={len(self.L_idx)}, n={n}")


def _lnc_star_anchored_fast(
    anchor_ctx: _AnchorContext,
    labels: np.ndarray,
    lnc_k: int = 50,
    lnc_alpha: float = 0.70,
    M_candidates: int = 300,
) -> float:
    """LNC*_a menggunakan prebuilt anchor context — tanpa rebuild KNN/landmark."""
    if not anchor_ctx.ok:
        return np.nan
    try:
        val = lnc_star(
            anchor_ctx.X_unit_a, labels, anchor_ctx.L_idx, anchor_ctx.knn_a,
            k=lnc_k, alpha=lnc_alpha,
            X_num=anchor_ctx.X_num_a, X_cat=anchor_ctx.X_cat_a,
            num_min=anchor_ctx.num_min_a, num_max=anchor_ctx.num_max_a,
            feature_mask_num=anchor_ctx.mask_num_a,
            feature_mask_cat=anchor_ctx.mask_cat_a,
            inv_rng=anchor_ctx.inv_rng_a,
            M_candidates=M_candidates,
        )
        return float(val) if np.isfinite(val) else np.nan
    except Exception:
        return np.nan


# ================================================================
# 2. lnc_star_anchored — public API (backward compat, builds fresh)
# ================================================================

def lnc_star_anchored(
    X_df: pd.DataFrame,
    labels: Sequence,
    Va: List[str],
    *,
    lm_frac: float = 0.20,
    central_frac: float = 0.80,
    boundary_frac: float = 0.20,
    lnc_k: int = 50,
    lnc_alpha: float = 0.70,
    M_candidates: int = 300,
    seed: int = 42,
    verbose: bool = False,
) -> float:
    """Public API — builds anchor context from scratch. For one-off calls."""
    labels_arr = np.asarray(labels)
    ctx = _AnchorContext(X_df, Va, labels_arr, lm_frac=lm_frac,
                         seed=seed, verbose=verbose)
    return _lnc_star_anchored_fast(ctx, labels_arr, lnc_k, lnc_alpha, M_candidates)


# ================================================================
# 3. _lnc_global_from_cache — reuse phase_a_cache
# ================================================================

def _lnc_global_from_cache(
    labels: np.ndarray,
    cols: List[str],
    phase_a_cache: Optional[PhaseACache],
    lnc_k: int = 50,
    lnc_alpha: float = 0.70,
) -> Tuple[float, bool]:
    """
    Compute LNC*(S*) reusing phase_a_cache.knn_index.
    Returns (lnc_score, passed).
    Falls back to NaN (pass) if cache not available.
    """
    if (phase_a_cache is None or not phase_a_cache.available
            or phase_a_cache.knn_index is None
            or phase_a_cache.X_unit_full is None):
        return np.nan, True  # can't compute → don't block

    mask_num, mask_cat = phase_a_cache.make_masks_for_subset(cols)
    try:
        n = phase_a_cache.n_samples
        M_cand = min(max(3 * lnc_k, 100), max(50, int(0.05 * n)))
        val = float(lnc_star(
            phase_a_cache.X_unit_full, labels,
            phase_a_cache.L_fixed, phase_a_cache.knn_index,
            k=lnc_k, alpha=lnc_alpha,
            X_num=phase_a_cache.X_num_full,
            X_cat=phase_a_cache.X_cat_full,
            num_min=phase_a_cache.num_min_full,
            num_max=phase_a_cache.num_max_full,
            feature_mask_num=mask_num,
            feature_mask_cat=mask_cat,
            inv_rng=phase_a_cache.inv_rng_full,
            M_candidates=M_cand,
        ))
        return val, True  # passed check is done by caller
    except Exception:
        return np.nan, True


# ================================================================
# 4. auto_select_algo_k_dav — OPTIMIZED
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
    Auto-K with DAV — optimized v1.1.8.

    Changes from v1.1.6:
    1. Clustering via subsample adapters (not full n)
    2. LNC* global via phase_a_cache (not fresh KNNIndex per trial)
    3. Anchor KNN+landmark built ONCE per auto_select call (not per trial)
    """
    best_k: Optional[int] = None
    best_algo: Optional[str] = None
    best_labels: Optional[np.ndarray] = None
    best_lnc_a: float = -np.inf
    best_lnc_global: float = np.nan
    history: List[Dict] = []

    cols = list(X_df.columns)

    # ── Build anchor context ONCE ──
    # Use a dummy label array for initial landmark selection
    # (landmarks are spatial, label-independent for anchor)
    dummy_labels = np.zeros(len(X_df), dtype=int)
    if phase_a_cache is not None and phase_a_cache.labels0 is not None:
        dummy_labels = phase_a_cache.labels0
    anchor_ctx = _AnchorContext(X_df, Va, dummy_labels,
                                lm_frac=lm_frac, seed=random_state,
                                verbose=verbose)

    for algo in algorithms:
        for k in c_range:

            # ── Clustering (subsample adapters) ──
            try:
                if algo in ("hac_gower", "hac_landmark"):
                    labels_k = hac_landmark_hybrid_adapter(
                        X_df, cat_idx, k, random_state, mode=hac_mode
                    )
                elif algo == "kprototypes":
                    from ..clustering.cluster_adapters import kprototypes_subsample_adapter
                    labels_k = kprototypes_subsample_adapter(
                        X_df, cat_idx, k, random_state, subsample_n=6000
                    )
                elif algo == "kamila":
                    from ..clustering.cluster_adapters import kamila_subsample_adapter
                    labels_k = kamila_subsample_adapter(
                        X_df, cat_idx, k, random_state, subsample_n=6000
                    )
                else:
                    from ..clustering.cluster_adapters import auto_adapter
                    labels_k = auto_adapter(X_df, cat_idx, k, random_state)
            except Exception as e:
                if verbose:
                    print(f"  [DAV] {algo} K={k} clustering gagal: {e}")
                continue

            if len(np.unique(labels_k)) < 2:
                continue

            # ── Guardrail: LNC*(S*) via cache ──
            lnc_global, _ = _lnc_global_from_cache(
                labels_k, cols, phase_a_cache,
                lnc_k=lnc_k, lnc_alpha=lnc_alpha,
            )
            global_ok = (np.isnan(lnc_global)
                         or lnc_global >= lnc_global_threshold)

            # ── Objective: LNC*_a(Va) via cached anchor context ──
            lnc_a = _lnc_star_anchored_fast(
                anchor_ctx, labels_k,
                lnc_k=lnc_k, lnc_alpha=lnc_alpha,
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

            if (global_ok
                    and np.isfinite(lnc_a)
                    and lnc_a >= lnc_anchor_threshold
                    and lnc_a > best_lnc_a):
                best_k, best_algo = k, algo
                best_labels = labels_k.copy()
                best_lnc_a = lnc_a
                best_lnc_global = lnc_global

    # ── Fallback ──
    if best_labels is None:
        if verbose:
            print("[DAV] Tidak ada K yang lulus. Fallback ke auto_select_algo_k().")
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
        "lnc_score": best_lnc_a,
        "lnc_global": best_lnc_global,
        "dav_applied": True,
        "anchor_cols": Va,
        "dav_history": history,
        "n_unique_labels": int(len(np.unique(best_labels))),
    }


# ================================================================
# 5. find_best_clustering_dav — Phase B entry point
# ================================================================

def find_best_clustering_dav(
    df_full: pd.DataFrame,
    top_subsets: List[List[str]],
    params: Any,
    Va: List[str],
    *,
    phase_a_cache: Optional[PhaseACache] = None,
    lnc_global_threshold: float = 0.50,
    lnc_anchor_threshold: float = 0.40,
    lm_frac: float = 0.20,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Phase B with DAV — optimized v1.1.8."""
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
