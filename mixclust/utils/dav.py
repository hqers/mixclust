# mixclust/utils/dav.py
#
# Domain Anchor Variable (DAV) — Phase B extension untuk MixClust
#
# v1.1.10 FIX: DAV Phase B 60x lebih lambat dari non-DAV
#   ROOT CAUSE: _AnchorContext membangun |L|=66K landmark (lm_frac=0.20 × 334K)
#   padahal Phase A cache hanya pakai |L|=1734 (c*sqrt(n)).
#   Kompleksitas LNC*_a = O(n × |L| × k) → 39x lebih mahal dari seharusnya.
#
#   FIX 1: Ganti formula |L| di _AnchorContext: lm_frac*n → c*sqrt(n)
#     Sebelum: max(sqrt(n), lm_frac*n) = max(578, 66845) = 66845
#     Sesudah: max(30, min(c*sqrt(n), cap_abs)) = 1734
#     Speedup: ~39x per LNC*_a call
#
#   FIX 2: Subsample data untuk AnchorContext (KNNIndex + LNC*_a)
#     KNNIndex dibangun pada subsample anchor_subsample_n (default 10K)
#     bukan full n=334K. Landmark dipilih dari subsample.
#     Speedup build: ~33x, LNC*_a tetap representative.
#
#   FIX 3: Cache AnchorContext antar subset berdasarkan Va_valid key
#     Subset 1,2,3 Susenas semua pakai Va=[DDS12,DDS13] → build 1x, reuse 3x.
#     _AnchorContextCache di find_best_clustering_dav menyimpan context per Va_key.
#
#   Estimasi speedup gabungan: ~40-100x
#   Target: Phase B DAV ≈ 2-3x Phase B non-DAV (~20-30 menit)
#
# v1.1.9 FIX:
#   1. DAV winner tidak dibandingkan langsung dengan fallback score (apple vs orange)
#   2. lnc_anchor_threshold default: 0.40 → 0.25
#   3. Log Va_valid vs Va_requested
#
# v1.1.8 FIX:
#   1. Clustering pakai subsample adapter (bukan full 334k)
#   2. LNC* global reuse phase_a_cache.knn_index (bukan bangun ulang)
#   3. Anchor KNN+landmark di-cache per subset (bukan per trial)
# ────────────────────────────────────────────────────────────────

from __future__ import annotations

import numpy as np
import pandas as pd
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
# 1. Prebuilt anchor context — dibangun SEKALI per (subset, Va_key)
# ================================================================

class _AnchorContext:
    """Cache KNN index + landmarks + arrays di ruang Va.

    v1.1.10: Tiga perubahan penting vs v1.1.8:
    (a) |L| dihitung dengan c*sqrt(n), bukan lm_frac*n.
        Ini konsisten dengan Phase A cache dan jauh lebih kecil
        pada data berskala besar (1734 vs 66845 untuk n=334K).
    (b) KNNIndex dan LNC*_a dievaluasi pada subsample
        (anchor_subsample_n baris), bukan full dataset.
        Pada n=334K, subsample 10K sudah representative.
    (c) Va_key property untuk cache lookup antar subset.
    """
    __slots__ = ('X_unit_a', 'knn_a', 'L_idx',
                 'X_num_a', 'X_cat_a', 'num_min_a', 'num_max_a',
                 'mask_num_a', 'mask_cat_a', 'inv_rng_a',
                 'Va_valid', 'Va_key', 'n_full', 'ok')

    def __init__(
        self,
        X_df: pd.DataFrame,
        Va: List[str],
        labels_init: np.ndarray,
        *,
        lm_c: float = 3.0,           # |L| = lm_c * sqrt(n), konsisten dengan Phase A
        lm_cap_abs: int = 3000,       # batas atas absolut landmark
        lm_floor: int = 30,           # batas bawah absolut landmark
        anchor_subsample_n: int = 10_000,  # subsample untuk KNNIndex & LNC*_a
        seed: int = 42,
        verbose: bool = False,
    ):
        self.ok = False
        self.Va_valid = [c for c in Va if c in X_df.columns]
        self.Va_key   = tuple(sorted(self.Va_valid))  # hashable cache key
        self.n_full   = len(X_df)

        # v1.1.10: log Va yang ditemukan vs diminta
        if verbose:
            missing = [c for c in Va if c not in X_df.columns]
            print(f"[DAV] Anchor context: Va diminta={Va}")
            if missing:
                print(f"[DAV]   Tidak ditemukan di subset: {missing}")
            print(f"[DAV]   Va aktif: {self.Va_valid}")

        if not self.Va_valid:
            if verbose:
                print("[DAV] Tidak ada Va yang ditemukan — skip anchor context")
            return

        # ── v1.1.10 FIX 2: subsample untuk efisiensi ──────────────────────
        n = len(X_df)
        if n > anchor_subsample_n:
            rng = np.random.default_rng(seed)
            # Stratified subsample berdasarkan labels_init agar distribusi terjaga
            idx_list = []
            uniq, counts = np.unique(labels_init, return_counts=True)
            for c, cnt in zip(uniq, counts):
                take = max(3, int(round(anchor_subsample_n * cnt / n)))
                pool = np.where(labels_init == c)[0]
                take = min(take, len(pool))
                idx_list.append(rng.choice(pool, size=take, replace=False))
            idx_sub = np.sort(np.unique(np.concatenate(idx_list)))
            if len(idx_sub) > anchor_subsample_n:
                idx_sub = rng.choice(idx_sub, size=anchor_subsample_n, replace=False)
                idx_sub = np.sort(idx_sub)
            X_anchor = X_df[self.Va_valid].iloc[idx_sub]
            labels_sub = labels_init[idx_sub]
            if verbose:
                print(f"[DAV]   Subsample: {n:,} → {len(idx_sub):,} rows untuk anchor context")
        else:
            X_anchor   = X_df[self.Va_valid]
            labels_sub = labels_init

        n_sub = len(X_anchor)

        # Gower arrays di ruang Va (pada subsample)
        self.X_num_a, self.X_cat_a, self.num_min_a, self.num_max_a, \
            self.mask_num_a, self.mask_cat_a, self.inv_rng_a = \
            prepare_mixed_arrays_no_label(X_anchor)

        # Unit-norm untuk ANN (pada subsample)
        try:
            self.X_unit_a, _, _ = build_features(
                X_anchor, label_col=None, scaler_type="standard", unit_norm=True
            )
        except Exception as e:
            if verbose:
                print(f"[DAV] build_features gagal: {e}")
            return

        # ── v1.1.10 FIX 1: |L| = lm_c * sqrt(n_sub) ─────────────────────
        # Konsisten dengan Phase A (c*sqrt(n)) dan jauh lebih kecil dari lm_frac*n
        m = int(np.clip(lm_c * np.sqrt(n_sub), lm_floor, min(lm_cap_abs, n_sub - 1)))
        try:
            self.L_idx = select_landmarks_cluster_aware(
                self.X_unit_a, labels_sub, m,
                central_frac=0.80, boundary_frac=0.20,
                per_cluster_min=3, seed=seed,
            )
        except Exception as e:
            if verbose:
                print(f"[DAV] select_landmarks gagal: {e}")
            return

        # KNN index di ruang Va (pada subsample)
        try:
            self.knn_a = KNNIndex(self.X_unit_a, try_hnsw=True, verbose=False)
        except Exception as e:
            if verbose:
                print(f"[DAV] KNNIndex gagal: {e}")
            return

        self.ok = True
        if verbose:
            print(f"[DAV] Anchor context OK: |Va|={len(self.Va_valid)}, "
                  f"|L|={len(self.L_idx)}, n_sub={n_sub:,} (dari {n:,})")


def _lnc_star_anchored_fast(
    anchor_ctx: _AnchorContext,
    labels: np.ndarray,
    lnc_k: int = 50,
    lnc_alpha: float = 0.70,
    M_candidates: int = 300,
) -> float:
    """LNC*_a menggunakan prebuilt anchor context.

    v1.1.10: LNC*_a dievaluasi pada subsample (n_sub rows) bukan full n.
    Ini konsisten karena anchor context dibangun pada subsample yang sama.
    """
    if not anchor_ctx.ok:
        return np.nan
    try:
        val = lnc_star(
            anchor_ctx.X_unit_a, labels[:len(anchor_ctx.X_unit_a)],
            anchor_ctx.L_idx, anchor_ctx.knn_a,
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
# 2. lnc_star_anchored — public API (backward compat)
# ================================================================

def lnc_star_anchored(
    X_df: pd.DataFrame,
    labels: Sequence,
    Va: List[str],
    *,
    lm_c: float = 3.0,
    anchor_subsample_n: int = 10_000,
    lnc_k: int = 50,
    lnc_alpha: float = 0.70,
    M_candidates: int = 300,
    seed: int = 42,
    verbose: bool = False,
) -> float:
    """Public API — builds anchor context from scratch. For one-off calls."""
    labels_arr = np.asarray(labels)
    ctx = _AnchorContext(
        X_df, Va, labels_arr,
        lm_c=lm_c, anchor_subsample_n=anchor_subsample_n,
        seed=seed, verbose=verbose,
    )
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
    """Compute LNC*(S*) reusing phase_a_cache.knn_index."""
    if (phase_a_cache is None or not phase_a_cache.available
            or phase_a_cache.knn_index is None
            or phase_a_cache.X_unit_full is None):
        return np.nan, True

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
        return val, True
    except Exception:
        return np.nan, True


# ================================================================
# 4. auto_select_algo_k_dav — v1.1.10
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
    lnc_anchor_threshold: float = 0.25,
    lm_c: float = 3.0,
    anchor_subsample_n: int = 10_000,
    lnc_k: int = 50,
    lnc_alpha: float = 0.70,
    hac_mode: str = "hybrid",
    lambda_weight: float = 0.60,
    random_state: int = 42,
    verbose: bool = True,
    # v1.1.10: accept pre-built context for cross-subset caching
    _prebuilt_anchor_ctx: Optional[_AnchorContext] = None,
    _prebuilt_idx_sub: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Auto-K with DAV — v1.1.10.

    Perubahan dari v1.1.9:
    1. _AnchorContext pakai |L|=c*sqrt(n_sub) bukan lm_frac*n (~39x speedup)
    2. Anchor context dibangun pada subsample (anchor_subsample_n) bukan full n
    3. Menerima _prebuilt_anchor_ctx untuk cache antar subset
    """
    best_k: Optional[int] = None
    best_algo: Optional[str] = None
    best_labels: Optional[np.ndarray] = None
    best_lnc_a: float = -np.inf
    best_lnc_global: float = np.nan
    history: List[Dict] = []

    cols = list(X_df.columns)

    # ── Gunakan pre-built context atau build baru ─────────────────────────
    if _prebuilt_anchor_ctx is not None and _prebuilt_anchor_ctx.ok:
        anchor_ctx = _prebuilt_anchor_ctx
        idx_sub    = _prebuilt_idx_sub
        if verbose:
            print(f"[DAV] Reusing cached anchor context "
                  f"(Va={list(anchor_ctx.Va_key)}, |L|={len(anchor_ctx.L_idx)})")
    else:
        dummy_labels = np.zeros(len(X_df), dtype=int)
        if phase_a_cache is not None and phase_a_cache.labels0 is not None:
            dummy_labels = phase_a_cache.labels0
        anchor_ctx, idx_sub = _build_anchor_context_with_idx(
            X_df, Va, dummy_labels,
            lm_c=lm_c, anchor_subsample_n=anchor_subsample_n,
            seed=random_state, verbose=verbose,
        )

    if verbose and anchor_ctx.ok:
        n_va_found = len(anchor_ctx.Va_valid)
        n_va_req   = len(Va)
        if n_va_found < n_va_req:
            print(f"[DAV] ⚠ Hanya {n_va_found}/{n_va_req} Va ditemukan. "
                  f"LNC*_a mungkin kurang representatif.")

    for algo in algorithms:
        for k in c_range:

            # ── Clustering (subsample adapters) ───────────────────────────
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

            # ── Guardrail: LNC*(S*) via cache ─────────────────────────────
            lnc_global, _ = _lnc_global_from_cache(
                labels_k, cols, phase_a_cache,
                lnc_k=lnc_k, lnc_alpha=lnc_alpha,
            )
            global_ok = np.isnan(lnc_global) or lnc_global >= lnc_global_threshold

            # ── Objective: LNC*_a(Va) — pada subsample ────────────────────
            # v1.1.10: labels_k dipotong ke panjang subsample
            labels_for_anchor = (labels_k[idx_sub]
                                 if idx_sub is not None and anchor_ctx.ok
                                 else labels_k)
            lnc_a = _lnc_star_anchored_fast(
                anchor_ctx, labels_for_anchor,
                lnc_k=lnc_k, lnc_alpha=lnc_alpha,
            )

            entry = dict(algo=algo, k=k, lnc_a=lnc_a,
                         lnc_global=lnc_global, global_ok=global_ok)
            history.append(entry)

            if verbose:
                g   = "✓" if global_ok else "✗"
                a   = f"{lnc_a:.4f}" if np.isfinite(lnc_a) else "nan"
                g_s = f"{lnc_global:.4f}" if np.isfinite(lnc_global) else "nan"
                print(f"  [DAV] {algo} K={k}: "
                      f"LNC*_a={a}  LNC*_global={g_s} [{g}]")

            if (global_ok
                    and np.isfinite(lnc_a)
                    and lnc_a >= lnc_anchor_threshold
                    and lnc_a > best_lnc_a):
                best_k, best_algo = k, algo
                best_labels = labels_k.copy()
                best_lnc_a  = lnc_a
                best_lnc_global = lnc_global

    # ── Fallback ──────────────────────────────────────────────────────────
    if best_labels is None:
        if verbose:
            best_trial = max(
                (e for e in history if np.isfinite(e.get('lnc_a', float('nan')))),
                key=lambda x: x['lnc_a'], default=None
            )
            if best_trial:
                print(f"[DAV] Tidak ada K yang lulus threshold={lnc_anchor_threshold:.2f}. "
                      f"Best LNC*_a={best_trial['lnc_a']:.4f} (K={best_trial['k']}). "
                      f"Fallback ke auto_select_algo_k().")
            else:
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


def _build_anchor_context_with_idx(
    X_df: pd.DataFrame,
    Va: List[str],
    labels_init: np.ndarray,
    *,
    lm_c: float = 3.0,
    lm_cap_abs: int = 3000,
    lm_floor: int = 30,
    anchor_subsample_n: int = 10_000,
    seed: int = 42,
    verbose: bool = False,
) -> Tuple[_AnchorContext, Optional[np.ndarray]]:
    """Build _AnchorContext dan kembalikan juga idx_sub yang dipakai."""
    n = len(X_df)
    idx_sub: Optional[np.ndarray] = None

    if n > anchor_subsample_n:
        rng = np.random.default_rng(seed)
        idx_list = []
        uniq, counts = np.unique(labels_init, return_counts=True)
        for c, cnt in zip(uniq, counts):
            take = max(3, int(round(anchor_subsample_n * cnt / n)))
            pool = np.where(labels_init == c)[0]
            take = min(take, len(pool))
            idx_list.append(rng.choice(pool, size=take, replace=False))
        idx_sub = np.sort(np.unique(np.concatenate(idx_list)))
        if len(idx_sub) > anchor_subsample_n:
            idx_sub = rng.choice(idx_sub, size=anchor_subsample_n, replace=False)
            idx_sub = np.sort(idx_sub)

    ctx = _AnchorContext(
        X_df, Va, labels_init,
        lm_c=lm_c, lm_cap_abs=lm_cap_abs, lm_floor=lm_floor,
        anchor_subsample_n=anchor_subsample_n,
        seed=seed, verbose=verbose,
    )
    return ctx, idx_sub


# ================================================================
# 5. find_best_clustering_dav — Phase B entry point v1.1.10
# ================================================================

def find_best_clustering_dav(
    df_full: pd.DataFrame,
    top_subsets: List[List[str]],
    params: Any,
    Va: List[str],
    *,
    phase_a_cache: Optional[PhaseACache] = None,
    lnc_global_threshold: float = 0.50,
    lnc_anchor_threshold: float = 0.25,
    lm_c: float = 3.0,
    anchor_subsample_n: int = 10_000,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Phase B with DAV — v1.1.10.

    FIX v1.1.10: AnchorContext cache antar subset berdasarkan Va_valid key.
    Subset yang berbeda tapi mengandung Va yang sama (e.g. semua punya DDS12+DDS13)
    akan berbagi AnchorContext yang sama tanpa rebuild.

    FIX v1.1.9: DAV winner selalu diprioritaskan atas fallback result.
    """
    if not top_subsets:
        return {}

    t0 = perf_counter()
    cat_cols_full = list(
        df_full.select_dtypes(include=["object", "category", "bool"]).columns
    )
    algorithms  = getattr(params, "auto_algorithms", None) or ["kprototypes", "hac_gower"]
    c_range     = range(params.c_min, params.c_max + 1)
    hac_mode    = getattr(params, "hac_mode", "hybrid")
    lambda_weight = getattr(params, "cluster_adapter_lambda", 0.6)
    rand        = params.random_state

    if verbose:
        print(f"\n[DAV Phase B v1.1.10] {len(top_subsets)} subset | Va={Va} | "
              f"c_range={list(c_range)} | "
              f"anchor_thr={lnc_anchor_threshold:.2f} | "
              f"guardrail LNC*(S*)>={lnc_global_threshold} | "
              f"anchor_sub={anchor_subsample_n:,}")

    # ── v1.1.10 FIX 3: cache AnchorContext antar subset ──────────────────
    # Key = tuple(sorted(Va_valid yang ditemukan di subset))
    # Subset berbeda dengan Va yang sama berbagi context yang sama
    _anchor_cache: Dict[tuple, Tuple[_AnchorContext, Optional[np.ndarray]]] = {}

    # Dapatkan labels0 untuk subsample stratified
    dummy_labels = np.zeros(len(df_full), dtype=int)
    if phase_a_cache is not None and phase_a_cache.labels0 is not None:
        dummy_labels = phase_a_cache.labels0

    best_overall: Optional[Dict[str, Any]] = None
    all_history: Dict[tuple, Dict] = {}

    for i, subset in enumerate(top_subsets):
        if not subset:
            continue
        subset_key = tuple(sorted(subset))
        df_sub  = df_full[subset]
        cat_sub = [c for c in subset if c in cat_cols_full]
        cat_idx = cat_cols_to_index(df_sub, cat_sub)

        if verbose:
            print(f"\n  Subset #{i+1}/{len(top_subsets)}: {subset}")

        # ── Tentukan Va_valid untuk subset ini ───────────────────────────
        Va_valid_here = [c for c in Va if c in subset]
        va_cache_key  = tuple(sorted(Va_valid_here))

        # ── Lookup atau build AnchorContext ───────────────────────────────
        if va_cache_key in _anchor_cache:
            prebuilt_ctx, prebuilt_idx = _anchor_cache[va_cache_key]
            if verbose:
                print(f"  [DAV] Cache hit: Va_key={list(va_cache_key)}, "
                      f"|L|={len(prebuilt_ctx.L_idx) if prebuilt_ctx.ok else 0}")
        else:
            prebuilt_ctx, prebuilt_idx = _build_anchor_context_with_idx(
                df_sub, Va, dummy_labels,
                lm_c=lm_c, anchor_subsample_n=anchor_subsample_n,
                seed=rand, verbose=verbose,
            )
            _anchor_cache[va_cache_key] = (prebuilt_ctx, prebuilt_idx)

        current = auto_select_algo_k_dav(
            X_df=df_sub,
            cat_idx=cat_idx,
            algorithms=algorithms,
            c_range=c_range,
            Va=Va,
            phase_a_cache=phase_a_cache,
            lnc_global_threshold=lnc_global_threshold,
            lnc_anchor_threshold=lnc_anchor_threshold,
            lm_c=lm_c,
            anchor_subsample_n=anchor_subsample_n,
            lnc_k=getattr(params, "lnc_k", 50),
            lnc_alpha=getattr(params, "lnc_alpha", 0.7),
            hac_mode=hac_mode,
            lambda_weight=lambda_weight,
            random_state=rand,
            verbose=verbose,
            _prebuilt_anchor_ctx=prebuilt_ctx,
            _prebuilt_idx_sub=prebuilt_idx,
        )

        current["subset"] = subset
        all_history[subset_key] = current

        # ── v1.1.9: fair comparison — DAV winner > fallback ──────────────
        if _should_update(current, best_overall):
            best_overall = current
            if verbose:
                da    = current.get("dav_applied", False)
                lnca  = current.get('lnc_score', float('nan'))
                lncg  = current.get('lnc_global', float('nan'))
                la_s  = f"{lnca:.4f}" if np.isfinite(lnca) else "nan"
                lg_s  = f"{lncg:.4f}" if np.isfinite(lncg) else "nan"
                print(f"  ★ Best K={current.get('k')} algo={current.get('algo')} "
                      f"LNC*_a={la_s} LNC*_global={lg_s} "
                      f"{'[DAV ✓]' if da else '[fallback]'}")

    elapsed = perf_counter() - t0
    if verbose:
        k_final   = best_overall.get('k') if best_overall else None
        dav_final = best_overall.get('dav_applied', False) if best_overall else False
        n_cached  = len(_anchor_cache)
        print(f"\n[DAV Phase B DONE] K*={k_final} "
              f"{'[DAV ✓]' if dav_final else '[fallback]'} "
              f"| anchor contexts built: {n_cached} "
              f"| {elapsed:.1f}s")

    if best_overall:
        best_overall["all_run_history"] = {str(k): v for k, v in all_history.items()}

    return best_overall or {}


def _should_update(cur: Dict, best: Optional[Dict]) -> bool:
    """v1.1.9: fair comparison — DAV winner tidak dibandingkan dengan fallback score."""
    if best is None:
        return True
    c_dav = cur.get("dav_applied", False)
    b_dav = best.get("dav_applied", False)
    if c_dav and not b_dav:
        return True   # DAV selalu menang atas fallback
    if not c_dav and b_dav:
        return False  # fallback tidak bisa mengalahkan DAV
    if c_dav and b_dav:
        return cur.get("lnc_score", -np.inf) > best.get("lnc_score", -np.inf)
    return cur.get("score_adj", -np.inf) > best.get("score_adj", -np.inf)
