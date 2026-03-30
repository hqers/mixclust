# mixclust/clustering/controller.py
#
# ARSITEKTUR FIX: Phase B menggunakan kembali precomputed cache dari Phase A
#
# SEBELUM (lambat):
#   Setiap trial (subset, algo, K) di Phase B:
#     1. cluster ulang → ok (diperlukan untuk auto-K)
#     2. prepare_mixed_arrays_no_label(df_subset) → re-Gower dari scratch ❌
#     3. score_internal() → L-Sil dari nol ❌
#     4. _compute_lnc_for_subset() → LNC* dari nol ❌
#   Total: ~30 detik/trial × 150 trial = 75 menit
#
# SESUDAH (cepat):
#   Phase B menerima PhaseACache dari api.py
#   Setiap trial (subset, algo, K):
#     1. cluster ulang → ok (diperlukan untuk auto-K)
#     2. buat mask subset dari cache → O(|cols|), microseconds ✅
#     3. L-Sil via L_fixed + mask → O(n·|L|) ✅
#     4. LNC* via L_fixed yang sama → O(n·|L|) ✅
#   Total: ~0.5 detik/trial × 150 trial = ~1-2 menit
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations
import numpy as np
import pandas as pd
import random
from time import perf_counter

from sklearn.metrics import davies_bouldin_score as _dbi
from sklearn.metrics import calinski_harabasz_score as _chi

from .cluster_adapters import (
    hac_gower_adapter,
    kprototypes_adapter,
    kprototypes_subsample_adapter,   # FIX v1.1.5
    kmodes_adapter,
    auto_adapter,
)

try:
    from ..metrics.lsil import lsil_using_prototypes_gower
    from ..metrics.lsil import lsil_using_landmarks   # ← tambah import langsung
except Exception:
    lsil_using_prototypes_gower = None
    lsil_using_landmarks = None

try:
    from ..metrics.silhouette import full_silhouette_gower_subsample
except Exception:
    full_silhouette_gower_subsample = None

try:
    from ..metrics.lnc_star import lnc_star
except Exception:
    lnc_star = None

try:
    from ..core.knn_index import KNNIndex
except Exception:
    KNNIndex = None

from ..core.preprocess import preprocess_mixed_data, prepare_mixed_arrays_no_label
from sklearn.preprocessing import normalize
from ..core.landmarks import select_landmarks_kcenter, select_landmarks_cluster_aware
from ..core.adaptive import adaptive_landmark_count
from ..core.prototypes import build_prototypes_by_cluster_gower

# Import PhaseACache
from ..aufs.phase_a_cache import PhaseACache

from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple, Dict, Any


# ================================================================
# Data classes
# ================================================================

@dataclass
class AutoClustResult:
    algo: Optional[str] = None
    k: Optional[int] = None
    labels: Optional[np.ndarray] = None
    score: float = -np.inf
    score_adj: float = -np.inf
    lsil_score: float = np.nan
    lnc_score: float = np.nan
    pi_frag: float = 0.0
    pi_imb: float = 0.0
    dbi: float = np.nan
    chi: float = np.nan
    metric_used: Optional[str] = None
    n_unique_labels: int = 0

    def is_better_than(self, other: 'AutoClustResult', tie_breakers: Tuple[str, ...]) -> bool:
        if self.score_adj > other.score_adj + 1e-9:
            return True
        if abs(self.score_adj - other.score_adj) < 1e-9:
            return _tie_better(self.__dict__, other.__dict__, tie_breakers)
        return False


@dataclass
class StructuralControlResult:
    lnc_score: float = np.nan
    passed: bool = False
    threshold: float = 0.5
    action: str = "none"
    n_landmarks: int = 0
    message: str = ""
    timing_s: float = 0.0


# ================================================================
# Helpers
# ================================================================

def cat_cols_to_index(X_df: pd.DataFrame, cat_cols: List[str]) -> List[int]:
    name2pos = {c: i for i, c in enumerate(X_df.columns)}
    return [name2pos[c] for c in cat_cols if c in name2pos]


def _tie_better(a: Dict[str, Any], b: Dict[str, Any], tie_breakers: Tuple[str, ...]) -> bool:
    for tb in tie_breakers:
        if tb == "dbi":
            va, vb = a.get("dbi", np.inf), b.get("dbi", np.inf)
            if np.isfinite(va) and np.isfinite(vb) and va != vb:
                return va < vb
        elif tb == "chi":
            va, vb = a.get("chi", -np.inf), b.get("chi", -np.inf)
            if np.isfinite(va) and np.isfinite(vb) and va != vb:
                return va > vb
    return False


def estimate_gamma(X_df: pd.DataFrame, cat_idx: List[int], scale: float = 1.0) -> float:
    num_cols = [c for i, c in enumerate(X_df.columns) if i not in cat_idx]
    if not num_cols:
        return 1.0
    s = X_df[num_cols].std(ddof=0).replace(0, 1e-9).mean()
    return float(scale * s)


# ================================================================
# HAC Landmark-Hybrid (dari fix sebelumnya, tetap dipakai)
# ================================================================

def hac_landmark_hybrid_adapter(
    X_df: pd.DataFrame,
    cat_idx: List[int],
    k: int,
    random_state: int,
    *,
    lm_frac: float = 0.15,
    lm_cap: int = 300,
    lm_per_cluster_min: int = 3,
    mode: str = "hybrid",
) -> np.ndarray:
    n = len(X_df)
    if mode == "full_hac":
        return hac_gower_adapter(X_df, cat_idx, k, random_state)

    X_num, X_cat, num_min, num_max, mask_num, mask_cat, inv_rng = \
        prepare_mixed_arrays_no_label(X_df)

    m = min(int(lm_frac * n), lm_cap)
    m = max(m, k * lm_per_cluster_min)

    try:
        from ..core.features import build_features
        X_unit, _, _ = build_features(
            X_df, label_col=None, scaler_type="standard", unit_norm=True
        )
        L_idx = select_landmarks_kcenter(X_unit, m=m, seed=random_state)
    except Exception:
        L_idx = np.arange(min(m, n), dtype=int)

    if len(L_idx) < k:
        return kprototypes_adapter(X_df, cat_idx, k, random_state, max_iter=20)

    df_landmarks = X_df.iloc[L_idx].reset_index(drop=True)
    cat_idx_lm = [
        df_landmarks.columns.get_loc(X_df.columns[i])
        for i in cat_idx if X_df.columns[i] in df_landmarks.columns
    ]

    try:
        lm_labels = hac_gower_adapter(df_landmarks, cat_idx_lm, k, random_state)
    except Exception:
        return kprototypes_adapter(X_df, cat_idx, k, random_state, max_iter=20)

    if len(np.unique(lm_labels)) < 2:
        return kprototypes_adapter(X_df, cat_idx, k, random_state, max_iter=20)

    X_num_lm = X_num[L_idx]
    X_cat_lm = X_cat[L_idx] if X_cat.shape[1] > 0 else X_cat

    protos_lm = build_prototypes_by_cluster_gower(
        lm_labels, X_num_lm, X_cat_lm, num_min, num_max,
        per_cluster=1, sample_cap=len(L_idx),
        seed=random_state,
        feature_mask_num=mask_num, feature_mask_cat=mask_cat,
        inv_rng=inv_rng,
    )

    if not protos_lm:
        return kprototypes_adapter(X_df, cat_idx, k, random_state, max_iter=20)

    try:
        from ..core.gower import gower_to_one_mixed
        labels_all = np.zeros(n, dtype=int)
        proto_ids = sorted(protos_lm.keys())
        for i in range(n):
            best_c, best_d = proto_ids[0], np.inf
            for c in proto_ids:
                if not protos_lm[c]:
                    continue
                d = float(np.mean(gower_to_one_mixed(
                    X_num, X_cat, num_min, num_max, i, protos_lm[c],
                    feature_mask_num=mask_num, feature_mask_cat=mask_cat,
                    inv_rng=inv_rng,
                )))
                if d < best_d:
                    best_d, best_c = d, c
            labels_all[i] = best_c
        unique_labs = np.unique(labels_all)
        label_map = {old: new for new, old in enumerate(sorted(unique_labs))}
        return np.array([label_map[l] for l in labels_all], dtype=int)
    except Exception:
        return kprototypes_adapter(X_df, cat_idx, k, random_state, max_iter=20)


# ================================================================
# Skor Komposit J(algo) = λ·L-Sil + (1-λ)·LNC* - π_frag - π_imb
# ================================================================

def _compute_composite_score_J(
    lsil: float,
    lnc: float,
    labels: np.ndarray,
    k: int,
    *,
    lambda_weight: float = 0.6,
    penalty_lambda: float = 0.02,
    min_cluster_size_frac: float = 0.02,
    n_samples: int = 1,
) -> Tuple[float, float, float]:
    if np.isnan(lnc):
        base_score = float(lsil) if np.isfinite(lsil) else -1.0
    elif np.isnan(lsil):
        base_score = float(lnc) if np.isfinite(lnc) else -1.0
    else:
        base_score = lambda_weight * float(lsil) + (1 - lambda_weight) * float(lnc)

    min_size = max(3, int(min_cluster_size_frac * n_samples))
    cluster_counts = np.bincount(
        labels.astype(int) if labels.dtype.kind in 'iu'
        else np.unique(labels, return_inverse=True)[1]
    )
    n_tiny = int(np.sum(cluster_counts < min_size))
    pi_frag = 0.5 * n_tiny if n_tiny > 0 else 0.0

    if len(cluster_counts) > 1:
        sizes = cluster_counts.astype(float)
        cv = float(np.std(sizes) / (np.mean(sizes) + 1e-9))
        pi_imb = 0.1 * max(0.0, cv - 1.0)
    else:
        pi_imb = 0.0

    complexity_penalty = penalty_lambda * np.log(max(2, k))
    J = base_score - pi_frag - pi_imb - complexity_penalty
    return float(J), float(pi_frag), float(pi_imb)


# ================================================================
# INTI FIX: Evaluasi Phase B dengan cache Phase A
#
# Fungsi ini adalah pengganti score_internal() + _compute_lnc_for_subset()
# Memakai kembali X_num_full, L_fixed, protos0 dari Phase A
# via mask subset — tidak re-Gower dari scratch
# ================================================================

def _eval_with_phase_a_cache(
    cols: List[str],
    labels_new: np.ndarray,
    cache: PhaseACache,
    *,
    lsil_agg_mode: str = "topk",
    lsil_topk: int = 1,
    lnc_k: int = 20,
    lnc_alpha: float = 0.7,
) -> Tuple[float, float]:
    """
    Evaluasi (L-Sil, LNC*) untuk subset kolom `cols` dengan label baru
    `labels_new`, menggunakan kembali precomputed cache dari Phase A.

    v1.1.6: L-Sil dievaluasi pada Phase B subsample (~30k rows) jika tersedia.
    Speedup ~11x untuk n=334k (96s→9s per trial).

    Returns: (lsil_score, lnc_score)
    """
    if not cache.available:
        return np.nan, np.nan

    mask_num, mask_cat = cache.make_masks_for_subset(cols)

    # ── L-Sil ──
    lsil_score = np.nan
    try:
        if lsil_using_landmarks is not None:
            from ..metrics.lsil import compute_lsil_from_D
            from ..core.gower import gower_distances_to_landmarks

            # Pilih jalur: subsample Phase B (cepat) atau full (fallback)
            if cache._pb_available:
                # JALUR CEPAT: evaluasi pada ~30k rows
                labels_pb = labels_new[cache._pb_idx]
                lm_labels_pb = labels_pb[cache._pb_L]
                if len(np.unique(lm_labels_pb)) >= 2:
                    D = gower_distances_to_landmarks(
                        cache._pb_X_num, cache._pb_X_cat,
                        cache._pb_num_min, cache._pb_num_max,
                        cache._pb_L,
                        feature_mask_num=mask_num,
                        feature_mask_cat=mask_cat,
                        inv_rng=cache._pb_inv_rng,
                    )
                    score_val, _ = compute_lsil_from_D(
                        D, labels_pb, lm_labels_pb,
                        agg_mode=lsil_agg_mode, topk=lsil_topk,
                    )
                    lsil_score = float(score_val)
            else:
                # FALLBACK: full data
                lm_labels_new = labels_new[cache.L_fixed]
                if len(np.unique(lm_labels_new)) >= 2:
                    D = gower_distances_to_landmarks(
                        cache.X_num_full, cache.X_cat_full,
                        cache.num_min_full, cache.num_max_full,
                        cache.L_fixed,
                        feature_mask_num=mask_num,
                        feature_mask_cat=mask_cat,
                        inv_rng=cache.inv_rng_full,
                    )
                    score_val, _ = compute_lsil_from_D(
                        D, labels_new, lm_labels_new,
                        agg_mode=lsil_agg_mode, topk=lsil_topk,
                    )
                    lsil_score = float(score_val)
    except Exception:
        pass

    # ── LNC* ── (tetap pada full data — sudah prebuilt KNNIndex)
    lnc_score = np.nan
    try:
        if (lnc_star is not None
                and cache.knn_index is not None
                and cache.X_unit_full is not None):
            n = cache.n_samples
            M_cand = min(max(3 * lnc_k, 100), max(50, int(0.05 * n)))
            lnc_score = float(lnc_star(
                cache.X_unit_full, labels_new, cache.L_fixed, cache.knn_index,
                k=lnc_k, alpha=lnc_alpha,
                X_num=cache.X_num_full,
                X_cat=cache.X_cat_full,
                num_min=cache.num_min_full,
                num_max=cache.num_max_full,
                feature_mask_num=mask_num,
                feature_mask_cat=mask_cat,
                inv_rng=cache.inv_rng_full,
                M_candidates=M_cand,
            ))
    except Exception:
        pass

    return lsil_score, lnc_score


# ================================================================
# score_internal — fallback jika cache tidak tersedia
# ================================================================

def score_internal(
    X_df: pd.DataFrame,
    labels: np.ndarray,
    cat_idx: List[int],
    *,
    metric: str = "auto",
    ss_max_n: int = 1000,
    landmarks=None,
    lsil_m: int = 150,
    random_state: int = 42
) -> float:
    n = len(X_df)
    _lsil = lsil_using_prototypes_gower
    _ss_full = full_silhouette_gower_subsample

    use_ss = (
        (metric == "ss_gower")
        or (metric == "auto" and n < ss_max_n)
        or (_lsil is None and _ss_full is not None)
    )
    if use_ss:
        if _ss_full is None:
            raise RuntimeError("Silhouette Gower not available.")
        X_num, X_cat, num_min, num_max, mask_num, mask_cat, inv_rng = \
            prepare_mixed_arrays_no_label(X_df)
        ss, _, _ = _ss_full(
            X_num, X_cat, num_min, num_max, labels,
            max_n=ss_max_n,
            feature_mask_num=mask_num, feature_mask_cat=mask_cat,
            inv_rng=inv_rng
        )
        return float(ss)

    if _lsil is None and lsil_using_landmarks is None:
        raise RuntimeError("L-Sil not available.")

    X_num, X_cat, num_min, num_max, mask_num, mask_cat, inv_rng = \
        prepare_mixed_arrays_no_label(X_df)
    m = min(lsil_m, max(20, n // 10))
    if X_num.shape[1] > 0:
        X_unit = normalize(X_num, norm="l2")
        landmark_idx = select_landmarks_kcenter(X_unit, m=m, seed=random_state)
    else:
        landmark_idx = np.arange(min(m, n), dtype=int)

    # FIX v2.1: pakai lsil_using_landmarks (bukan lsil_using_prototypes_gower)
    # — tidak perlu build_prototypes, landmark langsung jadi referensi klaster
    if lsil_using_landmarks is not None:
        score = lsil_using_landmarks(
            labels, landmark_idx,
            X_num, X_cat, num_min, num_max,
            feature_mask_num=mask_num, feature_mask_cat=mask_cat,
            inv_rng=inv_rng, agg_mode="topk", topk=1
        )
    else:
        # Fallback ke wrapper lama (signature sudah kompatibel)
        score = _lsil(
            labels, landmark_idx,
            X_num, X_cat, num_min, num_max,
            feature_mask_num=mask_num, feature_mask_cat=mask_cat,
            inv_rng=inv_rng, agg_mode="topk", topk=1
        )
    return float(score)


# ================================================================
# auto_select_algo_k — Phase B evaluation
# ================================================================

def auto_select_algo_k(
    X_df: pd.DataFrame,
    cat_idx: List[int],
    algorithms: List[str],
    c_range: range,
    *,
    phase_a_cache: Optional[PhaseACache] = None,   # ← KUNCI BARU
    primary_metric: str = "auto",
    lambda_weight: float = 0.6,
    penalty_lambda: float = 0.02,
    min_cluster_size_frac: float = 0.02,
    tie_breakers: Tuple[str, ...] = ("dbi", "chi"),
    hac_mode: str = "hybrid",
    lnc_k: int = 20,
    lnc_alpha: float = 0.7,
    lsil_agg_mode: str = "topk",
    lsil_topk: int = 1,
    enable_screening: bool = True,
    screening_k_values: Tuple[int, ...] = (2, 3, 4),
    screening_prune_threshold: float = 0.15,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Phase B: Cari (algorithm, K) terbaik menggunakan composite score J(algo).

    Jika phase_a_cache tersedia (mode C dengan lsil_fixed_calibrated):
      → L-Sil dan LNC* dihitung via _eval_with_phase_a_cache()
      → Tidak re-Gower, tidak re-landmark → O(n·|L|) per trial
      → Estimasi: ~0.5s/trial vs ~30s/trial sebelumnya

    Jika cache tidak tersedia (fallback):
      → score_internal() dari scratch (perilaku lama)
    """
    use_cache = phase_a_cache is not None and phase_a_cache.available
    best_result = AutoClustResult(labels=np.zeros(len(X_df), dtype=int))
    n_samples = len(X_df)

    active_algorithms = list(algorithms)
    if n_samples > 2000 and "hac_gower" in active_algorithms:
        if hac_mode == "full_hac":
            active_algorithms = [a for a in active_algorithms if a != "hac_gower"]

    if not active_algorithms:
        active_algorithms = [a for a in algorithms if a != "hac_gower"]

    gamma_for_kproto = (
        estimate_gamma(X_df, cat_idx)
        if "kprototypes" in active_algorithms else None
    )

    cols = list(X_df.columns)

    def _run_algo(algo: str, k: int) -> Optional[np.ndarray]:
        try:
            if algo == "hac_gower":
                return hac_landmark_hybrid_adapter(
                    X_df, cat_idx, k, random_state, mode=hac_mode
                )
            elif algo == "kprototypes":
                return kprototypes_subsample_adapter(   # ← ganti dari kprototypes_adapter
                    X_df, cat_idx, k, random_state,
                    subsample_n=6000                    # ~18% dari 32k
                )
            elif algo == "kmodes":
                return kmodes_adapter(X_df, cat_idx, k, random_state)
            else:
                return auto_adapter(X_df, cat_idx, k, random_state)
        except Exception:
            return None

    def _eval_one(algo: str, k: int) -> Optional[AutoClustResult]:
        labels = _run_algo(algo, k)
        if labels is None:
            return None
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return None

        # ── Evaluasi: cache atau fallback ──
        if use_cache:
            # JALUR CEPAT: gunakan cache Phase A
            # O(n·|L|) — tidak re-Gower
            lsil, lnc = _eval_with_phase_a_cache(
                cols, labels, phase_a_cache,
                lsil_agg_mode=lsil_agg_mode,
                lsil_topk=lsil_topk,
                lnc_k=lnc_k,
                lnc_alpha=lnc_alpha,
            )
        else:
            # JALUR FALLBACK: score dari scratch
            try:
                lsil = score_internal(
                    X_df, labels, cat_idx,
                    metric=primary_metric,
                    random_state=random_state
                )
            except Exception:
                lsil = -1.0
            lnc = np.nan  # LNC* tidak dihitung di fallback (terlalu mahal)

        # Composite score J(algo)
        J, pi_frag, pi_imb = _compute_composite_score_J(
            lsil, lnc, labels, k,
            lambda_weight=lambda_weight,
            penalty_lambda=penalty_lambda,
            min_cluster_size_frac=min_cluster_size_frac,
            n_samples=n_samples,
        )

        dbi, chi = np.nan, np.nan
        try:
            X_num_df = X_df.select_dtypes(include=[np.number])
            if X_num_df.shape[1] >= 1 and len(unique_labels) > 1:
                dbi = _dbi(X_num_df, labels)
                chi = _chi(X_num_df, labels)
        except Exception:
            pass

        return AutoClustResult(
            algo=algo, k=k, labels=labels,
            score=float(lsil) if np.isfinite(lsil) else -1.0,
            score_adj=float(J),
            lsil_score=float(lsil) if np.isfinite(lsil) else np.nan,
            lnc_score=float(lnc) if np.isfinite(lnc) else np.nan,
            pi_frag=float(pi_frag),
            pi_imb=float(pi_imb),
            dbi=float(dbi) if np.isfinite(dbi) else np.nan,
            chi=float(chi) if np.isfinite(chi) else np.nan,
            metric_used="cached_lsil" if use_cache else primary_metric,
            n_unique_labels=len(unique_labels)
        )

    # ── Screening Awal (Bab VI.7.6 Langkah 2) ──
    active_after_screening = list(active_algorithms)

    if enable_screening and len(active_algorithms) > 1:
        k_screen = [k for k in screening_k_values if k in c_range]
        if not k_screen:
            k_screen = [list(c_range)[0]]

        screening_scores: Dict[str, List[float]] = {a: [] for a in active_algorithms}
        for algo in active_algorithms:
            for k in k_screen:
                res = _eval_one(algo, k)
                if res is not None:
                    screening_scores[algo].append(res.score_adj)

        algo_mean_scores = {
            a: float(np.mean(v)) if v else -np.inf
            for a, v in screening_scores.items()
        }
        best_screen_score = max(algo_mean_scores.values())
        active_after_screening = [
            a for a, s in algo_mean_scores.items()
            if s >= best_screen_score - screening_prune_threshold
        ]
        if not active_after_screening:
            active_after_screening = [max(algo_mean_scores, key=algo_mean_scores.get)]

        pruned = set(active_algorithms) - set(active_after_screening)
        if pruned:
            print(f"  [SCREENING] Pruned: {pruned} | Retained: {active_after_screening}")

    # ── Full Auto-K ──
    history: List[AutoClustResult] = []
    for algo in active_after_screening:
        for k in c_range:
            res = _eval_one(algo, k)
            if res is None:
                continue
            history.append(res)
            if best_result.algo is None or res.is_better_than(best_result, tie_breakers):
                best_result = res

    return asdict(best_result)


# ================================================================
# Structural Control (tidak berubah)
# ================================================================

def structural_control_lnc(
    X_df: pd.DataFrame,
    labels: np.ndarray,
    cat_cols: List[str],
    *,
    lnc_threshold: float = 0.5,
    lnc_k: int = 50,
    lnc_alpha: float = 0.7,
    landmark_mode: str = "cluster_aware",
    lm_c: float = 3.0,             # c dalam c*sqrt(n) (Theorem 1 JDSA)
    lm_cap_frac: float = 0.2,      # batas atas fraksional
    lm_per_cluster_min: int = 3,
    central_frac: float = 0.8,
    boundary_frac: float = 0.2,
    M_candidates: int = 300,
    try_hnsw: bool = True,
    random_state: int = 42,
    verbose: bool = True,
) -> StructuralControlResult:
    t0 = perf_counter()
    if lnc_star is None or KNNIndex is None:
        return StructuralControlResult(
            lnc_score=np.nan, passed=True, action="none",
            message="LNC* atau KNNIndex tidak tersedia.",
            timing_s=perf_counter() - t0
        )
    labels = np.asarray(labels)
    n = len(labels)
    if n == 0 or len(np.unique(labels)) < 2:
        return StructuralControlResult(
            lnc_score=np.nan, passed=False, action="warning",
            message="Data tidak cukup.",
            timing_s=perf_counter() - t0
        )
    try:
        from ..core.features import build_features
        X_unit, _, _ = build_features(
            X_df, label_col=None, scaler_type="standard", unit_norm=True
        )
    except Exception as e:
        return StructuralControlResult(
            lnc_score=np.nan, passed=True, action="none",
            message=f"build_features gagal: {e}",
            timing_s=perf_counter() - t0
        )
    X_num, X_cat, num_min, num_max, mask_num, mask_cat, inv_rng = \
        prepare_mixed_arrays_no_label(X_df)
    K = len(np.unique(labels))
    m = adaptive_landmark_count(n, K=K, c=lm_c, cap_frac=lm_cap_frac,
                                per_cluster_min=lm_per_cluster_min)
    if landmark_mode == "cluster_aware":
        L = select_landmarks_cluster_aware(
            X_unit, labels, m,
            central_frac=central_frac, boundary_frac=boundary_frac,
            per_cluster_min=lm_per_cluster_min, seed=random_state
        )
    else:
        L = select_landmarks_kcenter(X_unit, m=m, seed=random_state)
    try:
        knn_index = KNNIndex(X_unit, try_hnsw=try_hnsw, verbose=False)
    except Exception as e:
        return StructuralControlResult(
            lnc_score=np.nan, passed=True, action="none",
            message=f"KNNIndex gagal: {e}",
            timing_s=perf_counter() - t0
        )
    M_cand = min(max(M_candidates, 3 * lnc_k), max(50, int(0.05 * n)))
    try:
        lnc_score_val = lnc_star(
            X_unit, labels, L, knn_index,
            k=lnc_k, alpha=lnc_alpha,
            X_num=X_num, X_cat=X_cat,
            num_min=num_min, num_max=num_max,
            feature_mask_num=mask_num, feature_mask_cat=mask_cat,
            inv_rng=inv_rng, M_candidates=M_cand,
        )
    except Exception as e:
        return StructuralControlResult(
            lnc_score=np.nan, passed=True, action="none",
            message=f"LNC* gagal: {e}",
            timing_s=perf_counter() - t0
        )
    lnc_val = float(lnc_score_val) if np.isfinite(lnc_score_val) else 0.0
    elapsed = perf_counter() - t0
    passed = lnc_val >= lnc_threshold
    result = StructuralControlResult(
        lnc_score=lnc_val, passed=passed,
        threshold=lnc_threshold,
        action="accept" if passed else "warning",
        n_landmarks=len(L),
        message=(
            f"LNC*={lnc_val:.4f} >= threshold={lnc_threshold:.2f}. Adequate."
            if passed else
            f"LNC*={lnc_val:.4f} < threshold={lnc_threshold:.2f}. May be insufficient."
        ),
        timing_s=elapsed
    )
    if verbose:
        status = "✅ PASS" if passed else "⚠️  WARNING"
        print(f"[STRUCTURAL CONTROL] {status} | LNC*={lnc_val:.4f} "
              f"(thr={lnc_threshold:.2f}) | |L|={len(L)} | {elapsed:.2f}s")
    return result


# ================================================================
# PHASE B: find_best_clustering_from_subsets
# ================================================================

def find_best_clustering_from_subsets(
    df_full: pd.DataFrame,
    top_subsets: List[List[str]],
    params: Any,
    verbose: bool = True,
    *,
    phase_a_cache: Optional[PhaseACache] = None,   # ← KUNCI BARU
    run_structural_control: bool = True,
    lnc_threshold: float = 0.5,
    lnc_k: int = 50,
) -> Dict[str, Any]:
    """
    Phase B dengan cache Phase A.

    Jika phase_a_cache tersedia: evaluasi O(n·|L|) per trial
    Jika tidak: fallback ke perilaku lama O(n²) per trial
    """
    if not top_subsets:
        if verbose:
            print("[PHASE B] Tidak ada subset. Abort.")
        return {}

    cache_available = phase_a_cache is not None and phase_a_cache.available
    if verbose:
        mode_str = (
            f"CACHE MODE (|L|={phase_a_cache.n_landmarks}, reuse Phase A components)"
            if cache_available else "FALLBACK MODE (recompute from scratch)"
        )
        print(f"\n[BEGIN PHASE B] {len(top_subsets)} subset | {mode_str}")

    best_overall_result: Optional[Dict[str, Any]] = None
    all_run_history: Dict[Tuple[str, ...], Dict[str, Any]] = {}

    algorithms = params.auto_algorithms or ["kprototypes", "hac_gower"]
    c_range = range(params.c_min, params.c_max + 1)
    cat_cols_full = list(
        df_full.select_dtypes(include=['object', 'category', 'bool']).columns
    )

    hac_mode = getattr(params, 'hac_mode', 'hybrid')
    lambda_weight = getattr(params, 'cluster_adapter_lambda', 0.6)

    for i, subset in enumerate(top_subsets):
        if not subset:
            continue

        subset_key = tuple(sorted(subset))
        if verbose:
            print(f"  -> Subset #{i+1} ({len(subset)} fitur): {subset}...")

        df_subset = df_full[subset]
        cat_cols_subset = [c for c in subset if c in cat_cols_full]
        cat_idx_subset = cat_cols_to_index(df_subset, cat_cols_subset)

        if len(cat_idx_subset) == 0:
            from sklearn.cluster import KMeans
            km = KMeans(
                n_clusters=params.c_min,
                random_state=params.random_state, n_init=10
            )
            labels = km.fit_predict(
                df_subset.select_dtypes(
                    exclude=['object', 'category', 'bool']
                ).values
            )
            current = {
                "algo": "kmeans", "k": params.c_min,
                "labels": labels, "score": np.nan, "score_adj": np.nan,
            }
        else:
            current = auto_select_algo_k(
                X_df=df_subset,
                cat_idx=cat_idx_subset,
                algorithms=algorithms,
                c_range=c_range,
                phase_a_cache=phase_a_cache,   # ← teruskan cache
                hac_mode=hac_mode,
                lambda_weight=lambda_weight,
                lnc_k=getattr(params, 'lnc_k', 20),
                lnc_alpha=getattr(params, 'lnc_alpha', 0.7),
                enable_screening=getattr(params, 'enable_screening', True),
                screening_k_values=getattr(params, 'screening_k_values', (2, 3, 4)),
                random_state=params.random_state,
            )

        current["subset"] = subset
        all_run_history[subset_key] = current

        current_score = current.get("score_adj", -np.inf)
        best_score = (
            best_overall_result.get("score_adj", -np.inf)
            if best_overall_result else -np.inf
        )

        if best_overall_result is None or current_score > best_score:
            best_overall_result = current
            if verbose:
                print(
                    f"    ✨ Best baru! J={current_score:.4f} "
                    f"(L-Sil={current.get('lsil_score', float('nan')):.4f}, "
                    f"LNC*={current.get('lnc_score', float('nan')):.4f}, "
                    f"algo={current['algo']}, K={current['k']})"
                )

    if verbose and best_overall_result:
        print("\n[PHASE B DONE]")

    # ── Structural Control ──
    sc_result = None
    n_samples = len(df_full)
    uses_landmark = (n_samples > getattr(params, 'ss_max_n', 2000))

    if (
        run_structural_control
        and uses_landmark
        and best_overall_result is not None
        and best_overall_result.get("labels") is not None
    ):
        best_subset = best_overall_result.get("subset", [])
        best_labels = np.asarray(best_overall_result["labels"])

        if best_subset and len(np.unique(best_labels)) >= 2:
            df_best = df_full[best_subset]
            cat_cols_best = [c for c in best_subset if c in cat_cols_full]
            sc_result = structural_control_lnc(
                X_df=df_best,
                labels=best_labels,
                cat_cols=cat_cols_best,
                lnc_threshold=lnc_threshold,
                lnc_k=lnc_k,
                random_state=getattr(params, 'random_state', 42),
                verbose=verbose,
            )

    if best_overall_result:
        best_overall_result["all_run_history"] = all_run_history
        best_overall_result["structural_control"] = (
            asdict(sc_result) if sc_result is not None else None
        )
        best_overall_result["cache_used"] = cache_available

    return best_overall_result or {}


# ================================================================
# Wrapper: make_auto_cluster_fn
# ================================================================

def make_auto_cluster_fn(
    algorithms: List[str] = ["kprototypes", "hac_gower"],
    c_range: range = range(2, 10),
    metric: str = "auto",
    penalty_lambda: float = 0.02,
    hac_mode: str = "hybrid",
    lambda_weight: float = 0.6,
    random_state: int = 42
):
    def cluster_fn(
        X_df: pd.DataFrame, cat_idx: List[int],
        _k_unused: int, seed: Optional[int] = None
    ):
        rng = seed if seed is not None else random_state
        best = auto_select_algo_k(
            X_df, cat_idx, algorithms, c_range=c_range,
            phase_a_cache=None,   # no cache in standalone mode
            primary_metric=metric,
            hac_mode=hac_mode,
            lambda_weight=lambda_weight,
            random_state=rng,
            penalty_lambda=penalty_lambda,
        )
        try:
            cluster_fn._last = {
                "algo": best.get("algo"),
                "k": best.get("k"),
                "C": best.get("k"),
                "score": best.get("score"),
                "score_adj": best.get("score_adj"),
                "lsil_score": best.get("lsil_score"),
                "lnc_score": best.get("lnc_score"),
            }
        except Exception:
            pass
        return best["labels"]

    cluster_fn._history = []
    return cluster_fn