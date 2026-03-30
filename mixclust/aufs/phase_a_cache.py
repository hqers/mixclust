# mixclust/aufs/phase_a_cache.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
import numpy as np
import pandas as pd


@dataclass
class PhaseACache:
    # Gower arrays (full dataset, semua fitur)
    X_num_full: Optional[np.ndarray] = None
    X_cat_full: Optional[np.ndarray] = None
    num_min_full: Optional[np.ndarray] = None
    num_max_full: Optional[np.ndarray] = None
    mask_num_full: Optional[np.ndarray] = None
    mask_cat_full: Optional[np.ndarray] = None
    inv_rng_full: Optional[Any] = None

    # Peta posisi kolom → indeks di X_num_full / X_cat_full
    num_pos: Dict[str, int] = field(default_factory=dict)
    cat_pos: Dict[str, int] = field(default_factory=dict)

    # Landmark dan precomputed clustering dari Phase A
    L_fixed: Optional[np.ndarray] = None
    labels0: Optional[np.ndarray] = None
    protos0: Optional[Dict[int, List[int]]] = None

    # ── KNNIndex prebuilt — dibangun SEKALI ──
    knn_index: Optional[Any] = None
    X_unit_full: Optional[np.ndarray] = None

    # ── Phase B subsample — evaluasi pada subset rows untuk kecepatan ──
    # Dibangun sekali di _extract_phase_a_cache, dipakai ulang per trial
    _pb_idx: Optional[np.ndarray] = None     # indeks rows subsample
    _pb_X_num: Optional[np.ndarray] = None
    _pb_X_cat: Optional[np.ndarray] = None
    _pb_num_min: Optional[np.ndarray] = None
    _pb_num_max: Optional[np.ndarray] = None
    _pb_inv_rng: Optional[Any] = None
    _pb_L: Optional[np.ndarray] = None       # landmark indices relatif ke subsample
    _pb_n: int = 0
    _pb_available: bool = False

    # Meta
    n_landmarks: int = 0
    n_samples: int = 0
    available: bool = False

    def make_masks_for_subset(self, cols: List[str]):
        if self.X_num_full is None:
            return None, None
        mnum = None
        if self.X_num_full.shape[1] > 0:
            mnum = np.zeros(self.X_num_full.shape[1], dtype=bool)
            idxs = [self.num_pos[c] for c in cols if c in self.num_pos]
            if idxs:
                mnum[np.array(idxs)] = True
        mcat = None
        if self.X_cat_full.shape[1] > 0:
            mcat = np.zeros(self.X_cat_full.shape[1], dtype=bool)
            idxs = [self.cat_pos[c] for c in cols if c in self.cat_pos]
            if idxs:
                mcat[np.array(idxs)] = True
        return mnum, mcat

    def build_phase_b_subsample(self, phase_b_eval_n: int = 30_000,
                                 random_state: int = 42):
        """
        Bangun subsample untuk evaluasi Phase B.
        Landmarks di-remap ke indeks relatif subsample.
        Dipanggil SEKALI di _extract_phase_a_cache.
        """
        if not self.available or self.n_samples <= phase_b_eval_n:
            # Data cukup kecil — pakai full
            self._pb_available = False
            return

        rng = np.random.default_rng(random_state + 9999)
        n = self.n_samples

        # Stratified subsample berdasarkan labels0
        idx_list = []
        uniq, counts = np.unique(self.labels0, return_counts=True)
        for c, cnt in zip(uniq, counts):
            take = max(3, int(round(phase_b_eval_n * cnt / n)))
            pool = np.where(self.labels0 == c)[0]
            take = min(take, len(pool))
            idx_list.append(rng.choice(pool, size=take, replace=False))
        idx_pb = np.unique(np.concatenate(idx_list))
        if len(idx_pb) > phase_b_eval_n:
            idx_pb = rng.choice(idx_pb, size=phase_b_eval_n, replace=False)
        idx_pb = np.sort(idx_pb)

        self._pb_idx = idx_pb
        self._pb_n = len(idx_pb)
        self._pb_X_num = self.X_num_full[idx_pb] if self.X_num_full is not None else None
        self._pb_X_cat = self.X_cat_full[idx_pb] if self.X_cat_full is not None else None
        self._pb_num_min = self.num_min_full
        self._pb_num_max = self.num_max_full
        self._pb_inv_rng = self.inv_rng_full

        # Remap landmarks: cari indeks L_fixed yang ada di idx_pb
        # dan convert ke posisi relatif di subsample
        if self.L_fixed is not None:
            pb_set = set(idx_pb.tolist())
            # Landmark yang ada di subsample
            lm_in_pb = [l for l in self.L_fixed if l in pb_set]
            if len(lm_in_pb) >= 6:  # minimal landmark
                # Map absolute → relative index
                abs_to_rel = {abs_idx: rel_idx for rel_idx, abs_idx in enumerate(idx_pb)}
                self._pb_L = np.array([abs_to_rel[l] for l in lm_in_pb], dtype=int)
            else:
                # Terlalu sedikit landmark overlap — buat landmark baru dari subsample
                from ..core.adaptive import adaptive_landmark_count
                K = len(np.unique(self.labels0))
                m = adaptive_landmark_count(self._pb_n, K=K, c=2.0, cap_frac=0.2)
                labels_pb = self.labels0[idx_pb]
                # Stratified landmarks pada subsample
                L_list = []
                vals, cnts = np.unique(labels_pb, return_counts=True)
                for ci, cnt in zip(vals, cnts):
                    pool_rel = np.where(labels_pb == ci)[0]
                    take = max(3, int(round(m * cnt / self._pb_n)))
                    take = min(take, len(pool_rel))
                    L_list.extend(rng.choice(pool_rel, size=take, replace=False).tolist())
                self._pb_L = np.array(sorted(set(L_list)), dtype=int)[:m]

        self._pb_available = self._pb_L is not None and len(self._pb_L) >= 6
        if self._pb_available:
            print(f"[CACHE] Phase B subsample: n={self._pb_n:,}, |L_pb|={len(self._pb_L)}")


def _extract_phase_a_cache(
    reward_fn: Callable,
    df: pd.DataFrame,
    phase_b_eval_n: int = 30_000,
) -> PhaseACache:
    cache = PhaseACache()

    if not hasattr(reward_fn, '__phase_a_cache__'):
        return cache

    src = reward_fn.__phase_a_cache__
    cache.X_num_full = src.get('X_num_full')
    cache.X_cat_full = src.get('X_cat_full')
    cache.num_min_full = src.get('num_min_full')
    cache.num_max_full = src.get('num_max_full')
    cache.mask_num_full = src.get('mask_num_full')
    cache.mask_cat_full = src.get('mask_cat_full')
    cache.inv_rng_full = src.get('inv_rng_full')
    cache.num_pos = src.get('num_pos', {})
    cache.cat_pos = src.get('cat_pos', {})
    cache.L_fixed = src.get('L_fixed')
    cache.labels0 = src.get('labels0')
    cache.protos0 = src.get('protos0')
    cache.n_samples = src.get('n_samples', len(df))
    cache.n_landmarks = len(cache.L_fixed) if cache.L_fixed is not None else 0
    cache.available = (
        cache.X_num_full is not None
        and cache.L_fixed is not None
        and cache.labels0 is not None
    )

    # KNNIndex prebuilt
    if cache.available and cache.X_num_full.shape[1] > 0:
        try:
            from sklearn.preprocessing import normalize
            from ..core.knn_index import KNNIndex
            X_unit = normalize(cache.X_num_full, norm="l2")
            cache.X_unit_full = X_unit
            cache.knn_index = KNNIndex(X_unit, try_hnsw=True, verbose=False)
            print(f"[CACHE] KNNIndex prebuilt: n={cache.n_samples}, shape={X_unit.shape}")
        except Exception as e:
            print(f"[CACHE] KNNIndex prebuilt gagal (LNC* akan di-skip): {e}")
            cache.knn_index = None
            cache.X_unit_full = None

    # ── Phase B subsample — evaluasi L-Sil pada subset rows ──
    if cache.available:
        cache.build_phase_b_subsample(
            phase_b_eval_n=phase_b_eval_n,
            random_state=src.get('random_state', 42),
        )

    return cache