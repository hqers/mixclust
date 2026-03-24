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

    # ── BARU: KNNIndex prebuilt — dibangun SEKALI, dipakai ulang setiap trial ──
    # Membangun KNNIndex(n=32k) per trial adalah penyebab Phase B lambat
    knn_index: Optional[Any] = None        # KNNIndex object
    X_unit_full: Optional[np.ndarray] = None  # normalized X_num untuk KNN

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


def _extract_phase_a_cache(
    reward_fn: Callable,
    df: pd.DataFrame,
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

    # ── BARU: Bangun KNNIndex SEKALI di sini ──
    # Bukan per trial di _eval_with_phase_a_cache
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

    return cache