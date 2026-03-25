# mixclust/core/adaptive.py — v2.0 (c√n law, Theorem 1 JDSA paper)
import numpy as np


def adaptive_landmark_count(
    n: int,
    K: int = 3,
    c: float = 3.0,
    per_cluster_min: int = 3,
    cap_frac: float = 0.2,
    # legacy params — diterima tapi diabaikan (backward compat caller lama)
    lm_max_frac: float = None,
    lm_per_cluster: int = None,
    labels=None,
    n_total: int = None,
) -> int:
    """
    |L| = max(K * per_cluster_min, min(c * sqrt(n), cap_frac * n, n))

    Theorem 1 (Pratama et al. 2026 JDSA):
      Total complexity O(c * p * n^{3/2}) — subquadratic untuk semua n.

    Default c=3.0 sesuai paper (range c ∈ [1.5, 4]).

    Backward compat: parameter lama (lm_max_frac, lm_per_cluster, labels, n_total)
    diterima tapi diabaikan — caller lama tidak perlu diubah sekaligus.
    """
    # Handle caller lama yang pass labels + n_total sebagai positional
    if isinstance(n, np.ndarray) or (hasattr(n, '__len__') and not isinstance(n, int)):
        # n adalah 'labels' dari caller lama: adaptive_landmark_count(labels, n_total, ...)
        labels_arg = n
        n = int(n_total) if n_total is not None else len(labels_arg)
        K_from_labels = int(len(np.unique(np.asarray(labels_arg))))
        K = K_from_labels

    n = int(n)
    K = max(2, int(K))
    m_sqrt = int(c * np.sqrt(n))
    m_cap  = int(cap_frac * n)
    m_floor = K * per_cluster_min
    result = max(m_floor, min(m_sqrt, m_cap, n))
    return int(result)
