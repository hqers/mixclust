# src/mixclust/adaptive.py
import numpy as np, math, pandas as pd
def adaptive_landmark_count(labels, n_total, lm_max_frac=0.2, lm_per_cluster=5): 
    """
    m adaptif berdasar entropi distribusi klaster + batas 20% dari n.
    """
    labels = np.asarray(labels)
    counts = pd.Series(labels).value_counts()
    p = (counts / n_total).values
    H = -(p * np.log2(p + 1e-12)).sum()
    n_clusters = len(counts)
    m_cap = int(lm_max_frac * n_total)
    denom = max(1e-12, math.log2(max(2, n_clusters)))
    m = max(lm_per_cluster * n_clusters,
            min(int(n_total * H / denom), m_cap, n_total))
    return int(m), float(H), int(n_clusters)
