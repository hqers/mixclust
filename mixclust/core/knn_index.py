# mixclust/core/knn_index.py
from __future__ import annotations
import numpy as np

class KNNIndex:
    """
    Backend kNN:
      - Jika hnswlib tersedia dan try_hnsw=True → HNSW (cepat)
      - Jika tidak → sklearn NearestNeighbors (metric='cosine')
    Menyediakan:
      - kneighbors_batch(query_idx, k) -> (idxs, dists) shape (Q, k)
      - kneighbors_idx_dist(i, k)      -> (idxs, dists) shape (k,)
    """
    def __init__(self, X_unit: np.ndarray, try_hnsw: bool = True, verbose: bool = True):
        self.X = X_unit
        self.n, self.d = X_unit.shape
        self.backend = 'sklearn'
        self.nn = None
        self.hnsw = None

        if try_hnsw:
            try:
                import hnswlib
                self.backend = 'hnsw'
                self.hnsw = hnswlib.Index(space='cosine', dim=self.d)
                self.hnsw.init_index(max_elements=self.n, ef_construction=200, M=32)
                self.hnsw.add_items(self.X, np.arange(self.n))
                self.hnsw.set_ef(100)
                if verbose:
                    print("  ✓ HNSW index aktif")
            except Exception:
                self.backend = 'sklearn'

        if self.backend == 'sklearn':
            if verbose:
                print("  ✓ Sklearn NearestNeighbors index aktif")
            from sklearn.neighbors import NearestNeighbors
            self.nn = NearestNeighbors(metric='cosine').fit(self.X)

    def kneighbors_batch(self, query_idx: np.ndarray, k: int):
        """Batch query: k tetangga utk banyak indeks sekaligus."""
        query_idx = np.asarray(query_idx, dtype=int)   # ← tambahkan
        qX = self.X[query_idx]
        k_eff = int(min(int(k) + 1, int(self.n)))   # ← CAST WAJIB

        if self.backend == 'hnsw':
            ef = int(max(100, 2 * k_eff))          # ← CAST WAJIB
            self.hnsw.set_ef(ef)
            idxs, dists = self.hnsw.knn_query(qX, k=k_eff)  # k=INT
        else:
            from sklearn.neighbors import NearestNeighbors
            dists, idxs = self.nn.kneighbors(qX, n_neighbors=k_eff, return_distance=True)

        out_idx, out_dist = [], []
        for r, qi in enumerate(query_idx):
            ids = idxs[r].tolist()
            ds  = dists[r].tolist()
            if int(qi) in ids:
                j = ids.index(int(qi))
                ids.pop(j); ds.pop(j)
            out_idx.append(ids[:int(k)])           # ← pastikan slicing pakai int(k)
            out_dist.append(ds[:int(k)])
        return np.asarray(out_idx, dtype=int), np.asarray(out_dist, dtype=float)

    def kneighbors_idx_dist(self, i: int, k: int):
        """Single query; wrapper ke batch agar kompatibel dgn kode lama."""
        k = int(k)                                  # ← CAST WAJIB
        idxs, dists = self.kneighbors_batch(np.array([int(i)], dtype=int), k)
        return idxs[0], dists[0]
