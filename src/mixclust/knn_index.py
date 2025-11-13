# src/mixclust/knn_index.py
from __future__ import annotations
import numpy as np

class KNNIndex:
    """
    Indeks kNN di ruang COSINE untuk vektor unit-norm (X_unit).
    - Backend utama: hnswlib (jika tersedia)
    - Fallback: sklearn.neighbors.NearestNeighbors(metric='cosine')

    Catatan:
    - X_unit diasumsikan sudah unit-norm (||x||_2 = 1). Jika tidak, cosine-distance
      tetap bisa dipakai, namun kualitas ANN/HNSW bisa menurun.
    """

    def __init__(
        self,
        X_unit: np.ndarray,
        try_hnsw: bool = True,
        verbose: bool = True,
        ef_query_mul: int = 6,
    ):
        X_unit = np.asarray(X_unit, dtype=np.float32)
        if X_unit.ndim != 2 or X_unit.size == 0:
            raise ValueError("X_unit harus array 2D dengan ukuran > 0.")
        self.X = X_unit
        self.n, self.d = X_unit.shape

        self.backend = "sklearn"
        self.nn = None
        self.hnsw = None
        self.ef_query_mul = max(1, int(ef_query_mul))

        if try_hnsw and self.d > 0 and self.n > 1:
            try:
                import hnswlib  # type: ignore
                self.backend = "hnsw"
                self.hnsw = hnswlib.Index(space="cosine", dim=int(self.d))
                # Parameter sedikit dinaikkan agar lebih robust di data campuran
                self.hnsw.init_index(max_elements=int(self.n), ef_construction=300, M=32)
                self.hnsw.add_items(self.X, np.arange(self.n, dtype=np.int32))
                # ef query awal (akan diubah dinamis per query)
                self.hnsw.set_ef(200)
                if verbose:
                    print("  ✓ HNSW index aktif")
            except Exception as e:
                if verbose:
                    print(f"  ↩︎ Gagal inisialisasi HNSW ({e}); fallback ke Sklearn.")
                self._fallback_to_sklearn(verbose=verbose)
        else:
            self._fallback_to_sklearn(verbose=verbose)

    def _fallback_to_sklearn(self, verbose: bool = True):
        from sklearn.neighbors import NearestNeighbors
        self.backend = "sklearn"
        self.nn = NearestNeighbors(metric="cosine")
        self.nn.fit(self.X)
        self.hnsw = None
        if verbose:
            print("  ✓ Sklearn NearestNeighbors index aktif")

    # ---------------- Core API ----------------
    def kneighbors_batch(self, query_idx: np.ndarray, k: int):
        """
        Kembalikan (idx, dist) untuk setiap query di query_idx:
          - idx: np.ndarray[len(query_idx), k]
          - dist: np.ndarray[len(query_idx), k]
        Guard:
          - k<=0 → array kosong
          - k>n-1 → dipotong otomatis (karena self-neighbor dibuang)
        """
        query_idx = np.asarray(query_idx, dtype=int).ravel()
        if query_idx.size == 0:
            return (
                np.zeros((0, 0), dtype=int),
                np.zeros((0, 0), dtype=float),
            )

        # Pastikan index valid
        if np.any(query_idx < 0) or np.any(query_idx >= self.n):
            raise IndexError("query_idx mengandung indeks di luar [0, n).")

        k = int(k)
        if k <= 0:
            return (
                np.zeros((query_idx.size, 0), dtype=int),
                np.zeros((query_idx.size, 0), dtype=float),
            )

        # Ambil +1 karena akan membuang diri sendiri
        k_eff = min(k + 1, max(1, self.n))
        qX = self.X[query_idx]

        if self.backend == "hnsw":
            try:
                ef = max(200, int(self.ef_query_mul * k_eff))
                self.hnsw.set_ef(ef)
                idxs, dists = self.hnsw.knn_query(qX, k=k_eff)
            except Exception:
                # Fallback otomatis jika HNSW error runtime
                self._fallback_to_sklearn(verbose=True)
                from sklearn.neighbors import NearestNeighbors
                dists, idxs = self.nn.kneighbors(qX, n_neighbors=k_eff, return_distance=True)
        else:
            from sklearn.neighbors import NearestNeighbors
            dists, idxs = self.nn.kneighbors(qX, n_neighbors=k_eff, return_distance=True)

        # Buang self-neighbor dan potong ke k
        out_idx = np.empty((query_idx.size, 0), dtype=int)
        out_dist = np.empty((query_idx.size, 0), dtype=float)
        if k_eff == 0:
            return out_idx, out_dist

        pruned_idx = []
        pruned_dist = []
        for row, qi in enumerate(query_idx):
            ids = list(np.asarray(idxs[row], dtype=int))
            ds = list(np.asarray(dists[row], dtype=float))

            # Hapus diri sendiri jika ada
            try:
                j = ids.index(int(qi))
                ids.pop(j)
                ds.pop(j)
            except ValueError:
                # Tidak ada self-neighbor pada hasil (boleh terjadi di beberapa backend)
                pass

            # Potong ke k
            ids = ids[:k]
            ds = ds[:k]

            # Jika jumlah tetangga < k (mis. n==1), tetap kembalikan apa adanya
            pruned_idx.append(ids if ids else [])
            pruned_dist.append(ds if ds else [])

        # Konversi ke array 2D; jika baris memiliki panjang berbeda, pad dengan -1/np.nan
        max_k = max((len(x) for x in pruned_idx), default=0)
        if max_k == 0:
            return (
                np.zeros((query_idx.size, 0), dtype=int),
                np.zeros((query_idx.size, 0), dtype=float),
            )

        out_idx = -np.ones((query_idx.size, max_k), dtype=int)
        out_dist = np.full((query_idx.size, max_k), np.nan, dtype=float)
        for r in range(query_idx.size):
            L = len(pruned_idx[r])
            if L:
                out_idx[r, :L] = pruned_idx[r]
                out_dist[r, :L] = pruned_dist[r]

        return out_idx, out_dist

    def kneighbors_idx_dist(self, i: int, k: int):
        """
        Versi single-query: kembalikan (idx, dist) untuk 1 indeks i.
        """
        i = int(i)
        if i < 0 or i >= self.n:
            raise IndexError("Index i di luar [0, n).")
        idxs, dists = self.kneighbors_batch(np.array([i], dtype=int), k)
        return idxs[0], dists[0]
