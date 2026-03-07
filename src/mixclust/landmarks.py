# src/mixclust/landmarks.py
from typing import Callable, Optional, Tuple, Dict, List, Any, Iterable
import numpy as np
from sklearn.neighbors import NearestNeighbors

def select_landmarks_kcenter(X_unit, m, seed=42, verbose=False): 
    """
    K-center greedy di ruang cosine (unit-norm).
    Kompleksitas ~ O(m * n * d) via produk vektor (tanpa matriks jarak penuh).
    """
    n, d = X_unit.shape
    rng = np.random.RandomState(seed)
    first = rng.randint(0, n)
    L = [first]
    min_dist = 1.0 - (X_unit @ X_unit[first].reshape(-1,1)).squeeze()

    while len(L) < min(m, n):
        nxt = int(np.argmax(min_dist))
        L.append(nxt)
        d2 = 1.0 - (X_unit @ X_unit[nxt].reshape(-1,1)).squeeze()
        np.minimum(min_dist, d2, out=min_dist)
        if verbose and (len(L) % max(1, m//5) == 0):
            print(f"  • K-center: {len(L)}/{m}")
    return L

def _farthest_first_from_pool(X_unit, pool_idx, k, seed=42): 
    if k <= 0 or len(pool_idx) == 0:
        return []
    rng = np.random.RandomState(seed)
    chosen = [pool_idx[rng.randint(0, len(pool_idx))]]
    if k == 1: return chosen
    mins = 1.0 - (X_unit[pool_idx] @ X_unit[chosen[0]].reshape(-1,1)).squeeze()
    while len(chosen) < min(k, len(pool_idx)):
        nxt = pool_idx[int(np.argmax(mins))]
        chosen.append(nxt)
        d2 = 1.0 - (X_unit[pool_idx] @ X_unit[nxt].reshape(-1,1)).squeeze()
        mins = np.minimum(mins, d2)
    return chosen
    
def select_landmarks_cluster_aware(X_unit, labels, m, central_frac=0.8, boundary_frac=0.2, per_cluster_min=3, seed=42): 
    """
    Landmark per-klaster: mayoritas 'central' (dekat centroid), sebagian 'boundary' (jauh),
    dengan farthest-first di dalam pool agar tidak duplikatif.
    """
    labels = np.asarray(labels)
    n = len(labels)
    uniq, counts = np.unique(labels, return_counts=True)
    # kuota per klaster proporsional ukuran + per_cluster_min
    quota = {c: max(per_cluster_min, int(round(m * (counts[i]/n)))) for i, c in enumerate(uniq)}
    L = []

    for c in uniq:
        idx = np.where(labels == c)[0]
        centroid = X_unit[idx].mean(axis=0)
        centroid /= (np.linalg.norm(centroid) + 1e-12)
        sims = X_unit[idx] @ centroid
        ord_central  = idx[np.argsort(-sims)]  # dekat centroid
        ord_boundary = idx[np.argsort(sims)]   # jauh (boundary)

        m_c = quota[c]
        take_central  = int(round(central_frac  * m_c))
        take_boundary = max(0, m_c - take_central)

        # ambil pool lebih besar lalu farthest-first
        pool_central  = ord_central[: max(take_central*5,  take_central+5)]
        pool_boundary = ord_boundary[: max(take_boundary*5, take_boundary+5)]

        L += _farthest_first_from_pool(X_unit, pool_central,  take_central,  seed=seed)
        L += _farthest_first_from_pool(X_unit, pool_boundary, take_boundary, seed=seed)

    # rapikan jumlah
    if len(L) > m: L = L[:m]
    if len(L) < m:
        rest = list(set(range(n)) - set(L))
        L += rest[: m - len(L)]

    return sorted(L)

def mini_pam_refine(
    L: np.ndarray,
    cluster_labels: np.ndarray,
    D_nL: np.ndarray,
    get_dist_col: Callable[[int], np.ndarray],
    pool_indices: Optional[np.ndarray] = None,
    per_cluster_pool: int = 50,
    max_iter: int = 2,
    max_swaps: Optional[int] = None,
    early_stop_eps: float = 1e-4,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Lightweight medoid-style refinement over an existing landmark set.

    Parameters
    ----------
    L : np.ndarray, shape (|L|,)
        Global row indices of current landmarks.
    cluster_labels : np.ndarray, shape (n,)
        Cluster label for each data point (and for each landmark via L).
    D_nL : np.ndarray, shape (n, |L|)
        Cached distances from every point to current landmarks (columns aligned with L order).
    get_dist_col : callable
        Function that returns a distance column for a *candidate* point index 'u':
        get_dist_col(u) -> np.ndarray of shape (n,), distances from all points to candidate u.
        Implement this using your existing Gower distance routine (or a cache).
    pool_indices : np.ndarray, optional
        Global indices considered as potential replacements. If None, a small per-cluster
        reservoir is auto-sampled from non-landmark points.
    per_cluster_pool : int, default=50
        When pool_indices is None, how many candidates to sample per cluster (approx.).
    max_iter : int, default=2
        Number of refinement passes over the landmark set (1–2 is usually enough).
    max_swaps : int, optional
        Maximum number of accepted swaps per iteration; if None, up to |L|.
    early_stop_eps : float, default=1e-4
        Stop if the relative improvement in objective is <= early_stop_eps.
    random_state : int, optional
        Reproducibility for auto-sampling.

    Returns
    -------
    result : dict
        {
          "L_refined": np.ndarray (|L|,), new landmark indices (global),
          "D_nL_refined": np.ndarray (n, |L|), updated distance cache,
          "improvement": float, absolute decrease in objective,
          "rel_improvement": float, relative decrease,
          "n_swaps": int, total accepted swaps,
          "n_iters": int, iterations performed,
          "objective_before": float,
          "objective_after": float,
        }

    Notes
    -----
    - Objective minimized: mean of within-cluster distances to the nearest landmark
      from the same cluster (stabilizes the a-tilde component of Silhouette).
    - Swaps are constrained within the same cluster to preserve per-cluster coverage.
    - This function never computes pairwise point-point distances; it reuses D_nL
      and fetches distance columns for candidates via get_dist_col.
    """

    rng = np.random.default_rng(random_state)

    # Internal helpers
    def objective_mean_intra(D_nL_cur: np.ndarray, L_cur: np.ndarray) -> float:
        """Mean nearest-landmark distance within the same cluster."""
        labels = cluster_labels
        unique_c = np.unique(labels)
        total = 0.0
        count = 0
        # Build mapping cluster -> landmark cols
        lm_cols_by_c = {c: np.where(labels[L_cur] == c)[0] for c in unique_c}
        for c in unique_c:
            rows = np.where(labels == c)[0]
            cols = lm_cols_by_c[c]
            if cols.size == 0 or rows.size == 0:
                continue
            # nearest landmark within the same cluster
            dmin = np.min(D_nL_cur[np.ix_(rows, cols)], axis=1)
            total += float(np.sum(dmin))
            count += int(rows.size)
        return total / max(count, 1)

    def build_auto_pool(L_cur: np.ndarray) -> np.ndarray:
        """Auto-sample candidates per cluster excluding current landmarks."""
        n = cluster_labels.shape[0]
        mask_landmark = np.zeros(n, dtype=bool)
        mask_landmark[L_cur] = True
        candidates = []
        for c in np.unique(cluster_labels):
            pool_mask = (cluster_labels == c) & (~mask_landmark)
            pool_idx = np.where(pool_mask)[0]
            if pool_idx.size == 0:
                continue
            take = min(per_cluster_pool, pool_idx.size)
            choose = rng.choice(pool_idx, size=take, replace=False)
            candidates.append(choose)
        return np.unique(np.concatenate(candidates)) if len(candidates) else np.array([], dtype=int)

    # Initialization
    L_cur = np.array(L, copy=True)
    D_cur = np.array(D_nL, copy=True)
    n, Lk = D_cur.shape
    if Lk != L_cur.size:
        raise ValueError("D_nL columns must align with L indices (same |L|).")

    # Determine candidate pool
    if pool_indices is None:
        pool = build_auto_pool(L_cur)
    else:
        pool = np.array(pool_indices, copy=False).astype(int)
        # Ensure no duplicates with L
        pool = pool[~np.isin(pool, L_cur)]

    # Precompute cluster of each landmark col (for fast membership)
    lm_cluster = cluster_labels[L_cur]
    obj_before = objective_mean_intra(D_cur, L_cur)

    total_swaps = 0
    max_swaps = L_cur.size if max_swaps is None else max_swaps
    it = 0

    while it < max_iter and total_swaps < max_swaps:
        it += 1
        iter_swaps = 0
        improved = False

        # For each cluster, try beneficial swaps
        for c in np.unique(cluster_labels):
            # columns of landmarks belonging to cluster c
            col_c = np.where(lm_cluster == c)[0]
            if col_c.size == 0:
                continue

            # candidate pool rows belonging to the same cluster c
            pool_c = pool[cluster_labels[pool] == c]
            if pool_c.size == 0:
                continue

            # Rows (points) in cluster c for objective evaluation
            rows_c = np.where(cluster_labels == c)[0]
            if rows_c.size == 0:
                continue

            # Current per-point nearest distance within cluster c
            cur_block = D_cur[np.ix_(rows_c, col_c)]
            cur_nearest = np.min(cur_block, axis=1)

            # Try, for each landmark column j in cluster c, the best candidate u in pool_c
            for j_local, j_col in enumerate(col_c):
                if total_swaps >= max_swaps:
                    break

                # Distances to all points for best candidate (cache-as-you-go)
                best_u = None
                best_gain = 0.0
                best_col_vec = None

                for u in pool_c:
                    d_u = get_dist_col(u)  # shape (n,)
                    # Replace column j_col with candidate u, evaluate on rows_c
                    # New nearest = min( current other cols, candidate column )
                    if col_c.size > 1:
                        other_cols = np.delete(col_c, j_local)
                        other_block = D_cur[np.ix_(rows_c, other_cols)]
                        other_nearest = np.min(other_block, axis=1)
                        new_nearest = np.minimum(other_nearest, d_u[rows_c])
                    else:
                        new_nearest = d_u[rows_c]

                    # Gain = (old sum - new sum)
                    gain = float(np.sum(cur_nearest) - np.sum(new_nearest))
                    if gain > best_gain:
                        best_gain = gain
                        best_u = u
                        best_col_vec = d_u  # keep full vector for insertion

                # Accept the best swap for this column if it improves
                if best_u is not None and best_gain > 0:
                    # Update L_cur, D_cur column, cluster tag
                    L_cur[j_col] = best_u
                    D_cur[:, j_col] = best_col_vec
                    lm_cluster[j_col] = c

                    # Update current nearest for cluster c to reflect the accepted swap
                    cur_block = D_cur[np.ix_(rows_c, col_c)]
                    cur_nearest = np.min(cur_block, axis=1)

                    # Remove candidate from pool_c to avoid reusing the same one repeatedly
                    pool_c = pool_c[pool_c != best_u]
                    iter_swaps += 1
                    total_swaps += 1
                    improved = True

                    if total_swaps >= max_swaps:
                        break

            if total_swaps >= max_swaps:
                break

        # Check relative improvement for early stop
        obj_now = objective_mean_intra(D_cur, L_cur)
        rel_imp = (obj_before - obj_now) / max(abs(obj_before), 1e-12)
        if not improved or rel_imp <= early_stop_eps:
            obj_before = obj_now
            break
        obj_before = obj_now

    result = {
        "L_refined": L_cur,
        "D_nL_refined": D_cur,
        "improvement": float(objective_mean_intra(D_nL, L) - objective_mean_intra(D_cur, L_cur)),
        "rel_improvement": float(
            (objective_mean_intra(D_nL, L) - objective_mean_intra(D_cur, L_cur))
            / max(abs(objective_mean_intra(D_nL, L)), 1e-12)
        ),
        "n_swaps": total_swaps,
        "n_iters": it,
        "objective_before": float(objective_mean_intra(D_nL, L)),
        "objective_after": float(objective_mean_intra(D_cur, L_cur)),
    }
    return result


def build_candidate_pool(
    L: np.ndarray,
    cluster_labels: np.ndarray,
    boundary_scores: Optional[np.ndarray] = None,
    per_cluster: int = 50,
    include_boundary_ratio: float = 0.5,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Build a small per-cluster candidate pool for mini-PAM refinement.

    Parameters
    ----------
    L : np.ndarray
        Current landmark indices.
    cluster_labels : np.ndarray
        Cluster labels for all points.
    boundary_scores : np.ndarray, optional
        Score per point indicating boundary-ness (higher = nearer to decision margins).
        If provided, top-k by score will be prioritized.
    per_cluster : int
        Target candidates per cluster.
    include_boundary_ratio : float
        Fraction of per-cluster quota reserved for top boundary points (if available).
    random_state : int, optional
        Reproducibility.

    Returns
    -------
    np.ndarray
        Global indices of candidate points (excluding current landmarks).
    """
    rng = np.random.default_rng(random_state)
    n = cluster_labels.shape[0]
    mask_landmark = np.zeros(n, dtype=bool)
    mask_landmark[L] = True

    pool = []
    for c in np.unique(cluster_labels):
        idx = np.where((cluster_labels == c) & (~mask_landmark))[0]
        if idx.size == 0:
            continue

        k_boundary = int(round(per_cluster * include_boundary_ratio))
        k_random = max(per_cluster - k_boundary, 0)

        chosen = []
        if boundary_scores is not None and k_boundary > 0:
            # top-k by boundary score inside cluster c
            top = idx[np.argsort(-boundary_scores[idx])[: min(k_boundary, idx.size)]]
            chosen.append(top)

            # remove chosen from pool for the random part
            remaining_mask = np.ones(idx.size, dtype=bool)
            remaining_mask[np.isin(idx, top)] = False
            idx_remaining = idx[remaining_mask]
        else:
            idx_remaining = idx

        if k_random > 0 and idx_remaining.size > 0:
            rnd = rng.choice(idx_remaining, size=min(k_random, idx_remaining.size), replace=False)
            chosen.append(rnd)

        if chosen:
            pool.append(np.unique(np.concatenate(chosen)))

    return np.unique(np.concatenate(pool)) if pool else np.array([], dtype=int)
# ------------------------------
# util: subsample clustering + propagate + cluster-aware landmarks (uses existing funcs)
# ------------------------------

def subsample_and_propagate_labels(
    df_full: "pd.DataFrame",
    cat_cols_full: List[str],
    cluster_fn: Callable,
    n_clusters: int,
    random_state: int,
    subsample_n: Optional[int] = 20000,
    proto_sample_cap: int = 200,
    per_cluster_proto: int = 1
) -> Tuple[np.ndarray, Dict[int, List[int]], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    1) If n > subsample_n: cluster on random subsample -> labels_sub
    2) Propagate labels_sub to full rows via NN on numeric features (fast)
    3) Build prototypes on full using labels0 (using existing build_prototypes_by_cluster_gower)
    Returns: (labels0, protos0_dict, idx_sub, labels_sub)
    """
    from mixclust.prototypes import build_prototypes_by_cluster_gower
    from mixclust.aufs_samba.preprocess import prepare_mixed_arrays_no_label

    n = len(df_full)
    rng = np.random.default_rng(random_state)

    Xn_full, Xc_full, nmin_full, nmax_full, _, _, inv_full = prepare_mixed_arrays_no_label(df_full)

    idx_sub = None
    labels_sub = None

    if subsample_n is None or n <= subsample_n:
        # small dataset: cluster full
        cat_idx_full = [df_full.columns.get_loc(c) for c in cat_cols_full]
        labels0 = cluster_fn(df_full, cat_idx_full, n_clusters, random_state)
    else:
        idx_sub = rng.choice(n, size=subsample_n, replace=False)
        df_sub = df_full.iloc[idx_sub].reset_index(drop=True)
        cat_idx_sub = [df_sub.columns.get_loc(c) for c in df_sub.columns if c in cat_cols_full]
        labels_sub = cluster_fn(df_sub, cat_idx_sub, n_clusters, random_state)
        Xn_sub, Xc_sub, _, _, _, _, _ = prepare_mixed_arrays_no_label(df_sub)

        # propagate using numeric NN if numeric exists
        if Xn_sub.shape[1] > 0 and Xn_full.shape[1] > 0:
            nn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(Xn_sub)
            _, idx_nn = nn.kneighbors(Xn_full, return_distance=True)
            labels0 = labels_sub[idx_nn.ravel()]
        else:
            # fallback: string-signature matching
            sig_sub = df_sub.astype(str).agg('-'.join, axis=1).values
            sig_full = df_full.astype(str).agg('-'.join, axis=1).values
            map_idx = {s:i for i,s in enumerate(sig_sub)}
            labels0 = np.array([ labels_sub[map_idx.get(s, 0)] for s in sig_full ], dtype=int)

    # build prototypes on full arrays
    try:
        protos0 = build_prototypes_by_cluster_gower(
            labels0, Xn_full, Xc_full, nmin_full, nmax_full,
            per_cluster=per_cluster_proto, sample_cap=proto_sample_cap,
            seed=random_state, feature_mask_num=None, feature_mask_cat=None, inv_rng=inv_full
        )
    except Exception:
        protos0 = {}

    return np.asarray(labels0, dtype=int), protos0, idx_sub, labels_sub
    
def cluster_aware_landmarks_on_subsample(
    df_full: "pd.DataFrame",
    idx_sub: Optional[np.ndarray],
    labels_sub: Optional[np.ndarray],
    labels_full: np.ndarray,
    m_cap: int,
    per_cluster_min: int,
    random_state: int,
    select_landmarks_fn: Optional[Callable] = None
) -> np.ndarray:
    """
    Compute cluster-aware landmarks on subsample (if provided) using existing select_landmarks_cluster_aware,
    then map selected subsample indices to full indices. Fallback to stratified selection on full if necessary.
    """
    rng = np.random.default_rng(random_state)
    # try subsample path
    if idx_sub is not None and labels_sub is not None and select_landmarks_fn is not None:
        try:
            df_sub = df_full.iloc[idx_sub].reset_index(drop=True)
            # attempt build_features only for subsample (cheaper)
            from mixclust.features import build_features
            X_unit_sub, _, _ = build_features(df_sub, label_col=None, scaler_type="standard", unit_norm=True)
            sel_sub = select_landmarks_fn(X_unit_sub, labels_sub, m_cap,
                                          central_frac=0.8, boundary_frac=0.2, per_cluster_min=per_cluster_min, seed=int(rng.integers(1<<30)))
            sel_full = np.array([ int(idx_sub[i]) for i in sel_sub ], dtype=int)
            return sel_full[:m_cap]
        except Exception:
            pass

    # fallback: stratified on full labels
    return stratified_landmarks(labels_full, m_cap, per_cluster_min, rng)
