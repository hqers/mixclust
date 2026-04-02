"""
KAMILA — KAy-means for MIxed LArge data sets
==============================================
Python re-implementation based on the R package by
Foss & Markatou (JSS 2018, doi:10.18637/jss.v083.i13).

Semiparametric clustering for mixed-type data:
- Continuous: radial kernel density estimation (flexible, non-Gaussian)
- Categorical: multinomial model with kernel-smoothed probabilities
- Balances contribution of both types without manual weighting

Usage
-----
    from kamila import KAMILA
    model = KAMILA(n_clusters=3, n_init=10, random_state=42)
    labels = model.fit_predict(df, num_cols=["x1","x2"], cat_cols=["color"])
"""

from __future__ import annotations
import warnings
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

# ---------------------------------------------------------------------------
# Radial KDE (core of KAMILA's continuous component)
# ---------------------------------------------------------------------------

def _bw_nrd0(x: np.ndarray) -> float:
    """Bandwidth selection matching R's bw.nrd0 (Silverman's rule of thumb)."""
    n = len(x)
    if n < 2:
        return 1.0
    hi = np.std(x, ddof=1)
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    lo = min(hi, iqr / 1.34)
    if lo == 0:
        lo = hi if hi > 0 else abs(np.mean(x)) if np.mean(x) != 0 else 1.0
    return 0.9 * lo * n ** (-0.2)


def _bkde(x: np.ndarray, bandwidth: float, grid_size: int = 401,
          range_x: Tuple[float, float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Simple binned KDE approximation (like KernSmooth::bkde)."""
    if range_x is None:
        range_x = (min(x), max(x))
    grid = np.linspace(range_x[0], range_x[1], grid_size)
    # Use scipy gaussian_kde with specified bandwidth
    if len(np.unique(x)) < 2:
        return grid, np.ones(grid_size) / (range_x[1] - range_x[0] + 1e-10)
    try:
        kde = gaussian_kde(x, bw_method=bandwidth / np.std(x, ddof=1) if np.std(x, ddof=1) > 0 else 1.0)
        density = kde(grid)
    except Exception:
        density = np.ones(grid_size) / (range_x[1] - range_x[0] + 1e-10)
    return grid, density


def radial_kde(radii: np.ndarray, eval_points: np.ndarray, pdim: int) -> np.ndarray:
    """Estimate density of radial distances, then evaluate at given points.
    
    This is the key function that transforms Euclidean distances into
    density-based log-likelihoods for the continuous component.
    """
    MAXDENS = 1.0
    
    if len(radii) < 2 or np.std(radii) == 0:
        return np.ones(len(eval_points)) * 0.01
    
    bw = _bw_nrd0(radii)
    max_eval = max(np.max(eval_points), np.max(radii))
    grid_x, grid_y = _bkde(radii, bandwidth=bw, range_x=(0, max_eval))
    
    # Remove zero/negative density estimates
    nonneg = grid_y > 0
    if not np.all(nonneg):
        min_pos = np.min(grid_y[nonneg]) if np.any(nonneg) else 1e-10
        grid_y[~nonneg] = min_pos / 100
    
    # At bottom 5th percentile, replace with line through origin
    quant05 = np.percentile(grid_x, 5)
    coords_lt = grid_x < quant05
    if np.any(coords_lt):
        max_pt = np.max(np.where(coords_lt))
        if grid_x[max_pt] > 0:
            slope = grid_y[max_pt] / grid_x[max_pt]
            grid_y[coords_lt] = grid_x[coords_lt] * slope
    
    # Radial Jacobian transformation: f(r) / r^(p-1)
    rad_y = np.zeros_like(grid_y)
    rad_y[0] = 0
    with np.errstate(divide='ignore', invalid='ignore'):
        rad_y[1:] = grid_y[1:] / (grid_x[1:] ** (pdim - 1))
    rad_y = np.nan_to_num(rad_y, nan=0.0, posinf=MAXDENS)
    
    # Cap at MAXDENS
    rad_y[rad_y > MAXDENS] = MAXDENS
    
    # Normalize to area 1
    bin_width = grid_x[1] - grid_x[0] if len(grid_x) > 1 else 1.0
    total = bin_width * np.sum(rad_y)
    if total > 0:
        dens_r = rad_y / total
    else:
        dens_r = np.ones_like(rad_y) / len(rad_y)
    
    # Interpolate at eval_points
    kdes = np.interp(eval_points, grid_x, dens_r, left=dens_r[0], right=dens_r[-1])
    
    # Floor at small positive value
    kdes = np.maximum(kdes, 1e-300)
    
    return kdes


# ---------------------------------------------------------------------------
# Categorical kernel smoothing
# ---------------------------------------------------------------------------

def _smooth_joint_table(tab: np.ndarray, bw: float) -> np.ndarray:
    """Smooth a 2D contingency table (cluster x level) with categorical kernel."""
    k, l = tab.shape
    if bw == 0:
        return tab.astype(float)
    
    # Smooth along cluster dimension (rows)
    col_sums = tab.sum(axis=0)
    mid = np.zeros_like(tab, dtype=float)
    for i in range(k):
        for j in range(l):
            off_counts = col_sums[j] - tab[i, j]
            mid[i, j] = (1 - bw) * tab[i, j] + bw / max(k - 1, 1) * off_counts
    
    # Smooth along level dimension (columns)
    row_sums = mid.sum(axis=1)
    out = np.zeros_like(mid)
    for i in range(k):
        for j in range(l):
            off_counts = row_sums[i] - mid[i, j]
            out[i, j] = (1 - bw) * mid[i, j] + bw / max(l - 1, 1) * off_counts
    
    return out


# ---------------------------------------------------------------------------
# Core distance functions
# ---------------------------------------------------------------------------

def _weighted_euclidean_distances(pts: np.ndarray, means: np.ndarray,
                                   weights: np.ndarray) -> np.ndarray:
    """Compute weighted Euclidean distances from n points to k means.
    Returns (n, k) matrix."""
    n, p = pts.shape
    k = means.shape[0]
    dists = np.zeros((n, k))
    for j in range(k):
        diff = (pts - means[j]) * weights
        dists[:, j] = np.sqrt(np.sum(diff ** 2, axis=1))
    return dists


# ---------------------------------------------------------------------------
# KAMILA class
# ---------------------------------------------------------------------------

class KAMILA:
    """KAMILA clustering for mixed-type data.
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    n_init : int
        Number of random initializations (best is kept).
    max_iter : int
        Maximum iterations per initialization.
    cat_bw : float
        Bandwidth for categorical kernel smoothing (0.025 default).
    con_weights : array-like or None
        Weights for continuous variables (all 1.0 if None).
    cat_weights : array-like or None
        Weights for categorical variables (all 1.0 if None).
    random_state : int or None
        Seed for reproducibility.
    """
    
    def __init__(
        self,
        n_clusters: int = 2,
        n_init: int = 10,
        max_iter: int = 25,
        cat_bw: float = 0.025,
        con_weights: Optional[np.ndarray] = None,
        cat_weights: Optional[np.ndarray] = None,
        random_state: int | None = None,
    ):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.cat_bw = cat_bw
        self.con_weights = con_weights
        self.cat_weights = cat_weights
        self.random_state = random_state
        
        # Results
        self.labels_: Optional[np.ndarray] = None
        self.centers_: Optional[np.ndarray] = None
        self.cat_probs_: Optional[list] = None
        self.objective_: float = -np.inf
        self.n_iter_: int = 0
    
    def fit_predict(
        self, df: pd.DataFrame,
        num_cols: List[str],
        cat_cols: List[str],
    ) -> np.ndarray:
        """Fit KAMILA and return cluster labels (1-indexed like R, shifted to 0-indexed)."""
        # Prepare data
        con_var = df[num_cols].values.astype(float)
        n, p = con_var.shape
        q = len(cat_cols)
        
        # Encode categoricals as integer codes (1-based like R)
        cat_codes = np.zeros((n, q), dtype=int)
        cat_levels = []
        for ci, col in enumerate(cat_cols):
            cats = pd.Categorical(df[col])
            cat_codes[:, ci] = cats.codes + 1  # 1-based
            cat_levels.append(len(cats.categories))
        
        num_lev = np.array(cat_levels)
        
        # Set weights
        con_w = np.ones(p) if self.con_weights is None else np.asarray(self.con_weights)
        cat_w = np.ones(q) if self.cat_weights is None else np.asarray(self.cat_weights)
        
        rng = np.random.RandomState(self.random_state)
        k = self.n_clusters
        
        best_objective = -np.inf
        best_memb = np.zeros(n, dtype=int)
        best_centers = None
        best_probs = None
        best_n_iter = 0
        
        # Total distance for objective function
        global_mean = con_var.mean(axis=0, keepdims=True)
        total_dist = np.sum(_weighted_euclidean_distances(con_var, global_mean, con_w))
        
        for init in range(self.n_init):
            # Initialize means: uniform within range
            ranges = np.column_stack([con_var.min(axis=0), con_var.max(axis=0)])
            means_i = np.column_stack([
                rng.uniform(ranges[j, 0], ranges[j, 1], k)
                for j in range(p)
            ])  # (k, p)
            
            # Initialize categorical probs: Dirichlet(1,...,1)
            log_probs_cond = []
            for qi in range(q):
                alpha = np.ones(num_lev[qi])
                probs = rng.dirichlet(alpha, size=k)  # (k, n_levels)
                probs = np.maximum(probs, 1e-300)
                log_probs_cond.append(np.log(probs))
            
            memb_old = np.zeros(n, dtype=int)
            memb_new = np.zeros(n, dtype=int)
            degenerate = False
            
            for iteration in range(1, self.max_iter + 1):
                # 1. Weighted Euclidean distances to means (n x k)
                dist_i = _weighted_euclidean_distances(con_var, means_i, con_w)
                
                # 2. Minimum distances per point
                min_dist_i = dist_i.min(axis=1)
                
                # 3-4. Radial KDE: evaluate density at all distances
                all_eval = dist_i.ravel()
                kdes = radial_kde(min_dist_i, all_eval, pdim=p)
                log_dist_dens = np.log(np.maximum(kdes, 1e-300)).reshape(n, k)
                
                # 5. Categorical log-likelihoods (n x k)
                cat_log_liks = np.zeros((n, k))
                for qi in range(q):
                    codes_qi = cat_codes[:, qi] - 1  # 0-based for indexing
                    # log_probs_cond[qi] is (k, n_levels)
                    for kk in range(k):
                        cat_log_liks[:, kk] += cat_w[qi] * log_probs_cond[qi][kk, codes_qi]
                
                # 6. Combined log-likelihood
                all_log_liks = log_dist_dens + cat_log_liks
                
                # 7. Assign to clusters
                memb_old = memb_new.copy()
                memb_new = np.argmax(all_log_liks, axis=1) + 1  # 1-based
                
                # 8. Update means
                means_i = np.zeros((k, p))
                for kk in range(k):
                    mask = memb_new == (kk + 1)
                    if np.sum(mask) > 0:
                        means_i[kk] = con_var[mask].mean(axis=0)
                
                # 9. Update categorical probabilities (with kernel smoothing)
                log_probs_cond = []
                for qi in range(q):
                    # Build joint table (k x n_levels)
                    joint = np.zeros((k, num_lev[qi]), dtype=float)
                    for kk in range(k):
                        mask = memb_new == (kk + 1)
                        for lv in range(num_lev[qi]):
                            joint[kk, lv] = np.sum(cat_codes[mask, qi] == (lv + 1))
                    
                    # Smooth
                    if self.cat_bw > 0:
                        joint = _smooth_joint_table(joint.astype(int), self.cat_bw)
                    
                    # Conditional probs: P(level | cluster)
                    row_sums = joint.sum(axis=1, keepdims=True)
                    row_sums = np.maximum(row_sums, 1e-300)
                    probs = joint / row_sums
                    probs = np.maximum(probs, 1e-300)
                    log_probs_cond.append(np.log(probs))
                
                # 10. Check degenerate
                if len(np.unique(memb_new)) < k:
                    degenerate = True
                    break
                
                # Check convergence (require min 3 iterations)
                if iteration >= 3 and np.array_equal(memb_old, memb_new):
                    break
            
            # Compute objective: (WSS/BSS) * catLogLik
            if degenerate:
                objective = -np.inf
            else:
                cat_ll = np.sum(np.max(cat_log_liks, axis=1))
                win_dist = np.sum(dist_i[np.arange(n), memb_new - 1])
                bss = total_dist - win_dist
                wss_bss = win_dist / bss if bss > 0 else 100
                objective = wss_bss * cat_ll  # Both terms: smaller WSS/BSS * larger catLL = better
            
            if objective > best_objective:
                best_objective = objective
                best_memb = memb_new.copy()
                best_centers = means_i.copy()
                best_probs = [np.exp(lp) for lp in log_probs_cond]
                best_n_iter = iteration
        
        # Store results (convert to 0-indexed)
        self.labels_ = best_memb - 1
        self.centers_ = best_centers
        self.cat_probs_ = best_probs
        self.objective_ = best_objective
        self.n_iter_ = best_n_iter
        
        return self.labels_


# ---------------------------------------------------------------------------
# Prediction strength for number of clusters estimation
# ---------------------------------------------------------------------------

def kamila_auto_k(
    df: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
    k_range: range = range(2, 11),
    n_init: int = 10,
    n_cv_runs: int = 10,
    ps_thresh: float = 0.8,
    random_state: int = 42,
) -> Tuple[int, Dict]:
    """Estimate number of clusters using prediction strength method.
    
    Returns (best_k, results_dict).
    """
    rng = np.random.RandomState(random_state)
    n = len(df)
    n_test = n // 2
    
    k_list = list(k_range)
    ps_matrix = np.full((len(k_list), n_cv_runs), np.nan)
    
    for cv_run in range(n_cv_runs):
        for ki, k in enumerate(k_list):
            test_idx = rng.choice(n, n_test, replace=False)
            train_idx = np.setdiff1d(np.arange(n), test_idx)
            
            # Cluster test data
            test_model = KAMILA(n_clusters=k, n_init=n_init, random_state=rng.randint(2**31))
            test_labels = test_model.fit_predict(df.iloc[test_idx].reset_index(drop=True), num_cols, cat_cols)
            
            # Cluster training data & classify test into training clusters
            train_model = KAMILA(n_clusters=k, n_init=n_init, random_state=rng.randint(2**31))
            train_model.fit_predict(df.iloc[train_idx].reset_index(drop=True), num_cols, cat_cols)
            
            # Simple classification: assign test points to nearest training center
            test_con = df.iloc[test_idx][num_cols].values.astype(float)
            dists = _weighted_euclidean_distances(test_con, train_model.centers_, np.ones(len(num_cols)))
            te_into_tr = np.argmin(dists, axis=1)
            
            # Prediction strength: min over clusters of proportion of pairs
            ps_props = []
            for cl in range(k):
                cl_members = np.where(test_labels == cl)[0]
                cl_n = len(cl_members)
                if cl_n < 2:
                    continue
                # Count pairs where both assigned to same training cluster
                same_count = 0
                total_pairs = cl_n * (cl_n - 1) / 2
                for i in range(cl_n - 1):
                    for j in range(i + 1, cl_n):
                        if te_into_tr[cl_members[i]] == te_into_tr[cl_members[j]]:
                            same_count += 1
                ps_props.append(same_count / total_pairs if total_pairs > 0 else 0)
            
            if ps_props:
                ps_matrix[ki, cv_run] = min(ps_props)
    
    # Average prediction strength
    avg_ps = np.nanmean(ps_matrix, axis=1)
    se_ps = np.nanstd(ps_matrix, axis=1) / np.sqrt(n_cv_runs)
    ps_values = avg_ps + se_ps
    
    above_thresh = ps_values > ps_thresh
    if not np.any(above_thresh):
        best_k = k_list[np.argmax(ps_values)]
    else:
        best_k = max(np.array(k_list)[above_thresh])
    
    results = {
        "best_k": best_k,
        "k_range": k_list,
        "ps_values": dict(zip(k_list, ps_values)),
        "avg_ps": dict(zip(k_list, avg_ps)),
    }
    return best_k, results


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)
    n = 300
    X1 = np.concatenate([np.random.normal(0, 1, 100), np.random.normal(5, 1, 100), np.random.normal(2.5, 0.5, 100)])
    X2 = np.concatenate([np.random.normal(0, 1, 100), np.random.normal(5, 1, 100), np.random.normal(7, 0.5, 100)])
    colors = (["red"] * 50 + ["blue"] * 50 +
              ["green"] * 50 + ["yellow"] * 50 +
              ["red"] * 50 + ["green"] * 50)
    
    df = pd.DataFrame({"x1": X1, "x2": X2, "color": colors})
    ground_truth = np.array([0]*100 + [1]*100 + [2]*100)
    
    model = KAMILA(n_clusters=3, n_init=10, random_state=42)
    labels = model.fit_predict(df, num_cols=["x1", "x2"], cat_cols=["color"])
    
    # NMI
    from collections import Counter
    def nmi(a, b):
        n = len(a)
        ca, cb = Counter(a), Counter(b)
        joint = Counter(zip(a, b))
        mi = sum(ntp/n * np.log2((ntp/n) / (ca[t]/n * cb[p]/n))
                 for (t,p), ntp in joint.items() if ntp > 0)
        ha = -sum(c/n * np.log2(c/n) for c in ca.values() if c > 0)
        hb = -sum(c/n * np.log2(c/n) for c in cb.values() if c > 0)
        return 2 * mi / (ha + hb) if (ha + hb) > 0 else 0
    
    print(f"KAMILA smoke test:")
    print(f"  Clusters: {len(np.unique(labels))}")
    print(f"  Sizes: {np.bincount(labels)}")
    print(f"  NMI vs ground truth: {nmi(ground_truth, labels):.4f}")
    print(f"  Objective: {model.objective_:.4f}")
    print(f"  Centers:\n{model.centers_}")
