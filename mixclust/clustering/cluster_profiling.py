# mixclust/clustering/cluster_profiling.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union
import warnings
from scipy.stats import f_oneway, kruskal, chi2_contingency
from sklearn.preprocessing import LabelEncoder
import math


def profile_clusters(X_df: pd.DataFrame, labels: np.ndarray, cat_cols: list[str], topk: int = 10) -> dict:
    out = {}
    X_df = X_df.copy()
    X_df["__label__"] = labels
    groups = {k: v.drop(columns="__label__") for k, v in X_df.groupby("__label__")}

    # ringkasan dasar
    size = X_df["__label__"].value_counts().sort_index()
    out["size"] = size.to_dict()
    out["fraction"] = (size / len(X_df)).to_dict()

    # numerik
    num_cols = X_df.drop(columns=["__label__"] + cat_cols).select_dtypes(include=[np.number]).columns.tolist()
    num_stats = {}
    for col in num_cols:
        stats = {}
        global_mean = X_df[col].mean()
        arr_full = X_df[col].values
        for k, g in groups.items():
            vals = g[col].dropna().values
            stats[k] = {
                "mean": float(np.nanmean(vals)) if len(vals) else float(np.nan),
                "median": float(np.nanmedian(vals)) if len(vals) else float(np.nan),
                "iqr": float(np.nanpercentile(vals, 75) - np.nanpercentile(vals, 25)) if len(vals) else float(np.nan),
                "diff_vs_global": float(np.nanmean(vals) - global_mean) if len(vals) else float(np.nan)
            }
        # uji beda antar cluster
        try:
            vals = [g[col].dropna().values for g in groups.values()]
            if all(len(v) > 3 for v in vals):
                F, p = f_oneway(*vals)
            else:
                H, p = kruskal(*vals)
            stats["_anova_p"] = float(p)
        except Exception:
            stats["_anova_p"] = None

        # effect size aggregate (CIS-B)
        try:
            cis_val = cis_cohensd_feature(arr_full, X_df["__label__"].values, agg="mean")
            stats["_cis_cohensd_mean"] = float(cis_val)
        except Exception:
            stats["_cis_cohensd_mean"] = None

        num_stats[col] = stats
    out["numeric"] = num_stats

    # kategorik
    cat_stats = {}
    for col in cat_cols:
        stats = {}
        X_col_str = X_df[col].astype(str)
        global_prop = X_col_str.value_counts(normalize=True)
        for k, g in groups.items():
            prop = g[col].astype(str).value_counts(normalize=True)
            # Reindex ke semua kategori global, isi 0 kalau tidak ada di klaster ini
            prop = prop.reindex(global_prop.index, fill_value=0.0)
            lift = (prop / (global_prop + 1e-12)).sort_values(ascending=False)
            # Ganti inf/-inf dengan 0 (kategori tidak ada di global tapi ada di klaster)
            lift = lift.replace([np.inf, -np.inf], 0.0)
            stats[k] = {"top": lift.head(topk).to_dict(), "n_unique": int(g[col].nunique())}
        # chi2 & Cramer's V
        try:
            tab = pd.crosstab(X_df["__label__"], X_col_str)
            chi2, p, dof, exp = chi2_contingency(tab)
            stats["_chi2_p"] = float(p)
            stats["_cramers_v"] = float(cramers_v_from_crosstab(tab))
        except Exception:
            stats["_chi2_p"] = None
            stats["_cramers_v"] = None
        cat_stats[col] = stats
    out["categorical"] = cat_stats

    return out


# ---------- Cramer's V (kategori vs kategori) ----------
def cramers_v_from_crosstab(ct: pd.DataFrame) -> float:
    """ct: contingency table (pandas crosstab). Returns Cramer's V."""
    try:
        chi2, p, dof, exp = chi2_contingency(ct.values)
        n = ct.values.sum()
        r, c = ct.shape
        denom = min(r - 1, c - 1)
        if denom <= 0 or n == 0:
            return float(0.0)
        return float(math.sqrt((chi2 / n) / denom))
    except Exception:
        return None

# ---------- Cohen's d (two sample effect size) ----------
def cohens_d_two_groups(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a[~np.isnan(a)])
    b = np.asarray(b[~np.isnan(b)])
    if len(a) < 2 or len(b) < 2:
        return float(0.0)
    s1 = a.std(ddof=1)
    s2 = b.std(ddof=1)
    n1, n2 = len(a), len(b)
    # pooled sd
    pooled = math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / max(1, (n1 + n2 - 2)))
    if pooled == 0:
        return float(0.0)
    return float((a.mean() - b.mean()) / pooled)

# ---------- Lift for categorical proportions (cluster vs global) ----------
def lift_by_prop(cluster_prop: pd.Series, global_prop: pd.Series) -> pd.Series:
    """Return lift Series indexed by category (cluster_prop / global_prop)."""
    # Align indices, avoid divide by zero
    gp = global_prop.reindex(cluster_prop.index).fillna(1e-12)
    return (cluster_prop / gp).replace([np.inf, -np.inf], np.nan)

# ---------- CIS options ----------
def cis_cohensd_feature(feature: np.ndarray, labels: np.ndarray, agg: str = "mean") -> float:
    """CIS-B: aggregate of Cohen's d between each cluster vs rest. agg='mean' or 'max'."""
    uniq = np.unique(labels)
    scores = []
    for u in uniq:
        g_in = feature[labels == u]
        g_out = feature[labels != u]
        scores.append(abs(cohens_d_two_groups(g_in, g_out)))
    if not scores:
        return float(0.0)
    return float(np.mean(scores)) if agg == "mean" else float(np.max(scores))

def cis_mutual_info_discrete(feature, labels, n_bins: int = 10):
    """
    CIS-A: mutual information normalized by H(cluster).
    For numeric -> quantile binning first.
    """
    try:
        if np.issubdtype(feature.dtype, np.number):
            # safe qcut; if too few unique values fallback to ordinal encoding
            try:
                feat_cat = pd.qcut(feature, q=min(n_bins, len(feature)), duplicates="drop").astype(str)
            except Exception:
                feat_cat = pd.Series(feature).astype(str)
        else:
            feat_cat = pd.Series(feature).astype(str)
        le = LabelEncoder()
        x_enc = le.fit_transform(feat_cat)
        # mutual_info_classif expects 2D X
        from sklearn.feature_selection import mutual_info_classif
        mi = mutual_info_classif(x_enc.reshape(-1, 1), labels, discrete_features=True)
        # normalize by entropy of labels
        probs = np.bincount(labels) / len(labels)
        Hc = -np.sum([p * math.log(p + 1e-12) for p in probs if p > 0])
        return float(mi[0] / (Hc + 1e-12))
    except Exception:
        return None
