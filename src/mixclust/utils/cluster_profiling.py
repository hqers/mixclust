# dynamic_clustering/src/mixclust/utils/cluster_profiling.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union
import warnings
from scipy.stats import f_oneway, kruskal, chi2_contingency

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
        for k, g in groups.items():
            stats[k] = {
                "mean": float(g[col].mean()),
                "median": float(g[col].median()),
                "iqr": float(g[col].quantile(0.75) - g[col].quantile(0.25)),
                "diff_vs_global": float(g[col].mean() - global_mean)
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
            stats["_anova_p"] = np.nan
        num_stats[col] = stats
    out["numeric"] = num_stats

    # kategorik
    cat_stats = {}
    for col in cat_cols:
        stats = {}
        global_prop = X_df[col].astype(str).value_counts(normalize=True)
        for k, g in groups.items():
            prop = g[col].astype(str).value_counts(normalize=True)
            # top kategori by lift
            lift = (prop / (global_prop + 1e-12)).sort_values(ascending=False)
            stats[k] = {"top": lift.head(topk).to_dict()}
        # chi2 antar cluster
        try:
            tab = pd.crosstab(X_df["__label__"], X_df[col].astype(str))
            chi2, p, dof, exp = chi2_contingency(tab)
            stats["_chi2_p"] = float(p)
        except Exception:
            stats["_chi2_p"] = np.nan
        cat_stats[col] = stats
    out["categorical"] = cat_stats

    return out
