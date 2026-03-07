# src/mixclust/reporting/profiles.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from ..utils.cluster_profiling import profile_clusters

def build_profiles_table(
    df: pd.DataFrame,
    labels: np.ndarray,
    best_cols: List[str],
    numeric_keys: Optional[List[str]] = None,
    cat_keys: Optional[List[str]] = None,
    topk_cat: int = 5
) -> pd.DataFrame:
    """
    Build a compact profiles table.
    - df: original dataframe (contains best_cols)
    - labels: cluster labels (np.ndarray)
    - best_cols: list of features used (order arbitrary)
    - numeric_keys: optional list of numeric indicators to include; if None, auto-detect
    - cat_keys: optional list of categorical indicators to include; if None, use first few from best_cols
    """
    # determine categorical columns among best_cols
    cat_cols = [c for c in best_cols if df[c].dtype.name in ("object","category","bool")]
    prof = profile_clusters(df[best_cols].copy(), labels=labels, cat_cols=cat_cols, topk=topk_cat)

    # automatic numeric keys if user not provided
    if numeric_keys is None:
        # numeric among best_cols
        numeric_keys = [c for c in best_cols if c not in cat_cols and pd.api.types.is_numeric_dtype(df[c])]
        # pick up to 10 most "informative" if too many: prefer those with non-zero variance
        numeric_keys = [c for c in numeric_keys if df[c].nunique() > 1]
        numeric_keys = numeric_keys[:10]

    # automatic categorical keys if not provided
    if cat_keys is None:
        cat_keys = cat_cols[:min(len(cat_cols), 6)]

    rows = []
    size = prof.get("size", {})
    fraction = prof.get("fraction", {})

    # ensure consistent keys types (string/int)
    cluster_ids = sorted([int(k) for k in size.keys()]) if size else sorted(np.unique(labels).tolist())

    for k in cluster_ids:
        # fetch as string key if not present as int
        k_key = k if k in prof.get("size", {}) else str(k)
        row: Dict[str, Any] = {
            "cluster": int(k),
            "size": int(prof.get("size", {}).get(k_key, prof.get("size", {}).get(str(k), 0))),
            "fraction": float(prof.get("fraction", {}).get(k_key, prof.get("fraction", {}).get(str(k), 0.0)))
        }

        # numeric indicators: mean, std, diff_vs_global, cis (if available)
        for col in numeric_keys:
            stats = prof.get("numeric", {}).get(col, {})
            s_k = stats.get(k_key, stats.get(str(k), {}))
            row[f"{col}_mean"] = s_k.get("mean", np.nan) if isinstance(s_k, dict) else np.nan
            row[f"{col}_median"] = s_k.get("median", np.nan) if isinstance(s_k, dict) else np.nan
            row[f"{col}_iqr"] = s_k.get("iqr", np.nan) if isinstance(s_k, dict) else np.nan
            row[f"{col}_diff_vs_global"] = s_k.get("diff_vs_global", np.nan) if isinstance(s_k, dict) else np.nan

            # CIS if stored at numeric[col]["_cis_cohensd_mean"]
            cis_val = prof.get("numeric", {}).get(col, {}).get("_cis_cohensd_mean", np.nan)
            row[f"{col}_cis"] = cis_val

        # categorical indicators: top lifts
        for c in cat_keys:
            cat_info = prof.get("categorical", {}).get(c, {})
            top_dict = cat_info.get(k_key, cat_info.get(str(k), {})).get("top", {}) if isinstance(cat_info, dict) else {}
            # join topk as "category:lift"
            row[f"{c}_top_lifts"] = ";".join([f"{cat}:{val:.3f}" for cat, val in list(top_dict.items())[:topk_cat]]) if top_dict else ""

        rows.append(row)

    return pd.DataFrame(rows)
