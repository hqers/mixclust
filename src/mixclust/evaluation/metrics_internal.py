# src/mixclust/evaluation/metrics_internal.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score

def compute_internal_metrics(X_df: pd.DataFrame, labels: np.ndarray, cat_cols: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    # DBI/CHI di numerik saja (standar praktik)
    X_num = X_df.drop(columns=cat_cols).select_dtypes(include=[np.number])
    if X_num.shape[1] >= 1 and len(np.unique(labels)) > 1:
        try:
            out["dbi"] = float(davies_bouldin_score(X_num, labels))
        except Exception:
            out["dbi"] = np.nan
        try:
            out["chi"] = float(calinski_harabasz_score(X_num, labels))
        except Exception:
            out["chi"] = np.nan
    else:
        out["dbi"] = np.nan
        out["chi"] = np.nan
    return out
