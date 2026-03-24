# mixclust/core/preprocess.py
from __future__ import annotations
import numpy as np, pandas as pd
from typing import List, Tuple, Optional

def preprocess_mixed_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Imputasi ringan: numerik→median, kategorik→modus (tanpa transformasi)."""
    df2 = df.copy()
    cat_cols = df2.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = df2.select_dtypes(exclude=["object", "category", "bool"]).columns.tolist()

    if num_cols:
        med = df2[num_cols].median()
        df2[num_cols] = df2[num_cols].fillna(med)
    for c in cat_cols:
        if df2[c].isnull().any():
            mode = df2[c].mode()
            if len(mode): df2[c] = df2[c].fillna(mode[0])
    return df2, cat_cols, num_cols

def prepare_mixed_arrays_no_label(df: pd.DataFrame):
    """
    Komponen raw utk Gower:
      X_num(float32), X_cat(int32), num_min/max, mask_num/cat, inv_rng(1/range)
    """
    df2 = df.copy()
    cat_cols = df2.select_dtypes(include=['object','category','bool']).columns.tolist()
    num_cols = df2.select_dtypes(include=[np.number]).columns.tolist()

    if num_cols:
        X_num = df2[num_cols].values.astype(np.float32)
        num_min = X_num.min(axis=0).astype(np.float32)
        num_max = X_num.max(axis=0).astype(np.float32)
        inv_rng = (1.0 / np.maximum(num_max - num_min, 1e-9)).astype(np.float32)
    else:
        X_num = np.zeros((len(df2), 0), np.float32)
        num_min = np.zeros((0,), np.float32)
        num_max = np.ones((0,), np.float32)
        inv_rng = np.ones((0,), np.float32)

    X_cat_list = []
    for c in cat_cols:
        vals, _ = pd.factorize(df2[c].astype(str), sort=True)
        X_cat_list.append(vals.astype(np.int32))
    X_cat = np.vstack(X_cat_list).T if X_cat_list else np.zeros((len(df2), 0), np.int32)

    mask_num = np.ones(X_num.shape[1], dtype=bool) if X_num.shape[1] else None
    mask_cat = np.ones(X_cat.shape[1], dtype=bool) if X_cat.shape[1] else None
    return X_num, X_cat, num_min, num_max, mask_num, mask_cat, inv_rng
