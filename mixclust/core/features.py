# mixclust/core/features.py
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
import numpy as np, pandas as pd

def _onehot_encoder():
    try:
        return OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    except TypeError:
        return OneHotEncoder(sparse=False, handle_unknown='ignore')

def build_features(df, label_col=None, scaler_type="standard", unit_norm=True, return_label_text=True):
    """
    Returns:
      X_unit : (n,d) float32, fitur gabungan (OneHot kategori + scaled numerik), opsional unit-norm
      labels_text : array[str] jika return_label_text True; else array (tipe asli)
      cols: dict {'cat': list, 'num': list}
    """
    df = df.copy()
    if label_col is not None and label_col in df.columns:
        df[label_col] = df[label_col].astype(str)

    cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in cat_cols if c != label_col]

    # One-Hot kategori (lossless)
    if len(cat_cols) > 0:
        ohe = _onehot_encoder()
        X_cat = ohe.fit_transform(df[cat_cols])
    else:
        X_cat = np.zeros((len(df), 0), dtype=np.float32)

    # Scale numerik
    if len(num_cols) > 0:
        if scaler_type == "standard":
            scaler = StandardScaler()
            X_num = scaler.fit_transform(df[num_cols].values.astype(float))
        elif scaler_type == "robust":
            scaler = RobustScaler()
            X_num = scaler.fit_transform(df[num_cols].values.astype(float))
        else:
            X_num = df[num_cols].values.astype(float)
    else:
        X_num = np.zeros((len(df), 0), dtype=np.float32)

    X = np.hstack([X_cat, X_num]).astype(np.float32)

    if unit_norm:
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        X = X / norms

    if label_col is not None and label_col in df.columns:
        labels_text = df[label_col].values if return_label_text else df[label_col].values
    else:
        labels_text = None

    return X, labels_text, {'cat': cat_cols, 'num': num_cols}


def prepare_mixed_arrays(df, label_col=None):
    df2 = df.copy()
    if label_col is not None:
        df2[label_col] = df2[label_col].astype(str)

    cat_cols = df2.select_dtypes(include=['object','category']).columns.tolist()
    num_cols = df2.select_dtypes(include=[np.number]).columns.tolist()
    if label_col is not None:
        cat_cols = [c for c in cat_cols if c != label_col]

    # numerik (raw, tanpa scaling)
    if len(num_cols) > 0:
        X_num_raw = df2[num_cols].values.astype(np.float32)
        num_min = X_num_raw.min(axis=0)
        num_max = X_num_raw.max(axis=0)
        # >>> NEW: cache inverse range
        inv_rng = 1.0 / np.maximum(num_max - num_min, 1e-9)
        inv_rng = inv_rng.astype(np.float32)
    else:
        X_num_raw = np.zeros((len(df2), 0), dtype=np.float32)
        num_min = np.zeros((0,), dtype=np.float32)
        num_max = np.ones((0,), dtype=np.float32)
        inv_rng = np.ones((0,), dtype=np.float32)

    # kategorik → integer per kolom
    X_cat_list = []
    for c in cat_cols:
        vals, _ = pd.factorize(df2[c].astype(str), sort=True)  # 0..L-1
        X_cat_list.append(vals.astype(np.int32))
    X_cat_int = np.vstack(X_cat_list).T if len(X_cat_list) > 0 else np.zeros((len(df2), 0), dtype=np.int32)

    mask_num = np.ones(X_num_raw.shape[1], dtype=bool) if X_num_raw.shape[1] > 0 else None
    mask_cat = np.ones(X_cat_int.shape[1], dtype=bool) if X_cat_int.shape[1] > 0 else None

    return X_num_raw, X_cat_int, num_min.astype(np.float32), num_max.astype(np.float32), mask_num, mask_cat, inv_rng
