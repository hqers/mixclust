# src/mixclust/pipeline/calibration.py
from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def _to_1d_float(arr: Iterable) -> np.ndarray:
    a = np.asarray(arr, dtype=float).ravel()
    # drop NaN/inf
    mask = np.isfinite(a)
    return a[mask]


def _linear_fit(L: np.ndarray, S: np.ndarray) -> Tuple[float, float]:
    """
    Ordinary Least Squares for S ≈ a*L + b (with intercept).
    Returns (a, b).
    """
    if L.size == 0 or S.size == 0:
        return 1.0, 0.0
    X = np.vstack([L, np.ones_like(L)]).T
    a, b = np.linalg.lstsq(X, S, rcond=None)[0]
    return float(a), float(b)


def _robust_fit(L: np.ndarray, S: np.ndarray) -> Tuple[float, float]:
    """
    Robust linear fit via HuberRegressor if available; fallback to OLS.
    Returns (a, b) for S ≈ a*L + b.
    """
    try:
        from sklearn.linear_model import HuberRegressor
        X = L.reshape(-1, 1)
        model = HuberRegressor(epsilon=1.35)  # common default
        model.fit(X, S)
        a = float(model.coef_[0])
        b = float(model.intercept_)
        return a, b
    except Exception:
        return _linear_fit(L, S)


def _clip_scores(x: np.ndarray, lo: float = -1.0, hi: float = 1.0) -> np.ndarray:
    return np.clip(x, lo, hi)


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------

def calibrate_lsil_to_ss(
    L: Iterable,
    S: Iterable,
    *,
    robust: bool = False,
    clip_range: Tuple[float, float] = (-1.0, 1.0)
) -> Dict[str, float]:
    """
    Fit linear calibration S ≈ a * L + b.

    Parameters
    ----------
    L : iterable of float
        L-Sil (proxy) values.
    S : iterable of float
        Ground-truth SS(Gower) values.
    robust : bool
        Use robust regression (Huber) if available; fallback to OLS.
    clip_range : (low, high)
        Clip predictions & (optionally) inputs to this range on return stats.

    Returns
    -------
    dict with keys:
        a, b, r, r2, n, mae_train, rmse_train
    """
    L = _to_1d_float(L)
    S = _to_1d_float(S)
    n = int(min(L.size, S.size))
    if n == 0:
        return {"a": 1.0, "b": 0.0, "r": 0.0, "r2": 0.0, "n": 0, "mae_train": np.nan, "rmse_train": np.nan}

    L = L[:n]
    S = S[:n]

    fit_fn = _robust_fit if robust else _linear_fit
    a, b = fit_fn(L, S)

    S_hat = a * L + b
    S_hat = _clip_scores(S_hat, *clip_range)

    r = float(np.corrcoef(L, S)[0, 1]) if n >= 2 else 0.0
    r2 = float(r2_score(S, S_hat)) if n >= 2 else 0.0
    mae = float(np.mean(np.abs(S_hat - S)))
    rmse = float(np.sqrt(np.mean((S_hat - S) ** 2)))

    return {
        "a": float(a),
        "b": float(b),
        "r": float(r),
        "r2": float(r2),
        "n": int(n),
        "mae_train": mae,
        "rmse_train": rmse,
    }


def predict_ss_from_lsil(
    L: Iterable,
    a: float,
    b: float,
    *,
    clip_range: Tuple[float, float] = (-1.0, 1.0)
) -> np.ndarray:
    """
    Apply calibration S_hat = a * L + b with clipping.
    """
    L = _to_1d_float(L)
    S_hat = a * L + b
    return _clip_scores(S_hat, *clip_range)


def lodo_cv(
    df: pd.DataFrame,
    *,
    col_L: str = "L-Sil_proto",
    col_S: str = "SS_Gower",
    col_id: str = "Dataset",
    robust: bool = False,
    clip_range: Tuple[float, float] = (-1.0, 1.0)
) -> pd.DataFrame:
    """
    Leave-One-Dataset-Out cross-validation for linear calibration.
    Each row of df is assumed to be a dataset-level observation (aggregate).

    Returns a DataFrame with per-holdout errors and fold-specific params.
    Columns:
      Dataset, a_cv, b_cv, SS_hat_cv, SS_true, AE_cv, SE_cv
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=[col_id, "a_cv", "b_cv", "SS_hat_cv", "SS_true", "AE_cv", "SE_cv"])

    rows = []
    for i in range(len(df)):
        train = df.drop(index=i)
        test = df.iloc[i]

        L_tr = _to_1d_float(train[col_L].astype(float).values)
        S_tr = _to_1d_float(train[col_S].astype(float).values)

        if L_tr.size == 0 or S_tr.size == 0:
            # skip fold with insufficient train data
            rows.append({
                col_id: test.get(col_id, f"row_{i}"),
                "a_cv": np.nan,
                "b_cv": np.nan,
                "SS_hat_cv": np.nan,
                "SS_true": float(test[col_S]),
                "AE_cv": np.nan,
                "SE_cv": np.nan,
            })
            continue

        # fit on train
        fit_fn = _robust_fit if robust else _linear_fit
        a_cv, b_cv = fit_fn(L_tr, S_tr)

        # predict on the held-out
        L_te = float(test[col_L])
        S_te = float(test[col_S])
        S_hat_cv = float(a_cv * L_te + b_cv)
        S_hat_cv = float(np.clip(S_hat_cv, *clip_range))

        rows.append({
            col_id: test.get(col_id, f"row_{i}"),
            "a_cv": float(a_cv),
            "b_cv": float(b_cv),
            "SS_hat_cv": S_hat_cv,
            "SS_true": S_te,
            "AE_cv": abs(S_hat_cv - S_te),
            "SE_cv": (S_hat_cv - S_te) ** 2,
        })

    return pd.DataFrame(rows)


def lodo_summary(cv_df: pd.DataFrame, *, col_id: str = "Dataset") -> Dict[str, float]:
    """
    Summarize LODO-CV result.
    Returns MAE, RMSE, R (corr of preds vs true), and N folds used.
    """
    if cv_df is None or cv_df.empty:
        return {"N_folds": 0, "MAE": np.nan, "RMSE": np.nan, "R_pred_true": np.nan}

    df = cv_df.copy()
    df = df[np.isfinite(df["SS_hat_cv"]) & np.isfinite(df["SS_true"])]

    if df.empty:
        return {"N_folds": 0, "MAE": np.nan, "RMSE": np.nan, "R_pred_true": np.nan}

    mae = float(np.mean(np.abs(df["SS_hat_cv"] - df["SS_true"])))
    rmse = float(np.sqrt(np.mean((df["SS_hat_cv"] - df["SS_true"]) ** 2)))
    r = float(np.corrcoef(df["SS_hat_cv"], df["SS_true"])[0, 1]) if len(df) >= 2 else np.nan

    return {"N_folds": int(len(df)), "MAE": mae, "RMSE": rmse, "R_pred_true": r}


def fit_calibration_on_dataframe(
    df: pd.DataFrame,
    *,
    col_L: str = "L-Sil_proto",
    col_S: str = "SS_Gower",
    robust: bool = False,
    clip_range: Tuple[float, float] = (-1.0, 1.0)
) -> Dict[str, float]:
    """
    Convenience: fit calibration directly from a dataframe with columns L & S.
    Returns dict as in calibrate_lsil_to_ss.
    """
    if df is None or df.empty:
        return {"a": 1.0, "b": 0.0, "r": 0.0, "r2": 0.0, "n": 0, "mae_train": np.nan, "rmse_train": np.nan}

    L = _to_1d_float(df[col_L].astype(float).values)
    S = _to_1d_float(df[col_S].astype(float).values)
    return calibrate_lsil_to_ss(L, S, robust=robust, clip_range=clip_range)
