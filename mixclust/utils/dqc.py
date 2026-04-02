# mixclust/utils/dqc.py
# Data Quality Check — dijalankan sebelum AUFS-Samba
# Mendeteksi zero-variance, near-zero, dan high-missing features
# sehingga tidak masuk ke pipeline dan merusak reward landscape.

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple


def run_dqc(
    df: pd.DataFrame,
    *,
    zero_var_action: str = "drop",        # "drop" | "warn" | "ignore"
    near_zero_threshold: float = 0.998,   # top-1 frekuensi ≥ ini → near-zero
    near_zero_action: str = "drop",       # "drop" | "warn" | "ignore"
    missing_threshold: float = 0.5,       # missing ≥ ini → flag
    missing_action: str = "warn",         # "drop" | "warn" | "ignore"
    num_std_threshold: float = 1e-6,      # std numerik < ini → near-zero
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    """
    Data Quality Check untuk deteksi fitur bermasalah sebelum AUFS-Samba.

    Parameters
    ----------
    df : DataFrame fitur (sudah tanpa ID dan label)
    zero_var_action : apa yang dilakukan terhadap zero-variance features
    near_zero_threshold : threshold fraksi dominansi untuk near-zero
    near_zero_action : apa yang dilakukan terhadap near-zero features
    missing_threshold : threshold proporsi missing values
    missing_action : apa yang dilakukan terhadap high-missing features
    num_std_threshold : threshold std numerik untuk near-zero
    verbose : print report

    Returns
    -------
    df_clean : DataFrame setelah kolom bermasalah di-handle
    dropped_cols : list kolom yang di-drop
    report : DataFrame laporan per kolom
    """
    n = len(df)
    records = []
    to_drop = []

    for col in df.columns:
        series = df[col]
        dtype_kind = series.dtype.kind  # 'f'=float, 'i'=int, 'O'=object, 'b'=bool

        n_missing = series.isna().sum()
        missing_frac = n_missing / n if n > 0 else 0.0

        n_unique = series.nunique(dropna=True)
        vc = series.value_counts(normalize=True, dropna=True)
        top_frac = float(vc.iloc[0]) if len(vc) > 0 else 0.0
        top_val = vc.index[0] if len(vc) > 0 else None

        # Std untuk numerik
        std_val = None
        if dtype_kind in ('f', 'i', 'u'):
            try:
                std_val = float(series.dropna().std())
            except Exception:
                std_val = None

        # ── Klasifikasi ──────────────────────────────────────────────
        issue = None
        action_taken = "ok"

        # Zero variance: satu nilai unik (atau std=0)
        if n_unique <= 1 or (std_val is not None and std_val < num_std_threshold and dtype_kind in ('f','i','u')):
            issue = "zero_variance"
            action_taken = zero_var_action

        # Near-zero variance: satu nilai sangat dominan
        elif top_frac >= near_zero_threshold:
            issue = "near_zero_variance"
            action_taken = near_zero_action

        # High missing — hanya tambahkan jika bukan sudah zero/near_zero
        if missing_frac >= missing_threshold:
            if issue is None:
                issue = "high_missing"
                action_taken = missing_action
            else:
                # zero/near_zero sudah menang — hanya anotasi saja
                issue = issue + "+high_missing"
                # action_taken tetap dari zero/near_zero

        records.append({
            "column": col,
            "dtype": str(series.dtype),
            "n_unique": n_unique,
            "top_value": str(top_val)[:40] if top_val is not None else None,
            "top_frac": round(top_frac, 4),
            "missing_frac": round(missing_frac, 4),
            "std": round(std_val, 6) if std_val is not None else None,
            "issue": issue,
            "action": action_taken,
        })

        if issue and action_taken == "drop":
            to_drop.append(col)

    report = pd.DataFrame(records)
    flagged = report[report["issue"].notna()].copy()

    if verbose and len(flagged) > 0:
        print(f"\n[DQC] {len(df.columns)} fitur diperiksa → "
              f"{len(flagged)} bermasalah, {len(to_drop)} akan di-drop\n")
        for _, row in flagged.iterrows():
            icon = "⛔" if row["action"] == "drop" else "⚠️ "
            print(f"  {icon} {row['column']:35s} | {row['issue']:25s} | "
                  f"top={row['top_frac']:.1%} ({row['top_value']}) | "
                  f"missing={row['missing_frac']:.1%}")
        print()
    elif verbose:
        print(f"[DQC] {len(df.columns)} fitur diperiksa → semua OK ✓\n")

    df_clean = df.drop(columns=to_drop, errors="ignore")
    return df_clean, to_drop, report
