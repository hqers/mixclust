# mixclust/utils/dqc.py
# Data Quality Check — dijalankan sebelum AUFS-Samba
# Mendeteksi zero-variance, near-zero, high-missing, dan
# disguised_categorical (integer kode yang seharusnya kategorik).

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict


def run_dqc(
    df: pd.DataFrame,
    *,
    zero_var_action: str = "drop",        # "drop" | "warn" | "ignore"
    near_zero_threshold: float = 0.998,   # top-1 frekuensi ≥ ini → near-zero
    near_zero_action: str = "drop",       # "drop" | "warn" | "ignore"
    missing_threshold: float = 0.5,       # missing ≥ ini → flag
    missing_action: str = "warn",         # "drop" | "warn" | "ignore"
    num_std_threshold: float = 1e-6,      # std numerik < ini → near-zero
    # ── v1.1.13: disguised categorical detection ──────────────────
    disguised_cat_action: str = "warn",   # "cast" | "warn" | "ignore"
    disguised_cat_nunique_max: int = 20,  # int/float kolom dengan nunique ≤ ini
    disguised_cat_nunique_min: int = 3,   # dan nunique ≥ ini — mulai 3 karena binary (nunique=2)
                                             # sudah ditangani benar oleh Gower numerik (d identik
                                             # dengan Gower kategorik untuk 0/1)
    explicit_cat_cols: Optional[List[str]] = None,  # kolom yang user deklarasikan kategorik
    explicit_num_cols: Optional[List[str]] = None,  # kolom yang user deklarasikan numerik (tidak di-cast)
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
    disguised_cat_action : "cast" → otomatis cast ke 'category';
                           "warn" → hanya warning tanpa cast;
                           "ignore" → tidak deteksi
    disguised_cat_nunique_max : batas atas nunique untuk dianggap kategorik tersamar
    disguised_cat_nunique_min : batas bawah nunique (hindari deteksi kolom biner)
    explicit_num_cols : kolom yang user nyatakan numerik secara eksplisit,
                        tidak akan di-cast meskipun nunique kecil
    verbose : print report

    Returns
    -------
    df_clean : DataFrame setelah kolom bermasalah di-handle
               (termasuk kolom yang di-cast ke 'category')
    dropped_cols : list kolom yang di-drop
    report : DataFrame laporan per kolom
    """
    n = len(df)
    records = []
    to_drop = []
    df_clean = df.copy()

    # ── Proses explicit_cat_cols terlebih dahulu ──────────────────────────────
    explicit_cast_done = []
    if explicit_cat_cols:
        for c in explicit_cat_cols:
            if c in df_clean.columns:
                if df_clean[c].dtype.kind in ('i', 'u', 'f'):
                    df_clean[c] = df_clean[c].astype('category')
                    explicit_cast_done.append(c)
                elif df_clean[c].dtype == object:
                    df_clean[c] = df_clean[c].astype('category')
                    explicit_cast_done.append(c)
        if verbose and explicit_cast_done:
            print(f"[DQC] explicit_cat_cols: cast {len(explicit_cast_done)} kolom → 'category': "
                  f"{explicit_cast_done}")

    # ── Per kolom ─────────────────────────────────────────────────────────────
    for col in df_clean.columns:
        series = df_clean[col]
        dtype_kind = series.dtype.kind  # 'f'=float, 'i'=int, 'O'=object, 'b'=bool

        n_missing = series.isna().sum()
        missing_frac = n_missing / n if n > 0 else 0.0

        n_unique = series.nunique(dropna=True)
        vc = series.value_counts(normalize=True, dropna=True)
        top_frac = float(vc.iloc[0]) if len(vc) > 0 else 0.0
        top_val = vc.index[0] if len(vc) > 0 else None

        std_val = None
        if dtype_kind in ('f', 'i', 'u'):
            try:
                std_val = float(series.dropna().std())
            except Exception:
                std_val = None

        # ── Klasifikasi ───────────────────────────────────────────────────────
        issue = None
        action_taken = "ok"

        # Level 0: zero variance
        if n_unique <= 1 or (std_val is not None and std_val < num_std_threshold
                             and dtype_kind in ('f', 'i', 'u')):
            issue = "zero_variance"
            action_taken = zero_var_action

        # Level 1: near-zero variance
        elif top_frac >= near_zero_threshold:
            issue = "near_zero_variance"
            action_taken = near_zero_action

        # Level 2: high missing
        if missing_frac >= missing_threshold:
            if issue is None:
                issue = "high_missing"
                action_taken = missing_action
            else:
                issue = issue + "+high_missing"

        # ── v1.1.13 Level 3: disguised categorical ────────────────────────────
        # Kolom integer/float dengan nunique kecil → kemungkinan kode kategorik
        # Tidak apply jika sudah ada issue lain (zero_var/near_zero) atau
        # sudah explicit_cat atau sudah bertipe category/object/bool
        # atau user deklarasikan sebagai numerik via explicit_num_cols
        if (disguised_cat_action != "ignore"
                and issue is None
                and col not in (explicit_cat_cols or [])
                and col not in (explicit_num_cols or [])
                and dtype_kind in ('i', 'u', 'f')
                and disguised_cat_nunique_min <= n_unique <= disguised_cat_nunique_max):

            if disguised_cat_action == "cast":
                df_clean[col] = df_clean[col].astype('category')
                issue = "disguised_categorical"
                action_taken = "cast→category"
            else:  # "warn"
                issue = "disguised_categorical"
                action_taken = "warn"

        records.append({
            "column":       col,
            "dtype_orig":   str(df[col].dtype),
            "dtype_now":    str(df_clean[col].dtype),
            "n_unique":     n_unique,
            "top_value":    str(top_val)[:40] if top_val is not None else None,
            "top_frac":     round(top_frac, 4),
            "missing_frac": round(missing_frac, 4),
            "std":          round(std_val, 6) if std_val is not None else None,
            "issue":        issue,
            "action":       action_taken,
        })

        if issue and action_taken == "drop":
            to_drop.append(col)

    report = pd.DataFrame(records)
    flagged = report[report["issue"].notna()].copy()

    if verbose and len(flagged) > 0:
        n_cast     = (flagged["action"] == "cast→category").sum()
        n_drop     = (flagged["action"] == "drop").sum()
        n_warn     = flagged["action"].isin(["warn"]).sum()
        n_explicit = len(explicit_cast_done)

        print(f"\n[DQC] {len(df.columns)} fitur diperiksa → "
              f"{len(flagged)} perlu perhatian  "
              f"(drop={n_drop}, cast={n_cast}, warn={n_warn}, "
              f"explicit_cast={n_explicit})\n")

        for _, row in flagged.iterrows():
            if row["action"] == "drop":
                icon = "⛔"
            elif row["action"] == "cast→category":
                icon = "🔄"
            elif row["issue"] == "disguised_categorical":
                icon = "⚠️ "
            else:
                icon = "⚠️ "

            dtype_change = (f"  [{row['dtype_orig']} → {row['dtype_now']}]"
                            if row["dtype_orig"] != row["dtype_now"] else "")
            print(f"  {icon} {row['column']:35s} | {str(row['issue']):28s} | "
                  f"nunique={row['n_unique']:>4}  top={row['top_frac']:.1%} | "
                  f"action={row['action']}{dtype_change}")
        print()

        # Ringkasan khusus disguised_categorical
        dc_cols = flagged[flagged["issue"] == "disguised_categorical"]["column"].tolist()
        if dc_cols:
            print(f"  [DQC] disguised_categorical: {len(dc_cols)} kolom integer/float "
                  f"dengan nunique ≤ {disguised_cat_nunique_max} terdeteksi sebagai "
                  f"kemungkinan kode kategorik: {dc_cols}")
            if disguised_cat_action == "cast":
                print(f"  [DQC] Kolom tersebut otomatis di-cast ke dtype 'category'.")
                print(f"  [DQC] Untuk menonaktifkan: set disguised_cat_action='warn' atau 'ignore'")
                print(f"  [DQC] Untuk proteksi kolom numerik: explicit_num_cols=['kolom1', ...]")
            else:
                print(f"  [DQC] Kolom-kolom ini TIDAK di-cast (disguised_cat_action='warn').")
                print(f"  [DQC] Jika kolom tersebut memang kategorik, cast secara eksplisit di notebook:")
                print(f"  [DQC]   df['nama_kolom'] = df['nama_kolom'].astype('category')")
                print(f"  [DQC] Atau gunakan explicit_cat_cols=['kolom1', ...] di run_dqc()")
                print(f"  [DQC] Atau set disguised_cat_action='cast' untuk konversi otomatis.")
            print()

    elif verbose:
        print(f"[DQC] {len(df.columns)} fitur diperiksa → semua OK ✓\n")

    df_clean = df_clean.drop(columns=to_drop, errors="ignore")
    return df_clean, to_drop, report
