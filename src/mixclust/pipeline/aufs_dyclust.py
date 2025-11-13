# src/mixclust/pipeline/aufs_dyclust.py
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from mixclust.aufs_samba.api import run_aufs_samba, AUFSParams
from mixclust.pipeline.landmark_eval import evaluate_dataframe_phaseB
from mixclust.utils.cluster_adapters import auto_adapter

# Optional: saat ingin auto-K untuk Phase B (Engine C)
try:
    from mixclust.utils.controller import make_auto_cluster_fn
    _HAS_AUTO_CLUSTER = True
except Exception:
    _HAS_AUTO_CLUSTER = False


def _ensure_label_column(df: pd.DataFrame, label_col: Optional[str]) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Memisahkan label (kalau ada), mengembalikan (df_wo_label, y_text_or_none).
    Phase A (AUFS) tidak memakai label → harus dibuang dulu.
    """
    if label_col is not None and label_col in df.columns:
        y_text = df[label_col].astype(str).copy()
        return df.drop(columns=[label_col]).copy(), y_text
    return df.copy(), None


def _resolve_phaseB_cluster_fn_and_k(
    params: AUFSParams,
    fallback_fn: Callable[[pd.DataFrame, List[int], int, Optional[int]], np.ndarray],
    n_clusters_user: Optional[int],
) -> Tuple[Callable[[pd.DataFrame, List[int], int, Optional[int]], np.ndarray], Optional[int]]:
    """
    Menentukan cluster_fn dan K untuk Phase B.
    - Jika Engine C + auto_k=True dan make_auto_cluster_fn tersedia → pakai auto cluster function.
    - Jika tidak → pakai fallback_fn (default: auto_adapter) dan K = n_clusters_user.
    """
    try:
        mode = (params.engine_mode or "A").upper()
    except Exception:
        mode = "A"

    if (mode == "C") and getattr(params, "auto_k", False) and _HAS_AUTO_CLUSTER:
        # Auto-K di Phase B: _k_unused akan diabaikan oleh make_auto_cluster_fn
        cluster_fn = make_auto_cluster_fn(
            algorithms=params.auto_algorithms or ["kprototypes", "hac_gower"],
            c_range=range(params.c_min, params.c_max + 1),
            metric="auto",
            penalty_lambda=0.02,
            random_state=params.random_state,
        )
        # n_clusters akan diabaikan oleh make_auto_cluster_fn, isi dummy aman
        return cluster_fn, 2
    else:
        return fallback_fn, n_clusters_user


def run_aufs_with_landmark_eval(
    df_input: pd.DataFrame,
    label_col: Optional[str],
    aufs_params: AUFSParams,
    n_clusters: int,
    *,
    cluster_fn: Callable[[pd.DataFrame, List[int], int, Optional[int]], np.ndarray] = auto_adapter,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Pipeline praktis:
      Phase A: AUFS-Samba (feature selection) — tidak memakai label.
      Phase B: Landmark-based evaluation (L-Sil_proto, LNC*, SS_Gower) pada subset fitur terpilih.
    Return:
      {
        "selected_cols": List[str],
        "aufs_info": dict,
        "eval_result": dict
      }
    """
    if not isinstance(df_input, pd.DataFrame) or df_input.empty:
        raise ValueError("df_input harus berupa DataFrame dan tidak boleh kosong.")

    # --- Phase A: AUFS (drop label terlebih dahulu) ---
    df_phaseA, _ = _ensure_label_column(df_input, label_col)

    if verbose:
        print(f"[Phase A] AUFS-Samba: n={len(df_phaseA)}, d={df_phaseA.shape[1]} (label dibuang: {bool(label_col)})")

    selected_cols, aufs_info = run_aufs_samba(
        df_input=df_phaseA,
        n_clusters=n_clusters,
        cluster_fn=cluster_fn,     # clusterer untuk reward di Phase A (sesuai engine & params)
        params=aufs_params,
        verbose=verbose,
        return_info=True,
    )

    if verbose:
        print("\n[✔] AUFS selesai")
        print(f"  • Jumlah fitur terpilih: {len(selected_cols)}")
        print(f"  • Fitur: {selected_cols}")

    # --- Siapkan DF terpilih + label untuk Phase B ---
    if len(selected_cols) == 0:
        # fallback aman: bila tidak ada fitur terpilih, kembalikan info minimum
        return {
            "selected_cols": [],
            "aufs_info": aufs_info,
            "eval_result": {
                "Status": "No selected features",
                "N": int(len(df_input)),
            },
        }

    # Pastikan kolom ada di df_input (antisipasi preprocessing menghapus kolom non-eksis)
    cols_ok = [c for c in selected_cols if c in df_input.columns]
    if verbose and len(cols_ok) != len(selected_cols):
        missing = sorted(set(selected_cols) - set(cols_ok))
        print(f"[WARN] {len(missing)} fitur terpilih tidak ditemukan di df_input dan akan di-skip: {missing}")

    df_selected = df_input[cols_ok].copy()
    if label_col is not None and label_col in df_input.columns:
        df_selected[label_col] = df_input[label_col]

    # --- Phase B: Landmark-based evaluation pada subset terpilih ---
    #     Tentukan cluster_fn & K khusus Phase B (Engine C bisa auto-K)
    cluster_fn_B, k_for_B = _resolve_phaseB_cluster_fn_and_k(
        aufs_params, fallback_fn=cluster_fn, n_clusters_user=n_clusters
    )

    if verbose:
        mode = (aufs_params.engine_mode or "A").upper()
        print(f"[Phase B] Eval: mode={mode}, auto_k={getattr(aufs_params, 'auto_k', False)}")
        print(f"          cluster_fn={'auto_cluster_fn' if cluster_fn_B is not cluster_fn else 'same_as_phaseA'}; K={k_for_B}")

    eval_result = evaluate_dataframe_phaseB(
        df_selected,
        label_col=label_col,
        selected_cols=None,           # df_selected sudah dipersempit sebelumnya
        use_gt_labels=False,          # default: internal labels dari cluster_fn_B
        scaler_type="standard",
        unit_norm=True,
        landmark_mode="cluster_aware",
        lm_max_frac=aufs_params.lsil_m_frac if hasattr(aufs_params, "lsil_m_frac") else 0.2,
        lm_per_cluster=max(3, getattr(aufs_params, "lsil_per_cluster_min", 3)),
        central_frac=0.8,
        boundary_frac=0.2,
        knn_k_lnc=getattr(aufs_params, "k_lnc", 50),
        ss_max_n=aufs_params.ss_max_n,
        try_hnsw=True,
        cluster_fn=cluster_fn_B,
        n_clusters=k_for_B,
        random_state=aufs_params.random_state,
        verbose=verbose,
    )

    return {
        "selected_cols": cols_ok,
        "aufs_info": aufs_info,
        "eval_result": eval_result,
    }


def run_aufs_dyclust_end_to_end(
    df_input: pd.DataFrame,
    label_col: Optional[str],
    aufs_params: AUFSParams,
    n_clusters: int,
    *,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Wrapper sekali jalan:
      - Jalankan AUFS + Phase B
      - Satukan ringkasan praktis (termasuk delta L-Sil vs SS)
    """
    out = run_aufs_with_landmark_eval(
        df_input=df_input,
        label_col=label_col,
        aufs_params=aufs_params,
        n_clusters=n_clusters,
        cluster_fn=auto_adapter,
        verbose=verbose,
    )

    # Ringkas beberapa metrik kunci dari eval_result
    eval_res = (out.get("eval_result") or {})
    summary = {
        "k_selected": len(out.get("selected_cols", [])),
        "L-Sil_proto": eval_res.get("L-Sil_proto"),
        "LNC*": eval_res.get("LNC*"),
        "SS_Gower": eval_res.get("SS_Gower"),
        "MAE": eval_res.get("MAE"),
        "Runtime_phaseB_s": eval_res.get("Runtime(s)"),
    }
    out["summary"] = summary
    return out
