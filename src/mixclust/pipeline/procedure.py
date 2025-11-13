# src/mixclust/pipeline/procedure.py
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd

from mixclust.pipeline.landmark_eval import evaluate_dataframe_phaseB
from mixclust.controller.dynamic import ControllerConfig, DynamicController
from mixclust.pipeline.calibration import calibrate_lsil_to_ss

@dataclass
class ProcedureConfig:
    max_rounds: int = 5
    patience: int = 2              # berhenti jika tak ada perbaikan N kali berturut
    improve_tol: float = 1e-3      # ambang perbaikan SS_Gower
    use_gt_labels: bool = False
    scaler_type: str = "standard"
    unit_norm: bool = True
    landmark_mode: str = "cluster_aware"
    lm_max_frac: float = 0.20
    lm_per_cluster: int = 5
    central_frac: float = 0.8
    boundary_frac: float = 0.2
    knn_k_lnc: int = 50
    ss_max_n: int = 2000
    try_hnsw: bool = True
    random_state: int = 42


def _eval_once(
    df: pd.DataFrame,
    label_col: Optional[str],
    cfg: ProcedureConfig,
    overrides: Optional[Dict[str, Any]] = None,
    cluster_fn=None,
    n_clusters: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """Wrapper evaluasi Phase-B dengan dukungan overrides parameter incremental."""
    params = dict(
        use_gt_labels=cfg.use_gt_labels,
        scaler_type=cfg.scaler_type,
        unit_norm=cfg.unit_norm,
        landmark_mode=cfg.landmark_mode,
        lm_max_frac=cfg.lm_max_frac,
        lm_per_cluster=cfg.lm_per_cluster,
        central_frac=cfg.central_frac,
        boundary_frac=cfg.boundary_frac,
        knn_k_lnc=cfg.knn_k_lnc,
        ss_max_n=cfg.ss_max_n,
        try_hnsw=cfg.try_hnsw,
        cluster_fn=cluster_fn,
        n_clusters=n_clusters,
        random_state=cfg.random_state,
        verbose=verbose
    )
    if overrides:
        params.update(overrides)

    out = evaluate_dataframe_phaseB(
        df=df,
        label_col=label_col,
        selected_cols=None,
        **params
    )
    # inject per_proto/m info utk controller
    out["per_proto"] = params.get("per_cluster_proto_if_many", 1 if out.get("Clusters", 0) >= 8 else 2)
    out["m_max_frac"] = params.get("lm_max_frac", cfg.lm_max_frac)
    out["k_lnc"] = params.get("knn_k_lnc", cfg.knn_k_lnc)
    return out


def run_landmark_controller_procedure(
    df: pd.DataFrame,
    *,
    label_col: Optional[str] = None,
    controller_cfg: Optional[ControllerConfig] = None,
    proc_cfg: Optional[ProcedureConfig] = None,
    cluster_fn=None,
    n_clusters: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Orchestrator iteratif Phase-B:
      1) Evaluate → 2) Controller.decide → 3) Apply overrides → loop
      Berhenti jika: 'finalize', tidak ada perbaikan (patience), atau max_rounds.
    """
    ctrl = DynamicController(controller_cfg or ControllerConfig())
    cfg = proc_cfg or ProcedureConfig()

    history: List[Dict[str, Any]] = []
    best: Dict[str, Any] = {}
    best_ss: float = -np.inf
    patience_left = cfg.patience
    overrides: Dict[str, Any] = {}

    t0 = time.time()
    for r in range(1, cfg.max_rounds + 1):
        if verbose:
            print(f"\n[PROCEDURE] Round {r}/{cfg.max_rounds}")

        # 1) evaluasi
        res = _eval_once(
            df=df,
            label_col=label_col,
            cfg=cfg,
            overrides=overrides,
            cluster_fn=cluster_fn,
            n_clusters=n_clusters,
            verbose=verbose
        )
        history.append({"round": r, **res})

        ss = float(res.get("SS_Gower", -np.inf))
        lsil = float(res.get("L-Sil_proto", np.nan))
        if verbose:
            print(f"    SS={ss:.6f} | L-Sil={lsil:.6f} | MAE={abs(lsil-ss):.6f} | m={res.get('m')} | C={res.get('Clusters')}")

        # update best
        if ss > best_ss + cfg.improve_tol:
            best_ss = ss
            best = res
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left < 0:
                if verbose:
                    print("    ⏹ Berhenti (tidak ada perbaikan, patience habis).")
                break

        # 2) keputusan controller
        action = ctrl.decide({
            "N": res.get("N"), "Clusters": res.get("Clusters"),
            "m": res.get("m"),
            "L-Sil_proto": res.get("L-Sil_proto"),
            "LNC*": res.get("LNC*"),
            "SS_Gower": res.get("SS_Gower"),
            "MAE": res.get("MAE"),
            "k_lnc": res.get("k_lnc", cfg.knn_k_lnc),
            "per_proto": res.get("per_proto", 1),
            "m_max_frac": res.get("m_max_frac", cfg.lm_max_frac),
        })
        if verbose:
            print(f"    → Action: {action['action']} | {action['reason']}")

        if action["action"] == "finalize":
            if verbose:
                print("    ✅ Finalize diterapkan.")
            best["history"] = history
            best["ProcedureRuntime(s)"] = round(time.time() - t0, 2)
            return best

        # 3) apply overrides utk iterasi berikut
        overrides.update(action.get("overrides", {}))

        # sinkronkan cfg kalau override global (opsional)
        if "lm_max_frac" in overrides:
            cfg.lm_max_frac = float(overrides["lm_max_frac"])
        if "knn_k_lnc" in overrides:
            cfg.knn_k_lnc = int(overrides["knn_k_lnc"])

    # selesai oleh batas iterasi
    best["history"] = history
    best["ProcedureRuntime(s)"] = round(time.time() - t0, 2)
    return best


def quick_calibration_summary(results_table: pd.DataFrame) -> Dict[str, Any]:
    """
    Ringkas kalibrasi linear L-Sil → SS berdasarkan kumpulan hasil (multi dataset).
    """
    L = results_table["L-Sil_proto"].astype(float).values
    S = results_table["SS_Gower"].astype(float).values
    a, b, r = calibrate_lsil_to_ss(L, S)
    return {
        "a": float(a), "b": float(b), "corr": float(r),
        "SS_hat_example": float(a * np.mean(L) + b)
    }
