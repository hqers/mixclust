# mixclust/pipeline.py
# Updated: expose hac_mode, cluster_adapter_lambda, enable_screening
from __future__ import annotations
import os, json, time
from typing import Optional, Dict, Any, List, Union
import pandas as pd
import numpy as np

from .api import run_aufs_samba, AUFSParams
from .reporting.profiles import build_profiles_table
from .clustering.cluster_profiling import profile_clusters

def _infer_id_col(df: pd.DataFrame, id_col: Optional[str]) -> Optional[str]:
    if id_col and id_col in df.columns:
        return id_col
    for c in ("HHID", "Id", "ID", "id", "household_id"):
        if c in df.columns:
            return c
    return None

def _cat_cols(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if df[c].dtype.name in ("object","category","bool")]

def run_generic_end2end(
    df_ready: pd.DataFrame,
    outdir: str,
    *,
    id_col: Optional[str] = None,
    drop_cols: Optional[List[str]] = None,
    params: Optional[AUFSParams] = None,
    n_clusters_hint: int = 3,
    verbose: bool = True
) -> Dict[str, Any]:
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()

    idc = _infer_id_col(df_ready, id_col)
    drops = set(drop_cols or [])
    if idc: drops.add(idc)
    feat_df = df_ready.drop(
        columns=[c for c in drops if c in df_ready.columns],
        errors="ignore"
    ).copy()

    params = params or AUFSParams(
        engine_mode="C",
        auto_k=True, c_min=2, c_max=6,

        # Reward (Phase A)
        reward_metric="lsil_fixed_calibrated",
        lsil_topk=1,
        per_cluster_proto_if_many=1,
        lsil_proto_sample_cap=200,
        ss_max_n=2000,

        # Rerank dinonaktifkan
        enable_rerank=False,
        rerank_mode="none",
        rerank_topk=0,
        shadow_rerank=False,

        # Redundansi
        kmsnc_k=5,
        build_redundancy_parallel=True,
        build_redundancy_cache="cache/redundancy_k5.pkl",
        red_backend="loky",          # FIX: "auto" tidak valid di joblib
        mab_n_jobs=2,

        # SA
        sa_neighbor_mode="full",
        sa_min_size=2, sa_max_size=None,
        mab_T=12, mab_k=6,

        # ── FIX 1: HAC landmark-hybrid (sesuai disertasi Bab VI.6.4) ──
        # "hybrid" = HAC pada landmark saja, assignment via prototipe
        # "full_hac" = untuk ablation study Bab VI
        hac_mode="hybrid",

        # ── FIX 2: λ dalam J(algo) = λ·L-Sil + (1-λ)·LNC* ──
        # Sesuai disertasi Bab VI.6.3
        cluster_adapter_lambda=0.6,

        # ── FIX 3: Screening awal K ∈ {2,3,4} ──
        # Sesuai disertasi Bab VI.7.6 Langkah 2
        enable_screening=True,
        screening_k_values=(2, 3, 4),
        screening_prune_threshold=0.15,

        # Kandidat algoritma Phase B
        # kprototypes + hac_gower (akan diroute ke hybrid secara otomatis)
        auto_algorithms=["kprototypes", "hac_gower"],

        random_state=42, verbose=verbose, show_progress=verbose
    )

    best_cols, info = run_aufs_samba(
        df_input=feat_df,
        n_clusters=n_clusters_hint,
        cluster_fn=None,
        params=params,
        verbose=verbose,
        return_info=True
    )

    # Simpan config & features
    with open(os.path.join(outdir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(info.get("params", {}), f, ensure_ascii=False, indent=2)
    pd.Series(best_cols, name="feature").to_csv(
        os.path.join(outdir, "features.csv"), index=False
    )

    labels = info.get("final_labels")
    if labels is None:
        raise RuntimeError("final_labels kosong dari AUFS.")

    if idc and idc in df_ready.columns:
        assign = pd.DataFrame({idc: df_ready[idc].values, "cluster": labels})
    else:
        assign = pd.DataFrame({"row_id": np.arange(len(df_ready)), "cluster": labels})
    assign.to_csv(os.path.join(outdir, "cluster_assignments.csv"), index=False)

    sub = df_ready[best_cols].copy()
    sub["cluster"] = labels
    cat_cols_list = _cat_cols(sub, best_cols)

    prof = profile_clusters(
        sub.drop(columns=["cluster"]), labels=labels,
        cat_cols=cat_cols_list, topk=8
    )
    with open(os.path.join(outdir, "profiles.json"), "w", encoding="utf-8") as f:
        json.dump(prof, f, ensure_ascii=False, indent=2)

    table = build_profiles_table(df_ready, labels, best_cols)
    table.to_csv(os.path.join(outdir, "profiles_table.csv"), index=False)

    desc_lines = ["# Cluster Profiles (Generic)\n"]
    for _, r in table.sort_values("cluster").iterrows():
        k = int(r["cluster"]); frac = float(r.get("fraction", 0.0))*100
        picks = []
        for key in (
            "PlantProteinShare","FVShare","ProcessedShare","PreparedFoodShare",
            "TobaccoShare","EnergyBurden","NonFoodBurden","FoodShare",
            "DDS14_norm","DDS13_noTob_norm","DDS12_noTobPrep_norm"
        ):
            if key in r and pd.notna(r[key]):
                picks.append(f"{key}≈{float(r[key]):.2f}")
        desc_lines.append(
            f"- Cluster {k} ({frac:.1f}%): "
            + (", ".join(picks) if picks else "mixed indicators")
        )
    with open(os.path.join(outdir, "profiles.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(desc_lines))

    # Metrics — tambahkan info Phase B breakdown
    phase_b_info = {}
    if info.get("timing_s", {}).get("phaseB_s"):
        phase_b_info = {
            "phaseB_s": info["timing_s"]["phaseB_s"],
            "hac_mode_used": params.hac_mode,
            "composite_lambda": params.cluster_adapter_lambda,
            "screening_enabled": params.enable_screening,
        }

    # DAV info — masuk ke metrics jika DAV aktif
    _fb = info.get("phase_b_result", {}) or {}
    dav_info = {}
    if _fb.get("dav_applied"):
        dav_info = {
            "dav_applied":        True,
            "anchor_cols":        _fb.get("anchor_cols", []),
            "lnc_anchor":         _fb.get("lnc_score"),      # LNC*_a(Va) — objective DAV
            "lnc_global":         _fb.get("lnc_global"),     # LNC*(S*)   — guardrail
        }

    metrics = {
        "best_K": info.get("final_C"),
        "final_algo": info.get("final_algo"),
        "used_metric": info.get("used_metric"),
        "best_reward": info.get("best_reward"),          # L-Sil score (or SS_Gower if small dataset)
        "final_ss_gower": info.get("final_ss_gower"),   # SS_Gower recomputed after Phase B
        "structural_control": info.get("structural_control"),
        "timing_s": info.get("timing_s", {}),
        "phase_b_config": phase_b_info,
        "dav": dav_info or None,                         # None jika DAV tidak aktif
    }
    with open(os.path.join(outdir, "metrics_internal.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # ── Excel summary (satu baris per run, mudah dibandingkan antar seed/dataset) ──
    excel_row = {
        "dataset":        os.path.basename(outdir),
        "best_K":         metrics["best_K"],
        "final_algo":     metrics["final_algo"],
        "best_reward":    metrics["best_reward"],
        "final_ss_gower": metrics["final_ss_gower"],
        "lnc_anchor":     dav_info.get("lnc_anchor")  if dav_info else None,
        "lnc_global":     dav_info.get("lnc_global")  if dav_info else None,
        "dav_applied":    dav_info.get("dav_applied", False),
        "anchor_cols":    ", ".join(dav_info.get("anchor_cols", [])) if dav_info else "",
        "runtime_s":      round(time.time() - t0, 2),
    }
    xl_path = os.path.join(outdir, "summary.xlsx")
    pd.DataFrame([excel_row]).to_excel(xl_path, index=False)

    return {
        "outdir": outdir,
        "best_features": best_cols,
        "assign_path": os.path.join(outdir, "cluster_assignments.csv"),
        "profiles_json_path": os.path.join(outdir, "profiles.json"),
        "profiles_table_path": os.path.join(outdir, "profiles_table.csv"),
        "metrics_path": os.path.join(outdir, "metrics_internal.json"),
        "summary_xlsx_path": xl_path,
        "runtime_s": excel_row["runtime_s"],
    }