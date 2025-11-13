# src/mixclust/pipeline/generic_end2end.py
from __future__ import annotations
import os, json, time
from typing import Optional, Dict, Any, List, Union
import pandas as pd
import numpy as np

from mixclust.aufs_samba.api import run_aufs_samba, AUFSParams
from mixclust.reporting.profiles import build_profiles_table
from mixclust.utils.cluster_profiling import profile_clusters

# ---------- util kecil ----------
def _infer_id_col(df: pd.DataFrame, id_col: Optional[str]) -> Optional[str]:
    if id_col and id_col in df.columns:
        return id_col
    for c in ("HHID", "Id", "ID", "id", "household_id"):
        if c in df.columns:
            return c
    return None

def _cat_cols(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if df[c].dtype.name in ("object","category","bool")]

# ---------- runner generik ----------
def run_generic_end2end(
    df_ready: pd.DataFrame,
    outdir: str,
    *,
    id_col: Optional[str] = None,          # mis. "HHID" (opsional)
    drop_cols: Optional[List[str]] = None, # kolom non-fitur untuk dibuang sebelum AUFS
    params: Optional[AUFSParams] = None,
    n_clusters_hint: int = 3,              # dipakai hanya jika auto_k=False
    verbose: bool = True
) -> Dict[str, Any]:
    """
    End-to-end generik:
    - Input: df_ready (mixed-type, sudah bersih)
    - AUFS–DyClust (Engine C + auto-K default)
    - Simpan artefak: features.csv, cluster_assignments.csv, profiles.(json|md), metrics_internal.json, config.json
    """
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()

    # 1) siapkan kolom ID & fitur
    idc = _infer_id_col(df_ready, id_col)
    drops = set(drop_cols or [])
    if idc: drops.add(idc)
    feat_df = df_ready.drop(columns=[c for c in drops if c in df_ready.columns], errors="ignore").copy()

    # 2) default params: Engine C + auto-K
    params = params or AUFSParams(
        engine_mode="C",
        auto_k=True, c_min=2, c_max=6,
    
        # ⚡ PERCEPAT REWARD (Langkah #2)
        reward_metric="lsil_fixed_calibrated",   # ← was: lsil / silhouette_gower
        lsil_topk=1,
        per_cluster_proto_if_many=1,
        lsil_proto_sample_cap=200,
    
        ss_max_n=2000,
    
        # ⛔ MATIKAN RERANK (Langkah #5)
        enable_rerank=False,
        rerank_mode="none",
        rerank_topk=0,
        shadow_rerank=False,
    
        # ⚙️ REDUNDANSI (Langkah #1)
        kmsnc_k=5,
        build_redundancy_parallel=True,
        build_redundancy_cache="cache/redundancy_k5.pkl",
        mab_n_jobs=2,   # jangan -1 dulu biar tak OOM
    
        sa_neighbor_mode="full",
        sa_min_size=2, sa_max_size=None,
        mab_T=12, mab_k=6,
        random_state=42, verbose=verbose, show_progress=verbose
    )


    # 3) jalankan AUFS
    best_cols, info = run_aufs_samba(
        df_input=feat_df,
        n_clusters=n_clusters_hint,      # diabaikan kalau auto_k=True
        cluster_fn=None,                 # di-resolve internal oleh api.py (Engine C)
        params=params,
        verbose=verbose,
        return_info=True
    )

    # 4) simpan config & features
    with open(os.path.join(outdir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(info.get("params", {}), f, ensure_ascii=False, indent=2)
    pd.Series(best_cols, name="feature").to_csv(os.path.join(outdir, "features.csv"), index=False)

    # 5) ambil label final dan simpan assignment
    labels = info.get("final_labels")
    if labels is None:
        raise RuntimeError("final_labels kosong dari AUFS. Cek pipeline.")
    if idc and idc in df_ready.columns:
        assign = pd.DataFrame({idc: df_ready[idc].values, "cluster": labels})
    else:
        assign = pd.DataFrame({"row_id": np.arange(len(df_ready)), "cluster": labels})
    assign.to_csv(os.path.join(outdir, "cluster_assignments.csv"), index=False)

    # 6) profiling segmen (pola konsumsi) — generik
    sub = df_ready[best_cols].copy()
    sub["cluster"] = labels
    cat_cols = _cat_cols(sub, best_cols)

    # JSON profil lengkap (size, fraction, numeric means, cat lifts, p-values, dst.)
    prof = profile_clusters(sub.drop(columns=["cluster"]), labels=labels, cat_cols=cat_cols, topk=8)
    with open(os.path.join(outdir, "profiles.json"), "w", encoding="utf-8") as f:
        json.dump(prof, f, ensure_ascii=False, indent=2)

    # Tabel ringkas + narasi markdown
    table = build_profiles_table(df_ready, labels, best_cols)
    table.to_csv(os.path.join(outdir, "profiles_table.csv"), index=False)

    desc_lines = ["# Cluster Profiles (Generic)\n"]
    for _, r in table.sort_values("cluster").iterrows():
        k = int(r["cluster"]); frac = float(r.get("fraction", 0.0))*100
        picks = []
        for key in ("PlantProteinShare","FVShare","ProcessedShare","PreparedFoodShare","TobaccoShare",
                    "EnergyBurden","NonFoodBurden","FoodShare","DDS14_norm","DDS13_noTob_norm","DDS12_noTobPrep_norm"):
            if key in r and pd.notna(r[key]):
                picks.append(f"{key}≈{float(r[key]):.2f}")
        desc_lines.append(f"- Cluster {k} ({frac:.1f}%): " + (", ".join(picks) if picks else "mixed indicators"))
    with open(os.path.join(outdir, "profiles.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(desc_lines))

    # 7) metrik minimal (timing, algo, K)
    metrics = {
        "best_K": info.get("final_C"),
        "final_algo": info.get("final_algo"),
        "used_metric": info.get("used_metric"),
        "timing_s": info.get("timing_s", {}),
    }
    with open(os.path.join(outdir, "metrics_internal.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return {
        "outdir": outdir,
        "best_features": best_cols,
        "assign_path": os.path.join(outdir, "cluster_assignments.csv"),
        "profiles_json_path": os.path.join(outdir, "profiles.json"),
        "profiles_table_path": os.path.join(outdir, "profiles_table.csv"),
        "metrics_path": os.path.join(outdir, "metrics_internal.json"),
        "runtime_s": round(time.time() - t0, 2),
    }

if __name__ == "__main__":
    # contoh pakai CSV umum (opsional)
    # df = pd.read_csv("data/any_dataset.csv")
    # out = run_generic_end2end(df, outdir="runs/demo-generic", id_col="HHID")
    pass
