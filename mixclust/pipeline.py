# mixclust/pipeline.py
# Updated: expose hac_mode, cluster_adapter_lambda, enable_screening
from __future__ import annotations
import os, json, time
from typing import Optional, Dict, Any, List, Union
import pandas as pd
import numpy as np

from .api import run_aufs_samba, AUFSParams
from .utils.dqc import run_dqc
from .reporting.profiles import build_profiles_table
from .clustering.cluster_profiling import profile_clusters


def _sanitize_for_json(obj):
    """Rekursif ganti nan/inf dengan None agar JSON valid (RFC 8259)."""
    if isinstance(obj, float):
        if obj != obj or obj == float('inf') or obj == float('-inf'):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if (v != v or v == float('inf') or v == float('-inf')) else v
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _sanitize_for_json(obj.tolist())
    return obj


def _safe_json_dump(obj, f, **kwargs):
    json.dump(_sanitize_for_json(obj), f, **kwargs)

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
    n_clusters_hint: Optional[int] = None,  # v1.1.12: None = auto (midpoint of c_range)
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

    # v1.1.12: auto-compute n_clusters_hint if not provided
    # Uses midpoint of [c_min, c_max] — unbiased toward any K,
    # consistent with the auto-K search range in params.
    # If landmark_mode="kcenter" (auto for n>10K), hint only affects
    # labels0, not L_fixed — so midpoint is safe and principled.
    if n_clusters_hint is None:
        _p = params if params is not None else AUFSParams()
        _c_min = getattr(_p, 'c_min', 2)
        _c_max = getattr(_p, 'c_max', 8)
        n_clusters_hint = _c_min + (_c_max - _c_min) // 2
        if verbose:
            print(f"[pipeline] n_clusters_hint=auto → {n_clusters_hint} "
                  f"(midpoint of [{_c_min},{_c_max}])")

    # ── Data Quality Check — sebelum AUFS-Samba ─────────────────────────
    # Deteksi dan drop fitur zero-variance / near-zero sebelum masuk ke
    # reward landscape AUFS-Samba. Fitur seperti ini tidak terdeteksi oleh
    # reward karena tidak menurunkan L-Sil, tapi mengecilkan bobot fitur
    # lain via normalisasi Gower /p (contoh: AccessCommunication Susenas).
    feat_df, _dqc_dropped, _dqc_report = run_dqc(
        feat_df,
        zero_var_action="drop",
        near_zero_action="drop",
        near_zero_threshold=0.998,
        missing_action="warn",
        missing_threshold=0.5,
        verbose=verbose,
    )
    if _dqc_dropped:
        drops.update(_dqc_dropped)

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

    # Simpan DQC report
    _dqc_report.to_csv(os.path.join(outdir, "dqc_report.csv"), index=False)

    # Simpan config & features
    with open(os.path.join(outdir, "config.json"), "w", encoding="utf-8") as f:
        _safe_json_dump(info.get("params", {}), f, ensure_ascii=False, indent=2)
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
        _safe_json_dump(prof, f, ensure_ascii=False, indent=2)

    table = build_profiles_table(df_ready, labels, best_cols)
    table.to_csv(os.path.join(outdir, "profiles_table.csv"), index=False)

    # ── profiles.md — generik, pakai kolom yang benar-benar ada ──
    num_cols_best  = [c for c in best_cols if c not in cat_cols_list]
    cat_cols_best  = [c for c in best_cols if c in cat_cols_list]

    desc_lines = [
        "# Cluster Profiles\n",
        f"Dataset  : {len(df_ready):,} rows × {df_ready.shape[1]} cols  ",
        f"Features : {len(best_cols)} selected ({len(num_cols_best)} numeric, {len(cat_cols_best)} categorical)  ",
        f"K        : {len(set(labels))} clusters\n",
    ]

    # Ambil top-1 kategori per fitur kategorik dari prof
    prof_cat = prof.get("categorical", {})
    prof_num = prof.get("numeric", {})

    for k in sorted(set(labels)):
        frac = float(prof["fraction"].get(str(k), prof["fraction"].get(k, 0))) * 100
        size = int(prof["size"].get(str(k), prof["size"].get(k, 0)))
        parts = []

        # Numerik — tampilkan mean dan diff vs global
        for col in num_cols_best:
            d = prof_num.get(col, {}).get(str(k)) or prof_num.get(col, {}).get(k)
            if d and d.get("mean") is not None:
                diff = d.get("diff_vs_global", 0) or 0
                sign = "+" if diff >= 0 else ""
                parts.append(f"{col}={d['mean']:.2f}({sign}{diff:.2f})")

        # Kategorik — tampilkan top category dengan lift tertinggi
        for col in cat_cols_best:
            cd = prof_cat.get(col, {}).get(str(k)) or prof_cat.get(col, {}).get(k)
            if cd and cd.get("top"):
                top_cat = max(cd["top"], key=lambda x: cd["top"][x] or 0)
                top_lift = cd["top"][top_cat]
                if top_lift and top_lift > 1.0:
                    parts.append(f"{col}={top_cat}(lift={top_lift:.2f})")

        desc_lines.append(
            f"- Cluster {k} ({frac:.1f}%, n={size:,}): "
            + (", ".join(parts) if parts else "no dominant features")
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
        _safe_json_dump(metrics, f, ensure_ascii=False, indent=2)

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
        # ── v1.1.9: expose key metrics langsung agar tidak perlu buka JSON ──
        "best_K":     metrics.get("best_K"),
        "final_algo": metrics.get("final_algo"),
        "dav":        metrics.get("dav"),        # None jika DAV tidak aktif
    }