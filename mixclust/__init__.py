"""
mixclust — Adaptive mixed-type clustering for household survey data.

Install:
    pip install git+https://github.com/hqers/mixclust.git

Quickstart:
    from mixclust import run_generic_end2end, AUFSParams
    import pandas as pd

    df = pd.read_csv('HH_CLUSTERING_ready.csv', sep=';')
    result = run_generic_end2end(df, outdir='output/', id_col='HHID')

Paper:
    AUFS-Samba: https://doi.org/10.1109/ACCESS.2026.3653624
"""

__version__ = "1.0.7"
__author__  = "Hasta Pratama"
__email__   = "33220015@std.stei.itb.ac.id"

# ── Public API ────────────────────────────────────────────────────
from .pipeline import run_generic_end2end               # noqa: F401
from .api import run_aufs_samba, AUFSParams             # noqa: F401
from .metrics.lsil import lsil_using_prototypes_gower  # noqa: F401
from .metrics.lnc_star import lnc_star                  # noqa: F401
from .reporting.profiles import build_profiles_table    # noqa: F401
from .clustering.cluster_profiling import profile_clusters  # noqa: F401
from .reporting.save_artifacts import save_json, save_table  # noqa: F401

__all__ = [
    "run_generic_end2end",
    "run_aufs_samba",
    "AUFSParams",
    "lsil_using_prototypes_gower",
    "lnc_star",
    "build_profiles_table",
    "profile_clusters",
    "save_json",
    "save_table",
]
