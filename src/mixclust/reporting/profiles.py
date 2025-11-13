# src/mixclust/reporting/profiles.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from ..utils.cluster_profiling import profile_clusters

def build_profiles_table(df: pd.DataFrame, labels: np.ndarray, best_cols: List[str]) -> pd.DataFrame:
    cat_cols = [c for c in best_cols if df[c].dtype.name in ("object","category","bool")]
    prof = profile_clusters(df[best_cols].copy(), labels=labels, cat_cols=cat_cols, topk=8)
    # flatten ringkas: ukuran & mean indikator yang umum
    rows = []
    size = prof["size"]
    for k, n in sorted(size.items(), key=lambda x: int(x[0])):
        row = {"cluster": k, "size": n, "fraction": prof["fraction"].get(str(k), prof["fraction"].get(k, 0.0))}
        for key in ("share_staples","share_protein","share_fnv","share_processed",
                    "energy_burden","nonfood_burden","food_diversity_index"):
            v = prof.get("numeric", {}).get(key, {}).get(k, {}).get("mean", np.nan)
            row[key] = v
        rows.append(row)
    return pd.DataFrame(rows)
