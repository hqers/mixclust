# mixclust/reporting/save_artifacts.py
from __future__ import annotations
import os, json
import pandas as pd
from typing import Dict, Any

def save_json(path: str, obj: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def save_table(path: str, df: pd.DataFrame):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
