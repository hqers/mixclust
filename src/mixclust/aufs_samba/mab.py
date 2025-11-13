# dynamic_clustering/src/mixclust/aufs_samba/mab.py
from __future__ import annotations
import math, random
from typing import Callable, Dict, List, Tuple
import numpy as np
import pandas as pd
try:
    from tqdm.auto import tqdm   # tampil bagus di notebook & terminal
except Exception:
    def tqdm(x, **k):            # fallback diam (no-op) kalau tqdm gagal
        return x

def mab_explore(
    df: pd.DataFrame,
    reward_fn: Callable[[pd.DataFrame], float],
    T: int, k: int, rng: random.Random,
    red_matrix=None, red_threshold: float = 0.9,
    penalty_beta: float = 1.5, show_progress: bool = True
) -> Tuple[List[Tuple[List[str], float]], Dict[str, Dict[str, int]]]:
    feats = df.columns.tolist()
    stats = {f: {"bad": 0, "tot": 0} for f in feats}
    cache = {}
    out = []
    warmup = max(1, int(0.2 * T))

    for t in tqdm(range(T), desc="MAB", total=T, leave=True, mininterval=0.3, disable=not show_progress):
        w = []
        for f in feats:
            bad = stats[f]["bad"]; tot = max(1, stats[f]["tot"])
            w.append(math.exp(-penalty_beta * (bad / tot)))
        s = sum(w); w = [wi/s for wi in w] if s>1e-12 else [1/len(feats)]*len(feats)

        subset, tries = [], 0
        while len(subset) < min(k, len(feats)) and tries < 50*k:
            f = rng.choices(feats, weights=(None if t < warmup else w), k=1)[0]
            if f in subset: tries += 1; continue
            if red_matrix:
                if any(red_matrix.get(f, {}).get(g, 0.0) > red_threshold for g in subset):
                    tries += 1; continue
            subset.append(f); tries += 1
        if not subset: continue

        key = tuple(sorted(subset))
        rew = cache.get(key)
        if rew is None:
            rew = reward_fn(df[list(subset)])
            cache[key] = rew

        out.append((subset, rew))
        for f in subset:
            stats[f]["tot"] += 1
            if rew < np.mean([r for ss, r in out if f in ss]):  # heuristik “jelek”
                stats[f]["bad"] += 1
    return out, stats
