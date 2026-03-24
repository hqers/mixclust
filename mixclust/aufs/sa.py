# mixclust/aufs/sa.py
from __future__ import annotations
import math, numpy as np
from typing import Callable, Dict, List, Optional, Tuple
try:
    from tqdm.auto import tqdm   # tampil bagus di notebook & terminal
except Exception:
    def tqdm(x, **k):            # fallback diam (no-op) kalau tqdm gagal
        return x


def generate_neighbors(
    subset: List[str],
    all_feats: List[str],
    mode: str = "swap",
    *,
    min_size: int = 1,         # <<< NEW: batas minimal ukuran subset
    max_size: Optional[int] = None  # <<< NEW: batas maksimal (opsional)
):
    """
    Hasilkan tetangga dengan menjaga ukuran subset di dalam [min_size, max_size].
    - mode="swap": hanya swap (ukuran tetap)
    - mode="full": swap + add + drop (tapi drop tidak boleh < min_size; add tidak boleh > max_size)
    """
    res = []
    current = list(subset)
    outside = sorted(set(all_feats) - set(current))  # stabil

    # --- SWAP (selalu aman utk ukuran) ---
    if mode in ("swap", "full"):
        for f_out in outside:
            for f_in in sorted(current):
                if f_out == f_in:
                    continue
                nb = current.copy(); nb.remove(f_in); nb.append(f_out)
                res.append(("swap", f"{f_in}<->{f_out}", nb))

    if mode == "full":
        # --- ADD: hanya jika belum mencapai max_size (jika max_size diset) ---
        if (max_size is None) or (len(current) < max_size):
            for f_out in outside:
                nb = current + [f_out]
                # cek batas atas
                if (max_size is None) or (len(nb) <= max_size):
                    res.append(("add", f"+{f_out}", nb))

        # --- DROP: hanya jika hasil masih >= min_size ---
        if len(current) > min_size:
            for f_in in current:
                nb = current.copy(); nb.remove(f_in)
                if len(nb) >= min_size:
                    res.append(("drop", f"-{f_in}", nb))

    return res


def simulated_annealing(
    subset_init: List[str],
    all_features: List[str],
    eval_reward: Callable[[List[str]], float],
    iters: int = 300,
    T0: float = 1.0,
    Tmin: float = 1e-3,
    alpha: float = 0.95,
    rng=None,
    neighbor_mode: str = "swap",               # 'swap' | 'full'
    exploit_rate: Optional[float] = None,      # e.g. 0.3 → evaluasi 30% tetangga saat eksploitasi
    show_progress: bool = True,
    reward_cache: Optional[Dict[Tuple[str, ...], float]] = None,
    cache_key_mode: str = "sorted",            # 'sorted' | 'ordered'
    # >>> NEW: kontrol ukuran subset
    min_size: int = 1,                         # <<< set ke 2 jika ingin hindari subset size=1
    max_size: Optional[int] = None
) -> Tuple[List[str], float, Dict[str, int]]:
    """
    SA dengan kontrol ukuran subset: min_size <= |subset| <= max_size (jika diset).
    """
    cache: Dict[Tuple[str, ...], float] = {} if reward_cache is None else reward_cache

    def key(cols: List[str]) -> Tuple[str, ...]:
        return tuple(sorted(cols)) if cache_key_mode == "sorted" else tuple(cols)

    cache_hits = 0
    cache_miss = 0

    def get_reward(cols: List[str]) -> float:
        nonlocal cache_hits, cache_miss
        # pastikan cols memenuhi batas ukuran
        if (len(cols) < min_size) or (max_size is not None and len(cols) > max_size):
            return float("-inf")
        k = key(cols)
        v = cache.get(k)
        if v is None:
            v = eval_reward(cols)
            cache[k] = v
            cache_miss += 1
        else:
            cache_hits += 1
        return v

    if rng is None:
        import numpy as np
        rng = np.random.default_rng(42)

    # --- INIT: pastikan ukuran minimal terpenuhi ---
    current = list(subset_init) if subset_init else []
    pool = [f for f in all_features if f not in current]

    # kalau kosong total, ambil acak minimal min_size
    if not current and all_features:
        take = max(min_size, 1)
        take = min(take, len(all_features))
        current = rng.choice(all_features, size=take, replace=False).tolist()

    # jika current < min_size, tambah acak dari pool
    while len(current) < min_size and pool:
        pick = rng.integers(0, len(pool))
        current.append(pool.pop(pick))

    if not current:
        return [], float("-inf"), {"iters": 0, "cache_hits": 0, "cache_miss": 0, "cache_size": 0}

    current_reward = get_reward(current)
    best = list(current)
    best_reward = current_reward

    T = T0
    done_iters = 0
    pbar = tqdm(range(iters), desc="SA", total=iters, leave=True) if show_progress else range(iters)

    import time
    t0 = time.time()

    for _ in pbar:
        done_iters += 1

        nbrs = generate_neighbors(
            current, all_features, mode=neighbor_mode,
            min_size=min_size, max_size=max_size
        )
        if not nbrs:
            break

        # Eksplorasi vs eksploitasi
        if (T > 0.1) and (exploit_rate is None):
            idx = rng.integers(0, len(nbrs))
            _, _, cand = nbrs[idx]
            r_cand = get_reward(cand)
        else:
            eval_list = nbrs
            if exploit_rate is not None and len(nbrs) > 1:
                k = max(1, int(len(nbrs) * float(exploit_rate)))
                pick = rng.choice(len(nbrs), size=k, replace=False)
                eval_list = [nbrs[i] for i in pick]

            r_best = float("-inf"); cand_best = None
            for _, _, cand in eval_list:
                r = get_reward(cand)
                if r > r_best:
                    r_best = r; cand_best = cand
            r_cand = r_best; cand = cand_best  # type: ignore

        if cand is None:
            T = max(T * alpha, Tmin)
            if T <= Tmin: break
            continue

        # Metropolis
        accept = False
        if r_cand > current_reward:
            accept = True
        else:
            delta = r_cand - current_reward
            if T > 1e-12 and math.exp(delta / T) > rng.random():
                accept = True

        if accept:
            current = list(cand)
            current_reward = r_cand
            if current_reward > best_reward:
                best_reward = current_reward
                best = list(current)

        T = max(T * alpha, Tmin)
        if T <= Tmin:
            break

        if show_progress:
            print(f"[SA] Iter {done_iters}: k={len(current):2d} | Current={current_reward:.4f} | "
                  f"New={r_cand:.4f} | Best={best_reward:.4f} | Accept={accept} | T={T:.4f}")
            t0 = time.time()
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix({
                    "k": len(current),
                    "best": f"{best_reward:.4f}",
                    "hit%": f"{(100.0*cache_hits/max(1, cache_hits+cache_miss)):.0f}"
                })

    stats = {
        "iters": done_iters,
        "cache_hits": cache_hits,
        "cache_miss": cache_miss,
        "cache_size": len(cache)
    }
    return best, float(best_reward), stats

