# mixclust/core/prototypes.py
import numpy as np
from .gower import gower_to_one_mixed
def mean_gower_to_set(i, set_idx, X_num, X_cat, num_min, num_max,
                      feature_mask_num=None, feature_mask_cat=None, inv_rng=None): 
    if len(set_idx) == 0:
        return 0.0
    d = gower_to_one_mixed(X_num, X_cat, num_min, num_max, int(i), np.asarray(set_idx, dtype=int),
                           feature_mask_num=feature_mask_num, feature_mask_cat=feature_mask_cat, inv_rng=inv_rng)
    return float(np.mean(d))

def agg_to_set(i, set_idx, X_num, X_cat, num_min, num_max,
               feature_mask_num=None, feature_mask_cat=None,
               inv_rng=None, mode="topk", topk=1): 
    """
    Agregasi jarak Gower titik i → himpunan set_idx.
    mode: "mean" | "min" | "topk"  (default: topk=1)
    """
    if len(set_idx) == 0:
        return 0.0
    d = gower_to_one_mixed(
        X_num, X_cat, num_min, num_max, int(i), np.asarray(set_idx, dtype=int),
        feature_mask_num=feature_mask_num, feature_mask_cat=feature_mask_cat,
        inv_rng=inv_rng
    )
    if mode == "min":
        return float(np.min(d))
    elif mode == "topk":
        k = int(min(topk, len(d)))
        part = np.partition(d, k-1)[:k]
        return float(np.mean(part))
    else:  # "mean"
        return float(np.mean(d))

def build_prototypes_by_cluster_gower(labels, X_num, X_cat, num_min, num_max,
                                      per_cluster=2, sample_cap=400, seed=42,
                                      feature_mask_num=None, feature_mask_cat=None, inv_rng=None): 
    """
    Ambil sampai 'per_cluster' medoid per klaster dengan sampel dalam-klaster (≤ sample_cap).
    Heuristik: medoid#1 = argmin(sum jarak), medoid#2 = kandidat terjauh dari medoid#1 (opsional).
    """
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)
    protos = {}
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            protos[c] = []
            continue
        if len(idx) > sample_cap:
            idx = np.sort(rng.choice(idx, size=sample_cap, replace=False).astype(int))

        # hitung jarak total tiap kandidat ke semua kandidat (chunked)
        # untuk efisiensi, kita estimasi dengan 64 anchor acak
        anchors = idx if len(idx) <= 64 else np.sort(rng.choice(idx, size=64, replace=False).astype(int))
        total = np.zeros(len(idx), dtype=np.float32)
        for a in anchors:
            d = gower_to_one_mixed(X_num, X_cat, num_min, num_max, int(a), idx,
                                   feature_mask_num=feature_mask_num, feature_mask_cat=feature_mask_cat, inv_rng=inv_rng)
            total += d
        med1 = int(idx[np.argmin(total)])  # medoid utama

        chosen = [med1]
        if per_cluster >= 2 and len(idx) >= 2:
            # cari titik paling jauh rata-rata dari med1 (diversity)
            d_med1 = gower_to_one_mixed(X_num, X_cat, num_min, num_max, med1, idx,
                                        feature_mask_num=feature_mask_num, feature_mask_cat=feature_mask_cat, inv_rng=inv_rng)
            med2 = int(idx[np.argmax(d_med1)])
            if med2 != med1:
                chosen.append(med2)

        if per_cluster >= 3 and len(idx) >= 3:
            # tambah satu lagi: farthest-first dari set chosen
            dist_sum = np.zeros(len(idx), dtype=np.float32)
            for ch in chosen:
                d_ch = gower_to_one_mixed(X_num, X_cat, num_min, num_max, ch, idx,
                                          feature_mask_num=feature_mask_num, feature_mask_cat=feature_mask_cat, inv_rng=inv_rng)
                dist_sum += d_ch
            med3 = int(idx[np.argmax(dist_sum)])
            for m in (med3,):
                if m not in chosen:
                    chosen.append(m)

        protos[c] = chosen[:per_cluster]
    return protos
