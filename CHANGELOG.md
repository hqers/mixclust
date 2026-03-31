# CHANGELOG — mixclust

## v1.1.7 (2026-03-31)

### New: `phase_b_skip_lnc` parameter
- `controller.py`: parameter `skip_lnc` di `auto_select_algo_k()` dan `_eval_with_phase_a_cache()`
- Jika `True`, LNC* tidak dihitung per trial Phase B → hemat ~30s/trial
- LNC* tetap dihitung **sekali** di structural control akhir
- Composite score J = L-Sil saja saat skip (sudah di-handle `_compute_composite_score_J`)

### Wiring parameter lengkap
- `api.py`: `AUFSParams` ditambah `phase_b_skip_lnc: bool = False`
- `controller.py`: `find_best_clustering_from_subsets()` meneruskan `phase_b_skip_lnc` ke `auto_select_algo_k()`
- `controller.py`: `find_best_clustering_from_subsets()` juga meneruskan `lsil_topk` dari params

### Estimasi speedup (mode cepat, semua optimasi aktif)
```
SA          : 33 min → ~5 min   (calibrate_mode="none")
Phase B     : 3.1 jam → ~20 min (rerank_topk=6 + skip_lnc + subsample)
Total       : 3.8 jam → ~30 min
```

---

## v1.1.6 (2026-03-31)

### Phase B subsample evaluasi L-Sil
- `phase_a_cache.py`: tambah `build_phase_b_subsample()` — bangun subsample stratified ~30k rows sekali, dipakai ulang setiap trial Phase B
- `phase_a_cache.py`: `_extract_phase_a_cache()` menerima `phase_b_eval_n` parameter
- `controller.py`: `_eval_with_phase_a_cache()` gunakan subsample jika tersedia, fallback ke full data jika tidak
- `reward.py`: inject `random_state` ke `__phase_a_cache__` dict (kedua cabang: `lsil_fixed` dan `lsil_fixed_calibrated`)
- `api.py`: tambah `phase_b_eval_n: int = 30_000` di `AUFSParams`, wire ke `_extract_phase_a_cache()`

### Wiring parameter v2.2 ke `make_sa_reward()`
- `api.py`: tambah `lsil_eval_n`, `lsil_c_reward`, `subsample_n_cluster` di `AUFSParams`
- `api.py`: kedua pemanggilan `make_sa_reward()` (baris ~335 dan ~700) sudah di-wire

### Estimasi speedup Phase B
```
Per trial   : 96s → ~9s  (Gower pada 30k bukan 334k)
Phase B     : 4.8 jam → ~2.8 jam (LNC* masih pada full data)
```

---

## v1.1.5 (2026-03-31)

### 🔴 Fix kritis: `kprototypes_subsample_adapter` tidak di-import
- **Dampak**: semua trial kprototypes di Phase B **gagal diam-diam** (`NameError` → `except` → `return None` → skip). Hanya hac_gower/kmodes/auto yang berjalan.
- **Root cause**: `_run_algo()` memanggil `kprototypes_subsample_adapter(...)` tapi import block hanya punya `kprototypes_adapter`
- **Fix**: `controller.py` baris 32 — tambah `kprototypes_subsample_adapter` ke import

### Wiring parameter v2.2 (awal)
- `api.py`: tambah `lsil_eval_n`, `lsil_c_reward`, `subsample_n_cluster` di `AUFSParams`
- `api.py`: wire ke kedua `make_sa_reward()` call

### Housekeeping
- `__init__.py`: ekspor `lsil_using_landmarks`, version bump
- `pyproject.toml`: version bump

---

## v1.1.2 — v1.1.4 (2026-03-30, sesi sebelumnya)

### v1.1.2: Optimasi performa SA reward
- `reward.py` [A]: evaluasi L-Sil pada subsample 20k (bukan full 334k) → ~67x speedup per SA call
- `reward.py` [B]: initial clustering pada 6k subsample (bukan 20k) → build_reward ~120s → ~8s
- `redundancy.py`: hapus joblib, precompute premaps → 31×31 matrix ~1s (bukan ~40s)

### v1.1.1: Bug fix SA reward = -1.0
- `reward.py` [C]: fix pemanggilan `lsil_using_prototypes_gower` setelah refactor ke `lsil_using_landmarks` — argumen positional salah menyebabkan TypeError ter-catch, semua reward = -1.0
- `controller.py`: fix `_eval_with_phase_a_cache` — pakai `lsil_using_landmarks` + `lm_labels_new = labels_new[L_fixed]`
- `controller.py`: fix `score_internal` — pakai `lsil_using_landmarks` langsung

---

## v1.1.0 (2026-03-30)

- Refactor L-Sil dari prototype-based ke landmark-based (sesuai paper JDSA)
- `adaptive_landmark_count()`: `|L| = c√n` (Theorem 1)
- `lsil_using_landmarks()`: landmark sebagai referensi klaster, bukan prototype
- `lsil_using_prototypes_gower()`: backward-compat wrapper
- `PhaseACache`: infrastructure untuk Phase B cache reuse
- `_extract_phase_a_cache()`: ekstrak cache dari SA reward closure

---

## File yang Diubah (v1.1.7 vs v1.1.0)

| File | Path di repo |
|------|-------------|
| `controller.py` | `mixclust/clustering/controller.py` |
| `phase_a_cache.py` | `mixclust/aufs/phase_a_cache.py` |
| `reward.py` | `mixclust/aufs/reward.py` |
| `api.py` | `mixclust/api.py` |
| `__init__.py` | `mixclust/__init__.py` |
| `pyproject.toml` | `pyproject.toml` (root) |
| `redundancy.py` | `mixclust/aufs/redundancy.py` (v1.1.2, tidak berubah sejak itu) |
