# Changelog v1.1.6 — Phase B Speedup + Kualitas Clustering

Mencakup semua fix dari v1.1.5.

## Perubahan Utama

### 🚀 Phase B Speedup: subsample evaluasi L-Sil (~11x lebih cepat)

**Masalah:** Phase B memakan 4.8 jam dari total 5.4 jam. Setiap trial (180 total)
memanggil `gower_distances_to_landmarks()` pada **full 334k rows × 1734 landmarks**
→ ~96s per trial.

**Fix:** Evaluasi L-Sil di Phase B menggunakan subsample 30k rows (stratified).
Subsample dibangun **SEKALI** di `_extract_phase_a_cache()` dan di-reuse setiap trial.

**File yang berubah:**
- `phase_a_cache.py` — tambah `build_phase_b_subsample()`, fields `_pb_*`
- `controller.py` — `_eval_with_phase_a_cache()` pakai subsample jika tersedia
- `reward.py` — inject `random_state` ke `__phase_a_cache__` dict
- `api.py` — wire `phase_b_eval_n` parameter

**Estimasi speedup:**
- Per trial: 96s → ~9s
- Phase B total: 4.8 jam → ~26 menit
- Total pipeline: 5.4 jam → ~1 jam

### 🔧 Fix dari v1.1.5 (included)

- `controller.py`: import `kprototypes_subsample_adapter` (Phase B kprototypes silent fail)
- `api.py`: wire `lsil_eval_n`, `lsil_c_reward`, `subsample_n_cluster` ke `AUFSParams`

---

## Tips untuk Meningkatkan Kualitas Clustering

Dari hasil run sebelumnya, 5 dari 7 fitur terpilih hampir tidak membedakan cluster.
Berikut rekomendasi parameter di notebook:

```python
params = AUFSParams(
    # PENTING: topk=3 sesuai paper (bukan 1)
    lsil_topk=3,

    # SA lebih banyak iterasi untuk eksplorasi lebih baik
    sa_iters=80,

    # Rentang K lebih lebar untuk auto-K
    c_min=2,
    c_max=7,

    # Phase B subsample (default sudah 30k, bisa dinaikkan jika mau akurasi lebih)
    phase_b_eval_n=30_000,
)
```

**Kenapa `lsil_topk=1` bermasalah:** Dengan topk=1, L-Sil hanya melihat 1 landmark
terdekat per cluster. Fitur yang hampir konstan (seperti AccessCommunication) tidak
menambah noise sehingga tidak menurunkan skor — tapi juga tidak menambah informasi.
Dengan topk=3 (paper default), evaluasi lebih robust dan lebih sensitif terhadap
fitur yang benar-benar membedakan cluster.

---

## File yang Diubah

| File | Path di repo | Perubahan |
|------|-------------|-----------|
| `phase_a_cache.py` | `mixclust/aufs/phase_a_cache.py` | Phase B subsample infrastructure |
| `controller.py` | `mixclust/clustering/controller.py` | Subsample eval + import fix |
| `reward.py` | `mixclust/aufs/reward.py` | Inject random_state ke cache dict |
| `api.py` | `mixclust/api.py` | Wire semua parameter baru |
| `__init__.py` | `mixclust/__init__.py` | Version bump + ekspor |
| `pyproject.toml` | `pyproject.toml` | Version bump |

## Git Commit

```bash
git add mixclust/aufs/phase_a_cache.py mixclust/aufs/reward.py \
        mixclust/clustering/controller.py mixclust/api.py \
        mixclust/__init__.py pyproject.toml
git commit -m "perf+fix: Phase B subsample eval (~11x speedup), wire params (v1.1.6)

- phase_a_cache.py: add build_phase_b_subsample() for Phase B eval on 30k rows
- controller.py: _eval_with_phase_a_cache uses subsample, fix import
- reward.py: inject random_state to __phase_a_cache__ dict
- api.py: add phase_b_eval_n to AUFSParams, wire all v2.2 params
- Estimated: Phase B 4.8h → ~26min, total 5.4h → ~1h"
git tag v1.1.6
git push && git push --tags
```
