# CHANGELOG — mixclust

## v1.1.9 (2026-04-04)

### Fix: DAV Phase B memilih fallback padahal DAV winner tersedia

**Root cause — dua masalah terpisah:**

**Masalah A — apple vs orange comparison:**
`find_best_clustering_dav` membandingkan `score_adj` secara langsung antara:
- DAV winner: `score_adj = LNC*_a(Va)` — mengukur kohesi lokal di ruang Va
- Fallback: `score_adj = LNC*(S*)` — mengukur kohesi di ruang S* penuh

Keduanya mengukur hal berbeda dengan skala berbeda. Pada Susenas:
- Subset 1 DAV winner: LNC*_a = 0.4033, K=3
- Subset 2 fallback: score_adj = 0.6896, K=2
- Hasil: fallback menang → K*=2, padahal DAV berhasil menemukan K=3

**Masalah B — threshold terlalu ketat:**
`lnc_anchor_threshold=0.40` terlalu tinggi untuk data real-world berskala besar.
LNC*_a mengukur kohesi di ruang Va yang lebih sempit dari S*, sehingga skor
absolut lebih rendah. Subset 1 Susenas menghasilkan 0.4033 — hanya 0.0033
di atas threshold, hampir gagal karena noise numerik.

**Fix v1.1.9:**

1. **`find_best_clustering_dav`: fair comparison dengan `_should_update()`**
   - DAV winner selalu diprioritaskan atas fallback — tidak bandingkan score
   - DAV vs DAV: bandingkan LNC*_a
   - Fallback vs fallback: bandingkan score_adj (perilaku lama)
   - Fallback tidak bisa mengalahkan DAV winner

2. **`lnc_anchor_threshold` default: 0.40 → 0.25**
   Nilai 0.25 lebih realistis untuk LNC*_a pada data campuran berskala besar.
   Threshold lama (0.40) bisa tetap dipakai via parameter eksplisit jika diinginkan.

3. **`_AnchorContext`: log Va_valid vs Va_requested**
   Sekarang menampilkan variabel mana yang ditemukan vs tidak ada di subset,
   memudahkan debugging ketika subset tidak mengandung semua anchor variable.

4. **`auto_select_algo_k_dav`: log best LNC*_a saat fallback**
   Saat tidak ada K yang lulus, tampilkan best LNC*_a yang ditemukan dan
   threshold yang berlaku, agar pengguna bisa menyesuaikan threshold.

**Kasus nyata yang diperbaiki:**
Susenas Skenario B (seed=42) — subset 1 mengandung DDS12+DDS13:
- Sebelum fix: K*=2 (fallback menang dengan score 0.6896)
- Setelah fix: K*=3 (DAV winner dengan LNC*_a=0.4033 diprioritaskan)

### Fix: `run_generic_end2end` return dict tidak menyertakan hasil clustering

`pipeline.py` mengembalikan path file saja — `best_K`, `final_algo`, dan `dav`
tidak tersedia di return dict, sehingga notebook harus membuka `metrics_internal.json`
secara terpisah dan rawan `NameError` / `KeyError`.

**Fix:** Tambahkan tiga key ke return dict `run_generic_end2end`:
```python
"best_K":     metrics.get("best_K"),
"final_algo": metrics.get("final_algo"),
"dav":        metrics.get("dav"),   # None jika DAV tidak aktif
```

### File yang berubah

| File | Path |
|------|------|
| `dav.py` | `mixclust/utils/dav.py` |
| `pipeline.py` | `mixclust/pipeline.py` |
| `__init__.py` | `mixclust/__init__.py` |
| `pyproject.toml` | `pyproject.toml` |

### Git commit
```bash
git add mixclust/utils/dav.py mixclust/pipeline.py mixclust/__init__.py pyproject.toml
git commit -m "fix: DAV Phase B prioritaskan DAV winner atas fallback, threshold 0.40→0.25 (v1.1.9)

- find_best_clustering_dav: _should_update() — DAV winner tidak dibandingkan
  langsung dengan fallback score (apple vs orange)
- lnc_anchor_threshold default: 0.40 → 0.25 (lebih realistis untuk skala besar)
- _AnchorContext: log Va_valid vs Va_requested
- auto_select_algo_k_dav: log best LNC*_a saat fallback agar mudah tuning"
git tag v1.1.9
git push && git push --tags
```

---

## v1.1.8 (2026-04-01)

### Fix: DAV Phase B stuck / sangat lambat
**Root cause:** 3 bottleneck per trial DAV:
1. `kprototypes_adapter` pada full 334k rows (~60-120s)
2. `structural_control_lnc` bangun KNNIndex dari scratch per trial (~60-120s)
3. `lnc_star_anchored` bangun KNNIndex anchor dari scratch per trial (~30-60s)

Total per trial: ~150-300s × 96 trial = **4-8 jam** (atau hang)

**Fix v1.1.8:**
1. Clustering via `kprototypes_subsample_adapter` (6k) → ~5s
2. LNC* global via `phase_a_cache.knn_index` → ~15s (reuse, no rebuild)
3. Anchor KNN+landmark dibangun **SEKALI per subset** via `_AnchorContext` → ~10s per trial

**Estimasi:** Per trial 150s → ~30s. Phase B DAV: 4-8 jam → ~45 min.

### New: `_AnchorContext` class
- Cache KNN index + landmarks + Gower arrays di ruang Va
- Dibangun sekali di `auto_select_algo_k_dav`, reused per (algo, K) trial
- `_lnc_star_anchored_fast()` — menggunakan prebuilt context

### New: `_lnc_global_from_cache()`
- Compute LNC*(S*) guardrail menggunakan `phase_a_cache` yang sudah ada
- Tidak perlu bangun KNNIndex baru per trial

### KAMILA adapter (baru)
- `cluster_adapters.py`: `kamila_adapter()` dan `kamila_subsample_adapter()`
- `controller.py`: `_run_algo` mengenali `"kamila"` sebagai algoritma
- `dav.py`: `auto_select_algo_k_dav` mengenali `"kamila"`
- Opt-in via `auto_algorithms=["kprototypes", "hac_gower", "kamila"]`

### New: `run_dqc()` — Data Quality Check sebelum AUFS-Samba
- **`utils/dqc.py`**: modul baru, dipanggil otomatis di `pipeline.py`
- **Level 1 — Zero variance** (`zero_var_action="drop"`): drop otomatis
- **Level 2 — Near-zero variance** (`near_zero_action="drop"`, threshold 99.9%): drop otomatis
- **Level 3 — High missing** (`missing_action="warn"`, threshold 50%): warning saja
- Output: `dqc_report.csv` tersimpan di `outdir` setiap run

---

## v1.1.7 (2026-03-31)

### New: `phase_b_skip_lnc` parameter
- Skip LNC* per trial Phase B → hemat ~30s/trial
- LNC* tetap dihitung sekali di structural control akhir

---

## v1.1.6 (2026-03-31)

### Phase B subsample evaluasi L-Sil
- `phase_a_cache.py`: `build_phase_b_subsample()` — subsample ~30k rows
- `controller.py`: `_eval_with_phase_a_cache()` gunakan subsample
- `api.py`: `phase_b_eval_n: int = 30_000`

---

## v1.1.5 (2026-03-31)

### Fix kritis: `kprototypes_subsample_adapter` tidak di-import
- **Dampak:** semua kprototypes trial Phase B gagal diam-diam

---

## v1.1.2 — v1.1.4 (2026-03-30)

### v1.1.2: Optimasi performa
- `reward.py` [A]: L-Sil eval pada 20k subsample (~67x speedup per SA call)
- `reward.py` [B]: clustering pada 6k subsample (~15x speedup build_reward)
- `redundancy.py`: hapus joblib, precompute premaps (~40x speedup)

### v1.1.1: Bug fix SA reward = -1.0
- `reward.py` [C]: fix argumen `lsil_using_landmarks` setelah refactor

---

## v1.1.0 (2026-03-30)

- Refactor L-Sil: prototype → landmark (paper JDSA)
- `|L| = c√n` (Theorem 1), default c=3
- PhaseACache infrastructure
