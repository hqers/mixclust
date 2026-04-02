# CHANGELOG — mixclust

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
- **Level 1 — Zero variance** (`zero_var_action="drop"`):
  kolom dengan `nunique == 1` atau `std < 1e-6` → drop otomatis sebelum AUFS-Samba
- **Level 2 — Near-zero variance** (`near_zero_action="drop"`, threshold 99.9%):
  kolom di mana satu nilai mendominasi ≥ 99.9% baris → drop otomatis
- **Level 3 — High missing** (`missing_action="warn"`, threshold 50%):
  kolom dengan ≥ 50% missing → warning, tidak otomatis drop (keputusan domain)
- Output: `dqc_report.csv` tersimpan di `outdir` setiap run
- `if _dqc_dropped: drops.update(_dqc_dropped)` — sinkronkan ke drops set pipeline

**Motivasi:** Fitur zero-variance tidak terdeteksi oleh reward-based AUFS karena
reward tidak turun saat fitur ini ditambahkan — dia *netral*, bukan negatif.
Redundancy penalty juga tidak menangkapnya karena mengukur korelasi antar fitur,
bukan variasi internal. Akibatnya fitur bisa lolos ke `features.csv` tanpa
kontribusi diskriminatif dan mengecilkan bobot fitur lain via normalisasi Gower `/p`.

**Kasus nyata:** `AccessCommunication` di Susenas 2020 — semua 334,229 RT
bernilai "Tidak" → Cramér's V = 0.0000, masuk `features.csv` v1.1.6,
tidak terdeteksi sampai post-hoc profile inspection.

### File yang berubah

| File | Path |
|------|------|
| `dav.py` | `mixclust/utils/dav.py` |
| `cluster_adapters.py` | `mixclust/clustering/cluster_adapters.py` |
| `controller.py` | `mixclust/clustering/controller.py` |
| `phase_a_cache.py` | `mixclust/aufs/phase_a_cache.py` |
| `reward.py` | `mixclust/aufs/reward.py` |
| `api.py` | `mixclust/api.py` |
| `kamila.py` | `mixclust/clustering/kamila.py` (BARU) |
| `__init__.py` | `mixclust/__init__.py` |
| `pipeline.py` | `mixclust/pipeline.py` |
| `dqc.py` | `mixclust/utils/dqc.py` (BARU) |
| `pyproject.toml` | `pyproject.toml` |

### Git commit
```bash
git add mixclust/utils/dav.py mixclust/clustering/cluster_adapters.py \
        mixclust/clustering/controller.py mixclust/clustering/kamila.py \
        mixclust/aufs/phase_a_cache.py mixclust/aufs/reward.py \
        mixclust/api.py mixclust/__init__.py mixclust/pipeline.py \
        mixclust/utils/dqc.py pyproject.toml
git commit -m "fix+perf: DAV Phase B, KAMILA, DQC zero-variance detection (v1.1.8)

- dav.py: _AnchorContext prebuilt, _lnc_global_from_cache, subsample clustering
- cluster_adapters.py: kamila_adapter, kamila_subsample_adapter, updated auto_adapter
- controller.py: import fix, skip_lnc, subsample eval, kamila in _run_algo
- phase_a_cache.py: Phase B subsample infrastructure
- reward.py: random_state in cache dict
- api.py: all new params wired (phase_b_eval_n, phase_b_skip_lnc, v2.2 params)
- pipeline.py: DQC integrated (run_dqc before AUFS-Samba)
- utils/dqc.py: zero-variance + near-zero + high-missing detection"
git tag v1.1.8
git push && git push --tags
```

---

## v1.1.7 (2026-03-31)

### New: `phase_b_skip_lnc` parameter
- Skip LNC* per trial Phase B → hemat ~30s/trial
- LNC* tetap dihitung sekali di structural control akhir
- `controller.py`: `skip_lnc` param di `auto_select_algo_k()` dan `_eval_with_phase_a_cache()`
- `api.py`: `AUFSParams.phase_b_skip_lnc: bool = False`

### Estimasi (mode cepat, semua optimasi)
- SA: 33 min → ~5 min (calibrate_mode="none")
- Phase B: 3.1 jam → ~20 min (rerank_topk=6 + skip_lnc + subsample)
- Total: 3.8 jam → ~30 min

---

## v1.1.6 (2026-03-31)

### Phase B subsample evaluasi L-Sil
- `phase_a_cache.py`: `build_phase_b_subsample()` — subsample ~30k rows
- `controller.py`: `_eval_with_phase_a_cache()` gunakan subsample
- `reward.py`: inject `random_state` ke `__phase_a_cache__`
- `api.py`: `phase_b_eval_n: int = 30_000`

### Wiring parameter v2.2
- `api.py`: `lsil_eval_n`, `lsil_c_reward`, `subsample_n_cluster` di AUFSParams
- Kedua `make_sa_reward()` call di-wire

---

## v1.1.5 (2026-03-31)

### Fix kritis: `kprototypes_subsample_adapter` tidak di-import
- **Dampak:** semua kprototypes trial Phase B gagal diam-diam
- **Fix:** tambah import di `controller.py`

### Housekeeping
- `__init__.py`: ekspor `lsil_using_landmarks`

---

## v1.1.2 — v1.1.4 (2026-03-30)

### v1.1.2: Optimasi performa
- `reward.py` [A]: L-Sil eval pada 20k subsample (~67x speedup per SA call)
- `reward.py` [B]: clustering pada 6k subsample (~15x speedup build_reward)
- `redundancy.py`: hapus joblib, precompute premaps (~40x speedup)

### v1.1.1: Bug fix SA reward = -1.0
- `reward.py` [C]: fix argumen `lsil_using_landmarks` setelah refactor
- `controller.py`: fix `_eval_with_phase_a_cache` dan `score_internal`

---

## v1.1.0 (2026-03-30)

- Refactor L-Sil: prototype → landmark (paper JDSA)
- `|L| = c√n` (Theorem 1), default c=3
- PhaseACache infrastructure
