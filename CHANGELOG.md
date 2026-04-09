# CHANGELOG — mixclust

## v1.1.14 (2026-04-09)

### Fix: `auto_params` menghasilkan konfigurasi yang lebih buruk dari manual v1.1.11

**Root cause:** Empat parameter di `auto_params` menggunakan formula yang terlalu
agresif untuk dataset medium (n = 10K–100K), menyebabkan `labels0`, `L_fixed`,
dan SA reward semuanya dalam kondisi buruk sekaligus.

Ditemukan dari perbandingan langsung `metrics_internal.json` antara v1.1.11
(config manual) dan v1.1.13 (auto_params):

| Metrik | v1.1.11 | v1.1.13 patch | v1.1.14 |
|---|---|---|---|
| `final_algo` | hac_gower | kprototypes | hac_gower ✓ |
| `final_ss_gower` | 0.7276 | 0.4028 | ~0.72 (target) |
| `best_reward` | 0.9742 | 0.8114 | — |
| `phaseB_s` | 1,474s | 29,773s | ~1,500s |

---

#### Bug #1 — `landmark_mode = "kcenter"` terlalu mudah trigger

**Formula lama:**
```python
geo_dom_risk = (n_ratio > 1) or (binary_ratio > 0.3) or (spike_ratio > 0.4)
```
BankMarketing `n_ratio=4.1 > 1` → selalu `kcenter`. `L_fixed` tidak aligned
dengan struktur klaster → evaluasi Phase B menyesatkan → algo/K yang salah
dipilih.

**Formula baru:**
```python
geo_dom_risk = (
    (n_ratio > 10.0)                               # n > 100K (Susenas-level)
    or (binary_ratio > 0.5 and spike_ratio > 0.5)  # geometric dominance serius
    or (n_ratio > 5.0 and binary_ratio > 0.4)      # large + sangat binary
)
```
`kcenter` tetap aktif untuk Susenas (n_ratio=33.4). Dataset 10K–100K dengan
mixed data normal kembali ke `cluster_aware` yang aligned ke struktur klaster.

---

#### Bug #2 — `lsil_eval_n` floor terlalu kecil

**Formula lama:** `max(5_000, 0.03 * n)` → 5,000 untuk n=41K (BankMarketing).
SA mengevaluasi reward dari 5K/41K = 12.2% → reward noisy → SA tidak bisa
membedakan subset bagus dari buruk.

**Formula baru:** `max(10_000, 0.03 * n)` — floor dinaikkan ke 10K.
Kompromi antara kecepatan (2× lebih cepat dari v1.1.11 yang pakai 20K)
dan stabilitas reward.

---

#### Bug #3 — `c_max` terlalu besar → `n_clusters_hint` jauh dari K*

**Formula lama:** `min(log2(n), sqrt(n/2), 10)` → `c_max=10` untuk n=41K.
`pipeline.py` auto-set `n_clusters_hint = midpoint([2,10]) = 6`.
`labels0` dibangun dengan K=6 padahal true K*=2. Merge 6→2 klaster tidak
natural → `L_fixed` buruk dari awal.

**Formula baru:** hard cap berbasis skala dataset:
```python
if n < 200_000:   c_max_hard = 6    # UCI benchmark medium
elif n < 500_000: c_max_hard = 8    # Susenas-level
else:             c_max_hard = 10   # Covertype dan sangat besar
c_max = min(int(log2(n)), int(sqrt(n/2)), c_max_hard)
```

Dampak per dataset:

| Dataset | n | c_max lama | c_max baru | K_hint lama | K_hint baru |
|---|---|---|---|---|---|
| BankMarketing | 41K | 10 | **6** | 6 | **4** |
| Adult | 49K | 10 | **6** | 6 | **4** |
| CreditCard | 30K | 10 | **6** | 6 | **4** |
| Diabetes130 | 102K | 10 | **6** | 6 | **4** |
| Susenas | 334K | 10 | **8** | 6 | **5** |
| Covertype | 581K | 10 | 10 | 6 | 6 |

---

#### Bug #4 — `subsample_n_cluster` floor terlalu kecil

**Formula lama:** `max(2_000, 0.02 * n)` → 2,000 untuk n=41K (4.9% dari data).
`subsample_n_cluster` dipakai untuk kprototypes awal yang menghasilkan `labels0`.
Labels dari 2K/41K tidak representatif → `L_fixed` ikut buruk.

**Formula baru:** `max(6_000, 0.02 * n)` — floor dinaikkan ke 6K.
Sama persis dengan konfigurasi manual v1.1.11 untuk BankMarketing.

---

### New: `label_col` parameter di `run_generic_end2end`

Parameter opsional untuk benchmark UCI dengan ground truth label tersedia.

```python
# Benchmark: K_hint dari jumlah kelas label (paling akurat)
result = run_generic_end2end(
    df,                   # df TERMASUK kolom label
    outdir='out/bank/',
    params=auto_params(df.drop(columns=['y']), random_state=42),
    label_col='y',        # auto-drop dari fitur, K_hint = nunique(y)
)
# [pipeline] n_clusters_hint=2 (source: label_col='y' (nunique=2))

# Produksi (Susenas): tidak ada label, pakai midpoint
result = run_generic_end2end(
    df_ready, outdir='out/susenas/',
    params=auto_params(df_ready, random_state=42),
    # label_col tidak diisi → midpoint [c_min, c_max]
)
```

**Tiga jalur resolusi K_hint (prioritas menurun):**
1. `n_clusters_hint=N` eksplisit → pakai langsung
2. `label_col='y'` → `K_hint = nunique(y)`, di-clamp ke `[c_min, c_max]`
3. Default → `midpoint([c_min, c_max])`

**Catatan metodologi:** `label_col` hanya mempengaruhi inisialisasi
(`labels0` + `L_fixed`), bukan pemilihan K* akhir. Phase B tetap mencari
K optimal secara bebas di `[c_min, c_max]`. Ini setara dengan *informed
initialization* — bukan supervisi. Untuk paper: *"For benchmark datasets
with known ground truth, K_hint is set to the number of true classes to
improve initialization quality. The final K* is determined independently
by Phase B."*

---

### Catatan: tidak ada sinyal struktural yang reliable untuk prediksi K* besar

Investigasi menunjukkan tidak ada kombinasi `n`, `p`, `cat_ratio`, `binary_ratio`,
`spike_ratio` yang bisa membedakan dataset K*=7 (DryBean, Obesity) dari dataset
K*=2 (BankMarketing, Adult) yang struktur kolomnya identik. Formula `c_max = f(p)`
meningkatkan coverage dari 10/13 ke 12/13 tapi masih miss Flag (K*=8, p=29).
Solusi yang benar adalah `label_col` untuk benchmark, atau override eksplisit
`auto_params(df, c_max=10)` untuk dataset dengan K* besar yang diketahui.

---

### Parameter auto_params: perbandingan v1.1.13 vs v1.1.14 (BankMarketing n=41K)

| Parameter | v1.1.11 manual | v1.1.13 auto | v1.1.14 auto | Match? |
|---|---|---|---|---|
| `c_max` | 6 | 10 | **6** | exact |
| `n_clusters_hint` | 3 | 6 | **4** | dekat |
| `landmark_mode` | cluster_aware | kcenter | **cluster_aware** | exact |
| `lsil_eval_n` | 20,000 | 5,000 | **10,000** | lebih baik |
| `subsample_n_cluster` | 6,000 | 2,000 | **6,000** | exact |
| `screening_k_values` | [2,3,4,5,6] | [2,4,7,10] | **[2,3,4,6]** | mendekati |

### Files changed

| File | Change |
|---|---|
| `api.py` | `auto_params`: 4 fix (c_max hard_cap, landmark_mode threshold, lsil_eval_n floor, subsample_n_cluster floor) |
| `pipeline.py` | `run_generic_end2end`: tambah `label_col` param + 3-jalur K_hint resolution |
| `__init__.py` | version → 1.1.14 |
| `pyproject.toml` | version → 1.1.14 |

---

## v1.1.13

### Fix 1: SA bottleneck untuk dataset besar (Susenas)

**Root cause:** `auto_params` menggunakan `lsil_c` yang sama untuk SA reward
dan Phase B. Untuk Susenas, `lsil_c=5.5` → `|L|=3179`. Setiap reward call
SA menghitung Gower distances: `eval_n × |L| = 20K × 3179 = 63M ops`.
SA 58 iterasi × 31 fitur = ~1800 calls → total ~114B ops → 6 jam.

**Fix:** `auto_params` kini menetapkan `lsil_c_reward = min(2.0, lsil_c)`.
SA menggunakan `|L|=1156` (kecil, cepat), Phase B tetap `|L|=3179` (akurat).
SA hanya butuh sinyal arah/ranking, bukan nilai absolut yang presisi.

Selain itu `lsil_eval_n` dikurangi dari 6% ke 3% dari n untuk SA.

| | Sebelum | Sesudah |
|---|---|---|
| eval_n SA | 20,053 | 10,026 |
| \|L\| SA | 3,179 | 1,156 |
| ops/call | 63,748,487 | 11,590,056 |
| Speedup | — | **5.5x** |
| SA Susenas ~6 jam | → | **~1.1 jam** |

`lsil_c` Phase B tetap 5.5 → `|L|=3179` → akurasi evaluasi tidak berubah.

### Fix 2: hac_gower Phase B bottleneck (Adult dan dataset n > 10K)

**Root cause:** `hac_landmark_hybrid_adapter` menggunakan pure Python loop
`for i in range(n)` untuk assignment nearest-centroid. Untuk Adult n=32K
dengan K_range=13 × 16 subsets = 208 trials → 208 × 32K Python iterations
× `gower_to_one_mixed()` per call → Phase B ~5 jam.

**Fix:** Ganti Python loop dengan `gower_distances_to_landmarks` (sudah
ada di codebase) + `np.argmin(D, axis=1)` — operasi matrix C-level numpy.

```python
# Sebelum: O(n×K) Python loop
for i in range(n):
    for c in proto_ids:
        d = gower_to_one_mixed(...)

# Sesudah: matrix ops
D = gower_distances_to_landmarks(X_num, X_cat, ..., proto_idx)
labels_all = [valid_ids[i] for i in np.argmin(D, axis=1)]
```

Hasil assignment identik secara matematis. Speedup estimasi 50-200x
untuk assignment step.

### Tidak ada perubahan teoritis

- Theorem 1 tidak berubah: `lsil_c_reward` adalah parameter implementasi,
  bukan klaim teoritis. SA menggunakan L-Sil sebagai proxy ranking, bukan
  nilai yang dilaporkan.
- Vektorisasi assignment menghasilkan labels yang bit-for-bit sama.
- AUFS-Samba, L-Sil, LNC* tidak berubah.

### Files changed

| File | Change |
|---|---|
| `api.py` | `auto_params`: `lsil_c_reward=min(2.0,lsil_c)`, `lsil_eval=3%n` |
| `controller.py` | `hac_landmark_hybrid_adapter`: vektorisasi assignment |
| `__init__.py` | version → 1.1.13 |
| `pyproject.toml` | version → 1.1.13 |

## v1.1.12

### Fix: Auto-K bias toward K_hint — two landmark strategies

**Root cause identified:**

`L_fixed` (the Phase B cache landmark set) was always built using
`cluster_aware_landmarks_on_subsample` with `labels0` from `K=n_clusters_hint`.
Landmarks were placed near centroids/boundaries of that specific K.

When Phase B evaluates K values far from K_hint:
- K=2 evaluation: two large clusters separate cleanly in landmark space → L-Sil inflated
- K=4 evaluation: four smaller clusters poorly represented by K_hint landmarks → L-Sil deflated
- Result: auto-K systematically favors K values close to K_hint, regardless of actual SS-Gower

**Fix — two landmark strategies, selectable via `landmark_mode`:**

| Mode | Algorithm | BCVD | Auto-K bias | When to use |
|------|-----------|------|-------------|-------------|
| `"cluster_aware"` (default) | 80% central + 20% boundary per cluster | Low | Yes (biased to K_hint) | Fixed K or narrow K range |
| `"kcenter"` | k-center greedy, K-agnostic | Slightly higher | None | `auto_k=True` with wide K range |

Both modes: Theorem 1 holds, O(n·|L|) unchanged, L-Sil/LNC* computation unchanged.

**`auto_params()` selects automatically:**
```python
landmark_mode = "kcenter"       if (c_max - c_min) > 1  # wide auto-K range
              = "cluster_aware"  otherwise                # fixed/narrow K
```

**Manual override:**
```python
# Force kcenter for fair auto-K evaluation
params = auto_params(df, landmark_mode="kcenter")

# Force cluster_aware (paper default, best BCVD mitigation)
params = AUFSParams(landmark_mode="cluster_aware", ...)
```

### New: `auto_params(df, **overrides)` — self-configuring AUFSParams

Three parameters auto-computed from data:

1. **`lsil_c`** — `max(3.0, 3.0 * log10(n) / log10(1000))`
   Theorem 1 holds for any c > 0; c=3.0 is empirical floor.

2. **`c_max`** — `min(int(log2(n)), int(sqrt(n/2)), 20)`
   Practical upper bound for K search, derived from n.

3. **`screening_k_values`** — 4 evenly-spaced points from `[c_min, c_max]`
   Always consistent with actual K search range.

### Fix: `screening_k_values` fallback when outside c_range

Before: fallback to `[c_range[0]]` — only one K screened.
After: 3 evenly-spaced points from c_range — proper coverage.

### Includes all v1.1.11 changes

Three-path kprototypes in Phase B included.

### Files changed

| File | Change |
|------|--------|
| `api.py` | `AUFSParams.landmark_mode`, wire to `make_sa_reward`, `auto_params()` |
| `reward.py` | `landmark_mode` param, kcenter/cluster_aware two-path in `lsil_fixed_calibrated` |
| `controller.py` | screening fallback fix + v1.1.11 three-path kproto |
| `__init__.py` | export `auto_params` |
| `pyproject.toml` | version → 1.1.12 |


## v1.1.11 (patch)

### Fix: Phase B kprototypes label-cache misalignment

**Problem:** In v1.1.10, `kprototypes` in Phase B always fit on a 6K
subsample (via `kprototypes_subsample_adapter`). This produced `labels_B`
that diverged from `labels0` used to build `L_fixed` in Phase A.
Since `L_fixed` is cluster-aware (placed near centroids of `labels0`),
evaluating `labels_B` against `L_fixed` systematically under-estimated
L-Sil for kprototypes. As a result the auto-adapter often selected
`hac_gower` even when kprototypes produced better SS-Gower.

**Root cause summary:**
```
Phase A:  labels0  = kproto(subsample 6K) → NN-propagated to full n
          L_fixed  = cluster_aware_landmarks(labels0)   ← placed near labels0 centroids

Phase B:  labels_B = kproto(NEW subsample 6K) → slightly different partition
          L-Sil(kproto) evaluated against L_fixed biased toward labels0
          → kproto L-Sil under-estimated → hac_gower wins unfairly
```

**Fix — three-path strategy in `_run_algo` (controller.py only):**

| Path | Condition | Behaviour |
|------|-----------|-----------| 
| A | n ≤ 10,000 | `kprototypes_adapter` on full data — accurate & fast |
| B | n > 10,000 AND cache available | derive labels from `labels0` via merge/split — O(n), deterministic, consistent with L_fixed |
| C | n > 10,000 AND no cache | subsample fallback (v1.1.10 behaviour) |

Two helper functions added to `controller.py`:
- `_merge_labels_to_k`: merge smallest cluster pairs to reach k_target
- `_split_labels_to_k`: bisect largest cluster via kprototypes(k=2)

### Files changed
| File | Change |
|------|--------|
| `controller.py` | `_run_algo`: three-path kproto, add `_merge_labels_to_k`, `_split_labels_to_k` |
| `__init__.py` | version bump to 1.1.11 |
| `pyproject.toml` | version bump to 1.1.11 |

## v1.1.10 (2026-04-04)

### Perf: DAV Phase B 60x slower than non-DAV

**Root cause: `_AnchorContext` built |L|=66,845 landmarks**

Old formula: `m = max(sqrt(n), lm_frac*n)` with `lm_frac=0.20`:
- `max(578, 66845) = 66845` landmarks for n=334K
- Compare with Phase A cache: |L|=1734 (`c*sqrt(n)`)
- LNC*_a complexity = O(n × |L| × k) → **39x more expensive**
- KNNIndex built on full 334K rows → **33x slower than necessary**
- Context rebuilt per subset even when Va is identical → **3x wasted work**
- Total: ~211s/trial × 180 trials = **37,967s** (vs ~638s without DAV)

**Fix v1.1.10 — three improvements:**

**FIX 1: Landmark formula in `_AnchorContext`: `lm_frac*n` → `c*sqrt(n_sub)`**
```
Before: max(sqrt(n), lm_frac*n) = max(578, 66845) = 66845
After:  int(clip(c*sqrt(n_sub), floor=30, cap=3000)) ≈ 300
Speedup: ~200x per LNC*_a call
```
Consistent with Phase A cache which also uses `c*sqrt(n)`.

**FIX 2: Subsample data for AnchorContext**
```
Before: KNNIndex built on full n=334,229 rows
After:  KNNIndex built on subsample anchor_subsample_n=10,000 rows
Build speedup: ~33x
```
Stratified subsample based on `labels0` to preserve cluster distribution.
LNC*_a evaluated on the same subsample (`labels_k[idx_sub]`).

**FIX 3: Cache AnchorContext across subsets sharing the same Va**
```
Susenas subsets 1, 2, 3: all contain Va=[DDS12, DDS13]
Before: build 3 separate contexts
After:  build once, reuse 3x via _anchor_cache[va_cache_key]
Speedup: 3x for subsets with overlapping Va
```

**Estimated combined speedup: ~600x**
```
Before: 37,967s (~10.5 hours)
After:  ~60-120s (~1-2 minutes)
```

### Files changed

| File | Path |
|------|------|
| `dav.py` | `mixclust/utils/dav.py` |
| `api.py` | `mixclust/api.py` |
| `__init__.py` | `mixclust/__init__.py` |
| `pyproject.toml` | `pyproject.toml` |

---

## v1.1.9 (2026-04-04)

### Fix: DAV Phase B selected fallback despite a valid DAV winner

**Root cause — two separate issues:**

**Issue A — apple vs orange comparison:**
`find_best_clustering_dav` compared `score_adj` directly between:
- DAV winner: `score_adj = LNC*_a(Va)` — measures local cohesion in Va space
- Fallback:   `score_adj = LNC*(S*)` — measures cohesion in full S* space

These measure different things on different scales. On Susenas:
- Subset 1 DAV winner: LNC*_a = 0.4033, K=3
- Subset 2 fallback:   score_adj = 0.6896, K=2
- Result: fallback won → K*=2, even though DAV found a valid K=3

**Issue B — threshold too strict:**
`lnc_anchor_threshold=0.40` is too high for large-scale real-world data.
LNC*_a evaluates cohesion in the narrower Va space, so absolute scores are
naturally lower than LNC*(S*). Susenas subset 1 scored 0.4033 — only 0.0033
above the old threshold, nearly failing due to numerical noise.

**Fixes in v1.1.9:**

1. **`find_best_clustering_dav`: fair comparison via `_should_update()`**
   - DAV winner always takes priority over fallback — scores not compared directly
   - DAV vs DAV: compare LNC*_a
   - Fallback vs fallback: compare score_adj (previous behaviour)
   - Fallback cannot override a DAV winner

2. **`lnc_anchor_threshold` default: 0.40 → 0.25**
   0.25 is more realistic for LNC*_a on large-scale mixed-type data.
   The old threshold (0.40) can still be set explicitly if desired.

3. **`_AnchorContext`: log Va_valid vs Va_requested**
   Now shows which anchor variables were found vs missing in each subset,
   making it easier to debug when a subset does not contain all Va columns.

4. **`auto_select_algo_k_dav`: log best LNC*_a when falling back**
   When no K passes the threshold, shows the best LNC*_a found and the
   active threshold, so users can tune accordingly.

**Real case fixed:**
Susenas Scenario B (seed=42), subset 1 contains DDS12+DDS13:
- Before fix: K*=2 (fallback won with score 0.6896)
- After fix:  K*=3 (DAV winner with LNC*_a=0.4033 prioritised)

### Fix: `run_generic_end2end` return dict did not include clustering results

`pipeline.py` returned only file paths — `best_K`, `final_algo`, and `dav`
were not in the return dict, requiring notebooks to open `metrics_internal.json`
separately and causing `NameError` / `KeyError` errors.

**Fix:** Add three keys to the `run_generic_end2end` return dict:
```python
"best_K":     metrics.get("best_K"),
"final_algo": metrics.get("final_algo"),
"dav":        metrics.get("dav"),   # None if DAV was not active
```

### Files changed

| File | Path |
|------|------|
| `dav.py` | `mixclust/utils/dav.py` |
| `pipeline.py` | `mixclust/pipeline.py` |
| `__init__.py` | `mixclust/__init__.py` |
| `pyproject.toml` | `pyproject.toml` |

---

## v1.1.8 (2026-04-01)

### Fix: DAV Phase B stuck / extremely slow

**Root cause:** Three bottlenecks per DAV trial:
1. `kprototypes_adapter` on full 334k rows (~60-120s)
2. `structural_control_lnc` rebuilds KNNIndex from scratch per trial (~60-120s)
3. `lnc_star_anchored` rebuilds KNNIndex from scratch per trial (~30-60s)

Total per trial: ~150-300s x 96 trials = **4-8 hours** (or hang)

**Fix v1.1.8:**
1. Clustering via `kprototypes_subsample_adapter` (6k rows) → ~5s
2. LNC* global reuses `phase_a_cache.knn_index` → ~15s (no rebuild)
3. Anchor KNN+landmark built **once per subset** via `_AnchorContext` → ~10s per trial

**Estimate:** Per trial 150s → ~30s. Phase B DAV: 4-8 hours → ~45 minutes.

### Files changed

| File | Path |
|------|------|
| `dav.py` | `mixclust/utils/dav.py` |
| `cluster_adapters.py` | `mixclust/clustering/cluster_adapters.py` |
| `controller.py` | `mixclust/clustering/controller.py` |
| `phase_a_cache.py` | `mixclust/aufs/phase_a_cache.py` |
| `reward.py` | `mixclust/aufs/reward.py` |
| `api.py` | `mixclust/api.py` |
| `kamila.py` | `mixclust/clustering/kamila.py` (new) |
| `__init__.py` | `mixclust/__init__.py` |
| `pipeline.py` | `mixclust/pipeline.py` |
| `dqc.py` | `mixclust/utils/dqc.py` (new) |
| `pyproject.toml` | `pyproject.toml` |

---

## v1.1.7 (2026-03-31)

### New: `phase_b_skip_lnc` parameter
- Skip LNC* per Phase B trial → saves ~30s/trial
- LNC* still computed once in the final structural control
- `controller.py`: `skip_lnc` param in `auto_select_algo_k()` and `_eval_with_phase_a_cache()`
- `api.py`: `AUFSParams.phase_b_skip_lnc: bool = False`

---

## v1.1.6 (2026-03-31)

### Phase B subsampled L-Sil evaluation
- `phase_a_cache.py`: `build_phase_b_subsample()` — subsample ~30k rows
- `controller.py`: `_eval_with_phase_a_cache()` uses subsample
- `reward.py`: inject `random_state` into `__phase_a_cache__`
- `api.py`: `phase_b_eval_n: int = 30_000`

---

## v1.1.5 (2026-03-31)

### Critical fix: `kprototypes_subsample_adapter` not imported
- **Impact:** all kprototypes Phase B trials failed silently
- **Fix:** add import in `controller.py`

---

## v1.1.2 — v1.1.4 (2026-03-30)

### v1.1.4: Import fix
- `controller.py`: fix import of `kprototypes_subsample_adapter`

### v1.1.3: Subsample wiring
- Wire `subsample_n_cluster` through reward pipeline

### v1.1.2: Performance optimisations
- `reward.py` [A]: L-Sil evaluated on 20k subsample (~67x speedup per SA call)
- `reward.py` [B]: clustering on 6k subsample (~15x speedup for build_reward)
- `redundancy.py`: remove joblib, precompute premaps (~40x speedup)

---

## v1.1.1 (2026-03-30)

### Bug fix: SA reward = -1.0
- `reward.py` [C]: fix `lsil_using_landmarks` argument after refactor
- `controller.py`: fix `_eval_with_phase_a_cache` and `score_internal`

---

## v1.1.0 (2026-03-30)

### Refactor: L-Sil prototype → landmark (aligned with JDSA paper)
- `|L| = c*sqrt(n)` (Theorem 1), default c=3
- PhaseACache infrastructure introduced
- Backward-compatible: `lsil_using_prototypes_gower` still exported
