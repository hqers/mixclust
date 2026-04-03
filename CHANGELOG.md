# CHANGELOG — mixclust

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

### Git commit
```bash
git add mixclust/utils/dav.py mixclust/api.py mixclust/__init__.py pyproject.toml
git commit -m "perf: DAV 60x speedup — c*sqrt(n) landmark, anchor subsample, cross-subset cache (v1.1.10)"
git tag v1.1.10
git push && git push --tags
```

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

### Git commit
```bash
git add mixclust/utils/dav.py mixclust/pipeline.py mixclust/__init__.py pyproject.toml
git commit -m "fix: DAV Phase B prioritise DAV winner over fallback, threshold 0.40->0.25 (v1.1.9)

- find_best_clustering_dav: _should_update() prevents DAV winner from being
  compared directly against fallback score (apple vs orange)
- lnc_anchor_threshold default: 0.40 -> 0.25 (more realistic at large scale)
- _AnchorContext: log Va_valid vs Va_requested
- auto_select_algo_k_dav: log best LNC*_a on fallback to aid tuning
- pipeline.py: expose best_K, final_algo, dav in run_generic_end2end return dict"
git tag v1.1.9
git push && git push --tags
```

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

### New: `_AnchorContext` class
- Caches KNN index + landmarks + Gower arrays in Va space
- Built once in `auto_select_algo_k_dav`, reused per (algo, K) trial
- `_lnc_star_anchored_fast()` uses the prebuilt context

### New: `_lnc_global_from_cache()`
- Computes LNC*(S*) guardrail by reusing `phase_a_cache.knn_index`
- No new KNNIndex built per trial

### New: KAMILA adapter
- `cluster_adapters.py`: `kamila_adapter()` and `kamila_subsample_adapter()`
- `controller.py`: `_run_algo` recognises `"kamila"` as an algorithm
- `dav.py`: `auto_select_algo_k_dav` recognises `"kamila"`
- Opt-in via `auto_algorithms=["kprototypes", "hac_gower", "kamila"]`

### New: `run_dqc()` — Data Quality Check before AUFS-Samba
- **`utils/dqc.py`**: new module, called automatically in `pipeline.py`
- **Level 1 — Zero variance** (`zero_var_action="drop"`): auto-drop
- **Level 2 — Near-zero variance** (`near_zero_action="drop"`, threshold 99.9%): auto-drop
- **Level 3 — High missing** (`missing_action="warn"`, threshold 50%): warning only
- Output: `dqc_report.csv` saved to `outdir` on every run

**Motivation:** Zero-variance features are invisible to reward-based AUFS because
adding them does not lower L-Sil — they are neutral, not penalised. The redundancy
penalty also misses them since it measures inter-feature correlation, not internal
variance. As a result, features can slip into `features.csv` with no discriminative
contribution while diluting other features via Gower's `/p` normalisation.

**Real case:** `AccessCommunication` in Susenas 2020 — all 334,229 households
had value "No" → Cramér's V = 0.0000, entered `features.csv` in v1.1.6,
undetected until post-hoc profile inspection.

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

### Runtime estimate (fast mode, all optimisations enabled)
- SA: 33 min → ~5 min (`calibrate_mode="none"`)
- Phase B: 3.1 h → ~20 min (`rerank_topk=6` + `skip_lnc` + subsample)
- Total: 3.8 h → ~30 min

---

## v1.1.6 (2026-03-31)

### Phase B subsampled L-Sil evaluation
- `phase_a_cache.py`: `build_phase_b_subsample()` — subsample ~30k rows
- `controller.py`: `_eval_with_phase_a_cache()` uses subsample
- `reward.py`: inject `random_state` into `__phase_a_cache__`
- `api.py`: `phase_b_eval_n: int = 30_000`

### Parameter wiring v2.2
- `api.py`: `lsil_eval_n`, `lsil_c_reward`, `subsample_n_cluster` in AUFSParams

---

## v1.1.5 (2026-03-31)

### Critical fix: `kprototypes_subsample_adapter` not imported
- **Impact:** all kprototypes Phase B trials failed silently
- **Fix:** add import in `controller.py`

### Housekeeping
- `__init__.py`: export `lsil_using_landmarks`

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
