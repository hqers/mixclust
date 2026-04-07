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

### Git commit
```bash
git add mixclust/api.py \
        mixclust/aufs/reward.py \
        mixclust/clustering/controller.py \
        mixclust/__init__.py \
        pyproject.toml
git commit -m "fix+feat: landmark_mode, auto_params, screening fix, v1.1.11 kproto (v1.1.12)

- landmark_mode='kcenter': K-agnostic landmarks, eliminates auto-K bias
  toward K_hint when using auto_k=True with wide c_min/c_max range
- landmark_mode='cluster_aware' (default): JDSA paper default,
  80% central + 20% boundary, optimal BCVD mitigation
- auto_params(df): self-configuring lsil_c, c_max, screening_k_values,
  landmark_mode from data characteristics
- screening_k fallback: evenly-spaced across c_range (not just c_range[0])
- Includes v1.1.11: three-path kprototypes Phase B"
git tag v1.1.12
git push && git push --tags
```

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

**What does NOT change (by design):**
- `reward.py` — Phase A logic, labels0, L_fixed unchanged
- `landmarks.py` — landmark selection unchanged
- `lsil.py` / `lnc_star.py` — metric computation unchanged
- `phase_a_cache.py` — cache structure unchanged
- `mab.py` / `sa.py` — feature selection unchanged
- `S*` (selected features) — identical to v1.1.10

**Will results change vs v1.1.10?**

| Dataset size | Change expected? |
|---|---|
| n ≤ 6K (CMC, CylinderBands, HeartDisease, etc.) | No — Path A, same as full kproto |
| n 6K–10K (Obesity) | Possibly minor — now full kproto vs subsample |
| n > 10K (Adult, BankMarketing, Susenas) | Yes — Path B replaces subsample |

Auto-adapter claim preserved: both kprototypes and hac_gower still
compete in Phase B. Fix ensures the competition is fair.

**Random seed:** fully deterministic within v1.1.11 via AUFSParams.random_state.
Results differ from v1.1.10 by design (bug fix), not by randomness.

### Files changed
| File | Change |
|------|--------|
| `controller.py` | `_run_algo`: three-path kproto, add `_merge_labels_to_k`, `_split_labels_to_k` |
| `__init__.py` | version bump to 1.1.11 |
| `pyproject.toml` | version bump to 1.1.11 |

### Git commit
```bash
git add mixclust/clustering/controller.py mixclust/__init__.py pyproject.toml
git commit -m "fix: Phase B kproto label-cache misalignment — three-path strategy (v1.1.11)

- Path A (n<=10K): full kprototypes_adapter — deterministic, no subsample
- Path B (n>10K + cache): derive from labels0 via merge/split — O(n),
  consistent with L_fixed, eliminates L-Sil under-estimation
- Path C (fallback): v1.1.10 subsample behaviour

Helpers added: _merge_labels_to_k, _split_labels_to_k
No changes to reward.py, landmarks.py, lsil.py, lnc_star.py, sa.py, mab.py"
git tag v1.1.11
git push && git push --tags
```
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


