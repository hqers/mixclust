# Changelog v1.1.5 — Bug Fix Phase B + Parameter Wiring

## Bug Fixes

### 🔴 BUG KRITIS: `controller.py` — `kprototypes_subsample_adapter` tidak di-import

**Dampak:** Semua trial kprototypes di Phase B **gagal diam-diam** (NameError → except → return None → skip). Hanya hac_gower/kmodes/auto yang berjalan, sehingga Phase B tidak pernah mencoba kprototypes — yang biasanya adalah adapter terbaik untuk mixed-type data.

**Root cause:** Di `_run_algo()` baris 500 dipanggil `kprototypes_subsample_adapter(...)`, tapi import block (baris 32-37) hanya import `kprototypes_adapter`. Karena `except Exception: return None`, error ini tidak terlihat.

**Fix:** Tambah import:
```python
from .cluster_adapters import (
    hac_gower_adapter,
    kprototypes_adapter,
    kprototypes_subsample_adapter,   # ← FIX v1.1.5
    kmodes_adapter,
    auto_adapter,
)
```

### 🟡 BUG: `api.py` — Parameter v2.2 tidak di-wire ke `make_sa_reward()`

**Dampak:** `lsil_eval_n`, `lsil_c_reward`, `subsample_n_cluster` selalu default (20k, None, 6000) — tidak bisa dikontrol dari notebook via `AUFSParams`.

**Fix:**
1. Tambah 3 field baru di `AUFSParams`:
   ```python
   lsil_eval_n: int = 20_000
   lsil_c_reward: Optional[float] = None
   subsample_n_cluster: int = 6_000
   ```
2. Wire ke kedua pemanggilan `make_sa_reward()` (baris ~331 dan ~695).

### 🟡 BUG: `__init__.py` — Versi stuck di 1.1.0 + `lsil_using_landmarks` belum diekspor

**Fix:** Update `__version__ = "1.1.5"` + tambah ekspor `lsil_using_landmarks`.

---

## File yang Diubah

| File | Path di repo | Perubahan |
|------|-------------|-----------|
| `controller.py` | `mixclust/clustering/controller.py` | Tambah import `kprototypes_subsample_adapter` |
| `api.py` | `mixclust/api.py` | 3 field baru di `AUFSParams` + wire ke `make_sa_reward()` |
| `__init__.py` | `mixclust/__init__.py` | Version bump + ekspor baru |
| `pyproject.toml` | `pyproject.toml` | Version bump ke 1.1.5 |

## Git Commit

```bash
git add mixclust/clustering/controller.py mixclust/api.py \
        mixclust/__init__.py pyproject.toml
git commit -m "fix: Phase B kprototypes silent fail + wire v2.2 params (v1.1.5)

- controller.py: add missing import kprototypes_subsample_adapter
  (was causing all kprototypes trials to silently fail in Phase B)
- api.py: add lsil_eval_n, lsil_c_reward, subsample_n_cluster to
  AUFSParams and wire both make_sa_reward() calls
- __init__.py: export lsil_using_landmarks, bump version
- pyproject.toml: bump to 1.1.5"
git tag v1.1.5
git push && git push --tags
```

## Cara Verifikasi

Setelah push, jalankan notebook dan perhatikan log Phase B:

```
[BEGIN PHASE B] 15 subset | CACHE MODE (|L|=1734, reuse Phase A components)
  -> Subset #1 (12 fitur): [...]...
    ✨ Best baru! algo=kprototypes, K=4, J=0.2815
```

**Jika `kprototypes` muncul** di output → fix berhasil.
**Jika hanya `hac_gower` / `kmodes`** → fix belum terapply (cek import ulang).
