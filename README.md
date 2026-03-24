# mixclust

Adaptive mixed-type household clustering with AUFS-Samba, L-Sil, and LNC*.

## Install

```bash
pip install git+https://github.com/hqers/mixclust.git
```

## Version Check Cell

Paste cell ini di awal notebook untuk memastikan versi selalu up-to-date:

```python
import sys, subprocess

GITHUB_URL  = "git+https://github.com/hqers/mixclust.git"
MIN_VERSION = "1.0.7"  # update sesuai versi minimum yang diperlukan

def _ver(v):
    try: return tuple(int(x) for x in v.split("."))
    except: return (0,)

# Cek numpy/pyarrow conflict (umum di shared JupyterHub)
try:
    import pyarrow
except Exception:
    print("numpy/pyarrow conflict — downgrade numpy<2...")
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "-q", "--user", "numpy<2.0"])
    print("RESTART KERNEL lalu jalankan ulang cell ini.")
    raise SystemExit("Restart kernel diperlukan")

needs_install = False
try:
    import mixclust as _mc
    if _ver(_mc.__version__) < _ver(MIN_VERSION):
        print(f"Versi {_mc.__version__} < min {MIN_VERSION} — reinstall...")
        needs_install = True
    else:
        print(f"mixclust {_mc.__version__} ok (min: {MIN_VERSION})")
except ImportError:
    print("mixclust belum terinstall — install sekarang...")
    needs_install = True

if needs_install:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "-q", "--user",
                           "--upgrade-strategy", "only-if-needed",
                           GITHUB_URL])
    import importlib, mixclust
    importlib.reload(mixclust)
    print(f"mixclust {mixclust.__version__} berhasil diinstall")
else:
    import mixclust

print(f"mixclust versi: {mixclust.__version__}")
```

> **Catatan:** Ubah `MIN_VERSION` sesuai versi minimum yang diperlukan untuk notebook Anda.
> Changelog versi tersedia di [releases](https://github.com/hqers/mixclust/releases).

## Quickstart

```python
from mixclust import run_generic_end2end, AUFSParams
import pandas as pd

df = pd.read_csv('HH_CLUSTERING_ready.csv', sep=';')

result = run_generic_end2end(
    df,
    outdir='output/',
    id_col='HHID',
    verbose=True
)
```

## Tools

Generator notebook dan visualisasi tersedia di: https://hqers.my.id/mixclust/

## Papers

- AUFS-Samba: https://doi.org/10.1109/ACCESS.2026.3653624
