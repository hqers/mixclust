# mixclust

Adaptive mixed-type household clustering with AUFS-Samba, L-Sil, and LNC*.

## Install

```bash
pip install git+https://github.com/hqers/mixclust.git
```

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

## Papers

- AUFS-Samba: https://doi.org/10.1109/ACCESS.2026.3653624
