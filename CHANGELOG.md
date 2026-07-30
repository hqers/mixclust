# CHANGELOG — mixclust

## v1.1.20 (2026-07-30)

### Change: bentuk agregasi L-Sil dan LNC* menjadi kanonik

Menanggapi butir B3 penelaahan disertasi (Dr. Dimitri Mahayana, 28 Juli 2026)
mengenai bobot ganda kuadratik pada varian L-Sil.

**Temuan.** Bentuk kanonik Silhouette adalah rata-rata sederhana atas instance:

    S = (1/n) sum_i s_i = sum_k |C_k| * sbar_k / sum_k |C_k|

Karena setiap instance menyumbang sekali, sebuah klaster **sudah** menyumbang
sebanding dengan |C_k|. Hingga v1.1.19, `weighted=True` memberi setiap instance
bobot tambahan sebesar ukuran klasternya, sehingga kontribusi klaster menjadi
sebanding **|C_k|^2** — ukuran klaster masuk dua kali. Pada dataset dengan
klaster tidak seimbang, klaster minoritas praktis tidak berpengaruh: pada
Obesity (K=7) rasio pengaruh klaster terbesar terhadap terkecil naik dari
sekitar 29x menjadi 843x.

Tiga hal menguatkan perubahan ini. Metrik acuan `full_silhouette_gower`
bersifat kanonik, sehingga perbandingan sebelumnya tidak sepadan. Seluruh
pembuktian teoretis (Lemma 1, 2, 5, Teorema 2) dilakukan untuk bentuk kanonik.
Dan justifikasi yang tertulis di naskah, yaitu bahwa Silhouette klasik secara
implisit berbobot ukuran klaster, sesungguhnya membenarkan bentuk kanonik.

**Bukti empiris (11 dataset UCI, 5 seed, landmark identik, SS_G eksak):**

| Besaran | Eq.(4) berbobot | Eq.(3) kanonik |
|---|---|---|
| mean MAE | 0.1580 | **0.1517** |
| calibrated MAE | 0.0120 | **0.0116** |
| LODO MAE | 0.0145 | **0.0138** |
| Spearman rho | **0.9182** | 0.8818 |
| R^2 | **0.9775** | 0.9742 |
| sd antar seed | 0.0031 | 0.0031 |

Selisih kedua bentuk kecil: rata-rata 0.0065, terbesar 0.0280 (Automobile),
sekitar 4% dari MAE. Bentuk kanonik lebih baik pada seluruh metrik galat,
termasuk galat held-out yang menjadi angka utama. Keunggulan rho pada bentuk
berbobot bertumpu pada satu pertukaran peringkat (Adult <-> Automobile) dengan
selang kepercayaan yang bertumpang tindih hampir sepenuhnya.

Tidak ada satu pun dataset yang berpindah sisi terhadap ambang Structural
Control 0.5: sembilan tetap di atas, dua (Student Performance, Flags) tetap di
bawah. Margin terdekat 0.138, sementara penurunan LNC* terbesar hanya 0.062.

**Perubahan API:**

```python
# baru di AUFSParams
lsil_weighted: bool = False   # False = kanonik Eq.(3); True = berbobot Eq.(4)
```

Bentuk agregasi disimpan sebagai default tingkat modul dan ditetapkan oleh
`AUFSParams.__post_init__`, sehingga berlaku sejak objek params dibentuk
(termasuk lewat `auto_params`):

```python
from mixclust.metrics.lsil import set_default_weighted, get_default_weighted
```

Parameter `weighted` pada `mixclust.metrics.lsil` dan `use_weighted_mean` pada
`mixclust.metrics.lnc_star` kini bersentinel `None`, yang berarti "ikuti default
paket"; nilai eksplisit `True`/`False` tetap menang. Ini disengaja: `controller.py`
dan modul `utils/` memanggil kedua metrik tanpa menerima `params`, sehingga tanpa
default tingkat modul, menyetel `lsil_weighted=True` hanya akan mengubah jalur
reward dan menghasilkan **campuran dua bentuk agregasi dalam satu run**.

**Kompatibilitas mundur.** Setel `lsil_weighted=True` untuk memperoleh
perilaku v1.1.19 dan sebelumnya. Angka yang telah dilaporkan pada disertasi
(v1.1.17) dan paper JDSA dihasilkan dengan bentuk berbobot; keduanya perlu
dijalankan ulang, dan bentuk berbobot dilaporkan sebagai kolom pendamping.

**Cakupan.** Rilis ini **hanya** mengubah bentuk agregasi. Penyelarasan
seleksi landmark dengan paket `lsil` v1.0.0 (`pool_frac_cap`, konstruksi pool
boundary, metrik farthest-first) sengaja ditunda ke rilis berikutnya agar
dampak butir B3 dapat diatribusikan ke satu faktor.

## v1.1.19 (2026-07-22)

### Feature: Structural Control retry loop ke elite archive

Sebelumnya, *Structural Control* berbasis LNC* di `controller.py`
(`find_best_clustering_from_subsets`) bersifat diagnostik murni: jika
kandidat terbaik gagal ambang LNC* (`passed=False`), sistem hanya
mencatat `action="warning"` tanpa mengambil tindakan lanjutan.
Kandidat yang gagal tetap diterima sebagai keluaran akhir.

Rilis ini mengimplementasikan mekanisme retry yang sebelumnya hanya
berupa desain diagram (`module_iii_with_feedback_loop`, belum ada di
kode): jika kandidat awal gagal ambang LNC*, sistem sekarang mencoba
kandidat lain dari `all_run_history` (elite archive yang sudah
dikumpulkan sepanjang Phase B), diurutkan dari skor J (`score_adj`)
tertinggi ke terendah, hingga ditemukan kandidat yang lolos atau
batas percobaan tercapai.

**Parameter baru di `find_best_clustering_from_subsets`:**

```python
# Jumlah maksimum kandidat elite archive yang dicoba
# setelah kandidat awal gagal ambang LNC* (default: 3)
sc_max_retries: int = 3  # dapat diatur via params.sc_max_retries
```

**Perubahan pada `StructuralControlResult.action`:**

Jika seluruh kandidat retry tetap gagal ambang, `action` diset
menjadi `"rejected_all"` (sebelumnya diam-diam tetap `"warning"`),
agar status ini eksplisit di metadata hasil dan dapat dilacak dalam
audit reproduksibilitas.

**Kompatibilitas mundur:** jika kandidat awal sudah lolos ambang LNC*
(satu-satunya kondisi yang teramati pada seluruh hasil yang telah
divalidasi hingga rilis ini), perilaku identik dengan v1.1.18 —
tidak ada perubahan pada angka yang telah dilaporkan.

**Status:** patch minor. Belum ditandai sebagai rilis stabil v1.2.0;
jalur retry ini belum diuji dengan skenario LNC* gagal ambang secara
sintetis (lihat *Saran* pengembangan lanjutan). Penandaan v1.2.0 akan
menyusul setelah seluruh perbaikan lain untuk disertasi selesai.

## v1.1.18 (2026-06-01)

### Feature: Post-selection Redundancy Check

Penambahan mekanisme verifikasi redundansi pada subset final S* setelah SA
selesai. Sebelumnya, SA yang berjalan dalam `full-neighbor` mode (add/drop/swap)
tidak membawa filter redundansi, sehingga subset output SA berpotensi mengandung
pasangan fitur dengan kMSNC\* > threshold meskipun titik awal dari MAB sudah
non-redundan.

**Arsitektur redundancy control tiga lapis:**

```
MAB (kMSNC*)         SA (L-Sil)          Post-check (kMSNC*)
      ↓                    ↓                      ↓
Non-redundan        Kualitas klaster        Non-redundan
di INPUT            dioptimalkan            di OUTPUT
```

**Perubahan di `AUFSParams`:**

Parameter baru `enable_post_redundancy_check: bool = True` ditempatkan di
blok `# Redundancy` bersama parameter redundansi lainnya. Menggunakan
threshold yang sama dengan `mab_redundancy_threshold` (default 0.90).

```python
# Blok Redundancy di AUFSParams:
red_batch_size: int = 500
# Post-selection redundancy check (v1.1.18)
enable_post_redundancy_check: bool = True  # ← baru
```

**Fix: `red_matrix` dan `red_threshold` sebelumnya tidak diteruskan ke `mab_explore`:**

Dua pemanggilan `mab_explore` di `api.py` tidak meneruskan `red_matrix`
sehingga filter konstruksi MAB tidak pernah aktif (selalu `None`).
`mab_redundancy_threshold = 0.90` di `AUFSParams` tidak pernah digunakan.

```python
# Sebelum — filter konstruksi MAB tidak aktif:
mab_out, mab_stats = mab_explore(
    df, reward_for_mab, params.mab_T, k_resolved, rng_py,
)

# Sesudah — filter konstruksi MAB aktif:
mab_out, mab_stats = mab_explore(
    df, reward_for_mab, params.mab_T, k_resolved, rng_py,
    red_matrix=red_mat,
    red_threshold=params.mab_redundancy_threshold,
)
```

**Post-check verbose output:**

Jika `verbose=True`, post-check menampilkan notifikasi eksplisit:

```
# Jika tidak ada pasangan redundan (kasus normal):
[POST-CHECK] Verifikasi redundansi subset S* (6 fitur, threshold=0.9)
[POST-CHECK] PASSED — tidak ada pasangan redundan
             (semua kMSNC* dalam S* <= 0.9).
             Subset dipertahankan: ['WorkerShare', 'AgeHead', ...]

# Jika ada pasangan redundan yang di-drop:
[POST-CHECK] Verifikasi redundansi subset S* (7 fitur, threshold=0.9)
[POST-CHECK] Drop 'FiturA' (kMSNC*(FiturA,FiturB)=0.934 > 0.9)
[POST-CHECK] Subset diperbarui -> 6 fitur: ['FiturB', ...]
```

**Untuk menonaktifkan (ablation study):**

```python
params = AUFSParams(enable_post_redundancy_check=False)
```

### Files changed

| File | Change |
|---|---|
| `api.py` | Tambah `enable_post_redundancy_check` di `AUFSParams` blok Redundancy; fix dua pemanggilan `mab_explore` dengan `red_matrix=red_mat`; tambah post-check setelah SA di dua lokasi dengan verbose output |
| `__init__.py` | version → 1.1.18 |

---

## v1.1.17 (2026-04-13)

### Fix: `auto_k=False` dengan `engine_mode='C'` mengabaikan `n_clusters`

**Root cause:** Kondisi `if (params.engine_mode == "C") and params.auto_k:` di `api.py`
menyebabkan Phase B **sama sekali tidak dijalankan** ketika `auto_k=False`.
Tanpa Phase B, `final_C` diambil dari `auto_adapter._last` yang menyimpan K
dari SA reward call terakhir — bukan dari `n_clusters` yang user set.

**Akibat yang terlihat:**
- `best_K` berbeda dari `n_clusters` yang di-set (contoh: `n_clusters=2` → `best_K=4`)
- `final_algo=null` di metrics_internal.json
- `phaseB_s` tidak ada di timing
- Perbandingan apple-to-apple MixClust vs baseline (K_gt) tidak bisa dilakukan

**Fix:** Phase B tetap dijalankan meski `auto_k=False`, tapi `c_range`
dikunci ke `[n_clusters, n_clusters]` sehingga K tidak bisa berubah.

```python
# Sebelum: skip Phase B jika auto_k=False
if (params.engine_mode.upper() == "C") and params.auto_k:
    ...
    params_B.auto_k = True  # hardcode True — bug

# Sesudah: Phase B tetap jalan, tapi c_range dikunci
_run_phase_b = (params.engine_mode.upper() == "C") and (
    params.auto_k or (not params.auto_k and n_clusters_eff is not None)
)
if _run_phase_b:
    ...
    if params.auto_k:
        params_B.auto_k = True
    else:
        params_B.auto_k = False
        params_B.c_min = n_clusters_eff   # kunci K
        params_B.c_max = n_clusters_eff   # kunci K
```

**Perilaku per kasus:**

| `auto_k` | `engine_mode` | Phase B | c_range |
|---|---|---|---|
| True | C | ✓ jalan | [c_min, c_max] auto |
| False | C | ✓ jalan | **[n_clusters, n_clusters]** ← fix |
| False | A | ✗ skip | — |
| False | C + n_clusters=None | ✗ skip | — |

**Use case yang kini berfungsi:**
```python
# Perbandingan apple-to-apple MixClust vs AUFS-Samba (keduanya K=K_gt)
params = auto_params(df, auto_k=False, n_clusters=K_gt, random_state=42)
result = run_generic_end2end(df, outdir='out/', params=params)
# best_K == K_gt ✓ — dijamin
```

### Fix: `auto_params` print error `NoneType * float` untuk `lsil_c_reward=None`

Print statement verbose menghitung `|L|_SA = int(lsil_c_reward * n**0.5)` —
error jika `lsil_c_reward=None` (default untuk n ≤ 100K sejak v1.1.16).

**Fix:** Gunakan `(lsil_c_reward or lsil_c)` sebagai fallback.

```python
# Sebelum: TypeError untuk None
f"|L|_SA={int(auto['lsil_c_reward'] * n**0.5):,}"

# Sesudah: fallback ke lsil_c jika None
f"|L|_SA={int((auto['lsil_c_reward'] or auto['lsil_c']) * n**0.5):,}"
```

### Files changed

| File | Change |
|---|---|
| `api.py` | `auto_k=False` + `engine_mode='C'`: Phase B tetap jalan dengan `c_range=[K,K]`; fix print `lsil_c_reward=None` |
| `__init__.py` | version → 1.1.17 |
| `pyproject.toml` | version → 1.1.17 |


---

## v1.1.16 (2026-04-10)

### Fix: `lsil_c_reward` threshold berbasis n — konsistensi SA vs Phase B

**Root cause** — dikonfirmasi dari eksperimen Adult (J=nan) dan analisis 4 run BankMarketing:

Untuk n ≤ 100K, `lsil_c_reward=2.0` menyebabkan SA menggunakan landmark
yang berbeda dari Phase B. Inkonsisensi ini membuat subset yang dipilih SA
tidak optimal saat dievaluasi Phase B.

**Fix:**
```python
if n <= 100_000:
    c_reward = None   # SA pakai landmark yang sama dengan Phase B
else:
    c_reward = round(min(2.0, c_lsil), 1)  # speedup untuk n > 100K
```

| Dataset | n | c_reward lama | c_reward baru | |L_SA| == |L_PB|? |
|---|---|---|---|---|
| BankMarketing | 41K | 2.0 | **None** | ✓ konsisten |
| Adult | 49K | 2.0 | **None** | ✓ konsisten |
| Susenas | 334K | 2.0 | 2.0 | speedup tetap |

---

### Fix: `lsil_eval_n` floor dinaikkan 10K → 20K

Dari eksperimen: `lsil_eval=10K` menyebabkan SA menemukan kandidat subset
yang salah. `lsil_eval=20K` diperlukan agar SA menemukan subset yang benar,
dan hasilnya baru bisa diperbaiki oleh Phase B.

Untuk dataset kecil (n < 20K): `min(n, 20K) = n` → tidak ada efek.

---

### Fix: `phase_b_eval_n` floor dinaikkan 10K → 30K

Dari eksperimen 4 run BankMarketing:
- `pb_eval=10K` dengan subset benar (lsil_eval=20K) → SS-Gower 0.54
- `pb_eval=30K` dengan subset benar → SS-Gower **0.73** (identik v1.1.11)

Phase B perlu evaluasi yang cukup akurat untuk memilih pemenang yang benar
dari subset kandidat yang sudah benar.

Untuk dataset kecil (n < 30K): `min(n, 30K) = n` → tidak ada efek.

---

### Fix: `lsil_topk` cap di 3

`lsil_topk=4` (v1.1.15) menyebabkan SA 2.7× lebih lambat dari v1.1.11
tanpa peningkatan kualitas untuk dataset ≤ 100K. Kembali ke cap 3.

---

### Fix: all-numeric subset di Phase B menghasilkan `J=nan`

**Root cause:** `find_best_clustering_from_subsets` di `controller.py` punya
fallback khusus untuk subset tanpa kolom kategorik (`cat_idx=[]`). Fallback
ini menjalankan KMeans tapi menetapkan `score=nan, score_adj=nan` — subset
ini tidak pernah bisa menang di Phase B meski kualitasnya mungkin baik.

Untuk Adult, SA memilih subset all-numeric (`age`, `Education-num`, dll)
yang valid secara klaster tapi di-skip di Phase B dengan `J=nan`.

**Fix:** Subset all-numeric sekarang dievaluasi via `auto_select_algo_k`
dengan `cat_idx=[]` — KMeans tetap dipakai tapi dengan evaluasi L-Sil
yang proper via cache Phase A.

```python
# Sebelum: hardcode kmeans + score=nan
if len(cat_idx_subset) == 0:
    km = KMeans(n_clusters=params.c_min, ...)
    current = {"algo": "kmeans", ..., "score": np.nan, "score_adj": np.nan}

# Sesudah: auto_select_algo_k dengan cat_idx=[]
if len(cat_idx_subset) == 0:
    current = auto_select_algo_k(X_df=df_subset, cat_idx=[], ...)
```

---

### Konfirmasi dari eksperimen Obesity (n=2,111)

Obesity native v1.1.15 = Obesity force (lsil_eval=20K, pb_eval=30K):
- SS-Gower identik: **0.5643**
- best_reward identik: **0.9273**

Ini mengkonfirmasi `min(n, floor)` bekerja dengan benar — untuk dataset
kecil (n < floor), semua parameter di-clamp ke n sehingga force tidak
berpengaruh sama sekali.

Bonus: v1.1.15 lebih baik dari v1.1.11 untuk Obesity (SS 0.56 vs 0.28),
menunjukkan perbaikan di versi sebelumnya benar-benar membantu dataset kecil.

---

### Fix: DQC `disguised_cat_action` default berubah dari `'cast'` ke `'warn'`

 `disguised_cat_action` default berubah dari `'cast'` ke `'warn'`

Sebelumnya default `'cast'` menyebabkan kolom seperti `OccupationHeadSector`
(integer kode pekerjaan) otomatis di-cast ke `category` tanpa sepengetahuan user.
Default kini diubah ke `'warn'` — DQC hanya memberi peringatan, tidak mengubah dtype.

```
[DQC] ⚠️  OccupationHeadSector  | disguised_categorical | nunique=9 | action=warn
[DQC] disguised_categorical: 2 kolom terdeteksi: ['OccupationHeadSector', 'edu_level']
[DQC] Kolom-kolom ini TIDAK di-cast (disguised_cat_action='warn').
[DQC] Jika kolom tersebut memang kategorik, cast secara eksplisit di notebook:
[DQC]   df['nama_kolom'] = df['nama_kolom'].astype('category')
[DQC] Atau gunakan explicit_cat_cols=['kolom1', ...] di run_dqc()
[DQC] Atau set disguised_cat_action='cast' untuk konversi otomatis.
```

**Tiga cara menangani kolom yang terdeteksi:**

```python
# Cara 1: cast manual di notebook (paling eksplisit, direkomendasikan)
df['OccupationHeadSector'] = df['OccupationHeadSector'].astype('category')

# Cara 2: explicit_cat_cols di run_generic_end2end
result = run_generic_end2end(df, outdir='out/', ...)
# DQC dipanggil otomatis — pass via auto_params atau override pipeline

# Cara 3: aktifkan auto-cast (perilaku v1.1.13 lama)
# Di notebook sebelum run:
from mixclust.utils.dqc import run_dqc
df_clean, dropped, report = run_dqc(df, disguised_cat_action='cast')
```


---

### Files changed

| File | Change |
|---|---|
| `api.py` | `auto_params`: `lsil_c_reward` threshold n≤100K; floor `lsil_eval_n` 20K; floor `phase_b_eval_n` 30K; cap `lsil_topk=3` |
| `controller.py` | `find_best_clustering_from_subsets`: all-numeric subset pakai `auto_select_algo_k` bukan hardcode kmeans+nan |
| `__init__.py` | version → 1.1.16 |
| `pyproject.toml` | version → 1.1.16 |

## v1.1.15 (2026-04-10)

### Fix: `lsil_c` terlalu besar untuk dataset medium (n ≤ 100K)

**Root cause** — ditemukan dari 4 run eksperimental BankMarketing:

| Run | `lsil_c` | `phase_b_eval_n` | SS-Gower |
|---|---|---|---|
| v1.1.11 manual | 3.0 | 30K | **0.7276** ✓ |
| v1.1.14 auto | 4.6 | 10K | 0.4604 ✗ |
| override lsil_c=3 | 3.0 | 30K | **0.7276** ✓ |
| pb_eval=30K only | 4.6 | 30K | 0.4604 ✗ |

Run ke-4 membuktikan `phase_b_eval_n` bukan penyebab — SS-Gower tetap 0.46
meski `phase_b_eval_n` dinaikkan ke 30K. Penyebabnya murni `lsil_c=4.6`.

**Mekanisme:** `lsil_c` menentukan `|L_PB|` — jumlah landmark di Phase B.
Dengan `lsil_c=4.6`, `|L_PB|=933` tapi `lsil_c_reward=2.0` → `|L_SA|=405`.
SA "melihat" landscape dengan 405 landmark, Phase B mengevaluasi dengan 933
yang berbeda posisinya. Inkonsisensi ini menyebabkan subset yang dipilih SA
tidak optimal di landscape Phase B → SS-Gower turun drastis.

Susenas (n=334K) justru *butuh* `lsil_c=5.5` (`|L_PB|=3181`) untuk
SS-Gower tinggi — dataset besar dan heterogen butuh lebih banyak landmark
agar evaluasi Phase B akurat. Ini dikonfirmasi dari run Susenas yang
menghasilkan SS-Gower tertinggi yang pernah dicapai (0.538 vs 0.252 baseline).

**Fix di `auto_params`:**
```python
# Sebelum (v1.1.14): selalu log-proportional
c_lsil = max(3.0, 3.0 * log10(n) / log10(1000))  # 4.6 untuk n=41K

# Sesudah (v1.1.15): floor 3.0 untuk n ≤ 100K
if n <= 100_000:
    c_lsil = 3.0      # konsisten dengan v1.1.11, |L_PB|=608 untuk BankMarketing
else:
    c_lsil = max(3.0, 3.0 * log10(n) / log10(1000))  # 5.5 untuk Susenas
```

Threshold 100K adalah empirical breakpoint dari eksperimen:
- n=41K (BankMarketing): `lsil_c=3.0` → SS-Gower 0.73 ✓
- n=49K (Adult): `lsil_c=3.0` → konsisten
- n=102K (Diabetes130): `lsil_c=5.0` → dataset besar, log-proportional wajar
- n=334K (Susenas): `lsil_c=5.5` → SS-Gower 0.54 ✓

### Fix: `final_ss_gower` dan `best_reward` tidak di-expose di return dict pipeline

`run_generic_end2end` hanya mengembalikan `best_K`, `final_algo`, `dav`.
Akibatnya `res.get('final_ss_gower', 0)` selalu return 0 di notebook.

```python
# Sebelum:
return { "best_K": ..., "final_algo": ..., "dav": ... }

# Sesudah:
return {
    "best_K":         ...,
    "final_algo":     ...,
    "final_ss_gower": metrics.get("final_ss_gower"),  # ← fix
    "best_reward":    metrics.get("best_reward"),       # ← fix
    "dav":            ...,
}
```

### Fix: `dav=null` di `metrics_internal.json` meski DAV berjalan sukses

`api.py` tidak pernah menaruh `finalB` (hasil `find_best_clustering_dav`)
ke dalam `info` dict. `pipeline.py` membaca `info.get("phase_b_result", {})`
untuk mengambil `dav_applied`, `lnc_anchor`, `lnc_global` — tapi key ini
tidak pernah ada, sehingga `dav_info` selalu `{}` → `dav=null`.

```python
# Fix: tambahkan phase_b_result ke info dict di api.py
info = {
    ...
    "phase_b_result": {
        k: v for k, v in (finalB or {}).items()
        if k not in ("labels", "all_run_history")  # skip array besar
    },
}
```

Setelah fix, Skenario B (DAV aktif) akan menampilkan `lnc_anchor` dan
`lnc_global` yang benar di `metrics_internal.json`.

### Files changed

| File | Change |
|---|---|
| `dqc.py` | default `disguised_cat_action` → `'warn'`; pesan warning lebih informatif |
| `api.py` | `auto_params`: fix `lsil_c` threshold n≤100K; tambah `phase_b_result` ke `info` |
| `pipeline.py` | return dict: tambah `final_ss_gower` dan `best_reward` |
| `__init__.py` | version → 1.1.15 |
| `pyproject.toml` | version → 1.1.15 |

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
