# KAMUS DATA & DEFINISI KOLOM (DATA_DICTIONARY.md)

> **Rujukan angka resmi:** [`GROUND_TRUTH.md`](GROUND_TRUTH.md)
> **Kaidah indeks topik:** model Gensim 0-indexed; seluruh data laporan & dashboard 1-indexed.

---

## 1. Metadata Pengambilan Data (Scraping Metadata)

- **Sumber Data**: Repositori Institusi UIN Raden Fatah Palembang (`http://repository.radenfatah.ac.id/`)
- **Prodi Target**: Sistem Informasi (`structure_id = 73`)
- **Rentang Tahun**: 2021 – 2025
- **Tanggal cutoff scraping**: **19 September 2026** (dokumen yang diunggah setelah tanggal ini tidak termasuk)
- **Definisi Kolom Tahun (`year` / `Tahun`)**: diisi oleh `scraper.py` dari elemen HTML meta `<span class="date">` atau substring 4-digit tahun terbit skripsi pada halaman repositori EPrints.

---

## 2. Alur Data (Lineage)

```
repositori (332 terdaftar)
   └─> data/raw/thesis_scraped.csv .............. 312 dokumen (hasil scraping)
         └─> data/processed/thesis_for_pipeline.csv  309 dokumen (filter awal)
               └─> data/intermediate/dataset_preprocessed.csv  285 dokumen (korpus bersih)
                     ├─> model/lda_model.gensim .... K=8, C_V=0,4256
                     └─> outputs/*.csv .......... artefak resmi dashboard
                           └─> outputs/forecast/*.csv  statistik & proyeksi
```

| Tahap | Jumlah | Penyebab pengurangan |
| :--- | ---: | :--- |
| Populasi repositori | 332 | — (referensi administratif di Bab III) |
| `thesis_scraped.csv` | 312 | Selisih 20 dokumen terhadap angka populasi; hasil scraping aktual |
| `thesis_for_pipeline.csv` | 309 | Artefak scraping / kelengkapan field |
| `dataset_preprocessed.csv` | **285** | Duplikasi judul + filter dokumen dengan < 5 token |

Sebaran tahun pada korpus bersih (285): 2021: 29 · 2022: 78 · 2023: 46 · 2024: 47 · 2025: 85

---

## 3. Struktur & Definisi Kolom Berkas Dataset

### A. `data/raw/thesis_scraped.csv` (312 baris)
*Dataset mentah hasil ekstraksi scraper repositori.*
- `id` (integer): ID unik dokumen skripsi pada repositori EPrints.
- `title` (string): Judul lengkap skripsi mentah.
- `author` (string): Nama penulis skripsi.
- `year` (integer): Tahun publikasi skripsi (2021–2025).
- `abstract` (string): Teks abstrak mentah dalam Bahasa Indonesia atau Inggris.
- `url` (string): URL halaman detail repositori skripsi.

### B. `data/processed/thesis_for_pipeline.csv` (309 baris)
*Dataset terfilter awal setelah penghapusan artefak scraping.*
- `id`, `title`, `author`, `year`, `abstract`, `url`

### C. `data/intermediate/dataset_preprocessed.csv` (285 baris)
*Dataset bersih hasil pembersihan, tokenisasi, N-gram, dan stemming.*
- `ID` (integer): ID unik dokumen.
- `Judul` (string): Judul skripsi.
- `Abstrak` (string): Teks abstrak mentah.
- `Tahun` (integer): Tahun publikasi.
- `teks_gabung` (string): Gabungan Judul + Abstrak (atau hanya Judul bila abstrak terdeteksi bahasa Inggris).
- `teks_bersih` (string): Teks setelah pembersihan karakter khusus, lowercasing, dan pembatas baris.
- `tokens` (string representation of list): List token/n-gram dasar hasil stemming Sastrawi & penyaringan stopword. Rata-rata 58,1 token/dokumen.

Berkas pendukung di direktori yang sama:
- `dictionary.gensim` — kamus **1.849 kata/n-gram** setelah `filter_extremes(no_below=2, no_above=0.40)`.
- `corpus.pkl` — corpus Bag-of-Words (bukan TF-IDF).
- `phraser_bigram.pkl`, `phraser_trigram.pkl` — model Gensim `Phrases` (`min_count=2`, `threshold=7.0`).

### D. `model/topic_distribution.csv` (285 baris)
*Distribusi topik dominan per dokumen hasil inferensi LDA.*
- `ID`, `Judul`, `Tahun`
- `topik_dominan` (integer): ID topik dominan 1-based (1–8).
- `prob_dominan` (float): Probabilitas keanggotaan topik dominan (0.0–1.0).

### E. `outputs/model/labels.json` (8 entri) & `model/topic_labels.csv` (8 baris)
*Definisi label topik resmi — Human-in-the-Loop, terkunci dengan fingerprint SHA-256.*
- `topic_id` (integer): ID topik 1-based (1–8).
- `label` (string): Label final hasil analisis interpretatif peneliti atas top-15 words.
- `deskripsi` (string): Deskripsi ringkas cakupan topik.
- `top_words_fingerprint` (string): SHA-256 dari 10 kata teratas topik yang sudah diurutkan A-Z.
- `tingkat_kepercayaan` (string): `Tinggi` | `Sedang` | `Rendah`.
- `status` (string): Selalu `disetujui` (hasil validasi dosen pembimbing).

Fingerprint divalidasi oleh `src/topic_labels.py::validate_labels()` dan dijalankan otomatis di `src/modeling.py` serta `run_tests.py`.

### F. `outputs/topics/topic_counts_final.csv` (8 baris)
*Distribusi dokumen per topik — sumber angka pada halaman Beranda & Distribusi Topik.*
- `topik_dominan` (integer): 1–8
- `jumlah_dokumen` (integer): 32/30/46/33/41/49/33/21 — total **285**
- `persentase` (float): Persentase terhadap 285 dokumen.

### G. `outputs/topics/topic_distribution_final.csv` (285 baris)
*Distribusi dokumen per topik versi final (reparan 1 dokumen pada Topik 2 & 4).*
- `ID`, `Judul`, `Tahun`, `topik_dominan` (1-based), `prob_dominan`

### H. `outputs/topics/topic_by_year_final.csv` (5 baris)
*Matriks kontingensi jumlah dokumen per tahun × per topik — sumber halaman Tren Per Tahun.*
- `Tahun` (integer): 2021–2025
- Kolom `1`–`8`: jumlah dokumen per topik
- `total_dokumen` (integer): 29/78/46/47/85 — total 285
- `jumlah_judul_saja` (integer): dokumen yang abstraknya tidak tersedia sehingga judul dipakai sebagai pengganti — total 38
- `proporsi_topik_1` … `proporsi_topik_8` (float): proporsi (%) tiap topik terhadap total tahun tersebut

### I. `outputs/forecast/forecast_table.csv` (8 baris)
*Tabel proyeksi WMA 2026–2027 — sumber halaman Proyeksi.*
- `topik_id` (integer): 1–8
- `label` (string): Label topik dari `labels.json`
- `share_2021` … `share_2025` (float): Proporsi historis (%)
- `wma_2026` (float): Proyeksi WMA 2026 (%)
- `pita_bawah_2026`, `pita_atas_2026` (float): Interval ± 1 SD residual backtest
- `wma_2027` (float): Proyeksi WMA rekursif 2027 (%)
- `arah` (string): `naik (indikatif)` | `turun (indikatif)` | `tidak berubah bermakna`
- `slope_linear_pp_per_tahun` (float): Slope regresi linear OLS
- `p_trend_spearman`, `p_trend_terkoreksi` (float): p-value tren Spearman sebelum & sesudah koreksi Bonferroni
- `mae_pp`, `mae_relatif`, `rasio_vs_baseline` (float): Metrik walk-forward backtest
- `keandalan` (string): Interpretasi perbandingan terhadap baseline
- `catatan` (string): Catatan interpretatif

### J. `outputs/forecast/backtest_results.csv` (24 baris)
*Hasil walk-forward backtest MAE (tahun target 2023, 2024, 2025) untuk 8 topik × 3 metode.*
- `topik_id`, `metode` (`WMA` | `Rata-rata Historis` | `Naive`)
- `mae_pp` (float): Mean Absolute Error dalam poin persentase
- `mae_relatif` (float): MAE dibagi rata-rata proporsi 5 tahun
- `rasio_vs_baseline` (float): MAE metode ÷ MAE rata-rata historis

### K. `outputs/forecast/share_intervals.csv` (40 baris)
*Interval Keyakinan Wilson 95% untuk proporsi tiap topik per tahun.*
- `topik_id`, `tahun`, `k_count`, `n_total`, `share_pct`, `ci_lower_pct`, `ci_upper_pct`

### L. `outputs/model/evaluation_summary.csv`
*Rekapitulasi metrik evaluasi model final — sumber halaman Metodologi.*
- `Metrik`, `Nilai`, `Protokol_Keterangan`

### M. `model/hyperparameter_results.csv` (84 baris)
*Hasil grid search tahap awal: 7 nilai K (4–10) × 4 α × 3 η.*
- `k`, `alpha`, `eta`, `coherence_cv`, `coherence_umass`, `log_perplexity`, `training_time_seconds`

> **Catatan:** Nilai C_V model final (0,4256) **tidak berasal** dari grid search ini.
> Grid search memakai `passes=10`; model final dilatih ulang dengan `passes=30`, `iterations=500`.

### N. `model/experiment_history.csv`
*Riwayat seluruh eksperimen pelatihan yang pernah dijalankan (audit trail).*