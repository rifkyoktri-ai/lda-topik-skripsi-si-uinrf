# 📊 Pemetaan Topik Penelitian Skripsi Sistem Informasi — LDA K=8

> **Program Studi Sistem Informasi · UIN Raden Fatah Palembang · 2021–2025**
> Analisis 285 skripsi dengan Latent Dirichlet Allocation (LDA) + proyeksi
> Weighted Moving Average (WMA), disajikan sebagai dashboard interaktif.

---

## 1. Apa yang diteliti?

285 dokumen abstrak skripsi dipetakan ke **8 topik** menggunakan LDA,
lalu **diperkirakan** komposisinya untuk 2026–2027.

| | |
|:--|:--|
| Jumlah dokumen | **285** (dari 312 entri repositori) |
| Rentang tahun | 2021–2025 |
| Jumlah topik $K$ | **8** |
| Model | Gensim LDA, α=auto, η=auto, 30 passes, 500 iterasi, seed 42 |
| Koherensi $C_V$ | **0,4256** |

🔢 **[Rujukan angka resmi: `GROUND_TRUTH.md`](GROUND_TRUTH.md)** — file itu
dihasilkan otomatis oleh `src/ground_truth.py` dari artefak, jadi tidak
bisa berbeda dengan data sebenarnya.

---

## 2. Datanya dari mana?

```
Halaman indeks repositori (5 berkas HTML)
   → 312 entri skripsi
        → 309 lolos validasi
             → 285 lolos prapemrosesan  ← dipakai model
```

Rincian 27 dokumen yang tidak dipakai (`outputs/audit/excluded_documents.csv`):

| Alasan | Jumlah |
|:--|--:|
| Token < 5 setelah stopword & stemming | 20 |
| Judul duplikat | 4 |
| Tidak valid | 3 |
| **Total** | **27** |

> 312 − 27 = **285** ✅

Rincian per tahun: `outputs/audit/data_lineage_by_year.csv`

---

## 3. Bagaimana diproses?

```
  01 Business Understanding ─┐
  02 Data Understanding    ─┼─→ data/raw/ · data/processed/
  03 Data Preparation     ──┘        data/intermediate/  (285 dok, 1.849 vocab)
        │
        ├── 04 Modeling  ──────→ outputs/model/lda_model.gensim   (K=8)
        ├── 04b Grid Search ────→ outputs/model/hyperparameter_results.csv (140 konfig)
        │
  05 Evaluation     ─────────→ metrik & eksperimen
  06 Forecast       ─────────→ outputs/forecast/  (WMA, backtest, uji chi-square)
  07 Labeling       ─────────→ outputs/model/labels.json  (nama 8 topik, terkunci)
        │
  08 Build Final Artifacts ─→ outputs/topics/  (distribusi, jumlah, per tahun)
        │
        └──→ app/  Dashboard Streamlit  (8 halaman)
```

| # | Notebook | Masukan → Keluaran |
|:-:|:---|:---|
| 01 | Business Understanding | HTML repositori → `data/raw/thesis_scraped.csv` |
| 02 | Data Understanding | 312 → `data/processed/thesis_for_pipeline.csv` (309) |
| 03 | Data Preparation | 309 → `data/intermediate/dataset_preprocessed.csv` (285) |
| 04 | Modeling | korpus → **model LDA** (K=8) |
| 04b | Grid Search | 140 konfigurasi, K=4…10 |
| 05 | Evaluation | metrik koherensi, perplexity, kestabilan |
| 06 | Evaluation & Forecast | proyeksi WMA + backtest + uji statistik |
| 07 | Labeling | top-15 kata → **8 nama topik** (locked by SHA-256) |
| 08 | Build Final Artifacts | model → artefak final (inferensi penuh) |

> **Notebook 04 membuat model, Notebook 08 mengubahnya jadi artefak final**
> memakai *inferensi penuh* (tanpa hold-out 80/20), supaya setiap dokumen
> dinilai secara optimal.

---

## 4. Hasilnya apa?

### 4.1 Delapan topik

| # | Topik | Dokumen | (%) | Keyakinan |
|:-:|:---|--:|--:|:--:|
| 1 | Audit TI & Evaluasi Kesuksesan SI | 32 | 11,23% | Sedang |
| 2 | Pengembangan & Analisis Sistem Heterogen | 29 | 10,18% | Rendah |
| 3 | Analisis Kualitas Layanan & Risiko SI | 46 | 16,14% | Sedang |
| 4 | Kepuasan Pengguna Akhir (EUCS & TAM) | 34 | 11,93% | Tinggi |
| 5 | Manajemen Risiko TI & Analitik Data | 41 | 14,39% | Sedang |
| 6 | Pengembangan & Perencanaan Strategis SI | 49 | 17,19% | Sedang |
| 7 | Penerapan Metode & Framework Sistem Informasi | 33 | 11,58% | Rendah |
| 8 | Keamanan Informasi & SI Institusi Pendidikan | 21 | 7,37% | Rendah |
| | **Total** | **285** | **100%** | |

### 4.2 Angka resmi

| Angka | Nilai | Sumber file |
|:--|:--|:--|
| $C_V$ (model final) | **0,4256** | `outputs/model/evaluation_metrics.csv` |
| $U_{mass}$ | −10,7533 | idem |
| Log Perplexity | −7,4518 | idem |
| Topic Diversity (10/15) | 0,7125 / 0,6917 | `src/config.py` |
| Ukuran kosakata | 1.849 | `data/intermediate/dictionary.gensim` |
| Chi-square homogenitas | χ² = 23,9361, df = 28 | `outputs/forecast/composition_tests_report.md` |
| p-value | 0,6849 | idem |
| Cramér's V | 0,1449 | idem |
| p-permutasi | 0,6994 | idem |
| MAE agregat WMA | 4,46 pp | `outputs/forecast/backtest_results.csv` |
| Topik dengan tren signifikan | **0 dari 8** | `outputs/forecast/per_topic_tests.csv` |

---

## 5. Bagaimana cara memverifikasi?

Semua perintah lewat **satu pintu masuk**:

```bash
python run.py              # daftar perintah
python run.py test         # 34 unit test
  python run.py smoke         # 10 uji asap: semua modul jalan
python run.py structure    # 13 pemeriksaan bentuk repo
python run.py notebook-fast # jalankan pipeline tanpa latih ulang (~2 menit)
python run.py dashboard    # dashboard Streamlit
```

**50 test berjalan otomatis** dan mencakup:
pipeline & prapemrosesan · integritas korpus · fingerprint label ·
matematika WMA · sinkronisasi metrik · bentuk repo.

> Jika `make` terpasang di sistem Anda, `make test` melakukan hal yang sama.

---

## 🗺️ Dari mana asal setiap angka?

Pertanyaan "angka ini dari mana?" bisa dijawab hanya dengan melihat nama file.

| Angka di naskah | File sumber | Dibuat oleh |
|:---|:---|:---|
| 312 entri repositori | `data/raw/thesis_scraped.csv` | Notebook 01 |
| 309 → 285, 27 terbuang | `outputs/audit/data_lineage_by_year.csv` | Notebook 02 |
| Alasan 27 dokumen dibuang | `outputs/audit/excluded_documents.csv` | Notebook 02 |
| 1.849 kosakata | `data/intermediate/dictionary.gensim` | Notebook 03 |
| 140 konfigurasi grid | `outputs/model/hyperparameter_results.csv` | `src/grid_search.py` |
| $C_V$ = 0,4256 | `outputs/model/evaluation_metrics.csv` | `src/modeling.py` |
| Nama 8 topik | `outputs/model/labels.json` | `src/make_labels.py` → `src/labeling.py` |
| Jumlah dokumen per topik | `outputs/topics/topic_counts_final.csv` | Notebook 08 |
| Kata kunci per topik | `outputs/topics/top_words_final.csv` | Notebook 08 |
| Proporsi per tahun | `outputs/topics/topic_by_year_final.csv` | Notebook 08 |
| χ², p, V | `outputs/forecast/composition_tests_report.md` | `src/forecast.py` |
| Proyeksi 2026–2027 | `outputs/forecast/forecast_table.csv` | `src/forecast.py` |
| MAE backtest | `outputs/forecast/backtest_results.csv` | `src/forecast.py` |
| Uji tren per topik | `outputs/forecast/per_topic_tests.csv` | `src/forecast.py` |
| Keyakinan penugasan dokumen | `outputs/audit/assignment_audit.md` | `src/assignment_audit.py` |
| **Semua angka resmi sekaligus** | **`GROUND_TRUTH.md`** | `src/ground_truth.py` |

---

## 📂 Struktur folder

```
├── data/{raw,interim,processed}   ← DATA (sumber → antara → siap)
├── src/        (18 modul)         ← MESIN
├── notebooks/  (9)                ← LAPISAN LAPORAN (CRISP-DM)
├── outputs/                      ← HASIL pipeline
│   ├── model/    model LDA, metrik, nama topik
│   ├── topics/   distribusi, jumlah, kata kunci, per tahun
│   ├── forecast/ proyeksi, backtest, uji statistik
│   └── audit/    keyakinan penugasan, lineage, dokumen terbuang
├── app/        (11)               ← Dashboard Streamlit (8 halaman)
├── tests/      (6)                ← 50 unit test + uji asap
├── docs/       (murni .md)        ← dokumentasi
└── thesis/                        ← naskah skripsi (.docx)
```

### ⚠️ Mengapa `outputs/` ikut di-*commit*?

Dalam konvensi umum, folder `outputs/` justru **diabaikan** git karena isinya
sampah sementara. **Di repo ini lain.**

Semua isi `outputs/` adalah **hasil perhitungan yang dipakai langsung di
naskah skripsi**. Kalau tidak ikut di-*commit*, maka:
- orang yang clone repo tidak bisa memverifikasi satu pun angka
- dashboard kosong
- skripsi tidak dapat direproduksi

Karena itu `.gitignore` mengecualikannya secara eksplisit, dan
`tests/test_structure.py` mengecek hal itu setiap kali test dijalankan.

---

## 🧩 Modul `src/`

| Modul | Peran |
|:---|:---|
| `config.py` | **Semua path & parameter — satu sumber kebenaran** |
| `scraping.py` | Ekstraksi HTML → CSV (Notebook 01) |
| `data_lineage.py` | Jejak lineage + dokumen terbuang (Notebook 02) |
| `preprocessing.py` | 7 tahap prapemrosesan teks (Notebook 03) |
| `grid_search.py` | 140 konfigurasi, K=4…10 (Notebook 04b) |
| `modeling.py` | Pelatihan LDA final (Notebook 04) |
| `evaluation_lda.py` | Multi-seed, sweep K, perplexity (Notebook 05) |
| `forecast.py` | Chi-square, Wilson, WMA, backtest (Notebook 06) |
| `make_labels.py` | Sumber kebenaran nama 8 topik (Notebook 07) |
| `labeling.py` | Ekspor label + validasi fingerprint (Notebook 07) |
| `topic_labels.py` | Validasi fingerprint SHA-256 |
| `build_final.py` | Regenerasi artefak final (Notebook 08) |
| `ground_truth.py` | Generate `GROUND_TRUTH.md` dari artefak |
| `assignment_audit.py` | Hitung ulang laporan keyakinan penugasan |
| `indonesian_stopwords.py` | Stopword bahasa Indonesia + domain SI |
| `data_manager.py` | Helper normalisasi indeks topik |
| `topic_labels_cli.py` | CLI validasi label |

---

## 📚 Dokumentasi

| Berkas | Isi |
|:---|:---|
| [`GROUND_TRUTH.md`](GROUND_TRUTH.md) | 🔒 **Rujukan resmi** angka & metrik |
| [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md) | 🔒 Kamus data & definisi kolom |
| [`docs/EVAL_NOTES.md`](docs/EVAL_NOTES.md) | Panduan klaim untuk Bab V |
| [`docs/README_artefak_final.md`](docs/README_artefak_final.md) | Cara artefak final dibangun |
| [`docs/exploration/`](docs/exploration/) | Laporan lama — **bukan angka skripsi** |

> ⚠️ Isi `docs/exploration/` berasal dari eksplorasi awal dan **tidak
> reproducible**. Jangan dipakai sebagai rujukan angka.

---

## ⚖️ Catatan metodologis

1. **Inferensi penuh (Notebook 08).** `src/modeling.py` melatih pada 228
   dokumen (80%) lalu memprediksi 285 — dokumen di hold-out bisa "lompat"
   topik. Notebook 08 menghitung ulang dengan *inferensi penuh*.

2. **Pemilihan K=8, bukan K=9.** Grid search menunjukkan K=9 mencapai $C_V$
   tertinggi (0,4314) vs K=8 (0,3941) pada tahap grid. K=8 tetap dipilih atas
   pertimbangan gabungan: jumlah dokumen per topik lebih memadai (min. 21),
   $U_{mass}$ dan Topic Diversity lebih baik, dan strukturnya lebih mudah
   diinterpretasikan. Nilai $C_V$ final (0,4256) berasal dari pelatihan ulang
   dengan 30 passes — **bukan** dari grid search.

3. **Proyeksi WMA bersifat indikatif.** Hanya tersedia 5 titik historis.
   Walk-forward backtest menunjukkan WMA **tidak** lebih akurat dari rata-rata
   historis untuk **5 dari 8** topik; hanya Topics 5, 7, dan 8 lebih baik.
   Karena itu proyeksi hanya dipakai sebagai gambaran indikatif.

4. **Pelabelan Human-in-the-Loop.** Label ditetapkan oleh peneliti lewat
   analisis top-15 kata, divalidasi dosen pembimbing, lalu dikunci dengan
   fingerprint SHA-256 agar tidak pernah melayang ke model berbeda.

5. **Pelabelan dapat Direvisi.** Nama topik bisa diubah bila merepresentasikan
   isi. Topik 7 pernah diubah dua kali setelah pengujian terhadap kata kunci
   asli model — lihat `outputs/model/labels.json`.

6. **Keterbatasan penugasan dokumen.** 58 dokumen (20,4%) memiliki nilai
   probabilitas maksimum di bawah 0,60, artinya model tidak yakin terhadap
   penugasannya. Rincian: `outputs/audit/assignment_audit.md`.

---

## 🚨 Peringatan

> **Jangan menjalankan `python -m src.modeling`.**

Nilai $C_V$ = 0,4256 **tidak reproducible**: melatih ulang K=8 dengan
parameter identik menghasilkan 0,3817. Menjalankan perintah itu akan
menimpa model kanonik dan membuat seluruh angka skripsi tidak cocok.
Rincian penyebabnya ada di `outputs/model/README.md`.

---

## 📄 Deployment

```bash
git push
# Streamlit Community Cloud → Settings → Main file path: app/main.py
```

Panduan lengkap: [`docs/DEPLOY.md`](docs/DEPLOY.md) · `runtime.txt` wajib ada.