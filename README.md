# 📊 Pemetaan Topik Penelitian Skripsi Sistem Informasi (LDA K=8)

> **Program Studi Sistem Informasi · UIN Raden Fatah Palembang · 2021–2025**
> Analisis 285 skripsi dengan Latent Dirichlet Allocation (LDA) + proyeksi Weighted Moving Average (WMA),
> disajikan sebagai dashboard interaktif berbasis Streamlit.

---

## 🎯 Prinsip Repositori

> **Setiap artefak hasil harus bisa ditunjuk: notebook mana yang membuatnya, dan siapa yang memakainya.**

Peta lengkap ada di bagian [🗺️ Peta Artefak](#️-peta-artefak) di bawah.

---

## 📂 Struktur

```
├── notebooks/          ← LAPISAN LAPORAN (CRISP-DM, .ipynb)
├── src/                ← MESIN (modul .py, logika inti)
├── app/                ← Deployment (Streamlit, 8 halaman)
├── tests/              ← verifikasi otomatis
│
├── data/               ← data korpus (312 → 309 → 285 dokumen)
├── model/              ← model LDA + artefak eager
├── _fix_final/         ← artefak final (dashboard)
├── _label_work/        ← label 8 topik (terkunci)
├── _stats_forecast/    ← statistik & proyeksi WMA
└── docs/               ← laporan analisis
```

---

## 🔬 Alur CRISP-DM (Notebook)

| # | Notebook | Fase | Hasil Utama |
|:-:|:---|:---|:---|
| 01 | `01_business_understanding.ipynb` | Business Understanding | `data/raw/thesis_scraped.csv` (312) |
| 02 | `02_data_understanding.ipynb` | Data Understanding | `data/processed/thesis_for_pipeline.csv` (309) |
| 03 | `03_data_preparation.ipynb` | Data Preparation | `data/intermediate/*` (285 dok, 1.849 vocab) |
| 04 | `04_modeling_lda.ipynb` | Modeling | `model/lda_model.gensim` (K=8) |
| 04b | `04b_grid_search.ipynb` | Modeling | `model/hyperparameter_results.csv` (84 konfig) |
| 05 | `05_evaluation_lda.ipynb` | Evaluation | `experiments/*` |
| 06 | `06_evaluation_forecast.ipynb` | Evaluation | `_stats_forecast/*` |
| 07 | `07_labeling.ipynb` | Modeling | `_label_work/labels.json` |
| 08 | `08_build_final_artifacts.ipynb` | Evaluation | `_fix_final/*` ← **regenerasi artefak** |
| — | `app/` | **Deployment** | Dashboard Streamlit |

> **Penting:** Notebook 04 membuat model, **Notebook 08 mengubahnya jadi artefak final**
> menggunakan *inferensi penuh* (tanpa hold-out 80/20) agar setiap dokumen dinilai optimal.

---

## 🚀 Menjalankan

### Dashboard
```bash
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/main.py
```
→ `http://localhost:8501`

### Notebook
```bash
jupyter lab notebooks/
# atau jalankan dari terminal:
python run_notebook.py notebooks/08_build_final_artifacts.ipynb
```

### Pipeline Lengkap (urutan)
```bash
python run_notebook.py notebooks/01_business_understanding.ipynb
python run_notebook.py notebooks/02_data_understanding.ipynb
python run_notebook.py notebooks/03_data_preparation.ipynb
python -m src.grid_search              # opsional (~10 menit)
python -m src.modeling                 # opsional (~5 menit)
python -m src.evaluation_lda           # opsional (~10 menit)
python -m src.topic_labels_cli         # validasi label
python -m src.labeling                 # ekspor label CSV
python run_notebook.py notebooks/06_evaluation_forecast.ipynb
python run_notebook.py notebooks/08_build_final_artifacts.ipynb
```

### Verifikasi
```bash
python run_tests.py              # 5 unit test + fingerprint label
python -m src.topic_labels_cli   # validasi fingerprint
```

---

## 📊 Metrik Terverifikasi

| Metrik | Nilai | Cara Verifikasi |
|:---|---:|:---|
| Jumlah dokumen | **285** | 2021:29 · 2022:78 · 2023:46 · 2024:47 · 2025:85 |
| Ukuran kosakata | **1.849** | `data/intermediate/dictionary.gensim` |
| Jumlah topik $K$ | **8** |(alpha=asymmetric, eta=auto) |
| Coherence $C_V$ | **0,4256** | dihitung ulang dari `model/lda_model.gensim` |
| $U_{mass}$ | −10,7533 | *Online Variational Bayes* |
| Log Perplexity | −7,4518 | *Online Variational Bayes* |
| Topic Diversity | 0,7125 / 0,6917 | *Online Variational Bayes* |
| Passes / Iterations | 30 / 500 | *Online Variational Bayes* |
| $\alpha$ / $\eta$ / seed | asymmetric / auto / 42 | *Online Variational Bayes* |

Semua angka **dapat direproduksi ulang** dengan menjalankan notebook terkait.
Rujukan resmi: [`GROUND_TRUTH.md`](GROUND_TRUTH.md)

---

## 🗺️ Peta Artefak

> Setiap artefak → siapa pembuatnya → siapa pemakainya.

### Data

| Artefak | Dibuat oleh | Consumed by |
|:---|:---|:---|
| `data/raw/html_source/2021-2025.html` | Save Page As repositori | Notebook 01 |
| `data/raw/thesis_scraped.csv` | Notebook 01 | Notebook 02, 03 |
| `data/processed/thesis_for_pipeline.csv` | Notebook 02 | Notebook 03 |
| `data/intermediate/dataset_preprocessed.csv` | Notebook 03 | Notebook 04, 04b, 05, 08 |
| `data/intermediate/dictionary.gensim` | Notebook 03 | Notebook 04, 04b, 08 |
| `data/intermediate/corpus.pkl` | Notebook 03 | Notebook 04, 05, 08 |
| `docs/data_lineage_by_year.csv` | Notebook 02 | Lampiran Bab III |
| `docs/excluded_documents.csv` | Notebook 02 | Lampiran Bab III |

### Model

| Artefak | Dibuat oleh | Consumed by |
|:---|:---|:---|
| `model/hyperparameter_results.csv` | `src.grid_search` | Notebook 04b |
| `model/lda_model.gensim` | `src.modeling` | Notebook 05, 07, 08, app |
| `model/topic_distribution.csv` | `src.modeling` | Referensi perbandingan |
| `model/topic_labels.csv` / `.json` | `src.labeling` | Notebook 07, app |
| `model/evaluation_metrics.csv` | `src.modeling` | Dokumentasi |
| `model/lda_visualization.html` | `src.modeling` | Lampiran Bab IV |
| `experiments/*` | `src.evaluation_lda` | Lampiran Bab IV |

### Label

| Artefak | Dibuat oleh | Consumed by |
|:---|:---|:---|
| `_label_work/labels.json` | Notebook 07 (`src.labeling`) | Notebook 04, 06, 08, app |

### Statistik & Proyeksi

| Artefak | Dibuat oleh | Consumed by |
|:---|:---|:---|
| `_stats_forecast/forecast_table.csv` | Notebook 06 | app (halaman Proyeksi) |
| `_stats_forecast/backtest_results.csv` | Notebook 06 | Lampiran Bab IV |
| `_stats_forecast/share_intervals.csv` | Notebook 06 | app (halaman Tren) |
| `_stats_forecast/per_topic_tests.csv` | Notebook 06 | Lampiran Bab IV |
| `_stats_forecast/composition_tests_report.md` | Notebook 06 | Lampiran Bab IV |

### Artefak Final

| Artefak | Dibuat oleh | Consumed by |
|:---|:---|:---|
| `_fix_final/topic_distribution_final.csv` | **Notebook 08** | app, tests |
| `_fix_final/topic_counts_final.csv` | **Notebook 08** | app (Beranda, Distribusi) |
| `_fix_final/topic_by_year_final.csv` | **Notebook 08** | Notebook 06, app (Tren) |
| `_fix_final/top_words_final.csv` | **Notebook 08** | Notebook 07 |
| `_fix_final/evaluation_summary.csv` | **Notebook 08** | app (Metodologi) |

---

## 🧩 Modul `src/`

| Modul | Peran |
|:---|:---|
| `config.py` | Semua path & parameter — **satu sumber kebenaran** |
| `scraping.py` | Ekstraksi HTML → CSV (Notebook 01) |
| `data_lineage.py` | Jejak lineage + dokumen tereliminasi (Notebook 02) |
| `preprocessing.py` | 7 tahap prapemrosesan teks (Notebook 03) |
| `grid_search.py` | Grid search 84 konfigurasi (Notebook 04b) |
| `modeling.py` | Pelatihan LDA final (Notebook 04) |
| `labeling.py` | Human-in-the-Loop labeling (Notebook 07) |
| `evaluation_lda.py` | Multi-seed, sweep K, perplexity (Notebook 05) |
| `forecast.py` | Chi-square, Wilson, WMA, backtest (Notebook 06) |
| `build_final.py` | **Regenerasi artefak final** (Notebook 08) |
| `indonesian_stopwords.py` | Daftar stopword + domain SI |
| `topic_labels.py` | Validasi fingerprint SHA-256 |
| `topic_labels_cli.py` | CLI wrapper validasi label |
| `data_manager.py` | Helper normalisasi indeks topik |

---

## 📚 Dokumentasi

| Berkas | Isi |
|:---|:---|
| [`GROUND_TRUTH.md`](GROUND_TRUTH.md) | 🔒 **Rujukan resmi** angka & metrik |
| [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md) | 🔒 Kamus data & definisi kolom |
| `docs/stability_report.md` | Analisis kestabilan multi-seed |
| `docs/perplexity_report.md` | Analisis perplexity |
| `docs/topic_quality.md` | Mutu & koherensi topik |
| `docs/assignment_confidence.md` | Keyakinan penugasan dokumen |
| `_stats_forecast/EVAL_NOTES.md` | Panduan klaim untuk Bab V |

---

## ⚖️ Catatan Metodologis

1. **Inferensi penuh (Notebook 08).** `src.modeling` melatih pada 228 dokumen
   (80%) lalu memprediksi 285 — dokumen di hold-out (20%) bisa "lompat" topik.
   Notebook 08 menghitung ulang dengan *inferensi penuh* sehingga konsisten.

2. **Pemilihan K=8.** Grid search menunjukkan K=9 mencapai $C_V$ tertinggi
   (0,4314). K=8 dipilih atas pertimbangan gabungan: jumlah dokumen per topik
   lebih memadai (min. 21) dan struktur topik lebih mudah diinterpretasikan.

3. **Proyeksi WMA bersifat indikatif.** Hanya tersedia 5 titik historis (2021–2025).
   Walk-forward backtest menunjukkan WMA **tidak** mengalahkan baseline rerata
   historis untuk 7 dari 8 topik, sehingga proyeksi hanya indikatif.

4. **Pelabelan Human-in-the-Loop.** Label ditetapkan oleh peneliti melalui
   analisis top-15 words, divalidasi dosen pembimbing, lalu dikunci dengan
   fingerprint SHA-256 agar tidak pernah melayang ke model berbeda.

---

## 📄 Deployment

```bash
# Streamlit Community Cloud
git push
# Settings → Main file path: app/main.py
```
Panduan lengkap: [`DEPLOY.md`](DEPLOY.md)