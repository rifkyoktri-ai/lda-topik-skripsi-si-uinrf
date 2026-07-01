# LDA Topic Modeling — Skripsi SI UIN Raden Fatah

Sistem analisis topik otomatis berbasis **Latent Dirichlet Allocation (LDA)**
untuk memetakan tema penelitian skripsi Program Studi Sistem Informasi
UIN Raden Fatah Palembang.

## Hasil Model

| Metrik | Nilai |
|--------|-------|
| Jumlah topik (K) | 7 |
| Coherence c_v | **0.5124** (kategori: Good >= 0.50) |
| Coherence u_mass | -1.6518 |
| Log Perplexity | -6.5858 |
| Jumlah dokumen | 311 |
| Vocabulary | 2385 kata |
| Parameter alpha | symmetric |
| Parameter eta | symmetric |
| Passes | 30 |
| Iterations | 500 |
| Random state | 42 |

## 7 Topik yang Ditemukan

| # | Topik | Coherence |
|---|-------|-----------|
| 1 | Kepuasan Pengguna Sistem Informasi | 0.5355 |
| 2 | Sistem Informasi Akademik | 0.4788 |
| 3 | Analisis Kualitas Website (WebQual) | 0.4343 |
| 4 | Sistem Informasi Penjualan & Inventory | 0.3403 |
| 5 | Service Quality & User Satisfaction | 0.7161 |
| 6 | Sistem Informasi Konseling & Bimbingan | 0.4462 |
| 7 | Manajemen Risiko & Adopsi TI | 0.6354 |

## Prerequisites

- Python 3.11.9
- Windows 10/11
- RAM minimal 4GB (untuk KeyBERT model loading ~500MB)
- Virtual environment (wajib)

## Installation & Setup

```bash
# Clone repository
git clone https://github.com/rifkyoktri-ai/lda-topik-skripsi-si-uinrf.git
cd tes

# Buat virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Menjalankan Pipeline

Preprocessing data:
```bash
python preprocess.py
```

Training model LDA K=7:
```bash
python pipeline.py --no-auto-tune --num-topics 7 --passes 30 --iterations 500 --alpha symmetric --eta symmetric
```

Jalankan dashboard:
```bash
streamlit run app/main.py
```

Dashboard tersedia di `http://localhost:8501`

## Fitur Dashboard (8 Tab)

| Tab | Fitur |
|-----|-------|
| Overview | Statistik utama, distribusi topik, timeline per tahun |
| Visualisasi LDA | PyLDAvis interaktif (self-contained, tanpa CDN) |
| Model Metrics | Coherence score, perplexity, jumlah topik |
| Topic Analysis | Top words per topik, word cloud, daftar dokumen |
| Document Search | Pencarian dokumen berdasarkan judul/tahun/topik |
| Data Info | Statistik dataset, preview data |
| Prediksi Tren | Tren topik per tahun, forecast WMA 2026-2027 |
| Manage Labels | Edit label topik (tersimpan ke JSON + CSV) |

## Halaman Pendukung

| Halaman | Akses | Fungsi |
|---------|-------|--------|
| Evaluasi Model | `streamlit run app/pages/4_evaluasi_model.py` | Grid search K + pyLDAvis + justifikasi K=7 |
| Analisis Tren | `streamlit run app/pages/5_analisis_tren.py` | WMA forecast detail per topik |

## Struktur Direktori

```
tes/
├── app/
│   ├── main.py                          # Entry point dashboard (8 tab)
│   ├── pages/
│   │   ├── 4_evaluasi_model.py          # Grid search + pyLDAvis
│   │   └── 5_analisis_tren.py           # WMA trend forecast
│   └── __pycache__/
├── data/
│   ├── raw/                             # Data mentah (CSV hasil scraping)
│   ├── processed/                       # thesis_for_pipeline.csv
│   └── intermediate/
│       ├── dataset_preprocessed.csv     # Output preprocessing
│       └── dictionary.gensim            # Gensim dictionary
├── model/
│   ├── lda_model.gensim                 # Model terlatih K=7
│   ├── topic_labels.json                # Label 7 topik (KeyBERT + LABEL_MAPPING)
│   ├── topic_labels.csv                 # Backup CSV
│   ├── topic_distribution.csv           # Distribusi topik per dokumen
│   ├── hyperparameter_results.csv       # Hasil grid search 156 konfigurasi
│   ├── evaluation_metrics.csv           # Coherence, perplexity
│   ├── lda_visualization.html           # PyLDAvis self-contained
│   ├── pyldavis_lib.js                  # JS library untuk inlining
│   ├── trend_prediction.csv             # WMA 2026-2027
│   ├── top_words_per_topic.png          # Plot top words
│   └── tren_topik_per_tahun.png         # Plot tren tahunan
├── docs/
│   ├── justifikasi_k7.md                # Justifikasi metodologis K=7
│   ├── CHANGELOG.md                     # Riwayat perubahan
│   └── evaluation_plots/                # Plot evaluasi + word clouds
├── model/ (lengkap di atas)
├── notebook/                            # Jupyter notebooks eksplorasi
├── .streamlit/
│   └── config.toml                      # Tema blue-gold
├── auto_labeling.py                     # KeyBERT auto-labeling + LABEL_MAPPING
├── hyperparameter_tuning.py             # Grid search K=3-15 (156 konfigurasi)
├── indonesian_stopwords.py              # Stopwords (get_all + get_conservative)
├── pipeline.py                          # Pipeline LDA: preprocessing → model → labeling
├── preprocess.py                        # Text preprocessing (stemming, stopword removal)
├── trend_analyzer.py                    # WMA forecast
├── topic_optimization.py                # Optimal K search (coherence vs perplexity)
├── lda_evaluation.py                    # Evaluasi model
├── requirements.txt
└── .gitignore
```

## Metodologi Analisis Tren

Prediksi tren topik menggunakan **Weighted Moving Average (WMA)** dengan justifikasi:
- Data time-series hanya 5 titik (2021–2025) — terlalu sedikit untuk ARIMA atau exponential smoothing
- WMA memberi bobot linier lebih besar pada tahun terbaru, sesuai asumsi tren akademik yang berubah cepat
- Prediksi hanya 1 tahun ke depan (2026) untuk meminimalkan *error propagation*
- Prediksi 2027 bersifat **indikatif** (menggunakan data prediksi 2026 sebagai input, bukan data aktual)

**Kolom Arah** ditentukan berdasarkan slope regresi linier data historis 2021–2025, bukan perbandingan titik terakhir dengan prediksi. Ini memastikan arah mencerminkan tren jangka panjang, bukan fluktuasi tahun terakhir.

**Reliabilitas prediksi** (`pred_2027_reliability`) dikategorikan berdasarkan MAE_LOO (*leave-one-out* validation error):

| MAE_LOO | Reliabilitas | Arti |
|---------|-------------|------|
| < 0.05 | Tinggi | Error prediksi rendah — model WMA konsisten |
| 0.05–0.10 | Sedang | Error moderat — prediksi perlu interpretasi hati-hati |
| > 0.10 | Rendah | Error tinggi — prediksi tidak stabil |

## Troubleshooting

**Error: components.v1.html not found**
`import streamlit.components.v1 as components` — panggil `components.html()`, bukan `components.v1.html()`.

**PyLDAvis tidak muncul**
Pastikan `lda_visualization.html` sudah self-contained (pipeline otomatis meng-inline JS).

**ModuleNotFoundError: tqdm**
`pip install tqdm` — sudah ada di requirements.txt.
