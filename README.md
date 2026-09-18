# LDA Topic Modeling — Skripsi SI UIN Raden Fatah

Sistem analisis topik otomatis berbasis **Latent Dirichlet Allocation (LDA)** untuk memetakan tema penelitian skripsi Program Studi Sistem Informasi UIN Raden Fatah Palembang.

## Hasil Model Terbaru

| Metrik | Nilai |
|--------|-------|
| Jumlah topik (K) | **3** |
| Coherence $C_V$ | **0.3575** |
| Coherence $U_{Mass}$ | **-5.6618** |
| Train Log Perplexity | **-7.2401** |
| Hold-out Test Log Perplexity | **-8.5822** |
| Jumlah dokumen | **309** |
| Vocabulary | **2114 kata** |
| Parameter alpha | **auto** |
| Parameter eta | **auto** |
| Passes | **30** |
| Iterations | **500** |
| Random state | **42** |

## Topik yang Ditemukan (K=3)

| # | Topik | Jumlah Dokumen | Coherence ($C_V$) |
|---|-------|----------------|-------------------|
| 1 | Keamanan & Manajemen Risiko Sistem | 78 | 0.3259 |
| 2 | Evaluasi Kepuasan Pengguna Sistem | 92 | 0.4192 |
| 3 | Pengembangan Sistem Informasi Manajemen | 139 | 0.3274 |

## Prerequisites

- Python 3.11+
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

Training model LDA (K=3):
```bash
python pipeline.py --no-auto-tune --num-topics 3 --passes 30 --iterations 500 --alpha auto --eta auto
```

Jalankan dashboard Streamlit:
```bash
streamlit run app/main.py
```

Dashboard tersedia di `http://localhost:8501`

## Fitur Dashboard

| Halaman / Navigasi | Fitur |
|--------------------|-------|
| 📈 Overview | Statistik utama, distribusi topik dominan, timeline skripsi per tahun, dan Quality Alerts |
| 🔵 Visualisasi LDA | Interaktif PyLDAvis dashboard (self-contained, tanpa CDN eksternal) |
| 📊 Model Metrics | Coherence score ($C_V$, $U_{Mass}$), perplexity, jumlah topik |
| 🏷️ Topic Analysis | Top words per topik, word cloud, serta daftar dokumen relevan |
| 🔍 Document Search | Pencarian dokumen berdasarkan kata kunci/tahun/topik |
| 📖 Data Info | Statistik dataset & preview data preprocessed |
| 📈 Prediksi Tren | Tren topik per tahun & proyeksi forecast WMA (2026-2027) |
| ℹ️ Info Model | Detail parameter model LDA yang sedang digunakan |
| 🧪 Evaluasi Model | Halaman khusus grid search K + perbandingan metrik & justifikasi |
| 📊 Analisis Tren | Halaman khusus analisis tren & detail proyeksi WMA |

## Struktur Direktori

```
.
├── app/
│   ├── main.py                          # Entry point dashboard Streamlit
│   └── pages/
│       ├── 4_evaluasi_model.py          # Halaman evaluasi model & grid search
│       └── 5_analisis_tren.py           # Halaman analisis tren WMA
├── data/
│   ├── raw/                             # Data mentah (CSV hasil scraping)
│   ├── processed/                       # thesis_for_pipeline.csv
│   └── intermediate/
│       ├── dataset_preprocessed.csv     # Output preprocessing
│       └── dictionary.gensim            # Gensim dictionary
├── model/
│   ├── lda_model.gensim                 # Model terlatih LDA
│   ├── topic_labels.json                # Label topik (KeyBERT)
│   ├── topic_labels.csv                 # Backup label CSV
│   ├── topic_distribution.csv           # Distribusi topik per dokumen
│   ├── evaluation_metrics.csv           # Metrik evaluasi (Coherence, Perplexity)
│   ├── lda_visualization.html           # PyLDAvis self-contained HTML
│   ├── trend_prediction.csv             # Hasil proyeksi WMA
│   ├── top_words_per_topic.png          # Plot kata kunci per topik
│   └── tren_topik_per_tahun.png         # Plot tren per tahun
├── logs/                                # Log eksekusi pipeline
├── auto_labeling.py                     # Auto-labeling topik berbasis KeyBERT
├── data_manager.py                      # Data loader & state manager dashboard
├── indonesian_stopwords.py              # Custom stopwords bahasa Indonesia
├── pipeline.py                          # Pipeline utama LDA
├── preprocess.py                        # Preprocessing teks (cleansing, tokenize, stemming)
├── trend_analyzer.py                    # Algoritma forecasting WMA
├── hyperparameter_tuning.py             # Grid search penentuan K optimal
├── requirements.txt
└── README.md
```

## Metodologi Analisis Tren

Prediksi tren topik menggunakan **Weighted Moving Average (WMA)**:
- **Time-series**: Menggunakan data historis 5 tahun (2021–2025).
- **WMA Weighting**: Memberikan bobot linier lebih besar pada tahun-tahun terbaru untuk menangkap arah tren penelitian terkini.
- **Prediksi**: Menghitung estimasi proporsi topik untuk tahun 2026 dan 2027.
