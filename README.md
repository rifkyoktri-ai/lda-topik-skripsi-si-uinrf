# 📊 Pemetaan Topik Penelitian Skripsi Sistem Informasi (LDA K=8)

> **Program Studi Sistem Informasi · UIN Raden Fatah Palembang · 2021–2025**  
> Dashboard analisis & pemodelan topik Latent Dirichlet Allocation (LDA) untuk 285 skripsi terindeks.

---

## 🏛️ Struktur Repositori

| Berkas / Direktori | Peran |
| :--- | :--- |
| **`GROUND_TRUTH.md`** | 🔒 **Satu-satunya rujukan resmi** angka & metrik ($K=8$, $C_V=0{,}4256$, 285 dokumen). |
| **`DATA_DICTIONARY.md`** | 🔒 Kamus data, alur lineage, & definisi setiap kolom. |
| **`_fix_final/`** | 🔒 Artefak dataset corrected, distribusi topik, & kata kunci resmi. |
| **`_stats_forecast/`** | 🔒 Statistik (interval Wilson 95%), backtest walk-forward, proyeksi WMA, uji Spearman. |
| **`_label_work/labels.json`** | 🔒 Definisi 8 label topik resmi + fingerprint SHA-256. |
| **`model/`** | 🔒 Model Gensim (`lda_model.gensim`), PyLDAvis HTML, hasil grid search. |
| **`app/`** | 🟢 Dashboard Streamlit (8 halaman). |
| **`data/`** | Korpus mentah → bersih (312 → 309 → 285 dokumen). |
| **`docs/`** | Laporan analisis pendukung (stabilitas, perplexity, kualitas topik, konfigurasi). |

---

## 🚀 Menjalankan Dashboard

```bash
# 1. Aktifkan Virtual Environment (Windows PowerShell)
.\.venv\Scripts\activate

# 2. Install dependensi (opsional bila sudah terpasang)
pip install -r requirements.txt

# 3. Jalankan Dashboard Streamlit
streamlit run app/main.py
```

Aplikasi terbuka di `http://localhost:8501`.

---

## 🔬 Pipeline Reproduksi (opsional)

```bash
python preprocess.py          # data/processed -> data/intermediate (285 dok)
python pipeline.py            # pelatihan LDA K=8 + artefak model
python run_evaluation.py      # studi kestabilan multi-seed & sweep K
python _stats_forecast/stats_forecast.py   # statistik & proyeksi WMA
python run_tests.py           # 5 unit test konsistensi data
```

> **Catatan**: `pipeline.py` memvalidasi fingerprint `labels.json` terhadap model
> sebelum TRAINING. Ganti label hanya melalui procedure Human-in-the-Loop
> (`_label_work/make_labels.py`) agar lock tetap konsisten.

---

## 📊 Metrik Terverifikasi

| Metrik | Nilai |
| :--- | :---: |
| Jumlah dokumen | 285 (29/78/46/47/85) |
| Ukuran kosakata | 1.849 |
| Jumlah topik $K$ | 8 |
| Coherence $C_V$ | **0,4256** |
| $U_{mass}$ | −10,7533 |
| Log Perplexity | −7,4518 |
| $\alpha$ / $\eta$ / passes / iter | asymmetric / auto / 30 / 500 |
| Metode inferensi | Online Variational Bayes |

Semua angka di atas dapat direproduksi ulang dari `model/lda_model.gensim`.