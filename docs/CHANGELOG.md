## [v1.1.0] — 2026-06-29

### Fixed
- Tambah `tqdm>=4.66.0` ke requirements.txt (FIX 1)
- Tambah tanggal training di sidebar dashboard (FIX 2)
- Hapus unused import `Counter` dari auto_labeling.py (FIX 4)
- `components.v1.html()` → `components.html()` (import sudah `components` dari `streamlit.components.v1`)

### Added
- Justifikasi metodologis pemilihan K=7 di `docs/justifikasi_k7.md` (FIX 3A)
- Tabel grid search hasil hyperparameter tuning di halaman Evaluasi Model (FIX 3B)
- `save_topic_labels()` function di auto_labeling.py untuk integrasi pipeline
- LABEL_MAPPING diperluas (18 → 35 pattern) untuk mencakup lebih banyak topik
- Deduplikasi label otomatis (suffix top_words_filtered jika label_final duplikat)

### Notes
- Coherence score: **0.4259** (K=7, setelah retrain dengan stopwords terbaru)
- Model retrain: **Ya** — preprocessing diulang karena `indonesian_stopwords.py` berubah (172 baris, vocab 2082 → turun dari 2179)
- Perplexity: -6.9835, Vocabulary: 2082 kata unik, Dokumen: 311

### Topik Labels
| Topik | Label |
|-------|-------|
| 1 | Extreme Programming (XP) |
| 2 | Analisis Sentimen & NLP |
| 3 | Audit & Evaluasi SIAKAD |
| 4 | Teknologi Informasi Akademik |
| 5 | Service Quality & Kepuasan Pengguna |
| 6 | Extreme Programming (XP) (Usability) |
| 7 | Kepuasan Pengguna SI Akademik |
