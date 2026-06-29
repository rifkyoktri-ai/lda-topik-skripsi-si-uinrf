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

### Topik Labels (Final)
| Topik | Label | Coherence |
|-------|-------|-----------|
| 1 | Manajemen Risiko & Usability Website | 0.4234 |
| 2 | Analisis Sentimen & Klasifikasi Teks | 0.4021 |
| 3 | Kualitas Layanan Sistem Informasi | 0.5428 |
| 4 | Service Quality & User Satisfaction | 0.6012 |
| 5 | Extreme Programming (XP) | 0.5718 |
| 6 | Kepuasan Pengguna Sistem Informasi | 0.5607 |
| 7 | Usability & Layanan Akademik | 0.4905 |

## [v1.2.0] — 2026-06-29 (Post-Audit Review)

### Fixed
- Stopwords terlalu agresif → dibuat `get_conservative_stopwords()` (132 kata generik)
- Preprocessing ulang dengan stopwords konservatif → vocab naik 2082 → 2396
- Coherence turun 0.4559 → 0.4259 → sekarang **0.5132** (naik 12.6%)
- Overlap label XP (Topik 1 & 6 duplikat) → dihilangkan dengan deduplikasi otomatis
- LABEL_MAPPING diperluas: 35 → 60+ pattern untuk cakupan label lebih baik

### Added
- `get_conservative_stopwords()` di `indonesian_stopwords.py`
- Coherence per topik di `docs/justifikasi_k7.md`
- Tabel coherence per topik di justifikasi

### Notes
- Coherence final: **0.5132** (↑ 0.0873 dari retrain sebelumnya)
- Vocab: 2396 kata unik (↑ 314 dari 2082)
- Perplexity: -6.6562 (↑ dari -6.9835)
- Tidak ada topik dengan coherence < 0.30
- Semua 7 topik unik (tidak ada duplikat label)
- Parameter: alpha=symmetric, eta=symmetric (best K=7 dari grid search)
