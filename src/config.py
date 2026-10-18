"""
config.py — SATU SUMBER PENGATURAN untuk Seluruh Proyek

Semua lokasi folder dan parameter model dikumpulkan di sini.
Tujuannya agar tidak ada yang perlu menulis path secara manual
di banyak file berbeda — cukup ubah di sini, berlaku di mana-mana.

Cara pakai dari notebook:
    import sys; sys.path.insert(0, '..')   # dari folder notebooks/
    from src import config as C
    df = pd.read_csv(C.PREPROCESSED_CSV)
"""

from pathlib import Path

# ── LOKASI FOLDER ────────────────────────────────────────────────────────────
# ROOT = folder utama proyek (parent dari folder src/)
REPO = Path(__file__).resolve().parent.parent

# Folder data mentah (hasil scraping)
RAW_DIR = REPO / "data" / "raw"
HTML_SOURCE_DIR = RAW_DIR / "html_source"   # file HTML dari repository kampus
SCRAPED_CSV = RAW_DIR / "thesis_scraped.csv"  # hasil scraping mentah

# Folder data yang sudah dibersihkan manual
PROCESSED_DIR = REPO / "data" / "processed"
PIPELINE_CSV = PROCESSED_DIR / "thesis_for_pipeline.csv"

# Folder data hasil preprocessing (siap masuk model)
INTERMEDIATE_DIR = REPO / "data" / "intermediate"
PREPROCESSED_CSV = INTERMEDIATE_DIR / "dataset_preprocessed.csv"
DICTIONARY_PATH = INTERMEDIATE_DIR / "dictionary.gensim"  # kamus kata unik
CORPUS_PATH = INTERMEDIATE_DIR / "corpus.pkl"              # dokumen dalam format angka
PHRASER_BIGRAM = INTERMEDIATE_DIR / "phraser_bigram.pkl"   # model penggabungan 2 kata
PHRASER_TRIGRAM = INTERMEDIATE_DIR / "phraser_trigram.pkl" # model penggabungan 3 kata

# ── FOLDER HASIL (ARTEFAK OUTPUT) ────────────────────────────────────────────
# File-file ini dihasilkan oleh pipeline dan HARUS di-commit ke Git
# agar hasil skripsi bisa diverifikasi ulang kapan saja.
OUTPUTS_DIR = REPO / "outputs"
MODEL_DIR = OUTPUTS_DIR / "model"    # model LDA tersimpan + evaluasi
TOPICS_DIR = OUTPUTS_DIR / "topics"  # distribusi topik per dokumen & per tahun
AUDIT_DIR = OUTPUTS_DIR / "audit"    # laporan audit keyakinan penugasan
STATS_DIR = OUTPUTS_DIR / "forecast" # uji statistik + proyeksi

# ── FILE MODEL LDA ────────────────────────────────────────────────────────────
LDA_MODEL = MODEL_DIR / "lda_model.gensim"         # model LDA tersimpan
LDA_ID2WORD = MODEL_DIR / "lda_model.gensim.id2word"  # kamus internal model
HYPERPARAMS_CSV = MODEL_DIR / "hyperparameter_results.csv"  # hasil grid search
TOPIC_DISTRIBUTION = MODEL_DIR / "topic_distribution.csv"   # distribusi (lama)
TOPIC_LABELS_CSV = MODEL_DIR / "topic_labels.csv"           # nama topik (tabel)
TOPIC_LABELS_JSON = MODEL_DIR / "topic_labels.json"         # nama topik (JSON)
EVAL_METRICS_CSV = MODEL_DIR / "evaluation_metrics.csv"     # metrik evaluasi
EXPERIMENT_HISTORY = MODEL_DIR / "experiment_history.csv"   # riwayat percobaan

# ── FILE HASIL PER TAHAP ──────────────────────────────────────────────────────
TOPIC_COUNTS_FINAL = TOPICS_DIR / "topic_counts_final.csv"      # jumlah skripsi per topik
TOPIC_DIST_FINAL = TOPICS_DIR / "topic_distribution_final.csv"  # distribusi per dokumen (final)
TOPIC_BY_YEAR_FINAL = TOPICS_DIR / "topic_by_year_final.csv"    # tren topik per tahun
TOPIC_SENSITIVITY = TOPICS_DIR / "topic_by_year_sensitivity.csv"  # uji sensitivitas
TOP_WORDS_FINAL = TOPICS_DIR / "top_words_final.csv"            # kata kunci per topik
DATASET_CORRECTED = TOPICS_DIR / "dataset_corrected.csv"        # dataset yang dikoreksi
CORPUS_CORRECTED = TOPICS_DIR / "corpus_corrected.pkl"          # corpus yang dikoreksi
EVAL_SUMMARY_CSV = MODEL_DIR / "evaluation_summary.csv"         # rekap metrik evaluasi

# ── FILE AUDIT ────────────────────────────────────────────────────────────────
ASSIGNMENT_AUDIT_MD = AUDIT_DIR / "assignment_audit.md"   # laporan keyakinan penugasan
EXCLUDED_DOCUMENTS = AUDIT_DIR / "excluded_documents.csv" # daftar dokumen yang dibuang
DATA_LINEAGE_BY_YEAR = AUDIT_DIR / "data_lineage_by_year.csv"  # jumlah dokumen per tahap

# ── FILE LABEL (NAMA RESMI TOPIK) ─────────────────────────────────────────────
LABELS_JSON = MODEL_DIR / "labels.json"  # nama resmi 8 topik + sidik jari SHA-256

# ── GAMBAR ────────────────────────────────────────────────────────────────────
# Dihasilkan ulang oleh src/make_figures.py — jangan diedit manual
FIGURES_DIR = OUTPUTS_DIR / "figures"
WORDCLOUD_DIR = FIGURES_DIR / "wordclouds"

# ── STATISTIK & PROYEKSI ──────────────────────────────────────────────────────
EXPERIMENTS_DIR = OUTPUTS_DIR / "experiments"   # hasil uji kestabilan model

FORECAST_CSV = STATS_DIR / "forecast_table.csv"         # tabel proyeksi 2026-2027
BACKTEST_CSV = STATS_DIR / "backtest_results.csv"        # hasil uji mundur
SHARE_INTERVALS = STATS_DIR / "share_intervals.csv"      # interval kepercayaan Wilson
PER_TOPIC_TESTS = STATS_DIR / "per_topic_tests.csv"      # uji tren per topik
COMPOSITION_REPORT = STATS_DIR / "composition_tests_report.md"  # laporan chi-square
SENSITIVITY_REPORT = STATS_DIR / "sensitivity_report.md" # laporan uji sensitivitas
BACKTEST_SUMMARY = STATS_DIR / "backtest_summary.md"     # ringkasan backtest

# ── DOKUMENTASI ───────────────────────────────────────────────────────────────
DOCS_DIR = REPO / "docs"
EVAL_NOTES = DOCS_DIR / "EVAL_NOTES.md"
DATA_LINEAGE_CSV = AUDIT_DIR / "data_lineage_by_year.csv"
EXCLUDED_DOCS_CSV = AUDIT_DIR / "excluded_documents.csv"

# Folder log — dibuat saat runtime, tidak perlu sudah ada sebelumnya
LOGS_DIR = REPO / "logs"


# ── TAHUN & JUMLAH DOKUMEN ────────────────────────────────────────────────────
YEARS = [2021, 2022, 2023, 2024, 2025]   # tahun data yang dianalisis
PROJECTION_YEARS = [2026, 2027]           # tahun proyeksi
CUTOFF_DATE = "19 September 2026"         # batas akhir pengambilan data

TOTAL_DOCS_RAW = 312        # jumlah skripsi hasil scraping mentah
TOTAL_DOCS_PROCESSED = 309  # setelah pembersihan awal
TOTAL_DOCS_FINAL = 285      # setelah preprocessing penuh (yang masuk model)
VOCAB_SIZE = 1849            # jumlah kata unik dalam kamus model


# ── PENGATURAN PREPROCESSING ──────────────────────────────────────────────────
MIN_TOKEN_LENGTH = 3         # buang kata yang panjangnya ≤ 2 huruf
MIN_DOCUMENT_TOKENS = 5      # buang dokumen yang punya < 5 kata
MIN_NGRAM_COUNT = 2          # minimal kemunculan agar dua kata digabung jadi satu
NGRAM_THRESHOLD = 7.0        # ambang skor untuk penggabungan n-gram
DICT_NO_BELOW = 2            # buang kata yang muncul di < 2 dokumen
DICT_NO_ABOVE = 0.40         # buang kata yang muncul di > 40% dokumen
ENGLISH_ABSTRACT_THRESHOLD = 0.12  # jika > 12% kata adalah bahasa Inggris → anggap abstrak Inggris


# ── PARAMETER MODEL LDA FINAL ─────────────────────────────────────────────────
NUM_TOPICS = 8            # jumlah topik yang dipilih (K=8)
LDA_ALPHA = "auto"        # prior distribusi dokumen-topik (diestimasi dari data)
LDA_ETA = "auto"          # prior distribusi topik-kata (diestimasi dari data)
LDA_PASSES = 30           # berapa kali seluruh data dibaca ulang saat pelatihan
LDA_ITERATIONS = 500      # jumlah iterasi optimasi per dokumen
RANDOM_SEED = 42          # benih acak — dikunci agar hasil bisa direproduksi
TRAIN_TEST_SPLIT = 0.20   # 20% data disimpan sebagai data uji (hold-out)

# Catatan: model final dilatih dengan alpha="auto" (diestimasi dari data),
# bukan "asymmetric". Ini terbukti dari experiment_history.csv (baris 2026-09-26)
# dan vektor alpha ~0.0244 (hasil estimasi, bukan 0.125 milik mode "symmetric").

# ── PENGATURAN GRID SEARCH (PENCARIAN PARAMETER TERBAIK) ─────────────────────
GRID_MIN_K = 4              # jumlah topik minimum yang dicoba
GRID_MAX_K = 10             # jumlah topik maksimum yang dicoba
GRID_ALPHA_OPTIONS = ["symmetric", "asymmetric", "auto", 0.01, 0.1]  # pilihan alpha
GRID_ETA_OPTIONS = ["symmetric", "auto", 0.01, 0.1]                  # pilihan eta
GRID_PASSES = 10            # passes saat grid search (lebih sedikit agar cepat)
GRID_ITERATIONS = 400       # iterasi saat grid search


# ── PENGATURAN PROYEKSI WMA ───────────────────────────────────────────────────
WMA_WEIGHTS = [1, 2, 3, 4, 5]           # bobot WMA: tahun terbaru lebih berpengaruh
BACKTEST_TARGET_YEARS = [2023, 2024, 2025]  # tahun yang diuji dalam backtest
RELIABILITY_RATIO_THRESHOLD = 0.95      # batas rasio MAE WMA vs rata-rata historis


# ── METRIK RESMI (angka ini yang dipakai di skripsi) ─────────────────────────
# Dapat direproduksi ulang — lihat GROUND_TRUTH.md untuk verifikasi
COHERENCE_CV_OFFICIAL = 0.4256      # skor koherensi C_V
COHERENCE_UMASS_OFFICIAL = -10.7533 # skor koherensi U_Mass
LOG_PERPLEXITY_OFFICIAL = -7.4518   # log perplexity (semakin negatif = semakin baik)
TOPIC_DIVERSITY_TOP10 = 0.7125      # keragaman kata antar topik (top-10)
TOPIC_DIVERSITY_TOP15 = 0.6917      # keragaman kata antar topik (top-15)
PCT_THETA_UNDER_050 = 13.33         # % dokumen dengan keyakinan topik < 50%


def ensure_dirs():
    """Buat semua folder output jika belum ada. Aman dipanggil berkali-kali."""
    for d in (LOGS_DIR, DOCS_DIR, MODEL_DIR, TOPICS_DIR, STATS_DIR,
              AUDIT_DIR, INTERMEDIATE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def describe():
    """
    Tampilkan ringkasan konfigurasi proyek.
    Biasanya dipanggil di cell pertama setiap notebook untuk konfirmasi pengaturan.
    """
    lines = [
        "KONFIGURASI PROYEK",
        f"  Repo          : {REPO}",
        f"  Dokumen akhir : {TOTAL_DOCS_FINAL} ({', '.join(str(y) for y in YEARS)})",
        f"  Vocabulary    : {VOCAB_SIZE}",
        f"  Model LDA     : K={NUM_TOPICS}, alpha={LDA_ALPHA}, eta={LDA_ETA}, "
        f"passes={LDA_PASSES}, iterations={LDA_ITERATIONS}, seed={RANDOM_SEED}",
        f"  Inferensi     : Online Variational Bayes (bukan Gibbs Sampling)",
        f"  C_V resmi     : {COHERENCE_CV_OFFICIAL}",
        f"  Cutoff data   : {CUTOFF_DATE}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())