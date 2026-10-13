"""
config.py — SATU SUMBER KEBENARAN untuk seluruh proyek.

Semua path direktori dan parameter model dikumpulkan di sini supaya
notebook tidak lagi menulis path secara hardcoded di banyak tempat.

Notebook memakai:
    import sys; sys.path.insert(0, '..')      # dari folder notebooks/
    from src import config as C
    df = pd.read_csv(C.PREPROCESSED_CSV)
"""

from pathlib import Path

# ── LOKASI ──────────────────────────────────────────────────────────────────
# ROOT = parent folder dari src/
REPO = Path(__file__).resolve().parent.parent

# Data
RAW_DIR = REPO / "data" / "raw"
HTML_SOURCE_DIR = RAW_DIR / "html_source"
SCRAPED_CSV = RAW_DIR / "thesis_scraped.csv"

PROCESSED_DIR = REPO / "data" / "processed"
PIPELINE_CSV = PROCESSED_DIR / "thesis_for_pipeline.csv"

INTERMEDIATE_DIR = REPO / "data" / "intermediate"
PREPROCESSED_CSV = INTERMEDIATE_DIR / "dataset_preprocessed.csv"
DICTIONARY_PATH = INTERMEDIATE_DIR / "dictionary.gensim"
CORPUS_PATH = INTERMEDIATE_DIR / "corpus.pkl"
PHRASER_BIGRAM = INTERMEDIATE_DIR / "phraser_bigram.pkl"
PHRASER_TRIGRAM = INTERMEDIATE_DIR / "phraser_trigram.pkl"

# Model
MODEL_DIR = REPO / "outputs" / "model"
LDA_MODEL = MODEL_DIR / "lda_model.gensim"
LDA_ID2WORD = MODEL_DIR / "lda_model.gensim.id2word"   # dictionary vocab model
HYPERPARAMS_CSV = MODEL_DIR / "hyperparameter_results.csv"
TOPIC_DISTRIBUTION = MODEL_DIR / "topic_distribution.csv"
TOPIC_LABELS_CSV = MODEL_DIR / "topic_labels.csv"
TOPIC_LABELS_JSON = MODEL_DIR / "topic_labels.json"
EVAL_METRICS_CSV = MODEL_DIR / "evaluation_metrics.csv"
EXPERIMENT_HISTORY = MODEL_DIR / "experiment_history.csv"
HUMAN_VALIDATION_CSV = MODEL_DIR / "human_topic_validation.csv"

# Artefak hasil (dihasilkan pipeline, tapi WAJIB ikut commit agar
# skripsi dapat direproduksi - lihat catatan di README bagian outputs/)
TOPICS_DIR = REPO / "outputs" / "topics"
TOPIC_COUNTS_FINAL = TOPICS_DIR / "topic_counts_final.csv"
TOPIC_DIST_FINAL = TOPICS_DIR / "topic_distribution_final.csv"
TOPIC_BY_YEAR_FINAL = TOPICS_DIR / "topic_by_year_final.csv"
TOPIC_SENSITIVITY = TOPICS_DIR / "topic_by_year_sensitivity.csv"
TOP_WORDS_FINAL = TOPICS_DIR / "top_words_final.csv"
DATASET_CORRECTED = TOPICS_DIR / "dataset_corrected.csv"
CORPUS_CORRECTED = TOPICS_DIR / "corpus_corrected.pkl"
EVAL_SUMMARY_CSV = MODEL_DIR / "evaluation_summary.csv"

# Audit keyakinan penugasan dokumen
AUDIT_DIR = REPO / "outputs" / "audit"
ASSIGNMENT_AUDIT_MD = AUDIT_DIR / "assignment_audit.md"
EXCLUDED_DOCUMENTS = AUDIT_DIR / "excluded_documents.csv"
DATA_LINEAGE_BY_YEAR = AUDIT_DIR / "data_lineage_by_year.csv"

# Label (sumber kebenaran nama topik)
LABELS_JSON = MODEL_DIR / "labels.json"

# Statistik & proyeksi
STATS_DIR = REPO / "outputs" / "forecast"
FORECAST_CSV = STATS_DIR / "forecast_table.csv"
BACKTEST_CSV = STATS_DIR / "backtest_results.csv"
SHARE_INTERVALS = STATS_DIR / "share_intervals.csv"
PER_TOPIC_TESTS = STATS_DIR / "per_topic_tests.csv"
COMPOSITION_REPORT = STATS_DIR / "composition_tests_report.md"
SENSITIVITY_REPORT = STATS_DIR / "sensitivity_report.md"
BACKTEST_SUMMARY = STATS_DIR / "backtest_summary.md"
# Dokumentasi (murni .md - tidak ada .csv/.png di sini)
DOCS_DIR = REPO / "docs"
EVAL_NOTES = DOCS_DIR / "EVAL_NOTES.md"
DATA_LINEAGE_CSV = AUDIT_DIR / "data_lineage_by_year.csv"
EXCLUDED_DOCS_CSV = AUDIT_DIR / "excluded_documents.csv"

# Dibuat saat runtime, tidak perlu sudah ada
LOGS_DIR = REPO / "logs"


# ── TAHUN & KORPUS ───────────────────────────────────────────────────────────
YEARS = [2021, 2022, 2023, 2024, 2025]
PROJECTION_YEARS = [2026, 2027]
CUTOFF_DATE = "19 September 2026"

TOTAL_DOCS_RAW = 312
TOTAL_DOCS_PROCESSED = 309
TOTAL_DOCS_FINAL = 285
VOCAB_SIZE = 1849


# ── PRAPEMROSINGAN ───────────────────────────────────────────────────────────
MIN_TOKEN_LENGTH = 3          # buang token < 3 karakter (len(w) > 2)
MIN_DOCUMENT_TOKENS = 5       # buang dokumen dengan < 5 token
MIN_NGRAM_COUNT = 2           # Phrases(min_count=2)
NGRAM_THRESHOLD = 7.0
DICT_NO_BELOW = 2
DICT_NO_ABOVE = 0.40
ENGLISH_ABSTRACT_THRESHOLD = 0.12


# ── MODEL LDA FINAL ──────────────────────────────────────────────────────────
NUM_TOPICS = 8
LDA_ALPHA = "auto"
LDA_ETA = "auto"
LDA_PASSES = 30
LDA_ITERATIONS = 500
RANDOM_SEED = 42
TRAIN_TEST_SPLIT = 0.20       # hold-out 20% (untuk log perplexity)

# Catatan alpha: model final dilatih dengan alpha="auto" (estimasi dari data),
# bukan "asymmetric". Bukti: model/experiment_history.csv baris 2026-09-26
# (alpha=auto, eta=auto, C_V=0.4256) dan vektor alpha tersimpan ~0.0244
# (hasil estimasi), bukan 0.125 (nilai tetap 1/K milik "symmetric"/"asymmetric").

# Grid search
GRID_MIN_K = 4
GRID_MAX_K = 10
GRID_ALPHA_OPTIONS = ["symmetric", "asymmetric", "auto", 0.01, 0.1]
GRID_ETA_OPTIONS = ["symmetric", "auto", 0.01, 0.1]
GRID_PASSES = 10
GRID_ITERATIONS = 400


# ── PROYEKSI WMA ─────────────────────────────────────────────────────────────
WMA_WEIGHTS = [1, 2, 3, 4, 5]
BACKTEST_TARGET_YEARS = [2023, 2024, 2025]
RELIABILITY_RATIO_THRESHOLD = 0.95


# ── METRIK RESMI (dapat direproduksi ulang — lihat GROUND_TRUTH.md) ───────────
COHERENCE_CV_OFFICIAL = 0.4256
COHERENCE_UMASS_OFFICIAL = -10.7533
LOG_PERPLEXITY_OFFICIAL = -7.4518
TOPIC_DIVERSITY_TOP10 = 0.7125
TOPIC_DIVERSITY_TOP15 = 0.6917
PCT_THETA_UNDER_050 = 13.33


def ensure_dirs():
    """Pastikan semua direktori output ada."""
    for d in (LOGS_DIR, DOCS_DIR, MODEL_DIR, TOPICS_DIR, STATS_DIR,
              AUDIT_DIR, INTERMEDIATE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def describe():
    """Cetak konfigurasi — dipakai di cell pembuka tiap notebook."""
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