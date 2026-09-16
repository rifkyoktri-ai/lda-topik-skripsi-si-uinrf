import logging
import sys
import pandas as pd
import pickle
import os
import re
import argparse
from datetime import datetime

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from indonesian_stopwords import get_conservative_stopwords
from gensim import corpora
from gensim.models.phrases import Phrases, Phraser
from gensim.models import TfidfModel

# ---------------------------------------------------------------------------
# 1. DOMAIN-SPECIFIC STOPWORDS & BOILERPLATE ACADEMIC TERMS
# ---------------------------------------------------------------------------
# Additional academic boilerplate words commonly found in thesis abstracts
DOMAIN_STOPWORDS_INSTITUTIONAL = {
    'skripsi', 'mahasiswa', 'universitas', 'fakultas',
    'program', 'studi', 'jurusan', 'uin', 'raden', 'fatah',
    'palembang', 'negeri', 'islam'
}

DOMAIN_STOPWORDS_ACADEMIC = {
    'sistem', 'informasi', 'berbasis', 'metode', 'analisis', 'pengaruh',
    'implementasi', 'studi', 'kasus', 'aplikasi', 'rancang', 'bangun',
    'website', 'web', 'program', 'tugas', 'akhir', 'menggunakan',
    'penelitian', 'hasil', 'berdasarkan', 'tujuan', 'dilakukan', 'membuat',
    'dirancang', 'diperoleh', 'pengembangan', 'mengembangkan', 'tersebut',
    'dapat', 'pada', 'dengan', 'guna', 'serta', 'dalam', 'secara', 'yaitu',
    'prosedur', 'tahapan', 'penerapan', 'uji', 'pengujian', 'proses', 'dibuat',
    'pembahasan', 'peneliti', 'penulisan', 'disusun', 'dibangun', 'ditulis',
    'bertujuan', 'ditinjau', 'diukur', 'diuraikan',
    # --- Surgical Fix: Administrative & Domain Noise Ablation ---
    'bantu', 'layan', 'pelayanan', 'giat', 'kegiatan', 'legalisir',
    'dimensi', 'akademik', 'kampus', 'universitas', 'mahasiswa', 'dosen',
    'fakultas', 'nilai', 'data'
}

# ---------------------------------------------------------------------------
# POST-STEM STOPWORDS: root forms produced BY Sastrawi that bypass the
# pre-stem filter. Apply this set AFTER stemming (double-pass fix).
# ---------------------------------------------------------------------------
POST_STEM_STOPWORDS = {
    # Stemmed administrative / service words
    'layan', 'bantu', 'kelola', 'tunjuk', 'dukung', 'sedia', 'terima',
    # Stemmed quantitative-research jargon
    'variabel', 'tingkat', 'nilai', 'hasil', 'pengaruh',
    'faktor', 'indikator', 'dimensi', 'uji', 'ukur', 'hitung', 'atur',
    # NOTE: 'kualitas' and 'analis' RESTORED — vital IS-domain co-occurrence anchors
    # Stemmed generic adjectives / adverbs
    'baik', 'besar', 'perlu', 'mudah', 'tinggi', 'rendah', 'banyak',
    'dapat', 'ada', 'salah', 'laku', 'beri',
    # Stemmed institutional / educational words
    'sekolah', 'kerja', 'guna', 'pakai', 'tugas', 'akhir',
}

ENGLISH_INDICATOR_WORDS = {
    'the', 'and', 'this', 'that', 'with', 'from', 'for', 'was', 'were',
    'using', 'used', 'study', 'research', 'paper', 'results', 'method',
    'system', 'based', 'development', 'analysis', 'data', 'information',
    'faculty', 'university', 'student', 'journal', 'abstract'
}

# ---------------------------------------------------------------------------
# LOGGING & PATH SETUP
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'logs/preprocess_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

PROSES_DIR = 'data/intermediate'
os.makedirs(PROSES_DIR, exist_ok=True)
os.makedirs('logs', exist_ok=True)

# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------
def is_english_abstract(text: str, threshold: float = 0.12) -> bool:
    """
    FEATURE 3: Bilingual Filtering.
    Heuristic filter to detect if an abstract is predominantly English
    (e.g., duplicate English abstract), ensuring Sastrawi stemmer only
    processes Indonesian text.
    """
    if not isinstance(text, str) or not text.strip():
        return False
    words = re.findall(r'\b[a-z]+\b', text.lower())
    if not words:
        return False
    english_count = sum(1 for w in words if w in ENGLISH_INDICATOR_WORDS)
    ratio = english_count / len(words)
    return ratio >= threshold

def bersihkan(teks: str) -> str:
    """Basic lowercasing and character regex cleaning."""
    if not isinstance(teks, str):
        return ''
    teks = teks.lower()
    teks = re.sub(r'[^a-zA-Z\s]', ' ', teks)
    teks = re.sub(r'\s+', ' ', teks).strip()
    return teks

def tokenize_basic(teks: str, stopwords_set: set) -> list:
    """Initial tokenization filtering out short words and domain stopwords."""
    return [w for w in teks.split() if w not in stopwords_set and len(w) > 2]

def stem_token_or_compound(token: str, stemmer) -> str:
    """
    Safely stems unigrams or compounds (e.g., 'sistem_informasi' -> 'sistem_informasi')
    without breaking the compound structure.
    """
    if '_' in token:
        parts = token.split('_')
        stemmed_parts = [stemmer.stem(p) for p in parts]
        return '_'.join(stemmed_parts)
    return stemmer.stem(token)

# ---------------------------------------------------------------------------
# MAIN PIPELINE EXECUTOR
# ---------------------------------------------------------------------------
def run_preprocessing(use_tfidf: bool = False, min_ngram_count: int = 2):
    logger.info("=" * 60)
    logger.info("PREPROCESSING PIPELINE (REFACTORED ML ENGINEER VERSION)")
    logger.info("=" * 60)

    logger.info("\n[1/5] Memuat data...")
    raw_path = 'data/processed/thesis_for_pipeline.csv'
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Dataset mentah tidak ditemukan di {raw_path}")

    df = pd.read_csv(raw_path)
    logger.info(f"  Total Dokumen: {len(df)}")
    logger.info(f"  Dengan abstrak: {df['abstract'].notna().sum()}")

    # Combined stopword set
    all_stopwords = (
        get_conservative_stopwords() |
        DOMAIN_STOPWORDS_INSTITUTIONAL |
        DOMAIN_STOPWORDS_ACADEMIC
    )

    # ---------------------------------------------------------------------------
    # FEATURE 3: BILINGUAL FILTERING / TEXT SELECTION
    # ---------------------------------------------------------------------------
    logger.info("\n[2/5] Handling Text & Bilingual Filtering (Detecting English Abstracts)...")
    
    clean_texts = []
    dropped_en_count = 0

    for idx, row in df.iterrows():
        title_str = str(row['title']) if pd.notna(row['title']) else ''
        abstract_str = str(row['abstract']) if pd.notna(row['abstract']) else ''

        # If abstract is in English, fallback to title only to prevent Sastrawi stemmer corruption
        if is_english_abstract(abstract_str):
            dropped_en_count += 1
            combined = title_str
        else:
            combined = (title_str + ' ' + abstract_str).strip()

        clean_texts.append(combined)

    df['teks_gabung'] = clean_texts
    logger.info(f"  Dokumen dengan Abstrak Bahasa Inggris difilter (fallback ke judul): {dropped_en_count}")

    df['teks_bersih'] = df['teks_gabung'].apply(bersihkan)

    # Initial tokenization
    tokens_raw = [tokenize_basic(t, all_stopwords) for t in df['teks_bersih']]

    # ---------------------------------------------------------------------------
    # FEATURE 2: N-GRAM EXTRACTION (GENSIM BIGRAM & TRIGRAM PHRASER)
    # ---------------------------------------------------------------------------
    logger.info("\n[3/5] Building Gensim N-Gram Phrase Models (Bigram & Trigram)...")
    
    # Train Bigram model
    bigram_phrases = Phrases(tokens_raw, min_count=min_ngram_count, threshold=7.0, delimiter='_')
    bigram_phraser = Phraser(bigram_phrases)

    tokens_bigram = [bigram_phraser[doc] for doc in tokens_raw]

    # Train Trigram model on top of bigrams
    trigram_phrases = Phrases(tokens_bigram, min_count=min_ngram_count, threshold=7.0, delimiter='_')
    trigram_phraser = Phraser(trigram_phrases)

    tokens_ngram = [trigram_phraser[doc] for doc in tokens_bigram]

    # Save Phrase Models for inference
    bigram_phraser.save(f'{PROSES_DIR}/phraser_bigram.pkl')
    trigram_phraser.save(f'{PROSES_DIR}/phraser_trigram.pkl')
    logger.info(f"  Phraser Bigram & Trigram saved to {PROSES_DIR}/")

    # ---------------------------------------------------------------------------
    # SASTRAWI STEMMING & FILTERING
    # ---------------------------------------------------------------------------
    logger.info("\n[4/5] Stemming & Filtering Stopwords...")
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    # Combined post-stem filter: union of all_stopwords + POST_STEM_STOPWORDS
    # This performs the SECOND PASS — catching root forms produced by Sastrawi
    post_stem_filter = all_stopwords | POST_STEM_STOPWORDS

    stemmed_tokens_list = []
    for doc in tokens_ngram:
        # Pass 1 guard: filter pre-stem stopwords before feeding to stemmer
        pre_filtered = [t for t in doc if t not in all_stopwords and len(t) > 2]

        # Stemming
        stemmed_doc_raw = [stem_token_or_compound(t, stemmer) for t in pre_filtered]

        # Pass 2: filter post-stem stopwords (catches Sastrawi root forms)
        stemmed_doc = [
            t for t in stemmed_doc_raw
            if t not in post_stem_filter and len(t) > 2
        ]
        stemmed_tokens_list.append(stemmed_doc)

    df['tokens'] = stemmed_tokens_list

    # Filter out empty documents
    df = df[df['tokens'].apply(len) > 0].reset_index(drop=True)
    logger.info(f"  Total Dokumen setelah filter: {len(df)}")

    # ---------------------------------------------------------------------------
    # FEATURE 4: DICTIONARY, CORPUS & TF-IDF SPARSITY HANDLING
    # ---------------------------------------------------------------------------
    logger.info("\n[5/5] Building Dictionary & Vectorization Corpus (BoW vs TF-IDF)...")
    
    dictionary = corpora.Dictionary(df['tokens'].tolist())
    dictionary.filter_extremes(no_below=2, no_above=0.40)
    
    DICT_PATH = f'{PROSES_DIR}/dictionary.gensim'
    CORPUS_PATH = f'{PROSES_DIR}/corpus.pkl'

    dictionary.save(DICT_PATH)
    bow_corpus = [dictionary.doc2bow(doc) for doc in df['tokens'].tolist()]

    if use_tfidf:
        logger.info("  [SPARSITY TOGGLE] Mengaplikasikan TF-IDF Vectorization untuk meredam kata berfrekuensi tinggi...")
        tfidf_model = TfidfModel(bow_corpus)
        corpus = tfidf_model[bow_corpus]
        tfidf_model.save(f'{PROSES_DIR}/tfidf_model.gensim')
    else:
        logger.info("  [SPARSITY TOGGLE] Menggunakan Standard Bag-of-Words (BoW) Corpus...")
        corpus = bow_corpus

    with open(CORPUS_PATH, 'wb') as f:
        pickle.dump(corpus, f)

    logger.info(f"  Dictionary ({len(dictionary)} kata unik) & Corpus disimpankan di {PROSES_DIR}/")

    # Save Preprocessed DataFrame
    df_final = pd.DataFrame({
        'ID': df['id'],
        'Judul': df['title'],
        'Abstrak': df['abstract'].fillna(''),
        'Tahun': df['year'],
        'teks_gabung': df['teks_gabung'],
        'teks_bersih': df['teks_bersih'],
        'tokens': df['tokens'].apply(lambda x: str(x))
    })

    df_final.to_csv(f'{PROSES_DIR}/dataset_preprocessed.csv', index=False)

    logger.info("\n" + "=" * 60)
    logger.info("PREPROCESSING SELESAI!")
    logger.info(f"  Output CSV       : {PROSES_DIR}/dataset_preprocessed.csv")
    logger.info(f"  Total Dokumen    : {len(df_final)}")
    logger.info(f"  Ukuran Vocab     : {len(dictionary)} kata/n-gram")
    logger.info("=" * 60)

    # Top 20 terms log
    word_freq = [(dictionary[id], dictionary.dfs.get(id, 0)) for id in dictionary]
    word_freq.sort(key=lambda x: -x[1])
    logger.info("\n  Top 20 Terms / N-grams terbanyak:")
    for w, f in word_freq[:20]:
        logger.info(f"    {w:30s}: {f} dokumen")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Refactored Preprocessing Pipeline")
    parser.add_argument('--use-tfidf', action='store_true', help="Aktifkan TF-IDF vectorizationalih-alih standard BoW")
    parser.add_argument('--min-ngram-count', type=int, default=2, help="Minimum frekuensi untuk membentuk N-gram Phraser")
    args = parser.parse_args()

    run_preprocessing(use_tfidf=args.use_tfidf, min_ngram_count=args.min_ngram_count)

