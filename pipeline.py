import argparse
import logging
import sys
import os
import pickle
import ast
import multiprocessing
from datetime import datetime
from typing import List, Tuple, Optional, Dict

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from sklearn.model_selection import train_test_split

import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

from indonesian_stopwords import get_all_stopwords
from auto_labeling import label_topics_keybert, save_topic_labels

# ---------------------------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'logs/pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

PROSES_DIR = 'data/intermediate'
MODEL_DIR = 'model'

# ---------------------------------------------------------------------------
# FEATURE 4: EXPERIMENT TRACKING LOGGING FUNCTION
# ---------------------------------------------------------------------------
def log_experiment_history(
    random_state: int,
    k: int,
    alpha: str,
    eta: str,
    coherence_cv: float,
    coherence_umass: float,
    train_perplexity: float,
    test_perplexity: float,
    passes: int,
    iterations: int,
    total_docs: int,
    vocab_size: int
):
    """
    FEATURE 4: Experiment Tracking.
    Records model metadata and multi-metric metrics to CSV for full auditability.
    """
    history_file = os.path.join(MODEL_DIR, 'experiment_history.csv')
    new_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'random_state': random_state,
        'k_topics': k,
        'alpha': str(alpha),
        'eta': str(eta),
        'coherence_cv': round(coherence_cv, 4),
        'coherence_umass': round(coherence_umass, 4),
        'train_perplexity': round(train_perplexity, 4),
        'test_perplexity': round(test_perplexity, 4),
        'passes': passes,
        'iterations': iterations,
        'total_docs': total_docs,
        'vocab_size': vocab_size
    }
    
    df_new = pd.DataFrame([new_entry])
    if os.path.exists(history_file):
        df_history = pd.read_csv(history_file)
        df_history = pd.concat([df_history, df_new], ignore_index=True)
    else:
        df_history = df_new
        
    df_history.to_csv(history_file, index=False)
    logger.info(f"  [EXPERIMENT TRACKER] Run logged to {history_file}")


def get_dominant_topic(lda_model, corpus):
    dominant_topics = []
    for bow in corpus:
        topic_probs = lda_model.get_document_topics(bow, minimum_probability=0)
        dominant = max(topic_probs, key=lambda x: x[1])
        dominant_topics.append({
            'topik_dominan': dominant[0] + 1,
            'prob_dominan': round(dominant[1], 4)
        })
    return pd.DataFrame(dominant_topics)

# ---------------------------------------------------------------------------
# FEATURE 1: ASYMMETRIC PRIORS IN MODEL INITIALIZATION
# ---------------------------------------------------------------------------
def train_model(
    corpus,
    dictionary,
    num_topics: int,
    passes: int = 30,
    iterations: int = 500,
    alpha: str = 'auto',
    eta: str = 'auto',
    random_state: int = 42
) -> LdaModel:
    """
    RESCUE PROTOCOL: Adaptive Hyperparameters & BoW.
    Initializes LDA with auto document-topic prior (alpha='auto')
    and auto topic-word prior (eta='auto') to learn asymmetric distributions dynamically.
    """
    model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        alpha=alpha,
        eta=eta,
        passes=passes,
        iterations=iterations,
        random_state=random_state,
        per_word_topics=True
    )
    return model

# ---------------------------------------------------------------------------
# FEATURE 2 & 3: MULTI-METRIC COHERENCE & HOLD-OUT TEST PERPLEXITY TUNING
# ---------------------------------------------------------------------------
def auto_tune(
    corpus_train,
    corpus_test,
    dictionary,
    tokenized_docs,
    k_range: range = range(4, 13),
    passes: int = 20,
    iterations: int = 400,
    alpha: str = 'asymmetric',
    eta: str = 'auto',
    random_state: int = 42
) -> Tuple[int, float, float, float, float, LdaModel]:
    """
    FEATURE 2 & 3: Multi-Metric Coherence Validation & Hold-out Perplexity.
    Evaluates both c_v and u_mass, as well as train vs hold-out test perplexity across K.
    """
    eval_results = []
    models = {}

    for k in k_range:
        logger.info(f"  Auto-tune K={k} (Alpha={alpha}, Eta={eta})...")
        model = train_model(
            corpus_train, dictionary, k,
            passes=passes, iterations=iterations,
            alpha=alpha, eta=eta, random_state=random_state
        )
        
        # FEATURE 2: Multi-Metric Coherence (C_V and U_MASS)
        cm_cv = CoherenceModel(
            model=model,
            texts=tokenized_docs,
            dictionary=dictionary,
            coherence='c_v',
            processes=1
        )
        cv_score = cm_cv.get_coherence()

        cm_umass = CoherenceModel(
            model=model,
            corpus=corpus_train,
            dictionary=dictionary,
            coherence='u_mass',
            processes=1
        )
        umass_score = cm_umass.get_coherence()

        # FEATURE 3: Hold-out Perplexity (Train vs Test)
        train_perp = model.log_perplexity(corpus_train)
        test_perp = model.log_perplexity(corpus_test)

        models[k] = model
        eval_results.append({
            'k': k,
            'cv': cv_score,
            'umass': umass_score,
            'train_perp': train_perp,
            'test_perp': test_perp
        })
        logger.info(f"  K={k:2d} | C_V: {cv_score:.4f} | U_Mass: {umass_score:.4f} | Train Perp: {train_perp:.4f} | Test Perp: {test_perp:.4f}")

    # Rank by C_V as primary, tie-break with U_Mass
    eval_results.sort(key=lambda x: (-x['cv'], -x['umass']))
    best_res = eval_results[0]
    optimal_k = best_res['k']
    
    logger.info(f"\n  [OPTIMAL K SELECTION] K={optimal_k} | C_V={best_res['cv']:.4f} | U_Mass={best_res['umass']:.4f}")

    return (
        optimal_k,
        best_res['cv'],
        best_res['umass'],
        best_res['train_perp'],
        best_res['test_perp'],
        models[optimal_k]
    )


def plot_topic_words(lda_model, optimal_k: int):
    n_cols = min(3, optimal_k)
    n_rows = (optimal_k + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
    axes = axes.flatten()
    COLORS = plt.cm.tab10.colors

    for i in range(optimal_k):
        top_words = lda_model.show_topic(i, topn=10)
        words = [w for w, _ in top_words]
        weights = [p for _, p in top_words]
        ax = axes[i]
        ax.barh(words[::-1], weights[::-1], color=COLORS[i % len(COLORS)])
        ax.set_title(f'Topik {i + 1}', fontweight='bold')
        ax.set_xlabel('Bobot')
        ax.grid(axis='x', linestyle='--', alpha=0.5)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(f'{MODEL_DIR}/top_words_per_topic.png', dpi=150)
    plt.close()
    logger.info(f"  Plot top words -> {MODEL_DIR}/top_words_per_topic.png")


def plot_topic_distribution(df_result: pd.DataFrame, optimal_k: int):
    topic_counts = df_result['topik_dominan'].value_counts().sort_index()

    plt.figure(figsize=(10, 5))
    bars = plt.bar(
        [f'Topik {t}' for t in topic_counts.index],
        topic_counts.values,
        color=plt.cm.tab10.colors[:len(topic_counts)],
        edgecolor='white',
        linewidth=0.7
    )
    for bar, val in zip(bars, topic_counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 str(val), ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.title(f'Distribusi Jumlah Dokumen per Topik (K={optimal_k})',
              fontsize=13, fontweight='bold')
    plt.xlabel('Topik')
    plt.ylabel('Jumlah Dokumen')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{MODEL_DIR}/distribusi_topik.png', dpi=150)
    plt.close()
    logger.info(f"  Plot distribusi topik -> {MODEL_DIR}/distribusi_topik.png")


def plot_topic_trend(df_result: pd.DataFrame):
    pivot = df_result.groupby(['Tahun', 'topik_dominan']).size().unstack(fill_value=0)

    plt.figure(figsize=(12, 6))
    plt.imshow(pivot.values, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            plt.text(j, i, str(pivot.values[i, j]), ha='center', va='center', fontsize=9)
    plt.colorbar(label='Jumlah Dokumen')
    plt.xticks(ticks=range(pivot.shape[1]), labels=pivot.columns)
    plt.yticks(ticks=range(pivot.shape[0]), labels=pivot.index)
    plt.title('Distribusi Tren Topik per Tahun', fontsize=13, fontweight='bold')
    plt.xlabel('Topik')
    plt.ylabel('Tahun')
    plt.tight_layout()
    plt.savefig(f'{MODEL_DIR}/tren_topik_per_tahun.png', dpi=150)
    plt.close()
    logger.info(f"  Plot tren tahunan -> {MODEL_DIR}/tren_topik_per_tahun.png")

    pivot.to_csv(f'{MODEL_DIR}/topic_trend_counts.csv')
    pivot_normalized = pivot.div(pivot.sum(axis=1), axis=0)
    pivot_normalized.reset_index().to_csv(
        f'{MODEL_DIR}/topic_trend.csv', index=False, encoding='utf-8-sig'
    )
    logger.info(f"  Trend data (proportion) -> {MODEL_DIR}/topic_trend.csv")


def main():
    parser = argparse.ArgumentParser(description='Pipeline LDA Topic Modeling (Senior Data Scientist Version)')
    parser.add_argument('--num-topics', type=int, default=None,
                        help='Jumlah topik (default: auto-tune dari 4-12)')
    parser.add_argument('--passes', type=int, default=50,
                        help='Jumlah passes LDA (default: 50)')
    parser.add_argument('--iterations', type=int, default=400,
                        help='Jumlah iterasi LDA (default: 400)')
    parser.add_argument('--alpha', type=str, default='auto',
                        help='Alpha parameter (default: auto)')
    parser.add_argument('--eta', type=str, default='auto',
                        help='Eta parameter (default: auto)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed untuk reproduksibilitas (default: 42)')
    parser.add_argument('--label-method', type=str, default='keybert',
                        choices=['keybert'],
                        help='Metode pelabelan (default: keybert)')
    parser.add_argument('--no-auto-tune', action='store_true',
                        help='Nonaktifkan auto-tune (wajib set --num-topics)')
    parser.add_argument('--min-k', type=int, default=3,
                        help='K minimum untuk auto-tune (default: 3)')
    parser.add_argument('--max-k', type=int, default=10,
                        help='K maksimum untuk auto-tune (default: 10)')
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    logger.info("=" * 60)
    logger.info("PIPELINE TOPIC MODELING LDA (REFACTORED DATA SCIENTIST VERSION)")
    logger.info("=" * 60)

    logger.info(f"\n[1/6] Memuat data preprocessed...")
    df = pd.read_csv(f'{PROSES_DIR}/dataset_preprocessed.csv')
    df['tokens'] = df['tokens'].apply(ast.literal_eval)
    tokenized_docs = df['tokens'].tolist()
    logger.info(f"  Dokumen: {len(df)}")

    logger.info(f"\n[2/6] Memuat dictionary & corpus...")
    dict_path = f'{PROSES_DIR}/dictionary.gensim'
    corpus_path = f'{PROSES_DIR}/corpus.pkl'

    if os.path.exists(dict_path) and os.path.exists(corpus_path):
        dictionary = corpora.Dictionary.load(dict_path)
        with open(corpus_path, 'rb') as f:
            corpus = pickle.load(f)
        logger.info(f"  Dictionary: {len(dictionary)} kata/n-gram")
        logger.info(f"  Corpus: {len(corpus)} dokumen")
    else:
        dictionary = corpora.Dictionary(tokenized_docs)
        dictionary.filter_extremes(no_below=2, no_above=0.40)
        corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
        dictionary.save(dict_path)
        with open(corpus_path, 'wb') as f:
            pickle.dump(corpus, f)
        logger.info(f"  Dictionary: {len(dictionary)} kata/n-gram")

    # RESCUE PROTOCOL: Strictly Raw Bag-of-Words (BoW) Corpus
    # Any TF-IDF transformation is completely bypassed to prevent Dirichlet distribution collapse
    logger.info("  [RESCUE PROTOCOL] Training LDA EXCLUSIVELY on Raw Count Bag-of-Words (BoW) Corpus...")

    # ---------------------------------------------------------------------------
    # FEATURE 3: HOLD-OUT SET TESTING (80/20 TRAIN-TEST SPLIT)
    # ---------------------------------------------------------------------------
    logger.info("\n[3/6] Splitting Hold-out Test Set (80% Train / 20% Test)...")
    corpus_train, corpus_test = train_test_split(corpus, test_size=0.20, random_state=args.seed)
    logger.info(f"  Train corpus: {len(corpus_train)} dokumen | Hold-out test corpus: {len(corpus_test)} dokumen")

    # ---------------------------------------------------------------------------
    # FEATURE 1 & 2: TRAINING MODEL LDA WITH ASYMMETRIC PRIORS & MULTI-METRIC AUTO-TUNE
    # ---------------------------------------------------------------------------
    logger.info(f"\n[4/6] Training & Validasi Model LDA (Alpha='{args.alpha}', Eta='{args.eta}')...")

    if args.no_auto_tune or args.num_topics is not None:
        k = args.num_topics or 5
        logger.info(f"  Manual K={k} (auto-tune disabled)")
        lda_model = train_model(
            corpus_train, dictionary, k,
            passes=args.passes, iterations=args.iterations,
            alpha=args.alpha, eta=args.eta, random_state=args.seed
        )
        OPTIMAL_K = k
        
        cm_cv = CoherenceModel(
            model=lda_model, texts=tokenized_docs,
            dictionary=dictionary, coherence='c_v', processes=1
        )
        cv_final = cm_cv.get_coherence()
        
        cm_umass = CoherenceModel(
            model=lda_model, corpus=corpus_train,
            dictionary=dictionary, coherence='u_mass', processes=1
        )
        umass_final = cm_umass.get_coherence()

        train_perp_final = lda_model.log_perplexity(corpus_train)
        test_perp_final = lda_model.log_perplexity(corpus_test)
    else:
        logger.info(f"  Auto-tune K ({args.min_k}-{args.max_k})...")
        k_range = range(args.min_k, args.max_k + 1)
        (
            OPTIMAL_K,
            cv_final,
            umass_final,
            train_perp_final,
            test_perp_final,
            lda_model
        ) = auto_tune(
            corpus_train, corpus_test, dictionary, tokenized_docs,
            k_range=k_range, passes=args.passes, iterations=args.iterations,
            alpha=args.alpha, eta=args.eta, random_state=args.seed
        )

    # ---------------------------------------------------------------------------
    # FEATURE 4: LOG EXPERIMENT HISTORY TO CSV
    # ---------------------------------------------------------------------------
    log_experiment_history(
        random_state=args.seed,
        k=OPTIMAL_K,
        alpha=args.alpha,
        eta=args.eta,
        coherence_cv=cv_final,
        coherence_umass=umass_final,
        train_perplexity=train_perp_final,
        test_perplexity=test_perp_final,
        passes=args.passes,
        iterations=args.iterations,
        total_docs=len(corpus),
        vocab_size=len(dictionary)
    )

    plot_topic_words(lda_model, OPTIMAL_K)

    logger.info(f"\n[5/6] Menentukan topik dominan & labeling...")

    topic_df = get_dominant_topic(lda_model, corpus)
    df_result = pd.concat([df[['ID', 'Judul', 'Tahun']].reset_index(drop=True), topic_df], axis=1)

    plot_topic_distribution(df_result, OPTIMAL_K)
    plot_topic_trend(df_result)

    all_stopwords = get_all_stopwords()

    # Pass judul dokumen per topik ke KeyBERT
    topic_titles_dict = {}
    for _, row in df_result.iterrows():
        tid = int(row['topik_dominan'])
        if tid not in topic_titles_dict:
            topic_titles_dict[tid] = []
        topic_titles_dict[tid].append(str(row['Judul']))

    logger.info("  Labeling method: KeyBERT (dari judul skripsi per topik)")
    topic_labels = label_topics_keybert(lda_model, all_stopwords, topic_titles=topic_titles_dict)

    logger.info("  Menghitung per-topic coherence...")
    coherence_per_topic = {}
    try:
        for tid_str in topic_labels.keys():
            tid = int(tid_str)
            top_words = [w for w, _ in lda_model.show_topic(tid, topn=20)]
            cm_topic = CoherenceModel(
                texts=tokenized_docs,
                dictionary=dictionary,
                topics=[top_words],
                coherence='c_v',
                processes=1
            )
            coherence_per_topic[tid_str] = round(cm_topic.get_coherence(), 4)
    except Exception as e:
        logger.warning(f"  Gagal menghitung per-topic coherence: {e}")
        for tid_str in topic_labels.keys():
            coherence_per_topic[tid_str] = 0.0

    for tid_str, info in topic_labels.items():
        info['quality_coherence'] = coherence_per_topic.get(tid_str, 0.0)

    save_topic_labels(topic_labels, MODEL_DIR)

    from data_manager import to_one_indexed, to_zero_indexed

    labels_rows = []
    for tid_str, info in topic_labels.items():
        tid = to_one_indexed(int(tid_str))
        doc_count = len(df_result[df_result['topik_dominan'] == tid])
        labels_rows.append({
            'topic_id': tid,
            'label': info['label_final'],
            'description': f"Topic with {doc_count} documents",
            'keywords': ';'.join(info['top_words'][:5]),
            'label_score': info['score'],
            'quality_coherence': coherence_per_topic.get(tid_str, '')
        })

    labels_df = pd.DataFrame(labels_rows)
    labels_df.to_csv(f'{MODEL_DIR}/topic_labels.csv', index=False)
    logger.info(f"  topic_labels.csv -> {MODEL_DIR}/topic_labels.csv")

    logger.info(f"\n[6/6] Menyimpan model & visualisasi...")

    vis_data = gensimvis.prepare(
        lda_model, corpus, dictionary,
        mds='mmds',
        sort_topics=False
    )
    pyLDAvis.save_html(vis_data, f'{MODEL_DIR}/lda_visualization.html')
    
    viz_path = f'{MODEL_DIR}/lda_visualization.html'
    lib_path = f'{MODEL_DIR}/pyldavis_lib.js'
    if os.path.exists(lib_path):
        with open(viz_path, 'r', encoding='utf-8') as f:
            viz_html = f.read()
        with open(lib_path, 'r', encoding='utf-8') as f:
            lib_js = f.read()
        cdn_url = 'https://cdn.jsdelivr.net/gh/bmabey/pyLDAvis@3.4.0/pyLDAvis/js/ldavis.v1.0.0.js'
        viz_html = viz_html.replace(f'<script type="text/javascript" src="{cdn_url}"></script>',
                                    f'<script type="text/javascript">{lib_js}</script>')
        with open(viz_path, 'w', encoding='utf-8') as f:
            f.write(viz_html)
        logger.info(f"  JS library inlined -> self-contained HTML")

    lda_model.save(f'{MODEL_DIR}/lda_model.gensim')
    df_result.to_csv(f'{MODEL_DIR}/topic_distribution.csv', index=False)

    eval_df = pd.DataFrame({
        'Metrik': [
            'Jumlah Topik (K)',
            'Coherence Score (C_V)',
            'Coherence Score (U_Mass)',
            'Train Log Perplexity',
            'Hold-out Test Log Perplexity',
            'Alpha Prior',
            'Eta Prior'
        ],
        'Nilai': [
            OPTIMAL_K,
            round(cv_final, 4),
            round(umass_final, 4),
            round(train_perp_final, 4),
            round(test_perp_final, 4),
            str(args.alpha),
            str(args.eta)
        ]
    })
    eval_df.to_csv(f'{MODEL_DIR}/evaluation_metrics.csv', index=False)

    logger.info(f"\n" + "=" * 60)
    logger.info("PIPELINE SELESAI")
    logger.info(f"  K                   = {OPTIMAL_K}")
    logger.info(f"  Coherence (C_V)     = {cv_final:.4f}")
    logger.info(f"  Coherence (U_Mass)  = {umass_final:.4f}")
    logger.info(f"  Train Perplexity    = {train_perp_final:.4f}")
    logger.info(f"  Test Perplexity     = {test_perp_final:.4f}")
    logger.info("=" * 60)

    print("\n" + "=" * 60)
    print("PIPELINE SELESAI")
    print(f"  K                   = {OPTIMAL_K}")
    print(f"  Coherence (C_V)     = {cv_final:.4f}")
    print(f"  Coherence (U_Mass)  = {umass_final:.4f}")
    print(f"  Train Perplexity    = {train_perp_final:.4f}")
    print(f"  Hold-out Test Perp  = {test_perp_final:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()

