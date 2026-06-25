import multiprocessing
import pandas as pd
import numpy as np
import pickle
import ast
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from gensim import corpora
from gensim.models import LdaModel, CoherenceModel

import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

from modeling.keybert_labeler import label_topics_with_keybert
from stopwords_id import get_all_stopwords

PROSES_DIR = 'data/proses'
MODEL_DIR  = 'model'

def get_dominant_topic(lda_model, corpus):
    dominant_topics = []
    for bow in corpus:
        topic_probs = lda_model.get_document_topics(bow, minimum_probability=0)
        dominant = max(topic_probs, key=lambda x: x[1])
        dominant_topics.append({
            'topik_dominan'  : dominant[0] + 1,
            'prob_dominan'   : round(dominant[1], 4)
        })
    return pd.DataFrame(dominant_topics)

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 60)
    print("PIPELINE TOPIC MODELING LDA + KeyBERT")
    print("=" * 60)

    print("\n[1/5] Memuat data preprocessed...")
    df = pd.read_csv(f'{PROSES_DIR}/dataset_preprocessed.csv')
    df['tokens'] = df['tokens'].apply(ast.literal_eval)
    tokenized_docs = df['tokens'].tolist()
    print(f"  Dokumen : {len(df)}")

    print("\n[2/5] Memuat dictionary & corpus...")
    DICT_PATH   = f'{PROSES_DIR}/dictionary.gensim'
    CORPUS_PATH = f'{PROSES_DIR}/corpus.pkl'

    if os.path.exists(DICT_PATH) and os.path.exists(CORPUS_PATH):
        dictionary = corpora.Dictionary.load(DICT_PATH)
        with open(CORPUS_PATH, 'rb') as f:
            corpus = pickle.load(f)
        print(f"  Dictionary : {len(dictionary)} kata unik")
        print(f"  Corpus     : {len(corpus)} dokumen")
    else:
        print("  File tidak ditemukan, rebuild...")
        dictionary = corpora.Dictionary(tokenized_docs)
        dictionary.filter_extremes(no_below=2, no_above=0.85)
        corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
        dictionary.save(DICT_PATH)
        with open(CORPUS_PATH, 'wb') as f:
            pickle.dump(corpus, f)
        print(f"  Dictionary : {len(dictionary)} kata unik")

    print("\n[3/5] Training model LDA final (K=14, langsung)...")
    OPTIMAL_K = 14
    ALPHA     = 'auto'
    BETA      = 'auto'
    PASSES    = 20

    print(f"  K       : {OPTIMAL_K}")
    print(f"  Alpha   : {ALPHA}")
    print(f"  Beta    : {BETA}")
    print(f"  Passes  : {PASSES}")

    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=OPTIMAL_K,
        alpha=ALPHA,
        eta=BETA,
        passes=PASSES,
        random_state=42,
        per_word_topics=True
    )

    print("\n  Evaluasi model final...")
    coherence_model = CoherenceModel(
        model=lda_model,
        texts=tokenized_docs,
        dictionary=dictionary,
        coherence='c_v',
        processes=1
    )
    coherence_final = coherence_model.get_coherence()
    perplexity_final = lda_model.log_perplexity(corpus)

    print(f"  Coherence (CV)  : {coherence_final:.4f}")
    print(f"  Log Perplexity : {perplexity_final:.4f}")

    print("\nTop 10 kata per topik:")
    print("=" * 60)
    for i in range(OPTIMAL_K):
        top_words = lda_model.show_topic(i, topn=10)
        words = ' | '.join([f'{w} ({p:.3f})' for w, p in top_words])
        print(f"Topik {i+1:2d}: {words}")
    print("=" * 60)

    n_cols = 3
    n_rows = (OPTIMAL_K + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
    axes = axes.flatten()

    COLORS = plt.cm.tab10.colors

    for i in range(OPTIMAL_K):
        top_words = lda_model.show_topic(i, topn=10)
        words  = [w for w, _ in top_words]
        weights = [p for _, p in top_words]
        ax = axes[i]
        ax.barh(words[::-1], weights[::-1], color=COLORS[i % len(COLORS)])
        ax.set_title(f'Topik {i+1}', fontweight='bold')
        ax.set_xlabel('Bobot')
        ax.grid(axis='x', linestyle='--', alpha=0.5)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(f'{MODEL_DIR}/top_words_per_topic.png', dpi=150)
    plt.close()
    print(f"  Plot top words -> {MODEL_DIR}/top_words_per_topic.png")

    print("\n[4/5] Menentukan topik dominan & KeyBERT labeling...")

    topic_df = get_dominant_topic(lda_model, corpus)
    df_result = pd.concat([df[['Nama', 'Judul', 'Tahun']].reset_index(drop=True), topic_df], axis=1)

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
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 str(val), ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.title(f'Distribusi Jumlah Dokumen per Topik (K={OPTIMAL_K})', fontsize=13, fontweight='bold')
    plt.xlabel('Topik')
    plt.ylabel('Jumlah Dokumen')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{MODEL_DIR}/distribusi_topik.png', dpi=150)
    plt.close()
    print(f"  Plot distribusi topik -> {MODEL_DIR}/distribusi_topik.png")

    pivot = df_result.groupby(['Tahun', 'topik_dominan']).size().unstack(fill_value=0)

    plt.figure(figsize=(12, 6))
    sns.heatmap(
        pivot, annot=True, fmt='d', cmap='YlOrRd',
        linewidths=0.5, linecolor='grey',
        cbar_kws={'label': 'Jumlah Dokumen'}
    )
    plt.title('Distribusi Tren Topik per Tahun', fontsize=13, fontweight='bold')
    plt.xlabel('Topik')
    plt.ylabel('Tahun')
    plt.tight_layout()
    plt.savefig(f'{MODEL_DIR}/tren_topik_per_tahun.png', dpi=150)
    plt.close()
    print(f"  Plot tren tahunan -> {MODEL_DIR}/tren_topik_per_tahun.png")

    all_stopwords = get_all_stopwords()
    topic_labels = label_topics_with_keybert(lda_model, all_stopwords)

    labels_rows = []
    for tid_str, info in topic_labels.items():
        labels_rows.append({
            'topic_id'   : int(tid_str) + 1,
            'label'      : info['label_final'],
            'description': f"Topic with {len(df_result[df_result['topik_dominan'] == int(tid_str) + 1])} documents",
            'keywords'   : ';'.join(info['top_words'])
        })

    labels_df = pd.DataFrame(labels_rows)
    labels_df.to_csv(f'{MODEL_DIR}/topic_labels.csv', index=False)
    print(f"  topic_labels.csv -> {MODEL_DIR}/topic_labels.csv")
    for _, row in labels_df.iterrows():
        print(f"  Topik {row['topic_id']:2d}: {row['label']}")

    print("\n[5/5] Menyimpan model & visualisasi...")

    vis_data = gensimvis.prepare(
        lda_model, corpus, dictionary,
        mds='mmds',
        sort_topics=False
    )
    pyLDAvis.save_html(vis_data, f'{MODEL_DIR}/lda_visualization.html')
    print(f"  LDA viz -> {MODEL_DIR}/lda_visualization.html")

    lda_model.save(f'{MODEL_DIR}/lda_model.gensim')
    df_result.to_csv(f'{MODEL_DIR}/topic_distribution.csv', index=False)

    eval_df_final = pd.DataFrame({
        'Metrik'  : ['Jumlah Topik (K)', 'Coherence Score (CV)', 'Log Perplexity'],
        'Nilai'   : [OPTIMAL_K, round(coherence_final, 4), round(perplexity_final, 4)]
    })
    eval_df_final.to_csv(f'{MODEL_DIR}/evaluation_metrics.csv', index=False)

    print(f"\nSemua output disimpan ke folder {MODEL_DIR}/")
    print("=" * 60)
    print("PIPELINE SELESAI")
    print("=" * 60)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
