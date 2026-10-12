"""
Skrip Evaluasi Model & Robustness Study (TUGAS A - E)
Menjalankan studi kestabilan multi-seed, sweep K, perplexity train/test,
kualitas topik, dan distribusi proporsi per tahun.

Semua keluaran ditulis ke folder `experiments/` agar direktori root tetap bersih.
"""
import ast
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split
from pathlib import Path

from . import config as C

sys.stdout.reconfigure(line_buffering=True)

np.random.seed(42)

# Parameter model final sesuai GROUND_TRUTH.md
FINAL_K = 8
FINAL_ALPHA = 'asymmetric'
FINAL_ETA = 'auto'
FINAL_PASSES = 30
FINAL_ITERATIONS = 500
FINAL_SEED = 42

EXP_DIR = Path("experiments")
EXP_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = C.PREPROCESSED_CSV
DICT_PATH = C.DICTIONARY_PATH


def main():
    """Evaluasi komprehensif: stabilitas multi-seed, sweep K, perplexity,
    kualitas topik, dan distribusi proporsi per tahun."""
    DATA_PATH = C.PREPROCESSED_CSV
    DICT_PATH = C.DICTIONARY_PATH
    df = pd.read_csv(DATA_PATH)
    dictionary = Dictionary.load(str(DICT_PATH))

    texts = df['tokens'].apply(ast.literal_eval).tolist()
    corpus = [dictionary.doc2bow(t) for t in texts]

    print(f"Loaded {len(texts)} documents and {len(dictionary)} vocabulary terms.", flush=True)

    # ---------------------------------------------------------
    # TUGAS A: STABILITAS MULTI-SEED (K=8, N=10 SEED)
    # ---------------------------------------------------------
    print("\n=== TUGAS A: RUNNING MULTI-SEED STABILITY STUDY (N=10) ===", flush=True)
    N_SEEDS = 10
    seeds = [FINAL_SEED + i * 100 for i in range(N_SEEDS)]

    models = {}
    cv_scores = []
    umass_scores = []

    for s in seeds:
        lda = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=FINAL_K,
            alpha=FINAL_ALPHA,
            eta=FINAL_ETA,
            passes=FINAL_PASSES,
            iterations=FINAL_ITERATIONS,
            random_state=s
        )
        models[s] = lda

        cm_cv = CoherenceModel(model=lda, texts=texts, dictionary=dictionary, coherence='c_v', processes=1)
        cm_umass = CoherenceModel(model=lda, corpus=corpus, dictionary=dictionary, coherence='u_mass', processes=1)

        cv_val = cm_cv.get_coherence()
        umass_val = cm_umass.get_coherence()

        cv_scores.append(cv_val)
        umass_scores.append(umass_val)
        print(f"Seed {s:4d} -> C_V: {cv_val:.4f}, UMass: {umass_val:.4f}", flush=True)

    cv_arr = np.array(cv_scores)
    umass_arr = np.array(umass_scores)

    print(f"\n[Multi-Seed C_V Results] Mean: {cv_arr.mean():.4f}, SD: {cv_arr.std():.4f}, Min: {cv_arr.min():.4f}, Max: {cv_arr.max():.4f}", flush=True)
    print(f"[Multi-Seed UMass Results] Mean: {umass_arr.mean():.4f}, SD: {umass_arr.std():.4f}, Min: {umass_arr.min():.4f}, Max: {umass_arr.max():.4f}", flush=True)

    # Hungarian Matching & Jaccard Stability vs Seed 42
    base_lda = models[FINAL_SEED]

    def get_top_words_set(lda_m, topn=15):
        top_dict = {}
        for tid in range(lda_m.num_topics):
            words = [w for w, _ in lda_m.show_topic(tid, topn=topn)]
            top_dict[tid] = set(words)
        return top_dict

    base_top_words = get_top_words_set(base_lda, topn=15)

    jaccard_per_topic = {t: [] for t in range(FINAL_K)}
    matched_assignments = []

    base_doc_topics = [max(base_lda.get_document_topics(c), key=lambda x: x[1])[0] for c in corpus]

    for s in seeds:
        m = models[s]
        m_words = get_top_words_set(m, topn=15)

        cost_matrix = np.zeros((FINAL_K, FINAL_K))
        for i in range(FINAL_K):
            for j in range(FINAL_K):
                set1 = base_top_words[i]
                set2 = m_words[j]
                jacc = len(set1 & set2) / len(set1 | set2)
                cost_matrix[i, j] = 1.0 - jacc

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        s_to_base = {col: row for row, col in zip(row_ind, col_ind)}

        for row, col in zip(row_ind, col_ind):
            jaccard_per_topic[row].append(1.0 - cost_matrix[row, col])

        s_doc_topics = [max(m.get_document_topics(c), key=lambda x: x[1])[0] for c in corpus]
        s_doc_mapped = [s_to_base.get(t, t) for t in s_doc_topics]

        acc = np.mean(np.array(base_doc_topics) == np.array(s_doc_mapped))
        ari = adjusted_rand_score(base_doc_topics, s_doc_topics)
        matched_assignments.append({'seed': s, 'accuracy': acc, 'ari': ari})

    avg_jaccard_per_topic = {t: np.mean(jaccard_per_topic[t]) for t in range(FINAL_K)}
    stable_topics_count = sum(1 for v in avg_jaccard_per_topic.values() if v >= 0.5)

    print(f"\n--- Topic Jaccard Stability vs Seed {FINAL_SEED} ---", flush=True)
    for t, avg_j in avg_jaccard_per_topic.items():
        print(f"Topic {t+1}: Average Jaccard = {avg_j:.4f} (Stable: {avg_j >= 0.5})", flush=True)
    print(f"Total Stable Topics (Jaccard >= 0.5): {stable_topics_count} / {FINAL_K}", flush=True)

    avg_acc = np.mean([x['accuracy'] for x in matched_assignments])
    avg_ari = np.mean([x['ari'] for x in matched_assignments])
    print(f"Average Document Topic Assignment Match Rate: {avg_acc*100:.2f}%", flush=True)
    print(f"Average Adjusted Rand Index (ARI): {avg_ari:.4f}", flush=True)

    # ---------------------------------------------------------
    # TUGAS B: SWEEP K PADA SETTING FINAL (K=4..12, 3 SEED PER K)
    # ---------------------------------------------------------
    print("\n=== TUGAS B: SWEEP K (4 TO 12, 3 SEEDS PER K) ===", flush=True)
    k_range = range(4, 13)
    sweep_seeds = [42, 100, 200]

    sweep_results = []
    for k in k_range:
        k_cvs = []
        k_umass = []
        for s in sweep_seeds:
            lda = LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=k,
                alpha='asymmetric',
                eta=0.1,
                passes=30,
                iterations=300,
                random_state=s
            )
            cm_cv = CoherenceModel(model=lda, texts=texts, dictionary=dictionary, coherence='c_v', processes=1)
            cm_umass = CoherenceModel(model=lda, corpus=corpus, dictionary=dictionary, coherence='u_mass', processes=1)
            k_cvs.append(cm_cv.get_coherence())
            k_umass.append(cm_umass.get_coherence())

        mean_cv = np.mean(k_cvs)
        std_cv = np.std(k_cvs)
        mean_umass = np.mean(k_umass)
        std_umass = np.std(k_umass)

        sweep_results.append({
            'k': k,
            'cv_mean': mean_cv,
            'cv_std': std_cv,
            'umass_mean': mean_umass,
            'umass_std': std_umass
        })
        print(f"K={k:2d} -> C_V: {mean_cv:.4f} +/- {std_cv:.4f} | UMass: {mean_umass:.4f} +/- {std_umass:.4f}", flush=True)

    sweep_df = pd.DataFrame(sweep_results)
    sweep_df.to_csv(EXP_DIR / "k_sweep.csv", index=False)

    plt.figure(figsize=(9, 5))
    plt.errorbar(sweep_df['k'], sweep_df['cv_mean'], yerr=sweep_df['cv_std'], fmt='-o', capsize=5, color='#1f77b4', ecolor='#a6cee3', elinewidth=2, capthick=2)
    plt.title("LDA Topic Coherence (C_V) Across K (Mean +/- SD)", fontsize=12, fontweight='bold')
    plt.xlabel("Jumlah Topik (K)", fontsize=11)
    plt.ylabel("Coherence Score (C_V)", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(k_range)
    plt.tight_layout()
    plt.savefig(EXP_DIR / "k_sweep.png", dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # TUGAS C: PERPLEXITY TRAIN VS HOLD-OUT (80/20 STRATIFIED)
    # ---------------------------------------------------------
    print("\n=== TUGAS C: PERPLEXITY TRAIN VS HOLD-OUT ===", flush=True)
    train_texts, test_texts, train_df, test_df = train_test_split(
        texts, df, test_size=0.2, random_state=42, stratify=df['Tahun']
    )

    train_corpus = [dictionary.doc2bow(t) for t in train_texts]
    test_corpus = [dictionary.doc2bow(t) for t in test_texts]

    lda_eval = LdaModel(
        corpus=train_corpus,
        id2word=dictionary,
        num_topics=FINAL_K,
        alpha=FINAL_ALPHA,
        eta=FINAL_ETA,
        passes=FINAL_PASSES,
        iterations=FINAL_ITERATIONS,
        random_state=FINAL_SEED
    )

    train_perp = lda_eval.log_perplexity(train_corpus)
    test_perp = lda_eval.log_perplexity(test_corpus)

    print(f"Train Log Perplexity    : {train_perp:.4f} (Per-word bound)", flush=True)
    print(f"Hold-out Log Perplexity : {test_perp:.4f} (Per-word bound)", flush=True)
    print(f"Perplexity Gap (Test - Train): {test_perp - train_perp:.4f}", flush=True)

    # ---------------------------------------------------------
    # TUGAS D: TOPIC QUALITY & DIVERSITY (MODEL FINAL K=8, SEED 42)
    # ---------------------------------------------------------
    print("\n=== TUGAS D: TOPIC QUALITY & DIVERSITY (K=8, SEED 42) ===", flush=True)
    final_lda = base_lda

    top10_words = [w for tid in range(FINAL_K) for w, _ in final_lda.show_topic(tid, topn=10)]
    top15_words = [w for tid in range(FINAL_K) for w, _ in final_lda.show_topic(tid, topn=15)]

    td_10 = len(set(top10_words)) / len(top10_words)
    td_15 = len(set(top15_words)) / len(top15_words)

    print(f"Topic Diversity (Top-10): {td_10:.4f}", flush=True)
    print(f"Topic Diversity (Top-15): {td_15:.4f}", flush=True)

    cv_per_topic = []
    for tid in range(FINAL_K):
        words = [w for w, _ in final_lda.show_topic(tid, topn=10)]
        cm_t = CoherenceModel(topics=[words], texts=texts, dictionary=dictionary, coherence='c_v', processes=1)
        cv_per_topic.append(cm_t.get_coherence())

    print("\n--- Coherence (C_V) Per Topic ---", flush=True)
    for tid, cv_t in enumerate(cv_per_topic):
        print(f"Topic {tid+1}: {cv_t:.4f}", flush=True)

    word_counts = {}
    for w in top15_words:
        word_counts[w] = word_counts.get(w, 0) + 1

    frequent_words = [w for w, c in word_counts.items() if c >= 3]
    print(f"\nDomain-Specific Candidate Stopwords (appears in >=3 topics top-15): {frequent_words}", flush=True)

    final_lda.save(str(EXP_DIR / "candidate_lda_model.gensim"))

    top_words_rows = []
    for tid in range(FINAL_K):
        for rank, (w, prob) in enumerate(final_lda.show_topic(tid, topn=15), 1):
            top_words_rows.append({
                'topic_id': tid + 1,
                'rank': rank,
                'word': w,
                'probability': prob
            })
    pd.DataFrame(top_words_rows).to_csv(EXP_DIR / "top_words_by_topic.csv", index=False)

    # ---------------------------------------------------------
    # TUGAS E: THETA DISTRIBUTION & PROPORSI TOPIC PER TAHUN
    # ---------------------------------------------------------
    print("\n=== TUGAS E: DOCUMENT ASSIGNMENT & PROPORTIONS PER YEAR ===", flush=True)
    theta_matrix = np.zeros((len(corpus), FINAL_K))
    for i, c in enumerate(corpus):
        for tid, prob in final_lda.get_document_topics(c, minimum_probability=0.0):
            theta_matrix[i, tid] = prob

    max_theta = theta_matrix.max(axis=1)

    pct_40 = np.mean(max_theta < 0.4) * 100
    pct_50 = np.mean(max_theta < 0.5) * 100
    pct_60 = np.mean(max_theta < 0.6) * 100

    print(f"Max Theta < 0.4: {pct_40:.2f}% documents", flush=True)
    print(f"Max Theta < 0.5: {pct_50:.2f}% documents", flush=True)
    print(f"Max Theta < 0.6: {pct_60:.2f}% documents", flush=True)

    df['topik_dominan'] = theta_matrix.argmax(axis=1) + 1

    argmax_prop = df.groupby(['Tahun', 'topik_dominan']).size().unstack(fill_value=0)
    argmax_prop = argmax_prop.div(argmax_prop.sum(axis=1), axis=0) * 100
    argmax_prop.to_csv(EXP_DIR / "proporsi_argmax.csv")

    df_theta = pd.DataFrame(theta_matrix, columns=[f"topik_{i+1}" for i in range(FINAL_K)])
    df_theta['Tahun'] = df['Tahun']
    soft_prop = df_theta.groupby('Tahun').mean() * 100
    soft_prop.to_csv(EXP_DIR / "proporsi_soft.csv")

    diff_matrix = (argmax_prop - soft_prop.values).abs()
    max_diff = diff_matrix.max().max()

    print(f"\nMax difference between Argmax and Soft Proportion: {max_diff:.2f}% points", flush=True)
    print("Argmax Proportions Saved to proporsi_argmax.csv", flush=True)
    print("Soft Proportions Saved to proporsi_soft.csv", flush=True)

    print("\n=== ALL CALCULATIONS COMPLETED SUCCESSFULLY! ===", flush=True)



if __name__ == "__main__":
    main()
