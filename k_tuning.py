"""
k_tuning.py — Micro K-Selection Tuning Script
Lead Data Scientist: Empirical verification of optimal K for LDA.

Iterates K = [3, 4, 5] with strict hyperparameters:
  passes=50, iterations=400, alpha='auto', eta='auto'

Outputs:
  - C_V and U_Mass coherence per K
  - Pairwise cosine similarity matrix (flags overlaps > 0.4)
  - Clean comparative summary table
"""

import os
import sys
import pickle
import ast
import logging
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from gensim import corpora
from gensim.models import LdaModel, CoherenceModel

# ---------------------------------------------------------------------------
# SETUP
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,          # suppress Gensim training spam
    format='%(asctime)s [%(levelname)s] %(message)s'
)

PROSES_DIR = 'data/intermediate'
RANDOM_STATE = 42
K_RANGE     = [3, 4, 5]
PASSES      = 50
ITERATIONS  = 400
ALPHA       = 'auto'
ETA         = 'auto'
SIM_THRESHOLD = 0.4             # flag overlapping topic pairs above this

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def cosine_sim(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity between two probability distributions."""
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    return float(np.dot(vec_a, vec_b) / denom) if denom > 0 else 0.0


def topic_vectors(model: LdaModel, num_terms: int) -> np.ndarray:
    """Return full P(w|z) distributions for all topics as a 2D array."""
    vecs = []
    for tid in range(model.num_topics):
        vec = np.zeros(num_terms)
        for wid, prob in model.get_topic_terms(tid, topn=num_terms):
            vec[wid] = prob
        vecs.append(vec)
    return np.array(vecs)


def pairwise_cosine(model: LdaModel) -> dict:
    """
    Compute all pairwise cosine similarities between topics.
    Returns dict with keys=(i,j) and values=similarity score.
    """
    vecs = topic_vectors(model, model.num_terms)
    pairs = {}
    for i, j in combinations(range(model.num_topics), 2):
        pairs[(i, j)] = cosine_sim(vecs[i], vecs[j])
    return pairs


def count_overlapping_pairs(pairs: dict, threshold: float) -> int:
    return sum(1 for s in pairs.values() if s > threshold)


def max_pairwise_sim(pairs: dict) -> float:
    return max(pairs.values()) if pairs else 0.0

# ---------------------------------------------------------------------------
# LOAD PREPROCESSED DATA
# ---------------------------------------------------------------------------
print("\n" + "="*65)
print("  K-SELECTION MICRO-TUNING  |  K in {3, 4, 5}")
print("="*65)
print(f"  passes={PASSES}  |  iterations={ITERATIONS}  |  alpha={ALPHA}  |  eta={ETA}")
print(f"  Cosine-overlap threshold : > {SIM_THRESHOLD}")
print("="*65)

print("\n[1/3] Loading preprocessed corpus...")
df = pd.read_csv(f'{PROSES_DIR}/dataset_preprocessed.csv')
df['tokens'] = df['tokens'].apply(ast.literal_eval)
tokenized_docs = df['tokens'].tolist()

dict_path   = f'{PROSES_DIR}/dictionary.gensim'
corpus_path = f'{PROSES_DIR}/corpus.pkl'

dictionary = corpora.Dictionary.load(dict_path)
with open(corpus_path, 'rb') as f:
    corpus = pickle.load(f)

print(f"  Documents : {len(df)}")
print(f"  Vocab     : {len(dictionary)} tokens")

# 80/20 hold-out split
corpus_train, corpus_test = train_test_split(
    corpus, test_size=0.20, random_state=RANDOM_STATE
)
print(f"  Train     : {len(corpus_train)}  |  Test: {len(corpus_test)}")

# ---------------------------------------------------------------------------
# TRAINING LOOP
# ---------------------------------------------------------------------------
print(f"\n[2/3] Training LDA for K = {K_RANGE} ...\n")

results = []

for k in K_RANGE:
    sys.stdout.write(f"  -> K={k} training ... ")
    sys.stdout.flush()

    model = LdaModel(
        corpus=corpus_train,
        id2word=dictionary,
        num_topics=k,
        alpha=ALPHA,
        eta=ETA,
        passes=PASSES,
        iterations=ITERATIONS,
        random_state=RANDOM_STATE,
        per_word_topics=True
    )

    # --- Coherence C_V ---
    cm_cv = CoherenceModel(
        model=model,
        texts=tokenized_docs,
        dictionary=dictionary,
        coherence='c_v',
        processes=1
    )
    cv = cm_cv.get_coherence()

    # --- Coherence U_Mass ---
    cm_um = CoherenceModel(
        model=model,
        corpus=corpus_train,
        dictionary=dictionary,
        coherence='u_mass',
        processes=1
    )
    umass = cm_um.get_coherence()

    # --- Perplexity ---
    train_perp = model.log_perplexity(corpus_train)
    test_perp  = model.log_perplexity(corpus_test)

    # --- Pairwise Cosine Similarity ---
    pairs    = pairwise_cosine(model)
    max_sim  = max_pairwise_sim(pairs)
    overlaps = count_overlapping_pairs(pairs, SIM_THRESHOLD)

    # Top-5 words per topic (for display)
    top_words = {
        tid: [w for w, _ in model.show_topic(tid, topn=5)]
        for tid in range(k)
    }

    results.append({
        'K'           : k,
        'C_V'         : cv,
        'U_Mass'      : umass,
        'Train Perp'  : train_perp,
        'Test Perp'   : test_perp,
        'Max CosSim'  : max_sim,
        'Overlaps>0.4': overlaps,
        '_model'      : model,
        '_pairs'      : pairs,
        '_top_words'  : top_words,
    })

    status = "[!] OVERLAP" if overlaps > 0 else "[OK]"
    print(f"C_V={cv:.4f}  U_Mass={umass:.4f}  MaxSim={max_sim:.3f}  Overlaps={overlaps} {status}")

# ---------------------------------------------------------------------------
# COMPARATIVE TABLE
# ---------------------------------------------------------------------------
print("\n" + "="*65)
print("  COMPARATIVE SUMMARY TABLE")
print("="*65)
header = f"{'K':>3}  {'C_V':>7}  {'U_Mass':>8}  {'TrainPerp':>10}  {'TestPerp':>9}  {'MaxSim':>7}  {'Overlaps>0.4':>13}  {'Verdict':>12}"
print(header)
print("-"*65)

best_cv   = max(r['C_V'] for r in results)
best_row  = None

for r in results:
    is_best = r['C_V'] == best_cv
    verdict = "[BEST C_V]" if is_best else ""
    if is_best:
        best_row = r

    overlap_flag = f"{r['Overlaps>0.4']} [!]" if r['Overlaps>0.4'] > 0 else f"{r['Overlaps>0.4']}  [ok]"
    print(
        f"{r['K']:>3}  "
        f"{r['C_V']:>7.4f}  "
        f"{r['U_Mass']:>8.4f}  "
        f"{r['Train Perp']:>10.4f}  "
        f"{r['Test Perp']:>9.4f}  "
        f"{r['Max CosSim']:>7.3f}  "
        f"{overlap_flag:>15}  "
        f"{verdict}"
    )

print("="*65)

# ---------------------------------------------------------------------------
# PER-TOPIC TOP WORDS FOR BEST K
# ---------------------------------------------------------------------------
if best_row:
    k_best = best_row['K']
    print(f"\n[3/3] Top-5 Words per Topic  (Best K={k_best}):")
    print("-"*65)
    for tid, words in best_row['_top_words'].items():
        print(f"  Topik {tid+1:>2}: {', '.join(words)}")

    print("\n  Pairwise Cosine Similarity Matrix:")
    print("  " + "-"*40)
    for (i, j), sim in best_row['_pairs'].items():
        flag = "  [!] OVERLAP > 0.4" if sim > SIM_THRESHOLD else ""
        print(f"  Topik {i+1} <-> Topik {j+1}: {sim:.4f}{flag}")
    print("="*65)

# ---------------------------------------------------------------------------
# SAVE RESULTS TO CSV
# ---------------------------------------------------------------------------
df_results = pd.DataFrame([
    {k: v for k, v in r.items() if not k.startswith('_')}
    for r in results
])
out_path = 'model/k_tuning_results.csv'
os.makedirs('model', exist_ok=True)
df_results.to_csv(out_path, index=False)
print(f"\n  Results saved → {out_path}\n")
