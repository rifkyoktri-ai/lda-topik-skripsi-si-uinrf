import argparse
import pandas as pd
import time
from pathlib import Path
from tqdm import tqdm
from . import config as C
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
from itertools import product
import logging
import ast

# Disable gensim INFO logs
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARNING)

def preprocess_text_for_coherence(df, text_col='abstract'):
    """
    Ensure we have a list of list of strings for CoherenceModel(texts=...).
    """
    texts = []
    # If the dataset already has a preprocessed column (list of words as string), parse it
    if 'tokens' in df.columns:
        text_col = 'tokens'
    elif 'preprocessed' in df.columns:
        text_col = 'preprocessed'
    
    for text in df[text_col].fillna(''):
        # If it looks like a lisjt string, e.g., "['sistem', 'informasi']", evaluate it
        if isinstance(text, str) and text.startswith('[') and text.endswith(']'):
            try:
                words = ast.literal_eval(text)
                texts.append(words)
                continue
            except:
                pass
        
        # Fallback to simple split if it's just text
        words = text.lower().split()
        texts.append(words)
        
    return texts

def run_grid_search(args):
    print("="*60)
    print("LDA HYPERPARAMETER TUNING (Grid Search)")
    print("="*60)
    
    # 1. Load Data
    dict_path = Path(str(C.DICTIONARY_PATH))
    if not dict_path.exists():
        dict_path = C.LDA_ID2WORD
    if not dict_path.exists():
        raise FileNotFoundError(f"Dictionary tidak ditemukan di data/intermediate/dictionary.gensim maupun model/lda_model.gensim.id2word")
    
    print(f"Loading dictionary dari {dict_path}...")
    dictionary = Dictionary.load(str(dict_path))
    
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {data_path}")
        
    print(f"Loading dataset dari {data_path}...")
    df = pd.read_csv(data_path)
    
    texts = preprocess_text_for_coherence(df)
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    print(f"Total dokumen: {len(corpus)}")
    print(f"Ukuran vocabulary: {len(dictionary)}")
    
    # 2. Setup Grid Search
    k_range = list(range(args.min_k, args.max_k + 1))
    alpha_options = list(C.GRID_ALPHA_OPTIONS)
    eta_options = list(C.GRID_ETA_OPTIONS)
    
    grid = list(product(k_range, alpha_options, eta_options))
    total_combinations = len(grid)
    
    # Estimate time (assuming ~15 seconds per model for CPU)
    est_time_per_model = 15
    total_est_seconds = total_combinations * est_time_per_model
    mins, secs = divmod(total_est_seconds, 60)
    hours, mins = divmod(mins, 60)
    
    print(f"\nTotal kombinasi parameter: {total_combinations}")
    print(f"Estimasi waktu eksekusi: {int(hours)} jam {int(mins)} menit {int(secs)} detik")
    print(f"K = {args.min_k} hingga {args.max_k}")
    print(f"Alpha = {alpha_options}")
    print(f"Eta = {eta_options}")
    print("-"*60)
    
    results = []
    
    # 3. Execute Grid Search
    with tqdm(total=total_combinations, desc="Tuning", unit="model") as pbar:
        for k, alpha, eta in grid:
            start_time = time.time()
            
            # Train model
            # Parameters optimized for CPU
            model = LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=k,
                alpha=alpha,
                eta=eta,
                passes=args.passes,
                iterations=400,
                minimum_probability=0.01,
                random_state=42
            )
            
            train_time = time.time() - start_time
            
            # Calculate metrics
            try:
                # c_v requires texts
                cm_cv = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
                cv_score = cm_cv.get_coherence()
                
                # u_mass uses corpus
                cm_umass = CoherenceModel(model=model, corpus=corpus, dictionary=dictionary, coherence='u_mass')
                umass_score = cm_umass.get_coherence()
                
                # Perplexity
                perplexity = model.log_perplexity(corpus)
                
            except Exception as e:
                print(f"\nError calculating metrics for K={k}, alpha={alpha}, eta={eta}: {e}")
                cv_score = 0
                umass_score = 0
                perplexity = 0
            
            results.append({
                'k': k,
                'alpha': alpha,
                'eta': eta,
                'coherence_cv': cv_score,
                'coherence_umass': umass_score,
                'log_perplexity': perplexity,
                'training_time_seconds': train_time
            })
            
            pbar.update(1)

    # 4. Save Results
    results_df = pd.DataFrame(results)
    out_path = C.HYPERPARAMS_CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)
    
    print("\n" + "="*60)
    print(f"Tuning selesai. Hasil disimpan di: {out_path}")
    print("="*60)
    
    # 5. Top 5 configs
    top_5 = results_df.sort_values(by='coherence_cv', ascending=False).head(5)
    print("TOP 5 KONFIGURASI BERDASARKAN COHERENCE (C_V):")
    print(top_5[['k', 'alpha', 'eta', 'coherence_cv', 'coherence_umass']].to_string(index=False))
    
    # Conclusion text
    best_k = int(top_5.iloc[0]['k'])
    final_k = C.NUM_TOPICS
    print("\nCATATAN METODOLOGIS:")
    print(f"K optimal menurut C_V = {best_k}; model final yang dipakai skripsi = K={final_k}")
    print("(sumber: src/config.py -> NUM_TOPICS).")
    if best_k != final_k:
        print("Keduanya berbeda. Pilihan untuk sidang skripsi:")
        print(f"1. Pertahankan K={final_k} dan jelaskan alasannya (jumlah dokumen per topik,")
        print("   interpretabilitas) - inilah yang skripsi ini lakukan.")
        print(f"2. Retrain ke K={best_k} bila penguji menuntut K optimum C_V.")
    else:
        print("K keduanya sama - tidak ada konflik.")
    print("="*60)


def build_parser():
    """Argumen CLI untuk grid search."""
    parser = argparse.ArgumentParser(description="LDA Hyperparameter Tuning")
    parser.add_argument("--min_k", type=int, default=C.GRID_MIN_K,
                        help="Minimum jumlah topik")
    parser.add_argument("--max_k", type=int, default=C.GRID_MAX_K,
                        help="Maximum jumlah topik")
    parser.add_argument("--passes", type=int, default=C.GRID_PASSES,
                        help="Jumlah passes training LDA")
    parser.add_argument("--workers", type=int, default=1,
                        help="Jumlah workers (CPU)")
    parser.add_argument("--data_path", type=str, default=str(C.PREPROCESSED_CSV),
                        help="Path ke dataset")
    return parser


def main():
    """Entry point untuk dipanggil dari notebook maupun terminal."""
    run_grid_search(build_parser().parse_args())


if __name__ == "__main__":
    main()
