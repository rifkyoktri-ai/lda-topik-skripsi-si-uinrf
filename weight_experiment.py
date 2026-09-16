import os
import re
import sys
import pickle
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from indonesian_stopwords import get_conservative_stopwords

# Configure logging
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

DOMAIN_STOPWORDS_MINIMAL = {
    'skripsi', 'mahasiswa', 'universitas', 'fakultas',
    'program', 'studi', 'jurusan', 'uin', 'raden', 'fatah',
    'palembang', 'negeri', 'islam'
}

def bersihkan(teks):
    if not isinstance(teks, str):
        return ''
    teks = teks.lower()
    teks = re.sub(r'[^a-zA-Z\s]', ' ', teks)
    teks = re.sub(r'\s+', ' ', teks).strip()
    return teks

def main():
    print("=" * 65)
    print("EKSPERIMEN PEMBOBOTAN JUDUL VS ABSTRAK (TITLE WEIGHTING EXPERIMENT)")
    print("=" * 65)
    
    data_path = Path('data/processed/thesis_for_pipeline.csv')
    if not data_path.exists():
        raise FileNotFoundError(f"File dataset tidak ditemukan: {data_path}")
        
    print(f"1. Memuat data dari {data_path}...")
    df = pd.read_csv(data_path)
    
    print("2. Inisialisasi Sastrawi Stemmer & Stopwords...")
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    all_stopwords = get_conservative_stopwords() | DOMAIN_STOPWORDS_MINIMAL
    
    def process_text_to_tokens(text):
        cleaned = bersihkan(text)
        tokens = [w for w in cleaned.split() if w not in all_stopwords and len(w) > 2]
        stemmed = [stemmer.stem(t) for t in tokens]
        return stemmed

    print("3. Preprocessing Judul dan Abstrak secara terpisah...")
    processed_docs = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
        title_raw = str(row['title']) if pd.notna(row['title']) else ''
        abstract_raw = str(row['abstract']) if pd.notna(row['abstract']) else ''
        
        t_tokens = process_text_to_tokens(title_raw)
        a_tokens = process_text_to_tokens(abstract_raw)
        
        processed_docs.append({
            'id': row['id'],
            'title_tokens': t_tokens,
            'abstract_tokens': a_tokens
        })

    weights = [1, 2, 3, 5]
    k_list = [5, 7, 10]
    
    results = []
    
    print("\n4. Menjalankan training & evaluasi untuk setiap skema bobot...")
    for weight in weights:
        print(f"\n--- Menguji Skema Bobot Judul: {weight}x (Title:Abstract = {weight}:1) ---")
        
        tokenized_corpus = []
        for doc in processed_docs:
            combined = (doc['title_tokens'] * weight) + doc['abstract_tokens']
            if combined:
                tokenized_corpus.append(combined)
                
        # Build Gensim Dictionary & Corpus
        dictionary = corpora.Dictionary(tokenized_corpus)
        dictionary.filter_extremes(no_below=2, no_above=0.80)
        corpus = [dictionary.doc2bow(doc) for doc in tokenized_corpus]
        
        vocab_size = len(dictionary)
        total_docs = len(corpus)
        print(f"    Vocabulary Size: {vocab_size} kata | Total Dokumen: {total_docs}")
        
        for k in k_list:
            model = LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=k,
                alpha='auto',
                eta='auto',
                passes=20,
                iterations=400,
                random_state=42,
                per_word_topics=True
            )
            
            # Compute Coherence C_V
            cm_cv = CoherenceModel(
                model=model,
                texts=tokenized_corpus,
                dictionary=dictionary,
                coherence='c_v',
                processes=1
            )
            cv_score = cm_cv.get_coherence()
            
            # Compute Coherence U_Mass
            cm_umass = CoherenceModel(
                model=model,
                corpus=corpus,
                dictionary=dictionary,
                coherence='u_mass',
                processes=1
            )
            umass_score = cm_umass.get_coherence()
            
            # Compute Perplexity
            perplexity = model.log_perplexity(corpus)
            
            print(f"    K={k:2d} | Coherence C_V: {cv_score:.4f} | U_Mass: {umass_score:.4f} | Log Perplexity: {perplexity:.4f}")
            
            results.append({
                'bobot_judul': f"{weight}x",
                'weight_multiplier': weight,
                'k': k,
                'coherence_cv': cv_score,
                'coherence_umass': umass_score,
                'log_perplexity': perplexity,
                'vocab_size': vocab_size,
                'total_docs': total_docs
            })

    results_df = pd.DataFrame(results)
    
    # Save CSV results
    out_dir = Path("model")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "weight_scheme_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n5. Hasil eksperimen berhasil disimpan ke: {csv_path}")
    
    # Plotting results
    plot_dir = Path("docs/evaluation_plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / "weight_scheme_comparison.png"
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot Coherence C_V
    for k in k_list:
        sub_df = results_df[results_df['k'] == k]
        axes[0].plot(sub_df['weight_multiplier'], sub_df['coherence_cv'], marker='o', linewidth=2, label=f'K={k}')
        
    axes[0].set_title('Pengaruh Bobot Judul terhadap Coherence Score (C_V)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Pengali Bobot Judul (x)', fontsize=10)
    axes[0].set_ylabel('Coherence C_V (Makin tinggi makin baik)', fontsize=10)
    axes[0].set_xticks(weights)
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend()
    
    # Plot Log Perplexity
    for k in k_list:
        sub_df = results_df[results_df['k'] == k]
        axes[1].plot(sub_df['weight_multiplier'], sub_df['log_perplexity'], marker='s', linewidth=2, label=f'K={k}')
        
    axes[1].set_title('Pengaruh Bobot Judul terhadap Log Perplexity', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Pengali Bobot Judul (x)', fontsize=10)
    axes[1].set_ylabel('Log Perplexity (Makin mendekati 0 / kurang negatif makin baik)', fontsize=10)
    axes[1].set_xticks(weights)
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"6. Grafik perbandingan disimpan ke: {plot_path}")
    
    print("\n" + "=" * 65)
    print("RINGKASAN HASIL EKSPERIMEN PEMBOBOTAN JUDUL (K=7 Utama):")
    print("=" * 65)
    df_k7 = results_df[results_df['k'] == 7]
    print(df_k7[['bobot_judul', 'k', 'coherence_cv', 'coherence_umass', 'log_perplexity']].to_string(index=False))
    print("=" * 65)
    
    best_weight_k7 = df_k7.loc[df_k7['coherence_cv'].idxmax()]
    print(f"\nKESIMPULAN METODOLOGIS:")
    print(f"Pada K=7 (model utama skripsi), skema bobot judul terbaik adalah '{best_weight_k7['bobot_judul']}'")
    print(f"dengan Coherence C_V = {best_weight_k7['coherence_cv']:.4f} dan Log Perplexity = {best_weight_k7['log_perplexity']:.4f}.")
    print("=" * 65)

if __name__ == '__main__':
    main()
