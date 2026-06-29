import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from gensim.models import LdaModel
from wordcloud import WordCloud
import ast

def generate_evaluation_plots():
    print("="*60)
    print("MEMBUAT PLOT EVALUASI LDA")
    print("="*60)
    
    results_path = Path("model/hyperparameter_results.csv")
    if not results_path.exists():
        print(f"File {results_path} tidak ditemukan. Silakan jalankan hyperparameter_tuning.py terlebih dahulu.")
        return
        
    df = pd.read_csv(results_path)
    
    docs_dir = Path("docs/evaluation_plots")
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Elbow Curve (K vs Coherence C_V)
    # We take the max C_V for each K across all alpha/eta combinations
    max_cv_per_k = df.groupby('k')['coherence_cv'].max().reset_index()
    optimal_k = max_cv_per_k.loc[max_cv_per_k['coherence_cv'].idxmax()]['k']
    
    plt.figure(figsize=(10, 6))
    plt.plot(max_cv_per_k['k'], max_cv_per_k['coherence_cv'], marker='o', linestyle='-', linewidth=2, markersize=8)
    plt.axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal K = {int(optimal_k)}')
    plt.title('Elbow Curve: Jumlah Topik (K) vs Coherence (C_V)', fontsize=14)
    plt.xlabel('Jumlah Topik (K)', fontsize=12)
    plt.ylabel('Coherence Score (C_V)', fontsize=12)
    plt.xticks(max_cv_per_k['k'])
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(docs_dir / 'elbow_curve_cv.png', dpi=300)
    plt.close()
    print(f"Berhasil menyimpan: {docs_dir / 'elbow_curve_cv.png'}")
    
    # 2. Dual Axis Plot (C_V vs U_Mass)
    max_umass_per_k = df.groupby('k')['coherence_umass'].max().reset_index()
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Jumlah Topik (K)', fontsize=12)
    ax1.set_ylabel('Coherence C_V', color=color, fontsize=12)
    ax1.plot(max_cv_per_k['k'], max_cv_per_k['coherence_cv'], marker='o', color=color, linewidth=2, label='C_V (Max)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(max_cv_per_k['k'])
    
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Coherence U_Mass', color=color, fontsize=12)
    ax2.plot(max_umass_per_k['k'], max_umass_per_k['coherence_umass'], marker='s', color=color, linewidth=2, label='U_Mass (Max)')
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()  
    plt.title('Perbandingan Coherence: C_V vs U_Mass per K', fontsize=14)
    
    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center')
    
    plt.savefig(docs_dir / 'coherence_comparison.png', dpi=300)
    plt.close()
    print(f"Berhasil menyimpan: {docs_dir / 'coherence_comparison.png'}")
    
    # 3. Heatmap Alpha vs Eta for Optimal K
    df_opt = df[df['k'] == optimal_k]
    
    if not df_opt.empty and len(df_opt) > 1:
        # Convert objects to string for pivot
        df_opt = df_opt.copy()
        df_opt['alpha_str'] = df_opt['alpha'].astype(str)
        df_opt['eta_str'] = df_opt['eta'].astype(str)
        
        pivot_table = df_opt.pivot(index='alpha_str', columns='eta_str', values='coherence_cv')
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt='.4f', linewidths=.5)
        plt.title(f'Heatmap Coherence C_V (K={int(optimal_k)})', fontsize=14)
        plt.xlabel('Eta / Beta')
        plt.ylabel('Alpha')
        plt.tight_layout()
        plt.savefig(docs_dir / f'heatmap_hyperparameters_K{int(optimal_k)}.png', dpi=300)
        plt.close()
        print(f"Berhasil menyimpan: {docs_dir / f'heatmap_hyperparameters_K{int(optimal_k)}.png'}")

    # 4. Word Clouds
    model_path = Path("model/lda_model.gensim")
    if model_path.exists():
        print(f"\nMembuat Word Cloud dari model {model_path}...")
        try:
            model = LdaModel.load(str(model_path))
            wc_dir = docs_dir / "wordclouds"
            wc_dir.mkdir(parents=True, exist_ok=True)
            
            for topic_id in range(model.num_topics):
                top_words = dict(model.show_topic(topic_id, topn=20))
                
                # Multiply by 1000 so sizes are readable by wordcloud
                freq_dict = {w: float(v * 1000) for w, v in top_words.items()}
                
                wc = WordCloud(
                    width=800, height=400, 
                    background_color='white', 
                    colormap='viridis',
                    max_words=20
                ).generate_from_frequencies(freq_dict)
                
                plt.figure(figsize=(10, 5))
                plt.imshow(wc, interpolation='bilinear')
                plt.axis('off')
                plt.title(f'Word Cloud Topik {topic_id+1}', fontsize=16)
                plt.tight_layout(pad=0)
                plt.savefig(wc_dir / f'wordcloud_topic_{topic_id+1}.png', dpi=300)
                plt.close()
                
            print(f"Berhasil menyimpan word clouds ke {wc_dir}")
            
        except Exception as e:
            print(f"Gagal meload/membuat word cloud: {e}")
    else:
        print(f"Model LDA tidak ditemukan di {model_path} untuk membuat word cloud.")
        
    print("\n" + "="*60)
    print("EVALUASI SELESAI")
    print("="*60)


if __name__ == "__main__":
    generate_evaluation_plots()
