import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
from pathlib import Path

# ── Color Palette (Same as main) ──
PRIMARY = "#1E3A5F"
SECONDARY = "#2E86AB"
ACCENT = "#F39C12"
BG_LIGHT = "#F4F6F9"
TEXT = "#1A1A2E"
MUTED = "#7F8C8D"
SUCCESS = "#27AE60"
DANGER = "#E74C3C"
CARD_BG = "#FFFFFF"

st.set_page_config(page_title="Evaluasi Model", page_icon="📈", layout="wide")

st.markdown(f"""
    <style>
    .card {{
        background: {CARD_BG};
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04), 0 2px 12px rgba(0,0,0,0.04);
        border: 1px solid rgba(0,0,0,0.04);
    }}
    .section-header {{
        border-bottom: 3px solid {ACCENT};
        padding-bottom: 0.5rem;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}
    .section-header h3 {{
        margin: 0;
        font-size: 1.2rem;
        font-weight: 600;
        color: {TEXT};
    }}
    </style>
""", unsafe_allow_html=True)

def card_start():
    st.markdown('<div class="card">', unsafe_allow_html=True)
def card_end():
    st.markdown('</div>', unsafe_allow_html=True)
def section_header(title: str):
    st.markdown(f'<div class="section-header"><h3>{title}</h3></div>', unsafe_allow_html=True)

st.title("📈 Evaluasi Model LDA")
st.markdown("Halaman ini menyajikan metrik performa model dan hasil hyperparameter tuning untuk pemilihan jumlah topik (K) yang optimal.")

base_path = Path(__file__).parent.parent.parent
results_path = base_path / "model" / "hyperparameter_results.csv"
metrics_path = base_path / "model" / "evaluation_metrics.csv"
viz_path = base_path / "model" / "lda_visualization.html"

# 1. Load Tuning Results
card_start()
section_header("📊 Hasil Hyperparameter Tuning")
if results_path.exists():
    df = pd.read_csv(results_path)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("Top 10 Konfigurasi Terbaik (Berdasarkan C_V):")
        top_10 = df.sort_values(by='coherence_cv', ascending=False).head(10)
        st.dataframe(top_10[['k', 'alpha', 'eta', 'coherence_cv', 'coherence_umass']], hide_index=True)
        
    with col2:
        st.write("Elbow Curve (Jumlah Topik vs Coherence C_V):")
        max_cv_per_k = df.groupby('k')['coherence_cv'].max().reset_index()
        optimal_k = max_cv_per_k.loc[max_cv_per_k['coherence_cv'].idxmax()]['k']
        
        fig = px.line(max_cv_per_k, x='k', y='coherence_cv', markers=True, 
                      title="Elbow Curve", labels={'k': 'Jumlah Topik (K)', 'coherence_cv': 'Max Coherence (C_V)'})
        fig.add_vline(x=optimal_k, line_dash="dash", line_color="red", annotation_text=f"Optimal K={optimal_k}")
        fig.update_layout(height=400, margin=dict(l=20,r=20,t=40,b=20))
        st.plotly_chart(fig, use_container_width=True)
        
else:
    st.warning("Data hyperparameter_results.csv tidak ditemukan. Tampilkan metrik evaluasi dasar saja.")
    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)
        st.dataframe(metrics_df, hide_index=True)
    else:
        st.error("Tidak ada data metrik model yang ditemukan.")

with st.expander("📖 Penjelasan Metrik"):
    st.markdown("""
    * **Coherence Score (C_V):** Mengukur seberapa sering kata-kata teratas dalam topik muncul bersamaan dalam dokumen aslinya. Nilainya berkisar 0 hingga 1. Semakin mendekati 1, semakin baik dan mudah diinterpretasi topiknya.
    * **Coherence (U_Mass):** Mengukur kualitas topik secara intrinsik (menggunakan dokumen dari corpus latih). Nilainya selalu negatif, dan semakin mendekati 0 semakin baik.
    * **Log Perplexity:** Mengukur kebingungan model (seberapa baik model memprediksi data baru). Nilai yang lebih kecil (lebih negatif) menunjukkan performa prediksi yang lebih baik.
    * **Elbow Curve:** Grafik yang digunakan untuk melihat titik di mana penambahan jumlah topik tidak lagi memberikan peningkatan *coherence* yang signifikan (biasanya titik "siku" / puncaknya adalah K optimal).
    """)
card_end()

# 2. PyLDAvis
card_start()
section_header("📍 Visualisasi Interaktif Model (PyLDAvis)")
if viz_path.exists():
    with open(viz_path, 'r', encoding='utf-8') as f:
        html_string = f.read()
    st.caption("Klik pada gelembung topik di sebelah kiri untuk melihat persebaran kata kuncinya di sebelah kanan.")
    components.html(html_string, height=800, scrolling=True)
else:
    st.warning("File lda_visualization.html tidak ditemukan. Pastikan model telah dilatih dengan benar.")
card_end()

# 3. Grid Search per K
st.markdown("---")
card_start()
section_header("📊 Hasil Grid Search Hyperparameter")

if results_path.exists():
    hp_df = pd.read_csv(results_path)

    best_per_k = hp_df.loc[hp_df.groupby('k')['coherence_cv'].idxmax()].reset_index(drop=True)
    display_cols = best_per_k[['k', 'coherence_cv', 'coherence_umass', 'log_perplexity', 'alpha', 'eta']].copy()
    display_cols.columns = ['K Topik', 'Coherence CV', 'Coherence UMass', 'Log Perplexity', 'Alpha', 'Eta']
    display_cols['Coherence CV'] = display_cols['Coherence CV'].round(4)
    display_cols['Log Perplexity'] = display_cols['Log Perplexity'].round(2)

    def highlight_k7(row):
        return ['background-color: #fff3cd' if row['K Topik'] == 7 else '' for _ in row]

    st.markdown("Tabel di bawah menunjukkan konfigurasi **terbaik per jumlah topik (K)** berdasarkan metrik Coherence C_V:")
    st.dataframe(
        display_cols.style.apply(highlight_k7, axis=1),
        use_container_width=True,
        hide_index=True
    )

    with st.expander("📖 Mengapa K=7 dipilih meskipun bukan yang tertinggi secara metrik?"):
        st.markdown("""
        **Coherence score tertinggi tidak selalu berarti model terbaik** untuk analisis domain.

        K=7 dipilih berdasarkan pertimbangan:
        1. **Interpretabilitas** — K kecil (K=3) menghasilkan topik terlalu luas dan tidak informatif  
        2. **Relevansi domain** — Terdapat 6-8 area penelitian SI yang berbeda secara konseptual  
        3. **Konsistensi literatur** — Penelitian LDA pada dokumen akademik umumnya K=5-10  
        4. **Selisih tidak signifikan** — Perbedaan coherence K=3 dan K=7 (0.036) masih dalam margin variabilitas normal  
        """)
else:
    st.warning("File hyperparameter_results.csv tidak ditemukan. Jalankan hyperparameter_tuning.py terlebih dahulu.")
card_end()
