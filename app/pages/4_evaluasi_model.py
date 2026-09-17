import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
from pathlib import Path

# ── Color Palette (Enterprise Dark Mode) ──
PRIMARY = "#3B82F6"
SECONDARY = "#60A5FA"
ACCENT = "#F59E0B"
BG_DARK = "#0B0F19"
TEXT = "#F3F4F6"
MUTED = "#9CA3AF"
SUCCESS = "#10B981"
DANGER = "#EF4444"
CARD_BG = "#1F2937"
BORDER = "#374151"

st.set_page_config(page_title="Evaluasi Model", page_icon="📈", layout="wide")

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, .stApp {{
        background-color: {BG_DARK} !important;
        color: {TEXT} !important;
        font-family: 'Inter', -apple-system, sans-serif !important;
    }}

    .card {{
        background: {CARD_BG} !important;
        border-radius: 10px !important;
        padding: 1.5rem !important;
        margin-bottom: 1.5rem !important;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3) !important;
        border: 1px solid {BORDER} !important;
    }}
    .card:empty {{
        display: none !important;
    }}

    .section-header {{
        border-bottom: 2px solid {BORDER} !important;
        padding-bottom: 0.5rem !important;
        margin-bottom: 1.2rem !important;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}

    .section-header h3 {{
        margin: 0;
        font-size: 1.25rem;
        font-weight: 600;
        color: #FFFFFF !important;
    }}
    </style>
""", unsafe_allow_html=True)

def card_start():
    st.markdown('<div class="card">', unsafe_allow_html=True)
def card_end():
    st.markdown('</div>', unsafe_allow_html=True)
def section_header(title: str):
    st.markdown(f'<div class="section-header"><h3>{title}</h3></div>', unsafe_allow_html=True)

import sys
base_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(base_path))
from data_manager import get_file_hash

# ---------------------------------------------------------------------------
# DECOUPLED ARCHITECTURE: STREAMLIT CACHING FOR OFFLINE ARTIFACTS
# ---------------------------------------------------------------------------
@st.cache_data
def load_hyperparameter_results(csv_path: str, file_hash: str) -> pd.DataFrame:
    """Read-only cached loading of hyperparameter tuning results."""
    return pd.read_csv(csv_path)

@st.cache_data
def load_evaluation_metrics(csv_path: str, file_hash: str) -> pd.DataFrame:
    """Read-only cached loading of evaluation metrics."""
    return pd.read_csv(csv_path)

@st.cache_data
def load_human_validation_csv(csv_path: str, file_hash: str) -> pd.DataFrame:
    """Read-only cached loading of human expert validation dataset."""
    return pd.read_csv(csv_path)

@st.cache_data
def load_viz_html(html_path: str, file_hash: str) -> str:
    """Read-only cached loading of PyLDAvis self-contained HTML."""
    with open(html_path, 'r', encoding='utf-8') as f:
        return f.read()


st.title("📈 Evaluasi Model LDA (Decoupled UI Architecture)")
st.markdown("Halaman ini menyajikan metrik performa model, validasi pakar (*Human Intrusion Test*), dan hasil hyperparameter tuning.")

base_path = Path(__file__).parent.parent.parent
results_path = base_path / "model" / "hyperparameter_results.csv"
metrics_path = base_path / "model" / "evaluation_metrics.csv"
viz_path = base_path / "model" / "lda_visualization.html"
human_val_path = base_path / "model" / "human_topic_validation.csv"

# 1. Load Tuning Results
card_start()
section_header("📊 Hasil Hyperparameter Tuning & Metrics")
if results_path.exists():
    df = load_hyperparameter_results(str(results_path), get_file_hash(results_path))
    
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
    st.warning("Data hyperparameter_results.csv tidak ditemukan. Menampilkan metrik evaluasi dasar model utama:")
    if metrics_path.exists():
        metrics_df = load_evaluation_metrics(str(metrics_path), get_file_hash(metrics_path))
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
st.markdown("<br>", unsafe_allow_html=True)
section_header("📍 Visualisasi Interaktif Model (PyLDAvis)")
if viz_path.exists():
    html_string = load_viz_html(str(viz_path), get_file_hash(viz_path))
    st.caption("Klik pada gelembung topik di sebelah kiri untuk melihat persebaran kata kuncinya di sebelah kanan.")
    
    with st.container():
        components.html(html_string, width=1300, height=800, scrolling=True)
else:
    st.warning("File lda_visualization.html tidak ditemukan. Pastikan model telah dilatih dengan benar.")

# 3. Human-in-the-Loop Expert Validation Export Section
st.markdown("<br>", unsafe_allow_html=True)
card_start()
section_header("👩‍🏫 Human-in-the-loop: Export Dataset Validasi Pakar")
st.markdown("""
Sistem menyediakan dataset khusus yang telah diformat untuk pengujian kualitatif/intrusi topik oleh pakar (*Human Topic Intrusion Test*).
Dosen penguji atau pakar domain dapat menilai relevansi dokumen (skala 1-5) dan memberikan umpan balik pada setiap topik yang dihasilkan LDA.
""")

if human_val_path.exists():
    val_df = load_human_validation_csv(str(human_val_path), get_file_hash(human_val_path))
    st.dataframe(val_df.head(10), hide_index=True)
    
    csv_bytes = val_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Unduh Dataset Validasi Pakar (human_topic_validation.csv)",
        data=csv_bytes,
        file_name="human_topic_validation.csv",
        mime="text/csv"
    )
else:
    st.info("Jalankan `python auto_labeling.py` untuk mengekspor dataset validasi pakar.")
card_end()

# 4. Grid Search per K
st.markdown("---")
card_start()
section_header("📊 Hasil Grid Search Hyperparameter per K")

if results_path.exists():
    hp_df = load_hyperparameter_results(str(results_path), get_file_hash(results_path))

    best_per_k = hp_df.loc[hp_df.groupby('k')['coherence_cv'].idxmax()].reset_index(drop=True)
    display_cols = best_per_k[['k', 'coherence_cv', 'coherence_umass', 'log_perplexity', 'alpha', 'eta']].copy()
    display_cols.columns = ['K Topik', 'Coherence CV', 'Coherence UMass', 'Log Perplexity', 'Alpha', 'Eta']
    display_cols['Coherence CV'] = display_cols['Coherence CV'].round(4)
    display_cols['Log Perplexity'] = display_cols['Log Perplexity'].round(2)

    max_cv_k = display_cols.loc[display_cols['Coherence CV'].idxmax()]['K Topik']

    def highlight_best(row):
        return ['background-color: rgba(59, 130, 246, 0.25); font-weight: bold;' if row['K Topik'] == max_cv_k else '' for _ in row]

    st.markdown("Tabel di bawah menunjukkan konfigurasi **terbaik per jumlah topik (K)** berdasarkan metrik Coherence C_V (baris terbaik di-highlight):")
    st.dataframe(
        display_cols.style.apply(highlight_best, axis=1),
        use_container_width=True,
        hide_index=True
    )
card_end()
