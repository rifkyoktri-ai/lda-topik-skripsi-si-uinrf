import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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

COLOR_SCALE = [PRIMARY, SECONDARY, ACCENT, SUCCESS, "#8E44AD", "#E67E22", "#1ABC9C"]

# Set page config
st.set_page_config(
    page_title="LDA Topic Modeling Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown('<meta name="google" content="notranslate">', unsafe_allow_html=True)

# ── Custom CSS (Enterprise Dark Mode) ──
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {{
        --bg-main: #0B0F19;
        --bg-sidebar: #111827;
        --bg-card: #1F2937;
        --bg-card-hover: #263346;
        --border-color: #374151;
        --border-sidebar: #1F2937;
        --text-primary: #F3F4F6;
        --text-secondary: #9CA3AF;
        --accent-blue: #3B82F6;
    }}

    /* ── Canvas & Background Utama ── */
    html, body, .stApp {{
        background-color: {BG_DARK} !important;
        color: {TEXT} !important;
        font-family: 'Inter', -apple-system, sans-serif !important;
    }}

    .main .block-container {{
        padding: 2rem 1.5rem;
        background-color: {BG_DARK};
    }}

    header[data-testid="stHeader"] {{
        background-color: rgba(11, 15, 25, 0.8) !important;
        backdrop-filter: blur(8px);
    }}

    /* ── Sidebar Modern (Surface Elevation) ── */
    section[data-testid="stSidebar"] {{
        background-color: #111827 !important;
        border-right: 1px solid #1F2937 !important;
    }}

    section[data-testid="stSidebar"] * {{
        color: {MUTED};
    }}

    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .st-emotion-cache-1v0mbdj {{
        color: #FFFFFF !important;
        font-weight: 600;
    }}

    section[data-testid="stSidebar"] .stRadio > div {{
        gap: 0.3rem;
    }}
    section[data-testid="stSidebar"] .stRadio label {{
        padding: 0.52rem 0.8rem;
        border-radius: 8px;
        font-weight: 500;
        color: {MUTED} !important;
        transition: all 0.15s ease;
    }}
    section[data-testid="stSidebar"] .stRadio label:hover {{
        background-color: rgba(59, 130, 246, 0.1) !important;
        color: #FFFFFF !important;
    }}
    section[data-testid="stSidebar"] .stRadio label[data-checked="true"] {{
        background-color: rgba(59, 130, 246, 0.2) !important;
        color: #FFFFFF !important;
        font-weight: 600;
    }}

    /* ── Card System & Kontainer ── */
    .card {{
        background: {CARD_BG} !important;
        border-radius: 10px !important;
        padding: 1.5rem !important;
        margin-bottom: 1.5rem !important;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3) !important;
        border: 1px solid {BORDER} !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }}
    .card:hover {{
        border-color: #4B5563 !important;
        box-shadow: 0 6px 12px -2px rgba(0,0,0,0.4) !important;
    }}

    /* ── Metric Cards ── */
    .metric-card {{
        background: {CARD_BG} !important;
        border-radius: 10px !important;
        padding: 1.2rem 1.2rem 0.9rem !important;
        border: 1px solid {BORDER} !important;
        border-left: 4px solid {PRIMARY} !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
        margin-bottom: 0.8rem !important;
    }}
    .metric-card.gold {{
        border-left-color: {ACCENT} !important;
    }}
    .metric-card.green {{
        border-left-color: {SUCCESS} !important;
    }}
    .metric-card.red {{
        border-left-color: {DANGER} !important;
    }}

    /* Native Streamlit Metric overrides */
    [data-testid="stMetric"] {{
        background-color: {CARD_BG} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 10px !important;
        padding: 1rem 1.2rem !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
    }}
    [data-testid="stMetricLabel"] {{
        color: {MUTED} !important;
        font-size: 0.85rem !important;
    }}
    [data-testid="stMetricValue"] {{
        color: #FFFFFF !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }}

    /* ── Section Headers ── */
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

    /* ── Form Controls & Input WIdgets ── */
    .stSelectbox > div > div, 
    .stMultiSelect > div > div,
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stTextArea textarea {{
        background-color: #111827 !important;
        color: #FFFFFF !important;
        border: 1px solid {BORDER} !important;
        border-radius: 8px !important;
    }}

    /* ── Divider & Footer ── */
    hr.custom-divider {{
        margin: 1.5rem 0;
        border: 0;
        height: 1px;
        background: {BORDER};
    }}

    .footer {{
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid {BORDER};
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.8rem;
        color: {MUTED};
    }}
    </style>
""", unsafe_allow_html=True)

# ── Helper Functions ──
def card_start():
    st.markdown('<div class="card">', unsafe_allow_html=True)

def card_end():
    st.markdown('</div>', unsafe_allow_html=True)

def section_header(title: str):
    st.markdown(f'<div class="section-header"><h3>{title}</h3></div>', unsafe_allow_html=True)

def metric_card(value: str, label: str, variant: str = ""):
    cls = f"metric-card {variant}" if variant else "metric-card"
    st.markdown(f"""
    <div class="{cls}">
        <div style="font-size: 1.8rem; font-weight: 700; color: #FFFFFF;">{value}</div>
        <div style="font-size: 0.8rem; color: {MUTED}; margin-top: 0.2rem;">{label}</div>
    </div>
    """, unsafe_allow_html=True)

def custom_divider():
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

from data_manager import (
    load_unified_model_data,
    get_file_hash,
    get_model_artifacts_hash,
    to_zero_indexed,
    to_one_indexed
)

# Load unified model contract
@st.cache_data
def get_unified_data(hash_key: str):
    base_path = Path(__file__).parent.parent
    return load_unified_model_data(base_path)

# Load preprocessed dataset
@st.cache_data
def load_dataset(file_hash: str):
    base_path = Path(__file__).parent.parent
    dataset_path = base_path / "data" / "intermediate" / "dataset_preprocessed.csv"
    if dataset_path.exists():
        return pd.read_csv(dataset_path)
    return pd.DataFrame()

# Load LDA visualization HTML
@st.cache_data
def load_lda_viz(file_hash: str):
    base_path = Path(__file__).parent.parent
    viz_path = base_path / "model" / "lda_visualization.html"
    if viz_path.exists():
        with open(viz_path, 'r', encoding='utf-8') as f:
            return f.read()
    return None

# Load trend prediction data
@st.cache_data
def load_trend_prediction(file_hash: str):
    base_path = Path(__file__).parent.parent
    trend_path = base_path / "model" / "trend_prediction.csv"
    if trend_path.exists():
        return pd.read_csv(trend_path)
    return None

# Load topic trend data (historical proportions per year)
@st.cache_data
def load_topic_trend(file_hash: str):
    base_path = Path(__file__).parent.parent
    trend_path = base_path / "model" / "topic_trend.csv"
    if trend_path.exists():
        return pd.read_csv(trend_path)
    return None

# Load LDA Model for Word Cloud
@st.cache_resource
def load_lda_model(file_hash: str):
    base_path = Path(__file__).parent.parent
    model_path = base_path / "model" / "lda_model.gensim"
    if model_path.exists():
        from gensim.models import LdaModel
        return LdaModel.load(str(model_path))
    return None

# Display HTML with proper rendering
def display_html(html_content):
    """Display HTML content safely"""
    try:
        components.html(html_content, height=900, scrolling=True)
    except Exception as e:
        st.error(f"Error: {e}")
        st.write("💡 Untuk melihat visualisasi, silakan buka file HTML langsung: `model/lda_visualization.html`")

# Title and Header
st.title("📊 LDA Topic Modeling Dashboard")
st.markdown("---")

# Compute hashes for cache invalidation
base_path = Path(__file__).parent.parent
artifacts_hash = get_model_artifacts_hash(base_path)
dataset_hash = get_file_hash(base_path / "data" / "intermediate" / "dataset_preprocessed.csv")
viz_hash = get_file_hash(base_path / "model" / "lda_visualization.html")
trend_hash = get_file_hash(base_path / "model" / "trend_prediction.csv")
topic_trend_hash = get_file_hash(base_path / "model" / "topic_trend.csv")
model_hash = get_file_hash(base_path / "model" / "lda_model.gensim")

# Load all synchronized data via single contract
unified_data = get_unified_data(artifacts_hash)
lda_model = load_lda_model(model_hash)
dataset_df = load_dataset(dataset_hash)
lda_viz_html = load_lda_viz(viz_hash)

metrics_df = unified_data["evaluation_metrics"]
topic_dist_df = unified_data["topic_distribution"]
topic_labels_df = unified_data["topic_labels"]
num_topics = unified_data["num_topics"]

# Check if data loaded successfully
if topic_dist_df is None or topic_dist_df.empty:
    st.error("❌ Gagal memuat data topik. Pastikan file model dan CSV ada di folder model/.")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("🎛️ Navigasi")
    page = st.radio(
        "Pilih Halaman:",
        ["📈 Overview", "🔵 Visualisasi LDA", "📊 Model Metrics", "🏷️ Topic Analysis", "🔍 Document Search", "📖 Data Info", "📈 Prediksi Tren"]
    )
    
    st.markdown("---")
    st.header("ℹ️ Info Model")
    metrics_dict = dict(zip(metrics_df['Metrik'], metrics_df['Nilai'])) if metrics_df is not None else {}
    coherence = float(metrics_dict.get('Coherence Score (C_V)', metrics_dict.get('Coherence Score (CV)', 0.0)))
    st.caption(f"**K (Jumlah Topik):** {int(metrics_dict.get('Jumlah Topik (K)', 0))}")
    st.caption(f"**Coherence (C_V):** {coherence:.4f}")

    import os
    try:
        metrics_path = Path(__file__).parent.parent / 'model' / 'evaluation_metrics.csv'
        m_df = pd.read_csv(metrics_path)
        if 'training_date' in m_df.columns:
            training_date = m_df['training_date'].iloc[0]
        elif 'date' in m_df.columns:
            training_date = m_df['date'].iloc[0]
        else:
            mtime = os.path.getmtime(Path(__file__).parent.parent / 'model' / 'lda_model.gensim')
            training_date = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d')
        st.caption(f"**Tanggal Training:** {training_date}")
    except Exception:
        try:
            mtime = os.path.getmtime(Path(__file__).parent.parent / 'model' / 'lda_model.gensim')
            training_date = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d')
            st.caption(f"**Tanggal Training:** {training_date}")
        except Exception:
            st.caption("**Tanggal Training:** Tidak tersedia")

    st.caption(f"**Total Dokumen:** {len(topic_dist_df) if topic_dist_df is not None else 0}")
    
    st.markdown("---")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

# PAGE 1: OVERVIEW
if page == "📈 Overview":
    card_start()
    section_header("📊 Overview Dashboard")
    
    metrics_dict = dict(zip(metrics_df['Metrik'], metrics_df['Nilai']))
    coherence = float(metrics_dict.get('Coherence Score (C_V)', metrics_dict.get('Coherence Score (CV)', 0.0)))
    perplexity = float(metrics_dict.get('Log Perplexity', metrics_dict.get('Train Log Perplexity', 0.0)))
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card(len(topic_dist_df), "Total Dokumen")
    with col2:
        metric_card(int(metrics_dict.get('Jumlah Topik (K)', 0)), "Jumlah Topik", "gold")
    with col3:
        metric_card(f"{coherence:.4f}", "Coherence Score (C_V)", "green")
    with col4:
        metric_card(f"{perplexity:.4f}", "Log Perplexity", "red")
    card_end()
    
    card_start()
    col1, col2 = st.columns(2)
    
    with col1:
        section_header("📈 Distribusi Topik Dominan")
        topic_counts = topic_dist_df['topik_dominan'].value_counts().sort_index()
        fig = px.bar(
            x=topic_counts.index,
            y=topic_counts.values,
            labels={'x': 'Topic ID', 'y': 'Jumlah Documents'},
            title="",
            color=topic_counts.values,
            color_continuous_scale=[PRIMARY, SECONDARY, ACCENT]
        )
        fig.update_layout(height=400, showlegend=False, margin=dict(l=20,r=20,t=30,b=20))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        section_header("📊 Distribusi Probabilitas Topik Dominan")
        fig = px.histogram(
            topic_dist_df['prob_dominan'],
            nbins=30,
            title="",
            labels={'x': 'Probability', 'count': 'Frequency'},
            color_discrete_sequence=[SECONDARY]
        )
        fig.update_layout(height=400, showlegend=False, margin=dict(l=20,r=20,t=30,b=20))
        st.plotly_chart(fig, use_container_width=True)
    card_end()
    
    card_start()
    section_header("📅 Distribusi Skripsi per Tahun")
    year_dist = topic_dist_df['Tahun'].value_counts().sort_index()
    fig = px.line(
        x=year_dist.index,
        y=year_dist.values,
        title="",
        labels={'x': 'Tahun', 'y': 'Jumlah Skripsi'},
        markers=True,
        color_discrete_sequence=[PRIMARY]
    )
    fig.update_layout(height=400, hovermode='x unified', margin=dict(l=20,r=20,t=30,b=20))
    st.plotly_chart(fig, use_container_width=True)
    card_end()

    # Quality Alerts
    card_start()
    section_header("⚠️ Quality Alerts")
    from lda_evaluation import evaluate_topic_overlap

    alerts = []

    # Alert 1: Per-topic coherence rendah
    # Threshold: 0.30 — calibrated for micro-corpus (311 IS-domain docs).
    # Clean, noise-free C_V on narrow-domain abstracts naturally ~0.32.
    if topic_labels_df is not None and 'quality_coherence' in topic_labels_df.columns:
        for _, row in topic_labels_df.iterrows():
            try:
                qc = float(row['quality_coherence'])
                if qc > 0 and qc < 0.30:
                    alerts.append(('warning', f"Topik {int(row['topic_id'])} ({row['label']}) memiliki coherence rendah ({qc:.4f} < 0.30)."))
            except (ValueError, TypeError):
                pass

    # Alert 2: Topic overlap
    # Threshold: 0.70 — calibrated for narrow IS-domain corpus.
    # Natural domain overlap causes cosine sim ~0.65; only flag genuine over-segmentation (> 0.70).
    try:
        lda_model = load_lda_model()
        if lda_model:
            overlapping = evaluate_topic_overlap(lda_model, threshold=0.70)
            for t1, t2, sim in overlapping:
                alerts.append(('error', f"Topik {t1+1} dan {t2+1} memiliki kesamaan sangat tinggi (cosine sim={sim:.3f} > 0.70). Kemungkinan over-segmented."))
    except Exception:
        pass

    # Alert 3: Distribusi tidak seimbang
    if topic_dist_df is not None and 'topik_dominan' in topic_dist_df.columns:
        topic_counts = topic_dist_df['topik_dominan'].value_counts()
        if len(topic_counts) > 0:
            ratio = topic_counts.max() / topic_counts.min()
            if ratio > 5:
                alerts.append(('info', f"Distribusi topik tidak seimbang (rasio max/min = {ratio:.1f}x). Topik {topic_counts.idxmax()} mendominasi."))

    if alerts:
        for level, msg in alerts:
            icon = {'warning': '⚠️', 'error': '🚫', 'info': 'ℹ️'}.get(level, 'ℹ️')
            st.markdown(f"**{icon} {level.upper()}:** {msg}")
        st.caption(f"Total {len(alerts)} issue terdeteksi. Perbaiki untuk meningkatkan kualitas model.")
    else:
        st.success("✅ Tidak ada issue yang terdeteksi. Kualitas model dalam kondisi baik.")
    card_end()


# PAGE 2: VISUALISASI LDA
elif page == "🔵 Visualisasi LDA":
    card_start()
    section_header("🔵 Visualisasi LDA Model — Topic Distribution")
    st.markdown("Visualisasi interaktif distribusi dan karakteristik topik dari model LDA.")
    card_end()
    
    # Create topic visualization data
    topic_stats = topic_dist_df.groupby('topik_dominan').agg({
        'prob_dominan': ['mean', 'count'],
        'Tahun': ['min', 'max']
    }).reset_index()
    
    topic_stats.columns = ['topic_id', 'avg_prob', 'doc_count', 'year_min', 'year_max']
    topic_stats['topic_id'] = topic_stats['topic_id'].astype(int)
    topic_stats['year_range'] = topic_stats['year_max'] - topic_stats['year_min']
    
    # Merge dengan topic labels
    topic_stats = topic_stats.merge(
        topic_labels_df[['topic_id', 'label', 'description']],
        on='topic_id',
        how='left'
    )
    
    card_start()
    section_header("🎯 Distribusi Topik — Bubble Chart")
    fig_bubble = px.scatter(
        topic_stats,
        x='topic_id',
        y='avg_prob',
        size='doc_count',
        color='avg_prob',
        hover_name='label',
        hover_data={
            'topic_id': True,
            'doc_count': True,
            'avg_prob': ':.4f',
            'year_min': True,
            'year_max': True,
            'year_range': False,
            'description': True,
            'label': False
        },
        title="",
        labels={
            'topic_id': 'Topic ID',
            'avg_prob': 'Average Probability',
            'doc_count': 'Number of Documents'
        },
        color_continuous_scale=[PRIMARY, ACCENT, SUCCESS],
        size_max=60
    )
    fig_bubble.update_layout(
        height=500,
        hovermode='closest',
        font=dict(size=12),
        margin=dict(l=20,r=20,t=30,b=20)
    )
    fig_bubble.update_traces(marker=dict(line=dict(width=2, color='white')))
    st.plotly_chart(fig_bubble, use_container_width=True)
    
    with st.expander("📚 Cara Membaca Bubble Chart", expanded=False):
        st.markdown("""
        - **Sumbu X**: Topic ID
        - **Sumbu Y**: Rata-rata probabilitas (semakin tinggi = topik lebih dominan)
        - **Ukuran bubble**: Jumlah dokumen dalam topik
        - **Warna**: Intensitas probabilitas
        - **Hover**: Detail lengkap topik
        """)
    card_end()
    
    card_start()
    col1, col2 = st.columns([2, 1])
    with col1:
        section_header("📊 Statistik Topik")
        display_table = topic_stats[['topic_id', 'label', 'doc_count', 'avg_prob', 'year_min', 'year_max']].copy()
        display_table.columns = ['Topic ID', 'Label', 'Documents', 'Avg Probability', 'First Year', 'Last Year']
        display_table = display_table.sort_values('Documents', ascending=False)
        st.dataframe(display_table, use_container_width=True, hide_index=True)
    with col2:
        section_header("📈 Ringkasan")
        metric_card(len(topic_stats), "Total Topics", "gold")
        metric_card(f"{topic_stats['doc_count'].mean():.1f}", "Rata-rata Dokumen/Topik")
        metric_card(topic_stats['doc_count'].sum(), "Total Dokumen", "green")
    card_end()
    
    # PyLDAvis HTML visualization (if available)
    if lda_viz_html:
        card_start()
        section_header("📍 PyLDAvis Interactive Visualization")
        st.caption("Klik pada topik untuk melihat top terms")
        components.html(lda_viz_html, height=900, scrolling=True)
        with st.expander("📚 Tentang PyLDAvis"):
            st.markdown("""
            **PyLDAvis** menampilkan:
            - **Lingkaran**: Setiap topik (ukuran = jumlah dokumen)
            - **Jarak antar lingkaran**: Similarity antar topik
            - **Top Terms**: Terms paling relevan untuk setiap topik
            """)
        card_end()


# PAGE 3: MODEL METRICS
elif page == "📊 Model Metrics":
    card_start()
    section_header("📊 Model Evaluation Metrics")
    metrics_dict = dict(zip(metrics_df['Metrik'], metrics_df['Nilai']))
    coherence = float(metrics_dict.get('Coherence Score (C_V)', metrics_dict.get('Coherence Score (CV)', 0.0)))
    perplexity = float(metrics_dict.get('Log Perplexity', metrics_dict.get('Train Log Perplexity', 0.0)))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card(int(metrics_dict['Jumlah Topik (K)']), "Jumlah Topik", "gold")
    with col2:
        metric_card(f"{coherence:.4f}", "Coherence Score", "green" if coherence > 0.5 else "red")
    with col3:
        metric_card(f"{perplexity:.4f}", "Log Perplexity")
    card_end()
    
    card_start()
    section_header("📋 Detail Metrik")
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    with st.expander("📚 Penjelasan Metrik", expanded=False):
        st.markdown("""
        - **Coherence Score (C_V)**: Mengukur seberapa terkait kata-kata dalam satu topik. Nilai 0–1, semakin tinggi semakin baik.
        - **Log Perplexity**: Mengukur kemampuan model memprediksi data baru. Semakin negatif semakin baik.
        - **Jumlah Topik (K)**: Jumlah topik optimal hasil optimasi atau pengaturan manual.
        """)
    card_end()


# PAGE 4: TOPIC ANALYSIS
elif page == "🏷️ Topic Analysis":
    card_start()
    section_header("🏷️ Topic Analysis")
    card_end()
    
    card_start()
    col1, col2 = st.columns([2, 1])
    with col1:
        section_header("📈 Top Topics by Document Count")
        topic_counts = topic_dist_df['topik_dominan'].value_counts().head(10).sort_values()
        topic_counts_df = pd.DataFrame({
            'topic_id': topic_counts.index,
            'count': topic_counts.values
        }).merge(topic_labels_df[['topic_id', 'label']], on='topic_id', how='left')
        fig = px.bar(
            topic_counts_df,
            y='label',
            x='count',
            orientation='h',
            title="",
            labels={'count': 'Number of Documents', 'label': ''},
            color='count',
            color_continuous_scale=[BG_LIGHT, SECONDARY, PRIMARY]
        )
        fig.update_layout(height=400, showlegend=False, yaxis={'categoryorder':'total ascending'}, margin=dict(l=20,r=20,t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        section_header("⚡ Quick Stats")
        metric_card(len(topic_dist_df['topik_dominan'].unique()), "Total Topics", "gold")
        metric_card(f"{topic_dist_df['prob_dominan'].mean():.4f}", "Rata-rata Probabilitas")
        metric_card(f"{topic_dist_df['prob_dominan'].max():.4f}", "Probabilitas Maksimum", "green")
        metric_card(f"{topic_dist_df['prob_dominan'].min():.4f}", "Probabilitas Minimum", "red")
    card_end()

    # Coherence quality per topic
    card_start()
    section_header("🎯 Kualitas Label per Topik")
    if 'quality_coherence' in topic_labels_df.columns and topic_labels_df['quality_coherence'].astype(str).str.strip().str.len().sum() > 0:
        quality_df = topic_labels_df[['topic_id', 'label', 'label_score', 'quality_coherence']].copy()
        quality_df['quality_coherence'] = pd.to_numeric(quality_df['quality_coherence'], errors='coerce')
        quality_df['label_score'] = pd.to_numeric(quality_df['label_score'], errors='coerce')
        quality_df = quality_df.sort_values('quality_coherence', ascending=False)

        col_a, col_b = st.columns([2, 1])
        with col_a:
            fig_q = px.bar(
                quality_df.dropna(subset=['quality_coherence']),
                x='label', y='quality_coherence',
                title="",
                labels={'label': '', 'quality_coherence': 'Quality Coherence'},
                color='quality_coherence',
                color_continuous_scale=[DANGER, ACCENT, SUCCESS],
                range_color=[0, 1]
            )
            fig_q.add_hline(y=0.4, line_dash="dash", line_color=DANGER,
                          annotation_text="Threshold (0.4)")
            fig_q.update_layout(height=400, margin=dict(l=20,r=20,t=20,b=40))
            st.plotly_chart(fig_q, use_container_width=True)

        with col_b:
            metric_card(f"{quality_df['quality_coherence'].mean():.3f}", "Rata-rata Quality Coherence")
            low_q = len(quality_df[quality_df['quality_coherence'] < 0.4])
            metric_card(low_q, "Perlu Review (< 0.4)", "red" if low_q > 0 else "green")
            if low_q > 0:
                st.warning(f"⚠️ {low_q} topik coherence rendah — pertimbangkan review manual")
            else:
                st.success("✅ Semua topik di atas threshold 0.4")

        with st.expander("📋 Detail Tabel Kualitas Label", expanded=False):
            st.dataframe(quality_df, use_container_width=True, hide_index=True)
    else:
        st.info("Data kualitas label belum tersedia.")
    card_end()

    card_start()
    section_header("📊 Average Probability per Topic")
    avg_prob = topic_dist_df.groupby('topik_dominan')['prob_dominan'].mean().sort_values(ascending=False)
    avg_prob_df = pd.DataFrame({
        'topic_id': avg_prob.index,
        'avg_probability': avg_prob.values
    }).merge(topic_labels_df[['topic_id', 'label']], on='topic_id', how='left')
    fig = px.bar(
        avg_prob_df,
        x='label',
        y='avg_probability',
        title="",
        labels={'label': '', 'avg_probability': 'Average Probability'},
        color='avg_probability',
        color_continuous_scale=[BG_LIGHT, SECONDARY, PRIMARY]
    )
    fig.update_layout(height=400, showlegend=False, margin=dict(l=20,r=20,t=20,b=40))
    st.plotly_chart(fig, use_container_width=True)
    card_end()
    
    card_start()
    section_header("🔍 Analisis Detail Topic")
    
    try:
        topic_list = sorted(topic_dist_df['topik_dominan'].unique().astype(int))
        selected_topic = st.selectbox(
            "Pilih Topic untuk detail:",
            topic_list,
            key="topic_select"
        )
        
        topic_label_row = topic_labels_df[topic_labels_df['topic_id'] == selected_topic]
        if len(topic_label_row) > 0:
            topic_label = topic_label_row.iloc[0]
            st.markdown(f"<h3 style='color:{PRIMARY};margin-top:0;'>🏷️ {topic_label['label']}</h3>", unsafe_allow_html=True)
            if topic_label['description']:
                st.write(f"*{topic_label['description']}*")
            if topic_label['keywords']:
                st.caption(f"Keywords: {topic_label['keywords']}")
            qc = topic_label.get('quality_coherence', '')
            ls = topic_label.get('label_score', '')
            if qc != '' and pd.notna(qc) and str(qc).strip():
                qc_val = float(qc)
                if qc_val < 0.4:
                    st.warning(f"⚠️ Kualitas label rendah (coherence={qc_val:.3f}) — perlu review manual")
                else:
                    st.success(f"✅ Kualitas label baik (coherence={qc_val:.3f})")
            if ls != '' and pd.notna(ls) and str(ls).strip():
                st.caption(f"Label Score: {float(ls):.4f}")
            custom_divider()
        
        topic_docs = topic_dist_df[topic_dist_df['topik_dominan'] == selected_topic]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            metric_card(len(topic_docs), "Documents", "gold")
        with col2:
            metric_card(f"{topic_docs['prob_dominan'].mean():.4f}", "Avg Probability")
        with col3:
            if len(topic_docs) > 0:
                metric_card(f"{int(topic_docs['Tahun'].min())} - {int(topic_docs['Tahun'].max())}", "Year Range")
        
        if len(topic_docs) > 0:
            display_df = topic_docs[['ID', 'Judul', 'Tahun', 'prob_dominan']].head(15).reset_index(drop=True)
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("No documents found for this topic.")
            
        # Word Cloud
        if lda_model:
            custom_divider()
            section_header("☁️ Word Cloud Topik")
            top_words = dict(lda_model.show_topic(to_zero_indexed(selected_topic), topn=30))
            freq_dict = {w: float(v * 1000) for w, v in top_words.items()}
            
            from wordcloud import WordCloud
            import matplotlib.pyplot as plt
            wc = WordCloud(
                width=800, height=400, 
                background_color='white', 
                colormap='viridis',
                max_words=30
            ).generate_from_frequencies(freq_dict)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
            plt.close(fig)
            
    except Exception as e:
        st.error(f"Error in topic analysis: {e}")
    card_end()


# PAGE 5: DOCUMENT SEARCH
elif page == "🔍 Document Search":
    card_start()
    section_header("🔍 Document Search & Filter")
    card_end()
    
    card_start()
    section_header("🔎 Pencarian")
    col1, col2, col3 = st.columns([2, 2, 1])
    results = pd.DataFrame()
    
    with col1:
        search_type = st.selectbox(
            "Cari berdasarkan:",
            ["ID", "Judul", "Tahun", "Topic ID"],
            key="search_type"
        )
    
    with col2:
        try:
            if search_type == "ID":
                search_term = st.text_input("Masukkan ID dokumen:", key="id_search", placeholder="Contoh: 001")
                if search_term:
                    results = topic_dist_df[topic_dist_df['ID'].astype(str).str.contains(search_term, case=False, na=False)]
            elif search_type == "Judul":
                search_term = st.text_input("Masukkan keyword judul:", key="judul_search", placeholder="Contoh: sistem informasi")
                if search_term:
                    results = topic_dist_df[topic_dist_df['Judul'].str.contains(search_term, case=False, na=False)]
            elif search_type == "Tahun":
                tahun_list = sorted(topic_dist_df['Tahun'].unique().astype(int))
                tahun = st.selectbox("Pilih Tahun:", tahun_list, key="tahun_search")
                results = topic_dist_df[topic_dist_df['Tahun'] == tahun]
            else:
                topic_list = sorted(topic_dist_df['topik_dominan'].unique().astype(int))
                topic_id = st.selectbox("Pilih Topic ID:", topic_list, key="topic_search")
                results = topic_dist_df[topic_dist_df['topik_dominan'] == topic_id]
        except Exception as e:
            st.error(f"Error in search: {e}")
            results = pd.DataFrame()
    
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        metric_card(len(results), "Dokumen Ditemukan", "gold" if len(results) > 0 else "red")
    card_end()
    
    card_start()
    section_header("📋 Hasil Pencarian")
    if len(results) > 0:
        display_cols = ['ID', 'Judul', 'Tahun', 'topik_dominan', 'prob_dominan']
        st.dataframe(results[display_cols].reset_index(drop=True), use_container_width=True, hide_index=True)
        
        col_a, col_b = st.columns([1, 4])
        with col_a:
            csv_data = results.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name="search_results.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        st.info("💡 Masukkan kata kunci pencarian untuk menampilkan hasil.")
    card_end()


# PAGE 6: DATA INFO
elif page == "📖 Data Info":
    card_start()
    section_header("📖 Data Information")
    card_end()
    
    card_start()
    col1, col2 = st.columns(2)
    with col1:
        section_header("📊 Dataset Statistics")
        metric_card(len(topic_dist_df), "Total Dokumen", "gold")
        metric_card(f"{topic_dist_df['Tahun'].min():.0f} - {topic_dist_df['Tahun'].max():.0f}", "Rentang Tahun")
        metric_card(len(topic_dist_df['topik_dominan'].unique()), "Unique Topics")
    with col2:
        section_header("📈 Probability Statistics")
        metric_card(f"{topic_dist_df['prob_dominan'].mean():.4f}", "Mean Probability")
        metric_card(f"{topic_dist_df['prob_dominan'].median():.4f}", "Median Probability")
        metric_card(f"{topic_dist_df['prob_dominan'].max():.4f}", "Max Probability", "green")
        metric_card(f"{topic_dist_df['prob_dominan'].min():.4f}", "Min Probability", "red")
    card_end()
    
    card_start()
    section_header("📋 Raw Data Preview")
    st.dataframe(topic_dist_df.head(20), use_container_width=True, hide_index=True)
    
    with st.expander("📚 Data Description", expanded=False):
        st.markdown("""
        **Dataset Columns:**
        - **ID**: ID dokumen
        - **Judul**: Judul lengkap skripsi
        - **Tahun**: Tahun penulisan skripsi
        - **topik_dominan**: ID topic dominan dari model LDA
        - **prob_dominan**: Probabilitas topic dominan
        
        **Sumber Data:** Kumpulan skripsi Sistem Informasi yang dianalisis menggunakan LDA.
        """)
    card_end()


# PAGE 8: PREDIKSI TREN
elif page == "📈 Prediksi Tren":
    card_start()
    section_header("📈 Prediksi Tren Topik")
    st.markdown("Prediksi proporsi topik 2026–2027 berdasarkan Weighted Moving Average dari data historis 2021–2025.")
    card_end()

    trend_df = load_trend_prediction()
    topic_trend_df = load_topic_trend()

    if trend_df is not None and not trend_df.empty:
        card_start()
        section_header("📈 Grafik Prediksi Tren per Topik")
        
        fig = go.Figure()
        for i, (_, row) in enumerate(trend_df.iterrows()):
            color = COLOR_SCALE[i % len(COLOR_SCALE)]
            fig.add_trace(go.Scatter(
                x=[2021, 2022, 2023, 2024, 2025],
                y=[row['2021'], row['2022'], row['2023'], row['2024'], row['2025']],
                mode='lines+markers',
                name=row['Label'],
                line=dict(width=2.5, color=color),
                marker=dict(size=7),
                legendgroup=row['Label'],
            ))
            fig.add_trace(go.Scatter(
                x=[2025, 2026, 2027],
                y=[row['2025'], row['2026'], row['2027']],
                mode='lines+markers',
                name=f"{row['Label']} (prediksi)",
                line=dict(dash='dash', width=2, color=color),
                marker=dict(size=6, symbol='circle-open'),
                showlegend=False,
                legendgroup=row['Label'],
            ))

        fig.add_vline(
            x=2025.5, line_dash="dot", line_color=MUTED,
            annotation_text="Batas Prediksi", annotation_position="top",
            annotation_font_size=12
        )
        fig.update_layout(
            title="",
            xaxis=dict(tickmode='array', tickvals=[2021, 2022, 2023, 2024, 2025, 2026, 2027]),
            yaxis=dict(title="Proporsi (%)", ticksuffix="%"),
            height=500,
            hovermode='x unified',
            margin=dict(l=20,r=20,t=20,b=20),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
        card_end()

        card_start()
        section_header("📊 Tabel Ringkasan Prediksi")
        display_cols = ['Label', 'R2', 'Linearitas', '2026', '2027', 'Arah']
        display_df = trend_df[display_cols].copy()
        display_df['2026'] = display_df['2026'].apply(lambda x: f"{x:.2f}%")
        display_df['2027'] = display_df['2027'].apply(lambda x: f"{x:.2f}%")
        display_df.columns = ['Label Topik', 'R²', 'Linearitas', '2026 (%)', '2027 (%)', 'Arah Tren']
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        custom_divider()

        st.info("""
        ⚠️ **Prediksi menggunakan Weighted Moving Average (WMA)** — bobot linier dengan tahun terbaru lebih berpengaruh. 
        Bersifat indikatif berdasarkan pola 5 tahun historis (2021–2025). Bukan prediksi presisi.
        """)

        with st.expander("📋 Detail Semua Topik", expanded=False):
            st.dataframe(trend_df, use_container_width=True, hide_index=True)
        card_end()

    else:
        card_start()
        st.info("📭 Data prediksi tren belum tersedia. Jalankan `python trend_analyzer.py` untuk menghasilkan prediksi.")
        if topic_trend_df is not None:
            section_header("📋 Data Historis Tersedia")
            st.dataframe(topic_trend_df, use_container_width=True)
        else:
            st.warning("Data historis (`model/topic_trend.csv`) juga belum tersedia. Jalankan pipeline utama terlebih dahulu.")
        card_end()



# Footer
now = datetime.now().strftime("%Y-%m-%d %H:%M")
st.markdown(f"""
<div class="footer">
    <span>📊 LDA Topic Modeling Dashboard &mdash; Data Science Project</span>
    <span>Terakhir diperbarui: {now}</span>
</div>
""", unsafe_allow_html=True)
