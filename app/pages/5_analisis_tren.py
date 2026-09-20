import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# Atur path agar trend_analyzer dapat diimpor
base_path = Path(__file__).parent.parent.parent
sys.path.append(str(base_path))

try:
    from trend_analyzer import compute_wma, determine_trend_direction
except ImportError:
    st.error("Gagal mengimpor fungsi dari trend_analyzer. Pastikan file tersebut ada di direktori root.")

# ── Skema Warna ──
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

st.set_page_config(page_title="Analisis Tren", page_icon="📈", layout="wide")

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

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
    section[data-testid="stSidebar"] span {{
        color: #FFFFFF !important;
    }}

    /* Streamlit Native Multipage Nav Links */
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] {{
        background-color: #111827 !important;
        padding-top: 1rem;
    }}

    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] {{
        background-color: #111827 !important;
    }}

    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a {{
        background-color: transparent !important;
        border-radius: 8px !important;
        padding: 0.5rem 0.8rem !important;
        margin-bottom: 0.2rem !important;
        transition: all 0.15s ease !important;
    }}

    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a span {{
        color: {MUTED} !important;
    }}

    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover {{
        background-color: rgba(59, 130, 246, 0.1) !important;
    }}

    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover span {{
        color: #FFFFFF !important;
    }}

    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] {{
        background-color: rgba(59, 130, 246, 0.2) !important;
    }}

    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] span {{
        color: #FFFFFF !important;
        font-weight: 600 !important;
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

st.title("📊 Analisis & Proyeksi Tren Topik")
st.markdown("Analisis popularitas topik dari tahun ke tahun beserta proyeksi menggunakan *Weighted Moving Average* (WMA).")

from data_manager import load_unified_model_data, get_model_artifacts_hash

@st.cache_data
def load_trend_data(artifacts_hash: str):
    unified = load_unified_model_data(base_path)
    df_dist = unified["topic_distribution"]
    labels_df = unified["topic_labels"]
    labels = {}
    if not labels_df.empty:
        labels = dict(zip(labels_df['topic_id'], labels_df['label']))
    return df_dist, labels

artifacts_hash = get_model_artifacts_hash(base_path)
df_dist, dict_labels = load_trend_data(artifacts_hash)

if df_dist is None or df_dist.empty:
    st.warning("File model/topic_distribution.csv tidak ditemukan. Harap pastikan model LDA sudah berjalan.")
    st.stop()

card_start()
col1, col2, col3 = st.columns(3)
with col1:
    topics = sorted(df_dist['topik_dominan'].unique())
    topic_options = {t: dict_labels.get(t, f"Topik {t}") for t in topics}
    selected_topic = st.selectbox("Pilih Topik:", list(topic_options.keys()), format_func=lambda x: topic_options[x])
    
with col2:
    min_year = int(df_dist['Tahun'].min())
    max_year = int(df_dist['Tahun'].max())
    year_range = st.slider("Rentang Tahun Historis:", min_value=min_year, max_value=max_year, value=(min_year, max_year))

with col3:
    wma_window = st.selectbox("Window WMA (Tahun):", [3, 5, 7], index=0)
card_end()

# Run Analyzer
try:
    
    # Hitung tren untuk topik yang dipilih
    # Filter dataset berdasarkan rentang tahun
    df_filtered = df_dist[(df_dist['Tahun'] >= year_range[0]) & (df_dist['Tahun'] <= year_range[1])]
    
    if df_filtered.empty:
        st.warning("Tidak ada data untuk rentang tahun yang dipilih.")
    else:
        # Hitung jumlah skripsi per tahun untuk topik spesifik
        yearly_counts = df_filtered[df_filtered['topik_dominan'] == selected_topic].groupby('Tahun').size()
        
        # Buat indeks lengkap tahun
        all_years = list(range(year_range[0], year_range[1] + 1))
        historical = pd.Series(index=all_years, data=[yearly_counts.get(y, 0) for y in all_years])
        
        # Lakukan prediksi WMA (3 tahun ke depan)
        weights = np.arange(1, wma_window + 1)
        
        forecast = []
        history = list(historical.values)
        
        for _ in range(3):
            if len(history) < wma_window:
                forecast.append(0)
                continue
            wma_val = np.dot(history[-wma_window:], weights) / weights.sum()
            forecast.append(wma_val)
            history.append(wma_val)
            
        forecast_years = [year_range[1] + 1, year_range[1] + 2, year_range[1] + 3]
        
        # Hitung deviasi standar data historis untuk interval kepercayaan
        std_dev = historical.std() if len(historical) > 1 else 0
        upper_bound = [f + std_dev for f in forecast]
        lower_bound = [max(0, f - std_dev) for f in forecast]
        
        card_start()
        section_header(f"📈 Tren & Proyeksi: {topic_options[selected_topic]}")
        
        fig = go.Figure()
        
        # Historical
        fig.add_trace(go.Scatter(
            x=historical.index, y=historical.values,
            mode='lines+markers', name='Historis',
            line=dict(color=PRIMARY, width=3),
            marker=dict(size=8)
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=forecast_years, y=forecast,
            mode='lines+markers', name='Proyeksi WMA',
            line=dict(color=ACCENT, width=3, dash='dash'),
            marker=dict(size=8)
        ))
        
        # Confidence Interval (Upper & Lower bound overlay)
        fig.add_trace(go.Scatter(
            name='Batas Atas (+1 STD)',
            x=forecast_years,
            y=upper_bound,
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            name='Interval Kepercayaan',
            x=forecast_years,
            y=lower_bound,
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(243, 156, 18, 0.2)',
            fill='tonexty',
            showlegend=True
        ))
        
        fig.update_layout(height=400, hovermode="x unified", xaxis_title="Tahun", yaxis_title="Jumlah Skripsi")
        st.plotly_chart(fig, use_container_width=True)
        card_end()
        
        # Summary & Export
        card_start()
        colA, colB = st.columns([1, 1])
        with colA:
            section_header("📝 Ringkasan Proyeksi")
            trend_status = "Meningkat" if forecast[-1] > historical.iloc[-1] else ("Menurun" if forecast[-1] < historical.iloc[-1] else "Stabil")
            st.write(f"Berdasarkan proyeksi WMA dengan window {wma_window}, tren topik ini diperkirakan akan **{trend_status}** dalam 3 tahun ke depan.")
            
            st.write(f"- Tahun {forecast_years[0]}: **{forecast[0]:.1f}** skripsi (±{std_dev:.1f})")
            st.write(f"- Tahun {forecast_years[-1]}: **{forecast[-1]:.1f}** skripsi (±{std_dev:.1f})")
            
        with colB:
            section_header("📥 Export Data")
            export_df = pd.DataFrame({
                'Tahun': list(historical.index) + forecast_years,
                'Nilai': list(historical.values) + forecast,
                'Tipe': ['Historis'] * len(historical) + ['Proyeksi'] * len(forecast)
            })
            
            st.download_button(
                label="Unduh CSV Hasil Proyeksi",
                data=export_df.to_csv(index=False),
                file_name=f"proyeksi_{selected_topic}.csv",
                mime="text/csv",
                use_container_width=True
            )
        card_end()
except Exception as e:
    st.error(f"Terjadi kesalahan saat memproses data tren: {e}")
