import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# Setup path so trend_analyzer can be imported
base_path = Path(__file__).parent.parent.parent
sys.path.append(str(base_path))

try:
    from trend_analyzer import compute_wma, determine_trend_direction
except ImportError:
    st.error("Gagal mengimpor fungsi dari trend_analyzer. Pastikan file tersebut ada di direktori root.")

# ── Color Palette ──
PRIMARY = "#1E3A5F"
SECONDARY = "#2E86AB"
ACCENT = "#F39C12"
BG_LIGHT = "#F4F6F9"
TEXT = "#1A1A2E"
MUTED = "#7F8C8D"
SUCCESS = "#27AE60"
DANGER = "#E74C3C"
CARD_BG = "#FFFFFF"

st.set_page_config(page_title="Analisis Tren", page_icon="📈", layout="wide")

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

st.title("📊 Analisis & Proyeksi Tren Topik")
st.markdown("Analisis popularitas topik dari tahun ke tahun beserta proyeksi menggunakan *Weighted Moving Average* (WMA).")

# Load distribution data
@st.cache_data
def load_trend_data():
    dist_path = base_path / "model" / "topic_distribution.csv"
    labels_path = base_path / "model" / "topic_labels.csv"
    
    if not dist_path.exists():
        return None, None
        
    df = pd.read_csv(dist_path)
    
    # Try loading labels for friendly names
    labels = {}
    if labels_path.exists():
        lbl_df = pd.read_csv(labels_path)
        if 'topic_id' in lbl_df.columns and 'label' in lbl_df.columns:
            labels = dict(zip(lbl_df['topic_id'], lbl_df['label']))
            
    return df, labels

df_dist, dict_labels = load_trend_data()

if df_dist is None:
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
    
    # We will compute trend for the selected topic
    # Filter dataset first based on year range
    df_filtered = df_dist[(df_dist['Tahun'] >= year_range[0]) & (df_dist['Tahun'] <= year_range[1])]
    
    if df_filtered.empty:
        st.warning("Tidak ada data untuk rentang tahun yang dipilih.")
    else:
        # Calculate yearly count for the specific topic
        yearly_counts = df_filtered[df_filtered['topik_dominan'] == selected_topic].groupby('Tahun').size()
        
        # We need a complete index of years
        all_years = list(range(year_range[0], year_range[1] + 1))
        historical = pd.Series(index=all_years, data=[yearly_counts.get(y, 0) for y in all_years])
        
        # Perform WMA Forecast (next 3 years)
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
        
        # Calculate standard deviation of historical data for confidence interval
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
