import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import json, os

# ── Load Data ───────────────────────────────────────────────────────────────
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

labels_path = os.path.join(REPO, "outputs/model", "labels.json")
words_path = os.path.join(REPO, "outputs", "topics", "top_words_final.csv")
dist_path = os.path.join(REPO, "outputs", "topics", "topic_distribution_final.csv")

with open(labels_path, 'r', encoding='utf-8') as f:
    labels = json.load(f)

df_words = pd.read_csv(words_path)
df_dist = pd.read_csv(dist_path)

label_dict = {t['topic_id']: t['label'] for t in labels}
select_options = [f"Topik {t['topic_id']} — {t['label']}" for t in labels]

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("<h1>Detail Topik</h1>", unsafe_allow_html=True)

# ── Selectbox ───────────────────────────────────────────────────────────────
selected_option = st.selectbox("Pilih topik", options=select_options)
topic_id = int(selected_option.split()[1])

# Data topik terpilih
lbl_obj = labels[topic_id - 1]
words_df = df_words[df_words['topik_id'] == topic_id].head(15)

topic_docs_all = df_dist[df_dist['topik_dominan'] == topic_id]
docs = topic_docs_all.sort_values('prob_dominan', ascending=False).head(5)

doc_count = len(topic_docs_all)
avg_confidence = topic_docs_all['prob_dominan'].mean() if doc_count > 0 else 0.0

# ── BARIS ATAS — tiga kolom (40/30/30) ──────────────────────────────────────
col1, col2, col3 = st.columns([40, 30, 30])

with col1:
    st.markdown(f"<h3 style='color: #1A56A0; margin-bottom: 0;'>Topik {topic_id}: {lbl_obj['label']}</h3>", unsafe_allow_html=True)
    st.write(lbl_obj['deskripsi'])
    
    kepercayaan = lbl_obj.get('tingkat_kepercayaan', 'Sedang')
    if kepercayaan in ['Tinggi', 'Sedang']:
        badge_style = "background-color: #DCFCE7; color: #166534; padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 0.85rem;"
    else:
        badge_style = "background-color: #FEE2E2; color: #991B1B; padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 0.85rem;"
    st.markdown(f"<span style='{badge_style}'>Kepercayaan Label: {kepercayaan}</span>", unsafe_allow_html=True)

with col2:
    st.metric("Dokumen", f"{doc_count}", help="dokumen dalam topik ini")

with col3:
    st.metric("Rata-rata Probabilitas", f"{avg_confidence:.1%}", help="keyakinan penugasan")

if topic_id in [2, 8]:
    st.warning(
        "Topik ini bersifat heterogen. Dokumen di dalamnya berasal dari "
        "berbagai metodologi yang tidak membentuk tema tunggal. "
        "Gunakan label ini dengan hati-hati."
    )

st.divider()

# ── DUA KOLOM UTAMA (45/55) ──────────────────────────────────────────────────
col_left, col_right = st.columns([45, 55])

with col_left:
    st.subheader("Peta Kata Kunci")
    
    # Generate Wordcloud
    freq = dict(zip(words_df['word'], words_df['probability']))
    freq_int = {w: int(round(p * 1000)) for w, p in freq.items()}
    
    wc = WordCloud(
        width=400,
        height=280,
        background_color='white',
        colormap='Blues',
        max_words=15,
        prefer_horizontal=0.85,
        min_font_size=10,
        max_font_size=80,
        relative_scaling=0.6,
        collocations=False
    ).generate_from_frequencies(freq_int)
    
    fig_wc, ax = plt.subplots(figsize=(4, 2.8))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout(pad=0)
    st.pyplot(fig_wc, use_container_width=True)
    plt.close(fig_wc)
    
    st.caption("Ukuran kata proporsional terhadap probabilitas dalam topik.")
    
    st.markdown("#### 15 Kata Kunci Teratas")
    
    # Horizontal Bar Chart
    df_plot = words_df.sort_values('probability', ascending=True)
    fig_bar = px.bar(
        df_plot,
        x='probability',
        y='word',
        orientation='h',
        color_discrete_sequence=['#1A56A0']
    )
    fig_bar.update_layout(
        height=350,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title=None,
        yaxis_title=None,
        showlegend=False,
        template='plotly_white'
    )
    fig_bar.update_xaxes(showgrid=False, tickformat='.4f')
    fig_bar.update_yaxes(showgrid=False)
    st.plotly_chart(fig_bar, use_container_width=True)

with col_right:
    st.subheader("5 Dokumen Paling Representatif")
    st.caption("Diurutkan berdasarkan probabilitas penugasan topik tertinggi.")
    
    for idx, row in docs.iterrows():
        with st.container(border=True):
            col_a, col_b = st.columns([1, 4])
            with col_a:
                tahun_val = row['Tahun'] if 'Tahun' in row.index else row.get('tahun', '-')
                st.markdown(f"**{tahun_val}**")
            with col_b:
                st.markdown(f"`{row['prob_dominan']:.4f}`")
            judul = row['Judul'] if 'Judul' in row.index else row.get('judul', '-')
            st.markdown(f"**{judul}**")
