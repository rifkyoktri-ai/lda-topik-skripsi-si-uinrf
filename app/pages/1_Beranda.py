import streamlit as st
import pandas as pd
import json
import os
import plotly.graph_objects as go

# ── Load data resmi ─────────────────────────────────────────────────────────
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
counts_df = pd.read_csv(os.path.join(REPO, "_fix_final", "topic_counts_final.csv"))

with open(os.path.join(REPO, "_label_work", "labels.json"), "r", encoding="utf-8") as f:
    raw_labels = json.load(f)

labels_dict = {int(item["topic_id"]): item["label"] for item in raw_labels}

# ── Header Layout 60/40 ──────────────────────────────────────────────────────
col_txt, col_met = st.columns([6, 4])

with col_txt:
    st.markdown("<h1>Pemetaan Topik Penelitian</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#9CA3AF; font-size:1.05rem;'>"
        "Program Studi Sistem Informasi · UIN Raden Fatah Palembang · 2021–2025"
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "Analisis terhadap **285 skripsi** menggunakan Latent Dirichlet Allocation (LDA) "
        "mengidentifikasi **8 topik** penelitian. Tidak ditemukan pergeseran komposisi yang "
        "signifikan secara statistik selama 2021–2025 *(chi-square p = 0,69)*. Topik terbesar "
        "adalah **Pengembangan & Perencanaan Strategis SI** (17,2%) dan "
        "**Analisis Kualitas Layanan & Risiko SI** (16,1%)."
    )

with col_met:
    m1, m2 = st.columns(2)
    m1.metric("Skripsi dianalisis", "285")
    m2.metric("Topik teridentifikasi", "8")
    m1.metric("Koherensi model (C_V)", "0,4256")
    m2.metric("Rentang data", "2021–2025")

st.markdown("<hr>", unsafe_allow_html=True)

# ── Bar Chart Horizontal Plotly ──────────────────────────────────────────────
df = counts_df.sort_values("persentase", ascending=True).copy()
y_labels = []
for _, row in df.iterrows():
    lbl = labels_dict[int(row["topik_dominan"])]
    y_labels.append(lbl[:35] + ("…" if len(lbl) > 35 else ""))

fig = go.Figure(go.Bar(
    x=df["persentase"].tolist(),
    y=y_labels,
    orientation="h",
    marker=dict(color="#1A56A0", opacity=0.75),
    text=[f"{p:.1f}%" for p in df["persentase"]],
    textposition="outside",
    hovertemplate="<b>%{y}</b><br>Porsi: %{x:.1f}%<extra></extra>",
))

fig.update_layout(
    title="Distribusi Topik Keseluruhan",
    height=320,
    margin=dict(l=10, r=40, t=40, b=10),
    showlegend=False,
    xaxis=dict(showgrid=False, visible=False),
    yaxis=dict(showgrid=False, tickfont=dict(size=11)),
    template="plotly_white",
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Data cutoff: September 2026 · Cakupan scraping 2025 ~82% · "
    "Model: LDA K=8, Online Variational Bayes · C_V = 0,4256 (texts=v1, topn=20, processes=1)"
)
