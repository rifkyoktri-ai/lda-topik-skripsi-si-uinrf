import streamlit as st
import pandas as pd
import json
import os
import plotly.graph_objects as go
import artifacts

# Metrik resmi dibaca dari artefak (bukan hard-code) - lihat artifacts.load_metrics()
M = artifacts.load_metrics()

# ── Load data resmi ─────────────────────────────────────────────────────────
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
counts_df = pd.read_csv(os.path.join(REPO, "outputs", "topics", "topic_counts_final.csv"))

with open(os.path.join(REPO, "outputs/model", "labels.json"), "r", encoding="utf-8") as f:
    raw_labels = json.load(f)

labels_dict = {int(item["topic_id"]): item["label"] for item in raw_labels}

# ── Nilai turunan (dihitung, bukan diketik) ─────────────────────────────────
_dist = pd.read_csv(os.path.join(REPO, "outputs", "topics", "topic_distribution_final.csv"))
TH_AWAL, TH_AKHIR = int(_dist["Tahun"].min()), int(_dist["Tahun"].max())
RENTANG = f"{TH_AWAL}\u2013{TH_AKHIR}"
N_DOC = artifacts.idr(artifacts.metrik(M, "Total_Documents"), 0)
N_TOPIK = M["k"]
CV_TAMPIL = artifacts.idr(artifacts.metrik(M, "Coherence_Score_CV"), 4)

# Dua topik terbesar beserta label dan persentasenya
_atas = counts_df.nlargest(2, "persentase")
_besar = [
    (int(r["topik_dominan"]),
     labels_dict[int(r["topik_dominan"])],
     artifacts.idr(r["persentase"], 1))
    for _, r in _atas.iterrows()
]
(T1_ID, T1_LBL, T1_PCT), (T2_ID, T2_LBL, T2_PCT) = _besar

# ── Header Layout 60/40 ──────────────────────────────────────────────────────
col_txt, col_met = st.columns([6, 4])

with col_txt:
    st.markdown("<h1>Pemetaan Topik Penelitian</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#9CA3AF; font-size:1.05rem;'>"
        f"Program Studi Sistem Informasi · UIN Raden Fatah Palembang · {RENTANG}"
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"Analisis terhadap **{N_DOC} skripsi** menggunakan Latent Dirichlet Allocation (LDA) "
        f"mengidentifikasi **{N_TOPIK} topik** penelitian. Tidak ditemukan pergeseran komposisi yang "
        f"signifikan secara statistik selama {RENTANG} *(chi-square p = "
        f"{artifacts.idr(artifacts.metrik(M, 'chi2_p', 0.0), 2)})*. Topik terbesar "
        f"adalah **{T1_LBL}** ({T1_PCT}%) dan "
        f"**{T2_LBL}** ({T2_PCT}%)."
    )

with col_met:
    m1, m2 = st.columns(2)
    m1.metric("Skripsi dianalisis", N_DOC)
    m2.metric("Topik teridentifikasi", str(N_TOPIK))
    m1.metric("Koherensi model (C_V)", CV_TAMPIL)
    m2.metric("Rentang data", RENTANG)

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
    f"Model: LDA K={N_TOPIK}, Online Variational Bayes · "
    f"C_V = {CV_TAMPIL} (texts=v1, topn=20, processes=1)"
)
