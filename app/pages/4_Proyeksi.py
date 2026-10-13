import streamlit as st
import pandas as pd
import json
import os
import plotly.graph_objects as go
import plotly.express as px

# ── Load data resmi ─────────────────────────────────────────────────────────
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

tby = pd.read_csv(os.path.join(REPO, "outputs", "topics", "topic_by_year_final.csv"))
forecast_df = pd.read_csv(os.path.join(REPO, "outputs/forecast", "forecast_table.csv"))

with open(os.path.join(REPO, "outputs/model", "labels.json"), "r", encoding="utf-8") as f:
    raw_labels = json.load(f)

labels_map = {item["label"]: int(item["topic_id"]) for item in raw_labels}
labels_by_id = {int(item["topic_id"]): item["label"] for item in raw_labels}
labels_list = [item["label"] for item in raw_labels]

st.markdown("<h1>Proyeksi Komposisi Topik 2026–2027</h1>", unsafe_allow_html=True)
st.caption("Weighted Moving Average · Window 5 tahun · Indikatif")

st.warning(
    "Proyeksi ini dihasilkan menggunakan Weighted Moving Average (WMA) dengan 5 titik data tahunan. "
    "WMA tidak terbukti lebih akurat dari rata-rata historis untuk 7 dari 8 topik (walk-forward backtest). "
    "Angka 2026–2027 adalah gambaran indikatif, bukan prediksi yang dapat diandalkan secara statistik."
)

default_topics = [labels_by_id[i] for i in [3, 4, 5, 6] if i in labels_by_id]

selected_labels = st.multiselect(
    "Pilih topik proyeksi:",
    options=labels_list,
    default=default_topics,
)

if selected_labels:
    fig = go.Figure()

    # Color palette: 8 colors from Reversed Blues
    palette = px.colors.sequential.Blues_r
    color_indices = [int(i * (len(palette) - 1) / 7) for i in range(8)]
    colors_8 = [palette[idx] for idx in color_indices]

    hist_years = ["2021", "2022", "2023", "2024", "2025"]
    proj_years = ["2025", "2026", "2027"]

    for label in selected_labels:
        tid = labels_map[label]
        color = colors_8[(tid - 1) % 8]
        col_prop = f"proporsi_topik_{tid}"

        if col_prop not in tby.columns:
            continue

        y_hist = tby[col_prop].tolist()
        r = forecast_df[forecast_df["topik_id"] == tid]
        if len(r) == 0:
            continue
        row = r.iloc[0]

        wma_26 = float(row["wma_2026"])
        wma_27 = float(row["wma_2027"])
        pita_bawah_26 = float(row["pita_bawah_2026"])
        pita_atas_26 = float(row["pita_atas_2026"])

        # 1. HISTORIS (2021-2025)
        fig.add_trace(go.Scatter(
            x=hist_years, y=y_hist,
            name=label,
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=7, symbol="circle"),
            opacity=1.0,
        ))

        # 2. PROYEKSI (2025-2027)
        fig.add_trace(go.Scatter(
            x=proj_years,
            y=[y_hist[-1], wma_26, wma_27],
            name=f"{label} (proyeksi)",
            mode="lines+markers",
            line=dict(color=color, width=2, dash="dot"),
            marker=dict(size=8, symbol="diamond"),
            opacity=0.7,
            showlegend=False,
        ))

        # 3. ERROR BAR 2026
        err_plus = pita_atas_26 - wma_26
        err_minus = wma_26 - pita_bawah_26

        fig.add_trace(go.Scatter(
            x=["2026"],
            y=[wma_26],
            mode="markers",
            marker=dict(size=1, color=color, opacity=0),
            error_y=dict(
                type="data",
                symmetric=False,
                array=[err_plus],
                arrayminus=[err_minus],
                visible=True,
                thickness=2,
                width=6,
                color=color,
            ),
            opacity=0.5,
            showlegend=False,
            hoverinfo="skip",
        ))

    # Garis vertikal putus-putus antara 2025 dan 2026 (x=4)
    fig.add_vline(x=4, line_dash="dash", line_color="#9CA3AF", line_width=1)

    fig.add_annotation(
        x=2.0, y=29,
        text="Historis", showarrow=False,
        font=dict(color="#6B7280", size=11),
    )
    fig.add_annotation(
        x=5.5, y=29,
        text="Proyeksi", showarrow=False,
        font=dict(color="#6B7280", size=11),
    )

    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=20, b=80),
        legend=dict(orientation="h", y=-0.25, x=0),
        template="plotly_white",
    )
    fig.update_xaxes(
        showgrid=False,
        type="category",
        categoryorder="array",
        categoryarray=["2021", "2022", "2023", "2024", "2025", "2026", "2027"],
    )
    fig.update_yaxes(
        title="Proporsi (%)",
        range=[0, 30],
        showgrid=True,
        gridcolor="#E5E7EB",
        zerolinecolor="#E5E7EB",
    )

    st.plotly_chart(fig, use_container_width=True)

# ── Tabel di bawah chart ─────────────────────────────────────────────────────
table_df = forecast_df.copy()

# Rename columns
table_df = table_df.rename(columns={
    "label": "Topik",
    "share_2025": "Porsi 2025 (%)",
    "wma_2026": "WMA 2026 (%)",
    "wma_2027": "WMA 2027 (%)",
    "arah": "Arah",
    "keandalan": "Keandalan",
})

# Format 2 desimal
for col in ["Porsi 2025 (%)", "WMA 2026 (%)", "WMA 2027 (%)"]:
    table_df[col] = table_df[col].apply(lambda x: f"{float(x):.2f}")

st.dataframe(
    table_df[["Topik", "Porsi 2025 (%)", "WMA 2026 (%)", "WMA 2027 (%)", "Arah", "Keandalan"]],
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "Arah ditentukan dari perbandingan statistik, bukan selisih angka mentah. "
    "Seluruh topik menunjukkan komposisi yang tidak berubah bermakna pada level kepercayaan 95%."
)
