import streamlit as st
import pandas as pd
import json
import os
import plotly.graph_objects as go
import plotly.express as px
import artifacts

# Metrik resmi dibaca dari artefak (bukan hard-code)
M = artifacts.load_metrics()

# ── Load data resmi ─────────────────────────────────────────────────────────
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

tby = pd.read_csv(os.path.join(REPO, "_fix_final", "topic_by_year_final.csv"))
intervals = pd.read_csv(os.path.join(REPO, "_stats_forecast", "share_intervals.csv"))

with open(os.path.join(REPO, "_label_work", "labels.json"), "r", encoding="utf-8") as f:
    raw_labels = json.load(f)

labels_map = {item["label"]: int(item["topic_id"]) for item in raw_labels}
labels_list = [item["label"] for item in raw_labels]

st.markdown("<h1>Tren Komposisi Topik 2021–2025</h1>", unsafe_allow_html=True)

selected_labels = st.multiselect(
    "Pilih topik yang ditampilkan",
    options=labels_list,
    default=labels_list,
)

if selected_labels:
    fig = go.Figure()
    years = [2021, 2022, 2023, 2024, 2025]
    years_str = ["2021", "2022", "2023", "2024", "2025"]

    # Distinct vibrant qualitative colors for high legibility
    colors_8 = [
        "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728",
        "#9467BD", "#8C564B", "#E377C2", "#17BECF"
    ]

    def hex_to_rgba(hex_str, alpha=0.12):
        hex_str = hex_str.lstrip("#")
        r, g, b = tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
        return f"rgba({r}, {g}, {b}, {alpha})"

    for label in selected_labels:
        tid = labels_map[label]
        color = colors_8[(tid - 1) % 8]
        fill_rgba = hex_to_rgba(color, 0.12)
        col_prop = f"proporsi_topik_{tid}"

        if col_prop not in tby.columns:
            continue

        y_vals = tby[col_prop].tolist()
        ti = intervals[intervals["topik_id"] == tid]

        y_upper, y_lower = [], []
        for yr in years:
            r = ti[ti["tahun"] == yr]
            if len(r) > 0:
                y_upper.append(float(r["ci_upper_pct"].iloc[0]))
                y_lower.append(float(r["ci_lower_pct"].iloc[0]))
            else:
                y_upper.append(None)
                y_lower.append(None)

        # Area shading Wilson 95% CI (fill='tonexty')
        if all(v is not None for v in y_upper + y_lower):
            fig.add_trace(go.Scatter(
                x=years_str, y=y_lower,
                mode="lines",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=years_str, y=y_upper,
                mode="lines",
                fill="tonexty",
                fillcolor=fill_rgba,
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
            ))

        # Main Solid Line
        fig.add_trace(go.Scatter(
            x=years_str, y=y_vals,
            name=label,
            mode="lines+markers",
            line=dict(color=color, width=2.5),
            marker=dict(size=7, color=color),
            hovertemplate=f"<b>{label}</b><br>Tahun: %{{x}}<br>Proporsi: %{{y:.2f}}%<extra></extra>",
        ))

    fig.update_layout(
        height=500,
        margin=dict(l=10, r=10, t=20, b=120),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(size=11)
        ),
        template="plotly_white",
    )
    fig.update_xaxes(showgrid=False, type="category")
    fig.update_yaxes(
        title="Proporsi (%)",
        range=[0, 30],
        showgrid=True,
        gridcolor="#E5E7EB",
        zerolinecolor="#E5E7EB",
    )

    st.plotly_chart(fig, use_container_width=True)

st.info(
    f"Uji homogenitas chi-square: p = {artifacts.idr(M['chi2_p'], 2)} — tidak ditemukan perubahan komposisi "
    "yang signifikan antar tahun. Tidak ada topik dengan tren yang terkonfirmasi "
    "secara statistik (Spearman + koreksi Bonferroni, semua p > 0,05). Fluktuasi "
    "yang terlihat kemungkinan merupakan variasi acak dari jumlah dokumen per tahun "
    "yang tidak merata (29–85 dokumen)."
)
