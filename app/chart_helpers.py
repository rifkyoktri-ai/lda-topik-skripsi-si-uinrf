"""
chart_helpers.py — Semua fungsi chart Plotly + set_style CSS global.
Dark mode, 8 warna kontras per topik.
"""
import streamlit as st
import plotly.graph_objects as go

# ── Palet warna 8 topik ──────────────────────────────────────────────────────
TOPIC_COLORS = [
    "#3B82F6",  # Topik 1: Royal Blue
    "#FF8C00",  # Topik 2: Dark Orange
    "#A855F7",  # Topik 3: Vivid Purple
    "#10B981",  # Topik 4: Emerald Green
    "#EC4899",  # Topik 5: Hot Pink
    "#06B6D4",  # Topik 6: Cyan
    "#EAB308",  # Topik 7: Golden Yellow
    "#F43F5E",  # Topik 8: Crimson Rose
]

BG_DARK    = "#0B0F19"
CARD_BG    = "#1F2937"
BORDER     = "#374151"
TEXT       = "#F3F4F6"
MUTED      = "#9CA3AF"
PRIMARY    = "#3B82F6"
GRID_COLOR = "#374151"

MARGIN = dict(l=10, r=10, t=35, b=10)


def set_style():
    st.markdown(
        """
        <style>
        html, body, .stApp {
            background-color: #0B0F19 !important;
            color: #F3F4F6 !important;
            font-family: 'Inter', -apple-system, sans-serif !important;
        }
        .main .block-container {
            padding: 2rem 1.5rem;
            background-color: #0B0F19;
        }
        header[data-testid="stHeader"] {
            background-color: rgba(11,15,25,0.85) !important;
            backdrop-filter: blur(8px);
        }
        section[data-testid="stSidebar"] {
            background-color: #111827 !important;
            border-right: 1px solid #374151 !important;
        }
        .main-title {
            color: #FFFFFF !important;
            font-size: 2rem !important;
            font-weight: 700 !important;
            margin-bottom: 0.2rem !important;
        }
        .sub-title {
            color: #9CA3AF !important;
            font-size: 1rem !important;
            margin-bottom: 1.5rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def _topic_color(topic_id: int) -> str:
    return TOPIC_COLORS[(int(topic_id) - 1) % len(TOPIC_COLORS)]


def _base_layout(fig, height: int = None):
    kw = dict(
        template="plotly_dark",
        paper_bgcolor=BG_DARK,
        plot_bgcolor=CARD_BG,
        font=dict(family="Inter, sans-serif", color=TEXT, size=12),
        margin=MARGIN,
        legend=dict(
            bgcolor="rgba(31,41,55,0.85)",
            bordercolor=BORDER,
            borderwidth=1,
            font=dict(color=TEXT, size=11),
        ),
    )
    if height:
        kw["height"] = height
    fig.update_layout(**kw)
    fig.update_xaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, tickfont=dict(color=MUTED))
    return fig


# ── Distribusi topik: bar horizontal ─────────────────────────────────────────
def make_distribution_bar(df_counts, labels_dict: dict):
    df = df_counts.sort_values("jumlah_dokumen", ascending=True).copy()
    y_labels, colors = [], []
    for _, row in df.iterrows():
        tid = int(row["topik_dominan"])
        lbl = labels_dict[tid]["label"]
        y_labels.append(lbl[:35] + ("…" if len(lbl) > 35 else ""))
        colors.append(_topic_color(tid))

    fig = go.Figure(go.Bar(
        x=df["jumlah_dokumen"].tolist(),
        y=y_labels,
        orientation="h",
        marker=dict(color=colors, line=dict(color=BORDER, width=0.5)),
        text=[f"  {p:.1f}%" for p in df["persentase"]],
        textposition="outside",
        textfont=dict(color=MUTED, size=11),
        hovertemplate="<b>%{y}</b><br>Jumlah: %{x} dokumen<extra></extra>",
    ))
    fig.update_xaxes(visible=False, showgrid=False)
    fig.update_yaxes(showgrid=False, tickfont=dict(color=TEXT, size=11))
    return _base_layout(fig, height=320)


# ── Donut chart distribusi ────────────────────────────────────────────────────
def make_donut_chart(df_counts, labels_dict: dict):
    df = df_counts.sort_values("topik_dominan").copy()
    labels_list = [labels_dict[int(r["topik_dominan"])]["label"][:28] for _, r in df.iterrows()]
    colors = [_topic_color(int(r["topik_dominan"])) for _, r in df.iterrows()]

    fig = go.Figure(go.Pie(
        labels=labels_list,
        values=df["jumlah_dokumen"].tolist(),
        hole=0.5,
        marker=dict(colors=colors, line=dict(color=BG_DARK, width=2)),
        textinfo="percent",
        textfont=dict(color=TEXT, size=11),
        hovertemplate="<b>%{label}</b><br>%{value} dokumen (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="v", x=1.02, y=0.5,
            font=dict(color=TEXT, size=10),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    return _base_layout(fig, height=340)


# ── Tren historis + Wilson CI ─────────────────────────────────────────────────
def make_trend_chart(df_by_year, df_intervals, selected_topics: list, labels_dict: dict):
    fig = go.Figure()
    year_col = "Tahun" if "Tahun" in df_by_year.columns else "tahun"
    years = [int(y) for y in df_by_year[year_col].tolist()]

    for topic_id in selected_topics:
        tid   = int(topic_id)
        col   = f"proporsi_topik_{tid}"
        color = _topic_color(tid)
        label = labels_dict[tid]["label"]

        if col not in df_by_year.columns:
            continue

        y_data = df_by_year[col].tolist()

        # Garis utama
        fig.add_trace(go.Scatter(
            x=years, y=y_data,
            name=label,
            mode="lines+markers",
            line=dict(color=color, width=2.5),
            marker=dict(size=7, color=color, line=dict(color=BG_DARK, width=1.5)),
            hovertemplate=f"<b>{label}</b><br>Tahun: %{{x}}<br>Proporsi: %{{y:.1f}}%<extra></extra>",
        ))

    fig.update_xaxes(
        showgrid=False,
        dtick=1,
        range=[2020.5, 2025.5],
        tickvals=[2021, 2022, 2023, 2024, 2025]
    )
    fig.update_yaxes(showgrid=True, gridcolor=BORDER, range=[0, 32], title="Proporsi (%)")
    fig.update_layout(legend=dict(orientation="h", y=-0.25, x=0))
    return _base_layout(fig, height=480)


# ── Forecast: historis solid + proyeksi putus-putus ───────────────────────────
def make_forecast_chart(df_by_year, df_forecast, selected_topics: list, labels_dict: dict):
    fig = go.Figure()

    for topic_id in selected_topics:
        tid   = int(topic_id)
        col   = f"proporsi_topik_{tid}"
        color = _topic_color(tid)
        label = labels_dict[tid]["label"]

        if col not in df_by_year.columns:
            continue

        y_hist = df_by_year[col].tolist()
        rows   = df_forecast[df_forecast["topik_id"] == tid]
        if len(rows) == 0:
            continue
        r = rows.iloc[0]

        # Historis: solid (2021-2025)
        fig.add_trace(go.Scatter(
            x=[2021, 2022, 2023, 2024, 2025], y=y_hist,
            name=label, mode="lines+markers",
            line=dict(color=color, width=2.5),
            marker=dict(size=7, color=color, line=dict(color=BG_DARK, width=1.5)),
        ))

        # Proyeksi: putus-putus, sambung dari 2025 (2025-2027)
        fig.add_trace(go.Scatter(
            x=[2025, 2026, 2027],
            y=[y_hist[-1], float(r["wma_2026"]), float(r["wma_2027"])],
            showlegend=False, mode="lines+markers",
            line=dict(color=color, width=2, dash="dot"),
            marker=dict(size=9, symbol="diamond", color=color,
                        line=dict(color=BG_DARK, width=1.5)),
            opacity=0.75,
        ))

        # Error bar 2026
        err_low  = float(r["wma_2026"]) - float(r["pita_bawah_2026"])
        err_high = float(r["pita_atas_2026"]) - float(r["wma_2026"])
        fig.add_trace(go.Scatter(
            x=[2026], y=[float(r["wma_2026"])],
            showlegend=False, mode="markers",
            marker=dict(color=color, size=1, opacity=0),
            error_y=dict(type="data", array=[err_high], arrayminus=[err_low],
                         color=color, thickness=2, width=6),
        ))

    # Garis pemisah historis/proyeksi
    fig.add_shape(
        type="line", xref="x", yref="paper",
        x0=2025, y0=0, x1=2025, y1=1,
        line=dict(color=MUTED, width=1, dash="dot"),
    )
    fig.add_annotation(x=2024, y=0.97, xref="x", yref="paper",
                       text="◀ Historis", showarrow=False,
                       font=dict(color=MUTED, size=10))
    fig.add_annotation(x=2026, y=0.97, xref="x", yref="paper",
                       text="Proyeksi ▶", showarrow=False,
                       font=dict(color=MUTED, size=10))

    fig.update_xaxes(
        showgrid=False,
        dtick=1,
        range=[2020.5, 2027.5],
        tickvals=[2021, 2022, 2023, 2024, 2025, 2026, 2027]
    )
    fig.update_yaxes(showgrid=True, gridcolor=BORDER, range=[0, 32])
    fig.update_layout(legend=dict(orientation="h", y=-0.25, x=0))
    return _base_layout(fig, height=500)


# ── Bar horizontal kata kunci topik ──────────────────────────────────────────
def make_topwords_chart(words: list, probs: list, topic_id: int = 1):
    words  = list(words)
    probs  = [float(p) for p in probs]
    color  = _topic_color(topic_id)
    max_p  = max(probs) if probs else 1
    opacs  = [0.4 + 0.6 * (p / max_p) for p in probs]

    # Ubah hex ke rgba
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    bar_colors = [f"rgba({r},{g},{b},{o:.2f})" for o in opacs]

    fig = go.Figure(go.Bar(
        x=probs, y=words,
        orientation="h",
        marker=dict(color=bar_colors, line=dict(color=BORDER, width=0.5)),
        text=[f"{p:.4f}" for p in probs],
        textposition="outside",
        textfont=dict(family="monospace", size=10, color=MUTED),
        hovertemplate="<b>%{y}</b>: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        yaxis=dict(autorange="reversed",
                   tickfont=dict(family="monospace", size=11, color=TEXT)),
        xaxis=dict(visible=False, showgrid=False),
    )
    return _base_layout(fig, height=400)
