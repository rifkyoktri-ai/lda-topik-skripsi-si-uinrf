import streamlit as st
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import artifacts
import chart_helpers

data = artifacts.load_all()
counts = data["counts"].copy()
labels = data["labels_dict"]

chart_helpers.set_style()

st.markdown('<h1 class="main-title">📊 Distribusi Topik Penelitian</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Sebaran dokumen skripsi berdasarkan topik utama (2021-2025).</p>', unsafe_allow_html=True)

# ── Tabel utama ──────────────────────────────────────────────────────────────
counts_display = counts.copy()
counts_display["No"]          = counts_display["topik_dominan"]
counts_display["Topik"]       = counts_display["topik_dominan"].apply(lambda x: labels[int(x)]["label"])
counts_display["Kepercayaan"] = counts_display["topik_dominan"].apply(
    lambda x: labels[int(x)].get("tingkat_kepercayaan", "Sedang")
)
counts_display["Catatan"] = counts_display["topik_dominan"].apply(
    lambda x: "Heterogen — keterbatasan model" if int(x) in [2, 8] else ""
)

st.dataframe(
    counts_display[["No", "Topik", "jumlah_dokumen", "persentase", "Kepercayaan", "Catatan"]],
    column_config={
        "No":             st.column_config.NumberColumn("No"),
        "jumlah_dokumen": st.column_config.NumberColumn("Jumlah"),
        "persentase":     st.column_config.ProgressColumn(
            "Porsi", format="%.1f%%", min_value=0, max_value=20
        ),
    },
    hide_index=True,
    use_container_width=True,
)

# ── Chart + Deskripsi ────────────────────────────────────────────────────────
col_pie, col_desc = st.columns([1, 1])

with col_pie:
    st.markdown("#### Proporsi Topik")
    fig = chart_helpers.make_donut_chart(counts, labels)
    st.plotly_chart(fig, use_container_width=True)

with col_desc:
    st.markdown("#### Deskripsi Singkat Tiap Topik")
    for tid in range(1, 9):
        lbl  = labels[tid]["label"]
        desc = labels[tid].get("deskripsi", "(Tidak ada deskripsi)")
        kep  = labels[tid].get("tingkat_kepercayaan", "Sedang")
        with st.expander(f"Topik {tid} — {lbl}"):
            badge_color = "#10B981" if kep == "Tinggi" else ("#EAB308" if kep == "Sedang" else "#EF4444")
            st.markdown(
                f"<span style='background:{badge_color};color:#000;padding:2px 8px;"
                f"border-radius:4px;font-size:0.78rem;font-weight:600'>{kep}</span>",
                unsafe_allow_html=True,
            )
            st.write(desc)

# ── Kotak peringatan ─────────────────────────────────────────────────────────
st.markdown(
    "<div style='background:#1e2d3d;border-left:4px solid #3B82F6;"
    "padding:1rem 1.2rem;border-radius:6px;margin-top:1.2rem;'>"
    "<strong style='color:#F3F4F6'>Catatan Keterbatasan Model</strong><br>"
    "<span style='color:#9CA3AF'>Topik 2 dan Topik 8 memiliki heterogenitas tinggi — "
    "dokumen di dalamnya beragam dan tidak membentuk tema tunggal. "
    "Ini merupakan keterbatasan model LDA pada korpus dengan variasi metodologi yang luas.</span>"
    "</div>",
    unsafe_allow_html=True,
)
