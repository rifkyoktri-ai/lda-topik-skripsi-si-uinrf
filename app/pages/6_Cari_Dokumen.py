import streamlit as st
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import artifacts
import chart_helpers

data = artifacts.load_all()
dataset = data["distribution"].copy()
labels_dict = data["labels_dict"]

chart_helpers.set_style()

st.markdown('<h1 class="main-title">🔍 Cari Dokumen Skripsi</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Pencarian interaktif berdasarkan kata kunci judul, topik, atau tahun lulus.</p>', unsafe_allow_html=True)

topic_options = {tid: f"Topik {tid}: {labels_dict[tid]['label']}" for tid in range(1, 9)}

col1, col2 = st.columns([3, 1])

with col1:
    query = st.text_input("Masukkan Kata Kunci Judul / Abstrak:", "")

with col2:
    selected_topic_label = st.selectbox("Filter Topik:", ["Semua"] + list(topic_options.values()))

filtered = dataset.copy()

# Tentukan nama kolom judul & abstrak (handling huruf kapital)
col_judul = "Judul" if "Judul" in filtered.columns else "judul"
col_abstrak = "Abstrak" if "Abstrak" in filtered.columns else "abstrak"
col_topik = "Topik Dominan" if "Topik Dominan" in filtered.columns else "topik_dominan"

if query.strip():
    filtered = filtered[
        filtered[col_judul].astype(str).str.contains(query, case=False, na=False) |
        filtered[col_abstrak].astype(str).str.contains(query, case=False, na=False)
    ]

if selected_topic_label != "Semua":
    selected_tid = [k for k, v in topic_options.items() if v == selected_topic_label][0]
    filtered = filtered[filtered[col_topik] == selected_tid]

st.markdown(f"**Menampilkan {len(filtered)} dari {len(dataset)} Dokumen**")
st.dataframe(filtered, use_container_width=True)
