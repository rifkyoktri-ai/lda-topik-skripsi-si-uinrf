import streamlit as st
import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import chart_helpers

st.set_page_config(
    page_title="Dashboard Skripsi SI UIN RF",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Terapkan styling global secara bersih tanpa bocor teks CSS
chart_helpers.set_style()

# ── Navigasi Bersih ─────────────────────────────────────────────────────────
pages = [
    st.Page("pages/1_Beranda.py", title="Beranda", icon="🏠", default=True),
    st.Page("pages/2_Distribusi_Topik.py", title="Distribusi Topik", icon="📊"),
    st.Page("pages/3_Tren_Per_Tahun.py", title="Tren Per Tahun", icon="📈"),
    st.Page("pages/4_Proyeksi.py", title="Proyeksi 2026–2027", icon="🔮"),
    st.Page("pages/5_Detail_Topik.py", title="Detail Topik", icon="🔍"),
    st.Page("pages/6_Cari_Dokumen.py", title="Cari Dokumen", icon="🔎"),
    st.Page("pages/7_Metodologi.py", title="Metodologi", icon="📐"),
    st.Page("pages/8_Evaluasi_Model.py", title="Evaluasi Model", icon="🧪"),
]

pg = st.navigation(pages)
pg.run()
