import streamlit as st
import pandas as pd
import json
import os
import hashlib
import artifacts

# Metrik resmi dibaca dari artefak (bukan hard-code)
M = artifacts.load_metrics()

# ── Paths Data Resmi ────────────────────────────────────────────────────────
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

forecast_path = os.path.join(REPO, "outputs/forecast", "forecast_table.csv")
labels_path = os.path.join(REPO, "outputs/model", "labels.json")
eval_path = os.path.join(REPO, "outputs", "model", "evaluation_summary.csv")

# SHA256 forecast_table.csv untuk footer
sha256_hash = ""
if os.path.exists(forecast_path):
    with open(forecast_path, "rb") as f:
        sha256_hash = hashlib.sha256(f.read()).hexdigest()[:8]

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("<h1>Metodologi Penelitian</h1>", unsafe_allow_html=True)

# ── BAGIAN 1 — Kerangka Kerja ───────────────────────────────────────────────
st.subheader("Pendekatan CRISP-DM")

crisp_dm_data = {
    "Fase": [
        "Business Understanding",
        "Data Understanding",
        "Data Preparation",
        "Modeling",
        "Evaluation",
        "Deployment",
    ],
    "Implementasi": [
        "Identifikasi pola topik riset Prodi SI",
        f"{N_DOC_TAMPIL} abstrak skripsi {RENTANG_TAHUN}",
        "Tokenisasi, stemming Sastrawi, filter stopword bahasa Indonesia + domain SI",
        "LDA Gensim, grid search K=4-10",
        "Koherensi C_V, walk-forward backtest MAE, uji komposisi chi-square",
        "Dashboard Streamlit 8 halaman",
    ],
}
st.dataframe(pd.DataFrame(crisp_dm_data), use_container_width=True, hide_index=True)

# ── BAGIAN 2 — Parameter Model ───────────────────────────────────────────────
st.subheader("Konfigurasi Model Final")

param_data = {
    "Parameter": [
        "Algoritma",
        "Implementasi",
        "Jumlah Topik (K)",
        "Alpha",
        "Eta",
        "Passes",
        "Iterations",
        "Random State",
        "Korpus",
    ],
    "Nilai": [
        "Latent Dirichlet Allocation (LDA)",
        "Gensim 4.4.0, Online Variational Bayes (bukan Gibbs Sampling)",
        str(M.get("k", 8)),
        str(M.get("alpha", "auto")).capitalize(),
        str(M.get("eta", "auto")).capitalize(),
        "30",
        "500",
        "42",
        f"{N_DOC_TAMPIL} dokumen ({RENTANG_TAHUN})",
    ],
}
st.dataframe(pd.DataFrame(param_data), use_container_width=True, hide_index=True)

# ── BAGIAN 3 — Metrik Evaluasi ──────────────────────────────────────────────
st.subheader("Metrik Evaluasi Model")

eval_df = pd.read_csv(eval_path) if os.path.exists(eval_path) else pd.DataFrame()

def get_eval_val(metrik_name):
    if eval_df.empty:
        return "-"
    res = eval_df[eval_df["Metrik"] == metrik_name]
    if not res.empty:
        return str(res["Nilai"].iloc[0])
    return "-"

cv_val = get_eval_val("Coherence_Score_CV")
umass_val = get_eval_val("Coherence_Score_UMass")
perp_val = get_eval_val("Train_Log_Perplexity")
div_val = get_eval_val("Topic_Diversity_Top10")
amb_val = get_eval_val("Pct_Max_Theta_Under_0.50")
if amb_val != "-":
    amb_val = f"{amb_val}%"

eval_table_data = {
    "Metrik": [
        "C_V",
        "UMass",
        "Log Perplexity",
        "Topic Diversity",
        "Dok. ambigu (max_theta < 0.5)",
    ],
    "Nilai": [cv_val, umass_val, perp_val, div_val, amb_val],
    "Keterangan": [
        "Protokol: texts=data/intermediate/ (1.849 vocab), topn=20, processes=1. Lebih tinggi lebih baik.",
        "Lebih tinggi (mendekati 0) lebih baik.",
        "Lebih rendah lebih baik.",
        "Proporsi kata unik antar topik (top-10).",
        "Persentase dokumen dengan keyakinan rendah.",
    ],
}
st.dataframe(pd.DataFrame(eval_table_data), use_container_width=True, hide_index=True)

_n_cfg = len(pd.read_csv(os.path.join(REPO, "outputs", "model", "hyperparameter_results.csv"),
                          encoding="utf-8-sig")) if os.path.exists(
    os.path.join(REPO, "outputs", "model", "hyperparameter_results.csv")) else 140

# K dan C_V resmi dibaca dari artefak (bukan hard-code)
_M = artifacts.load_metrics()
_k_final = _M.get("k", 8)
_cv_final = artifacts.idr(_M.get("Coherence_Score_CV", 0.4256), 4)
N_DOC_TAMPIL = artifacts.idr(_M.get("Total_Documents", 285), 0)
RENTANG_TAHUN = artifacts.rentang_tahun()

st.markdown(f"**Nilai C_V dari Grid Search ({_n_cfg} konfigurasi, K=4–10):**")
st.caption(
    "Nilai berikut adalah C_V maksimum untuk setiap K pada tahap grid search "
    f"(`model/hyperparameter_results.csv`, {_n_cfg} konfigurasi). Model final K={_k_final} "
    f"dilatih ulang dengan passes=30 dan iterations=500 sehingga menghasilkan "
    f"C_V = {_cv_final} — angka ini tidak berasal dari grid search, melainkan dari "
    "pelatihan final."
)

# C_V per-K dihitung dari artefak, bukan ditulis manual
_hp_path = os.path.join(REPO, "outputs", "model", "hyperparameter_results.csv")
_k_rows = []
if os.path.exists(_hp_path):
    _hp = pd.read_csv(_hp_path, encoding="utf-8-sig")
    _best = _hp.loc[_hp.groupby("k")["coherence_cv"].idxmax()]
    _k_rows = [[int(r["k"]), artifacts.idr(float(r["coherence_cv"]), 4)]
               for _, r in _best.sort_values("k").iterrows()]

_keterangan = {
    8: "Dipilih (interpretabilitas & minimum 21 dok./topik)",
    9: "C_V tertinggi pada grid search",
}
k_comp_data = {
    "K": [r[0] for r in _k_rows],
    "C_V (grid search)": [r[1] for r in _k_rows],
    "Keterangan": [_keterangan.get(r[0], "") for r in _k_rows],
}
st.dataframe(pd.DataFrame(k_comp_data), use_container_width=True, hide_index=True)

# ── BAGIAN 4 — Label Topik Resmi ─────────────────────────────────────────────
st.subheader("Topik yang Teridentifikasi")

if os.path.exists(labels_path):
    with open(labels_path, "r", encoding="utf-8") as f:
        labels_json = json.load(f)
    
    topic_rows = []
    for item in labels_json:
        topic_rows.append({
            "No": item.get("topic_id"),
            "Topik": item.get("label"),
            "Kepercayaan": item.get("tingkat_kepercayaan"),
        })
    st.dataframe(pd.DataFrame(topic_rows), use_container_width=True, hide_index=True)
else:
    st.error("labels.json tidak ditemukan")

st.caption(
    "Topik 2 dan Topik 8 bersifat heterogen. "
    "Label ditetapkan berdasarkan top-10 kata kunci "
    "dengan fingerprint SHA256 yang terkunci."
)

# ── BAGIAN 5 — Keterbatasan ──────────────────────────────────────────────────
st.subheader("Keterbatasan Penelitian")

st.write(
    "Penelitian ini memiliki beberapa keterbatasan. "
    "Pertama, hanya tersedia 5 titik data tahunan "
    "sehingga proyeksi WMA bersifat sangat indikatif. "
    "Kedua, 38 dokumen menggunakan judul sebagai "
    "pengganti abstrak karena abstrak tidak tersedia "
    "dalam bahasa Indonesia, terkonsentrasi di "
    "tahun 2021-2022. Ketiga, Topik 2 dan Topik 8 "
    "bersifat heterogen dan merupakan keterbatasan "
    "model pada korpus dengan variasi metodologi "
    "yang luas. Keempat, WMA tidak terbukti lebih "
    "akurat dari rata-rata historis untuk 7 dari 8 "
    "topik berdasarkan backtest walk-forward."
)

# ── FOOTER ───────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    f"Versi artefak: {sha256_hash} · Data cutoff: September 2026 · "
    "Skripsi S1 Program Studi Sistem Informasi UIN Raden Fatah Palembang"
)
