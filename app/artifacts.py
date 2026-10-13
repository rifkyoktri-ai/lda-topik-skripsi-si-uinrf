"""
artifacts.py  — Central data loader. JANGAN UBAH SUMBER DATA.
Semua angka dari GROUND_TRUTH.md.
"""
import os
import re
import json
import hashlib
import pandas as pd
import streamlit as st
import importlib
from gensim.models import LdaModel
import chart_helpers
importlib.reload(chart_helpers)

REPO      = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIX_DIR   = os.path.join(REPO, "_fix_final")
LABEL_DIR = os.path.join(REPO, "_label_work")
STATS_DIR = os.path.join(REPO, "_stats_forecast")
MODEL_DIR = os.path.join(REPO, "model")

FORECAST_FILE = os.path.join(STATS_DIR, "forecast_table.csv")
DIST_FILE     = os.path.join(FIX_DIR, "topic_distribution_final.csv")


def rentang_tahun() -> str:
    """
    Rentang tahun korpus, dihitung dari artefak (mis. "2021-2025").

    Dipakai pada judul halaman dan caption agar tahun tidak diketik manual
    di beberapa berkas sekaligus.
    """
    if os.path.exists(DIST_FILE):
        th = pd.read_csv(DIST_FILE, encoding="utf-8-sig")["Tahun"]
        return f"{int(th.min())}\u2013{int(th.max())}"
    return "2021\u20132025"   # cadangan bila artefak tidak ada

# Hash artefak untuk versioning
if os.path.exists(FORECAST_FILE):
    with open(FORECAST_FILE, "rb") as _f:
        ARTIFACT_VERSION = hashlib.sha256(_f.read()).hexdigest()[:8]
else:
    ARTIFACT_VERSION = "unknown"


def idr(value, decimals=2):
    """Angka gaya Indonesia: 0.6849 -> '0,68'"""
    return (f"{value:.{decimals}f}").replace(".", ",")


def load_metrics() -> dict:
    """
    Baca metrik resmi dari artefak - JANGAN hard-code di halaman.

    Sumber:
      _fix_final/evaluation_summary.csv        -> alpha, eta, C_V, vocab, dll
      _stats_forecast/composition_tests_report.md -> chi-square, p, Cramer's V

    Mengembalikan dict; kunci yang tidak ditemukan simply tidak ada.
    """
    out = {}

    ev = os.path.join(FIX_DIR, "evaluation_summary.csv")
    if os.path.exists(ev):
        df = pd.read_csv(ev, encoding="utf-8-sig")
        for _, r in df.iterrows():
            key = str(r["Metrik"]).strip()
            val = r["Nilai"]
            if key == "Model_Parameters":
                out["model_parameters"] = str(val)
                m = re.search(r"alpha=([^,]+),\s*eta=([^,]+)", str(val))
                if m:
                    out["alpha"] = m.group(1).strip()
                    out["eta"] = m.group(2).strip()
                m = re.search(r"K=(\d+)", str(val))
                if m:
                    out["k"] = int(m.group(1))
            else:
                try:
                    out[key] = float(val)
                except (TypeError, ValueError):
                    out[key] = val

    ct = os.path.join(STATS_DIR, "composition_tests_report.md")
    if os.path.exists(ct):
        with open(ct, "r", encoding="utf-8") as f:
            txt = f.read()
        for key, pattern in [
            ("chi2", r"Statistik \$\\chi\^2\$\*\*:\s*([\d.]+)"),
            ("chi2_df", r"Derajat Kebebasan \(\$dof\$\)\*\*:\s*(\d+)"),
            ("chi2_p", r"p-value Mentah \(Asimptotik\)\*\*:\s*([\d.]+)"),
            ("cramers_v", r"Cramer's V\*\*:\s*([\d.]+)"),
            ("perm_p", r"p-value Permutasi\*\*:\s*([\d.]+)"),
        ]:
            m = re.search(pattern, txt)
            if m:
                out[key] = float(m.group(1))
    return out


def load_all() -> dict:
    """
    Memuat semua data resmi. Mengembalikan dict dengan key:
      counts, yearly, intervals, distribution, forecast, dataset, labels
    labels: dict {int topic_id -> dict dengan key: label, deskripsi,
                  tingkat_kepercayaan, status}
    """
    labels_file = os.path.join(LABEL_DIR, "labels.json")
    with open(labels_file, "r", encoding="utf-8") as f:
        raw_labels = json.load(f)

    # Validasi 8 topik
    if len(raw_labels) != 8:
        raise ValueError(f"labels.json harus 8 entri, ditemukan {len(raw_labels)}")

    # Buat dict {int -> record} – gunakan integer sebagai key
    labels_dict = {int(item["topic_id"]): item for item in raw_labels}

    # Counts
    counts_df = pd.read_csv(os.path.join(FIX_DIR, "topic_counts_final.csv"))
    if counts_df["jumlah_dokumen"].sum() != 285:
        raise ValueError("Total dokumen di topic_counts_final.csv bukan 285")

    # Yearly
    yearly_df = pd.read_csv(os.path.join(FIX_DIR, "topic_by_year_final.csv"))

    # Intervals (Wilson CI)
    intervals_df = pd.read_csv(os.path.join(STATS_DIR, "share_intervals.csv"))

    # Distribution per dokumen
    dist_df = pd.read_csv(os.path.join(FIX_DIR, "topic_distribution_final.csv"))
    # Bersihkan newline di judul
    dist_df["Judul"] = dist_df["Judul"].str.replace(r"\s+", " ", regex=True).str.strip()

    # Dataset asli (untuk pencarian teks)
    dataset_df = pd.read_csv(os.path.join(FIX_DIR, "dataset_corrected.csv"))

    # Forecast / proyeksi
    forecast_df = pd.read_csv(FORECAST_FILE)
    if len(forecast_df) != 8:
        raise ValueError(f"forecast_table.csv harus 8 baris, ditemukan {len(forecast_df)}")

    return {
        "counts":          counts_df,
        "yearly":          yearly_df,
        "by_year":         yearly_df,
        "intervals":       intervals_df,
        "share_intervals": intervals_df,
        "distribution":    dist_df,
        "dataset":         dataset_df,
        "forecast":        forecast_df,
        "labels":          raw_labels,
        "labels_dict":     labels_dict,
        "metrics":         load_metrics(),
    }


@st.cache_resource
def load_model():
    return LdaModel.load(os.path.join(MODEL_DIR, "lda_model.gensim"))
