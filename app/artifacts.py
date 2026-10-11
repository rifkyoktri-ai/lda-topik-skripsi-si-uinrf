"""
artifacts.py  — Central data loader. JANGAN UBAH SUMBER DATA.
Semua angka dari GROUND_TRUTH.md.
"""
import os
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

# Hash artefak untuk versioning
if os.path.exists(FORECAST_FILE):
    with open(FORECAST_FILE, "rb") as _f:
        ARTIFACT_VERSION = hashlib.sha256(_f.read()).hexdigest()[:8]
else:
    ARTIFACT_VERSION = "unknown"


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
    }


@st.cache_resource
def load_model():
    return LdaModel.load(os.path.join(MODEL_DIR, "lda_model.gensim"))
