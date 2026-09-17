"""
Modul Manajemen Data & Normalisasi Indeks Topik

Modul ini menyelaraskan data antara:
- Model LDA Gensim (indeks 0-indexed)
- Distribusi topik (indeks 1-indexed)
- Label topik (JSON & CSV)
- Metrik evaluasi model
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from gensim.models import LdaModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NORMALISASI INDEKS TOPIK (0-INDEXED VS 1-INDEXED)
# ---------------------------------------------------------------------------
def to_one_indexed(topic_id_0_based: int) -> int:
    """Mengonversi ID topik berbasis 0 (Gensim) ke ID berbasis 1 (UI/Manusia)."""
    return int(topic_id_0_based) + 1


def to_zero_indexed(topic_id_1_based: int) -> int:
    """Mengonversi ID topik berbasis 1 (UI/Manusia) ke ID berbasis 0 (Gensim)."""
    return int(topic_id_1_based) - 1


# ---------------------------------------------------------------------------
# HASHING FILE UNTUK CACHE STREAMLIT
# ---------------------------------------------------------------------------
def get_file_hash(filepath: Path) -> str:
    """Menghitung nilai hash MD5 file untuk pembaruan cache Streamlit."""
    filepath = Path(filepath)
    if not filepath.exists():
        return "NON_EXISTENT"
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_model_artifacts_hash(base_path: Path) -> str:
    """Menghitung kombinasi hash dari seluruh berkas artifact model."""
    base_path = Path(base_path)
    files = [
        base_path / "model" / "lda_model.gensim",
        base_path / "model" / "topic_distribution.csv",
        base_path / "model" / "topic_labels.json",
        base_path / "model" / "topic_labels.csv",
        base_path / "model" / "evaluation_metrics.csv",
    ]
    combined = "_".join([get_file_hash(f) for f in files])
    return hashlib.md5(combined.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# PENYELARASAN DATA MODEL
# ---------------------------------------------------------------------------
def load_unified_model_data(base_path: Path) -> Dict[str, Any]:
    """
    Memuat dan menyelaraskan seluruh artifact model LDA.
    """
    base_path = Path(base_path)
    model_dir = base_path / "model"

    messages = []
    lda_model = None
    num_topics = 0

    # 1. Muat Model Gensim
    model_path = model_dir / "lda_model.gensim"
    if model_path.exists():
        try:
            lda_model = LdaModel.load(str(model_path))
            num_topics = lda_model.num_topics
            messages.append(f"Loaded LDA Model with num_topics = {num_topics}")
        except Exception as e:
            messages.append(f"Error loading LDA model: {e}")
    else:
        messages.append("LDA model file not found.")

    # 2. Muat Metrik Evaluasi
    metrics_df = pd.DataFrame()
    metrics_dict = {}
    metrics_path = model_dir / "evaluation_metrics.csv"
    if metrics_path.exists():
        try:
            metrics_df = pd.read_csv(metrics_path)
            metrics_dict = dict(zip(metrics_df["Metrik"], metrics_df["Nilai"]))
            if num_topics == 0 and "Jumlah Topik (K)" in metrics_dict:
                num_topics = int(metrics_dict["Jumlah Topik (K)"])
        except Exception as e:
            messages.append(f"Error loading evaluation metrics: {e}")

    # Fallback default jika model belum dimuat
    if num_topics == 0:
        num_topics = 3

    # 3. Muat & Normalisasi Distribusi Topik
    topic_dist_df = pd.DataFrame()
    dist_path = model_dir / "topic_distribution.csv"
    if dist_path.exists():
        try:
            topic_dist_df = pd.read_csv(dist_path)
            if "topik_dominan" in topic_dist_df.columns and not topic_dist_df.empty:
                min_t = topic_dist_df["topik_dominan"].min()
                max_t = topic_dist_df["topik_dominan"].max()
                if min_t == 0 and max_t == num_topics - 1:
                    topic_dist_df["topik_dominan"] = topic_dist_df["topik_dominan"] + 1
                    messages.append("Normalized topic_distribution topik_dominan from 0-indexed to 1-indexed.")
        except Exception as e:
            messages.append(f"Error loading topic distribution: {e}")

    # Hitung jumlah dokumen aktual per topic_id (1-based: 1..num_topics)
    topic_counts = pd.Series(0, index=range(1, num_topics + 1), dtype=int)
    if not topic_dist_df.empty and "topik_dominan" in topic_dist_df.columns:
        actual_counts = topic_dist_df["topik_dominan"].value_counts()
        for tid, count in actual_counts.items():
            if 1 <= int(tid) <= num_topics:
                topic_counts[int(tid)] = count
            else:
                messages.append(f"Warning: Found document assigned to invalid topic ID {tid} (outside 1..{num_topics}).")

    # Cek jika ada topik tanpa dokumen
    zero_doc_topics = [tid for tid, cnt in topic_counts.items() if cnt == 0]
    if zero_doc_topics:
        messages.append(f"Graceful handling: Topic(s) {zero_doc_topics} have zero assigned documents in current corpus.")

    # 4. Muat & Selaraskan Label Topik
    json_path = model_dir / "topic_labels.json"
    csv_path = model_dir / "topic_labels.csv"

    labels_map = {}
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                raw_json = json.load(f)
            for tid_key, info in raw_json.items():
                tid_int = int(tid_key)
                if tid_int < num_topics:
                    tid_1_based = to_one_indexed(tid_int)
                else:
                    tid_1_based = tid_int
                labels_map[tid_1_based] = info
        except Exception as e:
            messages.append(f"Error reading topic_labels.json: {e}")

    rows = []
    for tid in range(1, num_topics + 1):
        info = labels_map.get(tid, {})
        tid_0_based = to_zero_indexed(tid)

        model_top_words = []
        if lda_model:
            try:
                model_top_words = [w for w, _ in lda_model.show_topic(tid_0_based, topn=10)]
            except Exception:
                pass

        label_final = info.get("label_final") or info.get("label") or f"Topik {tid}"
        keywords = ";".join(info.get("top_words_filtered", info.get("top_words", model_top_words[:5])))
        score = info.get("score", "")
        coherence = info.get("quality_coherence", "")
        num_docs = int(topic_counts.get(tid, 0))

        rows.append({
            "topic_id": tid,
            "label": label_final,
            "description": f"Topik {tid}: {label_final} ({num_docs} dokumen)",
            "keywords": keywords,
            "label_score": score,
            "quality_coherence": coherence,
            "num_docs": num_docs
        })

    topic_labels_df = pd.DataFrame(rows).sort_values("topic_id").reset_index(drop=True)

    return {
        "lda_model": lda_model,
        "num_topics": num_topics,
        "topic_distribution": topic_dist_df,
        "topic_labels": topic_labels_df,
        "evaluation_metrics": metrics_df,
        "metrics_dict": metrics_dict,
        "topic_counts": topic_counts,
        "is_valid": lda_model is not None or not topic_dist_df.empty,
        "contract_messages": messages,
    }


def evaluate_topic_overlap(lda_model, threshold=0.70):
    """
    Deteksi topik yang terlalu mirip berdasarkan cosine similarity
    distribusi kata antar topik.
    """
    if lda_model is None:
        return []
    n_topics = lda_model.num_topics
    n_words = lda_model.num_terms
    topic_word_matrix = np.zeros((n_topics, n_words))

    for t in range(n_topics):
        words_probs = dict(lda_model.show_topic(t, topn=n_words))
        for word_id in range(n_words):
            word = lda_model.id2word[word_id]
            topic_word_matrix[t, word_id] = words_probs.get(word, 0.0)

    overlapping = []
    for i in range(n_topics):
        for j in range(i + 1, n_topics):
            norm_i = np.linalg.norm(topic_word_matrix[i])
            norm_j = np.linalg.norm(topic_word_matrix[j])
            if norm_i > 0 and norm_j > 0:
                sim = float(np.dot(topic_word_matrix[i], topic_word_matrix[j]) / (norm_i * norm_j))
                if sim > threshold:
                    overlapping.append((i, j, sim))

    return overlapping
