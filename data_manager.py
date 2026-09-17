"""
Centralized Data Contract & Index Normalization Module (Single Source of Truth)

This module eliminates decoupled state issues between:
- Gensim LdaModel (0-indexed topic IDs)
- topic_distribution.csv (1-indexed topik_dominan)
- topic_labels.json / topic_labels.csv (0-indexed or 1-indexed representations)
- evaluation_metrics.csv
- Streamlit application state & caches
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd
from gensim.models import LdaModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. STRICT INDEX NORMALIZATION (0-INDEXED VS 1-INDEXED)
# ---------------------------------------------------------------------------
def to_one_indexed(topic_id_0_based: int) -> int:
    """Convert 0-based Gensim topic ID to 1-based human/UI topic ID."""
    return int(topic_id_0_based) + 1


def to_zero_indexed(topic_id_1_based: int) -> int:
    """Convert 1-based human/UI topic ID to 0-based Gensim topic ID."""
    return int(topic_id_1_based) - 1


# ---------------------------------------------------------------------------
# 2. FILE HASHING FOR STREAMLIT CACHE INVALIDATION
# ---------------------------------------------------------------------------
def get_file_hash(filepath: Path) -> str:
    """Calculate MD5 hash of a file to invalidate Streamlit caches when file content changes."""
    filepath = Path(filepath)
    if not filepath.exists():
        return "NON_EXISTENT"
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_model_artifacts_hash(base_path: Path) -> str:
    """Generate a combined hash string for all model artifact files."""
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
# 3. UNIFIED DATA CONTRACT (SINGLE SOURCE OF TRUTH)
# ---------------------------------------------------------------------------
def load_unified_model_data(base_path: Path) -> Dict[str, Any]:
    """
    Loads, validates, and synchronizes all LDA model artifacts.

    Returns dictionary containing:
    - 'lda_model': Gensim LdaModel instance (or None if missing)
    - 'num_topics': int
    - 'topic_distribution': DataFrame with normalized 1-indexed 'topik_dominan'
    - 'topic_labels': DataFrame with synchronized 1-indexed 'topic_id', 'label', etc.
    - 'evaluation_metrics': DataFrame
    - 'metrics_dict': Dict of metric name -> value
    - 'topic_counts': Series of document counts per 1-based topic_id
    - 'is_valid': bool indicating contract integrity
    - 'contract_messages': List of warning/info messages
    """
    base_path = Path(base_path)
    model_dir = base_path / "model"

    messages = []
    lda_model = None
    num_topics = 0

    # 1. Load Gensim Model
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

    # 2. Load Evaluation Metrics
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

    # Fallback default if model not loaded
    if num_topics == 0:
        num_topics = 3

    # 3. Load & Normalize Topic Distribution
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

    # Compute actual document counts per topic_id (1-based: 1..num_topics)
    topic_counts = pd.Series(0, index=range(1, num_topics + 1), dtype=int)
    if not topic_dist_df.empty and "topik_dominan" in topic_dist_df.columns:
        actual_counts = topic_dist_df["topik_dominan"].value_counts()
        for tid, count in actual_counts.items():
            if 1 <= int(tid) <= num_topics:
                topic_counts[int(tid)] = count
            else:
                messages.append(f"Warning: Found document assigned to invalid topic ID {tid} (outside 1..{num_topics}).")

    # Check for zero-document topics
    zero_doc_topics = [tid for tid, cnt in topic_counts.items() if cnt == 0]
    if zero_doc_topics:
        messages.append(f"Graceful handling: Topic(s) {zero_doc_topics} have zero assigned documents in current corpus.")

    # 4. Load & Synchronize Topic Labels
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
