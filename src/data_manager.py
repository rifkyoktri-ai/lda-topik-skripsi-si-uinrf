"""
Modul Utilitas Data: Normalisasi Indeks Topik

Menyelaraskan konvensi indeks antara:
- Model LDA Gensim (indeks 0-indexed)
- Distribusi topik & label topik (indeks 1-indexed)
"""

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NORMALISASI INDEKS TOPIK (0-INDEXED VS 1-INDEXED)
# ---------------------------------------------------------------------------
def to_one_indexed(topic_id_0_based: int) -> int:
    """Mengonversi ID topik berbasis 0 (Gensim) ke ID berbasis 1 (UI/Manusia)."""
    return int(topic_id_0_based) + 1