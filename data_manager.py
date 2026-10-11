"""
Modul Utilitas Data: Normalisasi Indeks Topik & Hashing Artefak

Menyelaraskan konvensi indeks antara:
- Model LDA Gensim (indeks 0-indexed)
- Distribusi topik & label topik (indeks 1-indexed)

Fungsi di sini dipakai oleh `auto_labeling.py`, `pipeline.py`,
serta penghitungan fingerprint artefak pada halaman dashboard.
"""

import hashlib
import logging
from pathlib import Path

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
# HASHING FILE UNTUK FOOTER VERSI ARTEFAK
# ---------------------------------------------------------------------------
def get_file_hash(filepath) -> str:
    """Menghitung nilai hash MD5 file sebagai penanda versi artefak."""
    filepath = Path(filepath)
    if not filepath.exists():
        return "NON_EXISTENT"
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()