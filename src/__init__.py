"""
src — Paket modul bersama untuk seluruh pipeline LDA.

Modul di sini berisi definisi yang terlalu panjang atau terlalu umum untuk
ditempelkan di dalam notebook:

  config.py              semua path & parameter
  indonesian_stopwords.py daftar stopword bahasa Indonesia + domain SI
  topic_labels.py        validasi fingerprint SHA-256 label vs model
  data_manager.py        helper normalisasi indeks topik (0-based <-> 1-based)

Notebook mengimpor modul ini lewat:
    import sys; sys.path.insert(0, '..')
    from src import config as C
"""