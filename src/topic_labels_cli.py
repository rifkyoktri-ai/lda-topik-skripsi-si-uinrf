"""
topic_labels_cli.py — Validasi integritas label topik terhadap model LDA.

Memastikan fingerprint SHA-256 pada `_label_work/labels.json` masih cocok
dengan top-10 kata pada `model/lda_model.gensim`. Gagal keras (ValueError)
bila ada ketidakcocokan, jumlah topik beda, atau status bukan "disetujui".

Jalankan:  python -m src.topic_labels_cli
"""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src import config as C
    from src.topic_labels import validate_labels
else:
    from . import config as C
    from .topic_labels import validate_labels


def main():
    try:
        validate_labels(str(C.LDA_MODEL), str(C.LABELS_JSON))
        print("[SUCCESS] VALIDASI BERHASIL: Seluruh fingerprint label cocok 100% dengan model Gensim!")
        import json
        with open(C.LABELS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        for d in data:
            print(f"  [Topik {d['topic_id']}] {d['label']} "
                  f"(Kepercayaan: {d['tingkat_kepercayaan']}) -> OK")
    except Exception as exc:
        print(f"[ERROR] VALIDASI GAGAL: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()