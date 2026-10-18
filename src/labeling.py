"""
labeling.py — Pengelola Nama Resmi Topik LDA

File ini mengelola nama (label) untuk setiap topik LDA.
Nama topik tidak dibuat otomatis oleh komputer — melainkan ditetapkan
secara manual oleh peneliti setelah menganalisis kata kunci tiap topik,
lalu dikonfirmasi bersama dosen pembimbing ("Human-in-the-Loop").

Label resmi disimpan di: outputs/model/labels.json
File ini membacanya, memvalidasi keasliannya, dan menyediakannya untuk
dipakai oleh modeling.py dan dashboard.
"""
import json
import warnings
import argparse
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')
from gensim.models import LdaModel
from . import config as C
from .data_manager import to_one_indexed
from .topic_labels import validate_labels


def load_approved_labels(
    labels_path: str = str(C.LABELS_JSON),
    model_path: str = str(C.LDA_MODEL),
    validate: bool = True
) -> dict:
    """
    Muat nama-nama topik resmi dari file labels.json.

    Secara default, fungsi ini juga memverifikasi bahwa nama-nama tersebut
    masih cocok dengan model LDA yang sekarang (menggunakan sidik jari SHA-256).

    Parameter:
      labels_path — lokasi file labels.json
      model_path  — lokasi model LDA
      validate    — jika True (default), jalankan pengecekan sidik jari.
                    Set False hanya jika pemanggil sudah mengecek sebelumnya.

    Mengembalikan dict dengan format:
      {"0": {"topic_id": 1, "label_final": "...", ...},
       "1": {"topic_id": 2, ...}, ...}
    (kunci string 0-indexed untuk kompatibilitas dengan model Gensim)
    """
    if validate:
        validate_labels(model_path=model_path, labels_path=labels_path)

    with open(labels_path, 'r', encoding='utf-8') as f:
        labels_data = json.load(f)

    topic_labels_map = {}
    for item in labels_data:
        tid = item["topic_id"]
        tid_str = str(tid - 1)  # ubah ke 0-indexed untuk kunci internal
        topic_labels_map[tid_str] = {
            "topic_id": tid,
            "label_final": item["label"],
            "description": item.get("deskripsi", ""),
            "confidence": item.get("tingkat_kepercayaan", ""),
            "status": item.get("status", "")
        }
    return topic_labels_map


def export_topic_labels_json(
    topic_labels_map: dict,
    lda_model: LdaModel,
    dist_df: pd.DataFrame | None = None,
    output_path: str = str(C.TOPIC_LABELS_JSON),
) -> None:
    """
    Tulis cerminan label resmi ke file topic_labels.json.

    File ini sebelumnya mengandung label auto-labeling lama yang salah
    (label topik 7 berisi label milik topik 5). Sekarang label diambil
    dari labels.json agar topic_labels.json dan topic_labels.csv selalu identik.
    """
    # Hitung jumlah dokumen per topik (opsional, jika dist_df tersedia)
    num_docs = {}
    if dist_df is not None and 'topik_dominan' in dist_df.columns:
        num_docs = dist_df['topik_dominan'].value_counts().to_dict()

    payload = {}
    for tid_str, info in topic_labels_map.items():
        tid = info['topic_id']
        top_words = [w for w, _ in lda_model.show_topic(tid - 1, topn=10)]
        payload[tid_str] = {
            'topic_id': tid,
            'label_final': info['label_final'],
            'description': info['description'],
            'confidence': info['confidence'],
            'status': info['status'],
            'top_words': top_words,
            'num_docs': int(num_docs.get(tid, 0)),
        }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"  [LABEL JSON] Ditulis ke: {out_file}")


def run_auto_labeling():
    """
    Jalankan proses pelabelan dari terminal (command line).

    Membaca label dari labels.json, lalu mengekspor ke:
    - topic_labels.csv (tabel label per topik)
    - topic_labels.json (format JSON)
    """
    parser = argparse.ArgumentParser(description="Ekspor label topik LDA ke CSV & JSON")
    parser.add_argument("--model_path", type=str, default=str(C.LDA_MODEL), help="Lokasi file model LDA")
    parser.add_argument("--labels_path", type=str, default=str(C.LABELS_JSON), help="Lokasi file labels JSON resmi")
    parser.add_argument("--dist_path", type=str, default=str(C.TOPIC_DISTRIBUTION), help="Lokasi file distribusi topik CSV")
    parser.add_argument("--output", type=str, default=str(C.TOPIC_LABELS_CSV), help="Lokasi penyimpanan hasil label CSV")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model LDA tidak ditemukan: {model_path}")

    # Muat & validasi label resmi (akan berhenti jika sidik jari tidak cocok)
    topic_labels_map = load_approved_labels(args.labels_path, str(model_path))

    lda_model = LdaModel.load(str(model_path))

    # Buat dan simpan CSV label
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels_rows = []
    for tid_str, info in topic_labels_map.items():
        tid = info['topic_id']
        top_words = [w for w, _ in lda_model.show_topic(tid - 1, topn=5)]
        labels_rows.append({
            'topic_id': tid,
            'label': info['label_final'],
            'description': info['description'],
            'keywords': ';'.join(top_words),
            'confidence': info['confidence'],
            'status': info['status']
        })
    df_labels = pd.DataFrame(labels_rows)
    df_labels.to_csv(output_path, index=False)
    print(f"[OK] File label topik berhasil dibuat -> {output_path}")

    # Perbarui juga file topic_labels.json agar tidak ketinggalan
    dist_df = None
    if Path(args.dist_path).exists():
        dist_df = pd.read_csv(args.dist_path, encoding='utf-8-sig')
    export_topic_labels_json(topic_labels_map, lda_model, dist_df)


if __name__ == "__main__":
    run_auto_labeling()
