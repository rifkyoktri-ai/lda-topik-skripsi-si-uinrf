"""
Skrip Perubahan & Pembaruan auto_labeling.py
Menggunakan label terkunci dari _label_work/labels.json dengan pemindaian validasi fingerprint.
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
    Memuat label resmi yang telah disetujui dari _label_work/labels.json.

    Args:
        labels_path: Path ke berkas labels.json.
        model_path: Path ke model LDA Gensim.
        validate: Jika True, jalankan validasi fingerprint SHA-256 terhadap model
                  dan gagal keras bila tidak cocok. Set False hanya bila pemanggil
                  sudah memvalidasi model yang sama sebelumnya (mis. pipeline.py).

    Returns:
        dict dengan key string 0-indexed -> info label.
    """
    if validate:
        validate_labels(model_path=model_path, labels_path=labels_path)
    
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels_data = json.load(f)
        
    topic_labels_map = {}
    for item in labels_data:
        tid = item["topic_id"]
        # Format 0-indexed internal key string
        tid_str = str(tid - 1)
        topic_labels_map[tid_str] = {
            "topic_id": tid,
            "label_final": item["label"],
            "description": item.get("deskripsi", ""),
            "confidence": item.get("tingkat_kepercayaan", ""),
            "status": item.get("status", "")
        }
    return topic_labels_map


def export_human_validation_dataset(
    lda_model: LdaModel,
    corpus: list,
    dist_df: pd.DataFrame,
    output_path: str = str(C.HUMAN_VALIDATION_CSV)
) -> pd.DataFrame:
    """
    Mengekspor distribusi dokumen-topik beserta 15 kata teratas per topik
    ke file CSV untuk keperluan validasi pakar.
    """
    print(f"\n[EXPERT VALIDATION EXPORTER] Exporting Human Validation Dataset...")
    rows = []
    
    topic_top_words = {}
    for t_id in range(lda_model.num_topics):
        words = [w for w, _ in lda_model.show_topic(t_id, topn=15)]
        topic_top_words[to_one_indexed(t_id)] = ", ".join(words)
        
    for idx, row in dist_df.iterrows():
        doc_id = row.get('ID', idx)
        title = str(row.get('Judul', '')).replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ').strip()
        tahun = row.get('Tahun', '')
        dom_topic = int(row.get('topik_dominan', 1))
        prob = row.get('prob_dominan', 0.0)
        
        top_15_words = topic_top_words.get(dom_topic, "")
        
        rows.append({
            'document_id': doc_id,
            'tahun': tahun,
            'judul_skripsi': title,
            'assigned_dominant_topic': dom_topic,
            'topic_probability': prob,
            'topic_top15_words': top_15_words,
            'expert_relevance_score_1to5': '',
            'expert_notes': ''
        })
        
    export_df = pd.DataFrame(rows)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(out_file, index=False)
    print(f"  [HUMAN-IN-THE-LOOP] Expert validation CSV exported to: {out_file} ({len(export_df)} rows)")
    return export_df


def export_topic_labels_json(
    topic_labels_map: dict,
    lda_model: LdaModel,
    dist_df: pd.DataFrame | None = None,
    output_path: str = str(C.TOPIC_LABELS_JSON),
) -> None:
    """
    Menulis cerminan JSON dari label yang telah disetujui.

    Berkas ini sebelumnya tidak ditulis siapa pun dan masih memuat label
    hasil auto-labeling lama (label topik 7 berisi label milik topik 5),
    sehingga menyesatkan. Sekarang label_final diambil dari
    _label_work/labels.json agar identik dengan topic_labels.csv.
    """
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
    print(f"  [LABEL JSON] Cerminan label disetujui ditulis ke: {out_file}")


def run_auto_labeling():
    parser = argparse.ArgumentParser(description="Auto Labeling Topik LDA & Expert Validation Exporter")
    parser.add_argument("--model_path", type=str, default=str(C.LDA_MODEL), help="Path ke file model LDA")
    parser.add_argument("--labels_path", type=str, default=str(C.LABELS_JSON), help="Path ke file labels JSON")
    parser.add_argument("--dist_path", type=str, default=str(C.TOPIC_DISTRIBUTION), help="Path ke topic distribution CSV")
    parser.add_argument("--output", type=str, default=str(C.TOPIC_LABELS_CSV), help="Path untuk menyimpan hasil label (CSV)")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"LDA model tidak ditemukan: {model_path}")

    # Memuat & memvalidasi label resmi (Gagal keras jika fingerprint salah)
    topic_labels_map = load_approved_labels(args.labels_path, str(model_path))

    lda_model = LdaModel.load(str(model_path))

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
    print(f"✅ Topic labels CSV successfully created at -> {output_path}")

    # Cerminan JSON agar topic_labels.json tidak lagi basi/merusak
    dist_df = None
    if Path(args.dist_path).exists():
        dist_df = pd.read_csv(args.dist_path, encoding='utf-8-sig')
    export_topic_labels_json(topic_labels_map, lda_model, dist_df)


if __name__ == "__main__":
    run_auto_labeling()
