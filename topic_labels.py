"""
Modul Validasi Integritas Label Topik LDA vs Fingerprint Model Gensim
"""
import os
import sys
import json
import hashlib
from typing import Dict, Any
import gensim

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8', errors='ignore')

def compute_topic_fingerprint(lda_model, topic_id: int, topn: int = 10) -> str:
    """
    Menghitung hash SHA256 dari top-10 kata kunci terbanyak pada suatu topik (diurutkan A-Z).
    """
    raw_words = lda_model.show_topic(topic_id, topn=topn)
    words = sorted([str(w) for w, _ in raw_words])
    words_str = ",".join(words)
    return hashlib.sha256(words_str.encode('utf-8')).hexdigest()

def validate_labels(model_path: str = "model/lda_model.gensim", labels_path: str = "_label_work/labels.json") -> bool:
    """
    Memvalidasi integritas file labels.json terhadap model Gensim tersimpan.
    Gagal keras (ValueError) jika file tidak ada, jumlah topik != num_topics, atau fingerprint tidak cocok.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file tidak ditemukan di {model_path}")
    
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"File label tidak ditemukan di {labels_path}")
    
    lda_model = gensim.models.LdaModel.load(model_path)
    
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels_data = json.load(f)
        
    if len(labels_data) != lda_model.num_topics:
        raise ValueError(
            f"Jumlah topik di {labels_path} ({len(labels_data)}) tidak cocok dengan "
            f"jumlah topik model ({lda_model.num_topics})"
        )
        
    for item in labels_data:
        tid = item.get("topic_id")
        if tid is None or not (1 <= tid <= lda_model.num_topics):
            raise ValueError(f"ID Topik tidak valid: {tid}")
            
        topic_idx = tid - 1
        model_fp = compute_topic_fingerprint(lda_model, topic_idx, topn=10)
        expected_fp = item.get("top_words_fingerprint")
        
        if model_fp != expected_fp:
            raise ValueError(
                f"[Gagal Validasi Fingerprint] Topik {tid} ('{item.get('label')}') "
                f"memiliki fingerprint tidak cocok!\n"
                f"  Expected (labels.json) : {expected_fp}\n"
                f"  Computed (model Gensim): {model_fp}"
            )
            
        status = item.get("status")
        if status != "disetujui":
            raise ValueError(f"Status topik {tid} bukan 'disetujui' (ditemukan: {status})")
            
        confidence = item.get("tingkat_kepercayaan")
        if confidence not in ["Tinggi", "Sedang", "Rendah"]:
            raise ValueError(f"Tingkat kepercayaan topik {tid} tidak valid: {confidence}")

    return True

if __name__ == "__main__":
    try:
        res = validate_labels("model/lda_model.gensim", "_label_work/labels.json")
        print("[SUCCESS] VALIDASI BERHASIL: Seluruh fingerprint label cocok 100% dengan model Gensim!")
        with open("_label_work/labels.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
            for d in data:
                print(f"  [Topik {d['topic_id']}] {d['label']} (Kepercayaan: {d['tingkat_kepercayaan']}) -> OK")
    except Exception as e:
        print(f"[ERROR] VALIDASI GAGAL: {e}")
        exit(1)
