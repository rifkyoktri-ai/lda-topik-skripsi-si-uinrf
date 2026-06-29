import json
import warnings
import sys
import argparse
import pandas as pd
from pathlib import Path
import numpy as np

warnings.filterwarnings('ignore')
from gensim.models import LdaModel
from indonesian_stopwords import get_all_stopwords

LABEL_MAPPING = {
    "sistem informasi perpustakaan": "Sistem Informasi Perpustakaan & Digital Library",
    "perpustakaan digital": "Sistem Informasi Perpustakaan & Digital Library",
    "digital library": "Sistem Informasi Perpustakaan & Digital Library",
    "programming extreme": "Extreme Programming (XP)",
    "extreme programming": "Extreme Programming (XP)",
    "sistem informasi akademik": "Sistem Informasi Akademik",
    "sistem informasi keuangan": "Sistem Informasi Keuangan",
    "sistem informasi desa": "Sistem Informasi Desa (SID)",
    "sistem informasi geografis": "Sistem Informasi Geografis (SIG)",
    "sistem pendukung keputusan": "Sistem Pendukung Keputusan (SPK)",
    "sistem informasi akuntansi": "Sistem Informasi Akuntansi",
    "sistem informasi geografis sig": "Sistem Informasi Geografis (SIG)",
    "pendukung keputusan": "Sistem Pendukung Keputusan (SPK)",
    "penerimaan siswa": "Sistem Informasi Penerimaan Siswa",
    "penerimaan mahasiswa": "Sistem Informasi Penerimaan Mahasiswa",
    "informasi akademik": "Sistem Informasi Akademik",
    "sistem informasi inventory": "Sistem Informasi Inventory",
    "surat bas": "Sistem Informasi Surat & Arsip",
    "surat terima": "Sistem Informasi Surat & Arsip",
    "bas pilih": "Sistem Informasi Surat & Arsip",
    "audit evaluation": "Audit & Evaluasi Sistem Informasi",
    "audit siakad": "Audit & Evaluasi Sistem Informasi",
    "siakad": "Sistem Informasi Akademik (SIAKAD)",
    "service quality": "Service Quality & Kepuasan Pengguna",
    "service satisfaction": "Service Quality & Kepuasan Pengguna",
    "satisfaction akademik": "Kepuasan Pengguna SI Akademik",
    "satisfaction puas": "Kepuasan Pengguna Sistem Informasi",
    "technology akademik": "Teknologi Informasi Akademik",
    "evaluation siakad": "Audit & Evaluasi SIAKAD",
    "net success": "Kesuksesan Sistem Informasi",
    "webqual usability": "Analisis Kualitas Website (WebQual)",
    "usability webqual": "Analisis Kualitas Website (WebQual)",
    "layan mobile": "Mobile Government & Layanan Publik",
    "pusat layan": "Sistem Informasi Pelayanan Publik",
    "bantu sentimen": "Analisis Sentimen & NLP",
    "sentimen analis": "Analisis Sentimen & NLP",
    "score recall": "Analisis Sentimen & Klasifikasi Teks",
    "precision recall": "Analisis Sentimen & Klasifikasi Teks",
    "tingkat kepuasan": "Kepuasan Pengguna Sistem Informasi",
    "puas kepuasan": "Kepuasan Pengguna Sistem Informasi",
    "kepuasan pengguna": "Kepuasan Pengguna Sistem Informasi",
    "usability layan": "Usability & Layanan Akademik",
    "layan simak": "Usability & Layanan Akademik",
    "layan mahasiswa": "Usability & Layanan Akademik",
    "layan akademik": "Usability & Layanan Akademik",
    "usability akademik": "Usability & Layanan Akademik",
    "manajemen risiko": "Manajemen Risiko & Usability Website",
    "risiko website": "Manajemen Risiko & Usability Website",
    "risiko usability": "Manajemen Risiko & Usability Website",
    "usability risiko": "Manajemen Risiko & Usability Website",
    "usability quality": "Manajemen Risiko & Usability Website",
    "kualitas layan": "Kualitas Layanan Sistem Informasi",
    "sukses layan": "Kualitas Layanan Sistem Informasi",
    "monitoring kesuksesan": "Kualitas Layanan Sistem Informasi",
    "kesuksesan sistem": "Kualitas Layanan Sistem Informasi",
    "kesuksesan implementasi": "Kualitas Layanan Sistem Informasi",
    "technologies kepuasan": "Kepuasan Pengguna Sistem Informasi",
    "teknologi kepuasan": "Kepuasan Pengguna Sistem Informasi",
    "kepuasan teknologi": "Kepuasan Pengguna Sistem Informasi",
    "kepuasan pengguna sistem": "Kepuasan Pengguna Sistem Informasi",
    "satisfaction quality": "Service Quality & User Satisfaction",
    "quality satisfaction": "Service Quality & User Satisfaction",
    "satisfaction results": "Service Quality & User Satisfaction",
    "puas layan": "Kepuasan Pengguna Sistem Informasi",
    "jual barang": "Sistem Informasi Penjualan & Inventory",
    "palm startup": "Sistem Informasi Penjualan & Inventory",
    "konseling bimbing": "Sistem Informasi Konseling & Bimbingan",
    "nikah disaster": "Sistem Informasi Konseling & Bimbingan",
    "tuju bantu": "Usability & Layanan Akademik",
    "bantu tuju": "Usability & Layanan Akademik",
    "laku tuju": "Perilaku Pengguna & Adopsi TI",
    "laku bantu": "Perilaku Pengguna & Adopsi TI",
    "tuju akademik": "Sistem Informasi Akademik",
    "stok laku": "Sistem Informasi Inventory",
    "akademik technology": "Sistem Informasi Akademik",
    "kembang laku": "Manajemen Risiko & Adopsi TI",
    "mudah giat": "Usability & Kemudahan Pengguna",
    "sentimen kasus": "Analisis Sentimen & Klasifikasi Teks",
    "extreme programming": "Extreme Programming (XP)",
    "disaster mining": "Data Mining & Disaster Management",
    "mohon mining": "Data Mining & Klasifikasi",
    "score machine": "Machine Learning & Klasifikasi",
    "decision tree": "Decision Tree & Klasifikasi",
    "angket survei": "Survei & Analisis Data",
    "maturity capability": "Maturity & Capability Model",
    "information system": "Sistem Informasi Manajemen",
    "success model": "Kesuksesan Sistem Informasi",
    "model success": "Kesuksesan Sistem Informasi",
    "net benefits": "Kesuksesan Sistem Informasi",
    "benefits user": "Kesuksesan Sistem Informasi",
    "users satisfaction": "Kepuasan Pengguna Sistem Informasi",
}

def label_topics_keybert(lda_model, all_stopwords, topic_titles=None):
    print("\n" + "="*60)
    print("AUTO LABELING TOPIK LDA DENGAN KEYBERT")
    print("="*60)
    print(f"Jumlah topik: {lda_model.num_topics}")
    print("Model: paraphrase-multilingual-MiniLM-L12-v2 (CPU)")
    print("Estimasi waktu: 10-20 menit (CPU only)")
    print("="*60)

    try:
        print("\n[1/3] Loading KeyBERT model...")
        from keybert import KeyBERT
        from sentence_transformers import SentenceTransformer
        sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        kw_model = KeyBERT(model=sentence_model)
        keybert_loaded = True
        print("      KeyBERT model loaded.")
    except Exception as e:
        print(f"      Gagal meload KeyBERT (Mungkin isu memory): {e}")
        print("      Menggunakan fallback (Top words).")
        keybert_loaded = False
        kw_model = None

    print("\n[2/3] Memproses setiap topik...")
    topic_labels = {}

    is_1_indexed = False
    if topic_titles and min(topic_titles.keys()) == 1:
        is_1_indexed = True

    for topic_id in range(lda_model.num_topics):
        tid = topic_id + 1
        print(f"\n  Topik {tid}/{lda_model.num_topics}:")

        raw_words = lda_model.show_topic(topic_id, topn=20)
        filtered_words = [
            word for word, weight in raw_words
            if word.lower() not in all_stopwords
            and len(word) > 2
            and not word.isdigit()
            and word.isalpha()
        ]

        top_words_str = " ".join(filtered_words[:10])
        print(f"  Raw words (top 10): {[w for w, _ in raw_words[:10]]}")
        print(f"  Filtered words: {filtered_words[:10]}")

        t_key = topic_id + 1 if is_1_indexed else topic_id
        titles = topic_titles.get(t_key, []) if topic_titles else []
        num_docs = len(titles)

        doc_text = top_words_str
        if titles:
            doc_text += " " + " ".join(titles)

        best_label = ""
        best_score = 0.0
        keywords_extracted = []

        if keybert_loaded:
            try:
                keywords = kw_model.extract_keywords(
                    doc_text,
                    keyphrase_ngram_range=(1, 2),
                    stop_words=list(all_stopwords),
                    use_mmr=True,
                    diversity=0.5,
                    top_n=5
                )

                if keywords:
                    best_label = keywords[0][0].title()
                    best_score = float(keywords[0][1])
                    keywords_extracted = [k[0] for k in keywords]
                    print(f"  KeyBERT candidates: {[(k, round(s,3)) for k,s in keywords[:3]]}")
                else:
                    print(f"  KeyBERT tidak menghasilkan kandidat, fallback ke top words")
            except Exception as e:
                print(f"  KeyBERT error: {e}, fallback ke top words")

        if not best_label:
            best_label = " ".join(filtered_words[:3]).title()
            best_score = 0.0
            keywords_extracted = filtered_words[:5]

        # Apply LABEL_MAPPING
        label_auto = best_label
        combined = (label_auto + " " + " ".join(filtered_words)).lower()
        matched_key = None
        for pattern in sorted(LABEL_MAPPING.keys(), key=len, reverse=True):
            if pattern in combined:
                matched_key = pattern
                break
        if matched_key:
            best_label = LABEL_MAPPING[matched_key]
            print(f"  LABEL_MAPPING match: '{matched_key}' -> '{best_label}'")
        elif best_score < 0.25:
            fallback = " ".join(filtered_words[:3]).title()
            print(f"  KeyBERT score rendah ({best_score:.3f} < 0.25) "
                  f"dan tidak ada mapping -> fallback: '{fallback}'")
            best_label = fallback

        print(f"  label_auto  : {label_auto} (score: {best_score:.3f})")
        print(f"  label_final : {best_label}")

        topic_labels[str(topic_id)] = {
            "label_final": best_label,
            "label_auto": label_auto,
            "score": round(best_score, 4),
            "top_words": filtered_words[:10],
            "top_words_filtered": filtered_words[:10],
            "num_docs": num_docs
        }

    # Deduplicate: append suffix from top_words_filtered when label_final collides
    seen_labels = {}
    for tid_str, info in topic_labels.items():
        base = info["label_final"]
        if base in seen_labels:
            suffix = info["top_words_filtered"][0] if info["top_words_filtered"] else str(int(tid_str)+1)
            info["label_final"] = f"{base} ({suffix.title()})"
        else:
            seen_labels[base] = tid_str

    # Logging
    print("\n[3/3] Labeling selesai.")
    print("\n" + "=" * 60)
    print("RINGKASAN HASIL LABELING")
    print("=" * 60)
    for tid_str, info in topic_labels.items():
        print(f"  Topik {int(tid_str)+1:2d}: {info['label_final']}")
    print("=" * 60)

    return topic_labels


def validate_label_uniqueness(topic_labels):
    """
    Cek apakah ada 2+ topik dengan label_final identik setelah deduplication.
    Jika ada, tampilkan top_words masing-masing untuk membantu user mengedit
    label secara manual via dashboard.
    """
    from collections import Counter
    label_counts = Counter(v['label_final'] for v in topic_labels.values())
    duplicates = {lbl: cnt for lbl, cnt in label_counts.items() if cnt > 1}

    if duplicates:
        print("\n⚠️  DUPLIKASI LABEL TERDETEKSI:")
        for lbl, cnt in duplicates.items():
            topics = [tid for tid, v in topic_labels.items() if v['label_final'] == lbl]
            for tid in topics:
                info = topic_labels[tid]
                print(f"  Topik {int(tid)+1}: label='{lbl}' | "
                      f"top_words={info['top_words_filtered'][:5]}")
        print("Gunakan menu 'Manage Labels' di dashboard untuk mengedit.\n")

    return len(duplicates) == 0


def save_topic_labels(topic_labels, output_dir):
    output_path = Path(output_dir) / "topic_labels.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(topic_labels, f, indent=2, ensure_ascii=False)

    print(f"\n[3/3] Topic labels disimpan ke: {output_path}")
    print("\n" + "="*60)
    print("RINGKASAN HASIL LABELING")
    print("="*60)
    for tid_str, info in topic_labels.items():
        print(f"  Topik {int(tid_str)+1:2d}: {info['label_final']}")
    print("="*60)


def run_auto_labeling():
    parser = argparse.ArgumentParser(description="Auto Labeling Topik LDA menggunakan KeyBERT")
    parser.add_argument("--model_path", type=str, default="model/lda_model.gensim", help="Path ke file model LDA")
    parser.add_argument("--dist_path", type=str, default="model/topic_distribution.csv", help="Path ke topic distribution CSV")
    parser.add_argument("--output", type=str, default="model/topic_labels.csv", help="Path untuk menyimpan hasil label (CSV)")

    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"LDA model tidak ditemukan: {model_path}")

    print(f"Loading LDA model dari: {model_path}")
    lda_model = LdaModel.load(str(model_path))

    print(f"Loading document titles dari: {args.dist_path}")
    topic_titles = load_document_titles(args.dist_path)

    all_stopwords = get_all_stopwords()
    print(f"Total stopwords: {len(all_stopwords)} kata")

    topic_labels = label_topics_keybert(lda_model, all_stopwords, topic_titles)

    # Save via save_topic_labels (JSON)
    save_topic_labels(topic_labels, args.model_path.rsplit('/', 1)[0] if '/' in args.model_path else 'model')

    # Also save CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels_rows = []
    for tid_str, info in topic_labels.items():
        tid = int(tid_str) + 1
        labels_rows.append({
            'topic_id': tid,
            'label': info['label_final'],
            'description': f"Topic with {info['num_docs']} documents",
            'keywords': ';'.join(info['top_words'][:5]),
            'label_score': info['score'],
            'quality_coherence': ''
        })
    df_labels = pd.DataFrame(labels_rows)
    df_labels.to_csv(output_path, index=False)
    print(f"  CSV backup -> {output_path}")


def load_document_titles(dist_path: str) -> dict:
    topic_titles = {}
    try:
        if Path(dist_path).exists():
            df = pd.read_csv(dist_path)
            if 'topik_dominan' in df.columns and 'Judul' in df.columns:
                for idx, row in df.iterrows():
                    t = int(row['topik_dominan'])
                    if t not in topic_titles:
                        topic_titles[t] = []
                    title_clean = str(row['Judul']).replace('\r', ' ').replace('\n', ' ')
                    topic_titles[t].append(title_clean)
        return topic_titles
    except Exception as e:
        print(f"Warning: Gagal meload judul dokumen: {e}")
        return {}


if __name__ == "__main__":
    run_auto_labeling()
