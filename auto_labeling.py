import json
import warnings
import sys
import argparse
import pandas as pd
from pathlib import Path
import numpy as np
from typing import List, Dict, Optional, Tuple

warnings.filterwarnings('ignore')
from gensim.models import LdaModel
from indonesian_stopwords import get_all_stopwords
from data_manager import to_one_indexed, to_zero_indexed

# ---------------------------------------------------------------------------
# EKSPOR DATASET UNTUK VALIDASI PAKAR
# ---------------------------------------------------------------------------
def export_human_validation_dataset(
    lda_model: LdaModel,
    corpus: list,
    dist_df: pd.DataFrame,
    output_path: str = "model/human_topic_validation.csv"
) -> pd.DataFrame:
    """
    Mengekspor distribusi dokumen-topik beserta 15 kata teratas per topik
    ke file CSV untuk keperluan validasi pakar.
    """
    print(f"\n[EXPERT VALIDATION EXPORTER] Exporting Human Validation Dataset...")
    rows = []
    
    # Pre-extract top 15 words per topic
    topic_top_words = {}
    for t_id in range(lda_model.num_topics):
        words = [w for w, _ in lda_model.show_topic(t_id, topn=15)]
        topic_top_words[to_one_indexed(t_id)] = ", ".join(words)
        
    for idx, row in dist_df.iterrows():
        doc_id = row.get('ID', idx)
        title = row.get('Judul', '')
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
            'expert_relevance_score_1to5': '',  # Blank column for Human Expert Audit (1-5)
            'expert_notes': ''                  # Blank column for Human Expert Feedback
        })
        
    export_df = pd.DataFrame(rows)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(out_file, index=False)
    print(f"  [HUMAN-IN-THE-LOOP] Expert validation CSV exported to: {out_file} ({len(export_df)} rows)")
    return export_df


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

def label_topics_keybert(
    lda_model: LdaModel,
    all_stopwords: set,
    topic_titles: Optional[dict] = None
) -> dict:
    """
    Labeling topik LDA menggunakan KeyBERT.
    Input utama: judul skripsi per topik (bukan top words mentah).
    Fallback: top 3 kata jika KeyBERT gagal atau score rendah.
    """
    print("\n" + "="*60)
    print("AUTO LABELING TOPIK LDA")
    print("="*60)

    MANUAL_LABEL_OVERRIDE = {
        1: "Keamanan & Manajemen Risiko Sistem",
        2: "Evaluasi Kepuasan Pengguna Sistem",
        3: "Pengembangan Sistem Informasi Manajemen",
    }

    # Muat KeyBERT
    keybert_loaded = False
    kw_model = None
    try:
        print("[1/3] Loading KeyBERT...")
        from keybert import KeyBERT
        from sentence_transformers import SentenceTransformer
        sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        kw_model = KeyBERT(model=sentence_model)
        keybert_loaded = True
        print("      KeyBERT loaded.")
    except Exception as e:
        print(f"      KeyBERT gagal: {e}. Menggunakan fallback top words.")

    print("\n[2/3] Memproses setiap topik...")
    topic_labels = {}

    for topic_id in range(lda_model.num_topics):
        tid_display = topic_id + 1
        print(f"\n  --- Topik {tid_display} ---")

        # Ambil top words dari LDA (untuk fallback dan konteks tambahan)
        raw_words = lda_model.show_topic(topic_id, topn=20)
        top_words = [
            w for w, _ in raw_words
            if w.lower() not in all_stopwords
            and len(w) > 2
            and not w.isdigit()
        ]

        # Ambil judul dokumen untuk topik ini (input utama KeyBERT)
        titles = topic_titles.get(tid_display, []) if topic_titles else []
        titles = [t.replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ').strip() for t in titles]
        print(f"  Jumlah dokumen: {len(titles)}")
        print(f"  Top words LDA : {top_words[:7]}")

        best_label = ""
        best_score = 0.0

        if keybert_loaded and kw_model:
            try:
                # INPUT KEYBERT: gabungan judul dokumen (konteks natural)
                # Ambil maks 30 judul untuk efisiensi
                titles_text = " . ".join(titles[:30]) if titles else ""
                
                # Fallback ke top words kalau tidak ada judul
                if not titles_text.strip():
                    titles_text = " ".join(top_words[:10])

                keywords = kw_model.extract_keywords(
                    titles_text,
                    keyphrase_ngram_range=(1, 2),
                    stop_words=list(all_stopwords),
                    use_mmr=True,
                    diversity=0.4,
                    top_n=5
                )

                print(f"  KeyBERT hasil : {[(k, round(s,3)) for k,s in keywords[:3]]}")

                if keywords and keywords[0][1] >= 0.25:
                    best_label = keywords[0][0].title()
                    best_score = float(keywords[0][1])

            except Exception as e:
                print(f"  KeyBERT error: {e}")

        # Fallback: gunakan 3 kata pertama dari top words LDA
        if not best_label:
            best_label = " ".join(top_words[:3]).title()
            best_score = 0.0
            print(f"  Fallback label: {best_label}")

        # Override dengan label manual jika tersedia
        if tid_display in MANUAL_LABEL_OVERRIDE:
            best_label = MANUAL_LABEL_OVERRIDE[tid_display]
            best_score = 1.0  # manual = confidence penuh

        print(f"  Label final   : {best_label} (score: {best_score:.3f})")

        topic_labels[str(topic_id)] = {
            "label_final": best_label,
            "label_auto": best_label,
            "score": round(best_score, 4),
            "top_words": top_words[:10],
            "top_words_filtered": top_words[:10],
            "num_docs": len(titles)
        }

    # Deduplikasi label jika ada yang sama
    seen = {}
    for tid_str, info in topic_labels.items():
        base = info["label_final"]
        if base in seen:
            suffix = info["top_words"][0].title() if info["top_words"] else tid_str
            info["label_final"] = f"{base} ({suffix})"
        else:
            seen[base] = tid_str

    print("\n[3/3] Labeling selesai.")
    print("="*60)
    for tid_str, info in topic_labels.items():
        print(f"  Topik {int(tid_str)+1}: {info['label_final']}")
    print("="*60)

    return topic_labels


def save_topic_labels(topic_labels, output_dir):
    output_path = Path(output_dir) / "topic_labels.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(topic_labels, f, indent=2, ensure_ascii=False)

    print(f"\n[3/3] Topic labels disimpan ke: {output_path}")


def run_auto_labeling():
    parser = argparse.ArgumentParser(description="Auto Labeling Topik LDA & Expert Validation Exporter")
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

    topic_labels = label_topics_keybert(lda_model, all_stopwords, topic_titles)

    save_topic_labels(topic_labels, args.model_path.rsplit('/', 1)[0] if '/' in args.model_path else 'model')

    # Ekspor dataset validasi pakar
    if Path(args.dist_path).exists():
        dist_df = pd.read_csv(args.dist_path)
        export_human_validation_dataset(lda_model, [], dist_df, output_path="model/human_topic_validation.csv")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels_rows = []
    for tid_str, info in topic_labels.items():
        tid = to_one_indexed(int(tid_str))
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

