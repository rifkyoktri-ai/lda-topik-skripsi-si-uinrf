from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from stopwords_id import get_all_stopwords

LABEL_MAPPING = {
    "neighbor nearest"       : "K-Nearest Neighbor (KNN)",
    "nearest neighbor"       : "K-Nearest Neighbor (KNN)",
    "naive bayes"            : "Klasifikasi Naive Bayes",
    "support vector"         : "Support Vector Machine (SVM)",
    "decision tree"          : "Decision Tree Classifier",
    "random forest"          : "Random Forest",
    "neural network"         : "Jaringan Syaraf Tiruan (ANN)",
    "deep learning"          : "Deep Learning",
    "klasifikasi"            : "Klasifikasi & Data Mining",
    "prediksi"               : "Prediksi & Forecasting",
    "clustering"             : "Clustering & Pengelompokan",
    "sentimen"               : "Analisis Sentimen",
    "sentiment"              : "Analisis Sentimen",
    "mining"                 : "Data Mining",

    "extreme programming"    : "Extreme Programming (XP)",
    "rapid application"      : "Rapid Application Development (RAD)",
    "prototype"              : "Metode Prototyping",
    "agile"                  : "Agile Development",
    "scrum"                  : "Agile Scrum",

    "delone mclean"          : "Evaluasi Sistem DeLone & McLean",
    "technology acceptance"  : "Technology Acceptance Model (TAM)",
    "end user computing"     : "End User Computing Satisfaction (EUCS)",
    "eucs"                   : "End User Computing Satisfaction (EUCS)",
    "webqual"                : "Analisis Kualitas Website (WebQual)",
    "servqual"               : "Analisis Kualitas Layanan (ServQual)",
    "importance performance" : "Importance Performance Analysis (IPA)",
    "iso 9126"               : "Evaluasi Kualitas ISO 9126",
    "iso 25010"              : "Evaluasi Kualitas ISO 25010",
    "kepuasan"               : "Analisis Kepuasan Pengguna",
    "kualitas"               : "Analisis Kualitas Layanan",
    "usability"              : "Evaluasi Usabilitas Sistem",

    "beasiswa"               : "SPK Pemilihan Beasiswa",
    "spk"                    : "Sistem Pendukung Keputusan",
    "decision support"       : "Sistem Pendukung Keputusan",
    "ahp"                    : "SPK Metode AHP",
    "topsis"                 : "SPK Metode TOPSIS",
    "saw"                    : "SPK Metode SAW",

    "perpustakaan"           : "Sistem Informasi Perpustakaan",
    "library"                : "Sistem Informasi Perpustakaan",
    "akademik"               : "Sistem Informasi Akademik",
    "presensi"               : "Sistem Presensi & Absensi",
    "absensi"                : "Sistem Presensi & Absensi",
    "keuangan"               : "Sistem Informasi Keuangan",
    "inventory"              : "Sistem Manajemen Inventori",
    "stok"                   : "Sistem Manajemen Inventori",
    "kesehatan"              : "Sistem Informasi Kesehatan",
    "rekam medis"            : "Sistem Rekam Medis",
    "penjualan"              : "Sistem Informasi Penjualan",
    "peminjaman"             : "Sistem Informasi Peminjaman",
    "kepegawaian"            : "Sistem Informasi Kepegawaian",
    "pegawai"                : "Sistem Informasi Kepegawaian",
    "surat"                  : "Sistem Manajemen Surat",
    "arsip"                  : "Sistem Manajemen Arsip",
    "monitoring"             : "Sistem Monitoring",
    "audit"                  : "Audit Sistem Informasi",
    "cobit"                  : "Audit Sistem Informasi (COBIT)",

    "keamanan"               : "Keamanan Sistem & Jaringan",
    "enkripsi"               : "Keamanan & Kriptografi",
    "kriptografi"            : "Kriptografi & Enkripsi Data",
    "jaringan"               : "Jaringan Komputer",
    "network"                : "Jaringan Komputer",
    "firewall"               : "Keamanan Jaringan",

    "iot"                    : "Internet of Things (IoT)",
    "smart"                  : "Smart System & IoT",
    "gis"                    : "Geographic Information System (GIS)",
    "geographic"             : "Geographic Information System (GIS)",
    "mobile"                 : "Aplikasi Mobile",
    "android"                : "Pengembangan Aplikasi Android",
    "ecommerce"              : "E-Commerce & Transaksi Online",
    "toko online"            : "E-Commerce & Transaksi Online",
    "marketplace"            : "E-Commerce & Marketplace",
    "hukum"                  : "Sistem Informasi Hukum",
    "syariah"                : "Sistem Informasi Keuangan Syariah",
}

def apply_label_mapping(label_auto, top_words):
    combined = (label_auto + " " + " ".join(top_words)).lower()
    sorted_patterns = sorted(LABEL_MAPPING.keys(), key=len, reverse=True)
    for pattern in sorted_patterns:
        if pattern.lower() in combined:
            return LABEL_MAPPING[pattern]
    return label_auto.title()

def label_topics_with_keybert(lda_model, all_stopwords):
    print("[KeyBERT] Loading model paraphrase-multilingual-MiniLM-L12-v2...")
    print("[KeyBERT] Download ~500MB jika belum ada. Harap tunggu...")

    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    kw_model = KeyBERT(model=model)

    topic_labels = {}
    num_topics = lda_model.num_topics

    for topic_id in range(num_topics):
        print(f"[KeyBERT] Memproses Topik {topic_id + 1}/{num_topics}...")

        raw_words = lda_model.show_topic(topic_id, topn=20)

        filtered_words = [
            word for word, weight in raw_words
            if word.lower() not in all_stopwords
            and len(word) > 2
            and not word.isdigit()
        ]

        if len(filtered_words) < 3:
            topic_labels[str(topic_id)] = {
                "label_auto"  : f"Topik {topic_id + 1}",
                "score"       : 0.0,
                "top_words"   : filtered_words,
                "label_final" : f"Topik {topic_id + 1}"
            }
            continue

        doc = " ".join(filtered_words)

        try:
            keywords = kw_model.extract_keywords(
                doc,
                keyphrase_ngram_range=(2, 4),
                stop_words=list(all_stopwords),
                use_mmr=True,
                diversity=0.5,
                top_n=5
            )

            if keywords:
                best_label = keywords[0][0]
                best_score = keywords[0][1]
            else:
                best_label = " ".join(filtered_words[:3])
                best_score = 0.0

        except Exception as e:
            print(f"[KeyBERT] Warning Topik {topic_id}: {e}")
            best_label = " ".join(filtered_words[:3])
            best_score = 0.0

        label_final = apply_label_mapping(best_label, filtered_words)

        topic_labels[str(topic_id)] = {
            "label_auto"  : best_label,
            "score"       : round(best_score, 4),
            "top_words"   : filtered_words[:10],
            "label_final" : label_final
        }

        print(f"[KeyBERT] Topik {topic_id}: {label_final}")

    return topic_labels
