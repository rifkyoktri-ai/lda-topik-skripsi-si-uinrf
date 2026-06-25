import nltk
try:
    from nltk.corpus import stopwords
    NLTK_STOPWORDS = set(stopwords.words('indonesian'))
except LookupError:
    # Fallback if downloaded later
    NLTK_STOPWORDS = set()

CUSTOM_STOPWORDS_SI = {
    # Kata umum akademik yang muncul di hampir semua abstrak
    "penelitian", "sistem", "informasi", "metode", "menggunakan",
    "hasil", "data", "analisis", "berdasarkan", "tujuan",
    "dilakukan", "dapat", "dalam", "untuk", "dengan",
    "pada", "yang", "ini", "adalah", "sebagai",
    "studi", "kasus", "pengembangan", "implementasi", "rancang",
    "bangun", "berbasis", "menghasilkan", "digunakan", "dihasilkan",
    "proses", "teknik", "pendekatan", "output", "input",
    "bahwa", "telah", "akan", "dari", "atau",
    "oleh", "juga", "sehingga", "secara", "lebih",
    "sangat", "tidak", "ada", "hal", "cara",

    # Kata institusional
    "universitas", "islam", "negeri", "raden", "fatah",
    "palembang", "fakultas", "sains", "teknologi", "program",
    "studi", "mahasiswa", "skripsi", "tugas", "akhir",
    "dosen", "pembimbing", "penguji",

    # Kata teknis generik SI yang tidak membedakan topik
    "database", "aplikasi", "website", "web", "user",
    "pengguna", "interface", "fitur", "fungsi", "modul",
    "testing", "pengujian", "blackbox", "whitebox",
    "waterfall", "sdlc", "erd", "uml", "dfd",

    # ── KELOMPOK BARU 4: Entitas Lokal / Objek Studi Kasus (Noise) ──
    "sakit", "rumah", "toko", "agama", "sungai", "pondok", 
    "pesantren", "sekolah", "buku", "uang",

    # ── KELOMPOK BARU 1: Stopword Bahasa Inggris Umum ──
    # Kata Inggris generik yang sering muncul di abstrak skripsi SI
    # tapi tidak bermakna sebagai pembeda topik
    "the", "and", "of", "in", "to", "a", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do",
    "does", "did", "will", "would", "could", "should", "may",
    "might", "shall", "can", "need", "dare", "ought", "used",
    "this", "that", "these", "those", "it", "its", "with",
    "for", "on", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "above", "below", "between",
    "out", "off", "over", "under", "then", "than", "so", "but",
    "or", "if", "while", "although", "because", "since", "until",

    # Kata Inggris teknis generik yang muncul di hampir semua abstrak SI
    # tapi tidak membedakan topik
    "information", "system", "model", "method", "using", "based",
    "development", "application", "result", "analysis", "approach",
    "process", "management", "design", "study", "case", "research",
    "performance", "quality", "evaluation", "implementation",
    "framework", "testing", "user", "data", "technology", "fit",
    "human", "organization", "simple", "rapid", "unified", "rational",
    "growth", "importance", "acceptance", "computing", "end",
    "satisfaction", "process", "function", "feature", "module",

    # ── KELOMPOK BARU 2: Nama Lokasi & Institusi Lokal ──
    # Nama geografis dan institusi yang tidak bermakna sebagai topik SI
    "palembang", "sumatera", "selatan", "banyuasin", "ogan",
    "ilir", "komering", "musi", "rawas", "lahat", "prabumulih",
    "lubuklinggau", "pagaralam", "empat", "lawang", "muara",
    "kota", "kabupaten", "kecamatan", "desa", "kelurahan",
    "provinsi", "wilayah", "daerah", "nasional", "indonesia",
    "uin", "uml", "smk", "sma", "smp", "sd", "ma", "mts",
    "muhammadiyah", "nahdlatul", "ulama", "camat", "bupati",
    "walikota", "gubernur", "kantor", "dinas", "badan",
    "lembaga", "instansi", "perusahaan", "cabang", "pusat",

    # ── KELOMPOK BARU 3: Akronim & Kode Tidak Bermakna ──
    # Akronim yang terlalu spesifik dan membingungkan KeyBERT
    "eucs", "iso", "ipa", "hot", "mus", "iii", "ii", "iv",
    "vi", "vii", "viii", "ix", "xi", "xii", "spk", "dss",
    "erp", "crm", "scm", "pos", "lan", "wan", "vpn", "http",
    "https", "api", "sql", "php", "css", "html", "xml", "json",
    "ram", "cpu", "gpu", "os", "pc", "it", "ict", "ai", "ml",
}

def get_all_stopwords():
    try:
        nltk_sw = set(stopwords.words('indonesian'))
    except LookupError:
        nltk.download('stopwords')
        nltk_sw = set(stopwords.words('indonesian'))
    return nltk_sw.union(CUSTOM_STOPWORDS_SI)
