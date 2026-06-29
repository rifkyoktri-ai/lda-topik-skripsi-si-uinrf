import nltk
try:
    from nltk.corpus import stopwords
    NLTK_STOPWORDS = set(stopwords.words('indonesian'))
except LookupError:
    NLTK_STOPWORDS = set()

CUSTOM_STOPWORDS_SI = {
    # ── KATA UMUM BAHASA INDONESIA ──
    "yang", "dan", "di", "ke", "dari", "dengan", "untuk", "pada", "ini", "itu", 
    "adalah", "sebagai", "dalam", "oleh", "serta", "atau", "tidak", "akan", 
    "telah", "juga", "dapat", "sudah", "ada", "hal", "cara", "lebih", "sangat", 
    "semakin", "setelah", "sebelum", "ketika", "saat", "karena", "sehingga", 
    "maka", "bahwa", "secara", "antara", "tersebut", "yaitu", "yakni", "seperti", 
    "mengenai", "tentang", "sambil", "pula", "lalu", "pun", "para", "ia", "mereka", 
    "kami", "kita", "dia", "merupakan", "terdiri", "macam", "jenis", "terdapat", 
    "bagian", "menjadi", "berbagai", "memiliki", "menggunakan", "melakukan", 
    "memberikan", "mengetahui", "berdasarkan", "terhadap", "sebagaimana", "sejumlah", 
    "sering", "sesuai", "suatu", "sebuah", "seluruh", "semua", "setiap", "sendiri",
    "sekitar", "sebagian", "serupa", "seseorang", "sesuatu", "sisi", "situ", 
    "situasi", "soal", "sore", "spesifik", "standar", "status", "strategi", 
    "struktur", "studi", "subjek", "sudut", "suka", "sumber", "serta", "seputar", 
    "selalu", "selama", "selesai", "semenjak", "sementara", "sempat", "sepanjang", 
    "seperti", "sepertinya", "seringkali", "sesudah", "seterusnya", "soalnya", 
    "suatu", "sudah", "supaya", "tadi", "tahap", "tahun", "tak", "tambah", 
    "tampak", "tampaknya", "tandas", "tanya", "tapi", "tegas", "telah", "tempat", 
    "tengah", "tentang", "tentu", "tepat", "terakhir", "terasa", "terbanyak", 
    "terdahulu", "terdiri", "terhadap", "terjadi", "terkait", "terlalu", "terlihat", 
    "termasuk", "ternyata", "terus", "tetap", "tetapi", "tiap", "tiba", "tidak", 
    "tiga", "tinggi", "toh", "tunjuk", "turut", "tutur", "ujar", "umum", 
    "umumnya", "ungkap", "untuk", "usah", "usai", "utama", "waktu", "walau", 
    "walaupun", "wong", "yaitu", "yakin", "yakni", "yang", "zaman", "awal", 
    "akhir", "baru", "lama", "mungkin", "bisa", "harus", "boleh", "ingin", 
    "mau", "pernah", "belum", "sedang", "masih", "cukup", "kurang", "lebih", 
    "paling", "sangat", "sekali", "terlalu", "amat", "begitu", "sebegini", 
    "sebegitu", "demikian", "makin", "banyak", "sedikit", "beberapa", "semua", 
    "sebagian", "rata", "masing", "sendiri", "bersama", "lain", "satu", "dua", 
    "tiga", "empat", "lima", "enam", "tujuh", "delapan", "sembilan", "sepuluh", 
    "puluh", "ratus", "ribu", "juta", "milyar", "triliun", "nol", "pertama", 
    "kedua", "ketiga", "keempat", "kelima", "keenam", "ketujuh", "kedelapan", 
    "kesembilan", "kesepuluh",

    # ── KATA UMUM AKADEMIK ──
    "penelitian", "penulis", "hasil", "data", "analisis", "berdasarkan",
    "tujuan", "dilakukan", "metode", "pengembangan", "rancang", "bangun",
    "menghasilkan", "digunakan", "dihasilkan", "output", "input", "studi",
    "kasus", "proses", "sistem", "informasi", "aplikasi", "website", "web",
    "pengguna", "user", "skripsi", "tesis", "disertasi", "jurnal", "artikel", 
    "makalah", "prosiding", "konferensi", "seminar", "laporan", "publikasi", 
    "bab", "subbab", "lampiran", "daftar", "pustaka", "referensi", "kutipan", 
    "halaman", "gambar", "tabel", "grafik", "bagan", "diagram", "rumus", 
    "persamaan", "angka", "huruf", "simbol", "kata", "kalimat", "paragraf", 
    "teks", "dokumen", "berkas", "file", "folder", "direktori",

    # ── KATA METODOLOGI PENELITIAN ──
    "kuantitatif", "kualitatif", "deskriptif", "eksploratif",
    "eksperimen", "eksperimental", "regresi", "korelasi",
    "validasi", "verifikasi", "signifikan", "populasi", "sampel",
    "responden", "partisipan", "informan", "wawancara", "observasi",
    "kuesioner", "angket", "dokumentasi", "uji", "hipotesis",
    "normalitas", "reliabilitas", "validitas", "varian", "anova",
    "pengujian", "evaluasi", "kalibrasi", "optimasi", "simulasi", 
    "prediksi", "klasifikasi", "clustering", "sintesis", "desain", 
    "rancangan", "implementasi", "pembuatan", "pembangunan", 
    "penyusunan", "penulisan", "penerapan", "penggunaan", "pemanfaatan", 
    "pengelolaan", "pemeliharaan", "perbaikan", "peningkatan", 
    "penurunan", "perubahan", "perbedaan", "persamaan", "hubungan", 
    "pengaruh", "dampak", "efek", "manfaat", "kegunaan", "sasaran", 
    "target", "indikator", "kriteria", "parameter", "variabel", 
    "faktor", "kondisi", "situasi", "keadaan", "masalah", "kendala", 
    "hambatan", "tantangan", "peluang", "risiko", "ancaman", "solusi", 
    "alternatif", "pilihan", "keputusan", "rekomendasi", "kesimpulan", 
    "saran", "pendapat", "pandangan", "tanggapan", "komentar", "ulasan", 
    "review", "penilaian", "pengukuran", "perbandingan", "riset", 
    "survei", "percobaan", "coba", "narasumber", "pakar", "ahli", 
    "klien", "pelanggan", "konsumen", "masyarakat", "publik", "warga", 
    "penduduk", "individu", "kelompok", "organisasi", "lembaga", 
    "instansi", "perusahaan", "industri", "bisnis", "pasar", "produk", 
    "jasa", "layanan", "kualitas", "mutu", "kinerja", "performa", 
    "efisiensi", "efektivitas", "produktivitas", "keamanan", "keselamatan", 
    "kesehatan", "lingkungan", "sosial", "ekonomi", "budaya", "politik", 
    "hukum", "agama", "pendidikan", "pembelajaran", "pengajaran", 
    "kurikulum", "silabus", "materi", "modul", "buku", "diktat", "handout", 
    "tugas", "ujian", "latihan", "praktikum", "laboratorium", "kelas", 
    "sekolah", "kampus", "universitas", "institut", "akademi", 
    "politeknik", "fakultas", "jurusan", "program", "mahasiswa", 
    "dosen", "guru", "siswa", "pelajar", "alumni", "lulusan", "gelar", 
    "ijazah", "sertifikat", "transkrip", "prestasi", "beasiswa", "biaya", 
    "dana", "anggaran", "keuangan", "akuntansi", "manajemen", 
    "administrasi", "operasional", "produksi", "pemasaran", "penjualan", 
    "pembelian", "persediaan", "gudang", "logistik", "distribusi", 
    "transportasi", "komunikasi", "teknologi", "sains", "rekayasa", 
    "teknik", "seni", "perencanaan", "infrastruktur", "fasilitas", 
    "sarana", "prasarana", "peralatan", "perlengkapan", "mesin", 
    "kendaraan", "bahan", "material", "manusia", "alam", "energi", 
    "ruang", "lokasi", "wilayah", "daerah", "negara", "provinsi", 
    "kabupaten", "kota", "kecamatan", "desa", "kelurahan", "rt", "rw", 
    "jalan", "gedung", "bangunan", "rumah", "kantor", "pabrik", "toko", 
    "mall", "klinik", "puskesmas", "apotek", "hotel", "restoran", "kafe", 
    "bandara", "stasiun", "terminal", "pelabuhan",

    # ── KATA PENGEMBANGAN SISTEM ──
    "tahap", "analisa", "perancangan", "implementasi", "pengujian",
    "blackbox", "whitebox", "deploy", "maintenance", "testing",
    "use", "case", "diagram", "entity", "relationship", "flowchart",
    "dfd", "uml", "basis", "database", "server", "client",
    "interface", "modul", "arsitektur", "fungsionalitas", "program", 
    "software", "hardware", "komputer", "laptop", "jaringan", "internet", 
    "situs", "beranda", "menu", "tombol", "link", "tautan", "url", 
    "domain", "hosting", "cloud", "kolom", "baris", "field", "record", 
    "query", "sql", "nosql", "algoritma", "model", "framework", 
    "library", "plugin", "komponen", "fungsi", "prosedur", "objek", 
    "tipe", "argumen", "nilai", "langkah", "fase", "siklus", "iterasi",

    # ── BOILERPLATE TEMPLATE ABSTRAK ──
    "urgensi", "dasar", "tingkat", "efisiensi", "akurasi", "performa",
    "tata", "kelola", "operasional", "kait", "laksana", "metodologi",
    "terap", "ancang", "struktur", "sistematis", "cakup", "tahap",
    "utama", "butuh", "fungsional", "non", "kumpul", "primer", "sekunder",
    "bebas", "kendala", "teknis", "harap", "kontribusi", "nyata",
    "dokumen", "rekomendasi", "analitis", "solutif", "civitas",
    "manfaat", "instrumen", "dukung", "putus", "optimal", "publik",
    "internal", "organisasi", "fokus", "topik", "kerja", "institut",
    "instansi", "tahapan", "kinerja", "guna", "pihak", "terkait",
    "luaran", "berkelanjutan", "diawali", "fase", "bersifat",
    "berjalan", "sasaran", "berupa", "akhirnya", "berkaitan",

    # ── KATA INSTITUSIONAL ──
    "universitas", "islam", "negeri", "raden", "fatah", "palembang",
    "fakultas", "sains", "teknologi", "program", "studi", "mahasiswa",
    "skripsi", "tugas", "akhir", "dosen", "pembimbing", "penguji",
    "uin", "smk", "sma", "smp", "sd", "ma", "mts", "mi",

    # ── KATA TEKNIS SI YANG TERLALU GENERIK ──
    "teknologi", "informasi", "komunikasi", "komputer", "digital",
    "elektronik", "online", "offline", "daring", "luring",
    "platform", "perangkat", "lunak", "keras", "jaringan",

    # ── ENTITAS LOKAL / STUDI KASUS (noise) ──
    "sakit", "rumah", "toko", "agama", "sungai", "pondok",
    "pesantren", "sekolah", "buku", "uang", "kota", "kabupaten",
    "kecamatan", "desa", "kelurahan", "provinsi", "wilayah",
    "daerah", "nasional", "indonesia", "sumatera", "selatan",
    "banyuasin", "ogan", "ilir", "komering", "musi", "rawas",
    "lahat", "prabumulih", "lubuklinggau", "pagaralam", "empat",
    "lawang", "muara", "kantor", "dinas", "badan", "lembaga",
    "perusahaan", "cabang", "pusat", "muhammadiyah", "nahdlatul",
    "ulama", "camat", "bupati", "walikota", "gubernur",

    # ── STOPWORD BAHASA INGGRIS UMUM ──
    "the", "and", "of", "in", "to", "a", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do",
    "does", "did", "will", "would", "could", "should", "may",
    "might", "shall", "can", "need", "dare", "ought", "used",
    "this", "that", "these", "those", "it", "its", "with",
    "for", "on", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "above", "below", "between",
    "out", "off", "over", "under", "then", "than", "so", "but",
    "or", "if", "while", "although", "because", "since", "until",
    "about", "against", "within", "without", "some", "any",
    "every", "each", "both", "few", "more", "most", "other",

    # ── KATA INGGRIS GENERIK (tidak membedakan topik SI) ──
    "information", "system", "method", "using", "based",
    "application", "result", "approach", "study", "case", "data",
    "fit", "human", "organization", "process", "function",
    "feature", "module", "research", "development", "analysis",
    "model", "value", "test", "factor", "variable",

    # ── AKRONIM & KODE TIDAK BERMAKNA ──
    "eucs", "iso", "ipa", "hot", "mus", "iii", "ii", "iv",
    "vi", "vii", "viii", "ix", "xi", "xii", "pos", "lan",
    "wan", "vpn", "http", "https", "api", "sql", "php", "css",
    "html", "xml", "json", "ram", "cpu", "gpu", "os", "pc",
    "it", "ict",
}

def get_all_stopwords():
    try:
        nltk_sw = set(stopwords.words('indonesian'))
    except LookupError:
        nltk.download('stopwords')
        nltk_sw = set(stopwords.words('indonesian'))
    return nltk_sw.union(CUSTOM_STOPWORDS_SI)


def get_conservative_stopwords():
    """
    Stopwords minimal untuk preprocessing LDA.
    HANYA kata benar-benar generik — TIDAK termasuk kata teknis domain SI
    seperti 'sistem', 'informasi', 'aplikasi', 'data', 'teknologi', dll.
    """
    return {
        # Kata tugas bahasa Indonesia
        "yang", "dan", "di", "ke", "dari", "dengan", "untuk", "pada", "ini",
        "itu", "adalah", "sebagai", "dalam", "oleh", "serta", "atau", "tidak",
        "akan", "telah", "juga", "dapat", "sudah", "ada", "hal", "lebih",
        "sangat", "semakin", "setelah", "sebelum", "ketika", "saat", "karena",
        "sehingga", "maka", "bahwa", "secara", "antara", "tersebut", "yaitu",
        "yakni", "seperti", "mengenai", "tentang", "sambil", "pula", "lalu",
        "pun", "para", "ia", "mereka", "kami", "kita", "dia", "merupakan",
        "terdiri", "macam", "jenis", "terdapat", "bagian", "menjadi", "berbagai",
        "memiliki", "sebagaimana", "sejumlah", "sering", "suatu", "sebuah",
        "seluruh", "semua", "setiap", "sendiri", "sekitar", "sebagian",
        "selalu", "selama", "selesai", "semenjak", "sementara", "sempat",
        "sepanjang", "seringkali", "sesudah", "supaya", "tadi", "tak",
        "tampak", "tapi", "telah", "tempat", "tengah", "tentu", "terakhir",
        "terhadap", "terjadi", "terkait", "terlalu", "termasuk", "ternyata",
        "terus", "tetap", "tetapi", "tiap", "tiba", "toh", "turut",
        "umum", "umumnya", "untuk", "waktu", "walau", "walaupun", "wong",
        "yaitu", "yakin", "yakni", "zaman", "awal", "akhir", "baru", "lama",
        "mungkin", "bisa", "harus", "boleh", "ingin", "mau", "pernah",
        "belum", "sedang", "masih", "cukup", "kurang", "paling",
        "amat", "begitu", "demikian", "makin", "banyak", "sedikit",
        "beberapa", "rata", "masing", "bersama", "lain",
        # Angka dasar
        "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "delapan",
        "sembilan", "sepuluh", "puluh", "ratus", "ribu", "juta", "milyar", "nol",
        "pertama", "kedua", "ketiga",
        # Kata sangat generik konteks akademik (bukan teknis SI)
        "penelitian", "penulis", "skripsi", "tugas", "akhir",
        "bab", "halaman", "tabel", "gambar", "lampiran",
        # Stopwords Inggris sangat umum
        "the", "and", "of", "in", "to", "a", "is", "are", "was",
        "were", "be", "been", "being", "have", "has", "had", "do",
        "does", "did", "will", "would", "could", "should", "may",
        "might", "shall", "can", "need", "this", "that", "these",
        "those", "it", "its", "with", "for", "on", "at", "by",
        "from", "as", "into", "through", "during", "before", "after",
        "above", "below", "between", "out", "off", "over", "under",
        "then", "than", "so", "but", "or", "if", "while", "although",
        "because", "since", "until", "about", "against", "within",
        "without", "some", "any", "every", "each", "both", "few",
        "more", "most", "other",
    }
