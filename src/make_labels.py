# -*- coding: utf-8 -*-
"""
src/make_labels.py - SUMBER KEBENARAN nama 8 topik.

Berisi daftar label, tingkat keyakinan, dan deskripsi yang telah disetujui
untuk setiap topik. Skrip ini membangun outputs/model/labels.json, yang
kemudian dipakai oleh src/labeling.py untuk diekspor ke topic_labels.csv/.json
dan oleh seluruh dashboard.

Kenapa file ini penting:
    Fingerprint SHA-256 tiap label dikunci terhadap top-10 kata kunci model.
    Kalau model berubah, validate_labels() akan menolak dengan jelas. Ini
    yang mencegah nama topik "melayang" ke model berbeda tanpa disadari.

Catatan desain (perbaikan):
    1. Semua path lewat src/config.py - sebelumnya ditulis langsung
       ('model/lda_model.gensim'), yang rusak setelah folder dipindah.
    2. Fingerprint dihitung dengan fungsi yang SAMA dengan yang dipakai
       src/topic_labels.py - sebelumnya dihitung dua kali dari dua sumber
       berbeda.
    3. Travelled dalam fungsi main() + main guard, bukan di level modul,
       supaya bisa di-import tanpa memuat model.
    4. Mendukung --out supaya bisa diuji tanpa menyentuh labels.json asli.

Jalankan:
    python -m src.make_labels
    python src/make_labels.py --out /tmp/labels_uji.json
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

# Pastikan bisa jalan sebagai skrip berdiri sendiri maupun sebagai modul
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as C
from src.topic_labels import compute_topic_fingerprint


# ── LABEL YANG TELAH DISETUJUI ───────────────────────────────────────────────
# (label, tingkat_keyakinan, deskripsi)
#
# Topik 7 pernah diubah DUA KALI setelah pengujian terhadap kata kunci asli
# model:
#   1. "Rancang Bangun SI Berbasis Web & Mobile"
#   2. "Metode Pengembangan Perangkat Lunak & Antarmuka Pengguna"
#   3. "Penerapan Metode & Framework Sistem Informasi"
# Komponen "Antarmuka Pengguna" pada versi 2 dicabut karena tidak didukung:
# hanya 2 dari 33 dokumen yang menyebut UI/interface, dan salah satunya
# ("SI Percetakan Total Desain") adalah fashion, bukan antarmuka.
APPROVED_LABELS = {
    1: ('Audit TI & Evaluasi Kesuksesan SI',
        'Sedang',
        'Topik ini memuat skripsi seputar audit tata kelola IT, evaluasi kesuksesan/kualitas layanan sistem informasi instansi publik, serta analisis sentimen aplikasi.'),
    2: ('Pengembangan & Analisis Sistem Heterogen',
        'Rendah',
        'Topik ini berisi kelompok skripsi heterogen yang mencakup pengembangan sistem kepegawaian, data mining transaksi penjualan, SPK kelayakan kredit, serta evaluasi kesuksesan akademik.'),
    3: ('Analisis Kualitas Layanan & Risiko SI',
        'Sedang',
        'Topik ini berfokus pada analisis kualitas layanan aplikasi/SIAKAD, evaluasi risiko keamanan informasi, serta sentiment analysis ulasan pengguna.'),
    4: ('Kepuasan Pengguna Akhir (EUCS & TAM)',
        'Tinggi',
        'Topik ini secara spesifik mengelompokkan penelitian evaluasi kepuasan pengguna (EUCS) dan analisis penerimaan aplikasi digital.'),
    5: ('Manajemen Risiko TI & Analitik Data',
        'Sedang',
        'Topik ini memuat studi manajemen risiko teknologi informasi serta penerapan metode analitik data mentah (data mining, prediksi penjualan, dan NLP).'),
    6: ('Pengembangan & Perencanaan Strategis SI',
        'Sedang',
        'Topik ini mencakup skripsi mengenai perencanaan strategis sistem informasi, sistem pendukung keputusan seleksi, serta rancang bangun SI instansi pemerintah.'),
    7: ('Penerapan Metode & Framework Sistem Informasi',
        'Rendah',
        'Topik ini mengelompokkan skripsi yang menerapkan metode, framework, atau algoritma bernama pada masalah sistem informasi, seperti extreme programming, Rapid Application Development, CodeIgniter, Zachman Framework, Naive Bayes, SERVQUAL, dan Technology Acceptance Model. Dari 33 dokumen pada topik ini, 21 dokumen (64%) menyebut metode, framework, atau algoritma secara eksplisit dan 6 dokumen secara khusus memakai extreme programming sebagai kata kunci peringkat pertama topik. Label sebelumnya "Rancang Bangun SI Berbasis Web & Mobile" tidak lagi digunakan karena kata "sistem", "informasi", "aplikasi", "web", dan "implementasi" telah dihapus dari kosakata model oleh filter_extremes(no_above=0,40), sehingga sinyal rancang bangun tidak lagi tersedia secara diskriminatif.'),
    8: ('Keamanan Informasi & SI Institusi Pendidikan',
        'Rendah',
        'Topik ini merupakan kelompok topik berukuran paling kecil yang bersifat keranjang sisa, mencakup audit keamanan informasi, segmentasi pelanggan K-Means, dan sistem informasi sekolah.'),
}


def fingerprint_dari_model(lda_model, topik: int) -> str:
    """
    SHA-256 dari top-10 kata kunci topik, diurutkan A-Z.

    Memakai fungsi yang sama dengan validate_labels() supaya hanya ada SATU
    definisi sidik jari di seluruh repo.
    """
    return compute_topic_fingerprint(lda_model, topik - 1, topn=10)


def bangun_labels(lda_model, num_topics: int = 8) -> list:
    """Susun daftar label lengkap dengan fingerprint dari model."""
    if num_topics != len(APPROVED_LABELS):
        raise ValueError(
            "Jumlah label disetujui (%d) tidak cocok dengan jumlah topik "
            "model (%d). label topik mungkin belum ditulis."
            % (len(APPROVED_LABELS), num_topics)
        )

    hasil = []
    for tid in range(1, num_topics + 1):
        if tid not in APPROVED_LABELS:
            raise KeyError("Belum ada label yang disetujui untuk Topik %d" % tid)
        label, keyakinan, deskripsi = APPROVED_LABELS[tid]
        hasil.append({
            "topic_id": tid,
            "label": label,
            "deskripsi": deskripsi,
            "top_words_fingerprint": fingerprint_dari_model(lda_model, tid),
            "tingkat_kepercayaan": keyakinan,
            "status": "disetujui",
        })
    return hasil


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Bangun outputs/model/labels.json dari label yang "
                    "telah disetujui")
    parser.add_argument("--out", type=str, default=None,
                        help="tulis ke path lain (dipakai untuk pengujian)")
    parser.add_argument("--quiet", action="store_true",
                        help="hanya tampilkan ringkasan")
    args = parser.parse_args(argv)

    from gensim.models import LdaModel

    if not C.LDA_MODEL.exists():
        print("ERROR: model LDA tidak ditemukan: %s" % C.LDA_MODEL,
              file=sys.stderr)
        print("       Jangan menjalankan src/modeling.py - nilai C_V resmi "
              "tidak reproducible.", file=sys.stderr)
        return 1

    lda = LdaModel.load(str(C.LDA_MODEL))
    labels = bangun_labels(lda, lda.num_topics)

    tujuan = Path(args.out) if args.out else C.LABELS_JSON
    tujuan.parent.mkdir(parents=True, exist_ok=True)
    with open(tujuan, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)

    if not args.quiet:
        for d in labels:
            print('  Topik %d: %-46s keyakinan=%-6s sidik=%s...'
                  % (d["topic_id"], d["label"][:46],
                     d["tingkat_kepercayaan"], d["top_words_fingerprint"][:16]))
    print("\n  [OK] %d label ditulis ke %s"
          % (len(labels), tujuan.relative_to(C.REPO)
             if C.REPO in tujuan.parents else tujuan))
    return 0


if __name__ == "__main__":
    sys.exit(main())