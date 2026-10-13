import json
import hashlib
import pandas as pd
import gensim

df_words = pd.read_csv('_fix_final/top_words_final.csv')
lda = gensim.models.LdaModel.load('model/lda_model.gensim')

approved_labels = {
    1: ('Audit TI & Evaluasi Kesuksesan SI', 'Sedang', 'Topik ini memuat skripsi seputar audit tata kelola IT, evaluasi kesuksesan/kualitas layanan sistem informasi instansi publik, serta analisis sentimen aplikasi.'),
    2: ('Pengembangan & Analisis Sistem Heterogen', 'Rendah', 'Topik ini berisi kelompok skripsi heterogen yang mencakup pengembangan sistem kepegawaian, data mining transaksi penjualan, SPK kelayakan kredit, serta evaluasi kesuksesan akademik.'),
    3: ('Analisis Kualitas Layanan & Risiko SI', 'Sedang', 'Topik ini berfokus pada analisis kualitas layanan aplikasi/SIAKAD, evaluasi risiko keamanan informasi, serta sentiment analysis ulasan pengguna.'),
    4: ('Kepuasan Pengguna Akhir (EUCS & TAM)', 'Tinggi', 'Topik ini secara spesifik mengelompokkan penelitian evaluasi kepuasan pengguna (EUCS) dan analisis penerimaan aplikasi digital.'),
    5: ('Manajemen Risiko TI & Analitik Data', 'Sedang', 'Topik ini memuat studi manajemen risiko teknologi informasi serta penerapan metode analitik data mentah (data mining, prediksi penjualan, dan NLP).'),
    6: ('Pengembangan & Perencanaan Strategis SI', 'Sedang', 'Topik ini mencakup skripsi mengenai perencanaan strategis sistem informasi, sistem pendukung keputusan seleksi, serta rancang bangun SI instansi pemerintah.'),
    7: ('Metode Pengembangan Perangkat Lunak & Antarmuka Pengguna', 'Rendah', 'Topik ini memuat skripsi yang berfokus pada metode pengembangan perangkat lunak, pengelolaan data, serta perancangan antarmuka pengguna. Kata kunci terkuatnya adalah extreme programming, pengelolaan data, serta user interface, desain, dan android. Label sebelumnya "Rancang Bangun SI Berbasis Web & Mobile" tidak lagi digunakan karena kata "sistem", "informasi", "aplikasi", "web", dan "implementasi" telah dihapus dari kosakata model oleh filter_extremes(no_above=0,40), sehingga sinyal rancang bangun tidak lagi tersedia secara diskriminatif.'),
    8: ('Keamanan Informasi & SI Institusi Pendidikan', 'Rendah', 'Topik ini merupakan kelompok topik berukuran paling kecil yang bersifat keranjang sisa, mencakup audit keamanan informasi, segmentasi pelanggan K-Means, dan sistem informasi sekolah.')
}

labels_list = []

for tid in range(1, 9):
    top10_words = sorted([str(w) for w in df_words[df_words['topik_id'] == tid].head(10)['word']])
    words_str = ','.join(top10_words)
    fp_hash = hashlib.sha256(words_str.encode('utf-8')).hexdigest()
    
    lbl, conf, desc = approved_labels[tid]
    labels_list.append({
        'topic_id': tid,
        'label': lbl,
        'deskripsi': desc,
        'top_words_fingerprint': fp_hash,
        'tingkat_kepercayaan': conf,
        'status': 'disetujui'
    })
    print(f'Topik {tid}: label="{lbl}", fingerprint={fp_hash[:16]}..., confidence={conf}')

with open('_label_work/labels.json', 'w', encoding='utf-8') as f:
    json.dump(labels_list, f, indent=2, ensure_ascii=False)

print('Saved _label_work/labels.json successfully.')
