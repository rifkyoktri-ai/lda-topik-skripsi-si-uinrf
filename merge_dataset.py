import csv, re, os
from collections import OrderedDict
import openpyxl

def fix_bom(rows):
    for r in rows:
        r2 = {}
        for k, v in r.items():
            r2[k.replace('\ufeff', '')] = v
        r.clear()
        r.update(r2)

# ========== 1. Load all sources ==========

# Source A: thesis_lengkap.csv (530 entries, sebagian abstrak)
with open('data/raw/thesis_lengkap.csv', 'r', encoding='utf-8') as f:
    data_a = list(csv.DictReader(f))
fix_bom(data_a)

# Source B: thesis_2025.csv (92 entries, 88 abstrak)
with open('data/raw/thesis_2025.csv', 'r', encoding='utf-8') as f:
    data_b = list(csv.DictReader(f))
fix_bom(data_b)

# Source C: abstrak_30.csv (30 entries, ~12 abstrak)
with open('data/raw/abstrak_30.csv', 'r', encoding='utf-8') as f:
    data_c = list(csv.DictReader(f))
fix_bom(data_c)

# ========== 2. Build lookup for abstracts from B and C ==========

abstrak_lookup = {}

for r in data_b:
    ab = r.get('abstract', '').strip()
    rid = r.get('id', '').strip()
    if ab and ab not in ('ABSTRAK_TIDAK_ADA',) and not ab.startswith('FETCH') and not ab.startswith('ERROR'):
        abstrak_lookup[rid] = ab

for r in data_c:
    ab = r.get('abstract', '').strip()
    rid = r.get('id', '').strip()
    if ab:
        abstrak_lookup[rid] = ab

# ========== 3. Manual abstracts from user ==========

manual_abstracts = {
    '15203': 'Sistem Informasi Akademik adalah suatu sistem yang dibangun untuk mengelola data-data akademik sehingga memberikan kemudahan kepada pengguna dalam kegiatan administrasi akademik. Penelitian ini bertujuan untuk mengetahui tingkat kepuasan pengguna Sistem Informasi Akademik Universitas Tridinanti Palembang. Jumlah responden penelitian ini adalah 373 orang yang terdiri dari mahasiswa dan dosen. Analisis kepuasan yang dilakukan menggunakan variabel kepuasan pengguna yaitu End User Computing Satisfaction (EUCS) dan diukur dengan metode Kano. Metode Kano digunakan untuk mengukur dan mengkategorikan atribut berdasarkan seberapa baik atribut tersebut mampu memuaskan kebutuhan pengguna. Hasil dari penelitian ini menyatakan bahwa dari 6 kategori yang ada pada metode Kano hanya ada 3 kategori yang muncul yaitu kategori Must be, One Dimensional dan Attractive. Atribut yang paling berpengaruh terhadap peningkatan kepuasan dan ketidakpuasan pengguna dapat diketahui dari hasil analisa nilai better worse. Nilai kepuasan tertinggi yaitu pada atribut F2, Sistem informasi akademik memiliki struktur menu dan link yang mudah dipahami. Nilai ketidakpuasan tertinggi yaitu pada atribut C3, Sistem informasi akademik memiliki layanan layanan yang mendukung proses akademik secara lengkap.',
    '15204': 'Pengolahan Data Pada PT Surya Bumi Agrolanggeng yang berlokasi di Simpang Tais Kabupaten Pali masih dilakukan secara konvensional atau manual dengan menggunakan media kertas, Permasalahan yang timbul saat ini adalah sulitnya dalam melakukan cek sisa cuti yang tersedia, sistem pengingat daftar pegawai yang akan pensiun, serta penumpukan berkas yang diajukan pegawai naik pangkat, sehingga mengakibat kehilangan data,dan menyulitkan pencarian berkas. Metode yang digunakan dalam pengembangan sistem yaitu waterfall dan menggunakan alat bantu pengembangan sistem yaitu Flowchart serta Unified Modeling Language (UML) untuk memodelkan kebutuhan sistem yang terdiri dari use case diagram, sequence diagram, activity diagram dan class diagram. Menghasilkan sistem informasi kepegawaian yang dapat membantu kepegawaian seperti cek sisa cuti, pengajuan cuti, pengajuan mutasi, pengajuan naik pangkat, pengajuan pensiun, memperudah memperoleh informasi. serta didalam kepegawaian mempermudah proses pencarian, penyimpanan, perubahan data yang diinginkan. Sistem informasi ini dibangun dengan menggunakan framework CodeIgniter bahasa pemrograman PHP dan MySQL sebagai databasenya.',
    '15256': 'Sistem informasi E-KKN merupakan salah satu sistem informasi di lembaga penelitian dan pengabdian masyarakat(LP2M) Universitas Islam Negeri Raden Fatah Palembang yang memberikan pelayanan dan informasi mengenai kuliah kerja Nyata (KKN). Penelitian ini bertujuan untuk mengukur tingkat kesuksesan sistem informasi E-KKN berdasarkan pengaruh antar variabel pada model kesuksesan. Model kesuksesan yang digunakan menggunakan model Delone and Mclean IS Sucess Model. Hasil penelitian menunjukkan bahwa kualitas sistem dan kualitas pelayanan berpengaruh signifikan terhadap pemakaian, kuliatas informasi berpengaruh signifikan terhadap kepuasan pemakai, pemakaian berpengaruh signifikan terhadap kepuasan pemakai, dan kepuasan pengguna berpengaruh signifikan terhadap manfaat-manfaat bersih.',
    '15400': 'Dalam rangka meningkatkan kualitas pelayanan perusahaan, AJB Bumiputera 1912 Cabang Sekip Palembang membutuhkan suatu sistem pendukung yang dapat mempermudah dalam proses kegiatan layanan asuransi bagi nasabah atau pemegang polis-nya. selain itu, sistem ini juga diharapkan dapat meminimalisir biaya opersional perusahaan yang selama ini menjadi sebab utama menurunnya omset keuntungan bagi perusahaan asuransi di Indonesia. Layanan asuransi ini bertujuan untuk menarik minat para calon nasabah asuransi agar mau mengansuransi-kan diri baik jiwa, pendidikan maupun hari tuanya, yang selama ini enggan untuk ikut asuransi dengan alasan proses pencairan dana (klaim) yang lama atau proses penerbitan polis yang terkesan sulit. Penelitian ini bertujuan untuk merancang dan membangun Sistem Informasi Layanan Asuransi pada AJB Bumiputera 1912 Palembang yang mampu mempermudah proses penerbitan polis, angsuran premi, pengajuan klaim asuransi dan menjadi media promosi bagi produk-produk asuransi perusahaan serta diharapkan mampu memanajemen kegiatan agen asuransi dan pemegang polis. Metode penelitian yang digunakan yaitu metode penelitian kualitatif dengan metode pengumpulan data yaitu observasi, wawancara, studi kepustakaan, dan dokumentasi. Metode pengembangan sistem yang digunakan yaitu Prototype. Alat bantu analisis dan perancangan yang digunakan yaitu Data Flow Diagram (DFD), Use case Diagram, Class Diagram, Activity Diagram, Squence Diagram, serta perancangan database menggunakan Entity Relationship Diagram (ERD), Bahasa pemrograman yang digunakan yaitu Hypertext Preprocessor (PHP), Java (Eclips IDE), Framework Codeigniter serta database menggunakan MySQL, Metode pengujian menggunakan Blackbox testing. Diharapkan sistem dapat membantu kegiatan layanan asuransi terutama dalam kegiatan pengajuan klaim dan penerbitan polis asuransi.',
    '15417': 'Pada teknologi yang sangat pesat hampir setiap orang memiliki smartphone, mempermudah dalam mengakses, mengolah data, juga dalam berkomunikasi yang tidak lagi dibatasi oleh jarak dan waktu bahkan tempat yang jauh sekalipun, seiring dengan tingkat mobilitas yang tinggi, penelitian aplikasi tes buta warna berbasis mobilephone android yang biasanya hanya dapat dilakukan pada keperluan tertentu saja seperti tes kerja dan tes masuk dalam perguruan tinggi tertentu, sistem ini dibuat bertujuan memudahkan pengguna dalam memeriksa kelainan buta warna secara mandiri tanpa harus pergi ke dokter, Dalam pembuatan analisis dan perancangan aplikasi ini menggunakan metode pengembangan sistem, metode yang digunakan yaitu OOAD (Object Orientied Analysis Design) adalah metode model proses analisa dan perancangan berorientasi objek, kemudian melakukan analisis sistem dengan menggunakan pemodelan UML (Unified Modelling language) dimana perancangan sistem merupakan syarat untuk melakukan pengembangan dan memberikan gambaran secara umum kepada pengguna tentang sistem yang akan dibangun, perancangan sistem dalam pembuatan aplikasi tes buta warna ini terdiri dari Use Case Diagram, Activity Diagram dan Sequance Diagram, Class Diagram.',
    '15465': '',
    '15504': 'Sekolah Dasar Islam Terpadu Fathona Palembang banyak mengalami kesulitan terutama dalam mengolah data administrasi seperti laporan nilai siswa, data siswa, data guru, serta data inventaris sekolah. Administrasi pendidikan bertujuan untuk mendukung tercapainya semua tujuan kegiatan yang dilaksanakan pada suatu instansi pendidikan. Penelitian ini bertujuan untuk merancang dan membangun Sistem Informasi Administrasi pada SDIT Fathona Palembang agar mempermudah proses membuat laporan dan pengolahan data, serta memberikan informasi yang dibutuhkan. Metode penelitian yang digunakan yaitu metode penelitian kualitatif dengan metode pengumpulan data yaitu observasi, wawancara, studi kepustakaan, dan dokumentasi. Metode pengembangan sistem yang digunakan yaitu Prototipe. Alat bantu analisis dan perancangan yang digunakan yaitu Data Flow Diagram (DFD) serta perancangan database menggunakan Entity Relationship Diagram (ERD), Bahasa pemrograman yang digunakan yaitu Hypertext Preprocessor (PHP), serta database menggunakan MySQL, Metode pengujian menggunakan Blackbox testing. Diharapkan sistem dapat membantu kegiatan administrasi sekolah terutama dalam kegiatan laporan nilai siswa dan inventaris.',
    '15540': 'Analisis kesuksesan pada LP2M UIN Raden Fatah Palembang dilakukan untuk menganalisis kesuksesan sistem informasi E-KKN UIN Raden Fatah Palembang berdasarkan persepsi pengguna dengan menggunakan metode IS Impact Measurement Model. Proses analisis E-KKN UIN Raden Fatah Palembang dilakukan dengan 2 tahapan yaitu teknik uji instrumen dan pengukuran menggunakan model IS Impact, dengan beberapa faktor variabel yaitu; Individual Impact, Organizational Impact, System Quality, Information Quality untuk mengetahui sejauh mana dampak dari sistem informasi e-kkn terhadap pengguna dan organisasi. Proses pengukuran menggunakan metode IS Impact melibatkan 372 responden, dimana hasil pengukuran dampak individu, dampak organisasi, kualitas informasi mendapatkan hasil yang cukup baik, namun untuk kualitas sistem mendapatkan hasil kurang baik.',
    '15552': 'PT. Fastrata Buana Palembang merupakan salah satu bisnis unit dari PT. Kapal Api Global yang bergerak dalam bidang distributor consumer goods yang bertugas untuk mendistributorkan barang - barang yang ada ke seluruh wilayah Indonesia. Saat ini dalam kegiatan penjualannya PT. Frastrata Buana cabang Palembang masih menggunakan cara konvensional, keberadaan sistem yang digunakan selama ini belum dapat dijadikan sarana yang baik karena belum adanya suatu program komputer khusus seperti contoh proses pendataan, penjualan dan laporan terlebih dahulu dicatat secara manual dan disimpan dibuku besar hal ini menyebabkan terkadang adanya ketidakakuratan data saat akan dilakukan pendataan ulang dan juga tidak tersusun secara rapi. Tujuan penelitian ini adalah membuat sistem informasi penjualan barang yang dapat mengelola data produk, data pelanggan dan data penjualan barang. Dalam penelitian ini peneliti menggunakan metode Prototype sebagai metode pengembangan dan permodelan menggunakan UML (Unified Model Language). Sedangkan pembuatan aplikasinya sendiri menggunakan bahasa pemrograman PHP dengan MySQL untuk pengolahan Database. Dengan adanya sistem informasi penjualan berbasis website, maka diharapkan dapat membatu PT. Fastrata Buana cabang Palembang dalam mengelola data penjualan serta dapat menjadi sarana informasi dan promosi bagi perusahaan.',
    '15836': 'Persaudaraan Setia Hati Terate (PSHT) atau yang dikenal dengan SH Terate adalah suatu persaudaraan perguruan silat yang bertujuan mendidik dan membentuk manusia berbudi luhur, tahu benar dan salah, bertakwa kepada Tuhan Yang Maha Esa, mengajarkan kesetiaan pada hati sanubari sendiri serta mengutamakan persaudaraan antar warga (anggota) dan berbentuk sebuah organisasi yang merupakan rumpun atau aliran Persaudaraan Setia Hati (PSH), masalah yang dihadapi oleh pengurus pencak silat ini mengingat banyaknya anggota, jarak tiap tempat latian, kegiatan jauh dari pengurus cabang adalah kurangnya informasi pendataan data ranting, hasil pencatatan yang dilakukan pihak Pencak Silat Persaudaraan Setia Hati Terate tidak dapat diakses oleh pengurus dan anggota secara luas. Sistem informasi pengelolaan ini dapat mempermudah warga pengurus dalam mengelola data yang berkaitan dengan administrasi pengelolaan kegiatan yaitu data ranting, rayon, siswa yang akan mengikuti kegiatan. Sistem ini juga membantu ketua untuk melihat dan mencetak laporan-laporan serta mengelola data pengguna. Dalam penelitian ini menggunakan metode prototype sebagai metode pengembangan sistem dan pemodelan menggunakan DFD (Data Flow Diagram). Sedangkan pembuatan aplikasinya menggunakan bahasa pemprograman PHP. Hasil dari penelitian ini adalah menghasilkan informasi yang mempermudah pengurus warga cabang, ranting maupun rayon dalam proses administrasi pengelolaan kegiatan serta membantu tiap anggota untuk mudah dalam mengelola data administrasi agenda kegiatan siswa ataupun warga dalam organisasi.',
    '15952': '',
    '16159': 'PT Kereta Api Indonesia (KAI) perlu mengembangkan sistem yang menunjang proses penjualan tiket secara elektronik (e-ticketing), karena meskipun penggunaannya sudah relatif tinggi namun masih terdapat beberapa kekurangan. Penelitian ini menganalisis keberhasilan penggunaan sistem e-ticketing Rail Ticket System (RTS) pada PT. Kereta Api Indonesia (Persero) Divisi Regional III Palembang dengan menggunakan metode Human, Organization, Technology (HOT) Fit Model. Tujuan dari penelitian ini adalah untuk mengetahui seberapa besar pengaruh antar variabel human, organization, technology, dan net benefit serta kesesuaian di antaranya. Hasil evaluasi dari penelitian ini yang menggunakan sampel 15 responden menunjukkan bahwa penerapan sistem informasi e-ticketing sudah sangat tepat serta telah memberikan hasil yang positif terbukti dengan nilai uji korelasi dari nilai masing-masing variabel yaitu teknologi berpengaruh positif terhadap manusia dengan tingkat korelasi 0,952. Teknologi berpengaruh positif terhadap organisasi dengan tingkat korelasi 0,883. Manusia berpengaruh positif terhadap organisasi dengan tingkat korelasi 0,913. Dan manusia serta organisasi berpengaruh positif terhadap net benefit dengan masing-masing tingkat korelasi 0,868 dan 0,848. Hasil evaluasi menunjukkan bahwa sistem informasi Rail Ticket System (RTS) berhasil dimanfaatkan.',
    '16227': 'Wom Finance merupakan perusahaan yang bergerak pada bidang menimjaman dana maupun kredit, baik kredit sepeda motor, mobil, maupun dana tunai dengan jaminan BPKB. Wom Finance pada saat ini dalam pengambilan keputusan kredit yang dilakukan dengan langkah yang rumit yaitu calon konsumen mengajukan kredit pada Wom Finance dengan memenuhi dokumen dokumen yang diajukan, maka pihak pihak finance akan melakukan pengecekan dokumen permohonan yang telah ditentukan melalui interview dan survei lapangan. Untuk memutuskan pemberian Kredit pihak finance harus menseleksi sebaik mungkin untuk menghidari kredit macet. Seiring dengan adanya kemajuan teknologi, banyak aplikasi perbankan yang dapat dikembangkan secara komputerisasi. Sehingga pemrosesan data tidak hanya dapat dilakukan secara manual tetapi dapat dilakukan secara komputerisasi, dan hal ini dapat memberikan kemudahan bagi pihak finance, yaitu meminimalkan waktu pemrosesan data dan mengurangi terjadinya tunggakan konsumen kredit pada Wom Finance. Sistem Pendukung Keputusan Kelayakan Kredit Berbasis Web pada Wom Finance Palembang Menggunakan Metode TOPSIS (Technique For Others Reference by Similarity to Ideal Solution) dibangun dengan menggunakan perancangan Data Flow Diagram (DFD), metode pengembangan sistem waterfall, dan menggunakan bahasa pemrograman PHP, dan metode SPK Metode TOPSIS. Sistem ini diharapkan membantu pihak Wom Finance dalam pengajuan kredit.',
    '26995': 'Wedding Organizer Salon Bunda Fitry adalah sebuah jasa yang berfungsi khusus untuk melayani calon pengantin dan keluarga dalam perencanaan sampai pelaksanaan rangkaian acara pesta pernikahan. Sistem yang sedang berjalan di wedding organizer salon bunda fitry ini masih menggunakan sistem manual yaitu proses data pemesanan dan informasi paket masih menggunakan pembukuan sehingga masih sering mengalami masalah pada data pemesanan dan harga paket pernikahan. Setelah mengetahui permasalahan yang ada, maka dirancang sebuah sistem informasi pemesanan wedding organizer pada salon bunda fitry menggunakan metode pengembangan waterfall yang terdiri dari analisa, desain, pengodean, pengujian dan dan tahap pendukung, kemudian di implementasikan dengan bahasa pemrograman PHP dan MySQL sebagai database-nya. Tujuan dari penelitian ini adalah untuk memberikan kemudahan kepada karyawan wedding organizer dalam memproses data pemesanan dan harga serta memperluas area promosi penjualan paket pernikahan secara terkomputerisasi. Hasil dari penelitian ini adalah sebuah sistem informasi pemesanan wedding organizer yang dapat membantu pihak wedding organizer dan pengguna dalam memperoleh informasi secara cepat dan jelas, serta membantu operasional wedding organizer agar berjalan dengan lebih efektif dan efisien.',
    '31512': '',
    '46535': 'Pemanfaatan website sudah banyak digunakan, seperti pada Kantor Badan Amil Zakat Nasional (Baznas) Kota Palembang yang telah memanfaatkan website sebagai media informasi layanan publik. Kendala yang ada saat ini yaitu terdapat informasi yang tidak sesuai, navigasi yang tidak responsif, dan interaksi layanan yang buruk. Tujuan penelitian ini ialah untuk melakukan analisis pada kualitas Website menggunakan metode Webqual 4.0 dan Importance Performance Analysis yang berprioritas pada usability, information quality, service interaction quality dan overall, serta menghitung nilai harapan dan persepsi dari kualitas website. Data penelitian didapat dari 100 responden dengan menggunakan kuesioner. Hasil penelitian ini menghasilkan nilai harapan kualitas website Baznas Kota Palembang sebesar 4,36 sedangkan nilai persepsi kualitas website Baznas Kota Palembang sebesar 3,47 maka didapatlah nilai kualitas website sebesar -0,89 dari hasil tersebut menyatakan kualitas website Baznas Kota Palembang belum memenuhi harapan pengguna website, dan metode Importance Performance Analysis digunakan sebagai analisis yang berupa tingkat kesesuaian dan kuadran. Hasil analisis kuadran IPA didapatlah 6 atribut yang masuk kedalam kuadran A, 9 atribut pada kuadran B, 7 atribut pada kuadran C, dan 1 atribut pada kuadran D. Penelitian ini diharapkan dapat memberi masukan bagi pihak Baznas Kota Palembang untuk meningkatkan kualitas website kedepannya dan dapat memberikan website yang berkualitas kepada pengguna website Baznas Kota Palembang.',
    '48083': 'Perpustakaan merupakan lembaga yang menunjang pembelajaran siswa dan guru. Koleksi perpustakaan merupakan buku teks yang dijadikan acuan dalam proses pembelajaran. Pelayanan yang tersedia diperpustakaan MA Babussalam Payaraman masih menggunakan cara yang manual, seperti mencatat laporan peminjaman dan pengembalian buku yang ditulis disebuah buku besar serta mencatat ketersediaan buku yang jumlah cukup banyak yang tulis dibuku satu persatu. Pengelolaan perpustakaan yang masih dilakukan secara manual dapat menyulitkan dalam melakukan proses peminjaman dan pengembalian buku karena sering terjadi kekeliruan dan rentan terjadinya masalah. Untuk mengatasi masalah tersebut, maka dibuatlah sistem informasi perpustakaan ini untuk memudahkan petugas perpustakaan dalam mengelola peminjaman dan pengembalian buku, serta mengelola ketersediaan buku dan pembuatan laporan. Metode pengembangan yang dipakai pada penelitian diperpustakaan ini adalah metode Rapid Application Development yang memiliki tiga tahapan, yaitu merencanakan kebutuhan, mendesain sistem dan mengimplementasi sistem. Penelitian ini menghasilkan sebuah sistem informasi perpustakaan yang dapat memudahkan petugas perpustakaan dalam proses peminjaman dan pengembalian buku, serta membantu dalam pembuatan laporan dan mengelola ketersediaan buku secara sistematis.',
    '50640': 'PT Pupuk Sriwidjaja Palembang merupakan perusahaan besar yang beroperasi di sektor industri pupuk dan bahan kimia. Salah satu aspek penting dalam menjaga standar operasional dan kepatuhan terhadap regulasi adalah pelaksanaan audit internal. Namun, pengelolaan audit internal bagian sistem manajemen perusahaan ini masih menghadapi sejumlah tantangan, seperti sistem pengarsipan yang kurang tertata, proses audit yang memerlukan waktu lebih lama dari seharusnya, serta koordinasi monitoring yang belum optimal. Sebagai solusi untuk mengatasi kendala tersebut, digitalisasi melalui penerapan sistem informasi berbasis metode Rapid Application Development (RAD) menjadi pilihan strategis. Pendekatan RAD memungkinkan pengembangan sistem yang lebih cepat dan fleksibel dengan melibatkan masukan langsung dari pengguna, sehingga dapat lebih sesuai dengan kebutuhan perusahaan. Sistem informasi yang dirancang ini bertujuan untuk menyediakan platform terintegrasi yang mendukung pengarsipan data audit secara digital, mempercepat proses audit melalui otomatisasi, serta mempermudah monitoring dan pelaporan secara real-time. Dengan diimplementasikannya sistem informasi ini, pengelolaan arsip audit menjadi lebih terstruktur, waktu yang dibutuhkan dalam pelaksanaan audit dapat dipersingkat, dan koordinasi antar pengguna berlangsung lebih efektif. Hal ini secara keseluruhan berkontribusi pada peningkatan kualitas audit internal terhadap sistem manajemen di PT Pupuk Sriwidjaja Palembang.',
}

# ========== 4. Merge all data ==========

# Build final dataset for 2021-2025
target_years = {'2021', '2022', '2023', '2024', '2025'}
final_data = []

seen_ids = set()

for r in data_a:
    rid = r.get('id', '').strip()
    year = r.get('year', '').strip()
    title = r.get('title', '').strip()
    abstract = r.get('abstract', '').strip()

    if year not in target_years:
        continue
    if rid in seen_ids:
        continue
    seen_ids.add(rid)

    # Apply better abstract: manual > lookup > original
    if rid in manual_abstracts:
        ab = manual_abstracts[rid].strip()
        source = 'manual' if ab else 'tidak_ada'
    elif rid in abstrak_lookup:
        ab = abstrak_lookup[rid]
        source = 'repository'
    elif abstract and abstract not in ('ABSTRAK_TIDAK_ADA',) and not abstract.startswith('FETCH') and not abstract.startswith('ERROR'):
        ab = abstract
        source = 'repository'
    else:
        ab = ''
        source = 'tidak_ada'

    final_data.append({
        'id': rid,
        'year': year,
        'title': title,
        'abstract': ab,
        'source_abstrak': source
    })

print(f'Total data 2021-2025: {len(final_data)}')

with_ab = sum(1 for r in final_data if r['abstract'])
without_ab = len(final_data) - with_ab
print(f'Dengan abstrak: {with_ab}')
print(f'Tanpa abstrak: {without_ab}')

print(f'\nPer tahun:')
for y in sorted(target_years):
    yr = [r for r in final_data if r['year'] == y]
    wa = sum(1 for r in yr if r['abstract'])
    print(f'  {y}: {len(yr)} total, {wa} abstrak')

print(f'\nSumber abstrak:')
from collections import Counter
c = Counter(r['source_abstrak'] for r in final_data)
for k, v in c.most_common():
    print(f'  {k}: {v}')

# ========== 5. Save to Excel ==========

os.makedirs('data/processed', exist_ok=True)

wb = openpyxl.Workbook()
ws = wb.active
ws.title = 'Dataset 2021-2025'

# Header
headers = ['ID', 'Tahun', 'Judul', 'Abstrak', 'Sumber Abstrak']
ws.append(headers)

# Data
for r in final_data:
    ws.append([r['id'], r['year'], r['title'], r['abstract'], r['source_abstrak']])

# Auto-width
for col in ws.columns:
    max_length = 0
    col_letter = col[0].column_letter
    for cell in col:
        try:
            if cell.value:
                max_length = max(max_length, min(len(str(cell.value)), 80))
        except:
            pass
    ws.column_dimensions[col_letter].width = max(max_length + 2, 12)

excel_path = 'data/processed/dataset_lengkap_2021_2025.xlsx'
wb.save(excel_path)
print(f'\nExcel saved: {excel_path}')

# Also save CSV for pipeline
csv_path = 'data/processed/dataset_lengkap_2021_2025.csv'
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['id', 'year', 'title', 'abstract', 'source_abstrak'])
    w.writeheader()
    w.writerows(final_data)
print(f'CSV saved: {csv_path}')

# Also save just id + title + abstract for pipeline (teks_gabung will be created later)
pipe_path = 'data/processed/thesis_for_pipeline.csv'
with open(pipe_path, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['id', 'year', 'title', 'abstract'])
    w.writeheader()
    for r in final_data:
        w.writerow({'id': r['id'], 'year': r['year'], 'title': r['title'], 'abstract': r['abstract']})
print(f'Pipeline CSV saved: {pipe_path}')
