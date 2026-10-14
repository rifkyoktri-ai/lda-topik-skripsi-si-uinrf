# Panduan Persiapan Sidang

Dokumen ini dipakai untuk menjawab pertanyaan penguji dan dosen tentang **isi
penelitian** dan **cara sistem bekerja**. Semuanya ditulis dalam bahasa
sehari-hari.

> **Semua angka di dokumen ini diambil langsung dari file hasil kerja Anda.**
> Kalau ditanya "itu angka dari mana?", jawab dengan nama file yang tertulis
> di sebelah angkanya.

---

## 0. Tiga Hal yang Harus Anda Ingat

**1. Kode yang Anda tulis sendiri hanya 4.335 baris.**

```
src/   3.059 baris    mesin penelitian
app/   1.276 baris    dashboard
-------------------------------
total  4.335 baris
```

 Bandingkan dengan folder `.venv/` yang berisi **24.846 file** milik orang lain.

**2. Folder `.venv/` bukan karya Anda dan tidak pernah perlu dibuka.**

Isinya pustaka yang diunduh otomatis (`gensim`, `pandas`, `numpy`,
`streamlit`). Kalau folder itu dihapus, masih bisa dibangun ulang dari
`requirements.txt`. Tidak ada satu baris pun karya Anda di dalamnya.

**3. Semua angka penting ada di satu file: `src/config.py`.**

Kalau ditanya "di mana angka ini disimpan", jawab: **`src/config.py`**.
Itu satu-satunya tempat parameter penelitian didefinisikan.

---

## 1. Kalimat Pembuka — Hafal Tiga Kalimat Ini

Kalau ditanya *"Ceritakan penelitian Anda"*, jawab begini:

> *"Penelitian ini memetakan topik penelitian skripsi Program Studi Sistem
> Informasi UIN Raden Fatah Palembang secara otomatis menggunakan algoritma
> Latent Dirichlet Allocation, lalu memproyeksikan arah trennya untuk tahun
> 2026 dan 2027 menggunakan Weighted Moving Average. Hasilnya disajikan dalam
> satu dashboard interaktif.*
>
> *Dari 312 dokumen yang dikumpulkan, 285 dokumen lolos setelah pembersihan,
> lalu dianalisis menjadi 8 topik. Hasilnya menunjukkan komposisi topik tidak
> berubah secara signifikan antara tahun 2021 sampai 2025.*
>
> *Metodologi penelitian mengikuti enam fase CRISP-DM."*

**Tiga angka yang harus melekat di kepala:**

| Angka | Nilainya | Artinya |
|:--|--:|:--|
| dokumen bersih | **285** | dari 312 data awal |
| jumlah topik | **8** | hasil pemilihan K=8 |
| rentang tahun | **2021–2025** | ditambah proyeksi 2026–2027 |

---

## 2. Peta Folder — Mana yang Boleh Ditunjuk

### 🟢 Boleh dibuka dan ditunjuk ke penguji

| Folder | Isi | Kenapa penting |
|:--|:--|:--|
| `src/` | 19 file, 3.059 baris | **ini mesin penelitian Anda** |
| `app/` | 11 file, 1.276 baris | dashboard yang dihasilkan |
| `notebooks/` | 9 file | laporan enam fase CRISP-DM |
| `outputs/` | 43 file | **semua hasil beserta buktinya** |
| `tests/` | 6 file, 48 fungsi uji | bukti sistem tidak kacau |
| `thesis/` | naskah `.docx` | naskah Anda sendiri |

### 🟡 Jelaskan kalau diminta, jangan dibuka

| Folder | Isi |
|:--|:--|
| `data/` | 285 dokumen bersih dan 5 file HTML asli repository |
| `docs/` | 8 dokumen penjelasan teknis |

### 🔴 Jangan pernah dibuka di depan penguji

| Folder | Isi | Alasan |
|:--|:--|:--|
| `.venv/` | 703 MB, 24.846 file | seluruhnya kode orang lain |
| `.git/` | riwayat versi | urusan internal |

> **Kalau penguji membuka `.venv/` sendiri lalu bertanya "ini kode Anda?"**
> jawab: *"Bukan, Pak/Bu. Folder `.venv` berisi pustaka pihak ketiga yang
> diunduh otomatis. NumPy, Gensim, dan Streamlit adalah kode yang ditulis
> programmer lain. Kode orisinal penelitian saya ada di folder `src/`."*

---

## 3. Cara Kerja Sistem

### 3.1 Alur Enam Fase CRISP-DM

```
Business Understanding  ->  menjelaskan kebutuhan akademik
Data Understanding     ->  web scraping, 312 dokumen
Data Preparation       ->  5 tahap pembersihan, jadi 285 dokumen
Modeling               ->  LDA, menghasilkan 8 topik
Evaluation             ->  C_V, U_Mass, Perplexity, Topic Diversity
Deployment             ->  dashboard Streamlit
```

Setiap fase punya notebook-nya sendiri di folder `notebooks/`.

### 3.2 Lima Tahap Pembersihan (`src/preprocessing.py`)

| Tahap | Yang dilakukan |
|:--|:--|
| 1 | Muat data dan hapus duplikat judul |
| 2 | Deteksi bahasa abstrak dan filter teks |
| 3 | Bentuk frasa dua kata (bigram) dan tiga kata (trigram) |
| 4 | Stemming, yaitu memotong akhir kata |
| 5 | Buat kamus kata |

**Aturan yang dipakai:**

| Aturan | Nilai | Dipakai untuk |
|:--|--:|:--|
| panjang minimum token | 3 karakter | membuang kata terlalu pendek |
| minimum token per dokumen | 5 token | membuang dokumen terlalu tipis |
| `min_count` frasa | 2 | menentukan frasa |
| ambang frasa | 7,0 | menentukan frasa |

### 3.3 Kenapa Dipilih K = 8

Ada **140 konfigurasi** yang diuji, yaitu K=4 sampai K=10 yang dikalikan
5 pilihan alpha dan 4 pilihan eta.

Nilai koherensi tertinggi di tabel adalah **K=9 sebesar 0,4314**. Tapi K=9
ditolak karena pemisahan topiknya tidak memadai dan beberapa topik hanya
berisi sangat sedikit dokumen.

Di antara K=4 sampai K=10, **K=8 adalah nilai koherensi tertinggi yang
strukturnya masih bisa diinterpretasikan secara konsisten oleh peneliti.**

> **Kalau ditanya "kenapa bukan yang tertinggi?"** — jawaban itu sudah
> tertulis di naskah Bab IV. Memilih angka tertinggi bukan satu-satunya
> pilihan ilmiah. Yang penting alasannya bisa dijelaskan.

### 3.4 Cara WMA Bekerja

WMA adalah rata-rata bergerak tertimbang. Data tahun paling baru mendapat
bobot paling besar.

```
Bobot:   1   2   3   4   5        (data terbaru bobot 5)
Contoh:   5%  12%   9%  (tiga tahun terakhir)
 Rumus: (5 x 5 + 12 x 4 + 9 x 3) / (1 + 2 + 3 + 4 + 5)
```

**Dan ini wajib Anda ingat: WMA ternyata tidak akurat.** Sudah diuji dengan
walk-forward backtest, hasilnya lebih buruk daripada rata-rata biasa.

| Metode | Galat rata-rata (MAE) |
|:--|--:|
| **WMA** | 4,46 poin persentase |
| Rata-rata historis (pembanding) | **4,35 pp** ← lebih baik |
| Naive (ambil nilai tahun lalu) | 5,58 pp |

Karena itu seluruh proyeksi di sistem diberi label **"indikatif"**, bukan
prediksi yang bisa dipercaya. **Ini sikap ilmiah yang benar**, dan justru
jawaban seperti inilah yang membuat penelitian Anda terlihat jujur.

---

## 4. Lima File yang Paling Mungkin Ditanya Penguji

| File | Baris | Jawab kalau ditanya |
|:--|--:|:--|
| `src/config.py` | 120 | *"Ini tempat semua parameter penelitian dikumpulkan: jumlah dokumen, K, alpha, eta, bobot WMA, sampai semua angka metrik resmi."* |
| `src/preprocessing.py` | 225 | *"Ini pembersihan lima tahap. Masukan 312 dokumen, keluaran 285 dokumen."* |
| `src/modeling.py` | 464 | *"Ini yang melatih model LDA. File ini paling besar karena di dalamnya ada grid search 140 konfigurasi sekaligus pelatihan model final."* |
| `src/forecast.py` | 395 | *"Ini yang menghitung proyeksi WMA sekaligus uji walk-forward backtest-nya."* |
| `tests/test_structure.py` | 13.161 B | *"Ini pengujian otomatis yang memastikan folder dan file di proyek tidak kacau."* |

---

## 5. Lima Belas Pertanyaan yang Paling Sering Muncul

### ❓ Kenapa 312 dokumen menjadi 285?

> *"Sebanyak 27 dokumen dikeluarkan, dan alasannya saya catat satu per satu di
> file `outputs/audit/excluded_documents.csv`:"*

| Jumlah | Alasan |
|--:|:--|
| 20 | jumlah token kurang dari 5 setelah filtering |
| 4 | judul duplikat |
| 3 | gagal pada pembersihan scraping awal |

> *"Tidak ada dokumen yang dibuang tanpa alasan tercatat."*

---

### ❓ Kenapa K=8, padahal tabel menunjukkan K=9 lebih tinggi?

> *"Betul, K=9 punya koherensi 0,4314. Tapi struktur topiknya tidak terpisah
> jelas, dan beberapa topik hanya berisi sangat sedikit dokumen. Saya memilih
> K=8 karena koherensinya tetap tinggi dan seluruh topiknya bisa saya
> interpretasikan secara konsisten. Itu merupakan kompromi antara angka dan
> keterbacaan."*

---

### ❓ Kenapa alpha = auto, padahal tabel menunjukkan asymmetric tertinggi?

> *"Nilai alpha itu dihitung ulang setelah K=8 ditetapkan. Pada K=8, pilihan
> alpha auto justru menghasilkan koherensi tertinggi, yaitu 0,3941, dibanding
> asymmetric 0,3823. Saya memakai kriteria yang sama dengan pemilihan K,
> yaitu C_V."*
>
> *"Saya juga mencatatnya secara terbuka: pada K=8, metrik U_Mass menunjuk arah
> sebaliknya, yaitu asymmetric lebih baik. Itu saya akui sebagai
> keterbatasan."*

**Jawaban ini sangat kuat**, karena mengakui kelemahan dengan angka yang
terukur.

---

### ❓ Kenapa C_V hanya 0,4256? Bukannya rendah?

> *"Nilai itu berada di kategori sedang. Penyebabnya adalah pembersihan teks
> yang cukup ketat, dengan rata-rata 61 token per dokumen. Pembersihan itu
> meningkatkan presisi kata, tetapi ikut memperkecil konteks kemunculan
> bersama antar kata.*
>
> *"Yang lebih penting, secara kualitatif kedelapan topik tetap terpisah jelas
> dan mencerminkan bidang Sistem Informasi secara nyata."*

---

### ❓ Apakah C_V bisa direproduksi dengan melatih ulang?

> *"Nilai C_V resmi 0,4256 berasal dari model final yang sudah dilatih penuh
> dan tersimpan. Saya sudah menguji tingkat kemampuannya untuk direproduksi:
> melatih ulang dengan konfigurasi yang sama menghasilkan nilai yang sedikit
> berbeda. Jadi C_V 0,4256 adalah nilai model final yang tersimpan, bukan
> nilai yang selalu muncul setiap kali dilatih ulang. Karena itu saya
> menyimpannya sebagai artefak supaya bisa diperiksa."*

> ⚠️ **Jawab jujur.** Menyembunyikan ini jauh lebih berisiko. Kalau penguji
> mencoba mereplikasi dan mendapat angka berbeda, seluruh credibility
> penelitian Anda yang hilang.

---

### ❓ Kenapa proyeksinya tidak akurat?

> *"Saya sudah mengujinya dengan walk-forward backtest untuk tahun 2023, 2024,
> dan 2025. Hasilnya WMA tidak lebih akurat dari rata-rata historis untuk 5 dari
> 8 topik. Karena itu sistem saya tidak mengklaim proyeksi sebagai prediksi
> yang bisa dipercaya, tetapi sebagai gambaran arah tren."*

---

### ❓ Ada tanda perubahan topik yang signifikan dari 2021 ke 2025?

> *"Saya mengujinya dengan uji chi-square. Hasilnya chi-kuadrat sebesar 23,9361
> dengan p-value 0,6849. Karena p jauh lebih besar dari 0,05, tidak ada bukti
> statistik yang cukup untuk menyatakan komposisi topik berubah secara
> signifikan."*
>
> *"Nilai p dari uji asimtotik adalah 0,6849, dan dari uji permutasi Monte Carlo
> 10.000 iterasi adalah 0,6994. Keduanya menunjukkan hal yang sama."*

---

### ❓ Kenapa abstrak bahasa Inggris tidak dibuang?

> *"Justru sebaliknya, Pak/Bu. Abstrak berbahasa Inggris **tidak dibuang**.
> Sistem mendeteksi 38 dokumen seperti itu, dan untuk dokumen tersebut sistem
> memakai **judul saja** tanpa membaca abstraknya."*
>
> *"Alasannya, stemmer Sastrawi hanya bisa memproses bahasa Indonesia. Kalau
> abstrak bahasa Inggris ikut diproses, hasilnya menjadi rusak. Ini keputusan
> prapemrosesan, bukan pengurangan jumlah dokumen."*

**Rata-rata token membuktikan hal ini:**

| Jenis dokumen | Rata-rata token |
|:--|--:|
| Abstrak Inggris (judul saja) | 31,9 |
| Abstrak Indonesia (judul dan abstrak) | 65,9 |

---

### ❓ Berapa dokumen yang tidak masuk ke satu topik dengan jelas?

> *"38 dokumen dari 285, sekitar 13,33 persen, punya probabilitas topik dominan
> di bawah 0,50. Artinya isinya tersebar merata di beberapa topik. Saya catat
> ini sebagai keterbatasan, bukan saya sembunyikan, karena tabel distribusi
> topik dihitung berdasarkan topik dominan."*

---

### ❓ Bagaimana 8 topik ini ditentukan?

> *"Awalnya LDA memberi daftar kata kunci teratas untuk setiap topik. Kemudian
> saya baca satu per satu dan memberi nama yang sesuai dengan isi sebenarnya.
> Nama finalnya ada di `outputs/model/labels.json`, dan di sana ada sidik jari
> SHA-256 supaya bisa dipastikan nama itu masih cocok dengan model."*

---

### ❓ Kenapa 8 topik? Apa saja?

> *"Delapan topik, dengan jumlah dokumen masing-masing:"*
>
> *T1 32 · T2 29 · T3 46 · T4 34 · T5 41 · T6 49 · T7 33 · T8 21 dokumen*

---

### ❓ Bagaimana sistem ini bisa dipercaya?

> *"Ada 48 fungsi pengujian otomatis di folder `tests/`. Di antaranya, salah
> satunya memverifikasi bahwa sidik jari SHA-256 dari label topik cocok dengan
> model LDA, supaya nama topik tidak bisa bergeser diam-diam. Ada juga uji yang
> memastikan semua angka di naskah cocok dengan file hasil."*

---

### ❓ Kalau data repository bertambah, bagaimana?

> *"Pipeline-nya dirancang agar bisa dijalankan ulang dari awal. Ada satu pintu
> masuk, yaitu `run.py`, dengan perintah terpisah untuk setiap tahap. Acak
> sudah dikunci di angka 42 supaya hasilnya konsisten setiap kali dijalankan."*

---

### ❓ Teknologi apa saja yang dipakai?

> *"Bahasa Python, dengan pustaka Gensim untuk LDA, Sastrawi untuk stemming
> bahasa Indonesia, Scikit-learn untuk perhitungan umum, NLTK untuk daftar kata
>_STOP, dan Streamlit untuk dashboard. Versi semuanya tercatat di file
> `requirements.txt`."*

---

## 6. Kalau Tidak Tahu Jawabannya

Kalimat ini **tidak pernah salah** dan justru dihargai:

> *"Pertanyaan bagus, Pak/Bu. Saya perlu memeriksa ulang datanya agar tidak
> salah menjawab. Saya akan konfirmasi setelah adjourn."*

Yang **tidak boleh** dilakukan:

| ❌ Jangan | ✅ Lakukan |
|:--|:--|
| Menebak angka | Bilang akan diperiksa ulang |
| Mengarang alasan | Tunjuk file yang relevan |
| Menyalahkan alat | Jelaskan keputusan penelitian Anda |
| Diam saja | Ulangi pertanya dengan kalimat Anda sendiri |

---

## 7. Daftar Angka dan Asalnya

Kalau ditanya "itu angka dari mana?", tinggal sebut nama file di sebelah
kanan.

| Angka | Nilainya | Sumber file |
|:--|:--|:--|
| Dokumen mentah | 312 | `src/config.py` → `TOTAL_DOCS_RAW` |
| Dokumen bersih | 285 | `outputs/topics/dataset_corrected.csv` |
| Dokumen dikeluarkan | 27 | `outputs/audit/excluded_documents.csv` |
| Jumlah kosakata | 1.849 | `src/config.py` → `VOCAB_SIZE` |
| Rata-rata token | 61,4 | dihitung dari kolom `tokens` |
| Jumlah konfigurasi uji | 140 | `outputs/model/hyperparameter_results.csv` |
| K optimum | 8 | `src/config.py` → `NUM_TOPICS` |
| Alpha dan eta final | auto / auto | `src/config.py` → `LDA_ALPHA`, `LDA_ETA` |
| Coherence Score C_V | 0,4256 | `outputs/model/evaluation_summary.csv` |
| U_Mass | -10,7533 | `outputs/model/evaluation_summary.csv` |
| Log Perplexity | -7,4518 | `outputs/model/evaluation_summary.csv` |
| Topic Diversity (10 dan 15) | 0,7125 / 0,6917 | `outputs/model/evaluation_summary.csv` |
| Dokumen dengan theta di bawah 0,50 | 13,33% (38 dokumen) | `outputs/model/evaluation_summary.csv` |
| Uji chi-kuadrat | 23,9361 | `outputs/forecast/composition_tests_report.md` |
| p-value | 0,6849 | `outputs/forecast/composition_tests_report.md` |
| Cramér's V | 0,1449 | `outputs/forecast/composition_tests_report.md` |
| p-value permutasi | 0,6994 | `outputs/forecast/composition_tests_report.md` |
| Galat WMA | 4,46 pp | `outputs/forecast/backtest_summary.md` |
| Galat rata-rata historis | 4,35 pp | `outputs/forecast/backtest_summary.md` |
| Jumlah fungsi pengujian | 48 | folder `tests/` |
| Jumlah baris kode Anda | 4.335 | folder `src/` dan `app/` |

---

## 8. Dua Hal yang JANGAN Diklaim

### ⛔ 1. Jangan menyebut "34 abstrak Inggris"

Angka yang benar adalah **38 dokumen (13,3 persen)**, dihitung memakai fungsi
`is_english_abstract()` milik sistem sendiri.

Kalau angka 34 pernah muncul, itu berasal dari analisis pribadi dengan
ambang yang berbeda. **Angka 34 tidak berasal dari sistem.**

### ⛔ 2. Jangan bilang abstrak Inggris "dibuang"

Mereka **tetap ada** di dalam 285 dokumen, hanya dianalisis memakai judul
saja. Itu berbeda dari 27 dokumen yang benar-benar dikeluarkan.

---

## 9. Ringkasan Struktur Folder

```
lda-topik-skripsi-si-uinrf-main/
│
├── src/          🟢 19 file  - mesin penelitian (3.059 baris)
├── app/          🟢 11 file  - dashboard (1.276 baris)
├── notebooks/    🟢  9 file  - laporan enam fase CRISP-DM
├── outputs/      🟢 43 file  - hasil dan bukti angka
├── tests/        🟢  6 file  - 48 fungsi pengujian
├── thesis/       🟢  naskah.docx
│
├── data/         🟡 bahan mentah dan data bersih
├── docs/         🟡 dokumentasi teknis
│
├── .venv/        🔴 703 MB   - kode orang lain, jangan dibuka
├── .git/         🔴 riwayat versi
│
└── run.py        🟢  pintu masuk tunggal sistem
```

---

## Temuan yang Disimpan untuk Nanti

Empat hal di bawah ini **sudah diperiksa dan sudah diketahui**, tetapi
perbaikannya sengaja ditunda sampai Anda siap:

| # | Temuan |
|:--|:--|
| 1 | Separator desimal dipakai dua gaya (`0,4256` dan `0.4256`) |
| 2 | Naskah menjanjikan tiga pendekatan evaluasi, hanya dua yang bertajuk |
| 3 | Karakter `>` tertinggal di awal empat paragraf naskah |
| 4 | Jumlah abstrak Inggris di naskah perlu diselaraskan ke 38 dokumen |

Kalau penguji tidak menyentuhnya, tidak perlu Anda bawa. Kalau ditanya,
jawab jujur: *"sedang saya perbaiki."*