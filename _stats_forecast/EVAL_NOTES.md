# Catatan Evaluasi Metodologi & Batasan Analisis Statistik (EVAL_NOTES.md)

## 1. Metode Inferensi LDA
- **Algoritma Inferensi**: Model Latent Dirichlet Allocation (LDA) ini dilatih menggunakan **Online Variational Bayes (VB)** (Hoffman et al., 2010), **BUKAN Gibbs Sampling**.
- **Karakteristik Online VB**:
  - Mengoptimalkan *Evidence Lower Bound* (ELBO) secara deterministik lewat pembaruan stokastik mini-batch.
  - Memungkinkan penanganan dataset secara cepat dan efisien.
  - Menggunakan parameter konsentrasi dokumen-topik $\alpha$ *asymmetric* dan kata-topik $\eta$ *auto*.

## 2. Evaluasi Metrik Koherensi & Perpleksitas
- **$C_V = 0.4256$**: Metrik koherensi berbasis *sliding window* dan koefisien korelasi cosinus, dihitung dengan protokol resmi texts=data/intermediate/ (1.849 vocab), topn=20, processes=1, Gensim 4.4.0. Nilai ini berada pada rentang sedang untuk korpus skripsi akademik berbahasa Indonesia dengan terminologi multidisiplin.
- **$\text{UMass} = -10.7533$**: Metrik koherensi berbasis *log-conditional probability* kata dalam korpus asal (semakin mendekati 0 semakin baik).
- **Log Perplexity = -7.4518**: Mengukur seberapa baik model memprediksi sampel dokumen baru (semakin negatif/rendah *perplexity*, semakin baik kelayakan generatif model).

## 3. Weighted Moving Average (WMA) & Keterbatasannya
- **Metode WMA**: Memberikan bobot linier $1, 2, 3, 4, 5$ untuk tahun 2021 hingga 2025 ($W=5$).
- **Keterbatasan Utama**:
  1. **Ukuran Sampel Pendek ($N=5$ titik waktu)**: Data hanya tersedia untuk periode 2021–2025. Ekstrapolasi statistik kuat (seperti ARIMA atau Prophet) tidak valid secara metodologis.
  2. **Hasil Backtest Walk-Forward**: Rasio MAE WMA terhadap baseline Rata-rata Historis adalah $\ge 0.95$. Ini membuktikan bahwa WMA **tidak lebih baik** secara signifikan dibandingkan memprediksi proporsi topik menggunakan rerata historisnya.
  3. **Sifat Proyeksi**: Proyeksi 2026–2027 bersifat **INDIKATIF**, bukan kepastian kuantitatif.

## 4. Panduan Klaim untuk Penulisan Skripsi (Bab V)

### ✅ Kalimat yang AMAN dan VALID untuk Digunakan di Skripsi:
1. *"Berdasarkan uji Chi-Square homogenitas ($p > 0.05$), tidak terdapat bukti statistik yang cukup untuk menyatakan bahwa distribusi tren topik skripsi SI UINRF mengalami pergeseran komposisi yang signifikan antar tahun 2021–2025."*
2. *"Hasil proyeksi WMA 2026 bersifat indikatif dan menunjukkan fluktuasi proporsi topik yang relatif stabil dalam rentang batas kesalahan historisnya."*
3. *"Pengujian sensitivitas mengonfirmasi bahwa eksklusi dokumen berabstrak judul-saja tidak mengubah kesimpulan umum dinamika topik secara bermakna."*

### ❌ Klaim yang HARUS DIHINDARI (Dilarang Ditulis di Skripsi):
1. ❌ *"Model WMA terbukti sangat akurat dalam memprediksi jumlah skripsi tahun 2026."* (SALAH: Backtest membuktikan WMA tidak lebih baik dari rerata historis).
2. ❌ *"Topik X mengalami peningkatan pesat/signifikan secara statistik."* (SALAH: Tidak ada topik yang memiliki $p$-value tren Spearman $< 0.00625$ setelah koreksi Bonferroni).
3. ❌ *"Penerapan Gibbs Sampling pada LDA berhasil mengelompokkan topik."* (SALAH: Model dilatih menggunakan Online Variational Bayes).
