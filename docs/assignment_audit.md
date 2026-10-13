# AUDIT KEYAKINAN PENUGASAN DOKUMEN

_Dihitung otomatis oleh `src/assignment_audit.py` - jangan diedit manual._

**Sumber data:** `topic_distribution_final.csv`
**Tanggal pembuatan:** 13-10-2026 14:51

Angka pada bagian Keterbatasan Bab IV yang merujuk keyakinan penugasan dan fragmentasi dokumen analitik diturunkan dari laporan ini.

---

## 1. Keyakinan Penugasan

Nilai probabilitas maksimum tiap dokumen terhadap topik dominannya mengukur seberapa tegas model dalam menempatkan dokumen tersebut.

| Kelompok | Dokumen | Persentase |
|:---|---:|---:|
| Total dokumen | 285 | 100,00% |
| Sangat yakin (max-theta >= 0,99) | 179 | 62,81% |
| Kurang yakin (max-theta < 0,50) | 38 | 13,33% |
| **Ragu (max-theta < 0,60)** | **58** | **20,35%** |
| Rentang tengah (0,60 <= theta < 0,95) | 29 | 10,18% |

Median max-theta = 0,9953.

Distribusi bersifat bimodal: sebagian besar dokumen terkunci dengan keyakinan hampir mutlak, sementara sebagian dokumen benar-benar membentang antartopik. Kelompok "ragu" inilah yang paling berpotensi menjadi subjek interpretasi ulang.

## 2. Fragmentasi Dokumen Analitik Data

Dokumen yang 제목nya menyebut algoritma atau metode analitik data (K-Means, Apriori, Eclat, Naive Bayes, C4.5, FP-Growth, ARIMA, analisis sentimen, dan sejenisnya).

| Metrik | Nilai |
|:---|---:|
| Jumlah dokumen analitik | 26 |
| Persentase terhadap korpus | 9,12% |
| Jumlah topik tujuan | 8 |
| Masuk pada topik analitik data (T5) | 7 (26,9%) |
| Dokumen K-Means total | 3 |
| Dokumen K-Means pada topik analitik data | 0 |

Sebaran dokumen analitik data menurut topik dominan:

| Topik | Dokumen |
|---:|---:|
| T1 | 3 |
| T2 | 1 |
| T3 | 7 |
| T4 | 4 |
| T5 | 7 |
| T6 | 1 |
| T7 | 2 |
| T8 | 1 |

Dokumen analitik data tersebar hampir ke seluruh topik, padahal hanya sebagian kecil yang mendarat pada topik yang secara semantik paling dekat dengan analitik data. Ini menunjukkan pemisahan antartopik belum sepenuhnya tegas dan merupakan keterbatasan yang akan diperiksa ulang pada penelitian lanjutan.

---

_Perintah regenerasi: `python -m src.assignment_audit`_
