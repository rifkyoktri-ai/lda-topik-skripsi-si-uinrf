# TABEL REKONSILIASI ANGKA DRAF VS ANGKA FINAL (`DELTA_REPORT_ANGKA.md`)
**Tanggal Audit:** 2 Oktober 2026  
**Peran:** Engineer NLP / Statistik  
**Folder Output:** `_fix_final/`  

---

## 1. Perbandingan Distribusi Per Topik (Seluruh Periode)

| Topik ID | Jumlah Dokumen Draf | Jumlah Dokumen Final (Model Tersimpan + Correction) | Persentase Final (%) | Status Koreksi |
| :---: | :---: | :---: | :---: | :---: |
| **Topik 1** | 66 | **32** | 11.23% | **KOREKSI DIPERLUKAN** |
| **Topik 2** | 46 | **30** | 10.53% | **KOREKSI DIPERLUKAN** |
| **Topik 3** | 41 | **46** | 16.14% | **KOREKSI DIPERLUKAN** |
| **Topik 4** | 38 | **33** | 11.58% | **KOREKSI DIPERLUKAN** |
| **Topik 5** | 34 | **41** | 14.39% | **KOREKSI DIPERLUKAN** |
| **Topik 6** | 26 | **49** | 17.19% | **KOREKSI DIPERLUKAN** |
| **Topik 7** | 21 | **33** | 11.58% | **KOREKSI DIPERLUKAN** |
| **Topik 8** | 13 | **21** | 7.37% | **KOREKSI DIPERLUKAN** |
| **TOTAL** | **285** | **285** | **100.00%** | **SAMA** (Total Dokumen) |

---

## 2. Perbandingan Metrik Evaluasi Model

| Metrik Evaluasi | Nilai Draf Skripsi | Nilai Final Aktual | Status Koreksi | Catatan Metodologis |
| :--- | :---: | :---: | :---: | :--- |
| **Coherence Score ($C_V$)** | `0.4026` | **`0.3749`** | **KOREKSI DIPERLUKAN** | Nilai 0.4026 berasal dari $K=9$ di `hyperparameter_results.csv`, bukan $K=8$. |
| **Coherence Score ($U_{mass}$)** | Tidak Dilaporkan | **`-9.8165`** | **KOREKSI DIPERLUKAN** | Ditambahkan sebagai metrik pelengkap. |
| **Train Log Perplexity** | `-8.18` | **`-7.4664`** | **KOREKSI DIPERLUKAN** | Per-word log bound pada 285 dokumen korpus. |
| **Hold-out Test Perplexity** | `-16.09` | **`-11.0104`** | **KOREKSI DIPERLUKAN** | Pengujian 20% hold-out test set terstratifikasi. |
| **Jumlah Topik ($K$)** | 8 | **8** | **SAMA** | $K=8$ terkonfirmasi di `model/lda_model.gensim`. |
| **Passes / Iterations** | 50 / Auto | **30 / 500** | **KOREKSI DIPERLUKAN** | Model dilatih dengan passes=30, iterations=500. |
| **Alpha / Eta** | asymmetric / 0.1 | **asymmetric / auto** | **KOREKSI DIPERLUKAN** | Eta prior tersimpan adalah `'auto'`. |

---

## 3. Perbandingan Proporsi Topik per Tahun (%)

| Tahun | Topik Dominan Draf | Topik Dominan Final (Argmax) | Proporsi Topik Dominan Final | Status Koreksi |
| :---: | :---: | :---: | :---: | :---: |
| **2021** | Topik 1 | **Topik 3** (7 dokumen) | 24.14% | **KOREKSI DIPERLUKAN** |
| **2022** | Topik 1 | **Topik 6** (15 dokumen) | 19.23% | **KOREKSI DIPERLUKAN** |
| **2023** | Topik 1 | **Topik 6** (11 dokumen) | 23.91% | **KOREKSI DIPERLUKAN** |
| **2024** | Topik 1 | **Topik 5** (10 dokumen) | 21.28% | **KOREKSI DIPERLUKAN** |
| **2025** | Topik 1 | **Topik 3 & Topik 6** (15 dokumen masing-masing) | 17.65% masing-masing | **KOREKSI DIPERLUKAN** |

---

## 4. Kesimpulan Revisi Bab IV
Seluruh tabel frekuensi, proporsi tahunan, metrik coherence, dan parameter pelatihan pada draf skripsi memerlukan koreksi total sesuai nilai pada tabel di atas. Berkas `topic_distribution_final.csv`, `topic_counts_final.csv`, `topic_by_year_final.csv`, dan `evaluation_summary.csv` menjadi rujukan resmi utama.

## Update Setelah Reset (Oktober 2026)
Konfirmasi C_V final = 0.4256 (bukan 0.3749).
Konfirmasi semua angka di forecast_table.csv valid.
Halaman dashboard yang sudah selesai:
- Beranda
- Distribusi Topik
- Tren Per Tahun
- Proyeksi 2026-2027
- Detail Topik
- Cari Dokumen
- Tentang & Metodologi

Bagian skripsi yang masih perlu direvisi berdasarkan GROUND_TRUTH.md:
- Bab III: Protokol evaluasi C_V (texts=v1, topn=20, p=1).
- Bab IV: Angka C_V 0.4256, distribusi dokumen sesuai `topic_counts_final.csv`, dan proyeksi 2026-2027.
- Bab V: Kesimpulan tentang topik keranjang sisa (Topik 2 dan 8) dan tidak adanya tren signifikan yang ditemukan.
