# Peta Artefak `docs/evaluation_plots/`

**Status: BERKAS GAMBAR ARSIP — belum ada skrip penghasil di repositori.**

Berbeda dengan folder lain di repo ini, gambar-gambar di sini **tidak dapat
diregenerasi** dari pipeline. Semula dibuat secara ad-hoc saat analisis
eksplorasi, lalu dipertahankan sebagai bahan visual.

---

## Isi folder

| Berkas | Isi | Keterangan |
|:---|:---|:---|
| `elbow_curve_cv.png` | Kurva elbow C_V terhadap jumlah topik | Materi pembahasan pemilihan K |
| `coherence_comparison.png` | Perbandingan C_V antar topiknya | Materi pembahasan kualitas topik |
| `weight_scheme_comparison.png` | Perbandingan skema bobot (TF-IDF vs default) | Eksperimen prapemrosesan |
| `heatmap_hyperparameters_K3.png` | Heatmap hiperparameter pada K=3 | Eksperimen grid search |
| `wordclouds/wordcloud_topic_1..7.png` | Wordcloud 7 topik | **LENGKAP SEBAGIAN** |

---

## Dua hal yang perlu diketahui

**1. Wordcloud kurang 1 topik.**
Tersedia wordcloud untuk topik 1-7 saja. **Topik 8
("Keamanan Informasi & SI Institusi Pendidikan", 21 dokumen) tidak ada.**
Alasannya tidak tercatat di repo. Kalau someday folder ini diregenerasi,
pastikan 8 wordcloud dibuat.

Wordcloud yang tersedia juga dibuat untuk versi model yang lebih awal
( sebelum perbaikan jumlah dokumen per topik), sehingga label/urutan topik
pada sebagian gambar **tidak boleh** dipakai sebagai rujukan caption skripsi.

**2. Tidak ada skrip penghasil.**
Tidak ada modul di `src/` yang menulis ke folder ini. Untuk Peralta一步:

-要么 hapus folder ini, dan andalkan gambar yang tertanam di `Skripsi Rifky FIX.docx` (Gambar 7-9 sudah menyertai wordcloud & bar chart);
-要么 buat modul `src/plot_archive.py` yang mengulangainya agar reproducible.

**Selama ini belum diputuskan, janganBecomekan gambar di sini rujukan
angka resmi.** Rujukan resmi tetap `GROUND_TRUTH.md`.
