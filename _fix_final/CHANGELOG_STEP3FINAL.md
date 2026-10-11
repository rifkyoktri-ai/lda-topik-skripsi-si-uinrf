# CHANGELOG TAHAP 3 FINAL (`CHANGELOG_STEP3FINAL.md`)
**Tanggal Execution:** 2 Oktober 2026  
**Peran:** Engineer NLP / Statistik  

---

## Ringkasan Perubahan & Artefak yang Dihasilkan di `_fix_final/`

1. **`dataset_corrected.csv`**:
   - Berkas dataset 285 dokumen terbarukan dengan koreksi token, `teks_gabung`, dan `teks_bersih` untuk 4 dokumen dalam korpus (ID 23441, 29626, 25414, 57684). ID 46773 tereliminasi pada filter panjang token.
2. **`corpus_corrected.pkl` & `dictionary_corrected.gensim`**:
   - Vektor BoW corpus dan Dictionary gensim terbarukan dari dataset yang dikoreksi.
3. **`topic_distribution_final.csv`**:
   - Distribusi probabilitas 8 topik per dokumen untuk seluruh 285 dokumen beserta penetapan `topik_dominan` dan `prob_dominan`.
4. **`topic_counts_final.csv`**:
   - Tabel agregasi jumlah dokumen dan persentase per topik (seluruh periode 2021-2025).
5. **`topic_by_year_final.csv`**:
   - Distribusi dan proporsi topik per tahun (2021-2025) dilengkapi kolom `jumlah_judul_saja`.
6. **`top_words_final.csv`**:
   - Top-15 kata kunci dan probabilitas per topik dari `model/lda_model.gensim`.
7. **`evaluation_summary.csv`**:
   - Rangkuman metrik evaluasi final: $C_V = 0.3749$, $U_{mass} = -9.8165$, Log Perplexity = $-7.4664$, Topic Diversity Top-10 = $0.7875$, Top-15 = $0.7250$, % Max Theta $<0.50 = 2.11\%$, serta nilai Jaccard Top-15 per topik.
8. **`topic_by_year_sensitivity.csv`**:
   - Matriks uji sensitivitas proporsi tahunan (dengan vs tanpa 38 dokumen judul-saja).
9. **`DELTA_REPORT_ANGKA.md`**:
   - Tabel perbandingan berdampingan (side-by-side) antara Draf vs Final untuk revisi Bab IV.
10. **`test_final.py`**:
    - Skrip pengujian otomatis unit test (pytest) untuk memverifikasi integritas data dan konsistensi matematis.
