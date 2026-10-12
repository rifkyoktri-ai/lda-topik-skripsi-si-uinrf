# Laporan Uji Komposisi Topik x Tahun (2021-2025)

## 1. Ringkasan Statistik Contingency Table
- **Total Dokumen ($N$)**: 285
- **Dimensi Tabel**: 5 Tahun x 8 Topik (40 sel)
- **Derajat Kebebasan ($dof$)**: 28

## 2. Hasil Uji Chi-Square Homogenitas
- **Statistik $\chi^2$**: 23.9361
- **p-value Mentah (Asimptotik)**: 0.6849
- **Cramer's V**: 0.1449 (Mengindikasikan asosiasi sangat lemah hingga sedang)

## 3. Uji Permutasi Monte Carlo (10.000 iterasi, seed=42)
- **p-value Permutasi**: 0.6994
- **Kesimpulan Uji Homogenitas**: Tidak terdapat bukti statistik yang cukup untuk menyatakan komposisi topik berubah secara signifikan antar tahun (p >= 0.05)

## 4. Evaluasi Asumsi Nilai Harapan (Expected Frequencies)
- **Jumlah Sel dengan Nilai Harapan < 5**: 12 dari 40 sel (30.0%)
- **Peringatan Asumsi**: ⚠️ PERINGATAN: Lebih dari 20% sel memiliki nilai harapan < 5. Hasil uji Chi-Square asimptotik perlu ditafsirkan hati-hati; gunakan hasil uji permutasi Monte Carlo sebagai rujukan utama.
