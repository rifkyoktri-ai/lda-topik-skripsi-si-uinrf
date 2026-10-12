<!-- ============================================================
  LAPORAN EKSPLORASI - BUKAN ANGKA RESMI SKRIPSI
  Dihitung dengan inferensi 80/20 hold-out.
  Angka resmi (inferensi penuh): dokumen dengan max(theta) < 0,50 = 13,33%
  (bukan 2,11% seperti pada laporan ini).

  CATATAN REPRODUKTIBILITAS: berkas `proporsi_argmax.csv` dan
  `proporsi_soft.csv` yang disebut di bawah TIDAK lagi ada di repositori.
  Klaim selisih 2,62% poin tidak dapat diverifikasi ulang dari repo ini.
  ============================================================ -->

# LAPORAN KEYAKINAN PENUGASAN DOKUMEN (`assignment_confidence.md`)

## 1. Distribusi Nilai Maksimum $\theta$ (Max Probabilitas Topik Dokumen)
- **Persentase Dokumen dengan $\max(\theta) < 0.4$:** `0.00%` (0 dokumen)
- **Persentase Dokumen dengan $\max(\theta) < 0.5$:** `2.11%` (6 dokumen)
- **Persentase Dokumen dengan $\max(\theta) < 0.6$:** `11.23%` (32 dokumen)

---

## 2. Perbandingan Proporsi Topik per Tahun: Argmax vs Soft Expectation ($\bar{\theta}$)
- **Selisih Maksimum Proporsi:** `2.62%` poin.
- **Berkas Hasil Simpanan:**
  - `proporsi_argmax.csv`
  - `proporsi_soft.csv`
- **Kesimpulan:** Pendekatan *Hard Assignment (Argmax)* yang digunakan dalam draf skripsi memberikan perkiraan proporsi tahunan yang cukup dekat dengan *Soft Expectation* (selisih terbesar hanya 2.62% poin). Urutan topik utama per tahun tidak mengalami pergeseran dominasi secara signifikan.
