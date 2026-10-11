# LAPORAN EVALUASI PERPLEXITY (`perplexity_report.md`)

## 1. Per-word Log Perplexity Bound
Pengujian dilakukan dengan membagi korpus menjadi **80% Train (228 dokumen)** dan **20% Hold-out Test (57 dokumen)** secara terstratifikasi berdasarkan tahun skripsi.

- **Train Log Perplexity:** `-7.4664`
- **Hold-out Test Log Perplexity:** `-11.0104`
- **Perplexity Gap (Test - Train):** `-3.5440`

---

## 2. Analisis Kesenjangan (Generalization Gap)
- Log perplexity bernilai negatif (karena merupakan log-likelihood per kata). Nilai yang **lebih tinggi (mendekati 0)** menunjukkan performa prediktif yang lebih baik.
- Kesenjangan sebesar `-3.5440` antara data latih dan data uji mengindikasikan adanya *overfitting* moderat pada korpus latih. Dokumen pada hold-out test memuat beberapa kata yang jarang muncul pada dataset latih.
