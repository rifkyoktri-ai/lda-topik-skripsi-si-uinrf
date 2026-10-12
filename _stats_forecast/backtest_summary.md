# Ringkasan Backtest Walk-Forward (2023-2025)

## 1. Metodologi Walk-Forward Ekspansif
- **Periode Evaluasi**: Tahun $t \in [2023, 2024, 2025]$
- **Data Latih**: Seluruh data historis $< t$ (Secara ekspansif bertambah)
- **Metode Dui**:
  1. **WMA (Weighted Moving Average)**: Bobot linier $1..n$, window = $\min(n, 5)$
  2. **Rata-rata Historis**: Baseline rerata semua tahun sebelumnya
  3. **Naive**: Nilai proporsi tahun $t-1$

## 2. Hasil Agregat MAE (Poin Persentase / pp)
- **WMA MAE Rerata**: 4.46 pp
- **Rata-rata Historis MAE Rerata**: 4.35 pp
- **Naive MAE Rerata**: 5.58 pp
- **Rasio MAE WMA vs Rata-rata Historis**: 1.0259

## 3. Kesimpulan Evaluasi Backtest
- WMA secara keseluruhan TIDAK mengalahkan baseline Rata-rata Historis (Rasio >= 0.95), mengindikasikan fluktuasi proporsi antar tahun berisik dan tidak memiliki tren linier kuat.
