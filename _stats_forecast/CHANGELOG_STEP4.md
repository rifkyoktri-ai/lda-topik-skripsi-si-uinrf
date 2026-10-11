# CHANGELOG - STEP 4: STATISTICAL ANALYSIS & FORECASTING

## Ringkasan Perubahan & Artefak yang Dihasilkan

Seluruh pekerjaan analisis statistik dan proyeksi tren topik (2021-2025) diselesaikan di folder `_stats_forecast/` tanpa melatih ulang model LDA maupun mengubah artefak di `_fix_final/` dan `model/`.

### 1. Artefak Baru di Folder `_stats_forecast/`
- **`stats_forecast.py`**: Skrip Python utama yang mengeksekusi uji Chi-square homogenitas, permutasi Monte Carlo, korelasi Spearman, 2-proportion Z-test, interval kepercayaan Wilson 95%, backtest walk-forward ekspansif, proyeksi WMA 2026-2027, serta uji sensitivitas.
- **`composition_tests_report.md`**: Laporan uji Chi-square homogenitas ($\chi^2 = 23.9132, p = 0.6861$), Cramer's V ($0.1448$), dan Uji Permutasi Monte Carlo 10.000 iterasi ($p = 0.7031$).
- **`per_topic_tests.csv`**: Tabel hasil uji tren Spearman dan 2-proportion Z-test (2021 vs 2025) untuk 8 topik sebelum dan sesudah koreksi Bonferroni ($\alpha = 0.00625$).
- **`share_intervals.csv`**: Rentang proporsi per topik per tahun beserta interval kepercayaan Wilson 95% (lower & upper bounds).
- **`backtest_results.csv`**: Evaluasi MAE (poin persentase) walk-forward ekspansif (2023-2025) per topik untuk metode WMA, Rata-rata Historis, dan Naive.
- **`backtest_summary.md`**: Ringkasan evaluasi backtest yang menunjukkan bahwa WMA tidak mengalahkan Rata-rata Historis secara signifikan (Rasio MAE $\ge 0.95$).
- **`forecast_table.csv`**: Tabel proyeksi final WMA 2026-2027 memuat 8 topik, pita ketidakpastian $\pm 1$ SD residual, arah perubahan, serta statistik keandalan.
- **`sensitivity_report.md`**: Hasil uji sensitivitas pembuangan 38 dokumen berabstrak judul-saja yang mengonfirmasi kestabilan tren dan proyeksi.
- **`EVAL_NOTES.md`**: Catatan evaluasi metodologi LDA (Online Variational Bayes), metrik koherensi/perpleksitas, keterbatasan WMA, serta kalimat aman/terlarang untuk skripsi.
- **`test_stats.py`**: Suite pengujian unit yang memverifikasi rumus WMA, konsistensi sum 100%, Chi-square, penolakan window > 5, logika kolom arah, dan jumlah baris tabel forecast.

### 2. Hasil Pengujian Unit (`test_stats.py`)
Semua 6 pengujian unit lulus dengan sukses:
1. `test_wma_formula`: PASSED
2. `test_sum_shares_per_year`: PASSED
3. `test_chi2_contingency_known_table`: PASSED
4. `test_wma_window_rejection`: PASSED
5. `test_forecast_direction_logic`: PASSED
6. `test_forecast_table_rows`: PASSED
