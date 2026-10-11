# DISKREPANSI KONFIGURASI MODEL & HYPERPARAMETER (CONFIG_DISCREPANCIES.md)

## 1. Tabel Perbandingan Klaim Draf vs Kode vs Artefak Tersimpan

| Parameter / Konfigurasi | Draf Skripsi (Klaim Penulis) | Kode Python (`pipeline.py` / `hyperparameter_tuning.py`) | Artefak Model Tersimpan (`lda_model.gensim`) | Berkas Metrik (`evaluation_metrics.csv`) | Saran Koreksi (Siapa yang Harus Diubah) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Jumlah Topik ($K$)** | $K=8$ | Default auto-tune: 5..10 / manual CLI: $K=8$ | `num_topics` = 8 | `Jumlah Topik (K)` = 8 | **Konsisten** across all. |
| **Alpha Prior ($\alpha$)** | `'asymmetric'` | Default: `'asymmetric'` / CLI: `'auto'` | `alpha` = vector asymmetric (`[0.0255, 0.0221, ...]`) | `Alpha Prior` = `'asymmetric'` | **Ubah Teks Skripsi**: Jelaskan bahwa `asymmetric` menggunakan estimasi prior dari data latih Gensim. |
| **Eta Prior ($\eta$)** | `0.1` | Default tuning: `['symmetric', 0.01, 0.1]` / pipeline: `'auto'` | `eta` = vector auto per-word (`[0.5729, 0.1573, ...]`) | `Eta Prior` = `'0.1'` | **Ubah Berkas / Teks**: `evaluation_metrics.csv` menuliskan `0.1`, tetapi model asli dilatih menggunakan `eta='auto'` (vektor per-kata). Koreksi teks skripsi menjadi `eta='auto'` atau perbarui CSV metrik. |
| **Passes LDA** | 50 passes | CLI default `pipeline.py`: 50; fungsi `train_model()` default: 30 | `passes` = **30** | Tidak tercantum | **Ubah Teks Skripsi**: Model final tersimpan di `lda_model.gensim` sebenarnya dilatih dengan **30 passes**, bukan 50 passes. Koreksi teks skripsi dari 50 ke 30 passes. |
| **Iterasi LDA** | 400 iterasi | CLI default: 400; fungsi `train_model()` default: 500 | `iterations` = **500** | Tidak tercantum | **Ubah Teks Skripsi**: Model tersimpan dilatih dengan **500 iterasi**. Koreksi teks skripsi dari 400 ke 500 iterasi. |
| **Filter Panjang Kata** | Buang kata $<5$ karakter | `len(w) > 2` (mempertahankan kata $\ge 3$ karakter) | Vocab memuat kata 3-4 huruf (`spk`, `puas`, `jual`, `stok`) | Tidak tercantum | **Ubah Teks Skripsi**: Ganti uraian di Bab III menjadi *"membuang token kurang dari 3 karakter ($<3$)"*. |
| **Filter Minimal Token/Doc** | Tidak disebutkan di draf | `df['tokens'].apply(len) >= 5` (membuang doc $<5$ token) | 285 dokumen tersimpan | Tidak tercantum | **Ubah Teks Skripsi**: Tambahkan penjelasan kriteria eliminasi dokumen dengan $<5$ token di Bab III. |

---

## 2. Analisis Grid Search Hyperparameter (Kombinasi 84 vs 156 vs 208)

### Perbandingan Kombinasi:
1. **Draf Skripsi**:
   - Sebagian teks menyebutkan rentang $K=4..10$ dengan kombinasi grid search sebanyak **208 kombinasi** (klasifikasi bertentangan dengan opsi parameter).
   - Bagian lain menyebutkan $K=3..15$ dengan **156 kombinasi**.
2. **Kode Python (`hyperparameter_tuning.py`)**:
   - Parameter default: $K \in [3..15]$ (13 nilai), $\alpha \in ['symmetric', 'asymmetric', 0.01, 0.1]$ (4 nilai), $\eta \in ['symmetric', 0.01, 0.1]$ (3 nilai).
   - Total kombinasi yang dihasilkan kode: $13 \times 4 \times 3 = \mathbf{156\text{ kombinasi}}$.
3. **Hasil Tersimpan (`model/hyperparameter_results.csv`)**:
   - Berkas CSV tersimpan hanya memiliki **84 baris**.
   - Terdiri dari: $K \in [4..10]$ (7 nilai) $\times$ 4 $\alpha$ $\times$ 3 $\eta$ = $\mathbf{84\text{ kombinasi}}$.

---

## 3. Identifikasi Konfigurasi Final yang Menghasilkan Model

Berdasarkan analisis berkas `model/hyperparameter_results.csv`:
- Pada kombinasi $K=8, \alpha='asymmetric', \eta=0.1$, nilai Coherence $C_V$ yang dicapai pada hasil tuning adalah **0.4026**.
- Nilai $C_V = 0.4026$ ini tepat sama dengan nilai yang dicatat dalam `model/evaluation_metrics.csv`.
- Dengan demikian, model final $K=8$ diambil dari hasil tuning pada baris kombinasi $K=8, \alpha=\text{asymmetric}, \eta=0.1$, namun dieksekusi dengan `passes=30` dan `iterations=500` pada alur `pipeline.py`.
