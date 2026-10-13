# Peta Artefak `model/`

## `lda_model.gensim` — berasal dari mana?

**Jawaban jujur: artefak impor, bukan hasil pipeline versi sekarang.**

| Fakta | Nilai |
|:---|:---|
| Waktu dilatih | **26 September 2026, 05:29:13** |
| Sumber catatan | `model/experiment_history.csv` (baris terakhir) |
| Konfigurasi saat itu | K=8, **alpha=auto**, eta=auto, passes=30, iterations=500 |
| Korpus saat itu | 285 dokumen, 1.849 kata unik |
| C_V saat itu | 0,4256 |
| Stok | `random_state` — saat disimpan menjadi objek `RandomState`, seed asli tidak tersimpan |

### Yang bisa direproduksi

Seluruh **artefak turunan** dihitung ulang dari model ini dan konsisten:

```bash
python -m src.build_final     # -> _fix_final/  (C_V, distribusi topik, per tahun)
python -m src.forecast        # -> _stats_forecast/  (WMA, chi-square, backtest)
python -m src.ground_truth    # -> GROUND_TRUTH.md
python run_tests.py           # fingerprint label
```

C_V = 0,4256 **dapat diverifikasi ulang** dari `lda_model.gensim` + `data/intermediate/`.

### Yang TIDAK bisa direproduksi

Model itu sendiri. Melatih ulang K=8 dengan parameter yang sama pada korpus
sekarang menghasilkan C_V **0,3817**, bukan 0,4256. Penyebabnya bukan
pipeline rusak, melainkan:

- Versi dataset yang dipakai 26 Sept 2026 tidak ada di git (repo baru mulai
  11 Okt 2026).
- `random_state` tersimpan sebagai objek `RandomState` — seed aslinya hilang,
  dan C_V cukup sensitif terhadap seed (rentang 0,3557–0,3803 pada 10 seed).

Jadi: **angka 0,4256 punya asal yang tercatat, tapi file modelnya adalah artefak
impor.** Kalau ditanya dosen, jawaban yang benar: *"Model dilatih pada tahap
sebelum repo di-version-control; seluruh hasil evaluasi dan artefak turunan
dapat direproduksi ulang dari model tersebut."*

### Catatan alpha

Model dilatih dengan `alpha=auto` (estimasi dari data), **bukan**
`asymmetric`. Bukti: vektor alpha tersimpan ≈ 0,0244 (hasil estimasi),
sedangkan `asymmetric`/`symmetric` mengambang di 1/K = 0,125. Riwayat di
`experiment_history.csv` juga mencatat `alpha=auto`.

---

## Daftar berkas

| File | Isi | Dibuat oleh |
|:---|:---|:---|
| `lda_model.gensim` (+ `.state`, `.id2word`, `.expElogbeta.npy`) | Model LDA K=8, dilatih 26 Sept 2026 | **impor** (lihat di atas) |
| `topic_labels.json` | Label 8 topik + fingerprint SHA-256 | `src/modeling.py` |
| `topic_labels.csv` | Versi CSV label | `src/labeling.py` (notebook 07) |
| `topic_distribution.csv` | Distribusi probabilitas topik | `src/modeling.py` |
| `hyperparameter_results.csv` | 140 konfigurasi grid search | `src/grid_search.py` (notebook 04b) |
| `evaluation_metrics.csv` | Metrik ringkas model | `src/modeling.py` |
| `experiment_history.csv` | Rekam jejak 19 eksperimen (Sept–Okt 2026) | akumulasi manual |
| `human_topic_validation.csv` | Instrumen validasi olehexpert (40 dokumen) | `src/labeling.py` |
| `Lembar_Validasi_Pakar_LDA.xlsx` | Lembar cetak validasi expert | `src/labeling.py` |
| `top_words_per_topic.png`, `distribusi_topik.png` | Grafik keluaran pipeline | `src/modeling.py` |
| `lda_visualization.html`, `pyldavis_lib.js` | Eksplorasi pyLDAvis | `src/modeling.py` |
