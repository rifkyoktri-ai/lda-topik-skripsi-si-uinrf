# Peta Artefak `_fix_final/`

Folder ini berisi **artefak final** yang dipakai skripsi dan dashboard.
Semua file di sini dapat diregenerasi dari `model/lda_model.gensim` +
`data/intermediate/` memakai:

```bash
python -m src.build_final          # atau notebook 08
```

Prinsip repo ini: **setiap artefak harus bisa ditunjuk siapa pembuatnya.**

---

## Artefak yang dipakai skripsi

| File | Isi | Dibuat oleh |
|:---|:---|:---|
| `topic_counts_final.csv` | Jumlah dokumen per topik dominan (8 baris, total 285) | `src/build_final.py` |
| `topic_distribution_final.csv` | Matriks theta lengkap 285 x 8 | `src/build_final.py` |
| `topic_by_year_final.csv` | Kontingensi 5 tahun x 8 topik | `src/build_final.py` |
| `topic_by_year_sensitivity.csv` | Varian tanpa dokumen berabstrak judul-saja | `src/build_final.py` |
| `top_words_final.csv` | 15 kata kunci teratas per topik (120 baris) | `src/build_final.py` |
| `evaluation_summary.csv` | 9 metrik resmi model | `src/build_final.py` |

## Arsip eksperimen lama (JANGAN dipakai untuk penulisan)

| File | Kenapa disimpan | Kenapa bukan sumber skripsi |
|:---|:---|:---|
| `dataset_corrected.csv` | 285 baris, prapemrosesan versi lama | 281/285 baris identik dengan `data/intermediate/dataset_preprocessed.csv`; 4 baris berbeda karena versi stopword/prapemrosesan yang berbeda |
| `corpus_corrected.pkl` | BoW 285 dokumen versi lama | Terpasang pada dictionary 4.289 kata, sedangkan model final memakai 1.849 kata |

> Dua file ini hanya diperlukan bila ingin mereproduksi laporan eksplorasi di
> `docs/perplexity_report.md` dan `docs/topic_quality.md`. Keduanya memakai
> korpus 4.289 kata, sehingga angkanya **wajar berbeda** dari angka resmi.

---

## Protokol resmi vs protokal eksplorasi

| Aspek | Resmi (skripsi) | Eksplorasi (`docs/`) |
|:---|:---|:---|
| Dokumen | 285 (inferensi penuh) | 228 latih / 57 uji (80/20) |
| Kosakata | 1.849 kata | 4.289 kata |
| Log Perplexity | −7,4518 | −7,4664 |
| Topic Diversity (Top-10) | 0,7125 | 0,7875 |
| θ dominan < 0,50 | 13,33% | 2,11% |

**Rujukan penulisan skripsi: `GROUND_TRUTH.md`** (dibuat otomatis oleh
`python -m src.ground_truth`, diturunkan langsung dari artefak di atas).
