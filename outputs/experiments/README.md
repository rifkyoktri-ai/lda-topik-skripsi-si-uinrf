# outputs/experiments/

Keluaran **Notebook 05** (`src/evaluation_lda.py`) — studi evaluasi yang
menjadi bahan lampiran Bab IV:

| Berkas | Isi |
|:---|:---|
| `k_sweep.csv` | Nilai $C_V$ untuk tiap K dari sweep |
| `k_sweep.png` | Kurva elbow hasil sweep K |
| `top_words_by_topic.csv` | Kata kunci tiap topik dari model kandidat |
| `proporsi_argmax.csv` | Proporsi topik per tahun, pendekatan hard assignment |
| `proporsi_soft.csv` | Proporsi topik per tahun, pendekatan soft expectation |
| `candidate_lda_model.gensim` | Model kandidat (bukan model final) |

## ⚠️ Belum dibangkitkan ulang di repo ini

Folder ini masih kosong. Isinya pernah dibuat pada eksplorasi awal, tetapi
tidak ikut ter-*commit* karena `src/evaluation_lda.py` waktu itu menulis ke
`Path("experiments")` relatif di root repo dan folder tersebut ada di
`.gitignore`.

Akibatnya, klaim selisih 2,62 poin poin antara hard dan soft assignment pada
[`../exploration/assignment_confidence.md`](../exploration/assignment_confidence.md)
**belum bisa diverifikasi ulang**. Angka resmi skripsi tidak bergantung pada
berkas-berkas ini — semuanya berasal dari `outputs/topics/`.

## Cara membangkitkan

```bash
python -m src.evaluation_lda
```

Berkas akan ditulis ke folder ini, lalu ikut ter-*commit* bersama repo.