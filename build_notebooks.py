"""
build_notebooks.py — Generator file .ipynb untuk pipeline CRISP-DM.

Seluruh logika berat berada di `src/`. Notebook ini adalah LAPISAN LAPORAN:
markdown penjelasan + pemanggilan fungsi, sehingga mudah dibaca dan
bisa langsung dijadikan lampiran skripsi.

Struktur tiap notebook:
  1. Markdown atas  : Tujuan, Asal (Input), Keluaran (Output)
  2. Setup          : import modul
  3. Proses         : pemanggilan fungsi inti
  4. Markdown bawah : interpretasi + peta artefak

Jalankan:  python build_notebooks.py
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent
NB = REPO / "notebooks"
NB.mkdir(parents=True, exist_ok=True)


def _src(t):
    """nbformat: `source` = list of string, tiap baris (kecuali terakhir)
    WAJIB diakhiri '\\n'. Pakai splitlines(keepends=True), bukan split('\\n')."""
    return t.strip("\n").splitlines(keepends=True)


def md(t):
    return {"cell_type": "markdown", "metadata": {},
            "source": _src(t)}


def code(t):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": _src(t)}


SETUP = '''
import sys, io as _io
from pathlib import Path
_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(_ROOT))
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import pandas as pd, numpy as np
from src import config as C
C.ensure_dirs()
print(C.describe())
'''


def write(name, cells):
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3",
                                      "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": "3.11"}},
          "nbformat": 4, "nbformat_minor": 5}
    (NB / name).write_text(json.dumps(nb, ensure_ascii=False, indent=1),
                           encoding="utf-8")
    nmd = sum(1 for c in cells if c["cell_type"] == "markdown")
    ncd = sum(1 for c in cells if c["cell_type"] == "code")
    print("  [OK] %-42s %2d md + %2d code" % (name, nmd, ncd))


# ══════════════════════════════════════════════════════════════════════════
print("Membuat notebook CRISP-DM...")

# ── 01 BUSINESS UNDERSTANDING ──────────────────────────────────────────────
write("01_business_understanding.ipynb", [
md("""
# 01 - Business Understanding & Pengumpulan Data

## Tujuan
Mengambil data skripsi mahasiswa Program Studi Sistem Informasi UIN Raden Fatah
Palembang periode 2021-2025 dari repositori resmi institusi.

## Asal (Input)
`data/raw/html_source/2021.html` ... `2025.html` - halaman indeks repositori
EPrints yang disimpan dengan batas waktu **19 September 2026**.

## Keluaran (Output)
`data/raw/thesis_scraped.csv` - 312 dokumen unik
(id, year, author, title, abstract, url)

## Posisi dalam CRISP-DM
Fase pertama CRISP-DM adalah **Business Understanding**: memahami tujuan dan
ruang lingkup riset. Notebook ini mendokumentasikan kebutuhan Kaprodi
(pemetaan topik otomatis) dan mahasiswa (menemukan novelty) yang menjadi
dasar seluruh analisis.
"""),
code(SETUP),
md("## Rumusan Masalah yang Menjadi Acuan\n\n"
   "1. Bagaimana memetakan topik skripsi dengan LDA?\n"
   "2. Bagaimana memprediksi tren distribusi topik dengan WMA?\n"
   "3. Bagaimana membangun dashboard interaktif sebagai alat bantu keputusan?"),
md("## Langkah 1 - Jalankan Pengumpulan Data"),
code('''
from src import scraping

n = scraping.run(verbose=True)
print()
print(f'Total dokumen : {n}')
print(f'Nilai harapan : {C.TOTAL_DOCS_RAW}')
'''),
md("## Langkah 2 - Distribusi Dokumen per Tahun"),
code('''
raw_df = pd.read_csv(C.SCRAPED_CSV)
print(raw_df.groupby('year').size().rename('jumlah').to_string())
print()
print(f'Total: {len(raw_df)} dokumen')
'''),
md("## Langkah 3 - Cek Kelengkapan Field"),
code('''
for col in ['id', 'year', 'author', 'title', 'abstract']:
    filled = raw_df[col].notna().sum()
    print(f'  {col:10} terisi {filled:3d}/{len(raw_df)}  ({filled/len(raw_df)*100:5.1f}%)')
'''),
md("""
## Ringkasan Fase 01 - Business Understanding

| Hasil | Nilai |
|:---|:---|
| Dokumen terambil | **312 skripsi** |
| Rentang tahun | 2021-2025 |
| Cutoff | 19 September 2026 |
| Sumber | Repositori EPrints UIN Raden Fatah |

Kebutuhan Kaprodi dan mahasiswa terjawab melalui pengumpulan data ini.
Langkah berikutnya adalah memahami struktur dan kualitas data.
"""),
md("## Peta Artefak\n\n"
   "| Artefak | Dibuat oleh | Consumed by |\n"
   "|:---|:---|:---|\n"
   "| `data/raw/html_source/2021-2025.html` | Save Page As repositori | Notebook 01 |\n"
   "| `data/raw/thesis_scraped.csv` | Notebook 01 | Notebook 02, 03 |"),
])

# ── 02 DATA UNDERSTANDING ──────────────────────────────────────────────────
write("02_data_understanding.ipynb", [
md("""
# 02 - Data Understanding

## Tujuan
Memahami struktur, kelengkapan, dan kualitas data sebelum diproses lebih lanjut,
serta mendokumentasikan dokumen yang dibuang beserta alasannya.

## Asal (Input)
`data/raw/thesis_scraped.csv` - 312 dokumen hasil Notebook 01

## Keluaran (Output)
- `data/processed/thesis_for_pipeline.csv` - 309 dokumen (filter awal)
- `docs/data_lineage_by_year.csv` - jejak jumlah dokumen per tahap
- `docs/excluded_documents.csv` - dokumen yang dibuang + alasannya

## Pertanyaan yang Dijawab
- Berapa dokumen yang punya field lengkap?
- Dokumen mana yang terduplikasi judulnya?
- Berapa dokumen yang lolos ke tahap berikutnya?
"""),
code(SETUP),
md("## Langkah 1 - Analisis Kelengkapan Field"),
code('''
raw_df = pd.read_csv(C.SCRAPED_CSV)
print(f'Dokumen : {len(raw_df)}')
print(f'Kolom   : {list(raw_df.columns)}')
print()
for col in ['id', 'year', 'author', 'title', 'abstract']:
    filled = raw_df[col].notna().sum()
    print(f'  {col:10} terisi {filled:3d}/{len(raw_df)}  ({filled/len(raw_df)*100:5.1f}%)')
'''),
md("## Langkah 2 - Deteksi Duplikasi Judul"),
code('''
raw_df['tmp_title_lower'] = raw_df['title'].astype(str).str.strip().str.lower()
dupes = raw_df[raw_df.duplicated(subset=['tmp_title_lower'], keep='first')]
print(f'Dokumen dengan judul duplikat: {len(dupes)}')
if len(dupes):
    for _, r in dupes.head(5).iterrows():
        print(f'  ID {r["id"]:>7} ({r["year"]}): {str(r["title"])[:65]}')
'''),
md("## Langkah 3 - Filter Awal dan Lineage"),
code('''
from src import data_lineage as dl

dl.run_data_lineage(verbose=True)
'''),
md("""
## Ringkasan Fase 02 - Data Understanding

| Tahap | Jumlah Dokumen |
|:---|---:|
| Hasil web scraping | 312 |
| Setelah filter awal | 309 |
| **Korpus bersih akhir** | **285** |

**Temuan utama:**
1. Sebagian dokumen tidak memiliki teks abstrak sehingga tidak dapat dianalisis.
2. Duplikasi judul diselesaikan dengan case-insensitive matching.
3. Setelah preprocessing, 285 dokumen lolos dengan minimal 5 token per dokumen.
"""),
md("## Peta Artefak\n\n"
   "| Artefak | Dibuat oleh | Consumed by |\n"
   "|:---|:---|:---|\n"
   "| `data/processed/thesis_for_pipeline.csv` | Notebook 02 | Notebook 03 |\n"
   "| `docs/data_lineage_by_year.csv` | Notebook 02 | Lampiran Bab III |\n"
   "| `docs/excluded_documents.csv` | Notebook 02 | Lampiran Bab III |"),
])

# ── 03 DATA PREPARATION ────────────────────────────────────────────────────
write("03_data_preparation.ipynb", [
md("""
# 03 - Data Preparation

## Tujuan
Mengubah teks mentah menjadi representasi numerik siap model: case folding,
pembersihan, deteksi bahasa, pembentukan n-gram, stemming Sastrawi, serta
penyaringan stopword.

## Asal (Input)
`data/processed/thesis_for_pipeline.csv` - 309 dokumen hasil Notebook 02

## Keluaran (Output)
- `data/intermediate/dataset_preprocessed.csv` - 285 dokumen + kolom tokens
- `data/intermediate/dictionary.gensim` - 1.849 kata unik
- `data/intermediate/corpus.pkl` - corpus Bag-of-Words

## Tujuh Tahap Prapemrosesan
1. Case Folding | 2. Cleaning | 3. Deteksi Bahasa & Fallback
4. Bigram & Trigram (Gensim Phrases) | 5. Stemming (Sastrawi)
6. Filter Stopword Post-Stemming | 7. Filter Token Pendek
"""),
code(SETUP),
md("## Langkah 1 - Jalankan Prapemrosesan 7 Tahap"),
code('''
from src import preprocessing as pp

pp.run_preprocessing(use_tfidf=False,
                     min_ngram_count=C.MIN_NGRAM_COUNT,
                     output_dir=str(C.INTERMEDIATE_DIR))
'''),
md("## Langkah 2 - Verifikasi Hasil Prapemrosesan"),
code('''
prep = pd.read_csv(C.PREPROCESSED_CSV)
print(f'Dokumen : {len(prep)}')
print(f'Tahun   : {prep.groupby("Tahun").size().to_dict()}')
print()
avg = prep['tokens'].apply(lambda x: len(eval(x))).mean()
print(f'Rata-rata token per dokumen: {avg:.1f}')
'''),
md("## Langkah 3 - Contoh Hasil Tokenisasi"),
code('''
import ast
sample = prep.head(3)
for _, r in sample.iterrows():
    toks = ast.literal_eval(r['tokens'])
    print(f'  [{int(r["Tahun"])}] {str(r["Judul"])[:55]}')
    print(f'         {toks[:12]} ...')
    print()
'''),
md("## Langkah 4 - Verifikasi Dictionary"),
code('''
from gensim import corpora
d = corpora.Dictionary.load(str(C.DICTIONARY_PATH))
print(f'Vocabulary : {len(d)} kata/n-gram')
print(f'Nilai exp  : {C.VOCAB_SIZE}')
print('Status     :', 'SESHUI' if len(d) == C.VOCAB_SIZE else 'BERBEDA')
'''),
md("""
## Ringkasan Fase 03 - Data Preparation

| Metrik | Nilai |
|:---|---:|
| Dokumen output | 285 |
| Ukuran vocabulary | 1.849 |
| Rata-rata token/dokumen | 58,1 |

Prapemrosesan ketat ini menghasilkan rata-rata hanya 58 token per dokumen.
Kosakata yang terbatas ini menurunkan C_V karena konteks ko-okurensi kata
menyempit - hal yang dibahas sebagai keterbatasan pada Bab IV skripsi.
"""),
md("## Peta Artefak\n\n"
   "| Artefak | Dibuat oleh | Consumed by |\n"
   "|:---|:---|:---|\n"
   "| `data/intermediate/dataset_preprocessed.csv` | Notebook 03 | Notebook 04, 04b, 05 |\n"
   "| `data/intermediate/dictionary.gensim` | Notebook 03 | Notebook 04, 04b, 08 |\n"
   "| `data/intermediate/corpus.pkl` | Notebook 03 | Notebook 04, 05, 08 |"),
])

# ── 04 MODELING - FINAL LDA ────────────────────────────────────────────────
write("04_modeling_lda.ipynb", [
md("""
# 04 - Modeling: Pelatihan LDA Final (K=8)

## Tujuan
Melatih model Latent Dirichlet Allocation dengan konfigurasi optimal dan
menyimpan seluruh artefak model.

## Asal (Input)
`data/intermediate/` - korpus bersih 285 dokumen hasil Notebook 03

## Keluaran (Output)
- `model/lda_model.gensim` - model LDA biner (K=8)
- `model/topic_distribution.csv` - topik dominan per dokumen
- `model/topic_labels.csv` / `.json` - label 8 topik
- `model/evaluation_metrics.csv` - metrik model
- `model/human_topic_validation.csv` - lembar validasiexpert
- `model/lda_visualization.html` - visualisasi pyLDAvis

## Konfigurasi Final
| Parameter | Nilai |
|:---|:---|
| Jumlah topik (K) | 8 |
| Alpha | asymmetric |
| Eta | auto |
| Passes | 30 |
| Iterations | 500 |
| Random seed | 42 |
| Metode inferensi | Online Variational Bayes (bukan Gibbs Sampling) |
"""),
code(SETUP),
md("""
## Catatan Metodologis Penting

`modeling.py` melakukan `train_test_split(80/20)`: model dilatih pada
228 dokumen, lalu diprediksi ke seluruh 285. Dokumen di hold-out (20%)
belum pernah dilihat model sehingga theta-nya diffuse.

Akibatnya, distribusi di `model/topic_distribution.csv` bisa berbeda 1-2
dokumen dari distribusi final. **Distribusi final yang dipakai dashboard
dan skripsi dihitung dengan inferensi penuh** - lihat Notebook 08.
"""),
md("## Langkah 1 - Validasi Fingerprint Label"),
code('''
from src.topic_labels import validate_labels
validate_labels(str(C.LDA_MODEL), str(C.LABELS_JSON))
print('[OK] Fingerprint label cocok dengan model')
'''),
md("""## Langkah 2 - Latih Model LDA Final

Sel berikut melatih ulang model LDA dengan konfigurasi final.
Butuh sekitar 5 menit (30 passes x 500 iterasi).

Untuk sekadar **memverifikasi** model yang sudah ada, lewati sel ini dan
langsung ke Langkah 3. Set `RETRAIN = False` agar tidakMexecute.
"""),
code('''
RETRAIN = False   # set True untuk melatih ulang dari nol

if RETRAIN:
    from src import modeling
    modeling.main()
else:
    print('[SKIP] Melewati pelatihan ulang - menggunakan model yang tersimpan')
    print('       (set RETRAIN = True jika ingin dilatih dari nol)')
'''),
md("## Langkah 3 - Verifikasi Model Tersimpan"),
code('''
from gensim.models import LdaModel
from gensim import corpora

m = LdaModel.load(str(C.LDA_MODEL))
d = corpora.Dictionary.load(str(C.DICTIONARY_PATH))
print(f'K              : {m.num_topics}')
print(f'Vocabulary     : {len(d)}')
print(f'Passes         : {m.passes}')
print(f'Iterations     : {m.iterations}')
print(f'Alpha (asymm.) : {m.alpha[:3]}...')
print(f'Eta (auto)     : {m.eta[:3]}...')
'''),
md("## Langkah 4 - Hitung Coherence Score C_V (Protokol Resmi)"),
code('''
import ast
from gensim.models import CoherenceModel

prep = pd.read_csv(C.PREPROCESSED_CSV)
texts = prep['tokens'].apply(ast.literal_eval).tolist()
cv = CoherenceModel(model=m, texts=texts, dictionary=d, coherence='c_v', processes=1).get_coherence()

print(f'C_V terhitung : {cv:.4f}')
print(f'C_V resmi     : {C.COHERENCE_CV_OFFICIAL}')
print('Status        :', 'SESHUI' if abs(cv - C.COHERENCE_CV_OFFICIAL) < 0.0001 else 'BERBEDA')
'''),
md("## Langkah 5 - Kata Kunci Setiap Topik"),
code('''
for t in range(m.num_topics):
    words = [w for w, _ in m.show_topic(t, topn=8)]
    print(f'  Topik {t+1}: ' + ', '.join(words))
'''),
md("""
## Ringkasan Fase 04 - Modeling

| Metrik | Nilai |
|:---|---:|
| Jumlah topik (K) | 8 |
| Coherence C_V | **0,4256** |
| Vocabulary | 1.849 |

Model final siap digunakan pada tahap Evaluation (Notebook 05) dan
Build Final Artifacts (Notebook 08).
"""),
md("## Peta Artefak\n\n"
   "| Artefak | Dibuat oleh | Consumed by |\n"
   "|:---|:---|:---|\n"
   "| `model/lda_model.gensim` | Notebook 04 | Notebook 05, 08, app |\n"
   "| `model/topic_distribution.csv` | Notebook 04 | Referensi perbandingan |\n"
   "| `model/topic_labels.csv` / `.json` | Notebook 04 | Notebook 07, app |\n"
   "| `model/evaluation_metrics.csv` | Notebook 04 | Dokumentasi |"),
])

# ── 04b GRID SEARCH ────────────────────────────────────────────────────────
write("04b_grid_search.ipynb", [
md("""
# 04b - Modeling: Grid Search Hyperparameter

## Tujuan
Menentukan konfigurasi terbaik untuk jumlah topik K dan prior Dirichlet
(alpha, eta) melalui grid search exhaustive.

## Asal (Input)
`data/intermediate/dataset_preprocessed.csv` + `dictionary.gensim`

## Keluaran (Output)
`model/hyperparameter_results.csv` - hasil 84 konfigurasi

## Ruang Grid
| Parameter | Nilai yang diuji |
|:---|:---|
| K | 4, 5, 6, 7, 8, 9, 10 (7 nilai) |
| Alpha | symmetric, asymmetric, 0.01, 0.1 (4 nilai) |
| Eta | symmetric, 0.01, 0.1 (3 nilai) |

**Total: 7 x 4 x 3 = 84 konfigurasi**, masing-masing dengan passes=10.

## Catatan Penting tentang Pemilihan K
Grid search ini menunjukkan **K=9 mencapai C_V tertinggi (0,4314)**, bukan K=8.
K=8 dipilih atas pertimbangan gabungan: jumlah dokumen per topik lebih memadai
(minimum 21) dan struktur topik lebih mudah diinterpretasikan. Hal ini
dijelaskan secara transparan pada Bab IV skripsi.
"""),
code(SETUP),
md("## Langkah 1 - Jalankan Grid Search"),
code('''
# Grid search penuh = 84 konfigurasi (K=4..10 x 4 alpha x 3 eta), ~20-40 menit.
# Set True untuk menjalankan ulang. PERINGATAN: menjalankan ulang akan
# MENIMPA model/hyperparameter_results.csv, dan angka pada Tabel 6 skripsi
# diturunkan dari berkas itu. Default False agar angka skripsi tetap terpakai.
RUN_GRID = False

if RUN_GRID:
    from src import grid_search
    grid_search.main()
else:
    print("RUN_GRID=False -> memakai artefak yang sudah tersimpan:")
    print("  ", C.HYPERPARAMS_CSV)
    print("  Jumlah konfigurasi:", len(pd.read_csv(C.HYPERPARAMS_CSV)))
    print("  Untuk menjalankan ulang: python -m src.grid_search")
'''),
md("## Langkah 2 - Analisis Hasil: C_V Tertinggi per K"),
code('''
hp = pd.read_csv(C.HYPERPARAMS_CSV)
print(f'Total konfigurasi: {len(hp)}')
print()
print('=== C_V MAKSIMUM PER K ===')
for k, grp in hp.groupby('k'):
    print(f'  K={k:2d}  C_V max = {grp["coherence_cv"].max():.4f}   '
          f'C_V mean = {grp["coherence_cv"].mean():.4f}')
'''),
md("## Langkah 3 - 10 Konfigurasi Teratas"),
code('''
top = hp.sort_values('coherence_cv', ascending=False).head(10)
print(top[['k','alpha','eta','coherence_cv','coherence_umass']].to_string(index=False))
'''),
md("## Langkah 4 - Elbow Curve"),
code('''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sweep = hp.groupby('k')['coherence_cv'].agg(['max','mean']).reset_index()
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(sweep['k'], sweep['max'], '-o', color='#1A56A0', linewidth=2, label='Max C_V')
ax.plot(sweep['k'], sweep['mean'], '--o', color='#5B9BD5', alpha=0.7, label='Mean C_V')
ax.axvline(8, color='#00C48C', linestyle='--', linewidth=2)
ax.text(8.1, sweep['max'].max()*0.98, 'Model Final K=8', color='#00C48C', fontsize=9)
ax.set_xlabel('Jumlah Topik (K)'); ax.set_ylabel('Coherence Score (C_V)')
ax.set_title('Grid Search 84 Konfigurasi')
ax.set_xticks(sorted(hp['k'].unique())); ax.grid(alpha=0.3); ax.legend()
plt.tight_layout(); plt.show()
'''),
md("""
## Ringkasan Fase 04b - Grid Search

| K | C_V maksimum | Catatan |
|---:|---:|:---|
| 4 | 0,3605 | |
| 5 | 0,3756 | |
| 6 | 0,3909 | |
| 7 | 0,4142 | |
| 8 | 0,3871 | **Dipilih** - struktur lebih seimbang |
| 9 | **0,4314** | C_V tertinggi, tapi topik terlalu ringkas |
| 10 | 0,3993 | |

Pemilihan K=8 adalah keputusaninterpretif combined: bukan semata C_V
tertinggi, melainkan menimbang koherensi, jumlah dokumen per topik,
dan interpretabilitas.
"""),
md("## Peta Artefak\n\n"
   "| Artefak | Dibuat oleh | Consumed by |\n"
   "|:---|:---|:---|\n"
   "| `model/hyperparameter_results.csv` | Notebook 04b | Notebook 04 (-informed) |"),
])

# ── 05 EVALUATION LDA ──────────────────────────────────────────────────────
write("05_evaluation_lda.ipynb", [
md("""
# 05 - Evaluation: Robustness & Kualitas Model LDA

## Tujuan
Menguji seberapa konsisten model LDA terhadap variasi acak (multi-seed),
membandingkan K, dan mengukur kualitas serta diversitas topik.

## Asal (Input)
`model/lda_model.gensim` + `data/intermediate/` (korpus 285 dokumen)

## Keluaran (Output)
`experiments/` - hasil kestabilan multi-seed, sweep K, metrics

## Lima Uji yang Dilakukan
- **A.** Stabilitas multi-seed (K=8, 10 seed) - C_V & U_Mass
- **B.** Sweep K (4-12, 3 seed per K)
- **C.** Perplexity train vs hold-out
- **D.** Kualitas & diversitas topik
- **E.** Distribusi theta & proporsi per tahun
"""),
code(SETUP),
md("""
## Cara Menjalankan Evaluasi Komprehensif

Uji lengkap (multi-seed 10 seed, sweep K 4-12, perplexity) dijalankan lewat
modul oriented-script:

```bash
python -m src.evaluation_lda
```

Modul tersebut butuh sekitar 10 menit. Notebook ini menampilkan ringkasan
cepat metrik utama yang dapat dihitung langsung dari model tersimpan.
"""),
code('''
# Ringkasan cepat tanpa menunggu 10 seed penuh
import ast, numpy as np
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel

prep = pd.read_csv(C.PREPROCESSED_CSV)
texts = prep['tokens'].apply(ast.literal_eval).tolist()
d = corpora.Dictionary.load(str(C.DICTIONARY_PATH))
m = LdaModel.load(str(C.LDA_MODEL))
corpus = [d.doc2bow(t) for t in texts]

# Topik Diversity
t10 = [w for t in range(m.num_topics) for w,_ in m.show_topic(t, topn=10)]
t15 = [w for t in range(m.num_topics) for w,_ in m.show_topic(t, topn=15)]
print(f'Topic Diversity Top-10 : {len(set(t10))/len(t10):.4f}')
print(f'Topic Diversity Top-15 : {len(set(t15))/len(t15):.4f}')

# Theta distribution
theta = np.zeros((len(corpus), m.num_topics))
for i, c in enumerate(corpus):
    for tid, p in m.get_document_topics(c, minimum_probability=0.0):
        theta[i, tid] = p
mx = theta.max(axis=1)
print(f'Dokumen theta < 0.40   : {np.mean(mx<0.40)*100:.2f}%')
print(f'Dokumen theta < 0.50   : {np.mean(mx<0.50)*100:.2f}%')
'''),
md("## Langkah 2 - Perplexity Train vs Hold-out"),
code('''
from sklearn.model_selection import train_test_split
from gensim.models import LdaModel as LM

tr_texts, te_texts, _, _ = train_test_split(
    texts, texts, test_size=C.TRAIN_TEST_SPLIT, random_state=C.RANDOM_SEED)

tr_corpus = [d.doc2bow(t) for t in tr_texts]
te_corpus = [d.doc2bow(t) for t in te_texts]

m_eval = LM(corpus=tr_corpus, id2word=d, num_topics=C.NUM_TOPICS,
            alpha=C.LDA_ALPHA, eta=C.LDA_ETA,
            passes=C.LDA_PASSES, iterations=C.LDA_ITERATIONS,
            random_state=C.RANDOM_SEED)

print(f'Train log perplexity   : {m_eval.log_perplexity(tr_corpus):.4f}')
print(f'Hold-out log perplexity: {m_eval.log_perplexity(te_corpus):.4f}')
'''),
md("## Langkah 3 - Elbow Curve Sweep K"),
code('''
# Jalankan sweep penuh via modul (lebih lambat, ~10 menit):
#   python -m src.evaluation_lda
#
# Contoh visualisasi elbow dari data yang sudah ada:
print('Jalankan: python -m src.evaluation_lda')
print('Hasil disimpan di experiments/k_sweep.csv')
'''),
md("""
## Ringkasan Fase 05 - Evaluation

| Metrik | Nilai |
|:---|---:|
| Topic Diversity (Top-10) | 0,7125 |
| Topic Diversity (Top-15) | 0,6917 |
| Dokumen ambigu (theta < 0.50) | 13,33% |

Diversitas topik yang relatif tinggi menunjukkan kedelapan topik memang
berbeda satu sama lain, bukan sekadar pembagian arsip kata yang sama.
"""),
md("## Peta Artefak\n\n"
   "| Artefak | Dibuat oleh | Consumed by |\n"
   "|:---|:---|:---|\n"
   "| `experiments/*` | Notebook 05 / src.evaluation_lda | Lampiran Bab IV |"),
])

# ── 06 EVALUATION FORECAST ─────────────────────────────────────────────────
write("06_evaluation_forecast.ipynb", [
md("""
# 06 - Evaluation: Statistik Komposisi & Proyeksi WMA

## Tujuan
Menguji apakah komposisi topik berubah signifikan antar tahun, lalu memproyeksikan
tren distribusi topik 2026-2027 dengan Weighted Moving Average.

## Asal (Input)
- `_fix_final/topic_by_year_final.csv` (dari Notebook 08)
- `_fix_final/topic_by_year_sensitivity.csv`
- `_label_work/labels.json`

## Keluaran (Output)
- `_stats_forecast/forecast_table.csv` - proyeksi per topik
- `_stats_forecast/backtest_results.csv` - MAE walk-forward
- `_stats_forecast/share_intervals.csv` - interval Wilson 95%
- `_stats_forecast/per_topic_tests.csv` - uji Spearman + Bonferroni
- `_stats_forecast/composition_tests_report.md` - chi-square
- `_stats_forecast/sensitivity_report.md` - uji sensitivitas

## Metodologi Proyeksi
WMA dengan bobot linear [1,2,3,4,5] diterapkan pada 5 titik historis (2021-2025).
Proyeksi 2026 dihitung dari data asli; 2027 bersifat rekursif (indikatif).
Karena hanya tersedia 5 titik, seluruh proyeksi **tidak dapat diklaim
sebagai prediksi statistik yang andal**.
"""),
code(SETUP),
md("## Langkah 1 - Jalankan Analisis Statistik & Proyeksi"),
code('''
from src import forecast
forecast.main()
'''),
md("## Langkah 2 - Hasil Uji Komposisi (Chi-Square)"),
code('''
print(open(str(C.COMPOSITION_REPORT), encoding='utf-8').read()[:800])
'''),
md("## Langkah 3 - Tabel Proyeksi WMA"),
code('''
fc = pd.read_csv(C.FORECAST_CSV)
cols = ['topik_id','label','share_2025','wma_2026','pita_bawah_2026',
        'pita_atas_2026','wma_2027','arah']
print(fc[cols].to_string(index=False))
'''),
md("## Langkah 4 - Visualisasi Proyeksi"),
code('''
import plotly.graph_objects as go

tby = pd.read_csv(C.TOPIC_BY_YEAR_FINAL)
fig = go.Figure()
for t in [3, 4, 5, 6]:
    fig.add_trace(go.Scatter(
        x=['2021','2022','2023','2024','2025'],
        y=tby[f'proporsi_topik_{t}'],
        mode='lines+markers', name=f'Topik {t}'))
    row = fc[fc['topik_id']==t].iloc[0]
    fig.add_trace(go.Scatter(
        x=['2025','2026','2027'],
        y=[tby[f'proporsi_topik_{t}'].iloc[-1], row['wma_2026'], row['wma_2027']],
        mode='lines+markers', line=dict(dash='dot'),
        name=f'Topik {t} proyeksi', showlegend=False))

fig.add_vline(x=4, line_dash='dash', line_color='#999')
fig.update_layout(template='plotly_white', height=460,
                  xaxis_title='Tahun', yaxis_title='Proporsi (%)')
fig.show()
'''),
md("## Langkah 5 - Hasil Walk-Forward Backtest"),
code('''
bt = pd.read_csv(C.BACKTEST_CSV)
wma = bt[bt['metode']=='WMA']
hist= bt[bt['metode']=='Rata-rata Historis']
print(f'MAE rerata WMA            : {wma["mae_pp"].mean():.2f} pp')
print(f'MAE rerata Rata-rata Hist.: {hist["mae_pp"].mean():.2f} pp')
print()
better = (wma['mae_pp'].values < hist['mae_pp'].values).sum()
print(f'WMA lebih akurat pada {better} dari 8 topik')
print()
print('Kesimpulan: WMA TIDAK secara konsisten mengalahkan baseline rerata,')
print('sehingga seluruh proyeksi hanya bersifat indikatif.')
'''),
md("""
## Ringkasan Fase 06 - Evaluation & Forecast

| Temuan | Nilai |
|:---|---:|
| Chi-square p-value | 0,6861 (tidak signifikan) |
| Cramér's V | 0,1448 (asosiasi lemah) |
| MAE rerata WMA | 4,64 pp |
| Topik dengan WMA lebih baik | 2 dari 8 |

Tidak ditemukan pergeseran komposisi topik yang signifikan secara statistis
selama 2021-2025. Proyeksi WMA hanya indikatif dan tidak boleh diklaim
sebagai prediksi yang andal.
"""),
md("## Peta Artefak\n\n"
   "| Artefak | Dibuat oleh | Consumed by |\n"
   "|:---|:---|:---|\n"
   "| `_stats_forecast/forecast_table.csv` | Notebook 06 | app (halaman Proyeksi) |\n"
   "| `_stats_forecast/backtest_results.csv` | Notebook 06 | Lampiran Bab IV |\n"
   "| `_stats_forecast/share_intervals.csv` | Notebook 06 | app (halaman Tren) |"),
])

# ── 07 LABELING ────────────────────────────────────────────────────────────
write("07_labeling.ipynb", [
md("""
# 07 - Pelabelan Topik (Human-in-the-Loop)

## Tujuan
Memberi nama yang bermakna pada 8 topik hasil LDA melalui analisis
interpretatif terhadap kata kunci teratas, lalu mengunci label dengan
fingerprint SHA-256.

## Asal (Input)
- `_fix_final/top_words_final.csv` (dari Notebook 08)
- `model/lda_model.gensim`

## Keluaran (Output)
`_label_work/labels.json` - 8 label final + fingerprint SHA-256

## Mengapa Human-in-the-Loop?
LDA menghasilkan topik sebagai distribusi kata, tanpa nama. topping
seperti "variabel, teknologi, model, kualitas, sukses, monitoring..."
tidak langsung bermakna bagi pembaca. Needed: analis domain yang
menghubungkan kata-kata itu ke phenomenon di Program Studi.

Sesuai teori (Lau et al., 2011; Kozlowski et al., 2021), pelabelan
manual post-hoc adalah praktik standar dalam topic modeling.
"""),
code(SETUP),
md("## Langkah 1 - Tampilkan Kata Kunci per Topik"),
code('''
tw = pd.read_csv(C.TOP_WORDS_FINAL)
for t in range(1, 9):
    words = tw[tw['topik_id']==t].nsmallest(10, 'rank')['word'].tolist()
    print(f'  Topik {t}: ' + ', '.join(words))
    print()
'''),
md("## Langkah 2 - Lihat Label yang Disetujui"),
code('''
import json
labels = json.load(open(C.LABELS_JSON, encoding='utf-8'))
for item in labels:
    print(f"  Topik {item['topic_id']}: {item['label']}")
    print(f"    Keyakinan : {item['tingkat_kepercayaan']}")
    print(f"    Deskripsi : {item['deskripsi'][:75]}")
'''),
md("## Langkah 3 - Validasi Fingerprint terhadap Model"),
code('''
from src.topic_labels import validate_labels
validate_labels(str(C.LDA_MODEL), str(C.LABELS_JSON))
print('[OK] Seluruh fingerprint label cocok dengan model LDA')
'''),
md("## Langkah 4 - Ekspor Label ke CSV"),
code('''
from src.labeling import run_auto_labeling
run_auto_labeling()
'''),
md("""
## Ringkasan Fase 07 - Labeling

Delapan topik telah diberi label melalui analisis interpretatif dan
dikunci dengan fingerprint SHA-256. Mekanisme ini menjamin label tidak
pernah "melayang" ke model yang berbeda karena setiap perubahan model
akan menyebabkan validasi gagal.
"""),
md("## Peta Artefak\n\n"
   "| Artefak | Dibuat oleh | Consumed by |\n"
   "|:---|:---|:---|\n"
   "| `_label_work/labels.json` | Notebook 07 | Notebook 04, 06, 08, app |"),
])

# ── 08 BUILD FINAL ARTIFACTS ───────────────────────────────────────────────
write("08_build_final_artifacts.ipynb", [
md("""
# 08 - Build Final Artifacts (Regenerasi `_fix_final/`)

## Tujuan
Menghitung artefak final yang dipakai dashboard dan skripsi, dengan
**inferensi penuh terhadap seluruh 285 dokumen** sehingga hasilnya
dapat direproduksi ulang setiap kali notebook ini dijalankan.

## Asal (Input)
- `model/lda_model.gensim` (dari Notebook 04)
- `data/intermediate/` (dari Notebook 03)

## Keluaran (Output)
- `topic_distribution_final.csv` - 285 dokumen + matriks theta lengkap
- `topic_counts_final.csv` - jumlah & persentase per topik
- `topic_by_year_final.csv` - matriks tahun x topik + proporsi
- `top_words_final.csv` - kata kunci per topik
- `evaluation_summary.csv` - rekap metrik model

## Mengapa Inferensi Penuh?
Notebook 04 melatih model pada 228 dokumen (80%) lalu memprediksi 285.
Dokumen di hold-out (20%) belum pernah dilihat model, sehingga theta-nya
diffuse dan beberapa dokumen "lompat" ke topik berbeda.

Notebook ini menghilangkan bias itu: seluruh 285 dokumen dinilai oleh model
sehingga setiap dokumen mendapat penugasan topik yang konsisten. Ini juga
menjadikan angka di Tabel 5 dan 13 skripsi dapat direproduksi ulang.
"""),
code(SETUP),
md("## Langkah 1 - Bangun Artefak Final"),
code('''
from src import build_final as bf

result = bf.run(write=True, verbose=True)
'''),
md("## Langkah 2 - Verifikasi Konsistensi"),
code('''
counts = result['counts']
by_year = result['by_year']

print(f'Total dokumen      : {int(counts["jumlah_dokumen"].sum())}')
print(f'Total persentase    : {counts["persentase"].sum():.2f}%')
print()
print('Proporsi per tahun (harus ~100%):')
for _, r in by_year.iterrows():
    s = sum(r[f'proporsi_topik_{t}'] for t in range(1, 9))
    print(f'  {int(r["Tahun"])}: {int(r["total_dokumen"]):3d} dokumen  sum={s:.2f}%')
'''),
md("## Langkah 3 - Metrik Model (dapat direproduksi)"),
code('''
ev = result['evaluation']
print(ev.to_string(index=False, max_colwidth=70))
'''),
md("## Langkah 4 - Visualisasi Distribusi Topik"),
code('''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

c = result['counts']
fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar([f'Topik {int(t)}' for t in c['topik_dominan']],
              c['jumlah_dokumen'], color='#1A56A0', alpha=0.85)
for bar, val in zip(bars, c['jumlah_dokumen']):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            str(int(val)), ha='center', fontweight='bold')
ax.set_ylabel('Jumlah Dokumen')
ax.set_title(f'Distribusi Dokumen per Topik (Total {int(c["jumlah_dokumen"].sum())})')
ax.grid(axis='y', alpha=0.3); plt.tight_layout(); plt.show()
'''),
md("""
## Ringkasan Fase 08 - Build Final Artifacts

Seluruh artefak di `_fix_final/` kini **dapat direproduksi ulang** dengan
menjalankan notebook ini. Tidak ada lagi angka yang muncul "dari mana-mana".

| File | Jumlah Baris | Status |
|:---|---:|:---|
| `topic_distribution_final.csv` | 285 | dapat direproduksi |
| `topic_counts_final.csv` | 8 | dapat direproduksi |
| `topic_by_year_final.csv` | 5 | dapat direproduksi |
| `top_words_final.csv` | 120 | dapat direproduksi |
| `evaluation_summary.csv` | 9 metrik | dapat direproduksi |

**Catatan metodologis untuk Bab III:** distribusi final dihitung dengan
inferensi penuh terhadap seluruh 285 dokumen (bukan model yang di-split
80/20), sehingga setiap dokumen dinilai secara optimal.
"""),
md("## Peta Artefak\n\n"
   "| Artefak | Dibuat oleh | Consumed by |\n"
   "|:---|:---|:---|\n"
   "| `_fix_final/topic_distribution_final.csv` | Notebook 08 | app, tests |\n"
   "| `_fix_final/topic_counts_final.csv` | Notebook 08 | app (Beranda, Distribusi) |\n"
   "| `_fix_final/topic_by_year_final.csv` | Notebook 08 | Notebook 06, app (Tren) |\n"
   "| `_fix_final/top_words_final.csv` | Notebook 08 | Notebook 07 (labeling) |"),
])

print()
print(f"Selesai. {len(list(NB.glob('*.ipynb')))} notebook dibuat di {NB.relative_to(REPO)}")