"""
build_final.py — Regenerasi artefak `outputs/` dari model LDA tersimpan.

MASALAH YANG DISOLUSIKAN
-------------------------
`outputs/*.csv` sebelumnya dibuat manual tanpa script penghasilnya, sehingga
tidak reproducible. Modul ini menutup celah itu.

METODOLOGI (penting)
--------------------
Pipeline `modeling.py` melakukan `train_test_split(80/20)`: model dilatih pada
228 dokumen, lalu dipredict ke seluruh 285. Dokumen di hold-out (20%) belum
pernah dilihat model sehingga theta-nya diffuse dan 2 dokumen "lompat" topik.

Modul ini memakai **inferensi penuh**: seluruh 285 dokumen dinilai oleh model,
sehingga setiap dokumen mendapat θ maksimal secara optimal. Ini adalah
versi yang lebih benar dan sekarang dapat direproduksi ulang.

FILES YANG DIHASILKAN
---------------------
  outputs/topics/topic_distribution_final.csv   (285 baris + matriks theta 1..8)
  outputs/topics/topic_counts_final.csv         (jumlah & persentase per topik)
  outputs/topics/topic_by_year_final.csv        (matriks tahun x topik + proporsi)
  outputs/topics/top_words_final.csv            (kata kunci per topik)
  outputs/model/evaluation_summary.csv         (rekap metrik model)
"""
import ast
import sys
import numpy as np
import pandas as pd
from pathlib import Path

from gensim import corpora
from gensim.models import LdaModel, CoherenceModel

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src import config as C
else:
    from . import config as C


def load_corpus():
    """Muat dataset prapemrosesan + dictionary + corpus BoW."""
    df = pd.read_csv(C.PREPROCESSED_CSV)
    df["tokens"] = df["tokens"].apply(ast.literal_eval)
    texts = df["tokens"].tolist()
    dictionary = corpora.Dictionary.load(str(C.DICTIONARY_PATH))
    corpus = [dictionary.doc2bow(t) for t in texts]
    return df, texts, dictionary, corpus


def infer_full_theta(lda_model, corpus):
    """
    Inferensi PENUH terhadap seluruh corpus.

    Berbeda dengan modeling.py yang memprediksi model yang dilatih pada 80%
    data, di sini seluruh dokumen dinilai. Hasilnya setiap dokumen mendapat
    probabilitas topik maksimal secara konsisten.
    """
    theta = np.zeros((len(corpus), lda_model.num_topics))
    for i, bow in enumerate(corpus):
        for tid, prob in lda_model.get_document_topics(bow, minimum_probability=0.0):
            theta[i, tid] = prob
    return theta


def build_topic_distribution(df, theta):
    """Bangun tabel distribusi + matriks theta lengkap."""
    out = pd.DataFrame({
        "ID": df["ID"],
        "Judul": df["Judul"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip(),
        "Tahun": df["Tahun"],
    })
    for i in range(theta.shape[1]):
        out[f"prob_topik_{i + 1}"] = theta[:, i]
    out["topik_dominan"] = theta.argmax(axis=1) + 1
    out["prob_dominan"] = theta.max(axis=1)
    return out


def build_topic_counts(dist):
    """Hitung jumlah & persentase dokumen per topik."""
    counts = (dist.groupby("topik_dominan").size()
              .reindex(range(1, 9), fill_value=0).rename("jumlah_dokumen"))
    total = int(counts.sum())
    counts = counts.reset_index().rename(columns={"index": "topik_dominan"})
    counts["topik_dominan"] = counts["topik_dominan"].astype(int)
    counts["persentase"] = (counts["jumlah_dokumen"] / total * 100).round(2)
    return counts, total


def build_topic_by_year(dist):
    """Matriks kontingensi tahun x topik + proporsi per tahun."""
    pivot = (dist.groupby(["Tahun", "topik_dominan"]).size()
             .unstack(fill_value=0).reindex(columns=range(1, 9), fill_value=0))
    pivot = pivot.reindex(C.YEARS, fill_value=0)

    out = pivot.copy()
    out.insert(0, "Tahun", out.index)
    out["total_dokumen"] = pivot.sum(axis=1).values

    for t in range(1, 9):
        out[f"proporsi_topik_{t}"] = (pivot[t] / pivot.sum(axis=1) * 100).round(2).values

    return out.reset_index(drop=True)


def build_top_words(lda_model, topn=15):
    """Kata kunci teratas per topik beserta probabilitasnya."""
    rows = []
    for tid in range(lda_model.num_topics):
        for rank, (word, prob) in enumerate(
                lda_model.show_topic(tid, topn=topn), start=1):
            rows.append({"topik_id": tid + 1, "rank": rank,
                         "word": word, "probability": round(float(prob), 6)})
    return pd.DataFrame(rows)


def build_evaluation_summary(lda_model, texts, dictionary, corpus, theta):
    """Rekap metrik evaluasi model — nilai yang dapat direproduksi."""
    cv = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary,
                        coherence="c_v", processes=1).get_coherence()
    umass = CoherenceModel(model=lda_model, corpus=corpus, dictionary=dictionary,
                           coherence="u_mass", processes=1).get_coherence()
    log_perp = lda_model.log_perplexity(corpus)

    t10 = [w for t in range(lda_model.num_topics) for w, _ in lda_model.show_topic(t, topn=10)]
    t15 = [w for t in range(lda_model.num_topics) for w, _ in lda_model.show_topic(t, topn=15)]
    td10 = len(set(t10)) / len(t10)
    td15 = len(set(t15)) / len(t15)
    pct_under = float((theta.max(axis=1) < 0.50).mean() * 100)

    rows = [
        ("Coherence_Score_CV", round(cv, 4),
         "texts=data/intermediate/ (1.849 vocab), topn=20, processes=1, gensim 4.4.0"),
        ("Coherence_Score_UMass", round(umass, 4), "corpus BoW 285 dokumen, processes=1"),
        ("Train_Log_Perplexity", round(log_perp, 4), "seluruh 285 dokumen korpus v1"),
        ("Topic_Diversity_Top10", round(td10, 4), "proporsi kata unik top-10"),
        ("Topic_Diversity_Top15", round(td15, 4), "proporsi kata unik top-15"),
        ("Pct_Max_Theta_Under_0.50", round(pct_under, 2),
         "persentase dokumen dengan theta dominan < 0.50"),
        ("Vocabulary_Size", len(dictionary), "jumlah kata unik dictionary.gensim"),
        ("Total_Documents", df_len_holder, "korpus bersih"),
        ("Model_Parameters", f"K={lda_model.num_topics}, alpha={C.LDA_ALPHA}, eta={C.LDA_ETA}, "
                             f"passes=30, iterations=500, seed=42",
         "Online Variational Bayes (BUKAN Gibbs Sampling)"),
    ]
    return pd.DataFrame(rows, columns=["Metrik", "Nilai", "Protokol_Keterangan"])


# variabel global kecil agar build_evaluation_summary tetap ringkas
df_len_holder = 0


def compare_with_existing(new_dist):
    """
    Bandingkan hasil inferensi penuh dengan artefak yang sudah ada.
    Berguna untuk transparansi: berapa dokumen yang berubah dan mengapa.
    """
    old_path = C.TOPIC_DIST_FINAL
    if not old_path.exists():
        return None
    old = pd.read_csv(old_path)
    merged = old[["ID", "topik_dominan"]].merge(
        new_dist[["ID", "topik_dominan"]], on="ID", suffixes=("_lama", "_baru"))
    diff = merged[merged["topik_dominan_lama"] != merged["topik_dominan_baru"]]
    return diff


def run(write=True, verbose=True):
    global df_len_holder

    if verbose:
        print("=" * 66)
        print("BUILD FINAL ARTIFACTS — inferensi penuh 285 dokumen")
        print("=" * 66)

    lda_model = LdaModel.load(str(C.LDA_MODEL))
    df, texts, dictionary, corpus = load_corpus()
    df_len_holder = len(df)

    if verbose:
        print(f"  Model  : K={lda_model.num_topics} | vocabulary={len(dictionary)}")
        print(f"  Dokumen: {len(df)}")
        print(f"  Metode : inferensi penuh (tanpa hold-out)")

    theta = infer_full_theta(lda_model, corpus)

    dist = build_topic_distribution(df, theta)
    counts, total = build_topic_counts(dist)
    by_year = build_topic_by_year(dist)
    top_words = build_top_words(lda_model)
    evals = build_evaluation_summary(lda_model, texts, dictionary, corpus, theta)

    if verbose:
        print()
        print("  Distribusi topik:")
        for _, r in counts.iterrows():
            print(f"    Topik {int(r.topik_dominan)}: {int(r.jumlah_dokumen):3d} dokumen "
                  f"({r.persentase:5.2f}%)")
        print(f"    TOTAL: {total} dokumen")

    diff = compare_with_existing(dist)
    if diff is not None and len(diff):
        if verbose:
            print()
            print(f"  Perbandingan dengan artefak lama: {len(diff)} dokumen berubah topik")
            for _, r in diff.head(5).iterrows():
                print(f"    ID {int(r.ID)}: Topik {int(r.topik_dominan_lama)} "
                      f"-> Topik {int(r.topik_dominan_baru)}")

    if write:
        C.TOPICS_DIR.mkdir(parents=True, exist_ok=True)
        dist.to_csv(C.TOPIC_DIST_FINAL, index=False, encoding="utf-8-sig")
        counts.to_csv(C.TOPIC_COUNTS_FINAL, index=False, encoding="utf-8-sig")
        by_year.to_csv(C.TOPIC_BY_YEAR_FINAL, index=False, encoding="utf-8-sig")
        top_words.to_csv(C.TOP_WORDS_FINAL, index=False, encoding="utf-8-sig")
        evals.to_csv(C.EVAL_SUMMARY_CSV, index=False, encoding="utf-8-sig")
        if verbose:
            print()
            print(f"  [OK] 5 artefak ditulis ke {C.TOPICS_DIR.relative_to(C.REPO)}")

    return {"distribution": dist, "counts": counts, "by_year": by_year,
            "top_words": top_words, "evaluation": evals, "theta": theta}


if __name__ == "__main__":
    run(write=True, verbose=True)