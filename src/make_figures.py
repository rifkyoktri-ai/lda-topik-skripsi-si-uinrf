# -*- coding: utf-8 -*-
"""
src/make_figures.py - Bangkitkan ulang gambar untuk naskah skripsi.

KENAPA MODUL INI ADA
   _repo ini pernah menyimpan wordcloud topik 1-7 sebagai berkas PNG,
    TANPA skrip penghasilnya. Gambar yang tidak bisa dibangkitkan ulang
    tidak bisa diverifikasi, dan kalau label topik berubah, gambarnya
    diam-diam jadi kedaluwarsa. Modul ini menutup celah itu: setiap PNG
    punya skrip pembuatnya, dan bisa dibangkitkan ulang dari model.

Keluaran
    outputs/figures/wordclouds/wordcloud_topic_1..8.png

Sumber kata & bobot
    Langsung dari model LDA tersimpan (outputs/model/lda_model.gensim),
    bukan dari CSV mana pun - supaya selalu identik dengan model yang
    menghasilkan angka di skripsi.

Jalankan
    python -m src.make_figures            # semua topik
    python -m src.make_figures --topik 8  # hanya satu topik
"""
import argparse
import sys

import matplotlib
matplotlib.use("Agg")          # tanpa jendela; aman di server/headless
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from . import config as C

# ── Gaya visual (selaras dengan tema dashboard) ───────────────────────────────
LATAR = "#0B0F19"
WARNA = "#1F2937"
TEKS = "#F3F4F6"
AKSEN = "#3B82F6"
PALET = ["#3B82F6", "#60A5FA", "#93C5FD", "#22D3EE", "#34D399",
         "#FBBF24", "#F472B6", "#A78BFA", "#FB7185", "#FCD34D"]

# Jumlah kata yang ditampilkan per topik.
#
# Sengaja disamakan dengan 15 - jumlah kata kunci yang dicetak di Tabel 8
# naskah skripsi. Kalau angka ini diubah, gambar tidak lagi cocok dengan
# tabel dan akan sulit dipertanggungjawabkan saat sidang.
#
# Saat diuji dengan 45 kata, muncul istilah yang mengaburkan makna topik
# (mis. "extreme programming" dan "tracer study" pada Topik 8) karena
# keduanya hanya peringkat 45 dengan bobot 0,0036.
N_KATA = 15
SEED = 42                     # supaya hasilnya bisa direproduksi


def ambil_kata(lda_model, topik: int, n: int = N_KATA) -> dict:
    """
    Ambil n kata teratas beserta bobotnya langsung dari model.

    Tanda underscore pada bigram diganti spasi supaya terbaca
    ("bimbing_konseling" -> "bimbing konseling").
    """
    pasangan = lda_model.show_topic(topik - 1, topn=n)
    kata = {}
    for w, p in pasangan:
        if p <= 0:
            continue
        bersih = str(w).replace("_", " ").strip()
        if bersih:
            kata[bersih] = kata.get(bersih, 0.0) + float(p)
    return kata


def buat_wordcloud(kata: dict, judul: str, lebar=1200, tinggi=700) -> WordCloud:
    """Susun WordCloud. collocations=False supaya bigram tidak dipecah."""
    return WordCloud(
        width=lebar, height=tinggi,
        background_color=LATAR,
        colormap=None,
        font_path=None,
        random_state=SEED,
        collocations=False,
        prefer_horizontal=0.85,
        max_font_size=150,
        min_font_size=12,
        relative_scaling=0.35,
        margin=2,
    ).generate_from_frequencies(kata)


def simpan_satu(lda_model, topik: int, label: str, out_dir):
    """Bangkitkan dan simpan wordcloud satu topik."""
    import os
    os.makedirs(out_dir, exist_ok=True)

    kata = ambil_kata(lda_model, topik)
    if not kata:
        print("  [LEWAT] Topik %d tidak punya kata berbobot" % topik)
        return None

    # Kanvas diberi area khusus untuk judul dan kaki gambar, supaya tidak
    # tertimpa awan kata.
    fig = plt.figure(figsize=(13, 8.6), facecolor=LATAR)

    KEPALA = [0.0, 0.855, 1.0, 0.145]    # kiri, bawah, lebar, tinggi
    BADAN = [0.01, 0.085, 0.98, 0.755]
    KAKI = [0.0, 0.0, 1.0, 0.075]

    ax = fig.add_axes(BADAN)
    ax.set_facecolor(LATAR)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("tema", PALET)
    wc = buat_wordcloud(kata, label, lebar=1300, tinggi=1000)
    wc.colormap = cmap
    ax.imshow(wc, interpolation="bilinear")

    # kepala
    fig.text(0.035, 0.960, "TOPIK %d" % topik, color=AKSEN,
             fontsize=15, fontweight="bold", va="top")
    fig.text(0.035, 0.912, label, color=TEKS, fontsize=19, va="top",
             fontweight="bold")
    fig.add_artist(plt.Line2D([0.035, 0.965], [0.878, 0.878],
                              color="#374151", linewidth=1.2))

    # kaki: 6 kata teratas + keterangan sumber
    enam = sorted(kata, key=lambda w: -kata[w])[:6]
    fig.text(0.035, 0.042, " · ".join(enam), color="#93C5FD",
             fontsize=12.5, va="center")
    fig.text(0.965, 0.042,
             "UIN Raden Fatah · LDA K=8 · 2021–2025", color="#6B7280",
             fontsize=11, ha="right", va="center")

    path = "%s/wordcloud_topic_%d.png" % (out_dir, topik)
    fig.savefig(path, facecolor=LATAR, dpi=110)
    plt.close(fig)
    return path


def main():
    from gensim.models import LdaModel

    parser = argparse.ArgumentParser(description="Bangkitkan gambar skripsi")
    parser.add_argument("--topik", type=int, nargs="*", default=None,
                        help="nomor topik yang dibuat (default: semua)")
    args = parser.parse_args()

    lda = LdaModel.load(str(C.LDA_MODEL))

    import pandas as pd
    labels = pd.read_csv(C.TOPIC_LABELS_CSV, encoding="utf-8-sig")
    peta = dict(zip(labels["topic_id"].astype(int), labels["label"]))

    topik_list = args.topik or sorted(peta)
    print("\n[BANGKITKAN WORDCLOUD] %d topik -> %s"
          % (len(topik_list), C.WORDCLOUD_DIR.name + "/"))
    print("  sumber kata: %s" % C.LDA_MODEL.name)
    print("  %d kata teratas per topik, seed=%d" % (N_KATA, SEED))
    print()

    C.WORDCLOUD_DIR.mkdir(parents=True, exist_ok=True)
    for t in topik_list:
        path = simpan_satu(lda, t, peta.get(t, "?"), str(C.WORDCLOUD_DIR))
        if path:
            import os
            print("  [OK] Topik %-2d %-46s %6.0f KB"
                  % (t, peta.get(t, "?")[:46], os.path.getsize(path) / 1024))

    print("\n  Selesai. Gambar bisa dibangkitkan ulang kapan saja dengan:")
    print("    python -m src.make_figures\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())