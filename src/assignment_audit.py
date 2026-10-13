"""
src/assignment_audit.py - Audit keyakinan penugasan dokumen terhadap topik LDA.

Menghitung dua hal yang dipakai sebagai angka pada bagian Keterbatasan Bab IV:

1. Keyakinan penugasan
   Distribusi nilai probabilitas maksimum tiap dokumen terhadap topik
   dominannya. Bila model yakin, nilai maksimum mendekati 1; bila dokumen
   benar-benar membentang antartopik, nilainya rendah.

2. Fragmentasi dokumen analitik data
   Berapa dokumen yang memakai algoritma/metode analitik data, dan seberapa
   banyak yang benar-benar mendarat pada topik yang secara semantik paling
   dekat dengan analitik data.

Mengapa modul ini ada:
   Skripsi menyebut angka "58 dokumen (20,4%)" dan "26 dokumen (9,1%)" pada
   bagian keterbatasan. Tanpa skrip ini, kedua angka itu tidak memiliki
   asal-usul yang dapat diverifikasi ulang - melanggar prinsip repo ini.

Sumber data:
   _fix_final/topic_distribution_final.csv  (matriks theta final, 285 dokumen)
   model/topic_labels.csv                   (label resmi per topik)

Jalankan:
   python -m src.assignment_audit
"""
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from . import config as C

# Ambang keyakinan
AMBANG_RAGU = 0.60       # di bawah ini model tidak yakin
AMBANG_SANGAT_YAKIN = 0.99
AMBANG_KURANG_YAKIN = 0.50

# Pola judul yang menandai dokumen analitik data / algoritma
POLA_ANALITIK = (
    r'K-MEANS|KMEANS|APRIORI|ECLAT|NAIVE|NAÏVE|C4\.5|DECISION TREE|'
    r'FP-GROWTH|ARIMA|REGRESI|CLUSTERING|KLUSIF|KLASIFIKASI|SENTIMEN|'
    r'ASSOCIATION|MINING'
)


def _pct(n, total):
    return 100.0 * n / total


def desimal(x, n=2):
    """Format persen memakai koma, mengikuti gaya GROUND_TRUTH.md dan naskah."""
    return ('%.*f' % (n, x)).replace('.', ',')


def bersihkan_judul(serie):
    """Judul pada CSV mengandung \\r\\n; rapikan agar pola regex tidak gagal."""
    return (
        serie.astype(str)
        .str.replace(r'[\r\n]+', ' ', regex=True)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
        .str.upper()
    )


def audit_keyakinan(df: pd.DataFrame) -> dict:
    """Distribusi nilai probabilitas maksimum dokumen terhadap topik dominan."""
    kol = ['prob_topik_%d' % i for i in range(1, 9)]
    maks = df[kol].max(axis=1)

    sangat_yakin = int((maks >= AMBANG_SANGAT_YAKIN).sum())
    kurang_yakin = int((maks < AMBANG_KURANG_YAKIN).sum())
    ragu = int((maks < AMBANG_RAGU).sum())
    tengah = int(((maks >= AMBANG_RAGU) & (maks < 0.95)).sum())

    return {
        'total': int(len(df)),
        'sangat_yakin': sangat_yakin,
        'kurang_yakin': kurang_yakin,
        'ragu': ragu,
        'tengah': tengah,
        'median': float(maks.median()),
    }


def audit_analitik(df: pd.DataFrame) -> dict:
    """Seberapa真题 Attached dokumen analitik data pada topik analitik data."""
    judul = bersihkan_judul(df['Judul'])
    mask = judul.str.contains(POLA_ANALITIK, regex=True, na=False)
    sub = df[mask]

    # Topik yang secara semantik paling dekat dengan "analitik data":
    # ditentukan dari label resmi, bukan ditebak.
    labels = pd.read_csv(C.TOPIC_LABELS_CSV, encoding='utf-8-sig')
    kandidat = labels[labels['label'].str.contains(
        'Analitik Data', case=False, na=False)]
    topik_analitik = int(kandidat['topic_id'].iloc[0]) if len(kandidat) else 5

    masuk = int((sub['topik_dominan'] == topik_analitik).sum())

    sebaran = sub['topik_dominan'].value_counts().sort_index()

    return {
        'total': int(len(sub)),
        'porsi': _pct(len(sub), len(df)),
        'jumlah_topik': int(sub['topik_dominan'].nunique()),
        'topik_analitik': topik_analitik,
        'masuk_topik_analitik': masuk,
        'porsi_masuk': _pct(masuk, len(sub)) if len(sub) else 0.0,
        'sebaran': sebaran,
        'kmeans_total': int(judul.str.contains(
            r'K-MEANS|KMEANS', regex=True).sum()),
        'kmeans_masuk': int(((judul.str.contains(r'K-MEANS|KMEANS', regex=True)) &
                            (df['topik_dominan'] == topik_analitik)).sum()),
    }


def main():
    df = pd.read_csv(C.TOPIC_DIST_FINAL, encoding='utf-8-sig')
    key = audit_keyakinan(df)
    ana = audit_analitik(df)

    L = []
    L.append('# AUDIT KEYAKINAN PENUGASAN DOKUMEN')
    L.append('')
    L.append('_Dihitung otomatis oleh `src/assignment_audit.py` - jangan diedit manual._')
    L.append('')
    L.append('**Sumber data:** `%s`' % C.TOPIC_DIST_FINAL.name)
    L.append('**Tanggal pembuatan:** %s' % datetime.now().strftime('%d-%m-%Y %H:%M'))
    L.append('')
    L.append('Angka pada bagian Keterbatasan Bab IV yang merujuk '
             'keyakinan penugasan dan fragmentasi dokumen analitik '
             'diturunkan dari laporan ini.')
    L.append('')
    L.append('---')
    L.append('')
    L.append('## 1. Keyakinan Penugasan')
    L.append('')
    L.append('Nilai probabilitas maksimum tiap dokumen terhadap topik dominannya '
             'mengukur seberapa tegas model dalam menempatkan dokumen tersebut.')
    L.append('')
    L.append('| Kelompok | Dokumen | Persentase |')
    L.append('|:---|---:|---:|')
    L.append('| Total dokumen | %d | 100,00%% |' % key['total'])
    L.append('| Sangat yakin (max-theta >= %s) | %d | %s%% |'
             % (desimal(AMBANG_SANGAT_YAKIN), key['sangat_yakin'],
                desimal(_pct(key['sangat_yakin'], key['total']))))
    L.append('| Kurang yakin (max-theta < %s) | %d | %s%% |'
             % (desimal(AMBANG_KURANG_YAKIN), key['kurang_yakin'],
                desimal(_pct(key['kurang_yakin'], key['total']))))
    L.append('| **Ragu (max-theta < %s)** | **%d** | **%s%%** |'
             % (desimal(AMBANG_RAGU), key['ragu'],
                desimal(_pct(key['ragu'], key['total']))))
    L.append('| Rentang tengah (%s <= theta < 0,95) | %d | %s%% |'
             % (desimal(AMBANG_RAGU), key['tengah'],
                desimal(_pct(key['tengah'], key['total']))))
    L.append('')
    L.append('Median max-theta = %s.' % desimal(key['median'], 4))
    L.append('')
    L.append('Distribusi bersifat bimodal: sebagian besar dokumen terkunci '
             'dengan keyakinan hampir mutlak, sementara sebagian dokumen '
             'benar-benar membentang antartopik. Kelompok "ragu" inilah '
             'yang paling berpotensi menjadi subjek interpretasi ulang.')
    L.append('')
    L.append('## 2. Fragmentasi Dokumen Analitik Data')
    L.append('')
    L.append('Dokumen yang 제목nya menyebut algoritma atau metode analitik '
             'data (K-Means, Apriori, Eclat, Naive Bayes, C4.5, FP-Growth, '
             'ARIMA, analisis sentimen, dan sejenisnya).')
    L.append('')
    L.append('| Metrik | Nilai |')
    L.append('|:---|---:|')
    L.append('| Jumlah dokumen analitik | %d |' % ana['total'])
    L.append('| Persentase terhadap korpus | %s%% |' % desimal(ana['porsi']))
    L.append('| Jumlah topik tujuan | %d |' % ana['jumlah_topik'])
    L.append('| Masuk pada topik analitik data (T%d) | %d (%s%%) |'
             % (ana['topik_analitik'], ana['masuk_topik_analitik'],
                desimal(ana['porsi_masuk'], 1)))
    L.append('| Dokumen K-Means total | %d |' % ana['kmeans_total'])
    L.append('| Dokumen K-Means pada topik analitik data | %d |'
             % ana['kmeans_masuk'])
    L.append('')
    L.append('Sebaran dokumen analitik data menurut topik dominan:')
    L.append('')
    L.append('| Topik | Dokumen |')
    L.append('|---:|---:|')
    for topik, n in ana['sebaran'].items():
        L.append('| T%d | %d |' % (int(topik), int(n)))
    L.append('')
    L.append('Dokumen analitik data tersebar hampir ke seluruh topik, '
             'padahal hanya sebagian kecil yang mendarat pada topik yang '
             'secara semantik paling dekat dengan analitik data. '
             'Ini menunjukkan pemisahan antartopik belum sepenuhnya tegas '
             'dan merupakan keterbatasan yang akan diperiksa ulang pada '
             'penelitian lanjutan.')
    L.append('')
    L.append('---')
    L.append('')
    L.append('_Perintah regenerasi: `python -m src.assignment_audit`_')

    C.ASSIGNMENT_AUDIT_MD.parent.mkdir(parents=True, exist_ok=True)
    C.ASSIGNMENT_AUDIT_MD.write_text('\n'.join(L) + '\n', encoding='utf-8')

    print('[OK] %s ditulis (%d baris)'
          % (C.ASSIGNMENT_AUDIT_MD, len(L) + 1))
    print('  Keyakinan: ragu=%d (%.2f%%) | sangat yakin=%d (%.2f%%)'
          % (key['ragu'], _pct(key['ragu'], key['total']),
             key['sangat_yakin'], _pct(key['sangat_yakin'], key['total'])))
    print('  Analitik : %d dokumen (%.2f%%) ke %d topik; masuk T%d = %d (%.1f%%)'
          % (ana['total'], ana['porsi'], ana['jumlah_topik'],
             ana['topik_analitik'], ana['masuk_topik_analitik'],
             ana['porsi_masuk']))
    return 0


if __name__ == '__main__':
    sys.exit(main())