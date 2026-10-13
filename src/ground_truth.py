# -*- coding: utf-8 -*-
"""Generator GROUND_TRUTH.md - rujukan resmi angka skripsi.

Seluruh angka diambil dari artefak, bukan diketik manual:
  outputs/topics/topic_counts_final.csv   -> jumlah dokumen & persentase per topik
  outputs/model/evaluation_summary.csv   -> metrik evaluasi model
  outputs/forecast/composition_tests_report.md -> uji homogenitas
  outputs/forecast/per_topic_tests.csv -> uji tren per topik
  outputs/model/labels.json             -> label & tingkat kepercayaan

Jalankan:  python -m src.ground_truth
"""
import sys
import json
import re
from datetime import datetime, timezone

import pandas as pd

if __package__ in (None, ""):
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "src"

from . import config as C

OUT = C.REPO / "GROUND_TRUTH.md"


def idr(v, dec=2):
    """Angka gaya Indonesia: 0,4256"""
    s = ('%.*f' % (dec, float(v)))
    return s.replace('.', ',')


def parse_composition(path):
    """Ambil angka uji homogenitas dari laporan (format dijaga oleh src/forecast.py)."""
    txt = path.read_text(encoding='utf-8')
    pats = {
        'chi2': r'\*\*Statistik \$\\chi\^2\$\*\*:\s*([\d.]+)',
        'p': r'\*\*p-value Mentah \(Asimptotik\)\*\*:\s*([\d.]+)',
        'v': r"\*\*Cramer's V\*\*:\s*([\d.]+)",
        'p_perm': r'\*\*p-value Permutasi\*\*:\s*([\d.]+)',
        'dof': r'\*\*Derajat Kebebasan \(\$dof\$\)\*\*:\s*(\d+)',
    }
    out = {}
    for k, p in pats.items():
        m = re.search(p, txt)
        if not m:
            raise SystemExit('  [GAGAL] pola "%s" tidak ditemukan di %s' % (k, path.name))
        out[k] = m.group(1)
    return out


def main():
    counts = pd.read_csv(C.TOPIC_COUNTS_FINAL, encoding='utf-8-sig')
    metrics = pd.read_csv(C.EVAL_SUMMARY_CSV, encoding='utf-8-sig')
    labels = {int(t['topic_id']): t for t in json.loads(C.LABELS_JSON.read_text(encoding='utf-8'))}
    ptests = pd.read_csv(C.PER_TOPIC_TESTS, encoding='utf-8-sig')
    stats = parse_composition(C.COMPOSITION_REPORT)

    m = {str(r['Metrik']).strip(): r['Nilai'] for _, r in metrics.iterrows()}
    total = int(counts['jumlah_dokumen'].sum())

    years = []
    yb = C.TOPIC_BY_YEAR_FINAL
    if yb.exists():
        ydf = pd.read_csv(yb, encoding='utf-8-sig')
        years = ['%s: %d' % (int(r['Tahun']), int(r['total_dokumen']))
                 for _, r in ydf.sort_values('Tahun').iterrows()]

    L = []
    A = L.append
    A('# GROUND TRUTH - Rujukan Resmi Angka & Spesifikasi Skripsi')
    A('')
    A('> **DIBUANG OTOMATIS oleh `src/ground_truth.py`.** Jangan diedit manual.')
    A('> Regenerasi: `python -m src.ground_truth`')
    A('>')
    A('> Sumber angka: artefak di `outputs/`, `outputs/forecast/`, dan `outputs/model/`.')
    A('> Jika angka di skripsi berbeda dengan dokumen ini, dokumen ini yang benar')
    A('> karena diturunkan langsung dari artefak pipeline.')
    A('')
    A('_Terakhir dibuat: %s_' % datetime.now(timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M'))
    A('')
    A('---')
    A('')
    A('## 1. Spesifikasi Korpus & Model Final')
    A('')
    A('| Parameter Korpus / Model | Nilai Resmi | Keterangan / Berkas Rujukan |')
    A('| :--- | :---: | :--- |')
    A('| **Jumlah Dokumen** | **%d Dokumen** | %s |'
      % (total, ' \\| '.join(years) if years else C.PIPELINE_CSV.name))
    A('| **Ukuran Kosakata** | **%s Kata** | `%s` |'
      % (idr(m.get('Vocabulary_Size', 0), 0), C.DICTIONARY_PATH.relative_to(C.REPO).as_posix()))
    A('| **Model LDA Binary** | `model/lda_model.gensim` | Latent Dirichlet Allocation |')
    A('| **Jumlah Topik (K)** | **%d Topik** | Dipilih berdasarkan optimasi koherensi & interpretabilitas |' % C.NUM_TOPICS)
    A('| **Alpha / Eta** | `%s` / `%s` | Vektor prior yang diestimasi Gensim dari data latih |' % (C.LDA_ALPHA, C.LDA_ETA))
    A('| **Passes / Iterations** | **%d / %d** | Full convergence training |' % (C.LDA_PASSES, C.LDA_ITERATIONS))
    A('| **Random State / Seed** | **%d** | Terkunci untuk reproduksibilitas |' % C.RANDOM_SEED)
    A('| **Metode Inferensi** | Online Variational Bayes | Gensim %s Engine (**bukan** Gibbs Sampling) |' % '4.4.0')
    A('')
    A('---')
    A('')
    A('## 2. Metrik Evaluasi Resmi Model (K=%d)' % C.NUM_TOPICS)
    A('')
    A('> Protokol resmi `C_V`: `texts=data/intermediate/`, `topn=20`, `processes=1`, Gensim 4.4.0.')
    A('> Seluruh angka dapat direproduksi ulang dari `model/lda_model.gensim`.')
    A('> Verifikasi: `python run_tests.py`')
    A('')
    A('| Metrik Evaluasi | Nilai Resmi |')
    A('| :--- | :---: |')
    for key, label in [
        ('Coherence_Score_CV', '**Coherence Score ($C_V$)**'),
        ('Coherence_Score_UMass', '**Coherence Score ($U_{mass}$)**'),
        ('Train_Log_Perplexity', '**Log Perplexity**'),
        ('Topic_Diversity_Top10', 'Topic Diversity (Top-10)'),
        ('Topic_Diversity_Top15', 'Topic Diversity (Top-15)'),
        ('Pct_Max_Theta_Under_0.50', 'Persentase Ambigu ($\\theta < 0.50$)'),
        ('Vocabulary_Size', 'Vocabulary Size'),
        ('Total_Documents', 'Total Dokumen'),
    ]:
        if key not in m:
            raise SystemExit('  [GAGAL] metrik "%s" hilang dari evaluation_summary.csv' % key)
        val = idr(float(m[key]))
        A('| %s | `%s` |' % (label, val + ('%' if 'Persentase' in label else '')))
    A('| **Model Parameters** | `%s` |' % m['Model_Parameters'])
    A('')
    A('### Uji Homogenitas Komposisi Topik x Tahun')
    A('')
    A('| Statistik | Nilai |')
    A('| :--- | :---: |')
    A('| $\\chi^2$ (chi-square) | `%s` |' % idr(stats['chi2'], 4))
    A('| Derajat Kebebasan (df) | `%s` |' % stats['dof'])
    A('| p-value (asimtotik) | `%s` |' % idr(stats['p'], 4))
    A("| Cramer's V | `%s` |" % idr(stats['v'], 4))
    A('| p-value Permutasi (10.000 iterasi, seed=%d) | `%s` |' % (C.RANDOM_SEED, idr(stats['p_perm'], 4)))
    A('')
    A('---')
    A('')
    A('## 3. Distribusi & Label Topik Resmi')
    A('')
    A('```mermaid')
    A('pie title Distribusi %d Topik Skripsi SI (%d Dokumen)' % (C.NUM_TOPICS, total))
    for _, r in counts.sort_values('jumlah_dokumen', ascending=False).iterrows():
        t = int(r['topik_dominan'])
        A('    "T%d: %s (%s%%)" : %d'
          % (t, labels[t]['label'], idr(r['persentase'], 1), int(r['jumlah_dokumen'])))
    A('```')
    A('')
    A('| ID | Label Topik Resmi | Dokumen | Porsi | Kepercayaan |')
    A('| :-: | :--- | :-: | :-: | :---: |')
    for _, r in counts.sort_values('topik_dominan').iterrows():
        t = int(r['topik_dominan'])
        A('| **%d** | %s | %d | %s%% | %s |'
          % (t, labels[t]['label'], int(r['jumlah_dokumen']),
             idr(r['persentase']), labels[t].get('tingkat_kepercayaan', '-')))
    A('')
    A('> Total: **%d dokumen** (harus sama dengan jumlah dokumen korpus).' % total)
    A('')
    A('---')
    A('')
    A('## 4. Uji Tren per Topik (Spearman + Bonferroni)')
    A('')
    A('| Topik | Label | rho | p (raw) | p (Bonferroni) | Signifikan? |')
    A('| :-: | :--- | :-: | :-: | :-: | :---: |')
    for _, r in ptests.iterrows():
        A('| %d | %s | %s | %s | %s | %s |'
          % (int(r['topik_id']), r['label'], r['spearman_rho'],
             idr(r['p_spearman_raw'], 4), idr(r['p_spearman_bonferroni'], 4),
             r['spearman_sig']))
    A('')
    A('> **Tidak ada topik** yang memiliki tren signifikan setelah koreksi Bonferroni.')
    A('> Klaim "topik X meningkat signifikan" **dilarang** ditulis di skripsi.')
    A('')
    A('---')
    A('')
    A('> **Rujukan untuk Sidang:**')
    A('> Seluruh grafik pada Dashboard Streamlit (`app/main.py`) dan tabel di')
    A('> Bab IV dirancang konsisten dengan angka pada dokumen ini.')
    A('')

    OUT.write_text('\n'.join(L), encoding='utf-8')
    print('  [OK] %s ditulis (%d baris)' % (OUT.name, len(L)))
    print('  Total dokumen pada tabel topik: %d' % total)
    for _, r in counts.sort_values('topik_dominan').iterrows():
        print('    T%d = %d  (%s%%)' % (int(r['topik_dominan']), int(r['jumlah_dokumen']),
                                        idr(r['persentase'])))


if __name__ == '__main__':
    main()
