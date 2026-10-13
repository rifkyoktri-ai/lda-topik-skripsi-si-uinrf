# -*- coding: utf-8 -*-
"""
tests/test_metrics_sync.py

Menjaga agar konstanta metrik resmi di src/config.py SELALU sama dengan
model/evaluation_metrics.csv.

Mengapa modul ini perlu:
    Nilai C_V = 0,4256 saat ini hidup di DUA tempat sekaligus:
      1. model/evaluation_metrics.csv   -> dibaca dashboard (artifacts.load_metrics)
      2. src/config.py COHERENCE_CV_OFFICIAL -> dipakai pipeline & notebook
    Keduanya sekarang cocok, tapi tidak ada yang memaksa mereka tetap sinkron.
    Bila model dievaluasi ulang, dashboard dan pipeline bisa menampilkan
    angka berbeda tanpa ada yang menyadari.

Modul ini menutup celah tersebut: selisih > 1e-4 langsung menggagalkan test.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config as C  # noqa: E402


def _baca_metrik():
    """Baca model/evaluation_metrics.csv menjadi dict {nama: nilai}."""
    import pandas as pd

    df = pd.read_csv(C.EVAL_METRICS_CSV, encoding='utf-8-sig')
    return {str(r['Metrik']).strip(): r['Nilai'] for _, r in df.iterrows()}


class TestKonfigurasiVsArtefak(unittest.TestCase):
    """Konstanta di config.py harus sama persis dengan artefak."""

    def setUp(self):
        self.m = _baca_metrik()

    def _cek(self, konstanta, nama_artefak):
        nilai = float(self.m[nama_artefak])
        self.assertAlmostEqual(
            konstanta, nilai, places=4,
            msg="config.py %s=%r tapi %s di artefak = %r"
                % (nama_artefak, konstanta, nama_artefak, nilai)
        )

    def test_coherence_cv_cocok(self):
        """C_V resmi: config.py vs model/evaluation_metrics.csv"""
        self._cek(C.COHERENCE_CV_OFFICIAL, 'Coherence Score (C_V)')

    def test_coherence_umass_cocok(self):
        """U_Mass resmi: config.py vs artefak"""
        self._cek(C.COHERENCE_UMASS_OFFICIAL, 'Coherence Score (U_Mass)')

    def test_log_perplexity_cocok(self):
        """Log perplexity resmi: config.py vs artefak"""
        self._cek(C.LOG_PERPLEXITY_OFFICIAL, 'Train Log Perplexity')

    def test_konstanta_metric_lengkap(self):
        """Semua metrik yang di-hardcode di config.py punya padanan di artefak."""
        for nama in ['Coherence Score (C_V)', 'Coherence Score (U_Mass)',
                     'Train Log Perplexity']:
            self.assertIn(nama, self.m,
                          'Artefak %s tidak memuat metrik %r'
                          % (C.EVAL_METRICS_CSV.name, nama))

    def test_jumlah_topik_konsisten(self):
        """Jumlah topik di artefak harus sama dengan NUM_TOPICS di config."""
        self.assertEqual(int(float(self.m['Jumlah Topik (K)'])),
                         C.NUM_TOPICS)

    def test_ukuran_kosakata_konsisten(self):
        """Vocabulary Size di artefak harus cocok dengan config."""
        self.assertEqual(int(float(self.m['Vocabulary Size'])),
                         C.VOCAB_SIZE if hasattr(C, 'VOCAB_SIZE') else 1849)


if __name__ == '__main__':
    print("\nMenjalankan test sinkronisasi metrik...\n")
    unittest.main(verbosity=2)