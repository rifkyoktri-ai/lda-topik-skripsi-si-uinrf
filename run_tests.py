"""
run_tests.py — Test Runner & Verifikasi Konsistensi Data

Menjalankan 5 unit test konsistensi data yang menjadi bukti verifikasi pada
Bab III (fase Evaluation) dan Bab IV (subbab Evaluasi) skripsi, ditambah
validasi fingerprint label terhadap model LDA.

Jalankan:  python run_tests.py
"""
import sys
import unittest
import ast
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import pandas as pd

from src import config as C
from src.preprocessing import bersihkan, run_preprocessing


class TestPreprocessPipeline(unittest.TestCase):
    """Kelompok 1-2: preprocessing deterministik & bebas artefak newline."""

    def test_1_bersihkan_newline_and_escape_sequences(self):
        """
        Test 1: bersihkan() harus menangani \\n, \\\\n, \\r\\n, \\t, dan spasi ganda
        tanpa menghasilkan artefak 'nmetode' atau 'nkata'.
        """
        sample_input = "Sistem \\r\\ninformasi \\nmetode \\t perancangan \r\n kata\t kunci."
        cleaned = bersihkan(sample_input)

        words = cleaned.split()
        self.assertNotIn("nmetode", words)
        self.assertNotIn("nkata", words)
        for expected in ("metode", "kata", "informasi", "perancangan", "kunci"):
            self.assertIn(expected, words)

    def test_2_preprocessing_determinism(self):
        """
        Test 2: preprocessing bersifat deterministik (dua kali eksekusi
        menghasilkan output identik).
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            dir1 = Path(tmp_dir) / "run1"
            dir2 = Path(tmp_dir) / "run2"

            run_preprocessing(use_tfidf=False,
                              min_ngram_count=C.MIN_NGRAM_COUNT,
                              output_dir=str(dir1))
            run_preprocessing(use_tfidf=False,
                              min_ngram_count=C.MIN_NGRAM_COUNT,
                              output_dir=str(dir2))

            df1 = pd.read_csv(dir1 / "dataset_preprocessed.csv")
            df2 = pd.read_csv(dir2 / "dataset_preprocessed.csv")

            pd.testing.assert_frame_equal(df1, df2)


class TestCorpusIntegrity(unittest.TestCase):
    """Kelompok 3-5: integritas korpus & konsistensi label terhadap model."""

    def test_3_data_lineage_counts(self):
        """
        Test 3: jumlah dokumen pada setiap tahap lineage harus
        312 (raw) -> 309 (processed) -> 285 (preprocessed).
        """
        raw = pd.read_csv(C.SCRAPED_CSV)
        proc = pd.read_csv(C.PIPELINE_CSV)
        prep = pd.read_csv(C.PREPROCESSED_CSV)

        self.assertEqual(len(raw), 312, f"raw: expected 312, got {len(raw)}")
        self.assertEqual(len(proc), 309, f"processed: expected 309, got {len(proc)}")
        self.assertEqual(len(prep), 285, f"preprocessed: expected 285, got {len(prep)}")

    def test_4_document_id_uniqueness(self):
        """
        Test 4: ID dokumen unik dan sebaran per tahun sesuai GROUND_TRUTH.
        """
        prep = pd.read_csv(C.PREPROCESSED_CSV)

        self.assertTrue(prep["ID"].is_unique, "Duplikat ID terdeteksi")
        self.assertEqual(len(prep), 285)

        expected_years = {2021: 29, 2022: 78, 2023: 46, 2024: 47, 2025: 85}
        actual = prep.groupby("Tahun").size().to_dict()
        self.assertEqual(actual, expected_years, f"Sebaran tahun tidak sesuai: {actual}")

    def test_5_topic_distribution_consistency(self):
        """
        Test 5: jumlah dokumen per topik pada artefak final harus konsisten
        dengan topic_counts_final.csv yang dipakai dashboard (total 285).
        """
        counts = pd.read_csv(C.TOPIC_COUNTS_FINAL)
        dist = pd.read_csv(C.TOPIC_DIST_FINAL)

        self.assertEqual(int(counts["jumlah_dokumen"].sum()), 285)
        self.assertEqual(len(dist), 285)
        self.assertAlmostEqual(float(counts["persentase"].sum()), 100.0, delta=0.05)

        for row in counts.itertuples():
            self.assertEqual(
                int((dist["topik_dominan"] == row.topik_dominan).sum()),
                int(row.jumlah_dokumen),
                f"Topik {row.topik_dominan}: jumlah dokumen tidak konsisten",
            )


def _run_label_integrity():
    """Verifikasi tambahan: fingerprint label vs model LDA."""
    from src.topic_labels import validate_labels
    return validate_labels(str(C.LDA_MODEL), str(C.LABELS_JSON))


def _run_metrics_sync():
    """Verifikasi tambahan: konstanta metrik di config.py vs artefak resmi."""
    from tests.test_metrics_sync import TestKonfigurasiVsArtefak
    suite = unittest.TestLoader().loadTestsFromTestCase(TestKonfigurasiVsArtefak)
    result = unittest.TextTestRunner(verbosity=0).run(suite)
    return result.wasSuccessful(), result.testsRun


def _run_structure():
    """Verifikasi tambahan: bentuk repo (path, git tracking, folder kosong)."""
    from tests import test_structure as ts
    loader2 = unittest.TestLoader()
    suite2 = unittest.TestSuite()
    for name in dir(ts):
        obj = getattr(ts, name)
        if isinstance(obj, type) and issubclass(obj, unittest.TestCase):
            suite2.addTests(loader2.loadTestsFromTestCase(obj))
    result = unittest.TextTestRunner(verbosity=0).run(suite2)
    return result.wasSuccessful(), result.testsRun


def _run_smoke():
    """Verifikasi tambahan: semua modul src/ benar-benar bisa dijalankan."""
    from tests import test_smoke as ts
    loader3 = unittest.TestLoader()
    suite3 = unittest.TestSuite()
    for name in dir(ts):
        obj = getattr(ts, name)
        if isinstance(obj, type) and issubclass(obj, unittest.TestCase):
            suite3.addTests(loader3.loadTestsFromTestCase(obj))
    result = unittest.TextTestRunner(verbosity=0).run(suite3)
    return result.wasSuccessful(), result.testsRun


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestPreprocessPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestCorpusIntegrity))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 62)
    label_ok = True
    try:
        _run_label_integrity()
        print("[OK] Fingerprint SHA-256 labels.json cocok dengan model LDA")
    except Exception as exc:
        label_ok = False
        print(f"[GAGAL] Validasi fingerprint: {exc}")
    print("=" * 62)

    sync_ok, sync_n = _run_metrics_sync()
    if sync_ok:
        print(f"[OK] {sync_n} metrik resmi di config.py cocok dengan artefak")
    else:
        print("[GAGAL] Konstanta metrik config.py BERBEDA dari artefak resmi")
    print("=" * 62)

    struct_ok, struct_n = _run_structure()
    if struct_ok:
        print(f"[OK] {struct_n} pemeriksaan struktur repo lolos")
    else:
        print("[GAGAL] Struktur repo bermasalah - jalankan: python run.py structure")
    print("=" * 62)

    smoke_ok, smoke_n = _run_smoke()
    if smoke_ok:
        print(f"[OK] {smoke_n} uji asap: semua modul bisa dijalankan")
    else:
        print("[GAGAL] Ada modul yang tidak jalan - jalankan: "
              "python run.py smoke")
    print("=" * 62)

    total = result.testsRun + sync_n + struct_n + smoke_n
    if (not result.wasSuccessful() or not label_ok or not sync_ok
            or not struct_ok or not smoke_ok):
        sys.exit(1)
    print(f"\nSELURUH {total} UNIT TEST LULUS.")