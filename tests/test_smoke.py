# -*- coding: utf-8 -*-
"""
tests/test_smoke.py - Uji asap: semua modul src/ benar-benar bisa jalan.

KENAPA MODUL INI PERLU
    Kerusakan yang paling merusak di repo ini adalah "skrip penghasil tidak
    bisa dijalankan lagi". Contoh nyata: src/make_labels.py - sumber
    kebenaran 8 label topik - sempat rusak karena masih membaca path
    'model/lda_model.gensim' setelah folder dipindah ke outputs/.

    Test lain TIDAK menangkap itu, karena test hanya memverifikasi bahwa
    labels.json KONSISTEN dengan model. Tidak ada yang memverifikasi bahwa
    SKRIP YANG MEMBUATNYA masih jalan.

    Modul ini menutup celah itu.

JANJI TANPA EFEK SAMPING
    Tidak ada berkas di dalam repo yang disentuh. Semua keluaran ditulis ke
    folder sementara (tempfile) yang dihapus setelah selesai.

    Pelajaran: sebuah tes pernah menghapus artefak asli karena berhenti
    sebelum langkah pemulihan. Karena itu di sini dipakai tempdir, dan
    tidak pernah menimpa outputs/model/labels.json.

Jalankan:
    python tests/test_smoke.py
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config as C  # noqa: E402


MODUL = [p.stem for p in sorted((ROOT / "src").glob("*.py"))
         if p.name != "__init__.py"]


def jalankan(args, **kw):
    return subprocess.run([sys.executable] + args, cwd=str(ROOT),
                          capture_output=True, text=True, timeout=600, **kw)


class TestModulDapatDiimport(unittest.TestCase):
    """Semua modul harus bisa diimpor tanpa efek samping."""

    def test_semua_modul_import(self):
        gagal = []
        for m in MODUL:
            r = jalankan(["-c", "import sys; sys.path.insert(0,'.'); import src.%s" % m])
            if r.returncode != 0:
                baris = [l for l in r.stderr.strip().split("\n")
                         if "Error" in l or "error" in l]
                gagal.append("%s -> %s" % (m, baris[-1][:70] if baris else "?"))
        self.assertFalse(gagal, "Modul gagal di-import:\n  " + "\n  ".join(gagal))

    def test_tidak_ada_path_relatif(self):
        """
        Tidak boleh ada modul yang menulis ke path relatif.

        Dulu src/evaluation_lda.py menulis ke Path("experiments") - path
        relatif yang hanya benar bila CWD = root repo, dan melanggar
        prinsip config.py sebagai satu sumber path.

        Diperiksa lewat AST, bukan pencarian teks, supaya penyebutan di
        dalam docstring atau komentar tidak dianggap pelanggaran.
        """
        import ast

        # Hanya pemanggilan yang benar-benar menyentuh filesystem
        FUNGSI_FS = {"open", "mkdir", "to_csv", "to_excel", "to_json",
                     "savefig", "save", "read_csv", "read_text",
                     "write_text", "write", "load", "unlink",
                     "from_csv", "copy", "rmtree", "makedirs"}

        pelaku = []
        for f in sorted((ROOT / "src").glob("*.py")):
            if f.name == "config.py":
                continue      # config.py memang mendefinisikan titik ke REPO
            try:
                pohon = ast.parse(f.read_text(encoding="utf-8"))
            except SyntaxError as e:
                pelaku.append("%s: syntax error %s" % (f.name, e))
                continue

            for node in ast.walk(pohon):
                if not isinstance(node, ast.Call):
                    continue
                fname = ""
                if isinstance(node.func, ast.Name):
                    fname = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    fname = node.func.attr
                if fname not in FUNGSI_FS:
                    continue

                for arg in list(node.args) + [k.value for k in node.keywords]:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        s = arg.value
                        if not ("/" in s or "\\" in s):
                            continue
                        if len(s) > 120 or s.startswith(("http", "?", "<")):
                            continue
                        if s in ("rb", "wb", "r", "w", "utf-8"):
                            continue
                        pelaku.append(
                            "%s:%d  %s(%r)" % (f.name, node.lineno, fname, s[:56]))
        self.assertFalse(
            pelaku,
            "Path relatif di luar config.py:\n  " + "\n  ".join(sorted(set(pelaku))))


class TestMakeLabels(unittest.TestCase):
    """
    Uji paling penting: skrip pembuat labels.json harus bisa dijalankan
    dan hasilnya harus sama persis dengan berkas yang tersimpan.
    """

    def test_make_labels_bisa_dijalankan(self):
        with tempfile.TemporaryDirectory() as tmp:
            keluar = Path(tmp) / "labels_uji.json"
            r = jalankan(["-m", "src.make_labels", "--out", str(keluar),
                          "--quiet"])
            self.assertEqual(
                r.returncode, 0,
                "src.make_labels.py gagal jalan:\n%s" % r.stderr[-500:])
            self.assertTrue(keluar.exists(), "File keluaran tidak dibuat")

    def test_round_trip_labels_identik(self):
        """
        Hasil make_labels.py HARUS identik dengan outputs/model/labels.json.

        Kalau berbeda, berarti labels.json di repo sudah melenceng dari
        sumber kebenarannya.
        """
        with tempfile.TemporaryDirectory() as tmp:
            keluar = Path(tmp) / "labels_uji.json"
            r = jalankan(["-m", "src.make_labels", "--out", str(keluar),
                          "--quiet"])
            self.assertEqual(r.returncode, 0, r.stderr[-400:])

            hasil = json.loads(keluar.read_text(encoding="utf-8"))
            tersimpan = json.loads(C.LABELS_JSON.read_text(encoding="utf-8"))

            self.assertEqual(len(hasil), len(tersimpan),
                             "Jumlah entri label berbeda")
            for a, b in zip(hasil, tersimpan):
                for k in ("topic_id", "label", "deskripsi",
                          "top_words_fingerprint", "tingkat_kepercayaan",
                          "status"):
                    self.assertEqual(
                        a[k], b[k],
                        "Topik %d field %r berbeda:\n  hasil    : %r\n"
                        "  tersimpan: %r" % (a["topic_id"], k,
                                              str(a[k])[:60], str(b[k])[:60]))

    def test_fingerprint_pakai_satu_sumber(self):
        """
        make_labels.py tidak boleh menghitung SHA-256 sendiri.

        Sidik jari harus dihitung satu kali, di topic_labels.py, supaya
        tidak mungkin dua implementasi berbeda menghasilkan hash berbeda.
        """
        src = (ROOT / "src" / "make_labels.py").read_text(encoding="utf-8")
        kode = "\n".join(l.split("#")[0] for l in src.split("\n"))
        self.assertIn("compute_topic_fingerprint", kode,
                      "make_labels.py harus memakai compute_topic_fingerprint()")
        self.assertNotIn("hashlib.sha256", kode,
                         "make_labels.py tidak boleh menghitung hash sendiri")

    def test_tidak_menimpa_labels_json(self):
        """Argumen --out harus benar-benar dipakai (perlindungan artefak)."""
        sebelum = C.LABELS_JSON.read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            jalankan(["-m", "src.make_labels", "--out",
                      str(Path(tmp) / "x.json"), "--quiet"])
        sesudah = C.LABELS_JSON.read_bytes()
        self.assertEqual(
            sebelum, sesudah,
            "labels.json berubah padahal --out dipakai. Ini berbahaya - "
            "artifak asli akan tertimpa saat pengujian.")


class TestValidasiLabel(unittest.TestCase):
    """Fingerprint di labels.json harus cocok dengan model."""

    def test_validate_labels_lolos(self):
        from src.topic_labels import validate_labels
        self.assertTrue(validate_labels(str(C.LDA_MODEL), str(C.LABELS_JSON)))

    def test_delapan_label_disetujui(self):
        data = json.loads(C.LABELS_JSON.read_text(encoding="utf-8"))
        self.assertEqual(len(data), C.NUM_TOPICS)
        for d in data:
            self.assertEqual(d["status"], "disetujui",
                             "Topik %d belum disetujui" % d["topic_id"])
            self.assertIn(d["tingkat_kepercayaan"],
                          ("Tinggi", "Sedang", "Rendah"))


class TestExperimen(unittest.TestCase):
    """Folder eksperimen kini lewat config, bukan path relatif."""

    def test_experiments_dir_lewat_config(self):
        self.assertTrue(C.EXPERIMENTS_DIR.is_absolute(),
                        "EXPERIMENTS_DIR harus path absolut")
        self.assertTrue(str(C.EXPERIMENTS_DIR).startswith(str(ROOT)),
                        "EXPERIMENTS_DIR harus di dalam repo")
        self.assertEqual(C.EXPERIMENTS_DIR.parent, C.OUTPUTS_DIR)

    def test_experiments_tidak_di_gitignore(self):
        r = subprocess.run(["git", "check-ignore", "-q", "--no-index",
                            str(C.EXPERIMENTS_DIR.relative_to(ROOT))],
                           cwd=str(ROOT), capture_output=True)
        self.assertNotEqual(
            r.returncode, 0,
            "outputs/experiments/ masih di-gitignore; keluarannya tidak "
            "akan pernah ikut ter-commit")


if __name__ == "__main__":
    print("\nUji asap: semua modul harus benar-benar bisa jalan...\n")
    unittest.main(verbosity=2)