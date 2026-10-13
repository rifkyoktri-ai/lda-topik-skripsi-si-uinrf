# -*- coding: utf-8 -*-
"""
tests/test_structure.py - Validasi struktur repositori.

Test ini BERBEDA dari test konsistensi data. Yang diuji di sini adalah
"bentuk" repo: apakah semua path yang ditulis di kode benar-benar ada,
apakah file model masih ter-track git, apakah ada folder kosong.

Kenapa test ini perlu:
    1. PATah .gitignore. Repo ini mengabaikan *.gensim / *.npy lalu
       mengecualikannya per path. Kalau folder hasil berpindah dan negate
       di .gitignore tidak ikut diperbarui, model LDA LEPAS dari git
       secara senyap - semua test lain tetap hijau, tidak ada yang
       memberi tahu, tapi skripsi tidak bisa direproduksi.
    2. Path yang ditulis manual di app/ bisa salah tanpa errorifu版面.
    3. Folder kosong dan file mati menumpuk diam-diam.

Jalankan:
    python tests/test_structure.py
    python run.py structure
"""
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config as C  # noqa: E402


def git(*args):
    """Jalankan perintah git, kembalikan stdout atau None bila gagal."""
    try:
        r = subprocess.run(["git"] + list(args), cwd=str(ROOT),
                           capture_output=True, text=True, timeout=60)
        return r.stdout if r.returncode == 0 else None
    except Exception:
        return None


class TestPathKonfigurasi(unittest.TestCase):
    """Semua path yang ditulis di src/config.py harus benar-benar ada."""

    # Path yang dibuat saat runtime, jadi tidak harus sudah ada
    DINAMIS = {"LOGS_DIR"}

    def test_semua_path_konfigurasi_ada(self):
        hilang = []
        for name in dir(C):
            if not name.isupper():
                continue
            v = getattr(C, name)
            if isinstance(v, Path) and name not in self.DINAMIS:
                if not v.exists():
                    hilang.append("%s -> %s" % (
                        name, v.relative_to(ROOT) if ROOT in v.parents else v))
        self.assertFalse(
            hilang,
            "Path di src/config.py tidak ada:\n  " + "\n  ".join(hilang))

    def test_folder_ada(self):
        for folder in ("data", "src", "notebooks", "outputs", "app",
                       "tests", "docs", "thesis"):
            self.assertTrue((ROOT / folder).is_dir(),
                            "Folder wajib hilang: %s/" % folder)

    def test_subfolder_outputs_ada(self):
        for sub in ("model", "topics", "forecast", "audit"):
            self.assertTrue((C.OUTPUTS_DIR / sub).is_dir(),
                            "outputs/%s/ hilang" % sub)


class TestArtefakTerTrack(unittest.TestCase):
    """Jaring pengaman jebakan .gitignore."""

    MODEL_BESAR = [
        "outputs/model/lda_model.gensim",
        "outputs/model/lda_model.gensim.state",
        "outputs/model/lda_model.gensim.id2word",
        "outputs/model/lda_model.gensim.expElogbeta.npy",
        "outputs/topics/corpus_corrected.pkl",
        "data/intermediate/corpus.pkl",
    ]

    def test_artefak_besar_tertrack_git(self):
        """
        File binary besar WAJIB ter-track git.

        Kalau test ini gagal setelah pemindahan folder, hampir pasti
        negate di .gitignore tidak ikut diperbarui.
        """
        tracked = git("ls-files")
        self.assertIsNotNone(tracked, "git tidak bisa dijalankan")
        hilang = [f for f in self.MODEL_BESAR
                  if not (ROOT / f).exists() or f not in tracked]
        self.assertFalse(
            hilang,
            "Artefak berikut tidak ter-track git (model akan hilang saat clone):\n  "
            + "\n  ".join(hilang))

    def test_artefak_besar_ada_di_disk(self):
        hilang = [f for f in self.MODEL_BESAR if not (ROOT / f).exists()]
        self.assertFalse(hilang,
                         "Artefak hilang dari disk:\n  " + "\n  ".join(hilang))

    def test_outputs_tidak_di_gitignore(self):
        """Folder outputs/ sengaja di-commit; pastikan tidak ter-ignore."""
        for f in self.MODEL_BESAR[:4]:
            r = subprocess.run(["git", "check-ignore", "-q", "--no-index", f],
                               cwd=str(ROOT), capture_output=True)
            self.assertNotEqual(
                r.returncode, 0,
                "%s ter-ignore! Perbarui negate di .gitignore" % f)


class TestKebersihanRepo(unittest.TestCase):
    """Tidak ada folder kosong, file mati, atau sampah yang ter-commit."""

    FOLDER_BOLEH_KOSONG = {".git", ".venv", "logs", "__pycache__",
                           ".ipynb_checkpoints", ".pytest_cache", ".streamlit"}
    FILE_MATI = ["config.yaml", "build_notebooks.py", "k_sweep.csv"]

    def test_tidak_ada_folder_kosong(self):
        kosong = []
        for dp, dn, fn in os.walk(ROOT):
            d = Path(dp)
            if any(p in self.FOLDER_BOLEH_KOSONG or p.startswith('.')
                   for p in d.relative_to(ROOT).parts[:-1]):
                continue
            if d.name in self.FOLDER_BOLEH_KOSONG:
                continue
            isi = [f for f in d.iterdir() if f.name not in self.FOLDER_BOLEH_KOSONG]
            if not isi:
                kosong.append(str(d.relative_to(ROOT)) + "/")
        self.assertFalse(kosong, "Folder kosong:\n  " + "\n  ".join(kosong))

    def test_file_mati_tidak_ada(self):
        ada = [f for f in self.FILE_MATI if (ROOT / f).exists()]
        self.assertFalse(ada, "File mati masih ada: " + ", ".join(ada))

    def test_no_pycache_tercommit(self):
        tracked = git("ls-files") or ""
        sample = [f for f in tracked.splitlines() if "__pycache__" in f]
        self.assertFalse(sample, "Ada __pycache__ yang ter-commit")

    def test_docs_murni_markdown(self):
        """docs/ hanya dokumentasi - tidak boleh ada .csv/.png di dalamnya."""
        jahat = []
        docs = ROOT / "docs"
        if docs.is_dir():
            for p in docs.rglob("*"):
                if p.is_file() and p.suffix.lower() not in (".md", ".txt"):
                    jahat.append(str(p.relative_to(ROOT)))
        self.assertFalse(jahat, "docs/ harus murni markdown. Ditemukan:\n  "
                         + "\n  ".join(jahat))

    def test_naskah_tersimpan_di_thesis(self):
        self.assertTrue((ROOT / "thesis").is_dir(), "Folder thesis/ hilang")
        docx = list((ROOT / "thesis").glob("*.docx"))
        self.assertTrue(docx, "Naskah .docx tidak ada di thesis/")


class TestEntrypoint(unittest.TestCase):
    """Pintu masuk tunggal untuk pengguna awam."""

    def test_run_py_ada(self):
        self.assertTrue((ROOT / "run.py").exists(),
                        "run.py (pintu masuk utama) tidak ada")

    def test_pintu_masuk_terdaftar(self):
        txt = (ROOT / "run.py").read_text(encoding="utf-8")
        for cmd in ("test", "notebook", "audit", "dashboard", "all"):
            self.assertIn('"%s"' % cmd, txt,
                          "perintah %r tidak terdaftar di run.py" % cmd)


if __name__ == "__main__":
    print("\nValidasi struktur repositori...\n")
    unittest.main(verbosity=2)