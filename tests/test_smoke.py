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
import ast
import builtins
import json
import re
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

HALAMAN = sorted((ROOT / "app").rglob("*.py"))


def jalankan(args, **kw):
    return subprocess.run([sys.executable] + args, cwd=str(ROOT),
                          capture_output=True, text=True, timeout=600, **kw)


class TestHalamanDashboard(unittest.TestCase):
    """
    Halaman dashboard harus bisa dieksekusi tanpa NameError.

    INSIDEN
        app/pages/7_Metodologi.py sempat gagal total saat dibuka:
        NameError: name 'N_DOC_TAMPIL' is not defined.

        Penyebabnya bukan salah ketik. Saat refactor menghapus angka
        hard-code, definisi N_DOC_TAMPIL dan RENTANG_TAHUN dipindah ke
        BAWAH pemakaiannya - baris 41 dan 74 memakai, baris 127 baru
        mendefinisikan. Python membaca file dari atas ke bawah, jadi
        halaman langsung crash.

    MENGAPA PERLU DIJAGA
        Test lain hanya memeriksa isi artefak. Tidak ada yang memeriksa
        bahwa kode halaman dashboard masih bisa DIJALANKAN. Padahal
        halaman itu yang dilihat penguji.

    CARA KERJA
        Analisis AST: cari nama variabel yang dipakai pada baris N
        tetapi definisinya baru muncul pada baris M > N. Nama yang
        hanya terikat oleh parameter fungsi, target loop, atau baris
        yang sama diabaikan.
    """

    # Nama yang pasti sudah terikat: parameter fungsi, target loop,
    # dan target list comprehension. Semua muncul di baris yang sama
    # dengan pemakaiannya atau di baris sebelumnya.
    DILEWARKAN = {
        "m", "r", "fig", "df", "x", "i", "p", "t", "k", "v", "o", "col",
        "tid", "item", "word", "words", "probs", "color", "label",
        "labels_dict", "topic_id", "filtered", "sweep", "idx", "y",
        "hex_str", "table_df", "df_counts", "df_by_year", "colors",
        "selected_topics", "_base_layout", "_topic_color",
    }

    @staticmethod
    def _nama_terikat(pohon, batas_baris=None):
        """Kumpulan nama yang terikat. Tanpa batas = seluruh file."""
        hasil = set()
        for n in ast.walk(pohon):
            baris = getattr(n, "lineno", None)
            if baris is None:
                continue
            if batas_baris is not None and baris > batas_baris:
                continue
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                for a in n.names:
                    hasil.add((a.asname or a.name).split(".")[0])
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                                ast.ClassDef)):
                hasil.add(n.name)
            elif isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
                hasil.add(n.id)
            elif isinstance(n, ast.ExceptHandler) and n.name:
                hasil.add(n.name)
            elif isinstance(n, (ast.Global, ast.Nonlocal)):
                hasil.update(n.names)
        return hasil

    def test_tidak_ada_variabel_dipakai_sebelum_didefinisikan(self):
        import builtins

        bawaan = set(dir(builtins)) | {"__file__", "__name__", "__doc__"}
        salah = []

        for f in HALAMAN:
            pohon = ast.parse(f.read_text(encoding="utf-8"))
            semua = self._nama_terikat(pohon)
            for n in ast.walk(pohon):
                if not (isinstance(n, ast.Name)
                        and isinstance(n.ctx, ast.Load)):
                    continue
                if n.id in bawaan or n.id not in semua:
                    continue
                if n.id in self.DILEWARKAN:
                    continue
                sudah = self._nama_terikat(pohon, n.lineno)
                if n.id not in sudah:
                    salah.append("%s:%d  %s" % (f.name, n.lineno, n.id))

        self.assertFalse(
            salah,
            "Variabel dipakai SEBELUM didefinisikan (NameError saat "
            "dibuka di dashboard):\n  " + "\n  ".join(salah))

    def test_semua_halaman_tanpa_error_sintaks(self):
        rusak = []
        for f in HALAMAN:
            try:
                ast.parse(f.read_text(encoding="utf-8"))
            except SyntaxError as e:
                rusak.append("%s: %s" % (f.name, e))
        self.assertFalse(rusak, "Sintaks rusak:\n  " + "\n  ".join(rusak))


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


class TestKonsistensiNotebook(unittest.TestCase):
    """
    Notebook tidak boleh menyebut angka yang bertentangan dengan artefak.

    Kasus nyata: Notebook 04b masih menulis "Total: 7 x 4 x 3 = 84
    konfigurasi" dan judul grafik 'Grid Search 84 Konfigurasi' yang
    ter-hardcode - padahal hyperparameter_results.csv sudah berisi 140
    baris. Salahnya tertanam di notebook DAN tidak hilang meski
    dijalankan ulang, karena angkanya ditulis manual, bukan diturunkan
    dari data.
    """

    def _teks_notebook(self, nama):
        import glob as _glob
        p = ROOT / "notebooks" / nama
        self.assertTrue(p.exists(), "Notebook %s tidak ada" % nama)
        nb = json.loads(p.read_text(encoding="utf-8"))
        sumber = []
        for c in nb["cells"]:
            s = c.get("source", [])
            sumber.append(s if isinstance(s, str) else "".join(s))
        return "\n".join(sumber)

    def test_jumlah_konfigurasi_grid_cocok(self):
        """
        Notebook 04b tidak boleh menyebut jumlah konfigurasi yang berbeda
        dengan isi hyperparameter_results.csv.
        """
        import pandas as pd

        n_asli = len(pd.read_csv(C.HYPERPARAMS_CSV, encoding="utf-8-sig"))
        teks = self._teks_notebook("04b_grid_search.ipynb")

        # Semua sebutan "<angka> konfigurasi" harus sama dengan n_asli.
        # Dikecualikan "N Konfigurasi Teratas" - itu tabel peringkat atas,
        # bukan jumlah konfigurasi grid.
        ditemukan = []
        for m in re.finditer(r"(\d{1,4})\s+konfigurasi(.{0,20})", teks, re.I):
            if re.match(r"\s*teratas", m.group(2), re.I):
                continue
            ditemukan.append(int(m.group(1)))
        salah = [n for n in ditemukan if n != n_asli]
        self.assertFalse(
            salah,
            "Notebook 04b menyebut %s konfigurasi, sedangkan %s berisi "
            "%d baris. Jalankan ulang: python run_notebook.py "
            "notebooks/04b_grid_search.ipynb"
            % (sorted(set(salah)), C.HYPERPARAMS_CSV.name, n_asli))

    def test_judul_grafik_bukan_hardcode(self):
        """Angka pada judul grafik harus diturunkan dari data."""
        teks = self._teks_notebook("04b_grid_search.ipynb")
        self.assertNotRegex(
            teks, r"set_title\(['\"][^'\"]*\d{2,4}\s*Konfigurasi",
            "Judul grafik masih mengetik angka secara manual. "
            "Gunakan f-string, misalnya f'Grid Search {len(hp)} Konfigurasi'.")

    def test_tidak_ada_path_lama_di_notebook(self):
        """Tidak boleh ada path pra-restrukturisasi di notebook."""
        pola = re.compile(r"(?<![/\w])(_fix_final|_label_work|_stats_forecast)/"
                          r"|(?<![/\w])model/(lda_model|hyperparameter|"
                          r"topic_labels|evaluation_metrics|experiment_history)")
        salah = []
        for f in sorted((ROOT / "notebooks").glob("*.ipynb")):
            nb = json.loads(f.read_text(encoding="utf-8"))
            for c in nb["cells"]:
                s = c.get("source", [])
                teks = s if isinstance(s, str) else "".join(s)
                for m in pola.finditer(teks):
                    salah.append("%s: %s" % (f.name, m.group(0)))
        self.assertFalse(salah,
                         "Path lama masih ada di notebook:\n  "
                         + "\n  ".join(salah))


if __name__ == "__main__":
    print("\nUji asap: semua modul harus benar-benar bisa jalan...\n")
    unittest.main(verbosity=2)