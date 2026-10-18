# -*- coding: utf-8 -*-
"""
tests/test_thesis.py - Penjaga naskah skripsi.

INSIDEN YANG MENJADI ALASAN MODUL INI ADA
    Saat menghapus fitur validasi expert, saya menjalankan `git stash`
    untuk membandingkan perilaku kode sebelum dan sesudah. Naskah pada
    saat itu sedang dibuka di Word, sehingga `git stash pop` GAGAL
    sebagian: berkas naskah tertimpa sebagian dan isinya bercampur.

    Akibatnya berkas di disk mundur ke versi yang lebih lama:
        17 tabel, bukan 18
        15 media, bukan 11 (dua foto yang sudah dihapus muncul lagi)
        Gambar 2, 3, 4 kembali ke versi lama yang angkanya salah
        satu kemunculan 0,4256 hilang

    Tidak ada satu pun test yang menangkap itu. Test yang ada hanya
    memeriksa outputs/, src/, dan app/ - TIDAK PERNAH menyentuh naskah.
    Padahal naskah adalah hasil utama penelitian ini.

    Modul ini menutup celah itu.

APA YANG DIPERIKSA
    1. Naskah di disk harus sama dengan versi yang ter-commit. Ini
       menangkap perubahan yang tidak disengaja. Kalau memang mengubah
       naskah secara sadar, commit naskah itu lalu jalankan ulang.
    2. Naskah harus punya bentuk yang sudah diketahui: jumlah tabel,
       baris tabel, gambar, dan hyperlink.
    3. Naskah harus utuh sebagai berkas ZIP dan XML yang valid.
    4. Perbaikan teks yang sudah selesai harus masih ada.
    5. Berkas sementara dari Office tidak boleh tertinggal.

CATATAN PENTING
    Modul ini tidak melarang siapa pun mengubah naskah. Yang diperiksa
    adalah KESAMAAN dengan versi ter-commit, sehingga setiap perubahan
    harus disengaja.
"""
import hashlib
import re
import subprocess
import sys
import unittest
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NASKAH_REL = "thesis/Skripsi Rifky FIX.docx"
NASKAH = ROOT / NASKAH_REL

# Bentuk naskah yang harus bertahan. Angka diambil dari naskah versi
# bd5b8b9, setelah seluruh perbaikan gambar dan teks selesai.
HARAP_TABEL = 18
HARAP_BARIS_TABEL = 130
HARAP_GAMBAR = 11
HARAP_HYPERLINK = 85

# Berkas media yang TIDAK boleh ada: dua foto dokumentasi wawancara
# yang sudah dihapus karena sama sekali tidak punya keterangan.
TERLINDAR = {"image12.jpeg", "image13.jpeg"}

# Ukuran gambar yang harus cocok. Kalau berubah, berarti gambar di
# naskah bukan versi terbaru.
HARAP_MEDIA = {
    "image3.png": 159594,    # Gambar 2 - Alur CRISP-DM
    "image4.png": 100580,    # Gambar 3 - Fishbone
    "image5.png": 196497,    # Gambar 4 - Alur Tahapan
    "image10.png": 78819,    # Gambar 9 - Screenshot Metodologi
}

# Teks hasil perbaikan yang harus masih ada.
HARAP_TEKS = [
    ("Evaluasi Konsistensi Artefak", "sub-bagian Evaluasi"),
    ("ambang keandalan rasio 0,95", "penjelas selisih 5 dari 8 vs 2 dari 8"),
    ("deteksi berbasis rasio kata penanda", "keterbatasan bahasa Inggris"),
    ("Pemilihan prior Dirichlet", "justifikasi pemilihan alpha"),
    ("10,7533", "metrik U_Mass"),
    ("0,7125", "metrik Topic Diversity"),
    ("13,33", "persentase dokumen ambigu"),
]

# Teks versi lama yang tidak boleh muncul lagi.
DILARANG_TEKS = [
    ("7 dari 8", "angka WMA versi lama"),
    ("-7.4518", "separator desimal titik"),
]

PETUNJUK_SYNC = (
    "\n\nNaskah berbeda dari versi yang ter-commit.\n"
    "  Kalau ini memang disengaja:\n"
    '      git add "%s"\n'
    "      git commit -m \"docs: revisi naskah\"\n"
    "  Kalau ini TIDAK disengaja (naskah berubah sendiri):\n"
    '      git checkout HEAD -- "%s"'
    % (NASKAH_REL, NASKAH_REL)
)


def git_hash_blob(path_rel):
    """SHA-256 isi berkas pada HEAD, atau None bila tidak ter-track."""
    try:
        r = subprocess.run(["git", "cat-file", "blob", "HEAD:" + path_rel],
                           cwd=str(ROOT), capture_output=True, timeout=60)
        if r.returncode != 0:
            return None
        return hashlib.sha256(r.stdout).hexdigest()
    except Exception:
        return None


def muat_naskah():
    """Baca naskah, kembalikan (zip, xml, teks_polos)."""
    z = zipfile.ZipFile(NASKAH)
    xml = z.read("word/document.xml").decode("utf-8")
    polos = re.sub(r"<[^>]+>", "", re.sub(r"</w:p>", "\n", xml))
    polos = (polos.replace("&amp;", "&").replace("&lt;", "<")
             .replace("&gt;", ">"))
    return z, xml, polos


class TestNaskah(unittest.TestCase):
    """Naskah harus utuh, lengkap, dan sama dengan versi ter-commit."""

    @classmethod
    def setUpClass(cls):
        cls.z, cls.xml, cls.teks = muat_naskah()
        cls.media = {n.split("/")[-1] for n in cls.z.namelist()
                     if n.startswith("word/media/")}

    # ── 1. integritas berkas ────────────────────────────────────────────────
    def test_zip_naskah_utuh(self):
        rusak = self.z.testzip()
        self.assertIsNone(rusak,
                          "Entri zip rusak di naskah: %s" % rusak)

    def test_xml_naskah_valid(self):
        try:
            ET.fromstring(self.xml)
        except ET.ParseError as e:
            self.fail("document.xml bukan XML yang valid: %s" % e)

    # ── 2. struktur ──────────────────────────────────────────────────────────
    def test_jumlah_tabel(self):
        n = self.xml.count("<w:tbl>")
        self.assertEqual(n, HARAP_TABEL,
                         "Jumlah tabel berubah: %d (harus %d)%s"
                         % (n, HARAP_TABEL, PETUNJUK_SYNC))

    def test_jumlah_baris_tabel(self):
        n = len(re.findall(r"<w:tr(?:\s[^>]*?)?[ >/]", self.xml))
        self.assertEqual(n, HARAP_BARIS_TABEL,
                         "Jumlah baris tabel berubah: %d (harus %d)%s"
                         % (n, HARAP_BARIS_TABEL, PETUNJUK_SYNC))

    def test_jumlah_gambar(self):
        n = len(re.findall(r'r:embed="', self.xml))
        self.assertEqual(n, HARAP_GAMBAR,
                         "Jumlah gambar berubah: %d (harus %d)%s"
                         % (n, HARAP_GAMBAR, PETUNJUK_SYNC))

    def test_jumlah_hyperlink(self):
        n = len(re.findall(r"<w:hyperlink", self.xml))
        self.assertEqual(n, HARAP_HYPERLINK,
                         "Jumlah hyperlink berubah: %d (harus %d)%s"
                         % (n, HARAP_HYPERLINK, PETUNJUK_SYNC))

    def test_tidak_ada_media_yatim(self):
        """Jumlah berkas gambar harus sama dengan jumlah gambar terpakai."""
        n = len(re.findall(r'r:embed="', self.xml))
        self.assertEqual(len(self.media), n,
                         "Ada %d berkas gambar untuk %d gambar terpakai. "
                         "Bisa terjadi setelah penghapusan foto."
                         % (len(self.media), n))

    # ── 3. gambar yang harus sudah diperbarui ────────────────────────────────
    def test_gambar_adalah_versi_terbaru(self):
        salah = []
        for nama, ukuran in HARAP_MEDIA.items():
            if nama not in self.media:
                salah.append("%s hilang" % nama)
                continue
            n = len(self.z.read("word/media/" + nama))
            if n != ukuran:
                salah.append("%s = %d B, harus %d B (versi lama)"
                             % (nama, n, ukuran))
        self.assertFalse(salah,
                         "Gambar di naskah kembali ke versi lama:\n  "
                         + "\n  ".join(salah))

    def test_foto_tanpa_keterangan_tidak_ada(self):
        ada = sorted(self.media & TERLINDAR)
        self.assertFalse(ada,
                         "Foto dokumentasi tanpa keterangan muncul lagi: %s"
                         % ", ".join(ada))

    # ── 4. teks hasil perbaikan ──────────────────────────────────────────────
    def test_perbaikan_teks_masih_ada(self):
        hilang = ["%s (%s)" % (k, ket) for k, ket in HARAP_TEKS
                  if k not in self.teks]
        self.assertFalse(hilang,
                         "Perbaikan teks hilang dari naskah:\n  "
                         + "\n  ".join(hilang))

    def test_teks_lama_tidak_kembali(self):
        kembali = ["%s (%s)" % (k, ket) for k, ket in DILARANG_TEKS
                   if k in self.teks]
        self.assertFalse(kembali,
                         "Teks versi lama muncul lagi:\n  "
                         + "\n  ".join(kembali))

    def test_tidak_ada_karakter_gt_di_awal_paragraf(self):
        salah = [q.strip()[:56] for q in self.teks.split("\n")
                 if q.strip().startswith(">")]
        self.assertFalse(salah,
                         "Paragraf diawali karakter '>' (%d buah):\n  %s"
                         % (len(salah), "\n  ".join(salah)))

    # ── 5. kesamaan dengan versi yang ter-commit ────────────────────────────
    def test_naskah_sama_dengan_versi_tercommit(self):
        """
        Penjaga utama modul ini.

        Naskah boleh diubah, tapi harus disengaja. Perubahan yang tidak
        disengaja - seperti yang terjadi ketika `git stash pop` gagal -
        akan terdeteksi di sini.
        """
        if git_hash_blob(NASKAH_REL) is None:
            self.skipTest("naskah tidak ter-track di git, "
                          "perbandingan dilewati")

        di_disk = hashlib.sha256(NASKAH.read_bytes()).hexdigest()
        di_commit = git_hash_blob(NASKAH_REL)
        if di_disk != di_commit:
            self.fail(
                "Isi naskah di disk BERBEDA dari versi yang ter-commit.%s"
                % PETUNJUK_SYNC)

    # ── 6. kebersihan folder thesis ──────────────────────────────────────────
    def test_tidak_ada_berkas_sementara_office(self):
        sisa = [p.name for p in NASKAH.parent.iterdir()
                if p.name.startswith("~$")]
        self.assertFalse(sisa,
                         "Berkas sementara Office tertinggal: %s. "
                         "Tutup Word sepenuhnya, lalu hapus."
                         % ", ".join(sisa))


if __name__ == "__main__":
    print("\nPenjaga naskah skripsi...\n")
    unittest.main(verbosity=2)