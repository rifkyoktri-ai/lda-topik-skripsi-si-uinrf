#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run.py - SATU PINTU MASUK untuk seluruh perintah repo.

Kenapa ada berkas ini selain Makefile?
    Makefile adalah standar di Linux/macOS/Git Bash, tapi `make` tidak
    terpasang di banyak mesin Windows. Berkas ini memakai Python yang
    sudah ada di proyek, jadi jalan di mana saja.

Pakai:
    python run.py            (tanpa argumen) -> tampilkan daftar
    python run.py test
    python run.py notebook
    python run.py dashboard

Jalankan dari KONEKTOR:
    python run.py test       Test, notebook cepat, audit, validasi struktur
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def python_exe() -> str:
    """Pakai Python dari .venv bila ada, kalau tidak Python sistem."""
    for rel in (Path('.venv/Scripts/python.exe'), Path('.venv/bin/python')):
        p = ROOT / rel
        if p.exists():
            return str(p)
    return sys.executable


def run(args, desc):
    print("\n" + "=" * 62)
    print("  " + desc)
    print("=" * 62)
    r = subprocess.run([python_exe()] + args, cwd=str(ROOT))
    return r.returncode == 0


BANTUAN = """
  REPO SKRIPSI LDA - UIN RADEN FATAH
  ----------------------------------

  python run.py install       pasang dependensi
  python run.py test          27 unit test
  python run.py notebook      notebook 01-08 (lama, ~20 menit)
  python run.py notebook-fast 01-03 + 05-08 (lewati training, ~2 menit)
  python run.py audit         regenerate outputs/audit/assignment_audit.md
  python run.py figures       bangkitkan ulang wordcloud 8 topik
  python run.py structure     validasi struktur repo
  python run.py dashboard     dashboard Streamlit
  python run.py all           test -> notebook-fast -> audit
  python run.py clean         hapus __pycache__ & logs (outputs/ TIDAK disentuh)

  Dokumentasi penuh: README.md
"""

# (argumen, perintah, deskripsi)
URUTAN_NOTEBOOK_CEPAT = [
    "01_business_understanding", "02_data_understanding",
    "03_data_preparation", "05_evaluation_lda",
    "06_evaluation_forecast", "07_labeling",
    "08_build_final_artifacts",
]
URUTAN_NOTEBOOK_PENUH = [
    "01_business_understanding", "02_data_understanding",
    "03_data_preparation", "04_modeling_lda", "04b_grid_search",
    "05_evaluation_lda", "06_evaluation_forecast", "07_labeling",
    "08_build_final_artifacts",
]

PERINTAH = {
    "test": lambda: all([
        run(["run_tests.py"], "Unit test pipeline, integritas, label, metrik"),
        run(["tests/test_final.py"], "Konsistensi artefak final"),
        run(["tests/test_labels.py"], "Integritas label & fingerprint"),
        run(["tests/test_stats.py"], "Matematika WMA & uji statistik"),
    ]),
    "notebook": lambda: all(
        run(["run_notebook.py", "notebooks/%s.ipynb" % n], "Notebook " + n)
        for n in URUTAN_NOTEBOOK_PENUH),
    "notebook-fast": lambda: all(
        run(["run_notebook.py", "notebooks/%s.ipynb" % n], "Notebook " + n)
        for n in URUTAN_NOTEBOOK_CEPAT),
    "audit": lambda: run(["-m", "src.assignment_audit"],
                         "Audit keyakinan penugasan dokumen"),
    "figures": lambda: run(["-m", "src.make_figures"],
                           "Bangkitkan ulang wordcloud 8 topik"),
    "structure": lambda: run(["tests/test_structure.py"],
                             "Validasi struktur repo"),
    "dashboard": lambda: run(["-m", "streamlit", "run", "app/main.py"],
                             "Dashboard Streamlit"),
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(BANTUAN)
        return 0

    cmd = sys.argv[1]

    if cmd == "install":
        return 0 if run(["-m", "pip", "install", "-r", "requirements.txt"],
                        "Pasang dependensi") else 1

    if cmd == "clean":
        import shutil
        for d in ("logs", ".ipynb_checkpoints"):
            shutil.rmtree(ROOT / d, ignore_errors=True)
        for pc in ROOT.rglob("__pycache__"):
            if ".venv" not in pc.parts:
                shutil.rmtree(pc, ignore_errors=True)
        print("\n  Selesai. Folder outputs/ TIDAK disentuh - jangan dihapus manual.")
        return 0

    if cmd == "all":
        for tahap in ("test", "notebook-fast", "audit"):
            if not PERINTAH[tahap]():
                print("\n  GAGAL pada tahap: %s" % tahap)
                return 1
        print("\n  SEMUA SELESAI. Verifikasi GROUND_TRUTH.md untuk angka resmi.")
        return 0

    if cmd in PERINTAH:
        return 0 if PERINTAH[cmd]() else 1

    print("  Perintah tidak dikenal: %r" % cmd)
    print(BANTUAN)
    return 1


if __name__ == "__main__":
    sys.exit(main())