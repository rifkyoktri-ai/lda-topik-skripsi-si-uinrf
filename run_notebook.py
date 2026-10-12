"""
run_notebook.py — Eksekusi sel kode sebuah notebook .ipynb dari terminal.

Berguna untuk memverifikasi notebook berjalan tanpa perlu membuka Jupyter.
Default: hanya menjalankan sel kode, menyimpan output ke console.

Jalankan:  python run_notebook.py notebooks/01_business_understanding.ipynb
"""
import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent


def run(path, timeout_note=""):
    nb_path = Path(path)
    if not nb_path.is_absolute():
        nb_path = ROOT / nb_path
    nb = json.loads(nb_path.read_text(encoding="utf-8"))

    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    print(f"{'=' * 70}")
    print(f"MENJALANKAN: {nb_path.name}  ({len(code_cells)} sel kode)")
    print(f"{'=' * 70}")

    import os
    workdir = nb_path.parent
    cwd = os.getcwd()
    os.chdir(workdir)
    sys.path.insert(0, str(workdir))
    sys.path.insert(0, str(ROOT))

    g = {"__name__": "__main__", "__file__": str(nb_path)}
    failures = []

    try:
        for i, cell in enumerate(code_cells, start=1):
            src = "\n".join(cell["source"])
            if not src.strip():
                continue
            print(f"\n--- Sel {i} ---")
            try:
                exec(compile(src, f"<{nb_path.name} sel {i}>", "exec"), g)
            except Exception as exc:
                print(f"  [GAGAL sel {i}] {type(exc).__name__}: {exc}")
                failures.append((i, f"{type(exc).__name__}: {exc}"))
                break
    finally:
        os.chdir(cwd)

    print(f"\n{'=' * 70}")
    if failures:
        print(f"HASIL: GAGAL di sel {failures[0][0]}")
    else:
        print(f"HASIL: SUKSES - {len(code_cells)} sel kode berjalan tanpa error")
    print(f"{'=' * 70}")
    return len(failures) == 0


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "notebooks/08_build_final_artifacts.ipynb"
    ok = run(target)
    sys.exit(0 if ok else 1)