"""
run_notebook.py - Eksekusi sel kode notebook .ipynb dari terminal,
lalu MENYIMPAN output (teks + gambar) kembali ke dalam file .ipynb.

Mengapa:
  1. Output tersimpan -> notebook bisa langsung dipakai sebagai lampiran
     skripsi; angka & grafik terlihat tanpa harus di-run di depan dosen.
  2. CWD dipaksa ke root repo -> modul yang menulis berkas (log, report)
     tidak lagi scattering ke dalam folder notebooks/.

Jalankan:
    python run_notebook.py                          # semua notebook
    python run_notebook.py notebooks/08_build_final_artifacts.ipynb
    python run_notebook.py --no-save notebooks/03_data_preparation.ipynb
"""
import base64
import contextlib
import glob
import io
import json
import os
import sys
import traceback
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
NB_DIR = ROOT / "notebooks"


class _Cap(io.StringIO):
    """StringIO yang mendukung .reconfigure() - sel SETUP notebook memanggil
    sys.stdout.reconfigure(), sedangkan io.StringIO biasa tidak punya."""

    def reconfigure(self, *a, **k):
        return None


_PLOTLY_HTML = []          # figure plotly yang ditangkap pada sel berjalan


def _install_plotly_hook():
    """Tangkap fig.show() dari plotly sebagai HTML (matplotlib != plotly)."""
    try:
        import plotly.graph_objects as go
    except Exception:
        return
    if getattr(go.Figure, "_opencapture", False):
        return
    _orig_show = go.Figure.show

    def _show(self, *a, **k):
        try:
            _PLOTLY_HTML.append(
                self.to_html(include_plotlyjs="cdn", full_html=False,
                             config={"displayModeBar": False}))
        except Exception:
            return _orig_show(self, *a, **k)

    go.Figure.show = _show
    go.Figure._opencapture = True


def _capture_figures():
    """Ambil figure matplotlib yang masih terbuka -> daftar PNG base64."""
    imgs = []
    try:
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
        for num in plt.get_fignums():
            fig = plt.figure(num)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                        facecolor="white")
            imgs.append(base64.b64encode(buf.getvalue()).decode("ascii"))
            plt.close(fig)
    except Exception:
        pass
    return imgs


def run_one(nb_path, save=True, verbose=True):
    nb_path = Path(nb_path)
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    code_idx = 0
    total = len([c for c in nb["cells"] if c["cell_type"] == "code"])

    cwd = os.getcwd()
    argv0 = list(sys.argv)
    os.chdir(ROOT)                      # <- CWD = root repo
    sys.path.insert(0, str(ROOT))
    if str(ROOT) in sys.path[1:]:
        sys.path.remove(str(ROOT))
    # argparse di dalam notebook (grid_search, modeling, ...) harus TIDAK
    # membaca argumen runner ini
    sys.argv = [str(nb_path)]

    g = {"__name__": "__main__", "__file__": str(nb_path)}

    def _display(*objs, **kwargs):
        for o in objs:
            print(repr(o) if not hasattr(o, "_repr_html_") else
                  o._repr_html_())
    g["display"] = _display
    g["get_ipython"] = lambda: None
    _install_plotly_hook()

    ok = True
    try:
        for cell in nb["cells"]:
            if cell["cell_type"] != "code":
                continue
            code_idx += 1
            src = "".join(cell["source"])
            if not src.strip():
                cell["outputs"] = []
                cell["execution_count"] = None
                continue

            buf = _Cap()
            err = None
            try:
                with contextlib.redirect_stdout(buf), \
                        contextlib.redirect_stderr(_Cap()):
                    exec(compile(src, f"<{nb_path.name} sel {code_idx}>", "exec"), g)
            except Exception:
                err = traceback.format_exc()

            figs = _capture_figures()
            htmls = list(_PLOTLY_HTML)
            _PLOTLY_HTML.clear()
            out = []
            text = buf.getvalue()
            if text:
                out.append({"output_type": "stream", "name": "stdout",
                            "text": text.rstrip("\n").splitlines(keepends=True)})
            for b64 in figs:
                out.append({"output_type": "display_data",
                            "data": {"image/png": b64}, "metadata": {}})
            for h in htmls:
                out.append({"output_type": "display_data",
                            "data": {"text/html": h}, "metadata": {}})
            if err:
                out.append({"output_type": "error", "ename": "Exception",
                            "evalue": err.strip().splitlines()[-1],
                            "traceback": err.splitlines()})
                ok = False

            cell["outputs"] = out
            cell["execution_count"] = code_idx if not err else None

            if verbose:
                n_img = len(figs) + len(htmls)
                print("  sel %-2d  %-46s %s"
                      % (code_idx, src.strip().splitlines()[0][:44],
                         ("%d baris" % len(text.splitlines())) if text else "",
                         ))
                if n_img:
                    kinds = []
                    if figs:
                        kinds.append("%d PNG" % len(figs))
                    if htmls:
                        kinds.append("%d plotly" % len(htmls))
                    print("            + %s" % ", ".join(kinds))
            if err:
                print("  [GAGAL] %s" % err.strip().splitlines()[-1])
                break
    finally:
        os.chdir(cwd)
        sys.argv = argv0

    if save and ok:
        nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

    n_saved = sum(1 for c in nb["cells"]
                  if c["cell_type"] == "code" and c.get("outputs"))
    print("  -> %s | %d/%d sel punya output%s"
          % ("SUKSES" if ok else "GAGAL", n_saved, total,
             " (disimpan)" if save and ok else ""))
    return ok


def main():
    args = sys.argv[1:]
    save = "--no-save" not in args
    args = [a for a in args if not a.startswith("--")]

    if args:
        targets = [ROOT / a for a in args]
    else:
        targets = sorted(Path(p) for p in glob.glob(str(NB_DIR / "*.ipynb")))

    results = []
    for t in targets:
        print("=" * 72)
        print("MENJALANKAN: %s" % t.name)
        print("=" * 72)
        results.append((t.name, run_one(t, save=save)))
        print()

    print("=" * 72)
    print("RINGKASAN")
    print("=" * 72)
    for name, ok in results:
        print("  %-42s %s" % (name, "OK" if ok else "GAGAL"))
    bad = [n for n, o in results if not o]
    print("  Total: %d/%d berhasil" % (len(results) - len(bad), len(results)))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
