"""
data_lineage.py — Jejak Lineage Data & Pelacakan Dokumen Tereliminasi.

Menelusuri jumlah dokumen pada setiap tahap pipeline dan mencatat dokumen
yang dibuang beserta alasannya (duplikat judul / token < 5).

Output:
  docs/data_lineage_by_year.csv
  docs/excluded_documents.csv

Dipakai oleh: Notebook 02 (Data Understanding)
"""
import pandas as pd
from difflib import SequenceMatcher
from pathlib import Path

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src import config as C
else:
    from . import config as C


def run_data_lineage(verbose=True):
    """Hitung lineage per tahun + daftar dokumen tereliminasi."""
    C.DOCS_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(C.SCRAPED_CSV)
    proc_df = pd.read_csv(C.PIPELINE_CSV)
    try:
        prep_df = pd.read_csv(C.PREPROCESSED_CSV)
    except FileNotFoundError:
        prep_df = pd.DataFrame()

    if verbose:
        print("=" * 62)
        print("DATA LINEAGE & EXCLUDED DOCUMENTS TRACER")
        print("=" * 62)

    # ── Lineage per tahun ──────────────────────────────────────────────────
    rows = []
    for y in C.YEARS:
        n_raw = int((raw_df["year"] == y).sum())
        n_proc = int((proc_df["year"] == y).sum())
        n_prep = int((prep_df["Tahun"] == y).sum()) if len(prep_df) else 0
        rows.append({
            "tahun": y,
            "raw_scraped_count": n_raw,
            "processed_count": n_proc,
            "preprocessed_final_count": n_prep,
            "dropped_raw_to_processed": n_raw - n_proc,
            "dropped_preprocessed_dedup_and_length": n_proc - n_prep if len(prep_df) else None,
            "total_dropped": n_raw - n_prep if len(prep_df) else None,
        })

    lineage = pd.DataFrame(rows)
    lineage.to_csv(C.DATA_LINEAGE_CSV, index=False, encoding="utf-8-sig")

    if verbose:
        print("\nLINEAGE PER TAHUN")
        print(lineage.to_string(index=False))
        print(f"\n[OK] -> {C.DATA_LINEAGE_CSV.relative_to(C.REPO)}")

    # ── Dokumen tereliminasi ──────────────────────────────────────────────
    raw_ids = set(raw_df["id"])
    prep_ids = set(prep_df["ID"]) if len(prep_df) else set()
    proc_ids = set(proc_df["id"])

    excluded = []

    # Tahap 1: raw -> processed
    for did in sorted(raw_ids - proc_ids):
        r = raw_df[raw_df["id"] == did]
        if r.empty:
            continue
        r = r.iloc[0]
        excluded.append({
            "id": did, "title": r["title"], "year": r["year"],
            "stage": "raw_to_processed",
            "reason": "Tereliminasi pada pembersihan scraping awal",
            "similarity_score_with_duplicate": "",
        })

    # Tahap 2 & 3: duplikat judul & token < 5
    proc_df = proc_df.copy()
    proc_df["tmp_title_lower"] = proc_df["title"].astype(str).str.strip().str.lower()
    dup_ids = set(proc_df[proc_df.duplicated(subset=["tmp_title_lower"], keep="first")]["id"])

    for did in sorted(dup_ids):
        r = proc_df[proc_df["id"] == did]
        if r.empty:
            continue
        r = r.iloc[0]
        same = proc_df[(proc_df["tmp_title_lower"] == r["tmp_title_lower"]) &
                       (proc_df["id"] != did)]
        sim = ""
        reason = "Duplikat judul"
        if not same.empty:
            orig = same.iloc[0]
            sim = f"{SequenceMatcher(None, str(r['title']), str(orig['title'])).ratio():.4f}"
            reason = f"Duplikat judul dari dokumen ID {orig['id']}"
        excluded.append({
            "id": did, "title": r["title"], "year": r["year"],
            "stage": "preprocess_title_dedup",
            "reason": reason, "similarity_score_with_duplicate": sim,
        })

    # Token < 5
    if len(prep_df):
        for did in sorted((proc_ids - dup_ids) - prep_ids):
            r = proc_df[proc_df["id"] == did]
            if r.empty:
                continue
            r = r.iloc[0]
            excluded.append({
                "id": did, "title": r["title"], "year": r["year"],
                "stage": "preprocess_token_len_filter",
                "reason": "Jumlah token < 5 setelah stopword filtering & stemming",
                "similarity_score_with_duplicate": "",
            })

    ex_df = pd.DataFrame(excluded)
    if len(ex_df):
        ex_df.to_csv(C.EXCLUDED_DOCS_CSV, index=False, encoding="utf-8-sig")
        if verbose:
            print(f"\nDOKUMEN TERELIMINASI: {len(ex_df)}")
            print(ex_df.groupby("stage").size().to_string())
            print(f"\n[OK] -> {C.EXCLUDED_DOCS_CSV.relative_to(C.REPO)}")

    return lineage, ex_df


if __name__ == "__main__":
    run_data_lineage(verbose=True)