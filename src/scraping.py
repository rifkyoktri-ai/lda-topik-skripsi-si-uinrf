"""
scraping.py — Ekstraksi data skripsi dari HTML repositori EPrints.

Membaca file HTML lokal (2021.html - 2025.html) dari data/raw/html_source/
dan mengekstrak: id, year, author, title, abstract, url.

Output: data/raw/thesis_scraped.csv

Dipakai oleh: Notebook 01 (Business Understanding)
"""
import re
import csv
from pathlib import Path

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src import config as C
else:
    from . import config as C


def clean_text(text):
    """Bersihkan whitespace, newline, dan karakter aneh."""
    if not text:
        return ""
    text = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def extract_id_from_url(url):
    """Ekstrak ID numerik dari URL EPrints: /id/eprint/15400 -> 15400."""
    m = re.search(r"/(\d+)", url)
    return m.group(1) if m else ""


def parse_html_file(filepath, year):
    """
    Parse satu file HTML EPrints menjadi list of dict.

    Setiap entri skripsi berada dalam tag <p> dengan struktur:
      <a href="...">NAMA PENULIS</a> (YEAR) <em>JUDUL</em>
    """
    content = filepath.read_text(encoding="utf-8", errors="ignore")
    records = []

    for block in re.findall(r"<p[^>]*>(.*?)</p>", content, re.DOTALL | re.IGNORECASE):
        if f"({year})" not in block:
            continue

        um = re.search(r'href=["\']([^"\']+/\d+[^"\']*)["\']', block)
        url = um.group(1) if um else ""
        if url and url.startswith("/"):
            url = "https://repository.radenfatah.ac.id" + url
        doc_id = extract_id_from_url(url)

        am = re.search(r"<a[^>]*>([^<]+)</a>", block)
        author = clean_text(am.group(1)) if am else ""

        tm = re.search(r"<em>(.*?)</em>", block, re.DOTALL)
        if tm:
            title = clean_text(re.sub(r"<[^>]+>", "", tm.group(1)))
        else:
            parts = re.split(r"\(\d{4}\)", block, maxsplit=1)
            title = clean_text(re.sub(r"<[^>]+>", "", parts[1])) if len(parts) > 1 else ""

        if not title:
            continue

        abstract = ""
        dm = re.search(r'<div[^>]*class=["\']abstract["\'][^>]*>(.*?)</div>',
                       block, re.DOTALL | re.IGNORECASE)
        if dm:
            abstract = clean_text(re.sub(r"<[^>]+>", "", dm.group(1)))

        records.append({"id": doc_id, "year": year, "author": author,
                        "title": title, "abstract": abstract, "url": url})

    return records


def run(verbose=True):
    """Jalankan ekstraksi untuk semua tahun dan simpan ke CSV."""
    all_records = []

    for year in C.YEARS:
        html_path = C.HTML_SOURCE_DIR / f"{year}.html"
        if not html_path.exists():
            if verbose:
                print(f"  [SKIP] {year} - berkas HTML tidak ditemukan")
            continue
        records = parse_html_file(html_path, year)
        if verbose:
            print(f"  [{year}] {len(records):3d} skripsi ditemukan")
        all_records.extend(records)

    if verbose:
        print(f"\n  Total sebelum deduplikasi: {len(all_records)}")

    # Deduplikasi berdasarkan ID (fallback ke judul bila ID kosong)
    seen, unique_records = set(), []
    for r in all_records:
        key = r["id"] or r["title"]
        if key not in seen:
            seen.add(key)
            unique_records.append(r)

    if verbose:
        print(f"  Duplikat dibuang : {len(all_records) - len(unique_records)}")
        print(f"  Dokumen unik    : {len(unique_records)}")

    C.SCRAPED_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "year", "author", "title", "abstract", "url"]
    with open(C.SCRAPED_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(unique_records)

    if verbose:
        print(f"\n[OK] {len(unique_records)} skripsi -> {C.SCRAPED_CSV.relative_to(C.REPO)}")

    return len(unique_records)


if __name__ == "__main__":
    run(verbose=True)