"""
Scraper: HTML Repository UIN Raden Fatah -> CSV
Membaca file HTML lokal (2021.html - 2025.html) dari data/raw/
dan mengekstrak: id, year, author, title, abstract, url
Output: data/raw/thesis_scraped.csv
"""

import re
import csv
import os
from pathlib import Path

RAW_DIR = Path('data/raw')
OUTPUT_CSV = RAW_DIR / 'thesis_scraped.csv'
YEARS = [2021, 2022, 2023, 2024, 2025]


def clean_text(text: str) -> str:
    """Bersihkan whitespace, newline, dan karakter aneh."""
    if not text:
        return ''
    text = text.replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_id_from_url(url: str) -> str:
    """Ekstrak ID numerik dari URL EPrints, misal /id/eprint/15400 -> 15400"""
    match = re.search(r'/(\d+)', url)
    return match.group(1) if match else ''


def parse_html_file(filepath: Path, year: int) -> list:
    """
    Parse satu file HTML EPrints dan kembalikan list of dict.
    Setiap entri skripsi berada di dalam tag <p> dengan struktur:
      - <a href="...">NAMA PENULIS</a> (TAHUN) <em>JUDUL</em>
    """
    content = filepath.read_text(encoding='utf-8', errors='ignore')
    records = []

    # Ambil semua blok <p>...</p>
    p_blocks = re.findall(r'<p[^>]*>(.*?)</p>', content, re.DOTALL | re.IGNORECASE)

    for block in p_blocks:
        # Harus mengandung tahun dalam kurung, e.g. (2021)
        if f'({year})' not in block:
            continue

        # Ekstrak URL dan ID
        url_match = re.search(r'href=["\']([^"\']+/\d+[^"\']*)["\']', block)
        url = url_match.group(1) if url_match else ''
        # Lengkapi URL relatif
        if url and url.startswith('/'):
            url = 'https://repository.radenfatah.ac.id' + url
        doc_id = extract_id_from_url(url)

        # Ekstrak nama penulis (teks dalam tag <a>)
        author_match = re.search(r'<a[^>]*>([^<]+)</a>', block)
        author = clean_text(author_match.group(1)) if author_match else ''

        # Ekstrak judul (teks dalam tag <em> atau teks setelah (YEAR))
        title_match = re.search(r'<em>(.*?)</em>', block, re.DOTALL)
        if title_match:
            title = clean_text(re.sub(r'<[^>]+>', '', title_match.group(1)))
        else:
            # Fallback: ambil teks setelah (YEAR)
            after_year = re.split(r'\(\d{4}\)', block, maxsplit=1)
            title = clean_text(re.sub(r'<[^>]+>', '', after_year[1])) if len(after_year) > 1 else ''

        # Buang entri kosong
        if not title:
            continue

        # Ekstrak abstrak jika ada (biasanya tidak ada di halaman list, kosongkan)
        abstract = ''
        abstract_match = re.search(r'<div[^>]*class=["\']abstract["\'][^>]*>(.*?)</div>', block, re.DOTALL | re.IGNORECASE)
        if abstract_match:
            abstract = clean_text(re.sub(r'<[^>]+>', '', abstract_match.group(1)))

        records.append({
            'id': doc_id,
            'year': year,
            'author': author,
            'title': title,
            'abstract': abstract,
            'url': url,
        })

    return records


def main():
    all_records = []

    for year in YEARS:
        html_path = RAW_DIR / f'{year}.html'
        if not html_path.exists():
            print(f'[SKIP] {html_path} tidak ditemukan.')
            continue

        records = parse_html_file(html_path, year)
        print(f'[{year}] {len(records)} skripsi ditemukan.')
        all_records.extend(records)

    # Deduplikasi berdasarkan ID
    seen_ids = set()
    unique_records = []
    for r in all_records:
        key = r['id'] or r['title']
        if key not in seen_ids:
            seen_ids.add(key)
            unique_records.append(r)

    # Tulis ke CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['id', 'year', 'author', 'title', 'abstract', 'url']

    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(unique_records)

    print(f'\n[OK] Selesai! Total {len(unique_records)} skripsi unik disimpan ke:')
    print(f'   {OUTPUT_CSV.resolve()}')
    print(f'\nSample 3 baris pertama:')
    for r in unique_records[:3]:
        print(f"  [{r['year']}] {r['id']} | {r['author']} | {r['title'][:60]}...")


if __name__ == '__main__':
    main()
