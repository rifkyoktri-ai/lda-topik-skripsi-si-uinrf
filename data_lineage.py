"""
Script Lineage Data & Pelacakan Dokumen Tereliminasi
Proyek Skripsi LDA UIN Raden Fatah Palembang
"""

import pandas as pd
import numpy as np
from difflib import SequenceMatcher
from pathlib import Path

OUTPUT_DIR = Path('docs')


def run_data_lineage():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("DATA LINEAGE & EXCLUDED DOCUMENTS TRACER")
    print("=" * 60)

    raw_path = Path('data/raw/thesis_scraped.csv')
    proc_path = Path('data/processed/thesis_for_pipeline.csv')
    prep_path = Path('data/intermediate/dataset_preprocessed.csv')

    raw_df = pd.read_csv(raw_path)
    proc_df = pd.read_csv(proc_path)
    prep_df = pd.read_csv(prep_path)

    years = [2021, 2022, 2023, 2024, 2025]
    lineage_rows = []

    for y in years:
        raw_y = len(raw_df[raw_df['year'] == y])
        proc_y = len(proc_df[proc_df['year'] == y])
        prep_y = len(prep_df[prep_df['Tahun'] == y])

        dropped_proc = raw_y - proc_y
        dropped_prep = proc_y - prep_y

        lineage_rows.append({
            'tahun': y,
            'raw_scraped_count': raw_y,
            'processed_count': proc_y,
            'preprocessed_final_count': prep_y,
            'dropped_raw_to_processed': dropped_proc,
            'dropped_preprocessed_dedup_and_length': dropped_prep,
            'total_dropped': raw_y - prep_y
        })

    lineage_df = pd.DataFrame(lineage_rows)
    lineage_df.to_csv(OUTPUT_DIR / 'data_lineage_by_year.csv', index=False)
    print(f"docs/data_lineage_by_year.csv disimpankan ({len(lineage_df)} baris tahun)")

    # Pelacakan dokumen tereliminasi
    excluded = []

    raw_ids = set(raw_df['id'])
    proc_ids = set(proc_df['id'])
    prep_ids = set(prep_df['ID'])

    # Stage 1: Raw -> Processed
    dropped_proc_ids = raw_ids - proc_ids
    for did in sorted(dropped_proc_ids):
        r = raw_df[raw_df['id'] == did].iloc[0]
        excluded.append({
            'id': did,
            'title': r['title'],
            'year': r['year'],
            'stage': 'raw_to_processed',
            'reason': 'Tereliminasi pada pembersihan scraping awal (bukan prodi SI / artefak scraping)',
            'similarity_score_with_duplicate': ''
        })

    # Stage 2: Processed -> Preprocessed (Judul Duplikat & Token < 5)
    proc_df['temp_title_lower'] = proc_df['title'].astype(str).str.strip().str.lower()
    dups = proc_df[proc_df.duplicated(subset=['temp_title_lower'], keep='first')]

    dup_ids = set(dups['id'])
    for idx, r in dups.iterrows():
        orig = proc_df[(proc_df['temp_title_lower'] == r['temp_title_lower']) & (proc_df['id'] != r['id'])].iloc[0]
        sim = SequenceMatcher(None, str(r['title']), str(orig['title'])).ratio()
        excluded.append({
            'id': r['id'],
            'title': r['title'],
            'year': r['year'],
            'stage': 'preprocess_title_dedup',
            'reason': f"Duplikat judul dari dokumen ID {orig['id']}",
            'similarity_score_with_duplicate': f"{sim:.4f}"
        })

    # Stage 3: Eliminasi Token < 5
    remaining_proc_ids = proc_ids - dup_ids
    dropped_len_ids = remaining_proc_ids - prep_ids

    for did in sorted(dropped_len_ids):
        r = proc_df[proc_df['id'] == did].iloc[0]
        excluded.append({
            'id': did,
            'title': r['title'],
            'year': r['year'],
            'stage': 'preprocess_token_len_filter',
            'reason': 'Jumlah token < 5 setelah stopword filtering & stemming Sastrawi',
            'similarity_score_with_duplicate': ''
        })

    ex_df = pd.DataFrame(excluded)
    ex_df.to_csv(OUTPUT_DIR / 'excluded_documents.csv', index=False)
    print(f"docs/excluded_documents.csv disimpankan ({len(ex_df)} total dokumen tereliminasi)")

    # Print abstrak kosong
    print("\n=== DOKUMEN DENGAN ABSTRAK KOSONG IN THESIS_FOR_PIPELINE.CSV ===")
    null_abstr_df = proc_df[proc_df['abstract'].isna() | (proc_df['abstract'].astype(str).str.strip() == '') | (proc_df['abstract'].astype(str) == 'nan')]
    for idx, r in null_abstr_df.iterrows():
        print(f"  ID: {r['id']}, Tahun: {r['year']}, Judul: {r['title']}")

if __name__ == '__main__':
    run_data_lineage()
