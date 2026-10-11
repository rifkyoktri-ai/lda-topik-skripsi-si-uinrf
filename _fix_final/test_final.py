"""
Unit Test Verifikasi Integritas Data & Konsistensi Matematika (_fix_final/)
"""
import pandas as pd
import numpy as np

def test_topic_distribution_total_count():
    df_dist = pd.read_csv('_fix_final/topic_distribution_final.csv')
    assert len(df_dist) == 285, f"Expected 285 documents, got {len(df_dist)}"
    print("PASSED: test_topic_distribution_total_count (285 docs)")

def test_topic_counts_sum():
    df_counts = pd.read_csv('_fix_final/topic_counts_final.csv')
    assert df_counts['jumlah_dokumen'].sum() == 285, f"Expected total 285 documents, got {df_counts['jumlah_dokumen'].sum()}"
    assert np.isclose(df_counts['persentase'].sum(), 100.0, atol=0.05), f"Expected total percentage 100%, got {df_counts['persentase'].sum()}"
    print("PASSED: test_topic_counts_sum (285 docs, 100%)")

def test_topic_by_year_proportions_sum():
    df_year = pd.read_csv('_fix_final/topic_by_year_final.csv')
    for idx, row in df_year.iterrows():
        prop_sum = sum(row[f'proporsi_topik_{t}'] for t in range(1, 9))
        assert np.isclose(prop_sum, 100.0, atol=0.5), f"Year {row['Tahun']} proportion sum {prop_sum} is not 100%"
    print("PASSED: test_topic_by_year_proportions_sum (100% per year)")

def test_evaluation_summary_keys():
    df_eval = pd.read_csv('_fix_final/evaluation_summary.csv')
    metrics = set(df_eval['Metrik'])
    required_metrics = {
        'Coherence_Score_CV', 'Coherence_Score_UMass', 'Train_Log_Perplexity',
        'Topic_Diversity_Top10', 'Topic_Diversity_Top15', 'Pct_Max_Theta_Under_0.50',
        'Reproduksi_CV_0.4026'
    }
    for m in required_metrics:
        assert m in metrics, f"Missing metric {m} in evaluation_summary.csv"
    print("PASSED: test_evaluation_summary_keys (all metrics present)")

def test_consistency_dist_vs_by_year():
    df_dist = pd.read_csv('_fix_final/topic_distribution_final.csv')
    df_year = pd.read_csv('_fix_final/topic_by_year_final.csv').set_index('Tahun')
    
    for year, group in df_dist.groupby('Tahun'):
        year_counts = group['topik_dominan'].value_counts()
        for t in range(1, 9):
            cnt_dist = year_counts.get(t, 0)
            cnt_year = df_year.loc[year, str(t)]
            assert cnt_dist == cnt_year, f"Mismatch in Year {year} Topic {t}: dist has {cnt_dist}, by_year has {cnt_year}"
    print("PASSED: test_consistency_dist_vs_by_year (dist matches by_year)")

if __name__ == '__main__':
    test_topic_distribution_total_count()
    test_topic_counts_sum()
    test_topic_by_year_proportions_sum()
    test_evaluation_summary_keys()
    test_consistency_dist_vs_by_year()
    print("\nALL 5 UNIT TESTS PASSED SUCCESSFULLY!")
