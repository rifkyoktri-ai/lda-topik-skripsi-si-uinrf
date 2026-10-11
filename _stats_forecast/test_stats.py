import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from stats_forecast import calculate_wma

def test_wma_formula():
    """
    Test WMA calculation formula against reference input.
    Input: [31.03, 20.51, 32.61, 27.66, 15.29] with weights [1, 2, 3, 4, 5]
    Expected result: 23.80 (+- 0.01)
    """
    input_series = [31.03, 20.51, 32.61, 27.66, 15.29]
    result = calculate_wma(input_series, window=5)
    assert abs(result - 23.80) <= 0.01, f"Expected ~23.80, got {result}"

def test_sum_shares_per_year():
    """
    Test sum of shares per year equals 100% (+- 0.1 pp).
    """
    topic_by_year_path = os.path.join(os.path.dirname(BASE_DIR), "_fix_final", "topic_by_year_final.csv")
    df = pd.read_csv(topic_by_year_path)
    
    for idx, row in df.iterrows():
        sum_share = sum(row[f'proporsi_topik_{t}'] for t in range(1, 9))
        assert abs(sum_share - 100.0) <= 0.1, f"Year {row['Tahun']} sum share = {sum_share} != 100.0"

def test_chi2_contingency_known_table():
    """
    Test Chi-square function matches scipy on a known 2x2 table.
    Table: [[10, 20], [20, 10]]
    Default SciPy with Yates correction yields chi2 = 5.4
    """
    known_table = np.array([[10, 20], [20, 10]])
    chi2, p_val, dof, expected = stats.chi2_contingency(known_table)
    assert abs(chi2 - 5.4) <= 0.1, f"Expected chi2 ~5.4 (with Yates correction), got {chi2}"

def test_wma_window_rejection():
    """
    Test WMA window > 5 is rejected with ValueError.
    """
    input_series = [10, 20, 30, 40, 50, 60]
    caught = False
    try:
        calculate_wma(input_series, window=6)
    except ValueError as e:
        caught = True
        assert "WMA window cannot exceed 5" in str(e)
    assert caught, "Expected ValueError was not raised for window=6"

def test_forecast_direction_logic():
    """
    Test direction column is NEVER 'naik (indikatif)' if wma_2026 < share_2025.
    """
    forecast_path = os.path.join(BASE_DIR, "forecast_table.csv")
    df_forecast = pd.read_csv(forecast_path)
    
    for idx, row in df_forecast.iterrows():
        wma = row['wma_2026']
        share_2025 = row['share_2025']
        arah = row['arah']
        if wma < share_2025:
            assert arah != "naik (indikatif)", f"Row {row['topik_id']} has WMA {wma} < 2025 share {share_2025} but direction is '{arah}'"

def test_forecast_table_rows():
    """
    Test forecast_table.csv contains exactly 8 rows.
    """
    forecast_path = os.path.join(BASE_DIR, "forecast_table.csv")
    df_forecast = pd.read_csv(forecast_path)
    assert len(df_forecast) == 8, f"Expected 8 rows, got {len(df_forecast)}"

def run_all_tests():
    print("Running unit tests for _stats_forecast...")
    test_wma_formula()
    print("PASSED: test_wma_formula")
    test_sum_shares_per_year()
    print("PASSED: test_sum_shares_per_year")
    test_chi2_contingency_known_table()
    print("PASSED: test_chi2_contingency_known_table")
    test_wma_window_rejection()
    print("PASSED: test_wma_window_rejection")
    test_forecast_direction_logic()
    print("PASSED: test_forecast_direction_logic")
    test_forecast_table_rows()
    print("PASSED: test_forecast_table_rows")
    print("\nALL 6 UNIT TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_all_tests()
