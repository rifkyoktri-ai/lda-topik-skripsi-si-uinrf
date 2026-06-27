"""
Analisis Tren dan Prediksi menggunakan Weighted Moving Average (WMA)
Dipilih karena: data time-series hanya 5 titik dan bersifat fluktuatif
(tidak memiliki tren linear yang konsisten)
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from scipy import stats


def compute_wma(values: list, weights: list = None) -> float:
    n = len(values)
    if weights is None:
        weights = list(range(1, n + 1))

    weights = np.array(weights[:n], dtype=float)
    values = np.array(values, dtype=float)
    return float(np.dot(weights, values) / weights.sum())


def compute_r_squared(values: list) -> float:
    n = len(values)
    x = np.arange(n)
    y = np.array(values, dtype=float)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return float(r_value ** 2)


def determine_trend_direction(values: list, pred_2026: float) -> str:
    last_value = values[-1]
    diff = pred_2026 - last_value
    threshold = 0.01

    if diff > threshold:
        return "Meningkat"
    elif diff < -threshold:
        return "Menurun"
    else:
        return "Stabil"


def analyze_topic_trends(
    topic_trend_df: pd.DataFrame,
    topic_labels: dict,
    years: list = [2021, 2022, 2023, 2024, 2025],
    pred_years: list = [2026, 2027]
) -> pd.DataFrame:
    results = []
    topic_cols = [c for c in topic_trend_df.columns if c != 'Tahun']

    for col in topic_cols:
        try:
            topic_id = int(''.join(filter(str.isdigit, col))) - 1
        except:
            topic_id = topic_cols.index(col)

        historical = []
        for year in years:
            year_data = topic_trend_df[
                topic_trend_df['Tahun'] == year
            ][col]
            if len(year_data) > 0:
                historical.append(float(year_data.iloc[0]))
            else:
                historical.append(0.0)

        if len(historical) < 2:
            continue

        r2 = compute_r_squared(historical)
        weights = list(range(1, len(historical) + 1))
        pred_2026 = compute_wma(historical, weights)

        historical_for_2027 = historical[1:] + [pred_2026]
        pred_2027 = compute_wma(historical_for_2027, weights)

        pred_2026 = max(0.0, min(1.0, pred_2026))
        pred_2027 = max(0.0, min(1.0, pred_2027))

        trend = determine_trend_direction(historical, pred_2026)

        label = topic_labels.get(
            topic_id,
            topic_labels.get(str(topic_id), f"Topik {topic_id + 1}")
        )

        row = {
            "topic_id" : topic_id,
            "Label"    : label,
            "R2"       : round(r2, 4),
            "Metode"   : "Weighted Moving Average",
            "Arah"     : trend,
        }

        for year, val in zip(years, historical):
            row[str(year)] = round(val * 100, 2)

        row[str(pred_years[0])] = round(pred_2026 * 100, 2)
        row[str(pred_years[1])] = round(pred_2027 * 100, 2)

        results.append(row)

    result_df = pd.DataFrame(results)

    result_df['Linearitas'] = result_df['R2'].apply(
        lambda r: "Sangat Linier" if r >= 0.8
        else ("Cukup Linier" if r >= 0.5 else "Tidak Linier")
    )

    return result_df


def save_trend_results(trend_df: pd.DataFrame, output_dir: str = "model") -> None:
    output_path = Path(output_dir) / "trend_prediction.csv"
    trend_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Trend prediction disimpan ke: {output_path}")


def run_trend_analysis():
    base = Path(".")

    trend_path = base / "model" / "topic_trend.csv"
    if not trend_path.exists():
        print(f"File tidak ditemukan: {trend_path}")
        print("Jalankan pipeline utama terlebih dahulu.")
        return

    trend_df = pd.read_csv(trend_path)

    label_path = base / "model" / "topic_labels.json"
    if label_path.exists():
        with open(label_path, 'r', encoding='utf-8') as f:
            labels_data = json.load(f)
        topic_labels = {int(k): v['label_final'] for k, v in labels_data.items()}
    else:
        topic_labels = {}

    result = analyze_topic_trends(trend_df, topic_labels)
    save_trend_results(result)

    print("\nHASIL PREDIKSI TREN:")
    cols = ['Label', 'R2', 'Linearitas', 'Metode', 'Arah',
            '2021', '2022', '2023', '2024', '2025', '2026', '2027']
    print(result[cols].to_string(index=False))


if __name__ == "__main__":
    run_trend_analysis()
