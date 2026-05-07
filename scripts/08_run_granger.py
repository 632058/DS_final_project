"""Run Granger Causality test for tone/goldstein/n_material_conf -> protest_count.

Series are differenced to stationarity before testing.

Run: python scripts/08_run_granger.py
Output: output/ccf_granger/granger_summary.parquet
"""
from __future__ import annotations

import pandas as pd

from src.constants import FIPS_COUNTRIES, OUTPUT_DIR
from src.data_loader import load_country_weekly
from src.stat_tests import granger_test, stationarity_diff


def main() -> None:
    df = load_country_weekly()
    out_dir = OUTPUT_DIR / "ccf_granger"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for country in FIPS_COUNTRIES:
        df_c = (
            df[df['country'] == country]
            .sort_values('week_start')
            .reset_index(drop=True)
        )
        y_diff, d_y = stationarity_diff(df_c['protest_count'])
        for x_col in ['avg_tone', 'avg_goldstein', 'n_material_conf']:
            x_diff, d_x = stationarity_diff(df_c[x_col])
            common = y_diff.index.intersection(x_diff.index)
            result = granger_test(y_diff.loc[common], x_diff.loc[common], max_lag=12)
            rows.append({
                'country': country,
                'x_var': x_col,
                'd_y': d_y,
                'd_x': d_x,
                'best_lag': result['best_lag'],
                'best_p_value': result['best_p_value'],
                'reject_h0': result['reject_h0'],
            })

    out = pd.DataFrame(rows)
    out_path = out_dir / "granger_summary.parquet"
    out.to_parquet(out_path, index=False)
    print(out)


if __name__ == '__main__':
    main()
