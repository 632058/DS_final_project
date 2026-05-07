"""ADF stationarity test for protest_count, avg_tone, avg_goldstein per country.

Run: python scripts/05_run_adf.py
Output: output/arimax_results/adf_test.parquet
"""
from __future__ import annotations

import pandas as pd

from src.arimax_model import run_adf_test, select_difference_order
from src.constants import FIPS_COUNTRIES, OUTPUT_DIR
from src.data_loader import load_country_weekly


def main() -> None:
    df = load_country_weekly()
    targets = ['protest_count', 'avg_tone', 'avg_goldstein']

    rows = []
    for country in FIPS_COUNTRIES:
        df_c = df[df['country'] == country].sort_values('week_start')
        for col in targets:
            adf = run_adf_test(df_c[col], f'{country}_{col}')
            adf['suggested_d'] = select_difference_order(df_c[col])
            rows.append(adf)

    out = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "arimax_results" / "adf_test.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    print(out[['name', 'p_value', 'is_stationary', 'suggested_d']])


if __name__ == '__main__':
    main()
