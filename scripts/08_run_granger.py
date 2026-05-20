"""Run Granger Causality test for tone/goldstein/n_material_conf -> protest_count.

Series are differenced to stationarity before testing.
Window: PRE_WEEKS before + POST_WEEKS after outbreak date.

Run: python scripts/08_run_granger.py
Output: output/ccf_granger/granger_summary.parquet
"""
from __future__ import annotations

import pandas as pd

from src.constants import FIPS_COUNTRIES, OUTPUT_DIR
from src.data_loader import load_country_weekly
from src.stat_tests import granger_test, stationarity_diff

PRE_WEEKS  = 52
POST_WEEKS = 12

OUTBREAK_DATES = {
    'CE': '2022-03-31',  # 斯里蘭卡 2022 經濟崩潰大規模抗議
    'AR': '2018-08-06',  # 阿根廷 比索危機
    'CI': '2019-10-18',  # 智利 estallido social
    'TU': '2021-02-01',  # 土耳其 Bogaziçi 大學抗議
    'LE': '2019-10-17',  # 黎巴嫩 十月革命
}


def main() -> None:
    df = load_country_weekly()
    out_dir = OUTPUT_DIR / "ccf_granger"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for country in FIPS_COUNTRIES:
        anchor = pd.Timestamp(OUTBREAK_DATES[country])
        start  = anchor - pd.Timedelta(weeks=PRE_WEEKS)
        end    = anchor + pd.Timedelta(weeks=POST_WEEKS)

        df_c = (
            df[df['country'] == country]
            .sort_values('week_start')
        )
        df_c = df_c[
            (df_c['week_start'] > start) & (df_c['week_start'] <= end)
        ].reset_index(drop=True)

        print(f"{country}: {len(df_c)} 週  "
              f"({df_c['week_start'].min().date()} → {df_c['week_start'].max().date()})")

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
