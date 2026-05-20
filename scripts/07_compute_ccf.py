"""Compute Cross-Correlation Function for tone/goldstein/n_material_conf -> protest_count.

Pre-processing : ADF differencing to stationarity (stationarity_diff).
Window         : PRE_WEEKS before + POST_WEEKS after outbreak date.
Best lag       : directional — most negative for tone/goldstein, most positive for n_material_conf.

Merges logic from mine/compute_ccf_outbreak.py into the canonical script.

Run: python scripts/07_compute_ccf.py
Output: output/ccf_granger/ccf_<x_col>_to_protest.parquet (one per X variable)
"""
from __future__ import annotations

import pandas as pd

from src.constants import OUTPUT_DIR
from src.data_loader import load_country_weekly
from src.stat_tests import compute_ccf, stationarity_diff

PRE_WEEKS  = 52
POST_WEEKS = 12

OUTBREAK_DATES = {
    'CE': '2022-03-31',   # 斯里蘭卡：2022 年經濟崩潰大規模抗議
    'AR': '2018-08-06',   # 阿根廷：比索危機
    'CI': '2019-10-18',   # 智利：estallido social (18-O)
    'TU': '2021-02-01',   # 土耳其：Bogaziçi 大學抗議
    'LE': '2019-10-17',   # 黎巴嫩：十月革命 (17-O)
}

X_COLS = ['avg_tone', 'avg_goldstein', 'n_material_conf']
# expected sign per variable (determines directional best lag)
SIGN   = {'avg_tone': 'neg', 'avg_goldstein': 'neg', 'n_material_conf': 'pos'}


def get_window(df_country: pd.DataFrame, outbreak_date: str) -> pd.DataFrame:
    anchor = pd.Timestamp(outbreak_date)
    start  = anchor - pd.Timedelta(weeks=PRE_WEEKS)
    end    = anchor + pd.Timedelta(weeks=POST_WEEKS)
    mask   = (df_country['week_start'] > start) & (df_country['week_start'] <= end)
    return df_country.loc[mask].reset_index(drop=True)


def best_lag(row: pd.Series, sign: str) -> tuple:
    """Return (best_lag_incl0, r, best_lag_excl0, r_excl0) for a given direction."""
    if sign == 'neg':
        bl  = row.idxmin();              br  = row[bl]
        bl0 = row.drop('lag_0').idxmin(); br0 = row.drop('lag_0')[bl0]
    else:
        bl  = row.idxmax();              br  = row[bl]
        bl0 = row.drop('lag_0').idxmax(); br0 = row.drop('lag_0')[bl0]
    return bl, br, bl0, br0


def main() -> None:
    df = load_country_weekly()
    out_dir = OUTPUT_DIR / 'ccf_granger'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Window: {PRE_WEEKS} pre + {POST_WEEKS} post = {PRE_WEEKS + POST_WEEKS} weeks\n")

    for x_col in X_COLS:
        sign = SIGN[x_col]
        results = {}
        for country, outbreak in OUTBREAK_DATES.items():
            df_c   = df[df['country'] == country].sort_values('week_start').reset_index(drop=True)
            window = get_window(df_c, outbreak)
            x_stat, _ = stationarity_diff(window[x_col])
            y_stat, _ = stationarity_diff(window['protest_count'])
            results[country] = compute_ccf(x_stat, y_stat, max_lag=12)

        df_ccf = pd.DataFrame(results).T
        df_ccf.columns = [f'lag_{c}' for c in df_ccf.columns]
        df_ccf.to_parquet(out_dir / f'ccf_{x_col}_to_protest.parquet')

        print(f"=== {x_col} (取{'最負' if sign == 'neg' else '最正'}相關) ===")
        print(f"{'國家':>6}  {'最佳lag(含0)':>12}  {'r':>7}  {'最佳lag(排除0)':>14}  {'r':>7}")
        for country, row in df_ccf.iterrows():
            bl, br, bl0, br0 = best_lag(row, sign)
            print(f"{country:>6}  {bl:>12}  {br:>7.3f}  {bl0:>14}  {br0:>7.3f}")
        print()


if __name__ == '__main__':
    main()
