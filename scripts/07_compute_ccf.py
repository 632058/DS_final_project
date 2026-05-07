"""Compute Cross-Correlation Function for tone/goldstein/n_material_conf -> protest_count.

Run: python scripts/07_compute_ccf.py
Output: output/ccf_granger/ccf_<x_col>_to_protest.parquet (one per X variable)
"""
from __future__ import annotations

import pandas as pd

from src.constants import FIPS_COUNTRIES, OUTPUT_DIR
from src.data_loader import load_country_weekly
from src.stat_tests import compute_ccf


def main() -> None:
    df = load_country_weekly()
    out_dir = OUTPUT_DIR / "ccf_granger"
    out_dir.mkdir(parents=True, exist_ok=True)

    x_cols = ['avg_tone', 'avg_goldstein', 'n_material_conf']

    for x_col in x_cols:
        results = {}
        for country in FIPS_COUNTRIES:
            df_c = (
                df[df['country'] == country]
                .sort_values('week_start')
                .reset_index(drop=True)
            )
            ccf = compute_ccf(df_c[x_col], df_c['protest_count'], max_lag=12)
            results[country] = ccf
        df_ccf = pd.DataFrame(results).T
        df_ccf.columns = [f'lag_{c}' for c in df_ccf.columns]
        out_path = out_dir / f"ccf_{x_col}_to_protest.parquet"
        df_ccf.to_parquet(out_path)
        print(f"{x_col}: best lag per country (max |corr|)")
        print(df_ccf.abs().idxmax(axis=1))


if __name__ == '__main__':
    main()
