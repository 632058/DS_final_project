"""Run ARIMAX for one or more countries.

Run for all 5 countries:
    python scripts/06_run_arimax.py CE AR CI TU LE

Run for a subset (when splitting work between A and C):
    python scripts/06_run_arimax.py CE AR CI

Output:
- output/arimax_results/<country>.json (summary)
- output/arimax_results/<country>_forecast.parquet (predictions vs actual)
"""
from __future__ import annotations

import sys

from src.arimax_model import run_country_arimax
from src.constants import FIPS_COUNTRIES, OUTPUT_DIR
from src.data_loader import load_feature_matrix


def main(countries: list[str]) -> None:
    df = load_feature_matrix()
    out_dir = OUTPUT_DIR / "arimax_results"

    for country in countries:
        if country not in FIPS_COUNTRIES:
            print(f"Skipping unknown country: {country}")
            continue
        result = run_country_arimax(country, df, out_dir)
        print(f"{country}: order={tuple(result['order'])}, "
              f"AIC={result['aic']:.2f}, RMSE={result['rmse']:.2f}, "
              f"Ljung-Box p={result['ljung_box_pvalue']:.3f}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python scripts/06_run_arimax.py <COUNTRY> [<COUNTRY> ...]")
        sys.exit(1)
    main(sys.argv[1:])
