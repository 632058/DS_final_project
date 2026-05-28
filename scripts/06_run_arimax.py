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

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.arimax_model import run_country_arimax
from src.constants import FIPS_COUNTRIES, OUTPUT_DIR
from src.data_loader import load_feature_matrix


def _parse_order(value: str) -> tuple[int, int, int]:
    parts = [int(part) for part in value.split(',')]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("order must be formatted as p,d,q")
    return tuple(parts)


def _parse_lags(value: str) -> tuple[int, ...]:
    lags = tuple(int(part) for part in value.split(',') if part)
    if not lags:
        raise argparse.ArgumentTypeError("at least one lag is required")
    return lags


def main(
    countries: list[str],
    order: tuple[int, int, int] | None = None,
    max_p: int = 3,
    max_q: int = 3,
    exog_lag_set: tuple[int, ...] = (1, 2, 4),
    target_transform: str = "raw",
) -> None:
    df = load_feature_matrix()
    out_dir = OUTPUT_DIR / "arimax_results"

    for country in countries:
        if country not in FIPS_COUNTRIES:
            print(f"Skipping unknown country: {country}")
            continue
        result = run_country_arimax(
            country,
            df,
            out_dir,
            exog_lag_set=exog_lag_set,
            order=order,
            max_p=max_p,
            max_q=max_q,
            target_transform=target_transform,
        )
        print(f"{country}: order={tuple(result['order'])}, "
              f"transform={result['target_transform']}, "
              f"AIC={result['aic']:.2f}, RMSE={result['rmse']:.2f}, "
              f"Ljung-Box p={result['ljung_box_pvalue']:.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run ARIMAX for one or more countries.")
    parser.add_argument("countries", nargs="+", help="FIPS country codes, e.g. CE AR CI TU LE")
    parser.add_argument("--order", type=_parse_order, help="Explicit ARIMA order formatted as p,d,q")
    parser.add_argument("--max-p", type=int, default=3, help="Maximum AR order for AIC grid search")
    parser.add_argument("--max-q", type=int, default=3, help="Maximum MA order for AIC grid search")
    parser.add_argument(
        "--target-transform",
        choices=["raw", "log1p"],
        default="raw",
        help="Target scale used for SARIMAX fitting; metrics are always reported on count scale",
    )
    parser.add_argument(
        "--lags",
        type=_parse_lags,
        default=(1, 2, 4),
        help="Comma-separated lag set shared by all exogenous variables",
    )
    args = parser.parse_args()

    if len(sys.argv) < 2:
        parser.print_usage()
        sys.exit(1)
    main(
        args.countries,
        order=args.order,
        max_p=args.max_p,
        max_q=args.max_q,
        exog_lag_set=args.lags,
        target_transform=args.target_transform,
    )
