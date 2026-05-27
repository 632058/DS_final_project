"""Compare baseline ARIMAX lags against CCF-selected lag specs.

Run:
    python scripts/06b_compare_arimax_specs.py

Output:
    output/arimax_results/arimax_model_comparison.parquet
"""
from __future__ import annotations

import argparse
import warnings

import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning

from src.arimax_model import run_country_arimax
from src.constants import FIPS_COUNTRIES, OUTPUT_DIR
from src.data_loader import load_feature_matrix

X_COLS = ('avg_tone', 'avg_goldstein', 'n_material_conf')
SIGN = {'avg_tone': 'neg', 'avg_goldstein': 'neg', 'n_material_conf': 'pos'}
BASELINE_LAGS = (1, 2, 4)


def _best_nonzero_lag(row: pd.Series, sign: str) -> int:
    candidates = row.drop(labels=['lag_0'], errors='ignore')
    best_col = candidates.idxmin() if sign == 'neg' else candidates.idxmax()
    return int(best_col.replace('lag_', ''))


def load_ccf_selected_lags(ccf_dir) -> dict[str, dict[str, list[int]]]:
    """Return country -> feature -> [best_lag] from saved CCF outputs."""
    selected = {country: {} for country in FIPS_COUNTRIES}
    for x_col in X_COLS:
        path = ccf_dir / f'ccf_{x_col}_to_protest.parquet'
        if not path.exists():
            raise FileNotFoundError(f"Missing CCF output: {path}")

        df_ccf = pd.read_parquet(path)
        for country in FIPS_COUNTRIES:
            if country not in df_ccf.index:
                raise ValueError(f"Missing country {country} in {path}")
            selected[country][x_col] = [_best_nonzero_lag(df_ccf.loc[country], SIGN[x_col])]
    return selected


def main(max_p: int = 3, max_q: int = 3, target_transforms: tuple[str, ...] = ("raw", "log1p")) -> None:
    warnings.simplefilter("ignore", ConvergenceWarning)
    warnings.simplefilter("ignore", ValueWarning)

    df = load_feature_matrix()
    out_dir = OUTPUT_DIR / "arimax_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    ccf_lags = load_ccf_selected_lags(OUTPUT_DIR / "ccf_granger")
    rows = []
    for country in FIPS_COUNTRIES:
        specs = [
            ("baseline_lag_1_2_4", BASELINE_LAGS),
            ("ccf_selected_lag", ccf_lags[country]),
        ]
        for target_transform in target_transforms:
            for spec_name, exog_lag_set in specs:
                summary = run_country_arimax(
                    country,
                    df,
                    out_dir,
                    exog_lag_set=exog_lag_set,
                    max_p=max_p,
                    max_q=max_q,
                    spec_name=spec_name,
                    target_transform=target_transform,
                    save_outputs=False,
                )
                rows.append(summary)
                print(
                    f"{country} {spec_name} {target_transform}: order={tuple(summary['order'])}, "
                    f"RMSE={summary['rmse']:.2f}, MAE={summary['mae']:.2f}, "
                    f"Ljung-Box p={summary['ljung_box_pvalue']:.3f}"
                )

    comparison = pd.DataFrame(rows)
    comparison.to_parquet(out_dir / "arimax_model_comparison.parquet", index=False)
    print(f"\nSaved {out_dir / 'arimax_model_comparison.parquet'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compare ARIMAX lag specifications.")
    parser.add_argument("--max-p", type=int, default=3, help="Maximum AR order for AIC grid search")
    parser.add_argument("--max-q", type=int, default=3, help="Maximum MA order for AIC grid search")
    parser.add_argument(
        "--target-transform",
        choices=["raw", "log1p"],
        action="append",
        dest="target_transforms",
        help="Target transform to compare; repeat to include multiple transforms. Defaults to raw and log1p.",
    )
    args = parser.parse_args()
    transforms = tuple(args.target_transforms) if args.target_transforms else ("raw", "log1p")
    main(max_p=args.max_p, max_q=args.max_q, target_transforms=transforms)
