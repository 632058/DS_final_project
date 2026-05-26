"""Per-country XGBoost diagnostic — answer the question:
    Is the pooled model dragging TU down, or is TU's residual a method limit?

For each of the 5 target countries, train a country-specific XGBoost on the
SAME features (with that country's one-hot effectively redundant) and the
SAME hyperparameters as the pooled model, then compare per-country test
metrics. A pooled-vs-per-country comparison surfaces whether:
- per-country model >> pooled  -> pooled model was bad for this country
- per-country model ~= pooled  -> the residual is structural (method limit)

Branches (CLI flags):
- ``--target-scope {all,domestic}``
- ``--target-transform {raw,log1p}``

Output (always written to output/ml_results/{scope}/{transform}/):
- per_country_xgb_diagnostic.parquet
    columns: country, n_train, n_test,
             pooled_rmse, pooled_mae, pooled_dir_acc,
             per_country_rmse, per_country_mae, per_country_dir_acc,
             rmse_delta_vs_pooled  (positive = per-country worse)
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.constants import FIPS_COUNTRIES
from src.data_loader import load_feature_matrix
from src.ml_cli import (
    add_branch_args,
    add_objective_args,
    build_xgb_objective_params,
    objective_suffix,
    resolve_paths,
    validate_objective_args,
)
from src.ml_models import (
    add_target,
    apply_target_transform,
    evaluate_predictions,
    invert_target_transform,
    prepare_xy,
    time_train_test_split,
    train_xgboost,
)

# Kept in sync with scripts/10_train_xgboost.py DEFAULT_PARAMS.
# Duplicated here because Python module names cannot start with a digit, so
# scripts/10_train_xgboost.py cannot be imported as a regular module.
DEFAULT_PARAMS = {
    'n_estimators': 500,
    'max_depth': 5,
    'learning_rate': 0.15,
    'subsample': 0.7,
    'colsample_bytree': 0.6,
    'min_child_weight': 5,
    'gamma': 0.0,
    'reg_alpha': 1.0,
    'reg_lambda': 20.0,
    'random_state': 42,
    'n_jobs': -1,
}


def _build_xgb_params(
    objective: str,
    tweedie_variance_power: float,
    quantile_alpha: float,
) -> dict:
    """Merge DEFAULT_PARAMS with objective-specific XGBoost settings."""
    params = dict(DEFAULT_PARAMS)
    params.update(
        build_xgb_objective_params(
            objective,
            tweedie_variance_power=tweedie_variance_power,
            quantile_alpha=quantile_alpha,
        )
    )
    return params


def _per_country_metrics(pred_df: pd.DataFrame) -> dict[str, dict]:
    """Compute RMSE / MAE / dir_acc per country from a predictions DataFrame."""
    out: dict[str, dict] = {}
    for country, grp in pred_df.groupby('country'):
        err = grp['actual'].to_numpy() - grp['predicted'].to_numpy()
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))
        if len(grp) >= 2:
            g = grp.sort_values('week_start')
            a = np.sign(np.diff(g['actual'].to_numpy()))
            p = np.sign(np.diff(g['predicted'].to_numpy()))
            dir_acc = float(np.mean(a == p))
        else:
            dir_acc = float('nan')
        out[country] = {'rmse': rmse, 'mae': mae, 'dir_acc': dir_acc,
                        'n_test': int(len(grp))}
    return out


def _train_and_predict(
    df: pd.DataFrame,
    scope: str,
    transform: str,
    params: dict,
) -> tuple[pd.DataFrame, dict]:
    """Train XGBoost on df, return predictions DataFrame and overall metrics."""
    df = add_target(df, scope=scope)
    train, test = time_train_test_split(df)
    train = train.sort_values('week_start').reset_index(drop=True)
    X_train, y_train, _ = prepare_xy(train, scope=scope)
    X_test, y_test, mask_test = prepare_xy(test, scope=scope)

    y_train_model = apply_target_transform(y_train, transform)
    model = train_xgboost(X_train, pd.Series(y_train_model), params)
    y_pred_model = model.predict(X_test)
    y_pred = np.clip(invert_target_transform(y_pred_model, transform), 0, None)

    meta = test.loc[mask_test, ['country', 'week_start']].reset_index(drop=True)
    pred_df = meta.copy()
    pred_df['actual'] = y_test.to_numpy()
    pred_df['predicted'] = y_pred
    return pred_df, evaluate_predictions(pred_df)


def main(
    scope: str,
    transform: str,
    objective: str,
    tweedie_variance_power: float,
    quantile_alpha: float,
) -> None:
    validate_objective_args(
        objective, transform, tweedie_variance_power, quantile_alpha
    )

    paths = resolve_paths(scope=scope, transform=transform)
    suffix = objective_suffix(
        objective,
        tweedie_variance_power=tweedie_variance_power,
        quantile_alpha=quantile_alpha,
    )
    params = _build_xgb_params(objective, tweedie_variance_power, quantile_alpha)

    fm = load_feature_matrix()

    # --- Pooled baseline ---
    pooled_pred, pooled_overall = _train_and_predict(fm, scope, transform, params)
    pooled_per_country = _per_country_metrics(pooled_pred)

    # --- Per-country models ---
    rows = []
    print(
        f"\n=== Per-country diagnostic "
        f"(scope={scope}, transform={transform}, objective={objective}) ==="
    )
    print(f"Pooled overall: rmse={pooled_overall['rmse']:.3f} "
          f"mae={pooled_overall['mae']:.3f} "
          f"dir_acc={pooled_overall['directional_accuracy']:.3f}")

    for country in FIPS_COUNTRIES:
        sub = fm[fm['country'] == country].copy()
        if sub.empty:
            continue
        # Drop is_<country> one-hots (now redundant within a single country)
        one_hots = [c for c in sub.columns if c.startswith('is_')]
        sub = sub.drop(columns=one_hots)

        # Time split inside this country
        sub_pred, _ = _train_and_predict(sub, scope, transform, params)
        per_country_only = _per_country_metrics(sub_pred)[country]
        n_train_country = (sub['week_start'] < sub_pred['week_start'].min()).sum()

        pool_m = pooled_per_country[country]
        row = {
            'country': country,
            'n_train': int(n_train_country),
            'n_test': pool_m['n_test'],
            'pooled_rmse': pool_m['rmse'],
            'pooled_mae': pool_m['mae'],
            'pooled_dir_acc': pool_m['dir_acc'],
            'per_country_rmse': per_country_only['rmse'],
            'per_country_mae': per_country_only['mae'],
            'per_country_dir_acc': per_country_only['dir_acc'],
            'rmse_delta_vs_pooled': per_country_only['rmse'] - pool_m['rmse'],
            'objective': objective,
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    print("\n", out.round(3).to_string(index=False))

    out_path = paths.out_dir / f"per_country_xgb_diagnostic{suffix}.parquet"
    out.to_parquet(out_path, index=False)
    print(f"\nSaved {out_path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_branch_args(parser)
    add_objective_args(parser)
    return parser


if __name__ == '__main__':
    args = _build_parser().parse_args()
    main(
        scope=args.target_scope,
        transform=args.target_transform,
        objective=args.objective,
        tweedie_variance_power=args.tweedie_variance_power,
        quantile_alpha=args.quantile_alpha,
    )
