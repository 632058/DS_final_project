"""Train XGBoost main model on feature_matrix and persist artifacts for SHAP.

Branches (CLI flags):
- ``--target-scope {all,domestic}``     which protest count to predict
- ``--target-transform {raw,log1p}``    forward transform on y
- ``--objective {squarederror,poisson,tweedie,quantile}``  loss function

Count-aware objectives (poisson, tweedie) require ``--target-transform raw``
because they already handle the heavy-tailed distribution via their link
function. Combining them with ``log1p`` would double-transform and is
rejected at the CLI layer.

Run examples:
    python scripts/10_train_xgboost.py
    python scripts/10_train_xgboost.py --target-transform log1p
    python scripts/10_train_xgboost.py --target-scope domestic --target-transform log1p
    python scripts/10_train_xgboost.py --objective poisson
    python scripts/10_train_xgboost.py --objective tweedie --tweedie-variance-power 1.5
    python scripts/10_train_xgboost.py --objective quantile --quantile-alpha 0.5

Outputs (always written to output/ml_results/{scope}/{transform}/):
- xgboost_model{_suffix}.joblib
- X_test{_suffix}.parquet
- y_test{_suffix}.parquet
- xgboost_predictions{_suffix}.parquet
- xgboost_metrics{_suffix}.parquet

The ``_suffix`` is empty for ``--objective squarederror`` (canonical names,
preserves backward compatibility with downstream scripts). For other
objectives the suffix encodes the objective and its hyperparameter, e.g.
``_poisson``, ``_tweedie_1.5``, ``_quantile_0.5``.

For the default branch (scope=all, transform=raw, objective=squarederror)
the canonical files are ALSO mirrored to the legacy output/ml_results/ top
level for backward compat with other members' SHAP / figure scripts.
"""
from __future__ import annotations

import argparse
import shutil

import joblib
import numpy as np
import pandas as pd

from src.data_loader import load_feature_matrix
from src.ml_cli import (
    BranchPaths,
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

DEFAULT_PARAMS = {
    'n_estimators': 300,
    'max_depth': 3,
    'learning_rate': 0.03,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
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


def _mirror_to_top_level(paths: BranchPaths, filenames: list[str]) -> None:
    """Copy branch outputs to the legacy top-level paths when applicable."""
    if not paths.also_top_level:
        return
    for name in filenames:
        src = paths.out_dir / name
        if src.exists():
            shutil.copy2(src, paths.legacy_out_dir / name)


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
    xgb_params = _build_xgb_params(objective, tweedie_variance_power, quantile_alpha)

    df = load_feature_matrix()
    df = add_target(df, scope=scope)
    train, test = time_train_test_split(df)

    X_train, y_train, _ = prepare_xy(train, scope=scope)
    X_test, y_test, mask_test = prepare_xy(test, scope=scope)

    y_train_model = apply_target_transform(y_train, transform)
    model = train_xgboost(X_train, pd.Series(y_train_model), xgb_params)

    y_pred_model = model.predict(X_test)
    y_pred = np.clip(invert_target_transform(y_pred_model, transform), 0, None)

    test_meta = test.loc[mask_test, ['country', 'week_start']].reset_index(drop=True)
    pred_df = test_meta.copy()
    pred_df['actual'] = y_test.to_numpy()
    pred_df['predicted'] = y_pred
    metrics = evaluate_predictions(pred_df)

    print(
        f"XGBoost scope={scope} transform={transform} "
        f"objective={objective}{f' p={tweedie_variance_power}' if objective == 'tweedie' else ''}"
        f"{f' alpha={quantile_alpha}' if objective == 'quantile' else ''}: "
        f"rmse={metrics['rmse']:.4f} mae={metrics['mae']:.4f} "
        f"dir_acc={metrics['directional_accuracy']:.4f}"
    )
    print("Per-country dir acc:", metrics['per_country_directional'])

    model_path = paths.out_dir / f"xgboost_model{suffix}.joblib"
    x_test_path = paths.out_dir / f"X_test{suffix}.parquet"
    y_test_path = paths.out_dir / f"y_test{suffix}.parquet"
    pred_path = paths.out_dir / f"xgboost_predictions{suffix}.parquet"
    metrics_path = paths.out_dir / f"xgboost_metrics{suffix}.parquet"

    joblib.dump(
        {
            'model': model,
            'feature_cols': list(X_train.columns),
            'transform': transform,
            'scope': scope,
            'objective': objective,
            'objective_params': {
                k: v for k, v in xgb_params.items()
                if k in ('tweedie_variance_power', 'quantile_alpha')
            },
        },
        model_path,
    )
    X_test.to_parquet(x_test_path)
    y_test.to_frame(name='actual').to_parquet(y_test_path)
    pred_df.to_parquet(pred_path, index=False)

    metric_row = {
        'scope': scope,
        'transform': transform,
        'objective': objective,
        'tweedie_variance_power': (
            tweedie_variance_power if objective == 'tweedie' else float('nan')
        ),
        'quantile_alpha': (
            quantile_alpha if objective == 'quantile' else float('nan')
        ),
        'rmse': metrics['rmse'],
        'mae': metrics['mae'],
        'directional_accuracy': metrics['directional_accuracy'],
    }
    pd.DataFrame([metric_row]).to_parquet(metrics_path, index=False)

    # Only the canonical (squarederror) artefacts get mirrored to the legacy
    # top-level paths. Mirroring poisson/tweedie/quantile predictions would
    # silently break downstream scripts that assume reg:squarederror semantics.
    if objective == 'squarederror':
        _mirror_to_top_level(paths, [
            "xgboost_model.joblib",
            "X_test.parquet",
            "y_test.parquet",
            "xgboost_predictions.parquet",
            "xgboost_metrics.parquet",
        ])


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
