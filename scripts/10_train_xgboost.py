"""Train XGBoost main model on feature_matrix and persist artifacts for SHAP.

Branches (CLI flags):
- ``--target-scope {all,domestic}``     which protest count to predict
- ``--target-transform {raw,log1p}``    forward transform on y

Run examples:
    python scripts/10_train_xgboost.py
    python scripts/10_train_xgboost.py --target-transform log1p
    python scripts/10_train_xgboost.py --target-scope domestic --target-transform log1p

Outputs (always written to output/ml_results/{scope}/{transform}/):
- xgboost_model.joblib
- X_test.parquet
- y_test.parquet
- xgboost_predictions.parquet
- xgboost_metrics.parquet

For the default branch (scope=all, transform=raw) the same files are ALSO
mirrored to the legacy output/ml_results/ top level for backward compat with
other members' SHAP / figure scripts.
"""
from __future__ import annotations

import argparse
import shutil

import joblib
import numpy as np
import pandas as pd

from src.data_loader import load_feature_matrix
from src.ml_cli import BranchPaths, add_branch_args, resolve_paths
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


def _mirror_to_top_level(paths: BranchPaths, filenames: list[str]) -> None:
    """Copy branch outputs to the legacy top-level paths when applicable."""
    if not paths.also_top_level:
        return
    for name in filenames:
        src = paths.out_dir / name
        if src.exists():
            shutil.copy2(src, paths.legacy_out_dir / name)


def main(scope: str, transform: str, params: dict | None = None) -> None:
    paths = resolve_paths(scope=scope, transform=transform)

    df = load_feature_matrix()
    df = add_target(df, scope=scope)
    train, test = time_train_test_split(df)

    X_train, y_train, _ = prepare_xy(train, scope=scope)
    X_test, y_test, mask_test = prepare_xy(test, scope=scope)

    y_train_model = apply_target_transform(y_train, transform)
    model = train_xgboost(X_train, pd.Series(y_train_model), params or DEFAULT_PARAMS)

    y_pred_model = model.predict(X_test)
    y_pred = np.clip(invert_target_transform(y_pred_model, transform), 0, None)

    test_meta = test.loc[mask_test, ['country', 'week_start']].reset_index(drop=True)
    pred_df = test_meta.copy()
    pred_df['actual'] = y_test.to_numpy()
    pred_df['predicted'] = y_pred
    metrics = evaluate_predictions(pred_df)

    print(
        f"XGBoost scope={scope} transform={transform}: "
        f"rmse={metrics['rmse']:.4f} mae={metrics['mae']:.4f} "
        f"dir_acc={metrics['directional_accuracy']:.4f}"
    )
    print("Per-country dir acc:", metrics['per_country_directional'])

    joblib.dump(
        {
            'model': model,
            'feature_cols': list(X_train.columns),
            'transform': transform,
            'scope': scope,
        },
        paths.out_dir / "xgboost_model.joblib",
    )
    X_test.to_parquet(paths.out_dir / "X_test.parquet")
    y_test.to_frame(name='actual').to_parquet(paths.out_dir / "y_test.parquet")
    pred_df.to_parquet(paths.out_dir / "xgboost_predictions.parquet", index=False)

    metric_row = {
        'scope': scope,
        'transform': transform,
        'rmse': metrics['rmse'],
        'mae': metrics['mae'],
        'directional_accuracy': metrics['directional_accuracy'],
    }
    pd.DataFrame([metric_row]).to_parquet(
        paths.out_dir / "xgboost_metrics.parquet", index=False
    )

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
    return parser


if __name__ == '__main__':
    args = _build_parser().parse_args()
    main(scope=args.target_scope, transform=args.target_transform)
