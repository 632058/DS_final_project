"""Train XGBoost main model on feature_matrix and persist artifacts for SHAP.

Run examples:
    python scripts/10_train_xgboost.py
    python scripts/10_train_xgboost.py --target-transform log1p

Output (raw transform — backward compatible top-level paths):
- output/ml_results/xgboost_model.joblib
- output/ml_results/X_test.parquet
- output/ml_results/y_test.parquet
- output/ml_results/xgboost_predictions.parquet
- output/ml_results/xgboost_metrics.parquet

Output (log1p transform — suffixed paths):
- output/ml_results/xgboost_model_log1p.joblib
- output/ml_results/X_test_log1p.parquet
- output/ml_results/y_test_log1p.parquet
- output/ml_results/xgboost_predictions_log1p.parquet
- output/ml_results/xgboost_metrics_log1p.parquet
"""
from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd

from src.constants import OUTPUT_DIR
from src.data_loader import load_feature_matrix
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


def main(params: dict | None = None, transform: str = 'raw') -> None:
    df = load_feature_matrix()
    df = add_target(df)
    train, test = time_train_test_split(df)

    X_train, y_train, _ = prepare_xy(train)
    X_test, y_test, mask_test = prepare_xy(test)

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
        f"XGBoost transform={transform}: "
        f"rmse={metrics['rmse']:.4f} mae={metrics['mae']:.4f} "
        f"dir_acc={metrics['directional_accuracy']:.4f}"
    )
    print("Per-country dir acc:", metrics['per_country_directional'])

    suffix = '' if transform == 'raw' else f'_{transform}'
    out_dir = OUTPUT_DIR / "ml_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            'model': model,
            'feature_cols': list(X_train.columns),
            'transform': transform,
        },
        out_dir / f"xgboost_model{suffix}.joblib",
    )
    X_test.to_parquet(out_dir / f"X_test{suffix}.parquet")
    y_test.to_frame(name='actual').to_parquet(out_dir / f"y_test{suffix}.parquet")
    pred_df.to_parquet(
        out_dir / f"xgboost_predictions{suffix}.parquet", index=False
    )

    metric_row = {
        'transform': transform,
        'rmse': metrics['rmse'],
        'mae': metrics['mae'],
        'directional_accuracy': metrics['directional_accuracy'],
    }
    pd.DataFrame([metric_row]).to_parquet(
        out_dir / f"xgboost_metrics{suffix}.parquet", index=False
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--target-transform',
        choices=['raw', 'log1p'],
        default='raw',
    )
    return parser


if __name__ == '__main__':
    args = _build_parser().parse_args()
    main(transform=args.target_transform)
