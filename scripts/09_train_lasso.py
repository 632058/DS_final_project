"""Train Lasso baseline regressor on feature_matrix.

Run examples:
    python scripts/09_train_lasso.py
    python scripts/09_train_lasso.py --target-transform log1p
    python scripts/09_train_lasso.py --alpha 0.5 --target-transform raw

Output (raw transform — backward compatible top-level paths):
- output/ml_results/lasso_baseline.joblib
- output/ml_results/lasso_metrics.parquet
- output/ml_results/lasso_predictions.parquet

Output (log1p transform — suffixed paths, dual-branch infra arrives in Phase 2b):
- output/ml_results/lasso_baseline_log1p.joblib
- output/ml_results/lasso_metrics_log1p.parquet
- output/ml_results/lasso_predictions_log1p.parquet
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
    train_lasso,
)


def main(alpha: float = 1.0, transform: str = 'raw') -> None:
    df = load_feature_matrix()
    df = add_target(df)
    train, test = time_train_test_split(df)

    X_train, y_train, _ = prepare_xy(train)
    X_test, y_test, mask_test = prepare_xy(test)

    y_train_model = apply_target_transform(y_train, transform)
    model, scaler = train_lasso(X_train, pd.Series(y_train_model), alpha=alpha)

    y_pred_model = model.predict(scaler.transform(X_test))
    y_pred = np.clip(invert_target_transform(y_pred_model, transform), 0, None)

    test_meta = test.loc[mask_test, ['country', 'week_start']].reset_index(drop=True)
    pred_df = test_meta.copy()
    pred_df['actual'] = y_test.to_numpy()
    pred_df['predicted'] = y_pred
    metrics = evaluate_predictions(pred_df)

    print(
        f"Lasso alpha={alpha} transform={transform}: "
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
            'scaler': scaler,
            'feature_cols': list(X_train.columns),
            'transform': transform,
            'alpha': alpha,
        },
        out_dir / f"lasso_baseline{suffix}.joblib",
    )

    metric_row = {
        'alpha': alpha,
        'transform': transform,
        'rmse': metrics['rmse'],
        'mae': metrics['mae'],
        'directional_accuracy': metrics['directional_accuracy'],
    }
    pd.DataFrame([metric_row]).to_parquet(
        out_dir / f"lasso_metrics{suffix}.parquet", index=False
    )
    pred_df.to_parquet(
        out_dir / f"lasso_predictions{suffix}.parquet", index=False
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument(
        '--target-transform',
        choices=['raw', 'log1p'],
        default='raw',
    )
    return parser


if __name__ == '__main__':
    args = _build_parser().parse_args()
    main(alpha=args.alpha, transform=args.target_transform)
