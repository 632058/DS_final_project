"""Train XGBoost main model on feature_matrix and persist artifacts for SHAP.

Run: python scripts/10_train_xgboost.py
Output:
- output/ml_results/xgboost_model.joblib
- output/ml_results/X_test.parquet, y_test.parquet
- output/ml_results/xgboost_predictions.parquet
- output/ml_results/xgboost_metrics.parquet
"""
from __future__ import annotations

import joblib
import pandas as pd

from src.constants import OUTPUT_DIR
from src.data_loader import load_feature_matrix
from src.ml_models import (
    add_target,
    evaluate,
    prepare_xy,
    time_train_test_split,
    train_xgboost,
)

DEFAULT_PARAMS = {
    'n_estimators': 300,
    'max_depth': 5,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
}


def main(params: dict | None = None) -> None:
    df = load_feature_matrix()
    df = add_target(df)
    train, test = time_train_test_split(df)

    X_train, y_train, _ = prepare_xy(train)
    X_test, y_test, mask_test = prepare_xy(test)

    model = train_xgboost(X_train, y_train, params or DEFAULT_PARAMS)
    y_pred = model.predict(X_test)

    metrics = evaluate(y_test, y_pred)
    print(f"XGBoost test: {metrics}")

    out_dir = OUTPUT_DIR / "ml_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {'model': model, 'feature_cols': list(X_train.columns)},
        out_dir / "xgboost_model.joblib",
    )

    X_test.to_parquet(out_dir / "X_test.parquet")
    y_test.to_frame(name='actual').to_parquet(out_dir / "y_test.parquet")

    test_meta = test.loc[mask_test, ['country', 'week_start']].reset_index(drop=True)
    pred_df = test_meta.copy()
    pred_df['actual'] = y_test.values
    pred_df['predicted'] = y_pred
    pred_df.to_parquet(out_dir / "xgboost_predictions.parquet", index=False)

    pd.DataFrame([metrics]).to_parquet(out_dir / "xgboost_metrics.parquet", index=False)


if __name__ == '__main__':
    main()
