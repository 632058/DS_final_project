"""Train Lasso baseline regressor on feature_matrix.

Run: python scripts/09_train_lasso.py
Output:
- output/ml_results/lasso_baseline.joblib
- output/ml_results/lasso_metrics.parquet
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
    train_lasso,
)


def main(alpha: float = 1.0) -> None:
    df = load_feature_matrix()
    df = add_target(df)
    train, test = time_train_test_split(df)

    X_train, y_train, _ = prepare_xy(train)
    X_test, y_test, _ = prepare_xy(test)

    model, scaler = train_lasso(X_train, y_train, alpha=alpha)
    y_pred = model.predict(scaler.transform(X_test))
    metrics = evaluate(y_test, y_pred)
    print(f"Lasso alpha={alpha}: {metrics}")

    out_dir = OUTPUT_DIR / "ml_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': model, 'scaler': scaler, 'feature_cols': list(X_train.columns)},
                out_dir / "lasso_baseline.joblib")

    pd.DataFrame([{'alpha': alpha, **metrics}]).to_parquet(
        out_dir / "lasso_metrics.parquet", index=False
    )


if __name__ == '__main__':
    main()
