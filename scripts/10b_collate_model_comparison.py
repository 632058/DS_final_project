"""Collate per-model metrics into a single model_comparison.parquet.

This script is the single source of truth for cross-model comparison tables.
It reads metrics written by the individual training scripts and concatenates
them in a fixed schema, so figures and the report never disagree with the
per-model parquet files.

Run after both training scripts have produced fresh metrics:
    python scripts/09_train_lasso.py
    python scripts/10_train_xgboost.py
    python scripts/10b_collate_model_comparison.py

Inputs:
- output/ml_results/lasso_metrics.parquet
- output/ml_results/xgboost_metrics.parquet

Output:
- output/ml_results/model_comparison.parquet
"""
from __future__ import annotations

import sys

import pandas as pd

from src.constants import OUTPUT_DIR

METRIC_COLS = ['rmse', 'mae', 'directional_accuracy']


def _load_metrics(path, model_name: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing metrics file: {path}. "
            f"Run the corresponding training script first."
        )
    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError(f"{path} is empty")
    row = df.iloc[0]
    missing = [c for c in METRIC_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return {'Model': model_name, **{c: float(row[c]) for c in METRIC_COLS}}


def main() -> None:
    ml_dir = OUTPUT_DIR / "ml_results"
    lasso = _load_metrics(ml_dir / "lasso_metrics.parquet", "Lasso Baseline")
    xgb = _load_metrics(ml_dir / "xgboost_metrics.parquet", "XGBoost")

    comparison = pd.DataFrame([lasso, xgb], columns=['Model', *METRIC_COLS])
    out_path = ml_dir / "model_comparison.parquet"
    comparison.to_parquet(out_path, index=False)

    print(f"Wrote {out_path}")
    print(comparison.to_string(index=False))


if __name__ == '__main__':
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
