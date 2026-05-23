"""Collate per-model metrics into a single model_comparison.parquet.

This script is the single source of truth for cross-model comparison tables.
It reads metrics written by the individual training scripts and concatenates
them in a fixed schema, so figures and the report never disagree with the
per-model parquet files.

Branches (CLI flags):
- ``--target-scope {all,domestic}``
- ``--target-transform {raw,log1p}``

Run after both training scripts have produced fresh metrics for the SAME
branch, e.g. for the log1p main line:
    python scripts/09_train_lasso.py     --target-transform log1p
    python scripts/10_train_xgboost.py   --target-transform log1p
    python scripts/10b_collate_model_comparison.py --target-transform log1p

Input metrics live in output/ml_results/{scope}/{transform}/:
- lasso_metrics.parquet
- xgboost_metrics.parquet

Output: output/ml_results/{scope}/{transform}/model_comparison.parquet
(also mirrored to the legacy top-level path for the default scope=all,
transform=raw branch).
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

from src.ml_cli import add_branch_args, resolve_paths

METRIC_COLS = ['rmse', 'mae', 'directional_accuracy']


def _load_metrics(path: Path, model_name: str) -> dict:
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


def main(scope: str, transform: str) -> None:
    paths = resolve_paths(scope=scope, transform=transform)

    lasso = _load_metrics(paths.out_dir / "lasso_metrics.parquet", "Lasso Baseline")
    xgb = _load_metrics(paths.out_dir / "xgboost_metrics.parquet", "XGBoost")

    comparison = pd.DataFrame([lasso, xgb], columns=['Model', *METRIC_COLS])
    out_path = paths.out_dir / "model_comparison.parquet"
    comparison.to_parquet(out_path, index=False)

    print(f"Wrote {out_path}")
    print(comparison.to_string(index=False))

    if paths.also_top_level:
        legacy_path = paths.legacy_out_dir / "model_comparison.parquet"
        shutil.copy2(out_path, legacy_path)
        print(f"Mirrored to {legacy_path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_branch_args(parser)
    return parser


if __name__ == '__main__':
    args = _build_parser().parse_args()
    try:
        main(scope=args.target_scope, transform=args.target_transform)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
