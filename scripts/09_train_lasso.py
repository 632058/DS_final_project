"""Train Lasso baseline regressor on feature_matrix.

Branches (CLI flags):
- ``--target-scope {all,domestic}``     which protest count to predict
- ``--target-transform {raw,log1p}``    forward transform on y

By default the script runs :func:`train_lasso_cv` over the alpha grid
``np.logspace(-3, 1, 9)`` with :class:`TimeSeriesSplit` (n_splits=5),
which fixes the previous degeneracy where ``alpha=1.0`` on log1p targets
collapsed every coefficient to zero. Pass ``--alpha`` to override the CV
and fall back to a fixed regularization strength (matches the legacy
single-alpha behaviour).

Run examples:
    python scripts/09_train_lasso.py
    python scripts/09_train_lasso.py --target-transform log1p
    python scripts/09_train_lasso.py --target-scope domestic --target-transform log1p
    python scripts/09_train_lasso.py --alpha 0.5     # legacy fixed-alpha mode

Outputs (always written to output/ml_results/{scope}/{transform}/):
- lasso_baseline.joblib
- lasso_metrics.parquet
- lasso_predictions.parquet

For the default branch (scope=all, transform=raw) the same files are ALSO
mirrored to the legacy output/ml_results/ top level for backward compat with
other members' scripts.
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
    DEFAULT_LASSO_CV_ALPHAS,
    DEFAULT_LASSO_CV_SPLITS,
    add_target,
    apply_target_transform,
    evaluate_predictions,
    invert_target_transform,
    prepare_xy,
    time_train_test_split,
    train_lasso,
    train_lasso_cv,
)


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
    alpha: float | None,
    cv_splits: int,
) -> None:
    paths = resolve_paths(scope=scope, transform=transform)

    df = load_feature_matrix()
    df = add_target(df, scope=scope)
    train, test = time_train_test_split(df)

    X_train, y_train, _ = prepare_xy(train, scope=scope)
    X_test, y_test, mask_test = prepare_xy(test, scope=scope)

    y_train_model = apply_target_transform(y_train, transform)

    if alpha is None:
        model, scaler = train_lasso_cv(
            X_train,
            pd.Series(y_train_model),
            alphas=DEFAULT_LASSO_CV_ALPHAS,
            n_splits=cv_splits,
        )
        selected_alpha = float(model.alpha_)
        cv_used = True
        print(
            f"LassoCV selected alpha={selected_alpha:.6f} "
            f"from grid {DEFAULT_LASSO_CV_ALPHAS.tolist()} "
            f"(TimeSeriesSplit n_splits={cv_splits})"
        )
    else:
        model, scaler = train_lasso(X_train, pd.Series(y_train_model), alpha=alpha)
        selected_alpha = float(alpha)
        cv_used = False

    n_nonzero = int((model.coef_ != 0).sum())
    n_features = int(len(model.coef_))

    y_pred_model = model.predict(scaler.transform(X_test))
    y_pred = np.clip(invert_target_transform(y_pred_model, transform), 0, None)

    test_meta = test.loc[mask_test, ['country', 'week_start']].reset_index(drop=True)
    pred_df = test_meta.copy()
    pred_df['actual'] = y_test.to_numpy()
    pred_df['predicted'] = y_pred
    metrics = evaluate_predictions(pred_df)

    print(
        f"Lasso scope={scope} transform={transform} "
        f"alpha={selected_alpha:.6f} cv={cv_used} "
        f"nonzero={n_nonzero}/{n_features}: "
        f"rmse={metrics['rmse']:.4f} mae={metrics['mae']:.4f} "
        f"dir_acc={metrics['directional_accuracy']:.4f}"
    )
    print("Per-country dir acc:", metrics['per_country_directional'])

    joblib.dump(
        {
            'model': model,
            'scaler': scaler,
            'feature_cols': list(X_train.columns),
            'transform': transform,
            'scope': scope,
            'alpha': selected_alpha,
            'cv_used': cv_used,
            'cv_alphas_grid': DEFAULT_LASSO_CV_ALPHAS.tolist() if cv_used else None,
            'cv_n_splits': cv_splits if cv_used else None,
        },
        paths.out_dir / "lasso_baseline.joblib",
    )

    metric_row = {
        'scope': scope,
        'transform': transform,
        'alpha': selected_alpha,
        'cv_used': cv_used,
        'n_nonzero_coefs': n_nonzero,
        'n_features': n_features,
        'rmse': metrics['rmse'],
        'mae': metrics['mae'],
        'directional_accuracy': metrics['directional_accuracy'],
    }
    pd.DataFrame([metric_row]).to_parquet(
        paths.out_dir / "lasso_metrics.parquet", index=False
    )
    pred_df.to_parquet(paths.out_dir / "lasso_predictions.parquet", index=False)

    _mirror_to_top_level(paths, [
        "lasso_baseline.joblib",
        "lasso_metrics.parquet",
        "lasso_predictions.parquet",
    ])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--alpha',
        type=float,
        default=None,
        help=(
            "Fixed regularization strength. If omitted (default), runs "
            "LassoCV over np.logspace(-3, 1, 9) with TimeSeriesSplit."
        ),
    )
    parser.add_argument(
        '--cv-splits',
        type=int,
        default=DEFAULT_LASSO_CV_SPLITS,
        help="TimeSeriesSplit n_splits for LassoCV (ignored when --alpha is set).",
    )
    add_branch_args(parser)
    return parser


if __name__ == '__main__':
    args = _build_parser().parse_args()
    main(
        scope=args.target_scope,
        transform=args.target_transform,
        alpha=args.alpha,
        cv_splits=args.cv_splits,
    )
