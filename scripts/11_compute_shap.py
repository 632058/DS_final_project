"""Compute SHAP values for the XGBoost model and save them.

Branches (CLI flags):
- ``--target-scope {all,domestic}``
- ``--target-transform {raw,log1p}``
- ``--objective {squarederror,poisson,tweedie,quantile}``

Inputs (output/ml_results/{scope}/{transform}/):
- xgboost_model{_objective_suffix}.joblib
- X_test{_objective_suffix}.parquet

Outputs:
- output/ml_results/{scope}/{transform}/shap_values{_objective_suffix}.npy
- figures/{scope}/fig8_shap_summary_{transform}{_objective_suffix}.png
- For the default branch (scope=all, transform=raw, objective=squarederror)
  also mirrored to the legacy paths output/ml_results/shap_values.npy and
  figures/fig8_shap_summary.png so other members' scripts keep working.
"""
from __future__ import annotations

import argparse
import shutil

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.ml_cli import (
    add_branch_args,
    add_objective_args,
    objective_suffix,
    resolve_paths,
    validate_objective_args,
)
from src.viz_template import apply_style


def main(
    scope: str,
    transform: str,
    objective: str,
    tweedie_variance_power: float,
    quantile_alpha: float,
    max_display: int = 15,
) -> None:
    validate_objective_args(
        objective, transform, tweedie_variance_power, quantile_alpha
    )
    apply_style()
    paths = resolve_paths(scope=scope, transform=transform)
    suffix = objective_suffix(
        objective,
        tweedie_variance_power=tweedie_variance_power,
        quantile_alpha=quantile_alpha,
    )

    artifact = joblib.load(paths.out_dir / f"xgboost_model{suffix}.joblib")
    model = artifact['model']
    X_test = pd.read_parquet(paths.out_dir / f"X_test{suffix}.parquet")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    shap_path = paths.out_dir / f"shap_values{suffix}.npy"
    np.save(shap_path, shap_values)

    fig_name = f"fig8_shap_summary_{transform}{suffix}.png"
    fig_path = paths.fig_dir / fig_name

    shap.summary_plot(shap_values, X_test, show=False, max_display=max_display)
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"SHAP summary saved to {fig_path}")

    # Only the canonical (squarederror) artefact gets mirrored to legacy paths
    # so downstream report scripts always pull a consistent SHAP picture.
    if paths.also_top_level and objective == 'squarederror':
        shutil.copy2(shap_path, paths.legacy_out_dir / "shap_values.npy")
        shutil.copy2(fig_path, paths.legacy_fig_dir / "fig8_shap_summary.png")
        print(
            f"Mirrored to {paths.legacy_out_dir / 'shap_values.npy'} "
            f"and {paths.legacy_fig_dir / 'fig8_shap_summary.png'}"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_branch_args(parser)
    add_objective_args(parser)
    parser.add_argument('--max-display', type=int, default=15)
    return parser


if __name__ == '__main__':
    args = _build_parser().parse_args()
    main(
        scope=args.target_scope,
        transform=args.target_transform,
        objective=args.objective,
        tweedie_variance_power=args.tweedie_variance_power,
        quantile_alpha=args.quantile_alpha,
        max_display=args.max_display,
    )
