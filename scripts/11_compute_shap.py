"""Compute SHAP values for the XGBoost model and save them.

Branches (CLI flags):
- ``--target-scope {all,domestic}``
- ``--target-transform {raw,log1p}``

Inputs (output/ml_results/{scope}/{transform}/):
- xgboost_model.joblib
- X_test.parquet

Outputs:
- output/ml_results/{scope}/{transform}/shap_values.npy
- figures/{scope}/fig8_shap_summary_{transform}.png
- For the default branch (scope=all, transform=raw) also mirrored to the
  legacy paths output/ml_results/shap_values.npy and figures/fig8_shap_summary.png.
"""
from __future__ import annotations

import argparse
import shutil

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.ml_cli import add_branch_args, resolve_paths
from src.viz_template import apply_style


def main(scope: str, transform: str, max_display: int = 15) -> None:
    apply_style()
    paths = resolve_paths(scope=scope, transform=transform)

    artifact = joblib.load(paths.out_dir / "xgboost_model.joblib")
    model = artifact['model']
    X_test = pd.read_parquet(paths.out_dir / "X_test.parquet")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    shap_path = paths.out_dir / "shap_values.npy"
    np.save(shap_path, shap_values)

    fig_name = f"fig8_shap_summary_{transform}.png"
    fig_path = paths.fig_dir / fig_name

    shap.summary_plot(shap_values, X_test, show=False, max_display=max_display)
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"SHAP summary saved to {fig_path}")

    if paths.also_top_level:
        shutil.copy2(shap_path, paths.legacy_out_dir / "shap_values.npy")
        shutil.copy2(fig_path, paths.legacy_fig_dir / "fig8_shap_summary.png")
        print(
            f"Mirrored to {paths.legacy_out_dir / 'shap_values.npy'} "
            f"and {paths.legacy_fig_dir / 'fig8_shap_summary.png'}"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_branch_args(parser)
    parser.add_argument('--max-display', type=int, default=15)
    return parser


if __name__ == '__main__':
    args = _build_parser().parse_args()
    main(
        scope=args.target_scope,
        transform=args.target_transform,
        max_display=args.max_display,
    )
