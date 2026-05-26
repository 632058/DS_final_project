"""Generate Fig 5 (predicted vs actual scatter) and Fig 6 (residual diagnostics).

Both figures are colored by country using COUNTRY_COLORS so the Lebanon
over-prediction and Turkey under-prediction patterns are visible at a glance.

Branches (CLI flags):
- ``--target-scope {all,domestic}``
- ``--target-transform {raw,log1p}``
- ``--objective {squarederror,poisson,tweedie,quantile}``

Inputs (output/ml_results/{scope}/{transform}/):
- xgboost_predictions{_objective_suffix}.parquet
- xgboost_metrics{_objective_suffix}.parquet

Outputs:
- figures/{scope}/fig5_pred_vs_actual_{transform}{_objective_suffix}.png
- figures/{scope}/fig6_residual_plot_{transform}{_objective_suffix}.png
- figures/{scope}/fig5_per_country_{transform}{_objective_suffix}.png
- For the default branch (scope=all, transform=raw, objective=squarederror)
  also mirrored to legacy figures/fig5_pred_vs_actual.png, fig6_residual_plot.png,
  and fig5_per_country.png so other members' scripts keep working.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.constants import COUNTRY_COLORS, COUNTRY_NAMES, FIPS_COUNTRIES
from src.ml_cli import (
    add_branch_args,
    add_objective_args,
    objective_suffix,
    resolve_paths,
    validate_objective_args,
)
from src.viz_template import apply_style


def _country_legend_label(code: str, n: int, rmse: float) -> str:
    return f"{COUNTRY_NAMES[code]} ({code}) — n={n}, RMSE={rmse:.1f}"


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')


def main(
    scope: str,
    transform: str,
    objective: str,
    tweedie_variance_power: float,
    quantile_alpha: float,
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

    df = pd.read_parquet(paths.out_dir / f"xgboost_predictions{suffix}.parquet")
    metrics = pd.read_parquet(paths.out_dir / f"xgboost_metrics{suffix}.parquet").iloc[0]

    per_country: dict[str, tuple[int, float]] = {}
    for code, g in df.groupby('country'):
        rmse_c = float(np.sqrt(((g['actual'] - g['predicted']) ** 2).mean()))
        per_country[code] = (len(g), rmse_c)

    # ---------- Fig 5: Predicted vs Actual, colored by country ----------
    fig5, ax = plt.subplots(figsize=(8, 8))
    for code in FIPS_COUNTRIES:
        sub = df[df['country'] == code]
        if sub.empty:
            continue
        n, rmse_c = per_country[code]
        ax.scatter(
            sub['actual'], sub['predicted'],
            color=COUNTRY_COLORS[code],
            alpha=0.6, s=22, edgecolor='white', linewidth=0.3,
            label=_country_legend_label(code, n, rmse_c),
        )
    lim = float(max(df['actual'].max(), df['predicted'].max())) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', linewidth=1.0, label='Perfect prediction')
    ax.set_xlim(-5, lim)
    ax.set_ylim(-5, lim)
    ax.set_xlabel('Actual protest count')
    ax.set_ylabel('Predicted protest count')
    title_suffix = f" [{scope}/{transform}/{objective}]"
    ax.set_title(
        f'XGBoost: Predicted vs Actual  '
        f'(overall RMSE={metrics["rmse"]:.2f}, MAE={metrics["mae"]:.2f}){title_suffix}'
    )
    ax.legend(loc='upper left', fontsize=9)
    fig5.tight_layout()
    fig5_path = paths.fig_dir / f"fig5_pred_vs_actual_{transform}{suffix}.png"
    _save(fig5, fig5_path)
    plt.close(fig5)

    # ---------- Fig 6: Residual diagnostics, colored by country ----------
    df = df.copy()
    df['residual'] = df['actual'] - df['predicted']

    fig6, axes = plt.subplots(1, 2, figsize=(15, 6))

    for code in FIPS_COUNTRIES:
        sub = df[df['country'] == code]
        if sub.empty:
            continue
        axes[0].scatter(
            sub['predicted'], sub['residual'],
            color=COUNTRY_COLORS[code],
            alpha=0.6, s=22, edgecolor='white', linewidth=0.3,
            label=f"{COUNTRY_NAMES[code]} ({code})",
        )
    axes[0].axhline(0, color='k', linestyle='--', linewidth=1.0)
    axes[0].set_xlabel('Predicted protest count')
    axes[0].set_ylabel('Residual (actual - predicted)')
    axes[0].set_title(f'Residuals vs Predicted (by country){title_suffix}')
    axes[0].legend(loc='lower left', fontsize=9)

    bins = np.linspace(df['residual'].min(), df['residual'].max(), 35)
    residual_groups = [
        df.loc[df['country'] == code, 'residual'].values for code in FIPS_COUNTRIES
    ]
    colors = [COUNTRY_COLORS[code] for code in FIPS_COUNTRIES]
    labels = [f"{COUNTRY_NAMES[code]} ({code})" for code in FIPS_COUNTRIES]
    axes[1].hist(
        residual_groups, bins=bins, stacked=True,
        color=colors, label=labels,
        edgecolor='white', linewidth=0.3,
    )
    axes[1].axvline(0, color='k', linestyle='--', linewidth=1.0)
    axes[1].set_xlabel('Residual')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Residual distribution (stacked by country){title_suffix}')
    axes[1].legend(loc='upper right', fontsize=9)

    fig6.tight_layout()
    fig6_path = paths.fig_dir / f"fig6_residual_plot_{transform}{suffix}.png"
    _save(fig6, fig6_path)
    plt.close(fig6)

    # ---------- Fig 5b: per-country small multiples ----------
    # Each country plotted on its own axes range so TU's 197 outlier does not
    # crowd CE/AR/CI/LE into the bottom-left corner of the pooled figure.
    countries_present = [c for c in FIPS_COUNTRIES if not df[df['country'] == c].empty]
    n_panels = len(countries_present)
    fig5b, sm_axes = plt.subplots(1, n_panels, figsize=(3.4 * n_panels, 4))
    if n_panels == 1:
        sm_axes = [sm_axes]
    for ax, code in zip(sm_axes, countries_present):
        sub = df[df['country'] == code]
        n, rmse_c = per_country[code]
        local_lim = float(max(sub['actual'].max(), sub['predicted'].max())) * 1.1
        local_lim = max(local_lim, 1.0)
        ax.scatter(
            sub['actual'], sub['predicted'],
            color=COUNTRY_COLORS[code],
            alpha=0.65, s=22, edgecolor='white', linewidth=0.3,
        )
        ax.plot([0, local_lim], [0, local_lim], 'k--', linewidth=0.8)
        ax.set_xlim(-local_lim * 0.03, local_lim)
        ax.set_ylim(-local_lim * 0.03, local_lim)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(
            f"{COUNTRY_NAMES[code]} ({code})\nn={n}, RMSE={rmse_c:.1f}",
            fontsize=10,
        )
        ax.set_xlabel('Actual')
    sm_axes[0].set_ylabel('Predicted')
    fig5b.suptitle(
        f"XGBoost predictions per country  [{scope}/{transform}/{objective}]  "
        f"(overall RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f})",
        y=1.02, fontsize=12,
    )
    fig5b.tight_layout()
    fig5b_path = paths.fig_dir / f"fig5_per_country_{transform}{suffix}.png"
    _save(fig5b, fig5b_path)
    plt.close(fig5b)

    print(f"Saved {fig5_path}")
    print(f"Saved {fig6_path}")
    print(f"Saved {fig5b_path}")

    # Only the canonical (squarederror) figures get mirrored to the legacy
    # top-level paths so the report's main figures stay consistent.
    if paths.also_top_level and objective == 'squarederror':
        legacy_fig5 = paths.legacy_fig_dir / "fig5_pred_vs_actual.png"
        legacy_fig6 = paths.legacy_fig_dir / "fig6_residual_plot.png"
        legacy_fig5b = paths.legacy_fig_dir / "fig5_per_country.png"
        shutil.copy2(fig5_path, legacy_fig5)
        shutil.copy2(fig6_path, legacy_fig6)
        shutil.copy2(fig5b_path, legacy_fig5b)
        print(
            f"Mirrored to {legacy_fig5}, {legacy_fig6}, and {legacy_fig5b}"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_branch_args(parser)
    add_objective_args(parser)
    return parser


if __name__ == '__main__':
    args = _build_parser().parse_args()
    main(
        scope=args.target_scope,
        transform=args.target_transform,
        objective=args.objective,
        tweedie_variance_power=args.tweedie_variance_power,
        quantile_alpha=args.quantile_alpha,
    )
