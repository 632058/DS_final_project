"""Generate Fig 5 (predicted vs actual scatter) and Fig 6 (residual diagnostics).

Both figures are colored by country using COUNTRY_COLORS so the Lebanon
over-prediction and Turkey under-prediction patterns are visible at a glance.

Run: python scripts/15_make_ml_diagnostic_figures.py
Output: figures/fig5_pred_vs_actual.png, figures/fig6_residual_plot.png
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.constants import COUNTRY_COLORS, COUNTRY_NAMES, FIPS_COUNTRIES, OUTPUT_DIR
from src.viz_template import apply_style, save_fig


def _country_legend_label(code: str, n: int, rmse: float) -> str:
    return f"{COUNTRY_NAMES[code]} ({code}) — n={n}, RMSE={rmse:.1f}"


def main() -> None:
    apply_style()
    df = pd.read_parquet(OUTPUT_DIR / "ml_results" / "xgboost_predictions.parquet")
    metrics = pd.read_parquet(OUTPUT_DIR / "ml_results" / "xgboost_metrics.parquet").iloc[0]

    # Per-country RMSE for legend annotations
    per_country: dict[str, tuple[int, float]] = {}
    for code, g in df.groupby('country'):
        rmse_c = float(np.sqrt(((g['actual'] - g['predicted']) ** 2).mean()))
        per_country[code] = (len(g), rmse_c)

    # ---------- Fig 5: Predicted vs Actual, colored by country ----------
    fig, ax = plt.subplots(figsize=(8, 8))
    for code in FIPS_COUNTRIES:
        sub = df[df['country'] == code]
        if sub.empty:
            continue
        n, rmse_c = per_country[code]
        ax.scatter(
            sub['actual'],
            sub['predicted'],
            color=COUNTRY_COLORS[code],
            alpha=0.6,
            s=22,
            edgecolor='white',
            linewidth=0.3,
            label=_country_legend_label(code, n, rmse_c),
        )

    lim = float(max(df['actual'].max(), df['predicted'].max())) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', linewidth=1.0, label='Perfect prediction')
    ax.set_xlim(-5, lim)
    ax.set_ylim(-5, lim)
    ax.set_xlabel('Actual protest count')
    ax.set_ylabel('Predicted protest count')
    ax.set_title(
        f'XGBoost: Predicted vs Actual  (overall RMSE={metrics["rmse"]:.2f}, '
        f'MAE={metrics["mae"]:.2f})'
    )
    ax.legend(loc='upper left', fontsize=9)
    plt.tight_layout()
    save_fig(fig, 'fig5_pred_vs_actual')
    plt.close(fig)

    # ---------- Fig 6: Residual diagnostics, colored by country ----------
    df = df.copy()
    df['residual'] = df['actual'] - df['predicted']

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Left: residuals vs predicted, colored by country
    for code in FIPS_COUNTRIES:
        sub = df[df['country'] == code]
        if sub.empty:
            continue
        axes[0].scatter(
            sub['predicted'],
            sub['residual'],
            color=COUNTRY_COLORS[code],
            alpha=0.6,
            s=22,
            edgecolor='white',
            linewidth=0.3,
            label=f"{COUNTRY_NAMES[code]} ({code})",
        )
    axes[0].axhline(0, color='k', linestyle='--', linewidth=1.0)
    axes[0].set_xlabel('Predicted protest count')
    axes[0].set_ylabel('Residual (actual - predicted)')
    axes[0].set_title('Residuals vs Predicted (by country)')
    axes[0].legend(loc='lower left', fontsize=9)

    # Right: stacked histogram of residuals per country
    bins = np.linspace(df['residual'].min(), df['residual'].max(), 35)
    residual_groups = [
        df.loc[df['country'] == code, 'residual'].values for code in FIPS_COUNTRIES
    ]
    colors = [COUNTRY_COLORS[code] for code in FIPS_COUNTRIES]
    labels = [f"{COUNTRY_NAMES[code]} ({code})" for code in FIPS_COUNTRIES]
    axes[1].hist(
        residual_groups,
        bins=bins,
        stacked=True,
        color=colors,
        label=labels,
        edgecolor='white',
        linewidth=0.3,
    )
    axes[1].axvline(0, color='k', linestyle='--', linewidth=1.0)
    axes[1].set_xlabel('Residual')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Residual distribution (stacked by country)')
    axes[1].legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    save_fig(fig, 'fig6_residual_plot')
    plt.close(fig)
    print("Fig 5 and Fig 6 saved.")


if __name__ == '__main__':
    main()
