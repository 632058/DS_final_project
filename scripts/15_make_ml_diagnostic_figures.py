"""Generate Fig 5 (predicted vs actual scatter) and Fig 6 (residual diagnostics).

Run: python scripts/15_make_ml_diagnostic_figures.py
Output: figures/fig5_pred_vs_actual.png, figures/fig6_residual_plot.png
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from src.constants import OUTPUT_DIR
from src.viz_template import apply_style, save_fig


def main() -> None:
    apply_style()
    df = pd.read_parquet(OUTPUT_DIR / "ml_results" / "xgboost_predictions.parquet")
    metrics = pd.read_parquet(OUTPUT_DIR / "ml_results" / "xgboost_metrics.parquet").iloc[0]

    # Fig 5
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(df['actual'], df['predicted'], alpha=0.5, s=20)
    lim = max(df['actual'].max(), df['predicted'].max())
    ax.plot([0, lim], [0, lim], 'k--', label='Perfect prediction')
    ax.set_xlabel('Actual protest count')
    ax.set_ylabel('Predicted protest count')
    ax.set_title(f'XGBoost: Predicted vs Actual (RMSE={metrics["rmse"]:.2f})')
    ax.legend()
    plt.tight_layout()
    save_fig(fig, 'fig5_pred_vs_actual')
    plt.close(fig)

    # Fig 6
    residuals = df['actual'].values - df['predicted'].values
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(df['predicted'], residuals, alpha=0.5, s=20)
    axes[0].axhline(0, color='k', linestyle='--')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Residual (actual - predicted)')
    axes[0].set_title('Residuals vs Predicted')

    axes[1].hist(residuals, bins=30, edgecolor='black')
    axes[1].set_xlabel('Residual')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Residual Distribution')
    plt.tight_layout()
    save_fig(fig, 'fig6_residual_plot')
    plt.close(fig)
    print("Fig 5 and Fig 6 saved.")


if __name__ == '__main__':
    main()
