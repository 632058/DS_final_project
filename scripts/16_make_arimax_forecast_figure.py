"""Generate Fig 7: ARIMAX forecast vs actual per country.

Run: python scripts/16_make_arimax_forecast_figure.py
Output: figures/fig7_arimax_forecast.png
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from src.constants import COUNTRY_NAMES, FIPS_COUNTRIES, OUTPUT_DIR
from src.viz_template import apply_style, save_fig


def main() -> None:
    apply_style()

    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=False)
    for i, country in enumerate(FIPS_COUNTRIES):
        path = OUTPUT_DIR / "arimax_results" / f"{country}_forecast.parquet"
        if not path.exists():
            axes[i].text(0.5, 0.5, f'{country}: forecast not found',
                         ha='center', va='center', transform=axes[i].transAxes)
            continue
        pred = pd.read_parquet(path)
        ax = axes[i]
        ax.plot(pred['week_start'], pred['actual'], label='Actual', color='steelblue')
        ax.plot(pred['week_start'], pred['predicted'], label='Predicted', color='crimson')
        ax.fill_between(pred['week_start'], pred['ci_lower'], pred['ci_upper'],
                        color='crimson', alpha=0.2, label='95% CI')
        ax.set_title(f'{COUNTRY_NAMES[country]} ({country}) ARIMAX Forecast')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig(fig, 'fig7_arimax_forecast')
    plt.close(fig)
    print("Fig 7 saved.")


if __name__ == '__main__':
    main()
