"""Generate Fig 4: CCF heatmap (avg_tone -> protest, lag 0..12 weeks).

Run: python scripts/14_make_ccf_heatmap.py
Output: figures/fig4_ccf_heatmap.png
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.constants import COUNTRY_NAMES, FIPS_COUNTRIES, OUTPUT_DIR
from src.viz_template import apply_style, save_fig


def main() -> None:
    apply_style()
    df_ccf = pd.read_parquet(OUTPUT_DIR / "ccf_granger" / "ccf_avg_tone_to_protest.parquet")
    df_ccf = df_ccf.reindex(FIPS_COUNTRIES)
    df_ccf.index = [COUNTRY_NAMES[c] for c in df_ccf.index]
    df_ccf.columns = [c.replace('lag_', '') for c in df_ccf.columns]

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(
        df_ccf,
        cmap='RdBu_r',
        center=0,
        vmin=-0.5, vmax=0.5,
        annot=True, fmt='.2f',
        cbar_kws={'label': 'Cross-correlation'},
        ax=ax,
    )
    ax.set_xlabel('Lag (weeks): tone leads protest by k weeks')
    ax.set_ylabel('Country')
    ax.set_title('CCF: avg_tone(t-k) vs protest_count(t)')
    plt.tight_layout()
    save_fig(fig, 'fig4_ccf_heatmap')
    plt.close(fig)
    print("Fig 4 saved.")


if __name__ == '__main__':
    main()
