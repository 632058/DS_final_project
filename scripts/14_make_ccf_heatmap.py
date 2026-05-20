"""Generate CCF heatmaps for domestic protests and international conflicts.

Outputs (figures/):
  fig4a_ccf_tone_protest.png       avg_tone → protest (domestic)
  fig4b_ccf_goldstein_protest.png  avg_goldstein → protest (domestic)
  fig4c_ccf_tone_conflict.png      avg_tone → escalation (international)
  fig4d_ccf_goldstein_conflict.png avg_goldstein → escalation (international)

Run: python scripts/14_make_ccf_heatmap.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.constants import COUNTRY_NAMES, FIPS_COUNTRIES, OUTPUT_DIR
from src.viz_template import apply_style, save_fig

CONFLICT_ORDER = ['RUS↔UKR', 'ISR↔PSE', 'ARM↔AZE', 'IND↔PAK', 'USA↔IRN']


def make_domestic_heatmap(x_var: str, fig_name: str, title: str) -> None:
    df = pd.read_parquet(OUTPUT_DIR / 'ccf_granger' / f'ccf_{x_var}_to_protest.parquet')
    df = df.reindex(FIPS_COUNTRIES)
    df.index = [COUNTRY_NAMES[c] for c in df.index]
    df.columns = [c.replace('lag_', '') for c in df.columns]

    fig, ax = plt.subplots(figsize=(11, 4))
    sns.heatmap(
        df,
        cmap='RdBu_r', center=0, vmin=-0.7, vmax=0.7,
        annot=True, fmt='.2f',
        cbar_kws={'label': 'Cross-correlation'},
        ax=ax,
    )
    ax.set_xlabel('Lag (weeks): X leads protest_count by k weeks')
    ax.set_ylabel('Country')
    ax.set_title(title)
    plt.tight_layout()
    save_fig(fig, fig_name)
    plt.close(fig)
    print(f"Saved: figures/{fig_name}.png")


def make_conflict_heatmap(x_var: str, fig_name: str, title: str) -> None:
    df_raw = pd.read_parquet(OUTPUT_DIR / 'ccf_granger' / 'conflict_ccf_results.parquet')
    df_raw = df_raw[df_raw['x_var'] == x_var].copy()
    pivot = df_raw.pivot(index='short', columns='lag', values='r').reindex(CONFLICT_ORDER)
    pivot.columns = [str(c) for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(11, 4))
    sns.heatmap(
        pivot,
        cmap='RdBu_r', center=0, vmin=-0.7, vmax=0.7,
        annot=True, fmt='.2f',
        cbar_kws={'label': 'Cross-correlation'},
        ax=ax,
    )
    ax.set_xlabel('Lag (weeks): X leads escalation_count (code>15) by k weeks')
    ax.set_ylabel('Conflict pair')
    ax.set_title(title)
    plt.tight_layout()
    save_fig(fig, fig_name)
    plt.close(fig)
    print(f"Saved: figures/{fig_name}.png")


def main() -> None:
    apply_style()

    make_domestic_heatmap(
        'avg_tone', 'fig4a_ccf_tone_protest',
        'CCF: avg_tone(t−k) vs protest_count(t)  [52w pre + 12w post outbreak]',
    )
    make_domestic_heatmap(
        'avg_goldstein', 'fig4b_ccf_goldstein_protest',
        'CCF: avg_goldstein(t−k) vs protest_count(t)  [52w pre + 12w post outbreak]',
    )
    make_conflict_heatmap(
        'avg_tone', 'fig4c_ccf_tone_conflict',
        'CCF: avg_tone(t−k) vs escalation_count(t)  [52w pre + 12w post outbreak]',
    )
    make_conflict_heatmap(
        'avg_goldstein', 'fig4d_ccf_goldstein_conflict',
        'CCF: avg_goldstein(t−k) vs escalation_count(t)  [52w pre + 12w post outbreak]',
    )


if __name__ == '__main__':
    main()
