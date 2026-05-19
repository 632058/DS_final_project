"""Generate Fig 1 (tone vs protest dual-axis) and Fig 2 (tone/goldstein heatmap).

Run: python scripts/12_make_eda_figures.py
Output: figures/fig1_tone_vs_protest_per_country.png, figures/fig2_tone_goldstein_heatmap.png
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

from src.constants import COUNTRY_COLORS, COUNTRY_NAMES, FIPS_COUNTRIES
from src.data_loader import load_country_weekly
from src.viz_template import apply_style, save_fig


def make_fig1(df) -> None:
    fig, axes = plt.subplots(5, 1, figsize=(12, 18), sharex=False)
    for i, country in enumerate(FIPS_COUNTRIES):
        df_c = df[df['country'] == country].sort_values('week_start')
        ax1 = axes[i]
        ax2 = ax1.twinx()

        tone_smooth = df_c['avg_tone'].rolling(4, min_periods=1).mean()
        ax1.plot(df_c['week_start'], tone_smooth,
                 color='steelblue', linewidth=1.5, label='Avg Tone (4w MA)')
        ax1.set_ylabel('Avg Tone (media)', color='steelblue')
        ax1.tick_params(axis='y', labelcolor='steelblue')
        ax1.axhline(0, color='steelblue', linestyle=':', alpha=0.5)

        ax2.bar(df_c['week_start'], df_c['protest_count'],
                color=COUNTRY_COLORS[country], alpha=0.4, width=5,
                label='Protest count')
        ax2.set_ylabel('Protest count', color=COUNTRY_COLORS[country])
        ax2.tick_params(axis='y', labelcolor=COUNTRY_COLORS[country])
        ax2.grid(False)

        ax1.set_title(f'{COUNTRY_NAMES[country]} ({country})')
        ax1.set_xlabel('Week')

    plt.tight_layout()
    save_fig(fig, 'fig1_tone_vs_protest_per_country')
    plt.close(fig)


def _yearly_ticks(columns) -> tuple[list[int], list[str]]:
    """Pick one tick per year — the first column whose date falls in that year."""
    seen_years: set[int] = set()
    tick_idx: list[int] = []
    tick_labels: list[str] = []
    for i, ts in enumerate(columns):
        year = ts.year
        if year not in seen_years:
            seen_years.add(year)
            tick_idx.append(i)
            tick_labels.append(str(year))
    return tick_idx, tick_labels


def make_fig2(df) -> None:
    tone_pivot = df.pivot(index='country', columns='week_start',
                          values='avg_tone').reindex(FIPS_COUNTRIES)
    gold_pivot = df.pivot(index='country', columns='week_start',
                          values='avg_goldstein').reindex(FIPS_COUNTRIES)

    fig, axes = plt.subplots(2, 1, figsize=(16, 6))
    tick_idx, tick_labels = _yearly_ticks(tone_pivot.columns)

    sns.heatmap(tone_pivot, cmap='RdBu_r', center=0, vmin=-5, vmax=5,
                cbar_kws={'label': 'Avg Tone'}, ax=axes[0])
    axes[0].set_title('Avg Tone Over Time (per country)')
    axes[0].set_xlabel('')
    axes[0].set_xticks(tick_idx)
    axes[0].set_xticklabels(tick_labels, rotation=0, ha='center')
    axes[0].set_yticklabels([COUNTRY_NAMES[c] for c in tone_pivot.index], rotation=0)

    sns.heatmap(gold_pivot, cmap='RdBu_r', center=0, vmin=-5, vmax=5,
                cbar_kws={'label': 'Goldstein'}, ax=axes[1])
    axes[1].set_title('Avg Goldstein Over Time (per country)')
    axes[1].set_xlabel('Year')
    axes[1].set_xticks(tick_idx)
    axes[1].set_xticklabels(tick_labels, rotation=0, ha='center')
    axes[1].set_yticklabels([COUNTRY_NAMES[c] for c in gold_pivot.index], rotation=0)

    plt.tight_layout()
    save_fig(fig, 'fig2_tone_goldstein_heatmap')
    plt.close(fig)


def main() -> None:
    apply_style()
    df = load_country_weekly()
    make_fig1(df)
    make_fig2(df)
    print("Fig 1 and Fig 2 saved to figures/.")


if __name__ == '__main__':
    main()
