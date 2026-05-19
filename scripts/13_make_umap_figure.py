"""Generate Fig 3: UMAP projection of weekly event-mix vectors.

Run: python scripts/13_make_umap_figure.py
Output: figures/fig3_umap_eventmix.png
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import umap

from src.constants import COUNTRY_COLORS, COUNTRY_NAMES, FIPS_COUNTRIES
from src.data_loader import load_eventmix_weekly
from src.viz_template import apply_style, save_fig


def main() -> None:
    apply_style()
    df = load_eventmix_weekly()

    ratio_cols = [c for c in df.columns if c.startswith('ratio_')]
    X = df[ratio_cols].fillna(0).values

    reducer = umap.UMAP(n_neighbors=50, min_dist=0.5, random_state=42)
    embedding = reducer.fit_transform(X)

    df_plot = df[['country', 'week_start']].copy()
    df_plot['umap_x'] = embedding[:, 0]
    df_plot['umap_y'] = embedding[:, 1]

    fig, ax = plt.subplots(figsize=(10, 8))
    for country in FIPS_COUNTRIES:
        sub = df_plot[df_plot['country'] == country]
        ax.scatter(sub['umap_x'], sub['umap_y'],
                   color=COUNTRY_COLORS[country], label=COUNTRY_NAMES[country],
                   alpha=0.5, s=15)
    ax.set_xlabel('UMAP-1')
    ax.set_ylabel('UMAP-2')
    ax.set_title('UMAP of Weekly Event Mix (each point = one country-week)')
    ax.legend(loc='best')
    plt.tight_layout()
    save_fig(fig, 'fig3_umap_eventmix')
    plt.close(fig)
    print("Fig 3 saved.")


if __name__ == '__main__':
    main()
