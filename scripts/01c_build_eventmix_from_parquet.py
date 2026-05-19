"""Build country_eventmix_weekly from A's weekly_country_relation_directed_base.parquet.

Produces a (country, week_start) matrix with 20 ratio_<root_code> columns,
where each row sums to ~1.0 — the proportion of each CAMEO root code (01..20)
in that country's outgoing events for the given week.

Used by scripts/13_make_umap_figure.py (Fig 3 UMAP projection).

Run: python scripts/01c_build_eventmix_from_parquet.py
Output: data/country_eventmix_weekly.parquet
"""
from __future__ import annotations

import pandas as pd

from src.constants import DATA_DIR

DATA_PATH = DATA_DIR / "weekly_country_relation_directed_base.parquet"

# ISO-3 → FIPS 10-4 mapping for the 5 target countries
ISO_TO_FIPS = {
    'LKA': 'CE',  # Sri Lanka
    'ARG': 'AR',  # Argentina
    'CHL': 'CI',  # Chile
    'TUR': 'TU',  # Turkey
    'LBN': 'LE',  # Lebanon
}

# Full CAMEO root code range so every country-week has all 20 columns
ROOT_CODES = [f"{i:02d}" for i in range(1, 21)]
RATIO_COLS = [f"ratio_{rc}" for rc in ROOT_CODES]


def main() -> None:
    print(f"Reading {DATA_PATH} ...")
    df = pd.read_parquet(
        DATA_PATH,
        columns=['time_period', 'country_1', 'relation_code', 'raw_event_count'],
    )

    # Keep only rows where country_1 is one of the 5 target countries
    df = df[df['country_1'].isin(ISO_TO_FIPS)].copy()
    df['country'] = df['country_1'].map(ISO_TO_FIPS)
    df['week_start'] = pd.to_datetime(df['time_period'].astype(str), format='%Y%m%d')
    df['root_code'] = df['relation_code'].str[:2]

    # Drop any rows whose root_code falls outside 01..20 (defensive)
    df = df[df['root_code'].isin(ROOT_CODES)]

    # Aggregate event counts per (country, week_start, root_code)
    counts = (
        df.groupby(['country', 'week_start', 'root_code'])['raw_event_count']
        .sum()
        .unstack('root_code', fill_value=0)
        .reindex(columns=ROOT_CODES, fill_value=0)
    )

    # Convert to ratios per (country, week_start)
    row_totals = counts.sum(axis=1).replace(0, pd.NA)
    ratios = counts.div(row_totals, axis=0).fillna(0)

    ratios.columns = RATIO_COLS
    ratios = ratios.reset_index().sort_values(['country', 'week_start']).reset_index(drop=True)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "country_eventmix_weekly.parquet"
    ratios.to_parquet(out_path, index=False)

    print(f"Wrote {out_path}: shape={ratios.shape}")
    print(ratios.groupby('country').size())
    print()
    print("Row-sum sanity (should all be ~1.0):")
    print(ratios[RATIO_COLS].sum(axis=1).describe())


if __name__ == '__main__':
    main()
