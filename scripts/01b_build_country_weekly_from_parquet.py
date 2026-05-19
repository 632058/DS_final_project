"""Build country_weekly from A's weekly_country_relation_directed_base.parquet.

Replaces scripts/01_build_country_weekly.py when the raw DuckDB is unavailable.
A's data uses ISO-3 country codes and country-pair × relation_code granularity;
this script aggregates it to (country, week_start) with the same schema that
src/feature_engineering.py expects.

Run: python scripts/01b_build_country_weekly_from_parquet.py
Output: data/country_weekly.parquet
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


def main() -> None:
    print(f"Reading {DATA_PATH} ...")
    df = pd.read_parquet(DATA_PATH)

    # Keep only rows where country_1 is one of the 5 target countries
    df = df[df['country_1'].isin(ISO_TO_FIPS)].copy()
    df['country'] = df['country_1'].map(ISO_TO_FIPS)

    # Convert time_period (YYYYMMDD int) to week_start date
    df['week_start'] = pd.to_datetime(df['time_period'].astype(str), format='%Y%m%d')

    # Derive root code (first 2 chars of relation_code) for event-type counts
    df['root_code'] = df['relation_code'].str[:2]

    # Pre-compute mention-weighted tone / goldstein numerators
    df['tone_x_mentions'] = df['avg_tone'] * df['sum_mentions'].fillna(0)
    df['gstein_x_mentions'] = df['avg_goldstein'] * df['sum_mentions'].fillna(0)

    grp = df.groupby(['country', 'week_start'])

    agg = pd.DataFrame({
        'n_events': grp['raw_event_count'].sum(),
        'n_mentions_total': grp['sum_mentions'].sum(),
        'avg_tone': grp['avg_tone'].mean(),
        'avg_goldstein': grp['avg_goldstein'].mean(),
        'weighted_tone': grp['tone_x_mentions'].sum() / grp['sum_mentions'].sum(),
        'weighted_goldstein': grp['gstein_x_mentions'].sum() / grp['sum_mentions'].sum(),
    })

    # QuadClass-based counts (use raw_event_count as weight)
    for qc, col in [(1, 'n_verbal_coop'), (2, 'n_material_coop'),
                    (3, 'n_verbal_conf'), (4, 'n_material_conf')]:
        agg[col] = df[df['sample_quad_class'] == qc].groupby(
            ['country', 'week_start']
        )['raw_event_count'].sum()

    # CAMEO root-code-based event counts
    agg['protest_count'] = df[df['root_code'] == '14'].groupby(
        ['country', 'week_start']
    )['raw_event_count'].sum()

    agg['violence_count'] = df[df['root_code'].isin(['18', '19', '20'])].groupby(
        ['country', 'week_start']
    )['raw_event_count'].sum()

    agg['verbal_threat_count'] = df[df['root_code'].isin(['10', '11'])].groupby(
        ['country', 'week_start']
    )['raw_event_count'].sum()

    agg = agg.fillna(0).reset_index()
    agg = agg.sort_values(['country', 'week_start']).reset_index(drop=True)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "country_weekly.parquet"
    agg.to_parquet(out_path, index=False)

    print(f"Wrote {out_path}: shape={agg.shape}")
    print(agg.groupby('country').size())
    print(agg.dtypes)


if __name__ == '__main__':
    main()
