"""Build country_weekly aggregation from the directed-relation parquet source.

Source: data/weekly_country_relation_directed_base.parquet
  Schema: (time_period, country_1, country_2, relation_code, raw_event_count,
           sum_mentions, avg_tone, avg_goldstein, sample_quad_class, ...)
  country codes: ISO-3 (LKA, ARG, CHL, TUR, LBN, ...)

Output: output/country_weekly.parquet (FIPS 10-4 country codes)

Scope conventions (must stay consistent with the rest of the pipeline):
- Most aggregates use the "source country = target" scope (country_1 IN targets).
  This matches what `01b_build_country_weekly_from_parquet.py` produces and what
  downstream ML / feature scripts expect.
- protest_count_all : country_1 = target               (all-scope protest)
- protest_count     : country_1 = country_2 = target   (domestic-only protest)

ML pipeline currently uses protest_count_all as the prediction target; the
domestic-only protest_count is retained for backwards compatibility with
member C/D/F scripts that consume the legacy column name.

Run: python scripts/01_build_country_weekly.py
"""
from __future__ import annotations

import pandas as pd

from src.constants import DATA_DIR, OUTPUT_DIR

SOURCE_PATH = DATA_DIR / "weekly_country_relation_directed_base.parquet"

# ISO-3 -> FIPS 10-4 for the 5 target countries
ISO_TO_FIPS = {
    'LKA': 'CE',  # Sri Lanka
    'ARG': 'AR',  # Argentina
    'CHL': 'CI',  # Chile
    'TUR': 'TU',  # Turkey
    'LBN': 'LE',  # Lebanon
}


def _qc_sum(df: pd.DataFrame, quad_class: int) -> pd.Series:
    """Sum raw_event_count for rows whose sample_quad_class matches."""
    return (
        df[df['sample_quad_class'] == quad_class]
        .groupby(['country', 'week_start'])['raw_event_count']
        .sum()
    )


def _root_sum(df: pd.DataFrame, roots: list[str]) -> pd.Series:
    """Sum raw_event_count for rows whose CAMEO root code is in `roots`."""
    return (
        df[df['root_code'].isin(roots)]
        .groupby(['country', 'week_start'])['raw_event_count']
        .sum()
    )


def main() -> None:
    print(f"Reading {SOURCE_PATH} ...")
    raw = pd.read_parquet(SOURCE_PATH)

    # Keep target countries (as either source OR target side)
    # We need both `country_1 == target` rows (for most aggregates) and
    # `country_1 == country_2 == target` rows (for domestic protest only).
    iso_targets = list(ISO_TO_FIPS)
    src = raw[raw['country_1'].isin(iso_targets)].copy()
    src['country'] = src['country_1'].map(ISO_TO_FIPS)
    src['week_start'] = pd.to_datetime(
        src['time_period'].astype(str), format='%Y%m%d'
    )
    src['root_code'] = src['relation_code'].str[:2]
    src['tone_x_mentions'] = src['avg_tone'] * src['sum_mentions'].fillna(0)
    src['goldstein_x_mentions'] = src['avg_goldstein'] * src['sum_mentions'].fillna(0)

    grp = src.groupby(['country', 'week_start'])

    agg = pd.DataFrame({
        'n_events': grp['raw_event_count'].sum(),
        'n_mentions_total': grp['sum_mentions'].sum(),
        'avg_tone': grp['avg_tone'].mean(),
        'avg_goldstein': grp['avg_goldstein'].mean(),
        'weighted_tone': (
            grp['tone_x_mentions'].sum() / grp['sum_mentions'].sum()
        ),
        'weighted_goldstein': (
            grp['goldstein_x_mentions'].sum() / grp['sum_mentions'].sum()
        ),
    })

    # QuadClass-based conflict / cooperation counts (sample_quad_class is a
    # per-relation_code representative; this is the same convention used by
    # scripts/01b_build_country_weekly_from_parquet.py).
    agg['n_verbal_coop']   = _qc_sum(src, 1)
    agg['n_material_coop'] = _qc_sum(src, 2)
    agg['n_verbal_conf']   = _qc_sum(src, 3)
    agg['n_material_conf'] = _qc_sum(src, 4)

    # CAMEO root code aggregates.
    # protest_count_all uses the same `country_1 = target` scope as everything
    # else; it is the ML pipeline's primary target.
    agg['protest_count_all']   = _root_sum(src, ['14'])
    agg['violence_count']      = _root_sum(src, ['18', '19', '20'])
    agg['verbal_threat_count'] = _root_sum(src, ['10', '11'])

    # Domestic-only protest_count needs the stricter `country_1 == country_2`
    # filter, so it must be computed from `raw` (not from `src`, which only
    # keeps country_1 in targets but allows country_2 to be anything).
    domestic = raw[
        (raw['country_1'].isin(iso_targets))
        & (raw['country_2'] == raw['country_1'])
        & (raw['relation_code'].str[:2] == '14')
    ].copy()
    domestic['country'] = domestic['country_1'].map(ISO_TO_FIPS)
    domestic['week_start'] = pd.to_datetime(
        domestic['time_period'].astype(str), format='%Y%m%d'
    )
    agg['protest_count'] = (
        domestic.groupby(['country', 'week_start'])['raw_event_count'].sum()
    )

    out = agg.fillna(0).reset_index()
    out = out.sort_values(['country', 'week_start']).reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "country_weekly.parquet"
    out.to_parquet(out_path, index=False)

    print(f"Wrote {out_path}: shape={out.shape}")
    print()
    print("Rows per country:")
    print(out.groupby('country').size())
    print()
    print("Protest count totals (domestic vs all):")
    print(out.groupby('country')[['protest_count', 'protest_count_all']].sum())


if __name__ == '__main__':
    main()
