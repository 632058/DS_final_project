"""Build country_eventmix_weekly: 20-dim root code ratio vector per country-week.

Run: python scripts/03_build_country_eventmix_weekly.py
Output: output/country_eventmix_weekly.parquet
"""
from __future__ import annotations

import duckdb

from src.constants import DUCKDB_PATH, FIPS_COUNTRIES, OUTPUT_DIR

EVENTMIX_SQL = """
WITH base AS (
    SELECT
        ActionGeo_CountryCode AS country,
        DATE_TRUNC('week',
            strptime(LEFT(CAST(DATEADDED AS VARCHAR), 8), '%Y%m%d')
        ) AS week_start,
        EventRootCode,
        COUNT(*) AS n
    FROM gdelt_events
    WHERE ActionGeo_CountryCode IN ({countries})
      AND EventRootCode IS NOT NULL
    GROUP BY country, week_start, EventRootCode
),
totals AS (
    SELECT country, week_start, SUM(n) AS total
    FROM base
    GROUP BY country, week_start
)
SELECT
    b.country,
    b.week_start,
    b.EventRootCode AS root_code,
    b.n AS root_count,
    b.n * 1.0 / t.total AS ratio
FROM base b
JOIN totals t USING (country, week_start)
"""


def main() -> None:
    countries_in = ','.join(f"'{c}'" for c in FIPS_COUNTRIES)
    con = duckdb.connect(str(DUCKDB_PATH))
    df_long = con.sql(EVENTMIX_SQL.format(countries=countries_in)).df()
    con.close()

    df_wide = df_long.pivot_table(
        index=['country', 'week_start'],
        columns='root_code',
        values='ratio',
        fill_value=0,
    ).reset_index()

    rename = {c: f'ratio_{c}' for c in df_wide.columns
              if c not in ('country', 'week_start')}
    df_wide = df_wide.rename(columns=rename)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "country_eventmix_weekly.parquet"
    df_wide.to_parquet(out_path, index=False)

    print(f"Wrote {out_path}: shape={df_wide.shape}")


if __name__ == '__main__':
    main()
