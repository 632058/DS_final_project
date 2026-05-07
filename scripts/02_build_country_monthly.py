"""Build country_monthly aggregation table from gdelt_events.

Run: python scripts/02_build_country_monthly.py
Output: output/country_monthly.parquet
"""
from __future__ import annotations

import duckdb

from src.constants import DUCKDB_PATH, FIPS_COUNTRIES, OUTPUT_DIR

MONTHLY_SQL = """
SELECT
  ActionGeo_CountryCode AS country,
  DATE_TRUNC('month',
    strptime(LEFT(CAST(DATEADDED AS VARCHAR), 8), '%Y%m%d')
  ) AS month_start,
  COUNT(*) AS n_events,
  SUM(NumMentions) AS n_mentions_total,
  AVG(AvgTone) AS avg_tone,
  AVG(GoldsteinScale) AS avg_goldstein,
  SUM(AvgTone * NumMentions) / NULLIF(SUM(NumMentions), 0) AS weighted_tone,
  SUM(GoldsteinScale * NumMentions) / NULLIF(SUM(NumMentions), 0) AS weighted_goldstein,
  SUM(CASE WHEN QuadClass = 1 THEN 1 ELSE 0 END) AS n_verbal_coop,
  SUM(CASE WHEN QuadClass = 2 THEN 1 ELSE 0 END) AS n_material_coop,
  SUM(CASE WHEN QuadClass = 3 THEN 1 ELSE 0 END) AS n_verbal_conf,
  SUM(CASE WHEN QuadClass = 4 THEN 1 ELSE 0 END) AS n_material_conf,
  SUM(CASE WHEN EventRootCode = '14' THEN 1 ELSE 0 END) AS protest_count,
  SUM(CASE WHEN EventRootCode IN ('18','19','20') THEN 1 ELSE 0 END) AS violence_count,
  SUM(CASE WHEN EventRootCode IN ('10','11') THEN 1 ELSE 0 END) AS verbal_threat_count
FROM gdelt_events
WHERE ActionGeo_CountryCode IN ({countries})
GROUP BY country, month_start
ORDER BY country, month_start
"""


def main() -> None:
    countries_in = ','.join(f"'{c}'" for c in FIPS_COUNTRIES)
    con = duckdb.connect(str(DUCKDB_PATH))
    df = con.sql(MONTHLY_SQL.format(countries=countries_in)).df()
    con.close()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "country_monthly.parquet"
    df.to_parquet(out_path, index=False)

    print(f"Wrote {out_path}: shape={df.shape}")


if __name__ == '__main__':
    main()
