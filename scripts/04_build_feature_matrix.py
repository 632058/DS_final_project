"""Build feature_matrix.parquet from country_weekly.parquet.

Run: python scripts/04_build_feature_matrix.py
Output: output/feature_matrix.parquet
"""
from __future__ import annotations

from src.constants import OUTPUT_DIR
from src.data_loader import load_country_weekly
from src.feature_engineering import build_feature_matrix


def main() -> None:
    df_weekly = load_country_weekly()
    df_features = build_feature_matrix(df_weekly)

    out_path = OUTPUT_DIR / "feature_matrix.parquet"
    df_features.to_parquet(out_path, index=False)

    print(f"Wrote {out_path}: shape={df_features.shape}")
    print(f"Feature columns: {len(df_features.columns)}")


if __name__ == '__main__':
    main()
