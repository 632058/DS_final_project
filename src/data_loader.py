"""Common data loading helpers."""
import pandas as pd

from src.constants import DATA_DIR, OUTPUT_DIR


def load_country_weekly() -> pd.DataFrame:
    """Load country_weekly aggregation produced by member A.

    Reads OUTPUT_DIR (the 01_build_country_weekly.py target), not DATA_DIR.
    The OUTPUT_DIR copy distinguishes domestic-only `protest_count`
    (country_1 == country_2) from `protest_count_all`; the DATA_DIR copy
    produced by 01b lacks that split.
    """
    return pd.read_parquet(OUTPUT_DIR / "country_weekly.parquet")


def load_country_monthly() -> pd.DataFrame:
    """Load country_monthly aggregation produced by member A."""
    return pd.read_parquet(DATA_DIR / "country_monthly.parquet")


def load_eventmix_weekly() -> pd.DataFrame:
    """Load country_eventmix_weekly aggregation produced by member A."""
    return pd.read_parquet(DATA_DIR / "country_eventmix_weekly.parquet")


def load_feature_matrix() -> pd.DataFrame:
    """Load feature_matrix produced by member B."""
    return pd.read_parquet(OUTPUT_DIR / "feature_matrix.parquet")
