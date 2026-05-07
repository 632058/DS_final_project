"""Common data loading helpers."""
import pandas as pd

from src.constants import OUTPUT_DIR


def load_country_weekly() -> pd.DataFrame:
    """Load country_weekly aggregation produced by member A."""
    return pd.read_parquet(OUTPUT_DIR / "country_weekly.parquet")


def load_country_monthly() -> pd.DataFrame:
    """Load country_monthly aggregation produced by member A."""
    return pd.read_parquet(OUTPUT_DIR / "country_monthly.parquet")


def load_eventmix_weekly() -> pd.DataFrame:
    """Load country_eventmix_weekly aggregation produced by member A."""
    return pd.read_parquet(OUTPUT_DIR / "country_eventmix_weekly.parquet")


def load_feature_matrix() -> pd.DataFrame:
    """Load feature_matrix produced by member B."""
    return pd.read_parquet(OUTPUT_DIR / "feature_matrix.parquet")
