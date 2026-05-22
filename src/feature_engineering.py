"""Feature engineering pipeline for GDELT weekly data.

Produces feature_matrix from country_weekly:
- ensures every country has a continuous weekly index
- imputes missing values per column type
- adds rolling means / stds, lag, diff and pct_change features
- adds country one-hot for ML pooling
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.constants import LAG_RANGE, ROLLING_WINDOWS

COUNT_COLUMNS = [
    'n_events', 'n_mentions_total',
    'n_verbal_coop', 'n_material_coop',
    'n_verbal_conf', 'n_material_conf',
    'protest_count', 'violence_count', 'verbal_threat_count',
]

AVG_COLUMNS = [
    'avg_tone', 'avg_goldstein',
    'weighted_tone', 'weighted_goldstein',
]

ROLLING_LAG_TARGETS = ['avg_tone', 'avg_goldstein', 'n_material_conf']


def build_feature_matrix(df_weekly: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline.

    Parameters
    ----------
    df_weekly : DataFrame from load_country_weekly()

    Returns
    -------
    DataFrame with original columns plus rolling/lag/diff features and country one-hot.
    Sorted by (country, week_start).
    """
    df = df_weekly.copy()
    df = df.sort_values(['country', 'week_start']).reset_index(drop=True)
    df = ensure_continuous_weeks(df)
    df = impute_missing(df)
    df = add_rolling_features(df)
    df = add_lag_features(df)
    df = add_diff_features(df)
    df = add_v2_features(df)
    df = add_country_one_hot(df)
    return df


def ensure_continuous_weeks(df: pd.DataFrame) -> pd.DataFrame:
    """Re-index each country to have all weeks from min to max, fill gaps with NaN."""
    out = []
    for country, grp in df.groupby('country'):
        full_weeks = pd.date_range(
            start=grp['week_start'].min(),
            end=grp['week_start'].max(),
            freq='W-MON',
        )
        scaffold = pd.DataFrame({'country': country, 'week_start': full_weeks})
        merged = scaffold.merge(grp, on=['country', 'week_start'], how='left')
        out.append(merged)
    return (
        pd.concat(out, ignore_index=True)
        .sort_values(['country', 'week_start'])
        .reset_index(drop=True)
    )


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values:
    - count columns: 0 (no event = 0 events)
    - average columns: forward-fill, then back-fill, then country mean
      (cannot fill with 0 because 0 is a meaningful neutral value)
    """
    df = df.copy()

    for col in COUNT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    for col in AVG_COLUMNS:
        if col in df.columns:
            df[col] = df.groupby('country')[col].transform(
                lambda s: s.ffill().bfill().fillna(s.mean())
            )
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling mean and std features for ROLLING_LAG_TARGETS.

    Uses shift(1) before rolling so the feature for week t never includes week t.
    This prevents look-ahead bias.
    """
    df = df.copy()
    for col in ROLLING_LAG_TARGETS:
        if col not in df.columns:
            continue
        for window in ROLLING_WINDOWS:
            df[f'{col}_rolling_mean_{window}w'] = (
                df.groupby('country')[col]
                  .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
            )
            df[f'{col}_rolling_std_{window}w'] = (
                df.groupby('country')[col]
                  .transform(lambda s: s.shift(1).rolling(window, min_periods=2).std())
            )
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag features for ROLLING_LAG_TARGETS."""
    df = df.copy()
    for col in ROLLING_LAG_TARGETS:
        if col not in df.columns:
            continue
        for lag in LAG_RANGE:
            df[f'{col}_lag{lag}'] = df.groupby('country')[col].shift(lag)
    return df


def add_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 1-week diff, 4-week diff, and 4-week pct_change features."""
    df = df.copy()
    for col in ROLLING_LAG_TARGETS:
        if col not in df.columns:
            continue
        df[f'{col}_diff_1w'] = df.groupby('country')[col].diff(1)
        df[f'{col}_diff_4w'] = df.groupby('country')[col].diff(4)
        with np.errstate(divide='ignore', invalid='ignore'):
            df[f'{col}_pct_change_4w'] = df.groupby('country')[col].pct_change(4)
        df[f'{col}_pct_change_4w'] = df[f'{col}_pct_change_4w'].replace(
            [np.inf, -np.inf], np.nan
        )
    return df


def add_v2_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add v2 features suggested by E after Lasso baseline review.

    1. tone_goldstein_inter: avg_tone × avg_goldstein
       Captures joint signal of media negativity and conflict severity.

    2. Conflict ratio features: count / n_events
       Normalises by total event volume to remove news-cycle noise.

    3. Seasonal features: week_of_year, month
       Captures seasonal protest patterns.
    """
    df = df.copy()

    # 1. Interaction term
    df['tone_goldstein_inter'] = df['avg_tone'] * df['avg_goldstein']

    # 2. Conflict ratios (avoid divide-by-zero)
    n = df['n_events'].replace(0, np.nan)
    df['material_conf_ratio'] = df['n_material_conf'] / n
    df['verbal_conf_ratio']   = df['n_verbal_conf']   / n
    df['protest_ratio']       = df['protest_count']   / n

    # 3. Seasonal features
    df['week_of_year'] = df['week_start'].dt.isocalendar().week.astype(int)
    df['month']        = df['week_start'].dt.month

    return df


def add_country_one_hot(df: pd.DataFrame) -> pd.DataFrame:
    """Add country one-hot columns (is_<country>) without dropping the country column.

    Keeping country lets time-series members (C, D) filter by country,
    while ML members (E) can use the one-hot columns directly.
    """
    df = df.copy()
    one_hot = pd.get_dummies(df['country'], prefix='is')
    return df.join(one_hot)
