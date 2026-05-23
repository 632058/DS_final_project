"""Machine learning pipeline for protest count regression.

Pooled across 5 countries with country one-hot features.
Time-based train/test split (no shuffling) to avoid look-ahead bias.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from src.constants import TRAIN_RATIO

TARGET_COL = 'protest_count_next_week'

TransformKind = Literal['raw', 'log1p']

NON_FEATURE_COLS = {
    'week_start', 'country',
    # protest_count is the domestic-only legacy column; excluded so the
    # model does not see two scopes of the same signal.
    'protest_count',
    # protest_count_all (current-week, all-scope) IS a legitimate
    # autoregressive feature: target is protest_count_all.shift(-1) per
    # country, so using week-t's value to predict week-(t+1) is not leakage.
    TARGET_COL,
}


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add the next-week target as protest_count_all.shift(-1) per country.

    Uses protest_count_all (all events in the country) rather than the
    domestic-only protest_count, so the target stays on the same scope as
    the conflict-structure features (avg_tone, avg_goldstein, n_material_conf
    are all all-events-scope).
    """
    df = df.sort_values(['country', 'week_start']).copy()
    df[TARGET_COL] = df.groupby('country')['protest_count_all'].shift(-1)
    return df


def time_train_test_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
) -> tuple:
    """Split by week_start globally so train precedes test in time.

    All countries share the same cutoff date.
    """
    df = df.sort_values('week_start').reset_index(drop=True)
    cutoff_idx = int(len(df) * train_ratio)
    cutoff_week = df['week_start'].iloc[cutoff_idx]
    train = df[df['week_start'] < cutoff_week].copy()
    test = df[df['week_start'] >= cutoff_week].copy()
    return train, test


def prepare_xy(df: pd.DataFrame, target: str = TARGET_COL) -> tuple:
    """Drop non-feature columns and rows with NaN in features or target.

    Returns (X, y) plus the row mask (so callers can recover original metadata).
    """
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols].copy()
    y = df[target].copy()
    mask = y.notna() & X.notna().all(axis=1)
    return X[mask], y[mask], mask


def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """RMSE and MAE on the original scale.

    Does NOT compute directional accuracy. The previous implementation called
    ``np.diff(y_true.values)`` on a pooled-panel y, which mixes neighbouring
    rows from different countries at country boundaries. Use
    ``evaluate_predictions(pred_df)`` instead, which sorts within each country
    before taking diffs.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {'rmse': rmse, 'mae': mae}


def evaluate_predictions(pred_df: pd.DataFrame) -> dict:
    """Country-aware metrics from a predictions DataFrame.

    Parameters
    ----------
    pred_df : DataFrame with at least these columns:
        country, week_start, actual, predicted

    Returns
    -------
    dict with keys:
        rmse, mae                        : pooled across all rows
        directional_accuracy             : sample-weighted mean of per-country
                                           directional accuracy (sign of
                                           week-over-week change). NaN if every
                                           country has fewer than 2 rows.
        per_country_directional          : dict country -> directional accuracy
                                           (NaN for countries with <2 rows)
    """
    required = {'country', 'week_start', 'actual', 'predicted'}
    missing = required - set(pred_df.columns)
    if missing:
        raise ValueError(f"pred_df missing required columns: {sorted(missing)}")

    df = pred_df.sort_values(['country', 'week_start']).reset_index(drop=True)

    rmse = float(np.sqrt(mean_squared_error(df['actual'], df['predicted'])))
    mae = float(mean_absolute_error(df['actual'], df['predicted']))

    per_country: dict[str, float] = {}
    weighted_sum = 0.0
    total_weight = 0
    for country, grp in df.groupby('country', sort=True):
        if len(grp) < 2:
            per_country[country] = float('nan')
            continue
        actual_sign = np.sign(np.diff(grp['actual'].to_numpy()))
        pred_sign = np.sign(np.diff(grp['predicted'].to_numpy()))
        dir_acc = float(np.mean(actual_sign == pred_sign))
        per_country[country] = dir_acc
        weight = len(grp) - 1
        weighted_sum += dir_acc * weight
        total_weight += weight

    overall_dir = (
        float(weighted_sum / total_weight) if total_weight > 0 else float('nan')
    )

    return {
        'rmse': rmse,
        'mae': mae,
        'directional_accuracy': overall_dir,
        'per_country_directional': per_country,
    }


def apply_target_transform(y: pd.Series | np.ndarray, kind: TransformKind) -> np.ndarray:
    """Forward-transform the regression target prior to fitting.

    ``raw``   : identity
    ``log1p`` : log(1 + y); requires y >= 0
    """
    arr = np.asarray(y, dtype=float)
    if kind == 'raw':
        return arr
    if kind == 'log1p':
        if (arr < 0).any():
            raise ValueError("log1p target transform requires y >= 0")
        return np.log1p(arr)
    raise ValueError(f"unknown target transform: {kind!r}")


def invert_target_transform(y_pred: np.ndarray, kind: TransformKind) -> np.ndarray:
    """Inverse of ``apply_target_transform`` for model predictions.

    Negative predictions are NOT clipped here; callers should clip if their
    target is a non-negative count (e.g. ``np.clip(out, 0, None)``).
    """
    arr = np.asarray(y_pred, dtype=float)
    if kind == 'raw':
        return arr
    if kind == 'log1p':
        return np.expm1(arr)
    raise ValueError(f"unknown target transform: {kind!r}")


def train_lasso(X_train: pd.DataFrame, y_train: pd.Series, alpha: float = 1.0) -> tuple:
    """Lasso with feature standardization. Returns (model, fitted_scaler)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = Lasso(alpha=alpha, max_iter=10000)
    model.fit(X_scaled, y_train)
    return model, scaler


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict | None = None,
) -> xgb.XGBRegressor:
    """XGBoost regressor with reasonable defaults."""
    defaults = {
        'n_estimators': 300,
        'max_depth': 5,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
    }
    if params:
        defaults.update(params)
    model = xgb.XGBRegressor(**defaults)
    model.fit(X_train, y_train)
    return model
