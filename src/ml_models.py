"""Machine learning pipeline for protest count regression.

Pooled across 5 countries with country one-hot features.
Time-based train/test split (no shuffling) to avoid look-ahead bias.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from src.constants import TRAIN_RATIO

TARGET_COL = 'protest_count_next_week'

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
    """RMSE, MAE, and Directional Accuracy (sign of week-over-week change)."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    actual_diff = np.diff(y_true.values)
    pred_diff = np.diff(y_pred)
    directional = float(np.mean(np.sign(actual_diff) == np.sign(pred_diff))) \
        if len(actual_diff) > 0 else float('nan')
    return {'rmse': rmse, 'mae': mae, 'directional_accuracy': directional}


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
