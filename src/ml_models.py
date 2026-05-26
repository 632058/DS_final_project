"""Machine learning pipeline for protest count regression.

Pooled across 5 countries with country one-hot features.
Time-based train/test split (no shuffling) to avoid look-ahead bias.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.constants import TRAIN_RATIO

# Default alpha grid for Lasso cross-validation. Spans 5 decades so the
# selected alpha is unlikely to hit a boundary.
DEFAULT_LASSO_CV_ALPHAS = np.logspace(-3, 1, 9)
DEFAULT_LASSO_CV_SPLITS = 5

TARGET_COL = 'protest_count_next_week'

TransformKind = Literal['raw', 'log1p']
ScopeKind = Literal['all', 'domestic']

# Source column for the next-week target, keyed by target scope.
TARGET_SOURCE_BY_SCOPE: dict[str, str] = {
    'all': 'protest_count_all',
    'domestic': 'protest_count',
}

# Metadata columns that are never features regardless of scope.
_METADATA_COLS = frozenset({'week_start', 'country', TARGET_COL})


def _cross_scope_feature_columns(scope: str, columns: list[str]) -> set[str]:
    """Return all columns in ``columns`` that belong to the OTHER target scope.

    With Phase 2c there are two parallel autoregressive families:
        protest_count_all*       (all-scope: current week, lag, rolling, diff, ratio)
        protest_count*           (domestic: current week, lag, rolling, diff, ratio)
    The ``protest_count_all*`` family is a strict prefix superset of
    ``protest_count*`` so we cannot match by 'startswith' alone; instead we
    enumerate the cross-scope family explicitly.
    """
    if scope == 'all':
        # Drop the domestic family but keep the all-scope family. The all-scope
        # family also starts with 'protest_count' so we have to exclude any
        # column that starts with 'protest_count_all'.
        return {
            c for c in columns
            if c.startswith('protest_count') and not c.startswith('protest_count_all')
        } | {
            c for c in columns
            if c == 'protest_ratio_domestic'
        }
    if scope == 'domestic':
        return {
            c for c in columns
            if c.startswith('protest_count_all')
        } | {
            c for c in columns
            if c == 'protest_ratio'
        }
    raise ValueError(f"unknown target scope: {scope!r}")


def non_feature_cols(
    scope: str = 'all',
    columns: list[str] | None = None,
) -> frozenset[str]:
    """Columns excluded from the feature matrix for the given target scope.

    Always excludes the metadata columns and the target column.

    When ``columns`` is provided, also excludes every column that belongs to
    the OTHER target scope's autoregressive feature family. This keeps the
    all-branch model from training on domestic AR features (which add noise
    correlated with the target but only at the domestic scope) and vice
    versa. See :func:`_cross_scope_feature_columns`.

    Without ``columns`` the function returns only the metadata/target set,
    matching the legacy single-scope behaviour.
    """
    if scope not in TARGET_SOURCE_BY_SCOPE:
        raise ValueError(f"unknown target scope: {scope!r}")
    base = set(_METADATA_COLS)
    if columns is not None:
        base |= _cross_scope_feature_columns(scope, columns)
    return frozenset(base)


# Legacy alias preserved so importers that only need the default (all-scope)
# metadata/target exclusion set keep working unchanged.
NON_FEATURE_COLS = non_feature_cols('all')


def add_target(df: pd.DataFrame, scope: str = 'all') -> pd.DataFrame:
    """Add the next-week protest target as ``shift(-1)`` per country.

    The source column is chosen by ``scope``:
    - 'all'      -> ``protest_count_all`` (current ML main line)
    - 'domestic' -> ``protest_count``     (Actor1=Actor2=country, domestic only)

    The target column name is always ``TARGET_COL`` regardless of scope,
    so downstream metric / prediction schemas do not change between branches.
    """
    if scope not in TARGET_SOURCE_BY_SCOPE:
        raise ValueError(f"unknown target scope: {scope!r}")
    source_col = TARGET_SOURCE_BY_SCOPE[scope]
    if source_col not in df.columns:
        raise KeyError(
            f"add_target(scope={scope!r}) needs column {source_col!r} "
            f"but it is not in df.columns"
        )
    df = df.sort_values(['country', 'week_start']).copy()
    df[TARGET_COL] = df.groupby('country')[source_col].shift(-1)
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


def prepare_xy(
    df: pd.DataFrame,
    scope: str = 'all',
    target: str = TARGET_COL,
) -> tuple:
    """Drop non-feature columns and rows with NaN in features or target.

    Cross-scope autoregressive features (e.g. domestic ``protest_count_lag*``
    when training the all-scope branch) are pruned by passing ``df.columns``
    to :func:`non_feature_cols`, so each branch only sees its own AR family.

    Returns (X, y, mask) where ``mask`` lets callers recover the original
    metadata rows that survived NaN filtering.
    """
    drop = non_feature_cols(scope, columns=list(df.columns))
    feature_cols = [c for c in df.columns if c not in drop]
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


def train_lasso_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    alphas: np.ndarray | None = None,
    n_splits: int = DEFAULT_LASSO_CV_SPLITS,
) -> tuple:
    """LassoCV with time-series cross-validation.

    Returns ``(model, fitted_scaler)`` where ``model.alpha_`` holds the
    CV-selected regularization strength. Uses :class:`TimeSeriesSplit` so
    each validation fold draws from strictly later weeks than its training
    fold, avoiding look-ahead bias.

    Caller contract: ``X_train`` rows MUST be ordered by ``week_start``.
    Both :func:`time_train_test_split` and :func:`prepare_xy` preserve that
    ordering, so the standard pipeline satisfies this automatically.
    """
    if alphas is None:
        alphas = DEFAULT_LASSO_CV_ALPHAS
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    model = LassoCV(alphas=alphas, cv=tscv, max_iter=10000, n_jobs=-1)
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
