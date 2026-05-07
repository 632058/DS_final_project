"""ARIMAX time series modeling for protest count prediction.

End-to-end pipeline: ADF stationarity test, difference order selection,
ARIMAX (p, d, q) grid search by AIC, residual diagnostics, time-based
train/test split, and forecast export.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.constants import TRAIN_RATIO


def run_adf_test(series: pd.Series, name: str = "") -> dict:
    """Augmented Dickey-Fuller test for stationarity.

    p < 0.05 means series is stationary.
    """
    s = series.dropna()
    stat, pvalue, n_lags, n_obs, crit_vals, _ = adfuller(s, autolag='AIC')
    return {
        'name': name,
        'adf_statistic': float(stat),
        'p_value': float(pvalue),
        'n_lags': int(n_lags),
        'n_obs': int(n_obs),
        'critical_values': {k: float(v) for k, v in crit_vals.items()},
        'is_stationary': bool(pvalue < 0.05),
    }


def select_difference_order(series: pd.Series, max_d: int = 2) -> int:
    """Find smallest d such that diff^d(series) is stationary by ADF."""
    s = series.dropna()
    for d in range(max_d + 1):
        candidate = s.diff(d).dropna() if d > 0 else s
        if len(candidate) < 10:
            continue
        if adfuller(candidate, autolag='AIC')[1] < 0.05:
            return d
    return max_d


def fit_arimax(
    y: pd.Series,
    exog: pd.DataFrame,
    order: tuple | None = None,
    max_p: int = 3,
    max_q: int = 3,
):
    """Fit ARIMAX model. If order is None, grid-search (p, d, q) by AIC.

    d is chosen via ADF test. p and q are searched in [0, max_p] x [0, max_q].
    """
    if order is not None:
        chosen_order = order
    else:
        d = select_difference_order(y)
        best_aic = np.inf
        chosen_order = (1, d, 1)
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                try:
                    model = SARIMAX(
                        y, exog=exog, order=(p, d, q),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    res = model.fit(disp=False)
                    if res.aic < best_aic:
                        best_aic = res.aic
                        chosen_order = (p, d, q)
                except Exception:
                    continue

    final = SARIMAX(
        y, exog=exog, order=chosen_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    return {
        'order': chosen_order,
        'aic': float(final.aic),
        'bic': float(final.bic),
        'model': final,
        'residuals': final.resid,
    }


def diagnose_residuals(residuals: pd.Series, lags: int = 12) -> dict:
    """Ljung-Box test on residuals. p > 0.05 means residuals are white noise (good)."""
    lb = acorr_ljungbox(residuals.dropna(), lags=[lags], return_df=True)
    return {
        'ljung_box_stat': float(lb['lb_stat'].values[0]),
        'ljung_box_pvalue': float(lb['lb_pvalue'].values[0]),
        'is_white_noise': bool(lb['lb_pvalue'].values[0] > 0.05),
    }


def run_country_arimax(
    country: str,
    df_features: pd.DataFrame,
    output_dir: Path,
    target: str = 'protest_count',
    exog_lag_set: tuple = (1, 2, 4),
) -> dict:
    """End-to-end ARIMAX pipeline for one country.

    1. Filter feature_matrix to country, set time index
    2. Build exog from lag features (avoid look-ahead)
    3. Time-based 80/20 split
    4. Grid-search (p, d, q) by AIC on train
    5. Forecast on test, evaluate, save
    """
    df_c = (
        df_features[df_features['country'] == country]
        .sort_values('week_start')
        .set_index('week_start')
    )

    exog_cols = []
    for base in ['avg_tone', 'avg_goldstein', 'n_material_conf']:
        for lag in exog_lag_set:
            col = f'{base}_lag{lag}'
            if col in df_c.columns:
                exog_cols.append(col)

    y_full = df_c[target]
    exog_full = df_c[exog_cols]

    valid = y_full.notna() & exog_full.notna().all(axis=1)
    y_full = y_full[valid]
    exog_full = exog_full[valid]

    n_train = int(len(y_full) * TRAIN_RATIO)
    y_train = y_full.iloc[:n_train]
    y_test = y_full.iloc[n_train:]
    exog_train = exog_full.iloc[:n_train]
    exog_test = exog_full.iloc[n_train:]

    fit = fit_arimax(y_train, exog_train)
    diag = diagnose_residuals(fit['residuals'])

    forecast_obj = fit['model'].get_forecast(steps=len(y_test), exog=exog_test)
    pred = forecast_obj.predicted_mean
    ci = forecast_obj.conf_int(alpha=0.05)

    rmse = float(np.sqrt(((y_test.values - pred.values) ** 2).mean()))
    mae = float(np.abs(y_test.values - pred.values).mean())

    summary = {
        'country': country,
        'order': list(fit['order']),
        'aic': fit['aic'],
        'bic': fit['bic'],
        'ljung_box_pvalue': diag['ljung_box_pvalue'],
        'is_white_noise': diag['is_white_noise'],
        'rmse': rmse,
        'mae': mae,
        'n_train': int(n_train),
        'n_test': int(len(y_test)),
        'exog_cols': exog_cols,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f'{country}.json', 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2, default=str)

    forecast_df = pd.DataFrame({
        'week_start': y_test.index,
        'actual': y_test.values,
        'predicted': pred.values,
        'ci_lower': ci.iloc[:, 0].values,
        'ci_upper': ci.iloc[:, 1].values,
    })
    forecast_df.to_parquet(output_dir / f'{country}_forecast.parquet', index=False)

    return summary
