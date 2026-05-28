"""ARIMAX time series modeling for protest count prediction.

End-to-end pipeline: ADF stationarity test, difference order selection,
ARIMAX (p, d, q) grid search by AIC, residual diagnostics, time-based
train/test split, and forecast export.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning

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


def _with_inferred_datetime_freq(data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Attach inferred DatetimeIndex frequency so statsmodels does not warn."""
    index = data.index
    if not isinstance(index, pd.DatetimeIndex) or index.freq is not None:
        return data

    try:
        inferred_freq = pd.infer_freq(index)
    except ValueError:
        return data
    if inferred_freq is None:
        return data

    out = data.copy()
    out.index = pd.DatetimeIndex(index, freq=inferred_freq)
    return out


def _fit_sarimax(y: pd.Series, exog: pd.DataFrame, order: tuple):
    """Fit SARIMAX while suppressing expected model-search warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", ValueWarning)
        model = SARIMAX(
            y, exog=exog, order=order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        return model.fit(disp=False)


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
    y = _with_inferred_datetime_freq(y)
    exog = _with_inferred_datetime_freq(exog)

    if order is not None:
        chosen_order = order
    else:
        d = select_difference_order(y)
        best_aic = np.inf
        chosen_order = (1, d, 1)
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                try:
                    res = _fit_sarimax(y, exog, (p, d, q))
                    if res.aic < best_aic:
                        best_aic = res.aic
                        chosen_order = (p, d, q)
                except Exception:
                    continue

    final = _fit_sarimax(y, exog, chosen_order)

    return {
        'order': chosen_order,
        'aic': float(final.aic),
        'bic': float(final.bic),
        'converged': bool(final.mle_retvals.get('converged', True)),
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


def _transform_target(y: pd.Series, target_transform: str) -> pd.Series:
    """Transform the endogenous target before SARIMAX fitting."""
    if target_transform == 'raw':
        return y
    if target_transform == 'log1p':
        if (y < 0).any():
            raise ValueError('log1p target transform requires non-negative counts')
        # log1p keeps zero-count weeks finite and compresses large protest spikes.
        return np.log1p(y)
    raise ValueError(f"unknown target transform: {target_transform!r}")


def _inverse_transform_forecast(values, target_transform: str) -> np.ndarray:
    """Map forecasts back to the original count scale for metrics and export."""
    arr = np.asarray(values, dtype=float)
    if target_transform == 'raw':
        return arr
    if target_transform == 'log1p':
        # expm1 reverses log1p; clipping enforces the non-negative count support.
        return np.clip(np.expm1(arr), 0.0, None)
    raise ValueError(f"unknown target transform: {target_transform!r}")


def _format_lag_spec(exog_lag_set: Sequence[int] | Mapping[str, Sequence[int]]) -> dict:
    """Normalize shared or feature-specific lag specs for summary output."""
    if isinstance(exog_lag_set, Mapping):
        return {base: [int(lag) for lag in lags] for base, lags in exog_lag_set.items()}
    return {'all': [int(lag) for lag in exog_lag_set]}


def _select_exog_columns(
    df_country: pd.DataFrame,
    exog_lag_set: Sequence[int] | Mapping[str, Sequence[int]],
) -> list[str]:
    """Build lagged exogenous columns while avoiding same-week look-ahead."""
    exog_cols = []
    default_lags = None if isinstance(exog_lag_set, Mapping) else exog_lag_set
    for base in ['avg_tone', 'avg_goldstein', 'n_material_conf']:
        lags = exog_lag_set.get(base, ()) if isinstance(exog_lag_set, Mapping) else default_lags
        for lag in lags:
            if int(lag) <= 0:
                continue
            col = f'{base}_lag{int(lag)}'
            if col in df_country.columns:
                exog_cols.append(col)
    return exog_cols


def run_country_arimax(
    country: str,
    df_features: pd.DataFrame,
    output_dir: Path,
    target: str = 'protest_count',
    exog_lag_set: Sequence[int] | Mapping[str, Sequence[int]] = (1, 2, 4),
    order: tuple[int, int, int] | None = None,
    max_p: int = 3,
    max_q: int = 3,
    spec_name: str = 'baseline_lag_1_2_4',
    target_transform: str = 'raw',
    save_outputs: bool = True,
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

    exog_cols = _select_exog_columns(df_c, exog_lag_set)
    if not exog_cols:
        raise ValueError(f'No lagged exogenous columns found for {country}: {exog_lag_set!r}')

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

    y_train_model = _transform_target(y_train, target_transform)
    fit = fit_arimax(y_train_model, exog_train, order=order, max_p=max_p, max_q=max_q)
    diag = diagnose_residuals(fit['residuals'])

    forecast_obj = fit['model'].get_forecast(steps=len(y_test), exog=exog_test)
    pred_model = forecast_obj.predicted_mean
    ci = forecast_obj.conf_int(alpha=0.05)

    pred = _inverse_transform_forecast(pred_model, target_transform)
    ci_lower = _inverse_transform_forecast(ci.iloc[:, 0].values, target_transform)
    ci_upper = _inverse_transform_forecast(ci.iloc[:, 1].values, target_transform)

    rmse = float(np.sqrt(((y_test.values - pred) ** 2).mean()))
    mae = float(np.abs(y_test.values - pred).mean())

    summary = {
        'country': country,
        'spec_name': spec_name,
        'target': target,
        'target_transform': target_transform,
        'order': list(fit['order']),
        'requested_order': list(order) if order is not None else None,
        'max_p': int(max_p),
        'max_q': int(max_q),
        'exog_lag_set': _format_lag_spec(exog_lag_set),
        'aic': fit['aic'],
        'bic': fit['bic'],
        'converged': fit['converged'],
        'ljung_box_pvalue': diag['ljung_box_pvalue'],
        'is_white_noise': diag['is_white_noise'],
        'rmse': rmse,
        'mae': mae,
        'n_train': int(n_train),
        'n_test': int(len(y_test)),
        'exog_cols': exog_cols,
    }

    if save_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_stem = country if target_transform == 'raw' else f'{country}_{target_transform}'
        with open(output_dir / f'{output_stem}.json', 'w', encoding='utf-8') as fh:
            json.dump(summary, fh, indent=2, default=str)

        forecast_df = pd.DataFrame({
            'week_start': y_test.index,
            'actual': y_test.values,
            'predicted': pred,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'predicted_model_scale': pred_model.values,
            'ci_lower_model_scale': ci.iloc[:, 0].values,
            'ci_upper_model_scale': ci.iloc[:, 1].values,
        })
        forecast_df.to_parquet(output_dir / f'{output_stem}_forecast.parquet', index=False)

    return summary
