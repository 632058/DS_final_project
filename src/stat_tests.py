"""Cross-correlation and Granger causality tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, grangercausalitytests


def compute_ccf(x: pd.Series, y: pd.Series, max_lag: int = 12) -> pd.Series:
    """Cross-correlation: corr(x_{t-k}, y_t) for k = 0..max_lag.

    A positive lag k means x leads y by k weeks.
    Inputs are aligned on the intersection of their (cleaned) indices,
    then z-scored before correlation.
    """
    x_clean = x.dropna()
    y_clean = y.dropna()
    common_idx = x_clean.index.intersection(y_clean.index)
    x_aligned = x_clean.loc[common_idx].reset_index(drop=True)
    y_aligned = y_clean.loc[common_idx].reset_index(drop=True)

    n = len(x_aligned)
    if n < max_lag + 5:
        return pd.Series([np.nan] * (max_lag + 1),
                         index=range(max_lag + 1),
                         name=f'ccf_{x.name}_to_{y.name}')

    x_std = (x_aligned - x_aligned.mean()) / x_aligned.std(ddof=0)
    y_std = (y_aligned - y_aligned.mean()) / y_aligned.std(ddof=0)

    values = []
    for k in range(max_lag + 1):
        if k == 0:
            corr = float((x_std * y_std).sum() / n)
        else:
            corr = float((x_std.iloc[:-k].values * y_std.iloc[k:].values).sum() / n)
        values.append(corr)

    return pd.Series(values, index=range(max_lag + 1),
                     name=f'ccf_{x.name}_to_{y.name}')


def granger_test(y: pd.Series, x: pd.Series, max_lag: int = 12) -> dict:
    """H0: x does NOT Granger-cause y.

    Reports the minimum p-value across lags 1..max_lag and which lag achieved it.
    Reject H0 if the minimum p-value < 0.05 (note: no multiple-testing correction).
    """
    df = pd.concat([y, x], axis=1).dropna()
    df.columns = ['y', 'x']

    if len(df) < max_lag + 5:
        return {
            'p_values_per_lag': {},
            'best_lag': None,
            'best_p_value': float('nan'),
            'reject_h0': False,
        }

    # statsmodels >=0.14 removed the verbose argument; older versions still need it.
    try:
        raw = grangercausalitytests(df[['y', 'x']], maxlag=max_lag)
    except TypeError:
        raw = grangercausalitytests(df[['y', 'x']], maxlag=max_lag, verbose=False)

    p_values = {}
    for lag in range(1, max_lag + 1):
        p_values[lag] = float(raw[lag][0]['ssr_ftest'][1])

    best_lag = min(p_values, key=p_values.get)
    return {
        'p_values_per_lag': p_values,
        'best_lag': int(best_lag),
        'best_p_value': float(p_values[best_lag]),
        'reject_h0': bool(p_values[best_lag] < 0.05),
    }


def stationarity_diff(series: pd.Series, max_d: int = 2) -> tuple:
    """Difference series until stationary by ADF; return (diffed_series, d)."""
    s = series.dropna()
    for d in range(max_d + 1):
        candidate = s.diff(d).dropna() if d > 0 else s
        if len(candidate) < 10:
            continue
        if adfuller(candidate, autolag='AIC')[1] < 0.05:
            return candidate, d
    return s.diff(max_d).dropna(), max_d
