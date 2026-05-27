"""Smoke tests for src.arimax_model."""
import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning

from src.arimax_model import (
    diagnose_residuals,
    fit_arimax,
    run_adf_test,
    run_country_arimax,
    select_difference_order,
)


def test_adf_random_walk_not_stationary():
    np.random.seed(42)
    rw = pd.Series(np.cumsum(np.random.randn(200)))
    result = run_adf_test(rw, 'random_walk')
    assert not result['is_stationary']


def test_adf_white_noise_is_stationary():
    np.random.seed(42)
    s = pd.Series(np.random.randn(200))
    result = run_adf_test(s, 'noise')
    assert result['is_stationary']


def test_select_difference_order_random_walk():
    np.random.seed(42)
    rw = pd.Series(np.cumsum(np.random.randn(200)))
    d = select_difference_order(rw)
    assert d == 1


def test_diagnose_residuals_white_noise():
    np.random.seed(42)
    residuals = pd.Series(np.random.randn(200))
    result = diagnose_residuals(residuals)
    assert result['is_white_noise']


# ---------------------------------------------------------------------------
# fit_arimax
# ---------------------------------------------------------------------------

def test_fit_arimax_explicit_order_keys():
    np.random.seed(42)
    n = 80
    y = pd.Series(np.random.randn(n))
    exog = pd.DataFrame({'x': np.random.randn(n)})
    result = fit_arimax(y, exog, order=(1, 0, 1))
    assert set(result.keys()) >= {'order', 'aic', 'bic', 'model', 'residuals'}
    assert result['order'] == (1, 0, 1)
    assert isinstance(result['aic'], float)
    assert len(result['residuals']) == n


def test_fit_arimax_grid_search_valid_order():
    np.random.seed(42)
    n = 80
    y = pd.Series(np.random.randn(n))
    exog = pd.DataFrame({'x': np.random.randn(n)})
    result = fit_arimax(y, exog, max_p=1, max_q=1)
    p, d, q = result['order']
    assert 0 <= p <= 1
    assert 0 <= d <= 2
    assert 0 <= q <= 1


def test_fit_arimax_aic_is_finite():
    np.random.seed(0)
    n = 60
    y = pd.Series(np.random.randn(n))
    exog = pd.DataFrame({'x': np.random.randn(n)})
    result = fit_arimax(y, exog, order=(0, 0, 1))
    assert np.isfinite(result['aic'])
    assert np.isfinite(result['bic'])


def test_fit_arimax_suppresses_expected_statsmodels_warnings_for_weekly_index():
    rng = np.random.default_rng(9)
    weeks = pd.date_range('2022-01-03', periods=60, freq='W-MON')
    index_without_freq = pd.DatetimeIndex(weeks.to_numpy())
    y = pd.Series(rng.standard_normal(60), index=index_without_freq)
    exog = pd.DataFrame({'x': rng.standard_normal(60)}, index=index_without_freq)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fit_arimax(y, exog, order=(1, 0, 0))

    assert 'converged' in result
    assert not any(
        issubclass(item.category, (ConvergenceWarning, ValueWarning))
        for item in caught
    )


# ---------------------------------------------------------------------------
# run_country_arimax — uses build_feature_matrix to generate realistic input
# ---------------------------------------------------------------------------

def _make_feature_matrix(n_weeks: int = 60) -> pd.DataFrame:
    from src.feature_engineering import build_feature_matrix
    countries = ['CE', 'AR']
    rows = []
    weeks = pd.date_range('2022-01-03', periods=n_weeks, freq='W-MON')
    rng = np.random.default_rng(7)
    for c in countries:
        protest = rng.integers(0, 10, size=n_weeks).tolist()
        tone = rng.standard_normal(n_weeks).tolist()
        for i, w in enumerate(weeks):
            rows.append({
                'country': c, 'week_start': w,
                'n_events': 50, 'n_mentions_total': 500,
                'avg_tone': tone[i], 'avg_goldstein': rng.uniform(-2, 2),
                'weighted_tone': tone[i], 'weighted_goldstein': 0.0,
                'n_verbal_coop': 2, 'n_material_coop': 1,
                'n_verbal_conf': 1, 'n_material_conf': int(rng.integers(0, 5)),
                'protest_count': protest[i], 'violence_count': 0,
                'verbal_threat_count': 0,
            })
    return build_feature_matrix(pd.DataFrame(rows))


def test_run_country_arimax_saves_outputs(tmp_path):
    df = _make_feature_matrix()
    summary = run_country_arimax('CE', df, output_dir=tmp_path)
    assert (tmp_path / 'CE.json').exists()
    assert (tmp_path / 'CE_forecast.parquet').exists()


def test_run_country_arimax_summary_keys(tmp_path):
    df = _make_feature_matrix()
    summary = run_country_arimax('CE', df, output_dir=tmp_path)
    required = {
        'country', 'spec_name', 'target', 'order', 'requested_order', 'max_p', 'max_q',
        'target_transform', 'exog_lag_set', 'aic', 'bic', 'converged', 'rmse', 'mae',
        'n_train', 'n_test', 'exog_cols',
    }
    assert required.issubset(summary.keys())


def test_run_country_arimax_forecast_columns(tmp_path):
    df = _make_feature_matrix()
    run_country_arimax('CE', df, output_dir=tmp_path)
    forecast = pd.read_parquet(tmp_path / 'CE_forecast.parquet')
    assert {'week_start', 'actual', 'predicted', 'ci_lower', 'ci_upper'}.issubset(forecast.columns)


def test_run_country_arimax_metrics_are_finite(tmp_path):
    df = _make_feature_matrix()
    summary = run_country_arimax('AR', df, output_dir=tmp_path)
    assert np.isfinite(summary['rmse'])
    assert np.isfinite(summary['mae'])
    assert summary['n_train'] > 0
    assert summary['n_test'] > 0


def test_run_country_arimax_accepts_custom_lags_and_order_without_saving(tmp_path):
    df = _make_feature_matrix()
    summary = run_country_arimax(
        'CE',
        df,
        output_dir=tmp_path,
        exog_lag_set={
            'avg_tone': [1],
            'avg_goldstein': [2],
            'n_material_conf': [4],
        },
        order=(1, 0, 0),
        spec_name='custom_test',
        save_outputs=False,
    )
    assert summary['spec_name'] == 'custom_test'
    assert summary['order'] == [1, 0, 0]
    assert summary['requested_order'] == [1, 0, 0]
    assert summary['exog_cols'] == ['avg_tone_lag1', 'avg_goldstein_lag2', 'n_material_conf_lag4']
    assert not (tmp_path / 'CE.json').exists()
    assert not (tmp_path / 'CE_forecast.parquet').exists()


def test_run_country_arimax_log1p_writes_separate_nonnegative_forecast(tmp_path):
    df = _make_feature_matrix()
    summary = run_country_arimax(
        'CE',
        df,
        output_dir=tmp_path,
        order=(1, 0, 0),
        target_transform='log1p',
    )
    forecast = pd.read_parquet(tmp_path / 'CE_log1p_forecast.parquet')
    assert summary['target_transform'] == 'log1p'
    assert (tmp_path / 'CE_log1p.json').exists()
    assert not (tmp_path / 'CE.json').exists()
    assert {'predicted_model_scale', 'ci_lower_model_scale', 'ci_upper_model_scale'}.issubset(forecast.columns)
    assert (forecast[['predicted', 'ci_lower', 'ci_upper']] >= 0).all().all()
    assert np.isfinite(summary['rmse'])
    assert np.isfinite(summary['mae'])
