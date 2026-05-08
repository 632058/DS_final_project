"""Smoke tests for src.feature_engineering."""
import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    add_country_one_hot,
    add_diff_features,
    add_lag_features,
    add_rolling_features,
    build_feature_matrix,
    ensure_continuous_weeks,
    impute_missing,
)


def _make_dummy(n_weeks: int = 30) -> pd.DataFrame:
    countries = ['CE', 'AR']
    rows = []
    weeks = pd.date_range('2024-01-01', periods=n_weeks, freq='W-MON')
    for c in countries:
        for w in weeks:
            rows.append({
                'country': c,
                'week_start': w,
                'n_events': 10, 'n_mentions_total': 100,
                'avg_tone': -2.5, 'avg_goldstein': 0.5,
                'weighted_tone': -2.0, 'weighted_goldstein': 0.7,
                'n_verbal_coop': 1, 'n_material_coop': 1,
                'n_verbal_conf': 1, 'n_material_conf': 1,
                'protest_count': 0, 'violence_count': 0,
                'verbal_threat_count': 0,
            })
    return pd.DataFrame(rows)


def test_build_feature_matrix_runs():
    df = _make_dummy()
    out = build_feature_matrix(df)
    assert 'avg_tone_lag1' in out.columns
    assert 'avg_tone_rolling_mean_4w' in out.columns
    assert 'is_CE' in out.columns or 'is_AR' in out.columns


def test_ensure_continuous_weeks_fills_gaps():
    df = _make_dummy(n_weeks=10)
    df_gap = df.drop(df.index[3]).reset_index(drop=True)
    out = ensure_continuous_weeks(df_gap)
    assert out.groupby('country').size().nunique() == 1


def test_impute_missing_count_with_zero():
    df = _make_dummy()
    df.loc[0, 'protest_count'] = np.nan
    out = impute_missing(df)
    assert out.loc[0, 'protest_count'] == 0


def test_rolling_uses_shift_one():
    """Rolling mean for week t must NOT include week t itself."""
    df = _make_dummy(n_weeks=20)
    df.loc[df['country'] == 'CE', 'avg_tone'] = np.arange(20).astype(float)
    out = add_rolling_features(df)
    ce = out[out['country'] == 'CE'].sort_values('week_start').reset_index(drop=True)
    # First row's rolling_mean must be NaN (no past observations after shift(1))
    assert pd.isna(ce.loc[0, 'avg_tone_rolling_mean_4w'])
    # Second row's rolling_mean equals the first row's avg_tone (=0)
    assert ce.loc[1, 'avg_tone_rolling_mean_4w'] == pytest.approx(0.0)


def test_lag_features():
    df = _make_dummy(n_weeks=20)
    df.loc[df['country'] == 'CE', 'avg_tone'] = np.arange(20).astype(float)
    out = add_lag_features(df)
    ce = out[out['country'] == 'CE'].sort_values('week_start').reset_index(drop=True)
    assert ce.loc[1, 'avg_tone_lag1'] == pytest.approx(0.0)
    assert ce.loc[5, 'avg_tone_lag1'] == pytest.approx(4.0)


def test_country_one_hot_keeps_country_column():
    df = _make_dummy()
    out = add_country_one_hot(df)
    assert 'country' in out.columns
    assert any(c.startswith('is_') for c in out.columns)


# ---------------------------------------------------------------------------
# add_diff_features
# ---------------------------------------------------------------------------

def test_diff_1w_correct_value():
    df = _make_dummy(n_weeks=20)
    df.loc[df['country'] == 'CE', 'avg_tone'] = np.arange(20).astype(float)
    out = add_diff_features(df)
    ce = out[out['country'] == 'CE'].sort_values('week_start').reset_index(drop=True)
    assert pd.isna(ce.loc[0, 'avg_tone_diff_1w'])
    assert ce.loc[1, 'avg_tone_diff_1w'] == pytest.approx(1.0)
    assert ce.loc[5, 'avg_tone_diff_1w'] == pytest.approx(1.0)


def test_diff_does_not_cross_country_boundary():
    df = _make_dummy(n_weeks=20)
    out = add_diff_features(df)
    for country in ['CE', 'AR']:
        grp = out[out['country'] == country].sort_values('week_start').reset_index(drop=True)
        assert pd.isna(grp.loc[0, 'avg_tone_diff_1w'])


def test_diff_4w_correct_value():
    df = _make_dummy(n_weeks=20)
    df.loc[df['country'] == 'CE', 'avg_tone'] = np.arange(20).astype(float)
    out = add_diff_features(df)
    ce = out[out['country'] == 'CE'].sort_values('week_start').reset_index(drop=True)
    assert ce.loc[4, 'avg_tone_diff_4w'] == pytest.approx(4.0)


def test_pct_change_inf_replaced_by_nan():
    df = _make_dummy(n_weeks=20)
    df.loc[df['country'] == 'CE', 'avg_tone'] = 0.0
    out = add_diff_features(df)
    assert not np.isinf(out['avg_tone_pct_change_4w'].fillna(0)).any()


def test_diff_columns_created_for_all_targets():
    df = _make_dummy(n_weeks=20)
    out = add_diff_features(df)
    for target in ['avg_tone', 'avg_goldstein', 'n_material_conf']:
        assert f'{target}_diff_1w' in out.columns
        assert f'{target}_diff_4w' in out.columns
        assert f'{target}_pct_change_4w' in out.columns
