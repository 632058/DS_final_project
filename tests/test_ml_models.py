"""Smoke tests for src.ml_models."""
import numpy as np
import pandas as pd

from src.ml_models import (
    add_target,
    evaluate,
    prepare_xy,
    time_train_test_split,
)


def _make_dummy() -> pd.DataFrame:
    countries = ['CE', 'AR']
    rows = []
    weeks = pd.date_range('2024-01-01', periods=30, freq='W-MON')
    for c in countries:
        for i, w in enumerate(weeks):
            rows.append({
                'country': c,
                'week_start': w,
                'protest_count': i,
                'avg_tone_lag1': -2.0,
                'is_AR': 1 if c == 'AR' else 0,
            })
    return pd.DataFrame(rows)


def test_add_target_shifts_per_country():
    df = _make_dummy()
    out = add_target(df)
    ce = out[out['country'] == 'CE'].sort_values('week_start').reset_index(drop=True)
    assert ce.loc[0, 'protest_count_next_week'] == 1
    assert pd.isna(ce.iloc[-1]['protest_count_next_week'])


def test_time_split_no_overlap():
    df = _make_dummy()
    train, test = time_train_test_split(df, train_ratio=0.8)
    assert train['week_start'].max() < test['week_start'].min()


def test_prepare_xy_drops_non_features():
    df = _make_dummy()
    df = add_target(df)
    X, y, _ = prepare_xy(df)
    assert 'protest_count' not in X.columns
    assert 'protest_count_next_week' not in X.columns
    assert 'country' not in X.columns
    assert 'week_start' not in X.columns


def test_evaluate_perfect_prediction():
    y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    y_hat = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    m = evaluate(y, y_hat)
    assert m['rmse'] == 0
    assert m['directional_accuracy'] == 1.0
