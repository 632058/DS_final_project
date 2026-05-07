"""Smoke tests for src.arimax_model."""
import numpy as np
import pandas as pd

from src.arimax_model import (
    diagnose_residuals,
    run_adf_test,
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
