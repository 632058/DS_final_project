"""Smoke tests for src.stat_tests."""
import numpy as np
import pandas as pd

from src.stat_tests import compute_ccf, granger_test, stationarity_diff


def test_ccf_peak_at_known_lag():
    """If y_t = x_{t-3}, the CCF should peak at lag 3."""
    np.random.seed(42)
    n = 300
    x = pd.Series(np.random.randn(n), name='x')
    y = x.shift(3).rename('y')
    common = x.dropna().index.intersection(y.dropna().index)
    ccf = compute_ccf(x.loc[common], y.loc[common], max_lag=10)
    assert ccf.idxmax() == 3


def test_granger_returns_expected_structure():
    """Granger output must contain p-values per lag and a best_lag selection."""
    np.random.seed(42)
    n = 300
    x = pd.Series(np.random.randn(n), name='x')
    y = pd.Series(np.random.randn(n), name='y')
    result = granger_test(y, x, max_lag=4)
    assert set(result.keys()) >= {'p_values_per_lag', 'best_lag', 'best_p_value', 'reject_h0'}
    assert set(result['p_values_per_lag'].keys()) == {1, 2, 3, 4}
    assert result['best_lag'] in {1, 2, 3, 4}


def test_granger_independent_majority_does_not_reject():
    """Across many seeds, independent series should mostly NOT reject H0.

    With max_lag=4 and min-p over lags, false-positive rate is ~18%, so
    we only require that strictly less than half of seeds reject.
    """
    n = 300
    rejects = 0
    n_trials = 20
    for seed in range(n_trials):
        rng = np.random.default_rng(seed)
        x = pd.Series(rng.standard_normal(n), name='x')
        y = pd.Series(rng.standard_normal(n), name='y')
        if granger_test(y, x, max_lag=4)['reject_h0']:
            rejects += 1
    assert rejects < n_trials / 2


def test_granger_dependent_rejects():
    np.random.seed(42)
    n = 300
    x = pd.Series(np.random.randn(n), name='x')
    # y depends on lagged x
    y = x.shift(2).fillna(0) * 0.8 + np.random.randn(n) * 0.2
    y.name = 'y'
    result = granger_test(y, x, max_lag=5)
    assert result['reject_h0']


# ---------------------------------------------------------------------------
# stationarity_diff
# ---------------------------------------------------------------------------

def test_stationarity_diff_white_noise_returns_d0():
    np.random.seed(42)
    s = pd.Series(np.random.randn(200))
    result, d = stationarity_diff(s)
    assert d == 0
    assert len(result) > 0


def test_stationarity_diff_random_walk_returns_d1():
    np.random.seed(42)
    rw = pd.Series(np.cumsum(np.random.randn(200)))
    result, d = stationarity_diff(rw)
    assert d == 1
    assert len(result) > 0


def test_stationarity_diff_output_shorter_than_input():
    np.random.seed(42)
    rw = pd.Series(np.cumsum(np.random.randn(200)))
    result, d = stationarity_diff(rw)
    assert len(result) < len(rw)


def test_stationarity_diff_max_d_cap():
    np.random.seed(42)
    rw = pd.Series(np.cumsum(np.random.randn(200)))
    _, d = stationarity_diff(rw, max_d=1)
    assert d <= 1


def test_stationarity_diff_result_is_series():
    np.random.seed(0)
    s = pd.Series(np.random.randn(100))
    result, _ = stationarity_diff(s)
    assert isinstance(result, pd.Series)
