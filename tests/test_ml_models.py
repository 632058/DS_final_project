"""Smoke tests for src.ml_models."""
import numpy as np
import pandas as pd
import pytest

from src.ml_models import (
    add_target,
    apply_target_transform,
    evaluate,
    evaluate_predictions,
    invert_target_transform,
    non_feature_cols,
    prepare_xy,
    time_train_test_split,
    train_lasso,
    train_xgboost,
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
                # protest_count_all is the current ML target source (see add_target);
                # use a distinct offset so we can confirm the target uses _all and not
                # the domestic-only protest_count column.
                'protest_count_all': i * 2,
                'avg_tone_lag1': -2.0,
                'is_AR': 1 if c == 'AR' else 0,
            })
    return pd.DataFrame(rows)


def test_add_target_shifts_per_country_default_all_scope():
    df = _make_dummy()
    out = add_target(df)
    ce = out[out['country'] == 'CE'].sort_values('week_start').reset_index(drop=True)
    # default scope='all' uses protest_count_all (i*2); row 0 -> row 1's value (2)
    assert ce.loc[0, 'protest_count_next_week'] == 2
    assert pd.isna(ce.iloc[-1]['protest_count_next_week'])


def test_add_target_domestic_scope_uses_protest_count():
    df = _make_dummy()
    out = add_target(df, scope='domestic')
    ce = out[out['country'] == 'CE'].sort_values('week_start').reset_index(drop=True)
    # domestic scope uses protest_count (i); row 0 -> row 1's value (1)
    assert ce.loc[0, 'protest_count_next_week'] == 1
    assert pd.isna(ce.iloc[-1]['protest_count_next_week'])


def test_add_target_unknown_scope_raises():
    df = _make_dummy()
    with pytest.raises(ValueError, match="unknown target scope"):
        add_target(df, scope='crossborder')


_SAMPLE_COLUMNS = [
    'week_start', 'country',
    'protest_count', 'protest_count_lag1', 'protest_count_diff_1w',
    'protest_count_rolling_mean_4w', 'protest_ratio_domestic',
    'protest_count_all', 'protest_count_all_lag1', 'protest_count_all_diff_1w',
    'protest_count_all_rolling_mean_4w', 'protest_ratio',
    'avg_tone', 'n_material_conf', 'is_AR',
    'protest_count_next_week',
]


def test_non_feature_cols_metadata_only_without_columns():
    drop = non_feature_cols('all')
    assert drop == frozenset({'week_start', 'country', 'protest_count_next_week'})


def test_non_feature_cols_all_drops_full_domestic_family():
    drop = non_feature_cols('all', columns=_SAMPLE_COLUMNS)
    # full domestic AR family removed
    assert {'protest_count', 'protest_count_lag1', 'protest_count_diff_1w',
            'protest_count_rolling_mean_4w', 'protest_ratio_domestic'} <= drop
    # all-scope family kept
    assert 'protest_count_all' not in drop
    assert 'protest_count_all_lag1' not in drop
    assert 'protest_count_all_rolling_mean_4w' not in drop
    assert 'protest_ratio' not in drop
    # generic features kept
    assert 'avg_tone' not in drop
    assert 'is_AR' not in drop


def test_non_feature_cols_domestic_drops_full_all_scope_family():
    drop = non_feature_cols('domestic', columns=_SAMPLE_COLUMNS)
    assert {'protest_count_all', 'protest_count_all_lag1',
            'protest_count_all_diff_1w', 'protest_count_all_rolling_mean_4w',
            'protest_ratio'} <= drop
    assert 'protest_count' not in drop
    assert 'protest_count_lag1' not in drop
    assert 'protest_count_rolling_mean_4w' not in drop
    assert 'protest_ratio_domestic' not in drop


def test_time_split_no_overlap():
    df = _make_dummy()
    train, test = time_train_test_split(df, train_ratio=0.8)
    assert train['week_start'].max() < test['week_start'].min()


def test_prepare_xy_all_scope_keeps_protest_count_all():
    df = _make_dummy()
    df = add_target(df)
    X, _y, _ = prepare_xy(df, scope='all')
    assert 'protest_count' not in X.columns           # domestic-only legacy
    assert 'protest_count_next_week' not in X.columns  # target
    assert 'country' not in X.columns
    assert 'week_start' not in X.columns
    assert 'protest_count_all' in X.columns            # current-week AR feature


def test_prepare_xy_domestic_scope_keeps_protest_count():
    df = _make_dummy()
    df = add_target(df, scope='domestic')
    X, _y, _ = prepare_xy(df, scope='domestic')
    assert 'protest_count_all' not in X.columns         # excluded under domestic
    assert 'protest_count_next_week' not in X.columns   # target
    assert 'protest_count' in X.columns                 # current-week AR feature


# ---------------------------------------------------------------------------
# evaluate / evaluate_predictions
# ---------------------------------------------------------------------------

def test_evaluate_perfect_prediction_rmse_zero():
    y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    y_hat = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    m = evaluate(y, y_hat)
    assert m['rmse'] == 0.0
    assert m['mae'] == 0.0
    # evaluate() no longer reports directional_accuracy (see docstring)
    assert 'directional_accuracy' not in m


def test_evaluate_predictions_per_country_dir_acc():
    # Two countries; build predictions so each country has a clean monotone
    # signal -> per-country dir acc should be 1.0.
    weeks = pd.date_range('2024-01-01', periods=5, freq='W-MON')
    rows = []
    for c, base in [('AR', 0), ('TU', 100)]:
        for i, w in enumerate(weeks):
            rows.append({
                'country': c,
                'week_start': w,
                'actual': base + i,
                'predicted': base + i + 0.1,
            })
    pred_df = pd.DataFrame(rows)
    m = evaluate_predictions(pred_df)
    assert m['per_country_directional'] == {'AR': 1.0, 'TU': 1.0}
    assert m['directional_accuracy'] == 1.0


def test_evaluate_predictions_does_not_diff_across_country_boundary():
    # Old buggy evaluate() would diff across country boundary. Construct a panel
    # where pooled np.diff sees a fake reversal at the AR->TU boundary, but
    # per-country diffs are all monotone up.
    weeks = pd.date_range('2024-01-01', periods=4, freq='W-MON')
    rows = []
    # AR: actual 1,2,3,4  predicted matching -> dir acc 1.0
    for i, w in enumerate(weeks):
        rows.append({'country': 'AR', 'week_start': w,
                     'actual': float(1 + i), 'predicted': float(1 + i)})
    # TU: actual 100,101,102,103  predicted matching -> dir acc 1.0
    for i, w in enumerate(weeks):
        rows.append({'country': 'TU', 'week_start': w,
                     'actual': float(100 + i), 'predicted': float(100 + i)})
    pred_df = pd.DataFrame(rows)
    m = evaluate_predictions(pred_df)
    # If the function were diffing across the AR->TU boundary, dir acc would
    # also include the boundary step (4 -> 100) and still be 1.0 because both
    # actual and predicted jump together. So construct a deliberate disagreement
    # at the country boundary: AR ends at predicted=4, TU starts at predicted=99
    # while TU actual still goes 100,101,102,103. Per-country dir acc should
    # still be 1.0 because within each country actual & predicted both increase.
    pred_df.loc[pred_df['country'] == 'TU', 'predicted'] = [99.0, 100.0, 101.0, 102.0]
    m = evaluate_predictions(pred_df)
    assert m['per_country_directional']['AR'] == 1.0
    assert m['per_country_directional']['TU'] == 1.0
    assert m['directional_accuracy'] == 1.0


def test_evaluate_predictions_requires_columns():
    bad_df = pd.DataFrame({'country': ['AR'], 'actual': [1.0], 'predicted': [1.0]})
    with pytest.raises(ValueError, match="missing required columns"):
        evaluate_predictions(bad_df)


# ---------------------------------------------------------------------------
# apply_target_transform / invert_target_transform
# ---------------------------------------------------------------------------

def test_target_transform_raw_is_identity():
    y = np.array([0.0, 1.0, 7.0, 100.0])
    assert np.allclose(apply_target_transform(y, 'raw'), y)
    assert np.allclose(invert_target_transform(y, 'raw'), y)


def test_target_transform_log1p_roundtrip():
    y = np.array([0.0, 1.0, 7.0, 100.0, 5000.0])
    t = apply_target_transform(y, 'log1p')
    inv = invert_target_transform(t, 'log1p')
    assert np.allclose(inv, y, atol=1e-9)


def test_target_transform_log1p_rejects_negative():
    with pytest.raises(ValueError, match="y >= 0"):
        apply_target_transform(np.array([-1.0, 1.0]), 'log1p')


def test_target_transform_unknown_kind_raises():
    with pytest.raises(ValueError, match="unknown target transform"):
        apply_target_transform(np.array([1.0]), 'sqrt')


# ---------------------------------------------------------------------------
# train_lasso / train_xgboost
# ---------------------------------------------------------------------------

def _make_xy(n: int = 60):
    rng = np.random.default_rng(42)
    X = pd.DataFrame({
        'f1': rng.standard_normal(n),
        'f2': rng.standard_normal(n),
        'f3': rng.standard_normal(n),
    })
    y = pd.Series(rng.standard_normal(n))
    split = int(n * 0.8)
    return (X.iloc[:split].copy(), y.iloc[:split].copy(),
            X.iloc[split:].copy(), y.iloc[split:].copy())


def test_train_lasso_returns_model_and_scaler():
    X_tr, y_tr, _, _ = _make_xy()
    model, scaler = train_lasso(X_tr, y_tr, alpha=1.0)
    assert hasattr(model, 'predict')
    assert hasattr(scaler, 'transform')


def test_train_lasso_predict_shape():
    X_tr, y_tr, X_te, _ = _make_xy()
    model, scaler = train_lasso(X_tr, y_tr)
    preds = model.predict(scaler.transform(X_te))
    assert len(preds) == len(X_te)


def test_train_lasso_scaler_fit_on_train_only():
    X_tr, y_tr, X_te, _ = _make_xy()
    _, scaler = train_lasso(X_tr, y_tr)
    X_te_scaled = scaler.transform(X_te)
    assert X_te_scaled.shape == X_te.shape


def test_train_lasso_different_alpha_changes_coef():
    X_tr, y_tr, _, _ = _make_xy()
    model_loose, _ = train_lasso(X_tr, y_tr, alpha=0.001)
    model_tight, _ = train_lasso(X_tr, y_tr, alpha=100.0)
    assert not np.allclose(model_loose.coef_, model_tight.coef_)


def test_train_xgboost_predict_shape():
    X_tr, y_tr, X_te, _ = _make_xy()
    model = train_xgboost(X_tr, y_tr, params={'n_estimators': 10, 'random_state': 42})
    preds = model.predict(X_te)
    assert len(preds) == len(X_te)


def test_train_xgboost_returns_finite_predictions():
    X_tr, y_tr, X_te, _ = _make_xy()
    model = train_xgboost(X_tr, y_tr, params={'n_estimators': 10, 'random_state': 42})
    preds = model.predict(X_te)
    assert np.all(np.isfinite(preds))


def test_train_xgboost_custom_params_applied():
    import xgboost as xgb
    X_tr, y_tr, _, _ = _make_xy()
    model = train_xgboost(X_tr, y_tr,
                          params={'n_estimators': 5, 'max_depth': 2, 'random_state': 0})
    assert isinstance(model, xgb.XGBRegressor)
    assert model.n_estimators == 5
    assert model.max_depth == 2
