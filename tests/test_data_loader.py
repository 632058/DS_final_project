"""Smoke tests for src.data_loader."""
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch

from src import data_loader


def _write_dummy_parquet(tmp_path: Path, filename: str) -> pd.DataFrame:
    df = pd.DataFrame({
        'country': ['CE', 'AR'],
        'week_start': [pd.Timestamp('2024-01-01'), pd.Timestamp('2024-01-08')],
    })
    (tmp_path / filename).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(tmp_path / filename, index=False)
    return df


def test_load_country_weekly_returns_dataframe(tmp_path):
    expected = _write_dummy_parquet(tmp_path, 'country_weekly.parquet')
    with patch.object(data_loader, 'OUTPUT_DIR', tmp_path):
        df = data_loader.load_country_weekly()
    pd.testing.assert_frame_equal(df, expected)


def test_load_country_monthly_returns_dataframe(tmp_path):
    expected = _write_dummy_parquet(tmp_path, 'country_monthly.parquet')
    with patch.object(data_loader, 'OUTPUT_DIR', tmp_path):
        df = data_loader.load_country_monthly()
    pd.testing.assert_frame_equal(df, expected)


def test_load_eventmix_weekly_returns_dataframe(tmp_path):
    expected = _write_dummy_parquet(tmp_path, 'country_eventmix_weekly.parquet')
    with patch.object(data_loader, 'OUTPUT_DIR', tmp_path):
        df = data_loader.load_eventmix_weekly()
    pd.testing.assert_frame_equal(df, expected)


def test_load_feature_matrix_returns_dataframe(tmp_path):
    expected = _write_dummy_parquet(tmp_path, 'feature_matrix.parquet')
    with patch.object(data_loader, 'OUTPUT_DIR', tmp_path):
        df = data_loader.load_feature_matrix()
    pd.testing.assert_frame_equal(df, expected)


def test_load_missing_file_raises(tmp_path):
    with patch.object(data_loader, 'OUTPUT_DIR', tmp_path):
        with pytest.raises(Exception):
            data_loader.load_country_weekly()


def test_load_country_weekly_preserves_columns(tmp_path):
    df_in = pd.DataFrame({
        'country': ['CE'],
        'week_start': [pd.Timestamp('2024-01-01')],
        'protest_count': [3],
        'avg_tone': [-1.5],
    })
    df_in.to_parquet(tmp_path / 'country_weekly.parquet', index=False)
    with patch.object(data_loader, 'OUTPUT_DIR', tmp_path):
        df_out = data_loader.load_country_weekly()
    assert list(df_out.columns) == list(df_in.columns)
    assert len(df_out) == 1
