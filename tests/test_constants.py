"""Sanity checks for src.constants — structure, types, and value ranges."""
from pathlib import Path

from src.constants import (
    COUNTRY_COLORS,
    COUNTRY_NAMES,
    DATA_DIR,
    FIGURES_DIR,
    FIPS_COUNTRIES,
    LAG_RANGE,
    OUTPUT_DIR,
    PROJECT_ROOT,
    PROTEST_CODE,
    REPORT_DIR,
    ROLLING_WINDOWS,
    TRAIN_RATIO,
    VIOLENCE_CODES,
)


def test_project_root_is_path():
    assert isinstance(PROJECT_ROOT, Path)


def test_all_dirs_are_path_objects():
    for d in [DATA_DIR, OUTPUT_DIR, FIGURES_DIR, REPORT_DIR]:
        assert isinstance(d, Path)


def test_duckdb_path_under_data_dir():
    from src.constants import DUCKDB_PATH
    assert isinstance(DUCKDB_PATH, Path)
    assert DUCKDB_PATH.parent == DATA_DIR


def test_fips_countries_has_five_entries():
    assert len(FIPS_COUNTRIES) == 5


def test_fips_countries_are_strings():
    assert all(isinstance(c, str) for c in FIPS_COUNTRIES)


def test_country_names_keys_match_fips():
    assert set(COUNTRY_NAMES.keys()) == set(FIPS_COUNTRIES)


def test_country_colors_keys_match_fips():
    assert set(COUNTRY_COLORS.keys()) == set(FIPS_COUNTRIES)


def test_country_colors_are_hex():
    for code, color in COUNTRY_COLORS.items():
        assert color.startswith('#'), f"{code}: {color!r} is not hex"
        assert len(color) == 7, f"{code}: {color!r} is not a 6-digit hex"


def test_protest_code_is_string():
    assert isinstance(PROTEST_CODE, str)
    assert PROTEST_CODE == '14'


def test_violence_codes_are_list_of_strings():
    assert isinstance(VIOLENCE_CODES, list)
    assert all(isinstance(c, str) for c in VIOLENCE_CODES)


def test_rolling_windows_sorted_ascending():
    assert ROLLING_WINDOWS == sorted(ROLLING_WINDOWS)
    assert len(ROLLING_WINDOWS) >= 1


def test_lag_range_starts_at_one():
    assert LAG_RANGE[0] == 1
    assert all(isinstance(x, int) for x in LAG_RANGE)


def test_train_ratio_in_open_unit_interval():
    assert 0 < TRAIN_RATIO < 1
