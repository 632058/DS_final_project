"""Tests for src.ml_cli branch helpers."""
import argparse

import pytest

from src.constants import FIGURES_DIR, ML_RESULTS_DIR
from src.ml_cli import add_branch_args, resolve_paths


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_branch_args(parser)
    return parser


def test_defaults_are_all_and_raw():
    args = _build_parser().parse_args([])
    assert args.target_scope == 'all'
    assert args.target_transform == 'raw'


def test_resolve_paths_default_branch_is_top_level_mirror():
    paths = resolve_paths(scope='all', transform='raw')
    assert paths.scope == 'all'
    assert paths.transform == 'raw'
    assert paths.also_top_level is True
    assert paths.out_dir == ML_RESULTS_DIR / 'all' / 'raw'
    assert paths.fig_dir == FIGURES_DIR / 'all'
    assert paths.legacy_out_dir == ML_RESULTS_DIR
    assert paths.legacy_fig_dir == FIGURES_DIR


def test_resolve_paths_non_default_branch_does_not_mirror():
    for scope, transform in [
        ('all', 'log1p'),
        ('domestic', 'raw'),
        ('domestic', 'log1p'),
    ]:
        paths = resolve_paths(scope=scope, transform=transform)
        assert paths.also_top_level is False, (scope, transform)
        assert paths.out_dir == ML_RESULTS_DIR / scope / transform


def test_resolve_paths_creates_directories():
    paths = resolve_paths(scope='domestic', transform='log1p')
    assert paths.out_dir.exists()
    assert paths.fig_dir.exists()


def test_resolve_paths_rejects_unknown_branch():
    with pytest.raises(ValueError, match="unknown target scope"):
        resolve_paths(scope='crossborder', transform='raw')
    with pytest.raises(ValueError, match="unknown target transform"):
        resolve_paths(scope='all', transform='sqrt')
