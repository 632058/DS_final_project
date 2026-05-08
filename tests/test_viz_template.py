"""Smoke tests for src.viz_template."""
import matplotlib
matplotlib.use('Agg')  # headless — no display needed

import matplotlib.pyplot as plt
import pytest
from unittest.mock import patch

from src import viz_template
from src.viz_template import apply_style, get_country_palette, save_fig


def test_apply_style_removes_top_spine():
    apply_style()
    assert not plt.rcParams['axes.spines.top']


def test_apply_style_removes_right_spine():
    apply_style()
    assert not plt.rcParams['axes.spines.right']


def test_apply_style_sets_savefig_dpi():
    apply_style()
    assert plt.rcParams['savefig.dpi'] == 150


def test_apply_style_enables_grid():
    apply_style()
    assert plt.rcParams['axes.grid']


def test_get_country_palette_covers_all_fips():
    palette = get_country_palette()
    for code in ['CE', 'AR', 'CI', 'TU', 'LE']:
        assert code in palette


def test_get_country_palette_values_are_hex():
    palette = get_country_palette()
    for code, color in palette.items():
        assert color.startswith('#'), f"{code}: {color!r} is not a hex string"


def test_save_fig_creates_file(tmp_path):
    apply_style()
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    with patch.object(viz_template, 'FIGURES_DIR', tmp_path):
        save_fig(fig, 'test_figure')
    assert (tmp_path / 'test_figure.png').exists()
    plt.close(fig)


def test_save_fig_custom_extension(tmp_path):
    apply_style()
    fig, ax = plt.subplots()
    ax.plot([1])
    with patch.object(viz_template, 'FIGURES_DIR', tmp_path):
        save_fig(fig, 'test_svg', ext='svg')
    assert (tmp_path / 'test_svg.svg').exists()
    plt.close(fig)


def test_save_fig_creates_figures_dir(tmp_path):
    nested = tmp_path / 'new_dir'
    apply_style()
    fig, ax = plt.subplots()
    ax.plot([1])
    with patch.object(viz_template, 'FIGURES_DIR', nested):
        save_fig(fig, 'check_mkdir')
    assert nested.exists()
    plt.close(fig)


def test_save_fig_dpi(tmp_path):
    PIL = pytest.importorskip('PIL', reason='Pillow not installed')
    from PIL import Image

    apply_style()
    fig, ax = plt.subplots()
    ax.plot([1, 2])
    with patch.object(viz_template, 'FIGURES_DIR', tmp_path):
        save_fig(fig, 'dpi_check')
    img = Image.open(tmp_path / 'dpi_check.png')
    dpi_x, _ = img.info.get('dpi', (0, 0))
    assert abs(dpi_x - 150) < 1
    plt.close(fig)
