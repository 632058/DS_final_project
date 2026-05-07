"""Shared matplotlib style for all figures."""
import matplotlib.pyplot as plt

from src.constants import COUNTRY_COLORS, FIGURES_DIR


def apply_style() -> None:
    """Apply consistent matplotlib style. Call once at the top of each plotting script."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.titleweight': 'bold',
        'axes.labelsize': 11,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'legend.frameon': False,
        'legend.fontsize': 10,
        'figure.dpi': 100,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
    })


def get_country_palette() -> dict:
    """Return country code -> color hex mapping."""
    return COUNTRY_COLORS


def save_fig(fig, name: str, ext: str = 'png') -> None:
    """Save figure to figures/ with consistent dpi and naming."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f'{name}.{ext}', dpi=150, bbox_inches='tight')
