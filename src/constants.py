"""Project-wide constants. All members should import from here."""
from pathlib import Path

# Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_DIR = PROJECT_ROOT / "figures"
REPORT_DIR = PROJECT_ROOT / "report"

# Source database
DUCKDB_PATH = DATA_DIR / "gdelt_filtered_20251001_20260428.duckdb"

# Target countries (FIPS 10-4 codes, NOT ISO 3166)
FIPS_COUNTRIES = ['CE', 'AR', 'CI', 'TU', 'LE']

COUNTRY_NAMES = {
    'CE': 'Sri Lanka',
    'AR': 'Argentina',
    'CI': 'Chile',
    'TU': 'Turkey',
    'LE': 'Lebanon',
}

COUNTRY_COLORS = {
    'CE': '#E74C3C',  # red
    'AR': '#3498DB',  # blue
    'CI': '#2ECC71',  # green
    'TU': '#F39C12',  # orange
    'LE': '#9B59B6',  # purple
}

# CAMEO root codes used in this project
PROTEST_CODE = '14'
VIOLENCE_CODES = ['18', '19', '20']
VERBAL_THREAT_CODES = ['10', '11']

# Feature engineering hyperparameters
ROLLING_WINDOWS = [4, 8, 12]
LAG_RANGE = list(range(1, 13))

# Train/test split ratio
TRAIN_RATIO = 0.8

# ---------------------------------------------------------------------------
# ML pipeline output layout (dual-branch: target-scope × target-transform)
# ---------------------------------------------------------------------------

ML_RESULTS_DIR = OUTPUT_DIR / "ml_results"

# Scope = target column used for the next-week prediction
#   'all'      -> protest_count_all   (all-scope, current ML main line)
#   'domestic' -> protest_count       (Actor1=Actor2=country, domestic-only)
TARGET_SCOPES = ('all', 'domestic')

# Transform = forward function applied to y before fitting
#   'raw'   -> identity
#   'log1p' -> log(1+y)
TARGET_TRANSFORMS = ('raw', 'log1p')


def ml_results_subdir(scope: str, transform: str):
    """Return output/ml_results/{scope}/{transform}/ for the given branch.

    Does not create the directory; callers should mkdir(parents=True, exist_ok=True).
    """
    if scope not in TARGET_SCOPES:
        raise ValueError(f"unknown target scope: {scope!r}")
    if transform not in TARGET_TRANSFORMS:
        raise ValueError(f"unknown target transform: {transform!r}")
    return ML_RESULTS_DIR / scope / transform


def figures_subdir(scope: str):
    """Return figures/{scope}/ for the given branch.

    Callers should mkdir(parents=True, exist_ok=True).
    """
    if scope not in TARGET_SCOPES:
        raise ValueError(f"unknown target scope: {scope!r}")
    return FIGURES_DIR / scope


def is_default_branch(scope: str, transform: str) -> bool:
    """True if (scope, transform) is the legacy default that should also be
    mirrored to the top-level paths (output/ml_results/*, figures/*).

    Other members' scripts and report figures consume those top-level paths,
    so we always dual-write the default branch to preserve backward compat.
    """
    return scope == 'all' and transform == 'raw'
