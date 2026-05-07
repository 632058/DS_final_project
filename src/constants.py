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
