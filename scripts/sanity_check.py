"""Run sanity checks on the produced parquet outputs.

Run: python scripts/sanity_check.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.constants import OUTPUT_DIR


def check_country_weekly() -> bool:
    path = OUTPUT_DIR / "country_weekly.parquet"
    if not path.exists():
        print(f"[SKIP] {path} not found")
        return True
    df = pd.read_parquet(path)
    ok = True

    if df['country'].isna().any():
        print("[FAIL] country has NaN")
        ok = False
    if not df['avg_tone'].between(-100, 100).all():
        print("[FAIL] avg_tone out of [-100, 100]")
        ok = False
    if not df['avg_goldstein'].between(-10, 10).all():
        print("[FAIL] avg_goldstein out of [-10, 10]")
        ok = False
    for col in ['n_events', 'protest_count', 'violence_count']:
        if (df[col] < 0).any():
            print(f"[FAIL] {col} has negative values")
            ok = False
    if ok:
        print(f"[PASS] country_weekly: shape={df.shape}")
        print(df.groupby('country').size().to_string())
    return ok


def check_feature_matrix() -> bool:
    path = OUTPUT_DIR / "feature_matrix.parquet"
    if not path.exists():
        print(f"[SKIP] {path} not found")
        return True
    df = pd.read_parquet(path)
    ok = True

    numeric = df.select_dtypes(include='number')
    if numeric.isin([np.inf, -np.inf]).any().any():
        print("[FAIL] feature_matrix contains inf")
        ok = False
    sizes = df.groupby('country').size().nunique()
    if sizes != 1:
        print(f"[WARN] feature_matrix has unequal weeks per country (nunique={sizes})")
    if ok:
        print(f"[PASS] feature_matrix: shape={df.shape}")
    return ok


def main() -> None:
    results = [
        check_country_weekly(),
        check_feature_matrix(),
    ]
    if not all(results):
        sys.exit(1)


if __name__ == '__main__':
    main()
