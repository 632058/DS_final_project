"""Generate Fig 9 — prediction intervals (PI) from three quantile models.

Combines three XGBoost ``reg:quantileerror`` runs trained separately at
α=0.1 / 0.5 / 0.9 into a single visualization that surfaces predictive
uncertainty around the median forecast. Implements ``docs/model_evaluation_2026-05-24.md``
"用 XGBoost 兩個 quantile（0.1 / 0.9）跑兩遍，把 fig5 變成「pred + PI」散點圖。"

Branches (CLI flags):
- ``--target-scope {all,domestic}``
- ``--target-transform {raw,log1p}``

Inputs (output/ml_results/{scope}/{transform}/):
- xgboost_predictions_quantile_0.1.parquet  (lower)
- xgboost_predictions_quantile_0.5.parquet  (median, point prediction)
- xgboost_predictions_quantile_0.9.parquet  (upper)

Outputs (figures/{scope}/):
- fig9_pi_scatter_{transform}.png  — actual vs median, error bars = [Q0.1, Q0.9]
- fig9_pi_timeseries_{transform}.png  — per-country time series with PI band
- fig9_pi_coverage_{transform}.png  — empirical coverage by country
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.constants import COUNTRY_COLORS, COUNTRY_NAMES, FIPS_COUNTRIES
from src.ml_cli import add_branch_args, resolve_paths
from src.viz_template import apply_style

# Quantile alphas the script expects to find on disk.
LOWER_ALPHA = 0.1
MEDIAN_ALPHA = 0.5
UPPER_ALPHA = 0.9


def _load_quantile(out_dir, alpha: float) -> pd.DataFrame:
    path = out_dir / f"xgboost_predictions_quantile_{alpha:g}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing quantile prediction file: {path}. "
            f"Run: python scripts/10_train_xgboost.py "
            f"--target-scope ... --target-transform ... "
            f"--objective quantile --quantile-alpha {alpha}"
        )
    return pd.read_parquet(path)


def _merge_quantiles(out_dir) -> pd.DataFrame:
    """Return per-row DataFrame with actual, q_lower, q_median, q_upper."""
    lower = _load_quantile(out_dir, LOWER_ALPHA).rename(
        columns={'predicted': 'q_lower'}
    )
    median = _load_quantile(out_dir, MEDIAN_ALPHA).rename(
        columns={'predicted': 'q_median'}
    )
    upper = _load_quantile(out_dir, UPPER_ALPHA).rename(
        columns={'predicted': 'q_upper'}
    )

    merged = lower.merge(
        median[['country', 'week_start', 'q_median']],
        on=['country', 'week_start'],
    ).merge(
        upper[['country', 'week_start', 'q_upper']],
        on=['country', 'week_start'],
    )

    # XGBoost quantile training fits each level independently so q_lower can
    # occasionally exceed q_upper for a row. Sort the three quantiles within
    # each row so the PI band is monotone (a standard post-hoc fix).
    sorted_q = np.sort(
        merged[['q_lower', 'q_median', 'q_upper']].to_numpy(), axis=1
    )
    merged['q_lower'] = sorted_q[:, 0]
    merged['q_median'] = sorted_q[:, 1]
    merged['q_upper'] = sorted_q[:, 2]
    return merged


def _coverage_table(df: pd.DataFrame) -> pd.DataFrame:
    """Empirical coverage = fraction of rows with q_lower <= actual <= q_upper."""
    rows = []
    for code, g in df.groupby('country'):
        covered = ((g['actual'] >= g['q_lower']) & (g['actual'] <= g['q_upper'])).mean()
        width = (g['q_upper'] - g['q_lower']).mean()
        rows.append({
            'country': code,
            'n': int(len(g)),
            'coverage': float(covered),
            'mean_width': float(width),
        })
    pooled = (
        (df['actual'] >= df['q_lower']) & (df['actual'] <= df['q_upper'])
    ).mean()
    rows.append({
        'country': 'POOLED',
        'n': int(len(df)),
        'coverage': float(pooled),
        'mean_width': float((df['q_upper'] - df['q_lower']).mean()),
    })
    return pd.DataFrame(rows)


def _plot_pi_scatter(df: pd.DataFrame, title_suffix: str):
    """Scatter of actual vs q_median with [q_lower, q_upper] error bars."""
    fig, ax = plt.subplots(figsize=(9, 8))
    for code in FIPS_COUNTRIES:
        sub = df[df['country'] == code]
        if sub.empty:
            continue
        yerr = np.vstack([
            (sub['q_median'] - sub['q_lower']).clip(lower=0),
            (sub['q_upper'] - sub['q_median']).clip(lower=0),
        ])
        ax.errorbar(
            sub['actual'], sub['q_median'],
            yerr=yerr,
            fmt='o', markersize=4,
            color=COUNTRY_COLORS[code],
            ecolor=COUNTRY_COLORS[code], elinewidth=0.6,
            alpha=0.55, capsize=0,
            label=f"{COUNTRY_NAMES[code]} ({code}) — n={len(sub)}",
        )
    lim = float(max(df['actual'].max(), df['q_upper'].max())) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', linewidth=1.0, label='Perfect prediction')
    ax.set_xlim(-5, lim)
    ax.set_ylim(-5, lim)
    ax.set_xlabel('Actual protest count')
    ax.set_ylabel('Predicted (median) with [Q0.1, Q0.9]')
    ax.set_title(
        f'XGBoost quantile regression — median + 80% PI{title_suffix}'
    )
    ax.legend(loc='upper left', fontsize=9)
    fig.tight_layout()
    return fig


def _plot_pi_timeseries(df: pd.DataFrame, title_suffix: str):
    """Per-country time series with PI band shaded around median + actual line."""
    countries = [c for c in FIPS_COUNTRIES if not df[df['country'] == c].empty]
    n = len(countries)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.4 * n), sharex=False)
    if n == 1:
        axes = [axes]
    for ax, code in zip(axes, countries):
        sub = df[df['country'] == code].sort_values('week_start')
        ax.fill_between(
            sub['week_start'], sub['q_lower'], sub['q_upper'],
            color=COUNTRY_COLORS[code], alpha=0.2, linewidth=0,
            label='80% PI',
        )
        ax.plot(
            sub['week_start'], sub['q_median'],
            color=COUNTRY_COLORS[code], linewidth=1.4, label='Median forecast',
        )
        ax.plot(
            sub['week_start'], sub['actual'],
            color='black', linewidth=0.9, alpha=0.7, label='Actual',
        )
        covered = (
            (sub['actual'] >= sub['q_lower']) & (sub['actual'] <= sub['q_upper'])
        ).mean()
        ax.set_title(
            f"{COUNTRY_NAMES[code]} ({code}) — coverage={covered*100:.0f}%, "
            f"mean PI width={(sub['q_upper']-sub['q_lower']).mean():.1f}",
            fontsize=11,
        )
        ax.set_ylabel('Protest count')
        ax.legend(loc='upper left', fontsize=8, ncol=3)
    axes[-1].set_xlabel('Week')
    fig.suptitle(
        f"Quantile-regression prediction intervals over time{title_suffix}",
        y=1.005, fontsize=12,
    )
    fig.tight_layout()
    return fig


def _plot_coverage_bar(coverage: pd.DataFrame, title_suffix: str):
    """Bar chart of empirical coverage vs the nominal 80% level."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    # Per-country bars; POOLED row drawn separately to the right.
    per_country = coverage[coverage['country'].isin(FIPS_COUNTRIES)]
    pooled_row = coverage[coverage['country'] == 'POOLED'].iloc[0]
    xs = [COUNTRY_NAMES[c] for c in per_country['country']]
    ys = per_country['coverage'].to_numpy() * 100
    colors = [COUNTRY_COLORS[c] for c in per_country['country']]
    ax.bar(xs, ys, color=colors, alpha=0.85, edgecolor='white')
    ax.bar(['POOLED'], [pooled_row['coverage'] * 100],
           color='#555555', alpha=0.85, edgecolor='white')
    ax.axhline(80, color='k', linestyle='--', linewidth=1.0,
               label='Nominal 80% level')
    ax.set_ylim(0, 105)
    ax.set_ylabel('Empirical coverage (%)')
    ax.set_title(f'PI coverage vs nominal 80%{title_suffix}')
    ax.legend(loc='lower right', fontsize=9)
    for i, v in enumerate(list(ys) + [pooled_row['coverage'] * 100]):
        ax.text(i, v + 1.5, f'{v:.0f}%', ha='center', fontsize=9)
    fig.tight_layout()
    return fig


def main(scope: str, transform: str) -> None:
    apply_style()
    paths = resolve_paths(scope=scope, transform=transform)

    df = _merge_quantiles(paths.out_dir)
    coverage = _coverage_table(df)
    print(f"\n=== PI coverage (scope={scope}, transform={transform}) ===")
    print(coverage.round(3).to_string(index=False))

    title_suffix = f"  [{scope}/{transform}]"

    fig_scatter = _plot_pi_scatter(df, title_suffix)
    scatter_path = paths.fig_dir / f"fig9_pi_scatter_{transform}.png"
    fig_scatter.savefig(scatter_path, dpi=150, bbox_inches='tight')
    plt.close(fig_scatter)
    print(f"Saved {scatter_path}")

    fig_ts = _plot_pi_timeseries(df, title_suffix)
    ts_path = paths.fig_dir / f"fig9_pi_timeseries_{transform}.png"
    fig_ts.savefig(ts_path, dpi=150, bbox_inches='tight')
    plt.close(fig_ts)
    print(f"Saved {ts_path}")

    fig_cov = _plot_coverage_bar(coverage, title_suffix)
    cov_path = paths.fig_dir / f"fig9_pi_coverage_{transform}.png"
    fig_cov.savefig(cov_path, dpi=150, bbox_inches='tight')
    plt.close(fig_cov)
    print(f"Saved {cov_path}")

    # Persist merged PI table so the report can quote per-row PI widths.
    merged_path = paths.out_dir / f"xgboost_pi_merged_{transform}.parquet"
    df.to_parquet(merged_path, index=False)
    coverage_path = paths.out_dir / f"xgboost_pi_coverage_{transform}.parquet"
    coverage.to_parquet(coverage_path, index=False)
    print(f"Saved {merged_path}")
    print(f"Saved {coverage_path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_branch_args(parser)
    return parser


if __name__ == '__main__':
    args = _build_parser().parse_args()
    main(scope=args.target_scope, transform=args.target_transform)
