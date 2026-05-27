"""Generate publication-quality ARIMAX figures.

Run: python scripts/16_make_arimax_forecast_figure.py
Output:
- figures/fig7_arimax_forecast.png
- figures/fig7b_arimax_lag_comparison.png when comparison output exists
- figures/fig7c_log1p_sarimax_forecast.png when log1p forecasts exist
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/ds_final_project_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/ds_final_project_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.constants import COUNTRY_COLORS, COUNTRY_NAMES, FIGURES_DIR, FIPS_COUNTRIES, OUTPUT_DIR

SKILL_SCRIPT_DIRS = [
    Path.home() / ".codex" / "skills" / "beautiful-plots" / "scripts",
    Path.home() / ".dotfiles" / "codex" / ".codex" / "skills" / "beautiful-plots" / "scripts",
]
for skill_script_dir in SKILL_SCRIPT_DIRS:
    if skill_script_dir.exists():
        sys.path.insert(0, str(skill_script_dir))
        break

try:
    from paper_plot_utils import apply_paper_style, save_figure
except ImportError:  # pragma: no cover - fallback for environments without the local skill.
    from src.viz_template import apply_style as apply_paper_style

    def save_figure(fig, save_path, *, close=False, pad_inches=0.02):
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", pad_inches=pad_inches)
        if close:
            plt.close(fig)
        return path


FORECAST_DIR = OUTPUT_DIR / "arimax_results"
FORECAST_STEM = "fig7_arimax_forecast"
COMPARISON_STEM = "fig7b_arimax_lag_comparison"
LOG1P_FORECAST_STEM = "fig7c_log1p_sarimax_forecast"


def _save_png(fig, stem: str) -> None:
    save_figure(fig, FIGURES_DIR / f"{stem}.png")


def _load_country_summary(country: str, suffix: str = "") -> dict:
    path = FORECAST_DIR / f"{country}{suffix}.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_forecasts(suffix: str = "") -> dict[str, pd.DataFrame]:
    forecasts = {}
    for country in FIPS_COUNTRIES:
        path = FORECAST_DIR / f"{country}{suffix}_forecast.parquet"
        if path.exists():
            forecasts[country] = pd.read_parquet(path).sort_values("week_start")
    return forecasts


def _draw_forecast_panel(
    ax,
    x: pd.Series,
    actual: np.ndarray,
    predicted: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    actual_color: str,
    pred_color: str,
    ci_color: str,
) -> None:
    """Draw one forecast panel with Seaborn lines and a precomputed CI ribbon."""
    ax.fill_between(x, lower, upper, color=ci_color, alpha=0.18, linewidth=0)
    sns.lineplot(
        x=x,
        y=actual,
        ax=ax,
        color=actual_color,
        linewidth=2.1,
        marker="o",
        markersize=3.4,
        errorbar=None,
        sort=False,
    )
    sns.lineplot(
        x=x,
        y=predicted,
        ax=ax,
        color=pred_color,
        linewidth=2.2,
        errorbar=None,
        sort=False,
    )
    ax.axhline(0, color="#90A4AE", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel("")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    sns.despine(ax=ax)


def make_forecast_figure(
    forecasts: dict[str, pd.DataFrame],
    *,
    figsize: tuple[float, float] = (18, 10),
    summary_suffix: str = "",
    title: str = "ARIMAX Out-of-Sample Forecasts",
    forecast_label: str = "ARIMAX forecast",
    pred_color: str | None = None,
    ci_color: str | None = None,
) -> plt.Figure:
    fig, axes = plt.subplots(3, 2, figsize=figsize, sharex=False)
    axes_flat = axes.ravel()
    actual_color = "#263238"

    for ax, country in zip(axes_flat, FIPS_COUNTRIES):
        pred = forecasts.get(country)
        if pred is None or pred.empty:
            ax.text(0.5, 0.5, f"{country}: forecast not found", ha="center", va="center")
            ax.set_axis_off()
            continue

        summary = _load_country_summary(country, suffix=summary_suffix)
        x = pd.to_datetime(pred["week_start"])
        actual = pred["actual"].to_numpy(dtype=float)
        predicted = pred["predicted"].to_numpy(dtype=float)
        lower = pred["ci_lower"].to_numpy(dtype=float)
        upper = pred["ci_upper"].to_numpy(dtype=float)
        panel_pred_color = pred_color or COUNTRY_COLORS[country]
        panel_ci_color = ci_color or panel_pred_color

        _draw_forecast_panel(
            ax,
            x,
            actual,
            predicted,
            lower,
            upper,
            actual_color=actual_color,
            pred_color=panel_pred_color,
            ci_color=panel_ci_color,
        )

        rmse = summary.get("rmse")
        mae = summary.get("mae")
        metric_text = ""
        if rmse is not None and mae is not None:
            metric_text = f"  RMSE={rmse:.2f}, MAE={mae:.2f}"
        ax.set_title(f"{COUNTRY_NAMES[country]} ({country}){metric_text}", loc="left", pad=8)
        ax.set_ylabel("Weekly protest count")
        ax.tick_params(axis="x", rotation=25)
        ax.margins(x=0.01)

        y_max = np.nanmax([actual.max(), predicted.max(), upper.max()])
        ax.set_ylim(bottom=min(0, float(np.nanmin(lower)) * 0.05), top=max(1.0, y_max * 1.08))

    axes_flat[-1].set_axis_off()
    legend_handles = [
        Line2D([0], [0], color=actual_color, marker="o", linewidth=2.1, markersize=5, label="Actual"),
        Line2D([0], [0], color="#607D8B", linewidth=2.0, label=forecast_label),
        Line2D([0], [0], color="#607D8B", linewidth=8, alpha=0.25, label="95% interval"),
    ]
    axes_flat[-1].legend(handles=legend_handles, loc="center", frameon=False, fontsize=14)
    fig.suptitle(title, fontsize=22, fontweight="bold", y=0.99)
    plt.tight_layout()
    fig.subplots_adjust(top=0.92, hspace=0.46, wspace=0.25)
    return fig


def make_lag_comparison_figure(
    comparison: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (14, 5),
) -> plt.Figure:
    plot_df = comparison.copy()
    plot_df["Country"] = plot_df["country"].map(lambda c: f"{COUNTRY_NAMES[c]} ({c})")
    lag_label = plot_df["spec_name"].map({
        "baseline_lag_1_2_4": "Fixed",
        "ccf_selected_lag": "CCF",
    }).fillna(plot_df["spec_name"])
    if "target_transform" in plot_df.columns and plot_df["target_transform"].nunique() > 1:
        transform_label = plot_df["target_transform"].map({"raw": "raw", "log1p": "log1p"})
        plot_df["Model specification"] = transform_label + " + " + lag_label
    else:
        plot_df["Model specification"] = lag_label

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True)
    palette = {
        "Fixed": "#8C8C8C",
        "CCF": "#0072B2",
        "raw + Fixed": "#8C8C8C",
        "raw + CCF": "#0072B2",
        "log1p + Fixed": "#E69F00",
        "log1p + CCF": "#009E73",
    }
    hue_order = [label for label in palette if label in set(plot_df["Model specification"])]
    country_order = [f"{COUNTRY_NAMES[c]} ({c})" for c in FIPS_COUNTRIES]
    for ax, metric, label in zip(axes, ["rmse", "mae"], ["RMSE", "MAE"]):
        sns.barplot(
            data=plot_df,
            x="Country",
            y=metric,
            hue="Model specification",
            order=country_order,
            hue_order=hue_order,
            palette=palette,
            ax=ax,
            edgecolor="white",
            linewidth=0.8,
        )
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", fontsize=7, padding=2, rotation=90)

        # Mark the lowest-error model by outline/hatch so value labels stay readable.
        metric_max = float(plot_df[metric].max())
        for country_idx, country_label in enumerate(country_order):
            country_rows = plot_df[plot_df["Country"] == country_label]
            if country_rows.empty:
                continue
            best = country_rows.loc[country_rows[metric].idxmin()]
            hue_idx = hue_order.index(best["Model specification"])
            patch = ax.containers[hue_idx].patches[country_idx]
            patch.set_edgecolor("#212121")
            patch.set_linewidth(1.6)
            patch.set_hatch("///")

        ax.set_ylim(0, metric_max * 1.22)
        ax.set_title(label, loc="left", pad=8)
        ax.set_xlabel("")
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=25)
        ax.legend_.remove()

    handles, labels = axes[0].get_legend_handles_labels()
    best_handle = Patch(facecolor="white", edgecolor="#212121", hatch="///", linewidth=1.4)
    handles.append(best_handle)
    labels.append("Best model")
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("ARIMAX Lag Specification Comparison", fontsize=20, fontweight="bold", y=1.13)
    plt.tight_layout()
    fig.subplots_adjust(top=0.78, wspace=0.22)
    return fig


def main() -> None:
    apply_paper_style(font_scale=0.78)
    sns.set_theme(
        context="paper",
        style="whitegrid",
        rc={
            "axes.edgecolor": "#CFD8DC",
            "grid.color": "#ECEFF1",
            "grid.linewidth": 0.9,
        },
    )

    forecasts = _load_forecasts()
    forecast_fig = make_forecast_figure(forecasts)
    _save_png(forecast_fig, FORECAST_STEM)
    plt.close(forecast_fig)
    print(f"Saved {FIGURES_DIR / (FORECAST_STEM + '.png')}")

    log1p_forecasts = _load_forecasts(suffix="_log1p")
    if log1p_forecasts:
        log1p_fig = make_forecast_figure(
            log1p_forecasts,
            summary_suffix="_log1p",
            title="log1p-SARIMAX Back-Transformed Forecasts",
            forecast_label="log1p-SARIMAX forecast",
        )
        _save_png(log1p_fig, LOG1P_FORECAST_STEM)
        plt.close(log1p_fig)
        print(f"Saved {FIGURES_DIR / (LOG1P_FORECAST_STEM + '.png')}")

    comparison_path = FORECAST_DIR / "arimax_model_comparison.parquet"
    if comparison_path.exists():
        comparison = pd.read_parquet(comparison_path)
        comparison_fig = make_lag_comparison_figure(comparison)
        _save_png(comparison_fig, COMPARISON_STEM)
        plt.close(comparison_fig)
        print(f"Saved {FIGURES_DIR / (COMPARISON_STEM + '.png')}")


if __name__ == '__main__':
    main()
