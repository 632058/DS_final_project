"""CLI helpers shared by the ML training / collation / figure scripts.

Every script that reads or writes branch-specific artefacts goes through this
module, so the output layout and the backward-compatibility rules live in a
single place. To add a new branch dimension later, change this file plus
``src.constants.ml_results_subdir`` / ``figures_subdir`` only.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from src.constants import (
    FIGURES_DIR,
    ML_RESULTS_DIR,
    TARGET_SCOPES,
    TARGET_TRANSFORMS,
    figures_subdir,
    is_default_branch,
    ml_results_subdir,
)


@dataclass(frozen=True)
class BranchPaths:
    """Resolved output paths for a (scope, transform) branch.

    out_dir / fig_dir always point at the branch-specific subdirectory
    (``output/ml_results/{scope}/{transform}/`` and ``figures/{scope}/``).
    When ``also_top_level`` is True, scripts should ALSO mirror their
    primary outputs to the legacy top-level paths so other members' scripts
    continue to find them.
    """
    scope: str
    transform: str
    out_dir: Path
    fig_dir: Path
    legacy_out_dir: Path
    legacy_fig_dir: Path
    also_top_level: bool


def add_branch_args(parser: argparse.ArgumentParser) -> None:
    """Register --target-scope and --target-transform on the given parser.

    Defaults are ``all`` and ``raw``, matching the legacy single-branch
    behaviour so unflagged invocations remain backward compatible.
    """
    parser.add_argument(
        '--target-scope',
        choices=list(TARGET_SCOPES),
        default='all',
        help="Which protest count to predict (default: all).",
    )
    parser.add_argument(
        '--target-transform',
        choices=list(TARGET_TRANSFORMS),
        default='raw',
        help="Forward transform on y before fitting (default: raw).",
    )


# ---------------------------------------------------------------------------
# Objective handling (shared by XGBoost training / diagnostic / SHAP / figure
# scripts so the filename-suffix and CLI flag schema live in one place).
# ---------------------------------------------------------------------------

OBJECTIVE_CHOICES = ('squarederror', 'poisson', 'tweedie', 'quantile')

# CLI alias -> XGBoost objective string.
_XGB_OBJECTIVE_NAME = {
    'squarederror': 'reg:squarederror',
    'poisson': 'count:poisson',
    'tweedie': 'reg:tweedie',
    'quantile': 'reg:quantileerror',
}


def add_objective_args(parser: argparse.ArgumentParser) -> None:
    """Register --objective, --tweedie-variance-power, --quantile-alpha.

    Defaults preserve legacy ``reg:squarederror`` behaviour so unflagged
    invocations still produce the canonical output filenames.
    """
    parser.add_argument(
        '--objective',
        choices=list(OBJECTIVE_CHOICES),
        default='squarederror',
        help=(
            "XGBoost loss. squarederror is the legacy default; "
            "poisson/tweedie are count-aware (require --target-transform raw); "
            "quantile fits an asymmetric pinball loss at --quantile-alpha."
        ),
    )
    parser.add_argument(
        '--tweedie-variance-power',
        type=float,
        default=1.5,
        help="Tweedie variance power in (1, 2). 1=Poisson, 2=Gamma.",
    )
    parser.add_argument(
        '--quantile-alpha',
        type=float,
        default=0.5,
        help="Quantile level in (0, 1). 0.5 is median regression.",
    )


def objective_suffix(
    objective: str,
    tweedie_variance_power: float = 1.5,
    quantile_alpha: float = 0.5,
) -> str:
    """Filename suffix used to disambiguate non-default objectives.

    Empty for ``squarederror`` so canonical artefacts (e.g.
    ``xgboost_predictions.parquet``) keep working with downstream scripts
    that still hard-code them.
    """
    if objective == 'squarederror':
        return ''
    if objective == 'poisson':
        return '_poisson'
    if objective == 'tweedie':
        return f'_tweedie_{tweedie_variance_power:g}'
    if objective == 'quantile':
        return f'_quantile_{quantile_alpha:g}'
    raise ValueError(f"unknown objective: {objective!r}")


def build_xgb_objective_params(
    objective: str,
    tweedie_variance_power: float = 1.5,
    quantile_alpha: float = 0.5,
) -> dict:
    """Return only the objective-specific XGBoost params (not base hparams).

    Callers merge this into their own ``DEFAULT_PARAMS`` dict so the base
    XGBoost hyperparameters (n_estimators, max_depth, ...) stay co-located
    with the training script that owns them.
    """
    params: dict = {'objective': _XGB_OBJECTIVE_NAME[objective]}
    if objective == 'tweedie':
        params['tweedie_variance_power'] = tweedie_variance_power
    elif objective == 'quantile':
        params['quantile_alpha'] = quantile_alpha
        # XGBoost 2.x quantile loss requires the histogram tree method.
        params['tree_method'] = 'hist'
    return params


def validate_objective_args(
    objective: str,
    transform: str,
    tweedie_variance_power: float,
    quantile_alpha: float,
) -> None:
    """Reject objective + hparam combinations that double-transform y or
    fall outside valid hyperparameter ranges. Raises ``SystemExit`` with a
    message suitable for CLI error reporting.
    """
    if objective in ('poisson', 'tweedie') and transform != 'raw':
        raise SystemExit(
            f"--objective {objective} requires --target-transform raw "
            f"(got {transform!r}). Poisson/Tweedie already model the count "
            f"distribution; applying log1p on top double-transforms the target."
        )
    if objective == 'tweedie' and not (1.0 < tweedie_variance_power < 2.0):
        raise SystemExit(
            f"--tweedie-variance-power must be in (1, 2), got "
            f"{tweedie_variance_power}"
        )
    if objective == 'quantile' and not (0.0 < quantile_alpha < 1.0):
        raise SystemExit(
            f"--quantile-alpha must be in (0, 1), got {quantile_alpha}"
        )


def resolve_paths(scope: str, transform: str) -> BranchPaths:
    """Compute output directories for a single branch invocation.

    Creates the branch-specific subdirectories (and legacy top-level dirs
    when the branch is the default), so callers can write immediately.
    """
    out_dir = ml_results_subdir(scope, transform)
    fig_dir = figures_subdir(scope)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    also_top_level = is_default_branch(scope, transform)
    if also_top_level:
        ML_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    return BranchPaths(
        scope=scope,
        transform=transform,
        out_dir=out_dir,
        fig_dir=fig_dir,
        legacy_out_dir=ML_RESULTS_DIR,
        legacy_fig_dir=FIGURES_DIR,
        also_top_level=also_top_level,
    )
