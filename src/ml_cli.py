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
