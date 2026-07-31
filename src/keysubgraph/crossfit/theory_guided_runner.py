"""Resumable command planning for Stage-0 theory diagnostics."""

from __future__ import absolute_import, division, print_function

import sys
from pathlib import Path
from typing import List, Sequence, Tuple


def _command(*values) -> List[str]:
    return [str(value) for value in values]


def build_stage0_crossfit_commands(
    project_root: Path,
    crossfit_root: Path,
    output_root: Path,
    folds: Sequence[int] = (0, 1, 2),
    device: str = "cuda",
    fold_bootstrap_repeats: int = 2000,
    pooled_bootstrap_repeats: int = 10000,
    bootstrap_seed: int = 20260801,
    gw_max_iter: int = 100,
    gw_sinkhorn_iter: int = 100,
    gw_tolerance: float = 1.0e-7,
) -> List[Tuple[str, List[str], Path]]:
    project_root = Path(project_root).resolve()
    crossfit_root = Path(crossfit_root).resolve()
    output_root = Path(output_root).resolve()
    fold_values = tuple(int(value) for value in folds)
    if (
        not fold_values
        or len(set(fold_values)) != len(fold_values)
        or any(value < 0 for value in fold_values)
    ):
        raise ValueError("Stage-0 folds must be unique and non-negative")
    if fold_bootstrap_repeats < 1 or pooled_bootstrap_repeats < 1:
        raise ValueError("Stage-0 bootstrap repeats must be positive")
    if gw_max_iter < 1 or gw_sinkhorn_iter < 1 or gw_tolerance <= 0.0:
        raise ValueError("Stage-0 GW solver configuration is invalid")
    python = sys.executable
    assignments = crossfit_root / "assignments" / "fold_assignments.json"
    commands = []
    fold_dirs = []
    for fold in fold_values:
        source = crossfit_root / "fold_{}".format(fold)
        target = output_root / "fold_{}".format(fold)
        fold_dirs.append(target)
        commands.append(
            (
                "stage0_fold_{}".format(fold),
                _command(
                    python,
                    "-u",
                    project_root / "scripts" / "run_stage0_theory_diagnostics.py",
                    "--protocol",
                    source / "protocol" / "data_protocol.json",
                    "--hard-train-manifest",
                    source / "cache" / "train" / "manifest.json",
                    "--hard-test-manifest",
                    source / "cache" / "test" / "manifest.json",
                    "--fold-assignments",
                    assignments,
                    "--fold",
                    fold,
                    "--output-dir",
                    target,
                    "--device",
                    device,
                    "--bootstrap-repeats",
                    fold_bootstrap_repeats,
                    "--bootstrap-seed",
                    bootstrap_seed + fold,
                    "--gw-max-iter",
                    gw_max_iter,
                    "--gw-sinkhorn-iter",
                    gw_sinkhorn_iter,
                    "--gw-tolerance",
                    gw_tolerance,
                ),
                target / "manifest.json",
            )
        )
    commands.append(
        (
            "stage0_pooled_summary",
            _command(
                python,
                "-u",
                project_root / "scripts" / "summarize_theory_guided_upgrade.py",
                "--stage",
                "stage0",
                "--fold-dirs",
                *fold_dirs,
                "--output-dir",
                output_root / "pooled",
                "--bootstrap-repeats",
                pooled_bootstrap_repeats,
                "--bootstrap-seed",
                bootstrap_seed,
            ),
            output_root / "pooled" / "pooled_metrics.json",
        )
    )
    return commands
