"""Resumable command planning for Stage-0 theory diagnostics."""

from __future__ import absolute_import, division, print_function

import sys
from pathlib import Path
from typing import List, Sequence, Tuple

from keysubgraph.models.theory_guided_neural import THEORY_NEURAL_VARIANTS


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


def build_stage1_fold_commands(
    project_root: Path,
    crossfit_root: Path,
    output_root: Path,
    fold: int,
    variants: Sequence[str] = THEORY_NEURAL_VARIANTS,
    device: str = "cuda",
    seed: int = 42,
    epochs: int = 80,
    batch_size: int = 4,
    accumulation_steps: int = 2,
    num_workers: int = 2,
    gw_max_iter: int = 100,
    gw_sinkhorn_iter: int = 100,
    gw_tolerance: float = 1.0e-7,
) -> List[Tuple[str, List[str], Path]]:
    """Build one resumable Stage-1 outer-fold pipeline."""

    project_root = Path(project_root).resolve()
    crossfit_root = Path(crossfit_root).resolve()
    output_root = Path(output_root).resolve()
    fold = int(fold)
    if fold < 0:
        raise ValueError("Stage-1 fold must be non-negative")
    variants = tuple(str(value) for value in variants)
    if not variants or any(value not in THEORY_NEURAL_VARIANTS for value in variants):
        raise ValueError("Stage-1 variants are invalid")
    if batch_size * accumulation_steps < 8:
        raise ValueError("Stage-1 formal effective batch must be at least 8")
    source = crossfit_root / "fold_{}".format(fold)
    target = output_root / "fold_{}".format(fold)
    protocol = source / "protocol" / "data_protocol.json"
    selector = source / "selector" / "best_checkpoint.pt"
    python = sys.executable
    commands = []
    for split in ("train", "validation", "test"):
        cache = target / "cache" / split
        commands.append(
            (
                "cache_{}".format(split),
                _command(
                    python, "-u", project_root / "scripts" / "precompute_theory_neural_cache.py",
                    "--protocol", protocol,
                    "--split", split,
                    "--selector-checkpoint", selector,
                    "--output-dir", cache,
                    "--device", device,
                    "--num-workers", num_workers,
                    "--selection-seed", seed,
                    "--gw-max-iter", gw_max_iter,
                    "--gw-sinkhorn-iter", gw_sinkhorn_iter,
                    "--gw-tolerance", gw_tolerance,
                ),
                cache / "manifest.json",
            )
        )
    scaler = target / "cache" / "train_scaler.json"
    commands.append(
        (
            "fit_scaler",
            _command(
                python, "-u", project_root / "scripts" / "fit_theory_neural_scaler.py",
                "--train-manifest", target / "cache" / "train" / "manifest.json",
                "--output", scaler,
            ),
            scaler,
        )
    )
    for variant in variants:
        model_dir = target / "models" / "{}_seed{}".format(variant, seed)
        commands.append(
            (
                "train_{}".format(variant),
                _command(
                    python, "-u", project_root / "scripts" / "train_theory_guided_neural.py",
                    "--train-manifest", target / "cache" / "train" / "manifest.json",
                    "--validation-manifest", target / "cache" / "validation" / "manifest.json",
                    "--scaler", scaler,
                    "--variant", variant,
                    "--output-dir", model_dir,
                    "--device", device,
                    "--epochs", epochs,
                    "--batch-size", batch_size,
                    "--gradient-accumulation-steps", accumulation_steps,
                    "--num-workers", num_workers,
                    "--seed", seed,
                ),
                model_dir / "best_evaluation.json",
            )
        )
        commands.append(
            (
                "evaluate_{}".format(variant),
                _command(
                    python, "-u", project_root / "scripts" / "evaluate_theory_guided_neural.py",
                    "--manifest", target / "cache" / "test" / "manifest.json",
                    "--scaler", scaler,
                    "--checkpoint", model_dir / "best_checkpoint.pt",
                    "--output", model_dir / "outer_test_evaluation.json",
                    "--device", device,
                    "--batch-size", max(batch_size, 8),
                    "--num-workers", num_workers,
                    "--seed", seed,
                ),
                model_dir / "outer_test_evaluation.json",
            )
        )
        commands.append(
            (
                "diagnose_{}".format(variant),
                _command(
                    python, "-u", project_root / "scripts" / "diagnose_theory_guided_neural.py",
                    "--train-manifest", target / "cache" / "train" / "manifest.json",
                    "--validation-manifest", target / "cache" / "validation" / "manifest.json",
                    "--scaler", scaler,
                    "--checkpoint", model_dir / "best_checkpoint.pt",
                    "--output", model_dir / "diagnostics.json",
                    "--device", device,
                    "--batch-size", max(batch_size, 8),
                    "--num-workers", num_workers,
                    "--seed", seed,
                ),
                model_dir / "diagnostics.json",
            )
        )
    return commands
