"""Resumable three-fold plan for promoted representation-level F2."""

from __future__ import absolute_import, division, print_function

import sys
from pathlib import Path
from typing import List, Tuple


def _command(*values) -> List[str]:
    return [str(value) for value in values]


def build_svg_short_term_representation_f2_fold_commands(
    project_root: Path,
    source_crossfit_root: Path,
    g2_root: Path,
    output_root: Path,
    fold: int,
    short_term_seed: int,
    g2_seed: int = 43,
    fusion_seed: int = 42,
    device: str = "cuda",
    epochs: int = 80,
    num_workers: int = 0,
) -> List[Tuple[str, List[str], Path]]:
    project_root = Path(project_root).resolve()
    source_crossfit_root = Path(source_crossfit_root).resolve()
    g2_root = Path(g2_root).resolve()
    output_root = Path(output_root).resolve()
    if fold < 0 or epochs < 1 or num_workers < 0:
        raise ValueError("invalid representation F2 fold configuration")
    python = sys.executable
    source = source_crossfit_root / "fold_{}".format(fold)
    g2 = g2_root / "fold_{}".format(fold)
    target = output_root / "fold_{}".format(fold)
    protocol = source / "protocol" / "data_protocol.json"
    short_checkpoint = (
        source
        / "author_short_term_no_coord"
        / "training_seed{}".format(short_term_seed)
        / "best_checkpoint.pt"
    )
    g2_checkpoint = (
        g2 / "models" / "G2_seed{}".format(g2_seed) / "best_checkpoint.pt"
    )
    cache = target / "representation_cache"
    commands: List[Tuple[str, List[str], Path]] = []
    for split in ("train", "validation", "test"):
        split_output = cache / split
        commands.append(
            (
                "cache_{}".format(split),
                _command(
                    python,
                    "-u",
                    project_root
                    / "scripts"
                    / "precompute_svg_short_term_representation_f2.py",
                    "--protocol",
                    protocol,
                    "--split",
                    split,
                    "--short-term-checkpoint",
                    short_checkpoint,
                    "--g2-manifest",
                    source / "cache" / split / "manifest.json",
                    "--g2-scaler",
                    source / "scaler.json",
                    "--g2-spectral-manifest",
                    g2 / "spectral_cache" / split / "manifest.json",
                    "--g2-spectral-scaler",
                    g2 / "spectral_scaler.json",
                    "--g2-checkpoint",
                    g2_checkpoint,
                    "--output-dir",
                    split_output,
                    "--device",
                    device,
                    "--num-workers",
                    num_workers,
                ),
                split_output / "manifest.json",
            )
        )
    model = target / "model_seed{}".format(fusion_seed)
    commands.append(
        (
            "train_f2",
            _command(
                python,
                "-u",
                project_root
                / "scripts"
                / "train_svg_short_term_representation_f2.py",
                "--train-manifest",
                cache / "train" / "manifest.json",
                "--validation-manifest",
                cache / "validation" / "manifest.json",
                "--output-dir",
                model,
                "--device",
                device,
                "--epochs",
                epochs,
                "--batch-size",
                32,
                "--num-workers",
                num_workers,
                "--seed",
                fusion_seed,
                "--learning-rate",
                0.001,
                "--weight-decay",
                0.0001,
                "--gradient-clip",
                1.0,
                "--early-stopping-patience",
                15,
                "--minimum-epochs",
                5,
                "--residual-hidden-dim",
                64,
                "--dropout",
                0.20,
                "--initial-gate",
                0.01,
                "--residual-auxiliary-weight",
                0.25,
                "--gate-penalty-weight",
                0.001,
            ),
            model / "best_evaluation.json",
        )
    )
    for split in ("validation", "test"):
        output = model / "{}_evaluation.json".format(split)
        commands.append(
            (
                "evaluate_{}".format(split),
                _command(
                    python,
                    "-u",
                    project_root
                    / "scripts"
                    / "evaluate_svg_short_term_representation_f2.py",
                    "--manifest",
                    cache / split / "manifest.json",
                    "--checkpoint",
                    model / "best_checkpoint.pt",
                    "--output",
                    output,
                    "--threshold-strategy",
                    "balanced_accuracy",
                    "--device",
                    device,
                    "--batch-size",
                    64,
                    "--num-workers",
                    num_workers,
                ),
                output,
            )
        )
    return commands

