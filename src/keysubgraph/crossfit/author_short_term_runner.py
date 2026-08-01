"""Resumable cross-fit planning for the author short-term reproduction."""

from __future__ import absolute_import, division, print_function

import sys
from pathlib import Path
from typing import List, Optional, Tuple

from keysubgraph.models.author_short_term import AUTHOR_SHORT_TERM_PROFILES
from keysubgraph.training.author_short_term_trainer import (
    author_short_term_training_config,
)


AUTHOR_SHORT_TERM_BRANCH = "author_short_term_no_coord"


def _command(*values) -> List[str]:
    return [str(value) for value in values]


def build_author_short_term_crossfit_fold_commands(
    project_root: Path,
    output_root: Path,
    fold: int,
    profile: str,
    device: str = "cuda",
    seed: Optional[int] = None,
    epochs: int = 1000,
    batch_size: int = 32,
    evaluation_batch_size: int = 32,
    num_workers: int = 0,
) -> List[Tuple[str, List[str], Path]]:
    project_root = Path(project_root).resolve()
    output_root = Path(output_root).resolve()
    if fold < 0:
        raise ValueError("cross-fit fold must be non-negative")
    if profile not in AUTHOR_SHORT_TERM_PROFILES:
        raise ValueError("unsupported author short-term profile")
    if epochs < 1 or batch_size < 1 or evaluation_batch_size < 1:
        raise ValueError("invalid author short-term run configuration")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    frozen_seed = author_short_term_training_config(
        profile, epochs=epochs, seed=seed
    ).seed
    python = sys.executable
    fold_root = output_root / "fold_{}".format(fold)
    protocol = fold_root / "protocol" / "data_protocol.json"
    branch_root = fold_root / AUTHOR_SHORT_TERM_BRANCH
    frequency = branch_root / "community_frequency.json"
    training = branch_root / "training_seed{}".format(frozen_seed)
    evaluation = branch_root / "evaluation_seed{}".format(frozen_seed)
    checkpoint = training / "best_checkpoint.pt"
    commands = [
        (
            "community_frequency",
            _command(
                python,
                "-u",
                project_root
                / "scripts"
                / "fit_paper_short_term_community_frequency.py",
                "--protocol",
                protocol,
                "--output",
                frequency,
                "--outer-fold",
                fold,
            ),
            frequency,
        ),
        (
            "train",
            _command(
                python,
                "-u",
                project_root / "scripts" / "train_author_short_term.py",
                "--protocol",
                protocol,
                "--community-frequency",
                frequency,
                "--profile",
                profile,
                "--output-dir",
                training,
                "--device",
                device,
                "--epochs",
                epochs,
                "--batch-size",
                batch_size,
                "--num-workers",
                num_workers,
                "--seed",
                frozen_seed,
            ),
            training / "best_evaluation.json",
        ),
    ]
    for split in ("validation", "test"):
        commands.append(
            (
                "evaluate_{}".format(split),
                _command(
                    python,
                    "-u",
                    project_root
                    / "scripts"
                    / "evaluate_author_short_term.py",
                    "--protocol",
                    protocol,
                    "--checkpoint",
                    checkpoint,
                    "--split",
                    split,
                    "--threshold-strategy",
                    "balanced_accuracy",
                    "--output-dir",
                    evaluation,
                    "--device",
                    device,
                    "--batch-size",
                    evaluation_batch_size,
                    "--num-workers",
                    num_workers,
                ),
                evaluation / "{}_evaluation.json".format(split),
            )
        )
    return commands

