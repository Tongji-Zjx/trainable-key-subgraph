"""Stage planning for one resumable structured short-term cross-fit fold."""

from __future__ import absolute_import, division, print_function

import sys
from pathlib import Path
from typing import List, Tuple

from keysubgraph.models.structured_short_term import (
    PAPER_ALIGNED_VARIANT,
    STRUCTURED_SAFE_VARIANT,
)


def _command(*values) -> List[str]:
    return [str(value) for value in values]


def build_structured_short_term_crossfit_fold_commands(
    project_root: Path,
    output_root: Path,
    fold: int,
    device: str = "cuda",
    seed: int = 42,
    epochs: int = 80,
    batch_size: int = 4,
    evaluation_batch_size: int = 8,
    num_workers: int = 2,
    model_variant: str = STRUCTURED_SAFE_VARIANT,
) -> List[Tuple[str, List[str], Path]]:
    """Return ordered fold-local commands and completion artifacts."""

    project_root = Path(project_root).resolve()
    output_root = Path(output_root).resolve()
    if fold < 0:
        raise ValueError("cross-fit fold must be non-negative")
    if epochs < 1 or batch_size < 1 or evaluation_batch_size < 1:
        raise ValueError("invalid structured short-term training configuration")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if model_variant not in (
        STRUCTURED_SAFE_VARIANT,
        PAPER_ALIGNED_VARIANT,
    ):
        raise ValueError("unsupported structured short-term variant")

    python = sys.executable
    fold_root = output_root / "fold_{}".format(fold)
    protocol = fold_root / "protocol" / "data_protocol.json"
    branch_name = (
        "paper_aligned_short_term"
        if model_variant == PAPER_ALIGNED_VARIANT
        else "structured_short_term"
    )
    branch_root = fold_root / branch_name
    standardizer = branch_root / "standardizer.json"
    training = branch_root / "training_seed{}".format(seed)
    evaluation = branch_root / "evaluation_seed{}".format(seed)
    checkpoint = training / "best_checkpoint.pt"

    commands = [
        (
            "standardizer",
            _command(
                python,
                "-u",
                project_root
                / "scripts"
                / "fit_structured_short_term_standardizer.py",
                "--protocol",
                protocol,
                "--output",
                standardizer,
            ),
            standardizer,
        ),
        (
            "train",
            _command(
                python,
                "-u",
                project_root / "scripts" / "train_structured_short_term.py",
                "--protocol",
                protocol,
                "--standardizer",
                standardizer,
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
                seed,
                "--learning-rate",
                0.001,
                "--weight-decay",
                0.0001,
                "--gradient-clip",
                1.0,
                "--early-stopping-patience",
                15,
                "--selection-metric",
                "roc_auc",
                "--model-variant",
                model_variant,
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
                    / "evaluate_structured_short_term.py",
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
