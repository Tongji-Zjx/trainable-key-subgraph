"""Resumable paired OOF plan for corrected neural S/V and ST fusion."""

from __future__ import absolute_import, division, print_function

import sys
from pathlib import Path
from typing import List, Sequence, Tuple

from keysubgraph.models.neuralized_sv import NEURALIZED_SV_VARIANTS


def _command(*values) -> List[str]:
    return [str(value) for value in values]


def build_neuralized_sv_fold_commands(
    project_root: Path,
    crossfit_root: Path,
    output_root: Path,
    fold: int,
    variants: Sequence[str] = NEURALIZED_SV_VARIANTS,
    device: str = "cuda",
    seed: int = 42,
    epochs: int = 80,
    batch_size: int = 4,
    accumulation_steps: int = 2,
    gw_max_iter: int = 100,
    gw_sinkhorn_iter: int = 100,
    gw_tolerance: float = 1.0e-7,
) -> List[Tuple[str, List[str], Path]]:
    project_root = Path(project_root).resolve()
    crossfit_root = Path(crossfit_root).resolve()
    output_root = Path(output_root).resolve()
    fold = int(fold)
    variants = tuple(str(value) for value in variants)
    if fold < 0 or not variants:
        raise ValueError("invalid corrected neural S/V fold plan")
    if any(value not in NEURALIZED_SV_VARIANTS for value in variants):
        raise ValueError("unsupported corrected neural S/V variant")
    if batch_size * accumulation_steps < 8:
        raise ValueError("formal corrected neural S/V effective batch must be >=8")
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
                    python,
                    "-u",
                    project_root / "scripts" / "precompute_theory_neural_cache.py",
                    "--protocol",
                    protocol,
                    "--split",
                    split,
                    "--selector-checkpoint",
                    selector,
                    "--output-dir",
                    cache,
                    "--device",
                    device,
                    "--num-workers",
                    0,
                    "--selection-seed",
                    seed,
                    "--gw-max-iter",
                    gw_max_iter,
                    "--gw-sinkhorn-iter",
                    gw_sinkhorn_iter,
                    "--gw-tolerance",
                    gw_tolerance,
                ),
                cache / "manifest.json",
            )
        )
    scaler = target / "cache" / "train_scaler.json"
    commands.append(
        (
            "fit_scaler",
            _command(
                python,
                "-u",
                project_root / "scripts" / "fit_theory_neural_scaler.py",
                "--train-manifest",
                target / "cache" / "train" / "manifest.json",
                "--output",
                scaler,
            ),
            scaler,
        )
    )
    for variant in variants:
        model = target / "models" / "{}_seed{}".format(variant, seed)
        commands.append(
            (
                "train_{}".format(variant),
                _command(
                    python,
                    "-u",
                    project_root / "scripts" / "train_neuralized_sv.py",
                    "--train-manifest",
                    target / "cache" / "train" / "manifest.json",
                    "--validation-manifest",
                    target / "cache" / "validation" / "manifest.json",
                    "--scaler",
                    scaler,
                    "--variant",
                    variant,
                    "--output-dir",
                    model,
                    "--device",
                    device,
                    "--epochs",
                    epochs,
                    "--batch-size",
                    batch_size,
                    "--gradient-accumulation-steps",
                    accumulation_steps,
                    "--num-workers",
                    0,
                    "--seed",
                    seed,
                ),
                model / "best_evaluation.json",
            )
        )
        for split in ("validation", "test"):
            evaluation = model / "{}_evaluation.json".format(split)
            commands.append(
                (
                    "evaluate_{}_{}".format(variant, split),
                    _command(
                        python,
                        "-u",
                        project_root / "scripts" / "evaluate_neuralized_sv.py",
                        "--manifest",
                        target / "cache" / split / "manifest.json",
                        "--scaler",
                        scaler,
                        "--checkpoint",
                        model / "best_checkpoint.pt",
                        "--output",
                        evaluation,
                        "--device",
                        device,
                        "--batch-size",
                        max(batch_size, 8),
                        "--num-workers",
                        0,
                        "--seed",
                        seed,
                    ),
                    evaluation,
                )
            )
    return commands


def build_neuralized_sv_short_term_fusion_command(
    project_root: Path,
    source_crossfit_root: Path,
    neuralized_root: Path,
    output_root: Path,
    fold: int,
    variant: str,
    short_term_seed: int,
    neural_seed: int = 42,
) -> Tuple[str, List[str], Path]:
    if variant not in NEURALIZED_SV_VARIANTS:
        raise ValueError("unsupported corrected neural S/V fusion variant")
    project_root = Path(project_root).resolve()
    source = Path(source_crossfit_root).resolve() / "fold_{}".format(fold)
    neural = (
        Path(neuralized_root).resolve()
        / "fold_{}".format(fold)
        / "models"
        / "{}_seed{}".format(variant, neural_seed)
    )
    short = (
        source
        / "author_short_term_no_coord"
        / "evaluation_seed{}".format(short_term_seed)
    )
    output = (
        Path(output_root).resolve()
        / "fold_{}".format(fold)
        / "{}_seed{}".format(variant, neural_seed)
    )
    command = _command(
        sys.executable,
        "-u",
        project_root / "scripts" / "fit_evaluate_svg_v2_f0_fusion.py",
        "--fit-short-term",
        short / "validation_evaluation.json",
        "--fit-svg",
        neural / "validation_evaluation.json",
        "--evaluate-short-term",
        short / "test_evaluation.json",
        "--evaluate-svg",
        neural / "test_evaluation.json",
        "--fusion-protocol",
        "strict_crossfit",
        "--l1-weight",
        0.001,
        "--optimization-steps",
        2000,
        "--output-dir",
        output,
    )
    return "fuse_{}".format(variant), command, output / "evaluation.json"

