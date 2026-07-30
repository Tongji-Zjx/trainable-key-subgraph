"""Stage planning for one resumable SV Signed-GIN cross-fit fold."""

from __future__ import absolute_import, division, print_function

import sys
from pathlib import Path
from typing import List, Sequence, Tuple


SV_CROSSFIT_VARIANTS = (
    "sv_static_variation",
    "static_spectral_only",
    "static_spectral_variation_late_fusion",
    "signed_gin_variation",
    "signed_gin_static_variation",
    "signed_gin_multibranch_late_fusion",
    "signed_gin_static_anchor_residual",
    "signed_gin_static_anchor_residual_attention",
)


def _command(*values) -> List[str]:
    return [str(value) for value in values]


def build_sv_crossfit_fold_commands(
    project_root: Path,
    output_root: Path,
    fold: int,
    variants: Sequence[str] = ("signed_gin_static_variation",),
    device: str = "cuda",
    seed: int = 42,
    selector_epochs: int = 80,
    model_epochs: int = 80,
    num_workers: int = 2,
) -> List[Tuple[str, List[str], Path]]:
    """Return ordered commands and immutable completion artifacts."""

    project_root = Path(project_root).resolve()
    output_root = Path(output_root).resolve()
    if fold < 0:
        raise ValueError("cross-fit fold must be non-negative")
    if not variants or len(set(variants)) != len(variants):
        raise ValueError("cross-fit variants must be unique and non-empty")
    if any(value not in SV_CROSSFIT_VARIANTS for value in variants):
        raise ValueError("unsupported SV cross-fit variant")
    if selector_epochs < 1 or model_epochs < 1 or num_workers < 0:
        raise ValueError("invalid SV cross-fit training configuration")

    python = sys.executable
    fold_root = output_root / "fold_{}".format(fold)
    protocol = fold_root / "protocol" / "data_protocol.json"
    selector = fold_root / "selector"
    checkpoint = selector / "best_checkpoint.pt"
    cache = fold_root / "cache"
    scaler = fold_root / "scaler.json"
    commands = [
        (
            "selector",
            _command(
                python,
                "-u",
                project_root / "scripts" / "train_dual_selector.py",
                "--protocol",
                protocol,
                "--output-dir",
                selector,
                "--device",
                device,
                "--epochs",
                selector_epochs,
                "--batch-size",
                1,
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
            ),
            selector / "best_evaluation.json",
        )
    ]
    for split in ("train", "validation", "test"):
        split_dir = cache / split
        commands.append(
            (
                "cache_{}".format(split),
                _command(
                    python,
                    "-u",
                    project_root
                    / "scripts"
                    / "precompute_sv_signed_gin_cache.py",
                    "--protocol",
                    protocol,
                    "--selector-checkpoint",
                    checkpoint,
                    "--selection-mode",
                    "learned",
                    "--split",
                    split,
                    "--output-dir",
                    split_dir,
                    "--device",
                    device,
                    "--num-workers",
                    num_workers,
                    "--selection-seed",
                    seed,
                ),
                split_dir / "manifest.json",
            )
        )
    commands.append(
        (
            "scaler",
            _command(
                python,
                "-u",
                project_root / "scripts" / "fit_sv_signed_gin_scalers.py",
                "--train-manifest",
                cache / "train" / "manifest.json",
                "--output",
                scaler,
            ),
            scaler,
        )
    )
    for variant in variants:
        run = fold_root / "models" / "{}_seed{}".format(variant, seed)
        commands.append(
            (
                "train_{}".format(variant),
                _command(
                    python,
                    "-u",
                    project_root / "scripts" / "train_sv_signed_gin.py",
                    "--train-manifest",
                    cache / "train" / "manifest.json",
                    "--validation-manifest",
                    cache / "validation" / "manifest.json",
                    "--scaler",
                    scaler,
                    "--variant",
                    variant,
                    "--output-dir",
                    run,
                    "--device",
                    device,
                    "--epochs",
                    model_epochs,
                    "--batch-size",
                    4,
                    "--gradient-accumulation-steps",
                    2,
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
                    "composite_auc",
                ),
                run / "best_evaluation.json",
            )
        )
        if variant in (
            "static_spectral_only",
            "static_spectral_variation_late_fusion",
            "signed_gin_multibranch_late_fusion",
            "signed_gin_static_anchor_residual",
            "signed_gin_static_anchor_residual_attention",
        ):
            train_command = commands[-1][1]
            train_command.extend(
                [
                    "--message-mode",
                    "signed_normalized",
                    "--pooling",
                    "mean_std",
                    "--gin-residual",
                    "--gin-jumping-knowledge",
                    "--gin-compact-readout",
                    "--gin-batch-normalization",
                    "--auxiliary-loss-weight",
                    "0.25",
                ]
            )
        if variant in (
            "signed_gin_static_anchor_residual",
            "signed_gin_static_anchor_residual_attention",
        ):
            train_command.extend(
                [
                    "--static-anchor-epochs",
                    str(model_epochs),
                    "--residual-gate-penalty-weight",
                    "0.01",
                ]
            )
        if (
            variant
            == "signed_gin_static_anchor_residual_attention"
        ):
            train_command.append("--gin-residual-attention")
        commands.append(
            (
                "evaluate_{}".format(variant),
                _command(
                    python,
                    "-u",
                    project_root / "scripts" / "evaluate_sv_signed_gin.py",
                    "--manifest",
                    cache / "test" / "manifest.json",
                    "--scaler",
                    scaler,
                    "--checkpoint",
                    run / "best_checkpoint.pt",
                    "--threshold-strategy",
                    "balanced_accuracy",
                    "--output",
                    run / "outer_test_evaluation.json",
                    "--device",
                    device,
                    "--batch-size",
                    4,
                    "--num-workers",
                    num_workers,
                ),
                run / "outer_test_evaluation.json",
            )
        )
    return commands
