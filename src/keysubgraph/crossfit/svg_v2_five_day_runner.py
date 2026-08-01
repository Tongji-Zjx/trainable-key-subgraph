"""Resumable command planning for the frozen five-day SVG-v2 study."""

from __future__ import absolute_import, division, print_function

import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


SVG_V2_SCREEN_CANDIDATES = ("A1", "B1", "C3", "F1", "G2")
SVG_V2_COMBINATION_CANDIDATES = ("C3_F1", "C3_G2")
SVG_V2_ALL_CANDIDATES = (
    SVG_V2_SCREEN_CANDIDATES + SVG_V2_COMBINATION_CANDIDATES
)

SVG_V2_CANDIDATE_SPECS: Dict[str, Dict[str, object]] = {
    "A1": {
        "variant": "signed_gin_multibranch_late_fusion",
        "training_recipe": "author_a1",
        "spectral_sidecar": False,
    },
    "B1": {
        "variant": "svg_v2_b1_hks",
        "training_recipe": "current",
        "spectral_sidecar": True,
    },
    "C3": {
        "variant": "svg_v2_c3_hks_diffusion",
        "training_recipe": "current",
        "spectral_sidecar": True,
    },
    "F1": {
        "variant": "signed_gin_static_anchor_residual",
        "training_recipe": "current",
        "spectral_sidecar": False,
    },
    "G2": {
        "variant": "svg_v2_g2_signed_delta_q",
        "training_recipe": "current",
        "spectral_sidecar": True,
    },
    "C3_F1": {
        "variant": "svg_v2_c3_f1_residual",
        "training_recipe": "current",
        "spectral_sidecar": True,
    },
    "C3_G2": {
        "variant": "svg_v2_c3_g2",
        "training_recipe": "current",
        "spectral_sidecar": True,
    },
}


def _command(*values) -> List[str]:
    return [str(value) for value in values]


def _base_profile(command: List[str]) -> None:
    command.extend(
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


def build_svg_v2_fold_commands(
    project_root: Path,
    source_crossfit_root: Path,
    output_root: Path,
    fold: int,
    candidates: Sequence[str] = SVG_V2_SCREEN_CANDIDATES,
    mode: str = "screen",
    device: str = "cuda",
    seed: int = 42,
    model_epochs: int = 60,
    num_workers: int = 2,
) -> List[Tuple[str, List[str], Path]]:
    """Build a leakage-safe fold plan using existing frozen hard caches."""

    project_root = Path(project_root).resolve()
    source_crossfit_root = Path(source_crossfit_root).resolve()
    output_root = Path(output_root).resolve()
    if fold < 0 or model_epochs < 1 or num_workers < 0:
        raise ValueError("invalid SVG-v2 fold configuration")
    if mode not in ("screen", "confirmatory"):
        raise ValueError("SVG-v2 mode must be screen or confirmatory")
    if not candidates or len(candidates) != len(set(candidates)):
        raise ValueError("SVG-v2 candidates must be unique and non-empty")
    unknown = set(candidates).difference(SVG_V2_ALL_CANDIDATES)
    if unknown:
        raise ValueError("unsupported SVG-v2 candidates: {}".format(unknown))

    python = sys.executable
    source_fold = source_crossfit_root / "fold_{}".format(fold)
    base_cache = source_fold / "cache"
    base_scaler = source_fold / "scaler.json"
    fold_root = output_root / "fold_{}".format(fold)
    spectral_cache = fold_root / "spectral_cache"
    spectral_scaler = fold_root / "spectral_scaler.json"
    commands: List[Tuple[str, List[str], Path]] = []
    needs_spectral = any(
        bool(SVG_V2_CANDIDATE_SPECS[name]["spectral_sidecar"])
        for name in candidates
    )
    splits = (
        ("train", "validation", "test")
        if mode == "confirmatory"
        else ("train", "validation")
    )
    if needs_spectral:
        for split in splits:
            output = spectral_cache / split
            commands.append(
                (
                    "spectral_cache_{}".format(split),
                    _command(
                        python,
                        "-u",
                        project_root
                        / "scripts"
                        / "build_sv_spectral_diffusion_cache.py",
                        "--manifest",
                        base_cache / split / "manifest.json",
                        "--output-dir",
                        output,
                        "--device",
                        device,
                    ),
                    output / "manifest.json",
                )
            )
        commands.append(
            (
                "spectral_scaler",
                _command(
                    python,
                    "-u",
                    project_root
                    / "scripts"
                    / "fit_sv_spectral_diffusion_scaler.py",
                    "--train-manifest",
                    spectral_cache / "train" / "manifest.json",
                    "--output",
                    spectral_scaler,
                ),
                spectral_scaler,
            )
        )

    for candidate in candidates:
        spec = SVG_V2_CANDIDATE_SPECS[candidate]
        variant = str(spec["variant"])
        run = fold_root / "models" / "{}_seed{}".format(candidate, seed)
        train = _command(
            python,
            "-u",
            project_root / "scripts" / "train_sv_signed_gin.py",
            "--train-manifest",
            base_cache / "train" / "manifest.json",
            "--validation-manifest",
            base_cache / "validation" / "manifest.json",
            "--scaler",
            base_scaler,
            "--variant",
            variant,
            "--training-recipe",
            spec["training_recipe"],
            "--output-dir",
            run,
            "--device",
            device,
            "--epochs",
            model_epochs,
            "--static-anchor-epochs",
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
        )
        _base_profile(train)
        if bool(spec["spectral_sidecar"]):
            train.extend(
                [
                    "--spectral-train-manifest",
                    str(spectral_cache / "train" / "manifest.json"),
                    "--spectral-validation-manifest",
                    str(
                        spectral_cache
                        / "validation"
                        / "manifest.json"
                    ),
                    "--spectral-scaler",
                    str(spectral_scaler),
                ]
            )
        commands.append(
            ("train_{}".format(candidate), train, run / "best_evaluation.json")
        )
        if mode == "confirmatory":
            evaluate = _command(
                python,
                "-u",
                project_root / "scripts" / "evaluate_sv_signed_gin.py",
                "--manifest",
                base_cache / "test" / "manifest.json",
                "--scaler",
                base_scaler,
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
            )
            if bool(spec["spectral_sidecar"]):
                evaluate.extend(
                    [
                        "--spectral-manifest",
                        str(spectral_cache / "test" / "manifest.json"),
                        "--spectral-scaler",
                        str(spectral_scaler),
                    ]
                )
            commands.append(
                (
                    "evaluate_{}".format(candidate),
                    evaluate,
                    run / "outer_test_evaluation.json",
                )
            )
    return commands
