"""Resumable planning for the frozen three-day SVG improvement study."""

from __future__ import absolute_import, division, print_function

import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


SVG_THREE_DAY_SCREEN_CANDIDATES = ("D1", "H1", "E1")
SVG_THREE_DAY_COMBINATION_CANDIDATES = ("D1_H1",)
SVG_THREE_DAY_ALL_CANDIDATES = (
    ("BASELINE",)
    + SVG_THREE_DAY_SCREEN_CANDIDATES
    + SVG_THREE_DAY_COMBINATION_CANDIDATES
)
SVG_THREE_DAY_BUDGETS = (
    ("n35_e20", 0.35, 0.20),
    ("n50_e30", 0.50, 0.30),
    ("n65_e40", 0.65, 0.40),
)

SVG_THREE_DAY_CANDIDATE_SPECS: Dict[str, Dict[str, object]] = {
    "BASELINE": {
        "variant": "signed_gin_multibranch_late_fusion",
        "budget_source": "existing",
        "site_class_balanced": False,
    },
    "D1": {
        "variant": "svg_v2_d1_community_pooling",
        "budget_source": "middle",
        "site_class_balanced": False,
    },
    "H1": {
        "variant": "signed_gin_multibranch_late_fusion",
        "budget_source": "existing",
        "site_class_balanced": True,
    },
    "E1": {
        "variant": "svg_v2_e1_multi_budget",
        "budget_source": "multi",
        "site_class_balanced": False,
    },
    "D1_H1": {
        "variant": "svg_v2_d1_community_pooling",
        "budget_source": "middle",
        "site_class_balanced": True,
    },
}


def _command(*values) -> List[str]:
    return [str(value) for value in values]


def _default_svg_profile(command: List[str]) -> None:
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


def _budget_paths(fold_root: Path):
    return {
        name: {
            "root": fold_root / "budget_cache" / name,
            "scaler": fold_root / "budget_scalers" / "{}.json".format(name),
            "node_ratio": node_ratio,
            "edge_ratio": edge_ratio,
        }
        for name, node_ratio, edge_ratio in SVG_THREE_DAY_BUDGETS
    }


def build_svg_three_day_fold_commands(
    project_root: Path,
    source_crossfit_root: Path,
    output_root: Path,
    fold: int,
    candidates: Sequence[str] = SVG_THREE_DAY_SCREEN_CANDIDATES,
    mode: str = "screen",
    device: str = "cuda",
    seed: int = 42,
    selection_seed: int = 42,
    model_epochs: int = 60,
    num_workers: int = 0,
) -> List[Tuple[str, List[str], Path]]:
    """Build one leakage-safe fold without retraining the selector."""

    project_root = Path(project_root).resolve()
    source_crossfit_root = Path(source_crossfit_root).resolve()
    output_root = Path(output_root).resolve()
    if fold < 0 or model_epochs < 1 or num_workers < 0:
        raise ValueError("invalid three-day SVG fold configuration")
    if mode not in ("screen", "confirmatory"):
        raise ValueError("three-day SVG mode must be screen or confirmatory")
    if not candidates or len(candidates) != len(set(candidates)):
        raise ValueError("three-day SVG candidates must be unique and non-empty")
    unknown = set(candidates).difference(SVG_THREE_DAY_ALL_CANDIDATES)
    if unknown:
        raise ValueError("unsupported three-day SVG candidates: {}".format(unknown))

    python = sys.executable
    source_fold = source_crossfit_root / "fold_{}".format(fold)
    protocol = source_fold / "protocol" / "data_protocol.json"
    selector = source_fold / "selector" / "best_checkpoint.pt"
    existing_cache = source_fold / "cache"
    existing_scaler = source_fold / "scaler.json"
    fold_root = output_root / "fold_{}".format(fold)
    budgets = _budget_paths(fold_root)
    splits = (
        ("train", "validation", "test")
        if mode == "confirmatory"
        else ("train", "validation")
    )
    required_budget_names = set()
    for candidate in candidates:
        source = SVG_THREE_DAY_CANDIDATE_SPECS[candidate]["budget_source"]
        if source == "middle":
            required_budget_names.add("n50_e30")
        elif source == "multi":
            required_budget_names.update(budgets)

    commands: List[Tuple[str, List[str], Path]] = []
    for budget_name in sorted(required_budget_names):
        budget = budgets[budget_name]
        for split in splits:
            output = budget["root"] / split
            commands.append(
                (
                    "cache_{}_{}".format(budget_name, split),
                    _command(
                        python,
                        "-u",
                        project_root
                        / "scripts"
                        / "precompute_sv_signed_gin_cache.py",
                        "--protocol",
                        protocol,
                        "--selector-checkpoint",
                        selector,
                        "--selection-mode",
                        "learned",
                        "--split",
                        split,
                        "--output-dir",
                        output,
                        "--device",
                        device,
                        "--num-workers",
                        num_workers,
                        "--selection-seed",
                        selection_seed,
                        "--node-ratio",
                        budget["node_ratio"],
                        "--edge-ratio",
                        budget["edge_ratio"],
                    ),
                    output / "manifest.json",
                )
            )
        commands.append(
            (
                "scaler_{}".format(budget_name),
                _command(
                    python,
                    "-u",
                    project_root / "scripts" / "fit_sv_signed_gin_scalers.py",
                    "--train-manifest",
                    budget["root"] / "train" / "manifest.json",
                    "--output",
                    budget["scaler"],
                ),
                budget["scaler"],
            )
        )

    middle = budgets["n50_e30"]
    ordered_budgets = [budgets[name] for name, _, _ in SVG_THREE_DAY_BUDGETS]
    for candidate in candidates:
        spec = SVG_THREE_DAY_CANDIDATE_SPECS[candidate]
        source = str(spec["budget_source"])
        if source == "existing":
            cache = existing_cache
            scaler = existing_scaler
        else:
            cache = middle["root"]
            scaler = middle["scaler"]
        run = fold_root / "models" / "{}_seed{}".format(candidate, seed)
        train = _command(
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
            spec["variant"],
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
        )
        _default_svg_profile(train)
        if bool(spec["site_class_balanced"]):
            train.append("--site-class-balanced-sampler")
        if source == "multi":
            train.extend(["--multi-budget-train-manifests"])
            train.extend(
                str(item["root"] / "train" / "manifest.json")
                for item in ordered_budgets
            )
            train.extend(["--multi-budget-validation-manifests"])
            train.extend(
                str(item["root"] / "validation" / "manifest.json")
                for item in ordered_budgets
            )
            train.extend(["--multi-budget-scalers"])
            train.extend(str(item["scaler"]) for item in ordered_budgets)
        commands.append(
            ("train_{}".format(candidate), train, run / "best_evaluation.json")
        )
        if mode != "confirmatory":
            continue
        evaluate = _command(
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
        )
        if source == "multi":
            evaluate.extend(["--multi-budget-manifests"])
            evaluate.extend(
                str(item["root"] / "test" / "manifest.json")
                for item in ordered_budgets
            )
            evaluate.extend(["--multi-budget-scalers"])
            evaluate.extend(str(item["scaler"]) for item in ordered_budgets)
        commands.append(
            (
                "evaluate_{}".format(candidate),
                evaluate,
                run / "outer_test_evaluation.json",
            )
        )
    return commands
