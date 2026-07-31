"""Evaluate a frozen SV checkpoint with its validation-fit threshold."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.data_split import file_sha256  # noqa: E402
from keysubgraph.data.sv_signed_gin_dataset import (  # noqa: E402
    SVSignedGINDataset,
    create_sv_signed_gin_loader,
)
from keysubgraph.data.sv_theory_geometry import (  # noqa: E402
    SVTheoryAugmentedDataset,
)
from keysubgraph.models.sv_signed_gin import (  # noqa: E402
    SVSignedGINClassifier,
    SVSignedGINConfig,
)
from keysubgraph.training.sv_signed_gin_trainer import (  # noqa: E402
    load_sv_signed_gin_checkpoint,
    run_sv_signed_gin_epoch,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument("--theory-cache", type=Path)
    parser.add_argument("--theory-scaler", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--threshold-strategy",
        choices=("balanced_accuracy", "accuracy"),
        default="balanced_accuracy",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def _trusted_load(path, device):
    try:
        return torch.load(
            str(Path(path).resolve()),
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(str(Path(path).resolve()), map_location=device)


def _atomic_json(path, payload):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def main():
    args = parse_args()
    device = torch.device(args.device)
    raw = _trusted_load(args.checkpoint, device)
    model = SVSignedGINClassifier(
        SVSignedGINConfig(**raw["model_config"])
    ).to(device)
    checkpoint = load_sv_signed_gin_checkpoint(
        args.checkpoint, model, device
    )
    thresholds = checkpoint.get("validation_thresholds")
    if not isinstance(thresholds, dict) or args.threshold_strategy not in thresholds:
        raise ValueError("SV checkpoint has no frozen validation threshold")
    if model.config.uses_theory_geometry:
        if args.theory_cache is None or args.theory_scaler is None:
            raise ValueError(
                "theory-geometry evaluation requires its sidecar/scaler"
            )
        dataset = SVTheoryAugmentedDataset(
            args.manifest,
            args.scaler,
            args.theory_cache,
            args.theory_scaler,
            include_windows=model.config.uses_gin,
        )
    else:
        if args.theory_cache is not None or args.theory_scaler is not None:
            raise ValueError(
                "theory sidecars were supplied to a non-theory model"
            )
        dataset = SVSignedGINDataset(
            args.manifest,
            args.scaler,
            include_windows=model.config.uses_gin,
        )
    expected = checkpoint["provenance"]
    checks = (
        expected["protocol_sha256"]
        == dataset.manifest["protocol_sha256"],
        expected["selector_checkpoint_sha256"]
        == dataset.manifest["selector_checkpoint_sha256"],
        expected["selection_mode"] == dataset.manifest["selection_mode"],
        int(expected["selection_seed"])
        == int(dataset.manifest["selection_seed"]),
        expected["scaler_sha256"] == file_sha256(args.scaler),
    )
    if not all(checks):
        raise ValueError("SV evaluation provenance mismatch")
    if model.config.uses_theory_geometry:
        cache_hash_matches = True
        if dataset.split == "train":
            cache_hash_matches = (
                expected["theory_train_cache_sha256"]
                == file_sha256(args.theory_cache)
            )
        elif dataset.split == "validation":
            cache_hash_matches = (
                expected["theory_validation_cache_sha256"]
                == file_sha256(args.theory_cache)
            )
        theory_checks = (
            expected["theory_scaler_sha256"]
            == file_sha256(args.theory_scaler),
            cache_hash_matches,
        )
        if not all(theory_checks):
            raise ValueError(
                "SV theory evaluation provenance mismatch"
            )
    loader = create_sv_signed_gin_loader(
        dataset,
        batch_size=args.batch_size,
        seed=int(checkpoint["training_config"]["seed"]),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    threshold = float(thresholds[args.threshold_strategy])
    metrics = run_sv_signed_gin_epoch(
        model,
        loader,
        device,
        checkpoint["class_weights"],
        threshold=threshold,
        include_predictions=True,
    )
    result = {
        "artifact_type": "sv_hard_sgw_signed_gin_evaluation",
        "split": dataset.split,
        "variant": model.config.variant,
        "threshold_strategy": args.threshold_strategy,
        "threshold": threshold,
        "threshold_fit_split": "validation",
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "manifest_sha256": file_sha256(args.manifest),
        "scaler_sha256": file_sha256(args.scaler),
        "provenance": {
            "protocol_sha256": expected["protocol_sha256"],
            "selector_checkpoint_sha256": expected[
                "selector_checkpoint_sha256"
            ],
            "selection_mode": expected["selection_mode"],
            "selection_seed": int(expected["selection_seed"]),
            "training_seed": int(
                checkpoint["training_config"]["seed"]
            ),
        },
        "metrics": {
            key: value
            for key, value in metrics.items()
            if key != "predictions"
        },
        "predictions": metrics["predictions"],
    }
    if model.config.uses_theory_geometry:
        result["theory_feature_cache_sha256"] = file_sha256(
            args.theory_cache
        )
        result["theory_scaler_sha256"] = file_sha256(
            args.theory_scaler
        )
    _atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "split": dataset.split,
                "variant": model.config.variant,
                "roc_auc": metrics["roc_auc"],
                "site_stratified_roc_auc": metrics[
                    "site_stratified_roc_auc"
                ],
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "f1": metrics["f1"],
                "threshold": threshold,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
