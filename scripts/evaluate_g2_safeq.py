"""Evaluate G2-SafeQ with validation-frozen mixing and threshold."""

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
from keysubgraph.data.g2_safeq import G2SafeQDataset  # noqa: E402
from keysubgraph.training.g2_safeq_trainer import (  # noqa: E402
    create_g2_safeq_loader,
    load_g2_safeq_checkpoint,
    run_g2_safeq_epoch,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--threshold-strategy",
        choices=("balanced_accuracy", "accuracy"),
        default="balanced_accuracy",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


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
    dataset = G2SafeQDataset(args.manifest)
    model, checkpoint = load_g2_safeq_checkpoint(args.checkpoint, device)
    provenance = checkpoint.get("provenance", {})
    for name in (
        "protocol_sha256",
        "selector_checkpoint_sha256",
        "selection_mode",
        "selection_seed",
        "g2_checkpoint_sha256",
        "g2_scaler_sha256",
        "g2_spectral_scaler_sha256",
        "g2_variant",
        "transition_hidden_dim",
        "summary_dim",
        "frozen_g2",
        "train_only_scalers",
    ):
        if provenance.get(name) != dataset.manifest.get(name):
            raise ValueError("SafeQ evaluation provenance mismatch: {}".format(name))
    if dataset.summary_dim != model.config.summary_dim:
        raise ValueError("SafeQ evaluation feature dimension mismatch")
    mixing = checkpoint.get("mixing_selection")
    if not isinstance(mixing, dict) or mixing.get("fit_split") != "validation":
        raise ValueError("SafeQ checkpoint has no validation-fit mixing")
    selected = mixing["selected"]
    thresholds = checkpoint.get("validation_thresholds", {})
    if args.threshold_strategy not in thresholds:
        raise ValueError("SafeQ checkpoint has no frozen threshold")
    threshold = float(thresholds[args.threshold_strategy])
    loader = create_g2_safeq_loader(
        dataset,
        args.batch_size,
        int(checkpoint["training_config"]["seed"]),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    metrics = run_g2_safeq_epoch(
        model,
        loader,
        device,
        checkpoint["class_weights"],
        alpha=float(selected["alpha"]),
        beta=float(selected["beta"]),
        threshold=threshold,
        include_predictions=True,
    )
    result = {
        "artifact_type": "g2_safeq_evaluation",
        "split": dataset.split,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "manifest_sha256": file_sha256(args.manifest),
        "mixing_fit_split": "validation",
        "threshold_fit_split": "validation",
        "outer_test_tuning": False,
        "threshold_strategy": args.threshold_strategy,
        "threshold": threshold,
        "selected_mixing": selected,
        "fallback_to_frozen_g2": mixing["fallback_to_frozen_g2"],
        "metrics": {
            key: value for key, value in metrics.items() if key != "predictions"
        },
        "predictions": metrics["predictions"],
        "provenance": provenance,
    }
    _atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "split": dataset.split,
                "roc_auc": metrics["roc_auc"],
                "site_stratified_roc_auc": metrics[
                    "site_stratified_roc_auc"
                ],
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "alpha": float(selected["alpha"]),
                "beta": float(selected["beta"]),
                "fallback": mixing["fallback_to_frozen_g2"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
