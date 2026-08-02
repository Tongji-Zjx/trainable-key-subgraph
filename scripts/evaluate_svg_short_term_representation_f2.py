"""Evaluate a frozen representation-level F2 checkpoint."""

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
from keysubgraph.data.svg_short_term_representation_f2 import (  # noqa: E402
    SVGShortTermRepresentationF2Dataset,
)
from keysubgraph.training.svg_short_term_representation_f2_trainer import (  # noqa: E402
    create_svg_short_term_representation_f2_loader,
    load_svg_short_term_representation_f2_checkpoint,
    run_svg_short_term_representation_f2_epoch,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold-strategy", choices=("balanced_accuracy", "accuracy"), default="balanced_accuracy")
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
    dataset = SVGShortTermRepresentationF2Dataset(args.manifest)
    model, checkpoint = load_svg_short_term_representation_f2_checkpoint(
        args.checkpoint, device
    )
    if (
        dataset.g2_representation_dim
        != model.config.g2_representation_dim
        or dataset.short_term_representation_dim
        != model.config.short_term_representation_dim
    ):
        raise ValueError("representation F2 evaluation dimension mismatch")
    provenance = checkpoint["provenance"]
    for name in (
        "protocol_sha256",
        "short_term_checkpoint_sha256",
        "g2_checkpoint_sha256",
        "g2_scaler_sha256",
        "g2_spectral_scaler_sha256",
        "g2_variant",
    ):
        if provenance.get(name) != dataset.manifest.get(name):
            raise ValueError("representation F2 evaluation provenance mismatch")
    thresholds = checkpoint.get("validation_thresholds", {})
    if args.threshold_strategy not in thresholds:
        raise ValueError("representation F2 checkpoint has no frozen threshold")
    threshold = float(thresholds[args.threshold_strategy])
    loader = create_svg_short_term_representation_f2_loader(
        dataset,
        args.batch_size,
        int(checkpoint["training_config"]["seed"]),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    metrics = run_svg_short_term_representation_f2_epoch(
        model,
        loader,
        device,
        checkpoint["class_weights"],
        float(checkpoint["training_config"]["residual_auxiliary_weight"]),
        float(checkpoint["training_config"]["gate_penalty_weight"]),
        threshold=threshold,
        include_predictions=True,
    )
    result = {
        "artifact_type": "svg_short_term_representation_f2_evaluation",
        "split": dataset.split,
        "threshold_strategy": args.threshold_strategy,
        "threshold": threshold,
        "threshold_fit_split": "validation",
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "manifest_sha256": file_sha256(args.manifest),
        "frozen_base_encoders": True,
        "metrics": metrics,
        "provenance": provenance,
    }
    _atomic_json(args.output, result)
    print(
        "representation F2 {} AUC={} anchor_AUC={} BA={} gate={:.6f}".format(
            dataset.split,
            metrics["roc_auc"],
            metrics["anchor_roc_auc"],
            metrics["balanced_accuracy"],
            metrics["gate"],
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

