"""Evaluate a frozen low-capacity exact-SGW feature classifier."""

from __future__ import absolute_import, print_function

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

from keysubgraph.data.data_protocol import validate_data_protocol  # noqa: E402
from keysubgraph.data.data_split import file_sha256  # noqa: E402
from keysubgraph.data.dual_sgw_feature_dataset import (  # noqa: E402
    DualSGWFeatureDataset,
    create_dual_sgw_feature_loader,
)
from keysubgraph.models.dual_sgw_feature_classifier import (  # noqa: E402
    DualSGWFeatureClassifier,
    DualSGWFeatureClassifierConfig,
)
from keysubgraph.training.dual_sgw_feature_trainer import (  # noqa: E402
    binary_metrics,
    load_dual_sgw_feature_checkpoint,
    run_dual_sgw_feature_epoch,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def _device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _atomic_json(path, payload):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def _trusted_metadata(path, device):
    try:
        return torch.load(
            str(Path(path).resolve()),
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(str(Path(path).resolve()), map_location=device)


def main():
    args = parse_args()
    validate_data_protocol(args.protocol, PROJECT_ROOT)
    protocol_sha = file_sha256(args.protocol)
    dataset = DualSGWFeatureDataset(args.manifest, args.scaler)
    if dataset.split != args.split:
        raise ValueError("feature manifest split does not match --split")
    if dataset.manifest["protocol_sha256"] != protocol_sha:
        raise ValueError("feature manifest protocol hash mismatch")
    device = _device(args.device)
    metadata = _trusted_metadata(args.checkpoint, torch.device("cpu"))
    model = DualSGWFeatureClassifier(
        DualSGWFeatureClassifierConfig(**metadata["model_config"])
    )
    checkpoint = load_dual_sgw_feature_checkpoint(
        args.checkpoint, model, device
    )
    provenance = checkpoint["provenance"]
    for key in (
        "protocol_sha256",
        "selector_checkpoint_sha256",
        "selection_mode",
        "selection_seed",
    ):
        expected = (
            provenance[key]
            if key in provenance
            else dataset.manifest[key]
        )
        if dataset.manifest[key] != expected:
            raise ValueError("evaluation feature provenance mismatch")
    if provenance["sgw_scaler_sha256"] != file_sha256(args.scaler):
        raise ValueError("evaluation scaler hash mismatch")
    loader = create_dual_sgw_feature_loader(
        dataset,
        args.batch_size,
        seed=int(checkpoint["training_config"]["seed"]),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    class_weights = checkpoint["class_weights"].to(torch.float32)
    result = run_dual_sgw_feature_epoch(
        model.to(device),
        loader,
        device,
        class_weights,
        include_predictions=True,
    )
    predictions = result.pop("predictions")
    labels = [int(item["label"]) for item in predictions]
    probabilities = [
        float(item["positive_probability"]) for item in predictions
    ]
    thresholds = checkpoint.get("validation_thresholds")
    if not thresholds:
        raise ValueError("checkpoint has no frozen validation thresholds")
    payload = {
        "artifact": "dual_sgw_feature_classifier_evaluation",
        "schema_version": 1,
        "classifier_type": model.config.classifier_type,
        "seed": int(checkpoint["training_config"]["seed"]),
        "split": args.split,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "protocol_sha256": protocol_sha,
        "manifest_sha256": file_sha256(args.manifest),
        "scaler_sha256": file_sha256(args.scaler),
        "manifest_provenance": {
            key: dataset.manifest[key]
            for key in (
                "protocol_sha256",
                "selector_checkpoint_sha256",
                "selection_mode",
                "selection_seed",
            )
        },
        "validation_thresholds": thresholds,
        "loss": result["loss"],
        "metrics": {
            name: binary_metrics(labels, probabilities, threshold)
            for name, threshold in thresholds.items()
        },
        "predictions": predictions,
    }
    _atomic_json(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "classifier_type": payload["classifier_type"],
                "seed": payload["seed"],
                "split": payload["split"],
                "metrics": payload["metrics"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
