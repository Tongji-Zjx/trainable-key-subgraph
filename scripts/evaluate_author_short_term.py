"""Evaluate a frozen author short-term checkpoint with validation threshold."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.data_protocol import (  # noqa: E402
    protocol_node_name_policy,
    validate_data_protocol,
)
from keysubgraph.data.data_split import file_sha256  # noqa: E402
from keysubgraph.data.graph_dataset import GraphSequenceDataset  # noqa: E402
from keysubgraph.training.author_short_term_trainer import (  # noqa: E402
    create_author_short_term_evaluation_loader,
    evaluate_author_short_term,
    model_from_author_short_term_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), required=True
    )
    parser.add_argument(
        "--threshold-strategy",
        choices=("balanced_accuracy", "accuracy"),
        default="balanced_accuracy",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def _device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def main():
    args = parse_args()
    protocol = validate_data_protocol(args.protocol, PROJECT_ROOT)
    device = _device(args.device)
    model, checkpoint = model_from_author_short_term_checkpoint(
        args.checkpoint, device
    )
    if checkpoint.get("protocol_sha256") != file_sha256(args.protocol):
        raise ValueError("author short-term checkpoint protocol mismatch")
    thresholds = checkpoint.get("validation_thresholds", {})
    if args.threshold_strategy not in thresholds:
        raise ValueError("checkpoint has no frozen validation threshold")
    threshold = float(thresholds[args.threshold_strategy])
    paths = protocol["paths"]
    dataset = GraphSequenceDataset(
        PROJECT_ROOT / paths["dataset_root"],
        PROJECT_ROOT / paths["sample_index_csv"],
        PROJECT_ROOT / paths["splits_csv"],
        args.split,
        protocol["edge_presence_threshold"],
        node_name_policy=protocol_node_name_policy(protocol),
    )
    loader = create_author_short_term_evaluation_loader(
        dataset,
        args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    training = checkpoint["training_config"]
    metrics = evaluate_author_short_term(
        model,
        loader,
        device,
        checkpoint["positive_class_weight"],
        training["label_smoothing"],
        threshold,
    )
    predictions = metrics.pop("predictions")
    result = {
        "model_name": model.model_name,
        "profile": training["profile"],
        "checkpoint": str(args.checkpoint.resolve()),
        "protocol": str(args.protocol.resolve()),
        "split": args.split,
        "threshold_source": "frozen_validation",
        "threshold_fit_split": "validation",
        "threshold_strategy": args.threshold_strategy,
        "threshold": threshold,
        "uses_coordinates": False,
        "metrics": metrics,
        "predictions": predictions,
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "{}_evaluation.json".format(args.split)
    with json_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    csv_path = output_dir / "{}_predictions.csv".format(args.split)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "sample_key",
                "site",
                "label",
                "positive_probability",
                "prediction",
            ),
        )
        writer.writeheader()
        writer.writerows(predictions)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

