"""Evaluate a frozen structured short-term checkpoint without test tuning."""

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
from keysubgraph.data.graph_dataset import (  # noqa: E402
    GraphSequenceDataset,
    create_data_loader,
)
from keysubgraph.training import (  # noqa: E402
    evaluate_structured_short_term,
    model_from_structured_short_term_checkpoint,
)
from keysubgraph.models.structured_short_term import (  # noqa: E402
    is_paper_aligned_variant,
    variant_uses_pst,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        required=True,
    )
    parser.add_argument(
        "--threshold-strategy",
        choices=("balanced_accuracy", "accuracy"),
        default="balanced_accuracy",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
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
    model, checkpoint = model_from_structured_short_term_checkpoint(
        args.checkpoint,
        device,
    )
    uses_community_embedding = bool(
        is_paper_aligned_variant(model.config.variant)
    )
    uses_pst = variant_uses_pst(model.config.variant)
    protocol_hash = file_sha256(args.protocol)
    if checkpoint.get("protocol_sha256") != protocol_hash:
        raise ValueError("checkpoint protocol hash mismatch")
    if model.community_frequency is not None:
        if model.community_frequency.protocol_sha256 != protocol_hash:
            raise ValueError("checkpoint community frequency protocol mismatch")
        splits_hash = file_sha256(
            PROJECT_ROOT / protocol["paths"]["splits_csv"]
        )
        if model.community_frequency.train_manifest_sha256 != splits_hash:
            raise ValueError("checkpoint community frequency split mismatch")
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
    loader = create_data_loader(
        dataset,
        args.batch_size,
        seed=int(checkpoint["training_config"]["seed"]),
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    metrics = evaluate_structured_short_term(
        model,
        loader,
        device,
        checkpoint["class_weights"],
        threshold,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = metrics.pop("predictions")
    result = {
        "model_name": model.model_name,
        "checkpoint": str(args.checkpoint.resolve()),
        "protocol": str(args.protocol.resolve()),
        "split": args.split,
        "threshold_source": "frozen_validation",
        "threshold_fit_split": "validation",
        "threshold_strategy": args.threshold_strategy,
        "threshold": threshold,
        "metrics": metrics,
        "predictions": predictions,
    }
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
    summary_path = output_dir / "{}_summary.md".format(args.split)
    summary_path.write_text(
        "\n".join(
            (
                "# Structured Short-Term {} Evaluation".format(
                    args.split.title()
                ),
                "",
                "- Coordinate features: disabled",
                "- Raw community embedding: {}".format(
                    "enabled" if uses_community_embedding else "disabled"
                ),
                "- Sequence statistics branch: {}".format(
                    "enabled"
                    if (not uses_community_embedding or uses_pst)
                    else "disabled"
                ),
                "- Paper p_ST branch: {}".format(
                    "enabled" if uses_pst else "disabled"
                ),
                "- Model variant: `{}`".format(model.config.variant),
                "- Threshold source: validation ({})".format(
                    args.threshold_strategy
                ),
                "",
                "| AUROC | Balanced Accuracy | Accuracy | F1 | Threshold |",
                "|---:|---:|---:|---:|---:|",
                "| {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} |".format(
                    metrics["roc_auc"],
                    metrics["balanced_accuracy"],
                    metrics["accuracy"],
                    metrics["f1"],
                    threshold,
                ),
                "",
            )
        ),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    print("predictions:", csv_path)
    print("summary:", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
