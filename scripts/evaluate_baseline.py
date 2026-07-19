"""Evaluate a frozen signed sequence baseline on one manifest partition."""

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

from keysubgraph.data.baseline_collate import create_baseline_loader  # noqa: E402
from keysubgraph.data.baseline_dataset import BaselineHardSubgraphDataset  # noqa: E402
from keysubgraph.data.baseline_manifest import read_baseline_manifest  # noqa: E402
from keysubgraph.data.data_split import file_sha256  # noqa: E402
from keysubgraph.models.baseline_classifier import (  # noqa: E402
    BaselineModelConfig,
    SignedSequenceBaseline,
)
from keysubgraph.training.baseline_trainer import (  # noqa: E402
    evaluate_baseline,
    load_baseline_checkpoint,
    read_baseline_checkpoint_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = read_baseline_checkpoint_payload(
        args.checkpoint, device=torch.device("cpu")
    )
    structural_version = int(
        checkpoint.get("model_config", {}).get("structural_interface_version", 0)
    )
    if structural_version == 1:
        transform_path = args.checkpoint.resolve().parent / "structural_transform.json"
        if not transform_path.is_file():
            raise ValueError("structural checkpoint is missing structural_transform.json")
        if file_sha256(transform_path) != checkpoint.get("structural_transform_sha256"):
            raise ValueError("structural transform hash differs from checkpoint")
        with transform_path.open("r", encoding="utf-8") as handle:
            if json.load(handle) != checkpoint.get("structural_transform"):
                raise ValueError("structural transform payload differs from checkpoint")
    manifest_payload, manifest_records = read_baseline_manifest(args.manifest, PROJECT_ROOT)
    if manifest_payload["data_protocol_sha256"] != checkpoint["data_protocol_sha256"]:
        raise ValueError("checkpoint and evaluation manifest use different protocols")
    if manifest_payload["checkpoint_sha256"] != checkpoint["extractor_checkpoint_sha256"]:
        raise ValueError("checkpoint and evaluation manifest use different extractors")
    if manifest_payload.get("parent_manifest_sha256") != checkpoint.get(
        "parent_manifest_sha256"
    ):
        raise ValueError("checkpoint and evaluation manifest have different parents")
    if manifest_payload.get("downstream_splits_json_sha256") != checkpoint.get(
        "downstream_splits_json_sha256"
    ):
        raise ValueError("checkpoint and evaluation manifest use different downstream splits")
    if manifest_payload.get("subgraph_source", "key") != checkpoint.get(
        "subgraph_source", "key"
    ):
        raise ValueError("checkpoint and evaluation manifest use different subgraph sources")
    if manifest_payload.get("matched_control_manifest_sha256", "") != checkpoint.get(
        "matched_control_manifest_sha256", ""
    ):
        raise ValueError("checkpoint and evaluation manifest use different matched cohorts")
    if manifest_payload["split"] == "validation" and file_sha256(
        args.manifest
    ) != checkpoint["validation_manifest_sha256"]:
        raise ValueError("validation manifest differs from checkpoint selection data")

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    model = SignedSequenceBaseline(
        BaselineModelConfig(**checkpoint["model_config"])
    ).to(device)
    load_baseline_checkpoint(args.checkpoint, model, device=device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    dataset = BaselineHardSubgraphDataset(PROJECT_ROOT, args.manifest)
    loader = create_baseline_loader(
        dataset,
        args.batch_size,
        seed=int(checkpoint["training_config"]["seed"]),
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    metrics = evaluate_baseline(
        model,
        loader,
        device,
        torch.tensor(checkpoint["class_weights"], dtype=torch.float32, device=device),
        threshold=float(checkpoint["classification_threshold"]),
        max_batches=args.max_batches,
        include_predictions=True,
    )
    metrics.pop("labels")
    metrics.pop("probabilities")
    predictions = metrics.pop("predictions")
    record_map = {record.sample_key: record for record in manifest_records}
    for row in predictions:
        record = record_map[row["sample_key"]]
        row["session_id"] = record.session_id
    payload = {
        "schema_version": 1,
        "split": manifest_payload["split"],
        "evidence_level": manifest_payload["evidence_level"],
        "debug_limited_batches": args.max_batches,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "baseline_manifest_sha256": file_sha256(args.manifest),
        "classification_threshold_source": "checkpoint_validation",
        "subgraph_source": manifest_payload.get("subgraph_source", "key"),
        "matched_control_manifest_sha256": manifest_payload.get(
            "matched_control_manifest_sha256", ""
        ),
        "structural_group": checkpoint.get("model_config", {}).get(
            "structural_group", "neutral"
        ),
        "prior_mode": checkpoint.get("model_config", {}).get("prior_mode", "none"),
        "structural_transform_sha256": checkpoint.get(
            "structural_transform_sha256", ""
        ),
        "metrics": metrics,
        "predictions": predictions,
    }
    output = args.output or (
        args.checkpoint.resolve().parent
        / "{}_baseline_evaluation.json".format(manifest_payload["split"])
    )
    output = output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError("baseline evaluation output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(output))
    print(json.dumps({"output": str(output), "metrics": metrics}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
