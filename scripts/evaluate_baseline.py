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
    manifest_payload, _ = read_baseline_manifest(args.manifest, PROJECT_ROOT)
    if manifest_payload["data_protocol_sha256"] != checkpoint["data_protocol_sha256"]:
        raise ValueError("checkpoint and evaluation manifest use different protocols")
    if manifest_payload["checkpoint_sha256"] != checkpoint["extractor_checkpoint_sha256"]:
        raise ValueError("checkpoint and evaluation manifest use different extractors")
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
    labels = metrics.pop("labels")
    probabilities = metrics.pop("probabilities")
    del labels, probabilities
    payload = {
        "schema_version": 1,
        "split": manifest_payload["split"],
        "evidence_level": manifest_payload["evidence_level"],
        "debug_limited_batches": args.max_batches,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "baseline_manifest_sha256": file_sha256(args.manifest),
        "classification_threshold_source": "checkpoint_validation",
        "metrics": metrics,
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
