"""Evaluate a frozen checkpoint on an explicitly selected frozen split."""

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
from keysubgraph.data.graph_dataset import GraphSequenceDataset, create_data_loader  # noqa: E402
from keysubgraph.models.soft_extractor import SoftExtractorConfig, SoftGraphClassifier  # noqa: E402
from keysubgraph.training.trainer import (  # noqa: E402
    TrainingConfig,
    evaluate_model,
    load_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROJECT_ROOT / "configs" / "data_protocol.json")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test", "all"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = validate_data_protocol(args.protocol, PROJECT_ROOT)
    if args.split == "all" and protocol.get("experiment_mode") != "all_samples_exploratory":
        raise ValueError("--split all requires an all-sample protocol")
    checkpoint_payload = torch.load(
        str(args.checkpoint.resolve()), map_location="cpu", weights_only=False
    )
    if checkpoint_payload["data_protocol_sha256"] != file_sha256(args.protocol):
        raise ValueError("checkpoint does not match the frozen data protocol")
    if checkpoint_payload["edge_presence_threshold"] != protocol["edge_presence_threshold"]:
        raise ValueError("checkpoint edge threshold differs from the protocol")
    model = SoftGraphClassifier(SoftExtractorConfig(**checkpoint_payload["model_config"]))
    load_checkpoint(args.checkpoint, model, device=torch.device("cpu"))
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    paths = protocol["paths"]
    dataset = GraphSequenceDataset(
        PROJECT_ROOT / paths["dataset_root"],
        PROJECT_ROOT / paths["sample_index_csv"],
        PROJECT_ROOT / paths["splits_csv"],
        args.split,
        protocol["edge_presence_threshold"],
    )
    loader = create_data_loader(
        dataset,
        args.batch_size,
        seed=int(checkpoint_payload["training_config"]["seed"]),
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    training_config = TrainingConfig(**checkpoint_payload["training_config"])
    metrics = evaluate_model(
        model,
        loader,
        device,
        training_config,
        torch.tensor(checkpoint_payload["class_weights"], dtype=torch.float32, device=device),
        max_batches=args.max_batches,
        include_predictions=True,
    )
    payload = {
        "schema_version": 1,
        "split": args.split,
        "debug_limited_batches": args.max_batches,
        "exploratory_in_sample_evaluation": args.split == "all",
        "generalization_metrics_available": args.split == "test",
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "data_protocol_sha256": file_sha256(args.protocol),
        "metrics": metrics,
    }
    output = args.output or (
        args.checkpoint.resolve().parent / "{}_evaluation.json".format(args.split)
    )
    output = output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError("evaluation output already exists")
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
