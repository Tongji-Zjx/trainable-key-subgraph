"""Evaluate a cached-SGW dual checkpoint using its frozen validation threshold."""

from __future__ import absolute_import, print_function

import argparse
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.data_protocol import validate_data_protocol  # noqa: E402
from keysubgraph.data.data_split import file_sha256  # noqa: E402
from keysubgraph.data.dual_sgw_manifest import read_dual_sgw_manifest  # noqa: E402
from keysubgraph.data.exact_stse_dataset import (  # noqa: E402
    ExactSTSEDataset,
    create_exact_stse_loader,
)
from keysubgraph.models.dual_stse_hard_sgw import (  # noqa: E402
    DualSTSEHardSGWClassifier,
)
from keysubgraph.models.dual_stse_hard_sgw_loss import (  # noqa: E402
    DualSTSEHardSGWCriterion,
    DualSTSEHardSGWLossConfig,
)
from keysubgraph.training.dual_stse_hard_sgw_trainer import (  # noqa: E402
    load_dual_checkpoint,
    run_dual_epoch,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    protocol = validate_data_protocol(args.protocol, PROJECT_ROOT)
    protocol_sha = file_sha256(args.protocol)
    manifest, _, lookup = read_dual_sgw_manifest(args.manifest)
    if manifest["split"] != args.split:
        raise ValueError("evaluation manifest split mismatch")
    if manifest["protocol_sha256"] != protocol_sha:
        raise ValueError("evaluation manifest protocol mismatch")
    paths = protocol["paths"]
    dataset = ExactSTSEDataset(
        PROJECT_ROOT / paths["dataset_root"],
        PROJECT_ROOT / paths["sample_index_csv"],
        PROJECT_ROOT / paths["splits_csv"],
        args.split,
        protocol["edge_presence_threshold"],
        require_coordinates=False,
    )
    if {item.sample_key for item in dataset.assignments} != set(lookup):
        raise ValueError("evaluation SGW cache does not cover the split")
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    model = DualSTSEHardSGWClassifier().to(device)
    payload = load_dual_checkpoint(
        args.checkpoint,
        model,
        device,
        expected_protocol_sha256=protocol_sha,
    )
    threshold = payload.get("validation_threshold")
    if threshold is None:
        raise ValueError("dual checkpoint has no frozen validation threshold")
    loader = create_exact_stse_loader(
        dataset,
        args.batch_size,
        seed=int(payload["training_config"]["seed"]),
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    criterion = DualSTSEHardSGWCriterion(
        DualSTSEHardSGWLossConfig(**payload["loss_config"])
    )
    metrics = run_dual_epoch(
        model,
        loader,
        device,
        criterion,
        payload["stage"],
        payload["class_weights"],
        feature_lookup=lookup,
        threshold=float(threshold),
        include_predictions=True,
    )
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "stage": payload["stage"],
        "split": args.split,
        "selection_metric": payload["selection_metric"],
        "threshold_source": "frozen_validation_threshold",
        "threshold": float(threshold),
        "protocol_sha256": protocol_sha,
        "manifest_selection_mode": manifest["selection_mode"],
        "provenance": payload["provenance"],
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(
            result, handle, ensure_ascii=False, indent=2, sort_keys=True
        )
        handle.write("\n")
    printable = dict(result)
    printable["metrics"] = dict(metrics)
    printable["metrics"].pop("predictions", None)
    print(json.dumps(printable, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

