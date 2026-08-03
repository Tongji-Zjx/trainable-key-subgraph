"""Evaluate a revised critical-channel checkpoint with its frozen threshold."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.multiview_critical import MultiViewCriticalDataset, create_multiview_loader  # noqa: E402
from keysubgraph.models.multiview_critical import MultiViewCriticalClassifier, MultiViewCriticalConfig  # noqa: E402
from keysubgraph.training.multiview_critical_trainer import load_multiview_checkpoint, run_multiview_epoch  # noqa: E402
from keysubgraph.training.trainer import class_weights_from_labels  # noqa: E402
from keysubgraph.data.data_split import file_sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()
    device = torch.device(args.device)
    try:
        raw = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(str(args.checkpoint), map_location="cpu")
    model = MultiViewCriticalClassifier(MultiViewCriticalConfig(**raw["model_config"])).to(device)
    checkpoint = load_multiview_checkpoint(args.checkpoint, model, device)
    dataset = MultiViewCriticalDataset(PROJECT_ROOT, args.manifest, args.scaler)
    if checkpoint.get("scaler_sha256") != file_sha256(args.scaler):
        raise ValueError("multi-view checkpoint/scaler mismatch")
    for name in ("protocol_sha256", "selector_checkpoint_sha256", "feature_schema_sha256"):
        if checkpoint.get(name) != dataset.manifest.get(name):
            raise ValueError("multi-view checkpoint/manifest {} mismatch".format(name))
    loader = create_multiview_loader(
        dataset, args.batch_size, 0, False, args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    weights = class_weights_from_labels(dataset.labels).to(device)
    metrics = run_multiview_epoch(
        model, loader, device, weights, threshold=float(checkpoint["threshold"]),
        include_predictions=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"threshold": checkpoint["threshold"], "metrics": metrics}, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
