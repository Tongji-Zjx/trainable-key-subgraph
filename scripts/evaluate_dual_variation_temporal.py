"""Evaluate a frozen T1--T4 checkpoint with validation-fitted thresholds."""

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
from keysubgraph.data.dual_temporal_dataset import (  # noqa: E402
    DualTemporalDataset,
    create_dual_temporal_loader,
    shuffle_dual_temporal_batch,
)
from keysubgraph.models.dual_variation_temporal import (  # noqa: E402
    DualVariationTemporalClassifier,
    DualVariationTemporalConfig,
)
from keysubgraph.training.dual_sgw_feature_trainer import (  # noqa: E402
    binary_metrics,
)
from keysubgraph.training.dual_variation_temporal_trainer import (  # noqa: E402
    load_dual_temporal_checkpoint,
    run_dual_temporal_epoch,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--temporal-scaler", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--shuffle-time", action="store_true")
    parser.add_argument("--shuffle-seed", type=int, default=2026)
    return parser.parse_args()


def _device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _trusted_load(path):
    try:
        return torch.load(str(Path(path).resolve()), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(Path(path).resolve()), map_location="cpu")


class _ShuffledLoader(object):
    def __init__(self, loader, seed):
        self.loader = loader
        self.seed = int(seed)

    def __iter__(self):
        for batch in self.loader:
            yield shuffle_dual_temporal_batch(batch, self.seed)


def _atomic_json(path, payload):
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError("temporal evaluation output already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def main():
    args = parse_args()
    raw = _trusted_load(args.checkpoint)
    model = DualVariationTemporalClassifier(
        DualVariationTemporalConfig(**raw["model_config"])
    )
    device = _device(args.device)
    payload = load_dual_temporal_checkpoint(
        args.checkpoint, model, device
    )
    dataset = DualTemporalDataset(args.manifest, args.temporal_scaler)
    expected = {
        "temporal_scaler_sha256": file_sha256(args.temporal_scaler),
    }
    for key in (
        "protocol_sha256",
        "selector_checkpoint_sha256",
        "exact_head_checkpoint_sha256",
        "sgw_scaler_sha256",
        "exact_manifest_sha256",
        "selection_mode",
        "selection_seed",
    ):
        expected[key] = dataset.manifest[key]
    for key, value in expected.items():
        if payload["provenance"].get(key) != value:
            raise ValueError("evaluation provenance mismatch: {}".format(key))
    loader = create_dual_temporal_loader(
        dataset,
        args.batch_size,
        seed=args.shuffle_seed,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    if args.shuffle_time:
        loader = _ShuffledLoader(loader, args.shuffle_seed)
    class_weights = payload["class_weights"].to(torch.float32)
    result = run_dual_temporal_epoch(
        model.to(device),
        loader,
        device,
        class_weights,
        auxiliary_weight=float(
            payload["training_config"]["temporal_auxiliary_weight"]
        ),
        include_predictions=True,
    )
    labels = [int(item["label"]) for item in result["predictions"]]
    probabilities = [
        float(item["positive_probability"])
        for item in result["predictions"]
    ]
    thresholds = payload.get("validation_thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("checkpoint has no frozen validation thresholds")
    output = {
        "artifact": "dual_d3b_temporal_evaluation",
        "variant": model.config.variant,
        "split": dataset.split,
        "time_order": "shuffled" if args.shuffle_time else "original",
        "shuffle_seed": args.shuffle_seed if args.shuffle_time else None,
        "threshold_fit_split": payload["threshold_fit_split"],
        "thresholds": thresholds,
        "metrics": {
            name: binary_metrics(labels, probabilities, threshold)
            for name, threshold in thresholds.items()
        },
        "predictions": result["predictions"],
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "manifest_sha256": file_sha256(args.manifest),
        "temporal_scaler_sha256": file_sha256(args.temporal_scaler),
    }
    _atomic_json(args.output, output)
    print(json.dumps(output["metrics"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
