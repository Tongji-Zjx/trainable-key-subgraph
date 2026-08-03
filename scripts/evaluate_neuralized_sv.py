"""Evaluate corrected neuralized S/V with a frozen validation threshold."""

from __future__ import absolute_import, division, print_function

import argparse
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.theory_neural_dataset import (  # noqa: E402
    TheoryNeuralDataset,
    create_theory_neural_loader,
)
from keysubgraph.models.neuralized_sv import (  # noqa: E402
    NeuralizedSVClassifier,
    NeuralizedSVConfig,
)
from keysubgraph.training.theory_guided_neural_trainer import (  # noqa: E402
    _trusted_load,
    evaluate_theory_neural_classifier,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--threshold-strategy",
        choices=("balanced_accuracy", "accuracy"),
        default="balanced_accuracy",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = _trusted_load(args.checkpoint, torch.device("cpu"))
    model = NeuralizedSVClassifier(
        NeuralizedSVConfig(**checkpoint["model_config"])
    )
    dataset = TheoryNeuralDataset(PROJECT_ROOT, args.manifest, args.scaler)
    loader = create_theory_neural_loader(
        dataset,
        args.batch_size,
        args.seed,
        False,
        args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    result = evaluate_theory_neural_classifier(
        model.to(torch.device(args.device)),
        loader,
        args.checkpoint,
        torch.device(args.device),
        args.output,
        args.threshold_strategy,
        expected_provenance=checkpoint["provenance"],
    )
    metrics = result["metrics"]
    print("split:", dataset.split)
    print("AUC:", metrics["roc_auc"])
    print("site AUC:", metrics["site_stratified_roc_auc"])
    print("BA:", metrics["balanced_accuracy"])
    print("accuracy:", metrics["accuracy"])
    print("threshold:", result["threshold"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

