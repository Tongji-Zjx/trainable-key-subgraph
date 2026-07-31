"""Run frozen Stage-1 representation and mechanism diagnostics."""

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

from keysubgraph.analysis.theory_neural_diagnostics import (  # noqa: E402
    build_theory_neural_diagnostics,
    collect_theory_neural_diagnostic_inputs,
)
from keysubgraph.data.theory_neural_dataset import (  # noqa: E402
    TheoryNeuralDataset,
    create_theory_neural_loader,
)
from keysubgraph.models.theory_guided_neural import (  # noqa: E402
    TheoryGuidedNeuralClassifier,
    TheoryNeuralConfig,
)
from keysubgraph.training.theory_guided_neural_trainer import (  # noqa: E402
    _atomic_json,
    _trusted_load,
    load_theory_neural_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = _trusted_load(args.checkpoint, torch.device("cpu"))
    model = TheoryGuidedNeuralClassifier(
        TheoryNeuralConfig(**checkpoint["model_config"])
    ).to(device)
    load_theory_neural_checkpoint(args.checkpoint, model, device)
    datasets = [
        TheoryNeuralDataset(PROJECT_ROOT, path, args.scaler)
        for path in (args.train_manifest, args.validation_manifest)
    ]
    loaders = [
        create_theory_neural_loader(
            dataset, args.batch_size, args.seed, False, args.num_workers,
            pin_memory=args.device.startswith("cuda")
        ) for dataset in datasets
    ]
    result = build_theory_neural_diagnostics(*[
        collect_theory_neural_diagnostic_inputs(model, loader, device)
        for loader in loaders
    ])
    result["variant"] = model.config.variant
    _atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
