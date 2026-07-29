"""Run a dependency-light forward/loss/backward check for all SV variants."""

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

from keysubgraph.models.sv_signed_gin import (  # noqa: E402
    SV_SIGNED_GIN_VARIANTS,
    SVSignedGINBatch,
    SVSignedGINClassifier,
    SVSignedGINConfig,
    SVSignedGINSampleInput,
    SVSignedGINWindowInput,
)
from keysubgraph.training.sv_signed_gin_trainer import (  # noqa: E402
    balanced_classification_loss,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _sample(key, label, offset, node_count, window_count):
    windows = []
    for time_index in range(window_count):
        node = (
            torch.arange(node_count * 15, dtype=torch.float32)
            .reshape(node_count, 15)
            / 30.0
            + float(offset)
            + 0.02 * time_index
        )
        adjacency = torch.zeros((node_count, node_count))
        for index in range(node_count - 1):
            value = 0.2 + 0.03 * index
            if index % 2:
                value = -value
            adjacency[index, index + 1] = value
            adjacency[index + 1, index] = value
        windows.append(SVSignedGINWindowInput(node, adjacency))
    return SVSignedGINSampleInput(
        sample_key=key,
        label=label,
        windows=tuple(windows),
        static_features=torch.full((28,), float(offset)),
        variation=torch.full((16,), abs(float(offset))),
    )


def main():
    args = parse_args()
    device = torch.device(args.device)
    batch = SVSignedGINBatch(
        (
            _sample("dummy-0", 0, -0.5, 3, 2),
            _sample("dummy-1", 1, 0.5, 5, 3),
        )
    ).to(device)
    results = {}
    for variant in SV_SIGNED_GIN_VARIANTS:
        torch.manual_seed(42)
        improved = (
            variant == "signed_gin_multibranch_late_fusion"
        )
        model = SVSignedGINClassifier(
            SVSignedGINConfig(
                variant=variant,
                dropout=0.0,
                message_mode=(
                    "signed_normalized"
                    if improved
                    else "signed_weighted"
                ),
                pooling="mean_std" if improved else "attention",
                gin_residual=improved,
                gin_jumping_knowledge=improved,
            )
        ).to(device)
        output = model(batch)
        loss = balanced_classification_loss(
            output.logits,
            batch.labels.to(device),
            torch.ones(2, device=device),
        )
        loss.backward()
        gradient = sum(
            float(parameter.grad.abs().sum().detach().cpu())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        if not bool(torch.isfinite(output.logits).all()) or gradient <= 0.0:
            raise RuntimeError("SV local flow failed for {}".format(variant))
        results[variant] = {
            "logits_shape": list(output.logits.shape),
            "representation_shape": list(
                output.final_representation.shape
            ),
            "loss": float(loss.detach().cpu()),
            "gradient_absolute_sum": gradient,
        }
    print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
