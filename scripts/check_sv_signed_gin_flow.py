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


def _sample(
    key,
    label,
    offset,
    node_count,
    window_count,
    include_budget_views=True,
):
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
        windows.append(
            SVSignedGINWindowInput(
                node,
                adjacency,
                time_position=time_index,
                hks=torch.linspace(
                    0.1, 0.9, node_count * 6
                ).reshape(node_count, 6),
                diffusion_eigenvalues=torch.linspace(
                    0.1, 1.0, node_count
                ),
                diffusion_eigenvectors=torch.eye(node_count),
                spectral_delta_to_next=(
                    torch.linspace(-0.5, 0.5, 16)
                    if time_index + 1 < window_count
                    else None
                ),
                communities=torch.arange(node_count, dtype=torch.long) % 2,
            )
        )
    budget_views = ()
    if include_budget_views:
        budget_views = tuple(
            _sample(
                key + "-budget-{}".format(index),
                label,
                offset + 0.01 * index,
                node_count,
                window_count,
                include_budget_views=False,
            )
            for index in range(3)
        )
    return SVSignedGINSampleInput(
        sample_key=key,
        label=label,
        windows=tuple(windows),
        static_features=torch.full((28,), float(offset)),
        variation=torch.full((16,), abs(float(offset))),
        spectral_direction=torch.linspace(
            -0.25, 0.25, 16
        ) + float(offset),
        diffusion_geometry=torch.linspace(
            0.0, 1.0, 28
        ) + abs(float(offset)),
        budget_views=budget_views,
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
        residual_attention = (
            variant
            == "signed_gin_static_anchor_residual_attention"
        )
        overrides = {}
        if residual_attention:
            overrides.update(
                {
                    "message_mode": "signed_normalized",
                    "pooling": "mean_std",
                    "gin_residual": True,
                    "gin_jumping_knowledge": True,
                    "gin_compact_readout": True,
                    "gin_batch_normalization": True,
                }
            )
        model = SVSignedGINClassifier(
            SVSignedGINConfig(
                variant=variant,
                dropout=0.0,
                gin_residual_attention=residual_attention,
                **overrides
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
