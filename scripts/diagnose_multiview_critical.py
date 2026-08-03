"""Read-only representation, attention and signed-message diagnostics."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.multiview_critical import MultiViewCriticalDataset, create_multiview_loader  # noqa: E402
from keysubgraph.models.multiview_critical import MultiViewCriticalClassifier, MultiViewCriticalConfig  # noqa: E402
from keysubgraph.training.multiview_critical_trainer import load_multiview_checkpoint  # noqa: E402


def _matrix_summary(rows):
    if not rows:
        return {"sample_count": 0}
    matrix = torch.stack(rows).to(torch.float64)
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    energy = singular.square()
    probabilities = energy / energy.sum().clamp_min(1.0e-12)
    entropy = -(probabilities * probabilities.clamp_min(1.0e-12).log()).sum()
    effective_rank = float(torch.exp(entropy))
    normalized = torch.nn.functional.normalize(matrix, dim=1)
    cosine = normalized.matmul(normalized.transpose(0, 1))
    if matrix.shape[0] > 1:
        mean_cosine = float((cosine.sum() - matrix.shape[0]) / (matrix.shape[0] * (matrix.shape[0] - 1)))
    else:
        mean_cosine = None
    return {
        "sample_count": int(matrix.shape[0]),
        "dimension": int(matrix.shape[1]),
        "mean_feature_variance": float(matrix.var(dim=0, unbiased=False).mean()),
        "effective_rank": effective_rank,
        "normalized_effective_rank": effective_rank / float(max(1, matrix.shape[1])),
        "mean_pairwise_cosine": mean_cosine,
    }


def _attention_summary(rows):
    entropies, maxima, effective = [], [], []
    for weights in rows:
        weights = weights.detach().to(torch.float64).cpu().clamp_min(1.0e-12)
        entropy = float(-(weights * weights.log()).sum())
        entropies.append(entropy / math.log(max(2, weights.numel())))
        maxima.append(float(weights.max()))
        effective.append(float(math.exp(entropy)))
    if not rows:
        return {"count": 0}
    return {
        "count": len(rows),
        "mean_normalized_entropy": sum(entropies) / len(entropies),
        "mean_maximum_weight": sum(maxima) / len(maxima),
        "mean_effective_nodes": sum(effective) / len(effective),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    device = torch.device(args.device)
    try:
        payload = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(str(args.checkpoint), map_location="cpu")
    model = MultiViewCriticalClassifier(
        MultiViewCriticalConfig(**payload["model_config"])
    ).to(device)
    load_multiview_checkpoint(args.checkpoint, model, device)
    model.eval()
    dataset = MultiViewCriticalDataset(
        PROJECT_ROOT, args.manifest, args.scaler, max_samples=args.max_samples
    )
    loader = create_multiview_loader(dataset, 1, 0, False, args.num_workers, False)
    representations = defaultdict(list)
    attentions = defaultdict(list)
    messages = defaultdict(list)
    handles = []

    def encoder_hook(name):
        def hook(module, inputs, output):
            del module, inputs
            representations[name + "_window"].append(output.graph_embedding.detach().cpu())
            attentions[name].append(output.attention.detach().cpu())
            for index, state in enumerate(output.layer_states):
                representations["{}_layer_{}_graph_mean".format(name, index + 1)].append(
                    state.detach().mean(dim=0).cpu()
                )
        return hook

    for name in ("static_encoder", "object_encoder", "full_encoder"):
        encoder = getattr(model, name)
        handles.append(encoder.register_forward_hook(encoder_hook(name.replace("_encoder", ""))))
        for index, layer in enumerate(encoder.layers):
            def layer_hook(module, inputs, output, branch=name, layer_index=index):
                del inputs, output
                for key, value in getattr(module, "last_message_diagnostics", {}).items():
                    messages["{}_layer_{}_{}".format(branch, layer_index + 1, key)].append(float(value))
            handles.append(layer.register_forward_hook(layer_hook))
    q_absolute, delta_absolute = [], []
    with torch.no_grad():
        for batch in loader:
            output = model(batch.to(device))
            for sample in output.samples:
                representations["static_sample"].append(sample.static_representation.cpu())
                representations["evolution_sample"].append(sample.evolution_representation.cpu())
                representations["full_sample"].append(sample.full_representation.cpu())
                representations["critical_final"].append(sample.representation.cpu())
                if sample.q_predictions is not None:
                    q_absolute.extend(
                        float(value) for value in (sample.q_predictions - sample.q_targets).abs().flatten().cpu()
                    )
                if sample.delta_q_predictions is not None:
                    delta_absolute.extend(
                        float(value) for value in (sample.delta_q_predictions - sample.delta_q_targets).abs().flatten().cpu()
                    )
    for handle in handles:
        handle.remove()
    result = {
        "schema_version": 1,
        "artifact_type": "multiview_critical_read_only_diagnostic",
        "split": dataset.split,
        "sample_count": len(dataset),
        "representations": {
            name: _matrix_summary(rows) for name, rows in sorted(representations.items())
        },
        "attention": {
            name: _attention_summary(rows) for name, rows in sorted(attentions.items())
        },
        "signed_messages": {
            name: {
                "count": len(values),
                "mean": sum(values) / len(values),
                "minimum": min(values),
                "maximum": max(values),
            }
            for name, values in sorted(messages.items()) if values
        },
        "q_standardized_mae": sum(q_absolute) / len(q_absolute) if q_absolute else None,
        "delta_q_standardized_mae": sum(delta_absolute) / len(delta_absolute) if delta_absolute else None,
        "gates": {
            "static": float(torch.tanh(model.static_gate).cpu()),
            "temporal": float(torch.tanh(model.temporal.gate).cpu()),
            "v": float(torch.tanh(model.v_gate).cpu()),
            "g": float(torch.tanh(model.g_gate).cpu()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
