"""Frozen validation-only S/V/G channel contribution diagnostic."""

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

from keysubgraph.data.multiview_critical import (  # noqa: E402
    MultiViewCriticalDataset,
    create_multiview_loader,
)
from keysubgraph.models.multiview_critical import (  # noqa: E402
    MultiViewCriticalClassifier,
    MultiViewCriticalConfig,
)
from keysubgraph.training.dual_sgw_feature_trainer import (  # noqa: E402
    binary_metrics,
    fit_binary_threshold,
)
from keysubgraph.training.multiview_critical_trainer import (  # noqa: E402
    load_multiview_checkpoint,
)


def _load_payload(path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def _condition_representations(model, input_samples, output):
    rows = {"all": [], "mask_v": [], "mask_g": [], "static_only": []}
    v_gate = torch.tanh(model.v_gate)
    g_gate = torch.tanh(model.g_gate)
    legacy_gate = torch.tanh(model.legacy_v_gate)
    for sample_input, sample_output in zip(input_samples, output.samples):
        static = sample_output.static_representation
        v_residual = torch.zeros_like(static)
        if model.config.enable_v:
            v_residual = v_gate * model.v_projection(
                sample_output.evolution_representation
            )
        elif model.config.enable_legacy_v:
            v_residual = legacy_gate * model.legacy_v_projection(
                sample_input.legacy_variation
            )
        g_residual = torch.zeros_like(static)
        if model.config.enable_g:
            g_residual = g_gate * model.g_projection(
                sample_output.full_representation
            )
        rows["all"].append(static + v_residual + g_residual)
        rows["mask_v"].append(static + g_residual)
        rows["mask_g"].append(static + v_residual)
        rows["static_only"].append(static)
    return {name: torch.stack(values, dim=0) for name, values in rows.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()

    payload = _load_payload(args.checkpoint)
    model = MultiViewCriticalClassifier(
        MultiViewCriticalConfig(**payload["model_config"])
    ).to(torch.device(args.device))
    load_multiview_checkpoint(args.checkpoint, model, torch.device(args.device))
    model.eval()
    dataset = MultiViewCriticalDataset(
        PROJECT_ROOT, args.manifest, args.scaler, max_samples=args.max_samples
    )
    if dataset.split != "validation":
        raise ValueError("channel masking is a validation-only architecture diagnostic")
    loader = create_multiview_loader(
        dataset, args.batch_size, 0, False, args.num_workers, False
    )
    labels, probabilities = [], {
        name: [] for name in ("all", "mask_v", "mask_g", "static_only")
    }
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(torch.device(args.device))
            output = model(batch)
            representations = _condition_representations(
                model, batch.samples, output
            )
            labels.extend(int(value) for value in batch.labels.tolist())
            for name, values in representations.items():
                current = torch.softmax(model.classifier(values), dim=-1)[:, 1]
                probabilities[name].extend(
                    float(value) for value in current.detach().cpu().tolist()
                )
    all_threshold = fit_binary_threshold(
        labels, probabilities["all"], "balanced_accuracy"
    )
    conditions = {}
    all_auc = None
    for name in ("all", "mask_v", "mask_g", "static_only"):
        metrics = binary_metrics(labels, probabilities[name], all_threshold)
        if name == "all":
            all_auc = metrics["roc_auc"]
        metrics["delta_auc_vs_all"] = (
            None if metrics["roc_auc"] is None or all_auc is None
            else float(metrics["roc_auc"] - all_auc)
        )
        conditions[name] = metrics
    result = {
        "schema_version": 1,
        "artifact_type": "multiview_frozen_channel_masking_diagnostic",
        "split": dataset.split,
        "sample_count": len(dataset),
        "updated_parameter_count": 0,
        "shared_all_condition_threshold": all_threshold,
        "conditions": conditions,
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
