"""Read-only representation audit for the improved SV late-fusion model."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.analysis.sv_signed_gin_bottleneck import (  # noqa: E402
    representation_statistics,
)
from keysubgraph.data.sv_signed_gin_dataset import (  # noqa: E402
    SVSignedGINDataset,
    create_sv_signed_gin_loader,
)
from keysubgraph.data.data_split import file_sha256  # noqa: E402
from keysubgraph.models.sv_signed_gin import (  # noqa: E402
    SVSignedGINClassifier,
    SVSignedGINConfig,
)
from keysubgraph.training.sv_signed_gin_trainer import (  # noqa: E402
    load_sv_signed_gin_checkpoint,
    run_sv_signed_gin_epoch,
)


EXPECTED_VARIANTS = (
    "signed_gin_multibranch_late_fusion",
    "signed_gin_static_anchor_residual",
    "signed_gin_static_anchor_residual_attention",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument(
        "--validation-manifest", type=Path, required=True
    )
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def _trusted_load(path, device):
    try:
        return torch.load(
            str(Path(path).resolve()),
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(str(Path(path).resolve()), map_location=device)


def _collect(model, loader, device):
    values = {
        "gin_representation": [],
        "gin_normalized_representation": [],
        "gin_projection": [],
        "static_projection": [],
        "variation_projection": [],
        "final_representation": [],
    }
    labels = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch)
            labels.extend(int(value) for value in batch.labels.tolist())
            for name in values:
                tensor = getattr(output, name)
                if tensor is None:
                    raise RuntimeError(
                        "late-fusion diagnostic is missing {}".format(
                            name
                        )
                    )
                values[name].append(tensor.detach().cpu())
    arrays = {
        name: torch.cat(items, dim=0).numpy()
        for name, items in values.items()
    }
    labels_array = np.asarray(labels, dtype=np.int64)
    return {
        name: representation_statistics(value, labels_array)
        for name, value in arrays.items()
    }


def _attention_statistics(model, loader, device):
    normalized_entropy = []
    maximum_weight = []
    effective_nodes = []
    degree_correlations = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            moved = batch.to(device)
            output = model(moved)
            for sample, encoded in zip(
                moved.samples, output.encoder_outputs
            ):
                for window, weights in zip(
                    sample.windows, encoded.node_attention
                ):
                    total = weights.sum().clamp_min(1.0e-12)
                    probabilities = weights / total
                    entropy = -(
                        probabilities
                        * probabilities.clamp_min(1.0e-12).log()
                    ).sum()
                    node_count = int(probabilities.numel())
                    normalized = (
                        float(entropy.detach().cpu())
                        / float(np.log(node_count))
                        if node_count > 1
                        else 0.0
                    )
                    normalized_entropy.append(normalized)
                    maximum_weight.append(
                        float(probabilities.max().detach().cpu())
                    )
                    effective_nodes.append(
                        float(torch.exp(entropy).detach().cpu())
                    )
                    degree = (
                        window.adjacency.abs().sum(dim=-1)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    attention = (
                        probabilities.detach().cpu().numpy()
                    )
                    correlation = spearmanr(
                        attention, degree
                    ).correlation
                    if np.isfinite(correlation):
                        degree_correlations.append(float(correlation))

    def summary(values):
        array = np.asarray(values, dtype=np.float64)
        if array.size == 0:
            return None
        return {
            "count": int(array.size),
            "mean": float(array.mean()),
            "median": float(np.median(array)),
            "minimum": float(array.min()),
            "maximum": float(array.max()),
        }

    return {
        "normalized_entropy": summary(normalized_entropy),
        "maximum_node_weight": summary(maximum_weight),
        "effective_node_count": summary(effective_nodes),
        "attention_absolute_degree_spearman": summary(
            degree_correlations
        ),
    }


def _attention_ablation_metrics(
    model, loader, device, class_weights
):
    if not model.config.gin_residual_attention:
        return None
    parameter = model.encoder.attention_residual_gate_logit
    original = parameter.detach().clone()
    with torch.no_grad():
        parameter.fill_(-20.0)
    try:
        return run_sv_signed_gin_epoch(
            model,
            loader,
            device,
            class_weights,
            include_predictions=False,
        )
    finally:
        with torch.no_grad():
            parameter.copy_(original)


def _atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def _expected_provenance(train, validation, scaler_path):
    if train.split != "train" or validation.split != "validation":
        raise ValueError(
            "late-fusion diagnostic requires train/validation manifests"
        )
    if set(train.sample_keys).intersection(validation.sample_keys):
        raise ValueError(
            "late-fusion diagnostic train/validation samples overlap"
        )
    keys = (
        "protocol_sha256",
        "selector_checkpoint_sha256",
        "selection_mode",
        "selection_seed",
    )
    for key in keys:
        if train.manifest[key] != validation.manifest[key]:
            raise ValueError(
                "late-fusion manifests disagree on {}".format(key)
            )
    if train.manifest["selection_mode"] != "learned":
        raise ValueError(
            "late-fusion diagnostic requires learned selection"
        )
    return {
        "protocol_sha256": train.manifest["protocol_sha256"],
        "selector_checkpoint_sha256": train.manifest[
            "selector_checkpoint_sha256"
        ],
        "selection_mode": train.manifest["selection_mode"],
        "selection_seed": int(train.manifest["selection_seed"]),
        "train_manifest_sha256": file_sha256(train.manifest_path),
        "validation_manifest_sha256": file_sha256(
            validation.manifest_path
        ),
        "scaler_sha256": file_sha256(scaler_path),
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    json_path = output_dir / "diagnostic.json"
    markdown_path = output_dir / "summary.md"
    if json_path.exists() or markdown_path.exists():
        raise FileExistsError("late-fusion diagnostic output exists")
    device = torch.device(args.device)
    raw = _trusted_load(args.checkpoint, device)
    model = SVSignedGINClassifier(
        SVSignedGINConfig(**raw["model_config"])
    ).to(device)
    if model.config.variant not in EXPECTED_VARIANTS:
        raise ValueError(
            "diagnostic requires a supported safe-fusion model"
        )
    train = SVSignedGINDataset(args.train_manifest, args.scaler)
    validation = SVSignedGINDataset(
        args.validation_manifest, args.scaler
    )
    expected_provenance = _expected_provenance(
        train, validation, args.scaler
    )
    checkpoint = load_sv_signed_gin_checkpoint(
        args.checkpoint,
        model,
        device,
        expected_provenance=expected_provenance,
    )
    loaders = {
        "train": create_sv_signed_gin_loader(
            train,
            args.batch_size,
            int(checkpoint["training_config"]["seed"]),
            False,
            args.num_workers,
        ),
        "validation": create_sv_signed_gin_loader(
            validation,
            args.batch_size,
            int(checkpoint["training_config"]["seed"]),
            False,
            args.num_workers,
        ),
    }
    result = {
        "artifact_type": "sv_late_fusion_representation_diagnostic",
        "read_only": True,
        "parameter_updates": 0,
        "variant": model.config.variant,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "provenance": expected_provenance,
        "splits": {},
    }
    for name, loader in loaders.items():
        metrics = run_sv_signed_gin_epoch(
            model,
            loader,
            device,
            checkpoint["class_weights"],
            include_predictions=False,
        )
        result["splits"][name] = {
            "metrics": {
                key: value
                for key, value in metrics.items()
                if key
                in (
                    "roc_auc",
                    "site_stratified_roc_auc",
                    "balanced_accuracy",
                    "accuracy",
                    "f1",
                    "branch_metrics",
                    "fusion_weights",
                    "residual_gates",
                )
            },
            "representations": _collect(model, loader, device),
            "attention": _attention_statistics(
                model, loader, device
            ),
        }
        attention_ablation = _attention_ablation_metrics(
            model,
            loader,
            device,
            checkpoint["class_weights"],
        )
        if attention_ablation is not None:
            result["splits"][name]["attention_ablation_metrics"] = {
                key: attention_ablation.get(key)
                for key in (
                    "roc_auc",
                    "site_stratified_roc_auc",
                    "balanced_accuracy",
                    "accuracy",
                    "f1",
                )
            }
    validation_gin = result["splits"]["validation"][
        "representations"
    ]["gin_representation"]
    validation_projection = result["splits"]["validation"][
        "representations"
    ]["gin_projection"]
    result["checks"] = {
        "gin_representation_not_low_rank": (
            float(validation_gin["normalized_effective_rank"]) >= 0.10
        ),
        "gin_projection_not_nearly_collinear": (
            validation_projection["mean_pairwise_cosine"] is None
            or float(
                validation_projection["mean_pairwise_cosine"]
            )
            < 0.995
        ),
        "fusion_controls_nonnegative": all(
            float(value) >= 0.0
            for field in ("fusion_weights", "residual_gates")
            for value in (
                result["splits"]["validation"]["metrics"].get(
                    field, {}
                )
            ).values()
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(json_path, result)
    lines = [
        "# SV SignedGIN 改进版只读诊断",
        "",
        "- 参数更新量：0",
        "- Test 使用：否",
        "",
        "## Validation 表示",
        "",
        "| 表示 | 归一化有效秩 | 平均余弦 |",
        "|---|---:|---:|",
    ]
    for name, values in result["splits"]["validation"][
        "representations"
    ].items():
        cosine = values["mean_pairwise_cosine"]
        lines.append(
            "| {} | {:.6f} | {} |".format(
                name,
                values["normalized_effective_rank"],
                "N/A" if cosine is None else "{:.6f}".format(cosine),
            )
        )
    lines.extend(
        [
            "",
            "## Validation 分支与融合",
            "",
            "```json",
            json.dumps(
                result["splits"]["validation"]["metrics"],
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            "```",
            "",
            "## Validation Attention",
            "",
            "```json",
            json.dumps(
                {
                    "distribution": result["splits"][
                        "validation"
                    ]["attention"],
                    "masked_metrics": result["splits"][
                        "validation"
                    ].get("attention_ablation_metrics"),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            "```",
            "",
            "## 闸门",
            "",
        ]
    )
    for name, passed in result["checks"].items():
        lines.append(
            "- {}：{}".format(name, "通过" if passed else "未通过")
        )
    markdown_path.write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "diagnostic": str(json_path),
                "summary": str(markdown_path),
                "checks": result["checks"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
