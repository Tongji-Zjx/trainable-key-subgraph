"""Fair frozen-feature probes for selector-transfer comparisons."""

from __future__ import absolute_import, division, print_function

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from keysubgraph.data.sv_signed_gin_manifest import (
    read_sv_signed_gin_manifest,
)
from keysubgraph.training.dual_sgw_feature_trainer import binary_metrics
from keysubgraph.training.sv_signed_gin_trainer import (
    site_stratified_roc_auc,
)


def _load_probe_split(path: Path) -> Dict[str, Any]:
    manifest, records = read_sv_signed_gin_manifest(path)
    if not records:
        raise ValueError("selector-transfer probe manifest is empty")
    features = np.asarray(
        [
            np.concatenate(
                (
                    record.static_features.detach().cpu().numpy(),
                    record.variation.detach().cpu().numpy(),
                )
            )
            for record in records
        ],
        dtype=np.float64,
    )
    if features.ndim != 2 or features.shape[1] != 44:
        raise ValueError("selector-transfer probe requires 44-D features")
    if not bool(np.isfinite(features).all()):
        raise ValueError("selector-transfer probe features are non-finite")
    return {
        "manifest": manifest,
        "sample_keys": [record.sample_key for record in records],
        "labels": np.asarray([record.label for record in records], dtype=int),
        "sites": [str(record.site) for record in records],
        "features": features,
    }


def _validate_split_pair(
    train: Mapping[str, Any], validation: Mapping[str, Any]
) -> None:
    if train["split"] != "train":
        raise ValueError("selector-transfer train manifest is not train")
    if validation["split"] != "validation":
        raise ValueError(
            "selector-transfer validation manifest is not validation"
        )
    if train["protocol_sha256"] != validation["protocol_sha256"]:
        raise ValueError("selector-transfer manifests mix protocols")
    if set(train["labels"].tolist()) != {0, 1}:
        raise ValueError("selector-transfer train split needs both classes")
    if set(validation["labels"].tolist()) != {0, 1}:
        raise ValueError(
            "selector-transfer validation split needs both classes"
        )


def _manifest_data(path: Path) -> Dict[str, Any]:
    data = _load_probe_split(path)
    data.update(
        {
            "split": data["manifest"]["split"],
            "protocol_sha256": data["manifest"]["protocol_sha256"],
        }
    )
    return data


def evaluate_selector_transfer_feature_set(
    name: str,
    train: Mapping[str, Any],
    validation: Mapping[str, Any],
    provenance: Mapping[str, Any],
    seed: int = 42,
) -> Dict[str, Any]:
    """Fit one balanced linear probe using train only and score validation."""
    _validate_split_pair(train, validation)
    train_features_raw = np.asarray(train["features"], dtype=np.float64)
    validation_features_raw = np.asarray(
        validation["features"], dtype=np.float64
    )
    if (
        train_features_raw.ndim != 2
        or validation_features_raw.ndim != 2
        or train_features_raw.shape[1] != 44
        or validation_features_raw.shape[1] != 44
        or not bool(np.isfinite(train_features_raw).all())
        or not bool(np.isfinite(validation_features_raw).all())
    ):
        raise ValueError(
            "selector-transfer feature sets require finite 44-D features"
        )
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features_raw)
    validation_features = scaler.transform(validation_features_raw)
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=int(seed),
        solver="liblinear",
    )
    classifier.fit(train_features, train["labels"])
    train_probabilities = classifier.predict_proba(train_features)[:, 1]
    validation_probabilities = classifier.predict_proba(
        validation_features
    )[:, 1]
    train_metrics = binary_metrics(
        train["labels"].tolist(), train_probabilities.tolist(), 0.5
    )
    validation_metrics = binary_metrics(
        validation["labels"].tolist(),
        validation_probabilities.tolist(),
        0.5,
    )
    validation_metrics["site_stratified_roc_auc"] = (
        site_stratified_roc_auc(
            validation["labels"].tolist(),
            validation_probabilities.tolist(),
            validation["sites"],
        )
    )
    return {
        "name": str(name),
        "feature_definition": "static_28_plus_variation_16",
        "feature_dimension": 44,
        "probe": {
            "type": "logistic_regression",
            "class_weight": "balanced",
            "threshold": 0.5,
            "seed": int(seed),
            "scaler_fit_split": "train",
            "classifier_fit_split": "train",
        },
        "provenance": dict(provenance),
        "sample_keys": {
            "train": train["sample_keys"],
            "validation": validation["sample_keys"],
        },
        "labels": {
            "train": train["labels"].tolist(),
            "validation": validation["labels"].tolist(),
        },
        "train": train_metrics,
        "validation": validation_metrics,
        "validation_probabilities": validation_probabilities.tolist(),
    }


def _compare_selector_transfer_results(
    results: Sequence[Mapping[str, Any]], seed: int
) -> Dict[str, Any]:
    reference = results[0]
    for result in results[1:]:
        if result["sample_keys"] != reference["sample_keys"]:
            raise ValueError(
                "selector-transfer conditions do not align by sample"
            )
        if result["labels"] != reference["labels"]:
            raise ValueError(
                "selector-transfer conditions do not align by label"
            )
        if (
            result["provenance"]["protocol_sha256"]
            != reference["provenance"]["protocol_sha256"]
        ):
            raise ValueError("selector-transfer conditions mix protocols")
    reference_auc = reference["validation"]["roc_auc"]
    rows = []
    for result in results:
        metrics = result["validation"]
        rows.append(
            {
                "name": result["name"],
                "roc_auc": metrics["roc_auc"],
                "delta_auc_vs_reference": (
                    float(metrics["roc_auc"]) - float(reference_auc)
                ),
                "site_stratified_roc_auc": metrics[
                    "site_stratified_roc_auc"
                ],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1"],
            }
        )
    return {
        "schema_version": 1,
        "artifact_type": "selector_transfer_frozen_probe_comparison",
        "reference_condition": reference["name"],
        "test_used": False,
        "parameter_update_scope": (
            "independent balanced logistic probe per condition; "
            "selector and hard graphs frozen"
        ),
        "seed": int(seed),
        "rows": rows,
        "conditions": results,
    }


def evaluate_selector_transfer_condition(
    name: str,
    train_manifest: Path,
    validation_manifest: Path,
    seed: int = 42,
) -> Dict[str, Any]:
    """Fit one probe from immutable SV manifests."""
    train = _manifest_data(train_manifest)
    validation = _manifest_data(validation_manifest)
    manifest = train["manifest"]
    provenance = {
        "protocol_sha256": manifest["protocol_sha256"],
        "train_manifest": str(Path(train_manifest).resolve()),
        "validation_manifest": str(
            Path(validation_manifest).resolve()
        ),
        "selection_mode": manifest["selection_mode"],
        "selection_seed": int(manifest["selection_seed"]),
        "selector_checkpoint_sha256": manifest[
            "selector_checkpoint_sha256"
        ],
    }
    return evaluate_selector_transfer_feature_set(
        name, train, validation, provenance, seed=seed
    )


def compare_selector_transfer_feature_sets(
    conditions: Sequence[
        Tuple[
            str,
            Mapping[str, Any],
            Mapping[str, Any],
            Mapping[str, Any],
        ]
    ],
    seed: int = 42,
) -> Dict[str, Any]:
    if len(conditions) < 2:
        raise ValueError("selector-transfer comparison needs two conditions")
    results = [
        evaluate_selector_transfer_feature_set(
            name, train, validation, provenance, seed=seed
        )
        for name, train, validation, provenance in conditions
    ]
    return _compare_selector_transfer_results(results, seed)


def compare_selector_transfer_conditions(
    conditions: Sequence[Tuple[str, Path, Path]],
    seed: int = 42,
) -> Dict[str, Any]:
    if len(conditions) < 2:
        raise ValueError("selector-transfer comparison needs two conditions")
    results = [
        evaluate_selector_transfer_condition(
            name, train_manifest, validation_manifest, seed=seed
        )
        for name, train_manifest, validation_manifest in conditions
    ]
    return _compare_selector_transfer_results(results, seed)


def _format_optional(value: Any) -> str:
    return "N/A" if value is None else "{:.6f}".format(float(value))


def selector_transfer_probe_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Selector Full–Soft–Hard 公平冻结探针对照",
        "",
        "- 特征：冻结硬图的 28 维静态结构 + 16 维 Variation",
        "- 探针：各条件独立使用 train-only 标准化与平衡 Logistic",
        "- Test 使用：否",
        "- 主指标：Validation AUROC",
        "- 参考条件：`{}`".format(payload["reference_condition"]),
        "",
        "| 条件 | AUROC | ΔAUC | Site-AUC | BA | Accuracy | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {name} | {auc:.6f} | {delta:+.6f} | {site} | "
            "{ba:.6f} | {accuracy:.6f} | {f1:.6f} |".format(
                name=row["name"],
                auc=float(row["roc_auc"]),
                delta=float(row["delta_auc_vs_reference"]),
                site=_format_optional(row["site_stratified_roc_auc"]),
                ba=float(row["balanced_accuracy"]),
                accuracy=float(row["accuracy"]),
                f1=float(row["f1"]),
            )
        )
    lines.extend(
        [
            "",
            "> 所有 selector 与硬图均冻结；每个条件的探针只在 train "
            "拟合，validation 仅用于评估，未使用 test。",
            "",
        ]
    )
    return "\n".join(lines)


def write_selector_transfer_probe_artifacts(
    payload: Mapping[str, Any], output_dir: Path
) -> Dict[str, str]:
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("selector-transfer probe output exists")
    output_dir.mkdir(parents=True)
    json_path = output_dir / "comparison.json"
    markdown_path = output_dir / "summary.md"
    temporary = json_path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            payload, handle, ensure_ascii=False, indent=2, sort_keys=True
        )
        handle.write("\n")
    os.replace(str(temporary), str(json_path))
    temporary = markdown_path.with_suffix(".md.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(selector_transfer_probe_markdown(payload))
    os.replace(str(temporary), str(markdown_path))
    return {
        "comparison_json": str(json_path),
        "summary_markdown": str(markdown_path),
    }
