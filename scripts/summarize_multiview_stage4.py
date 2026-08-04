"""Summarize the frozen Stage-4 Author-ST/Critical/Fusion evaluation.

This reporter is deliberately read-only.  It verifies that Stage-2/3 model
choices were frozen without test use and that every model reuses exactly the
same validation-derived threshold on test before computing incremental AUROC.
"""

from __future__ import absolute_import, division, print_function

import argparse
import hashlib
import json
from pathlib import Path


def _read(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _metrics(payload):
    metrics = payload.get("metrics", payload)
    required = (
        "sample_count",
        "roc_auc",
        "balanced_accuracy",
        "accuracy",
        "f1",
    )
    missing = [name for name in required if name not in metrics]
    if missing:
        raise ValueError("Stage-4 evaluation metrics are incomplete: {}".format(missing))
    return metrics


def _threshold(payload):
    if "threshold" not in payload:
        raise ValueError("Stage-4 evaluation has no frozen threshold")
    return float(payload["threshold"])


def _load_pair(validation_path, test_path):
    validation = _read(validation_path)
    test = _read(test_path)
    validation_threshold = _threshold(validation)
    test_threshold = _threshold(test)
    if abs(validation_threshold - test_threshold) > 1.0e-12:
        raise ValueError("validation/test thresholds differ")
    return {
        "threshold": validation_threshold,
        "validation": _metrics(validation),
        "test": _metrics(test),
        "artifacts": {
            "validation": str(Path(validation_path)),
            "validation_sha256": _sha256(validation_path),
            "test": str(Path(test_path)),
            "test_sha256": _sha256(test_path),
        },
    }


def _number(value):
    return "N/A" if value is None else "{:.6f}".format(float(value))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for model in ("author", "critical", "fusion"):
        parser.add_argument(
            "--{}-validation".format(model), type=Path, required=True
        )
        parser.add_argument("--{}-test".format(model), type=Path, required=True)
    parser.add_argument("--stage2-selection", type=Path, required=True)
    parser.add_argument("--stage3-selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    stage2 = _read(args.stage2_selection)
    stage3 = _read(args.stage3_selection)
    for name, selection in (("Stage 2", stage2), ("Stage 3", stage3)):
        if selection.get("test_used") is not False:
            raise ValueError("{} selection is not validation-only".format(name))

    models = {}
    for name in ("author", "critical", "fusion"):
        models[name] = _load_pair(
            getattr(args, "{}_validation".format(name)),
            getattr(args, "{}_test".format(name)),
        )

    increments = {}
    for split in ("validation", "test"):
        fusion_auc = models["fusion"][split]["roc_auc"]
        increments[split] = {
            "fusion_minus_author_auc": (
                None
                if fusion_auc is None or models["author"][split]["roc_auc"] is None
                else float(fusion_auc - models["author"][split]["roc_auc"])
            ),
            "fusion_minus_critical_auc": (
                None
                if fusion_auc is None or models["critical"][split]["roc_auc"] is None
                else float(fusion_auc - models["critical"][split]["roc_auc"])
            ),
        }

    payload = {
        "schema_version": 1,
        "artifact_type": "multiview_stage4_frozen_summary",
        "architecture_selection_source": "validation_only",
        "test_used_for_selection": False,
        "test_threshold_refit": False,
        "stage2_selection": stage2,
        "stage3_selection": stage3,
        "models": models,
        "increments": increments,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "summary.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# 多视图关键子图 Stage 4 冻结评估",
        "",
        "- 架构选择：仅使用 validation",
        "- test 阈值重拟合：否",
        "- test 用于架构选择：否",
        "",
        "| 模型 | Split | AUROC | BA | Accuracy | F1 | Threshold |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("author", "critical", "fusion"):
        for split in ("validation", "test"):
            row = models[name][split]
            lines.append(
                "| {name} | {split} | {auc} | {ba} | {acc} | {f1} | {threshold} |".format(
                    name=name,
                    split=split,
                    auc=_number(row["roc_auc"]),
                    ba=_number(row["balanced_accuracy"]),
                    acc=_number(row["accuracy"]),
                    f1=_number(row["f1"]),
                    threshold=_number(models[name]["threshold"]),
                )
            )
    lines.extend(
        (
            "",
            "## 融合增量",
            "",
            "| Split | Fusion - Author AUROC | Fusion - Critical AUROC |",
            "|---|---:|---:|",
        )
    )
    for split in ("validation", "test"):
        lines.append(
            "| {split} | {author} | {critical} |".format(
                split=split,
                author=_number(increments[split]["fusion_minus_author_auc"]),
                critical=_number(increments[split]["fusion_minus_critical_auc"]),
            )
        )
    (args.output_dir / "summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
