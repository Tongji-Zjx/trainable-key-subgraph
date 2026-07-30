"""Summaries for formal full-soft-hard selector experiments."""

from __future__ import absolute_import, division, print_function

import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


FORMAL_DATASETS = ("wmrc", "adhd")
FORMAL_OBJECTIVES = ("current", "full_soft", "full_soft_hard")


def _read_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _optional_metric(metrics, primary, fallback=None):
    value = metrics.get(primary)
    if value is None and fallback is not None:
        value = metrics.get(fallback)
    return value


def summarize_selector_transfer_formal(root: Path) -> Dict[str, Any]:
    root = Path(root).resolve()
    datasets = {}
    for dataset in FORMAL_DATASETS:
        native = []
        for objective in FORMAL_OBJECTIVES:
            path = (
                root
                / "{}_{}".format(dataset, objective)
                / "best_evaluation.json"
            )
            if not path.is_file():
                raise FileNotFoundError(str(path))
            evaluation = _read_json(path)
            validation = evaluation["validation"]
            native.append(
                {
                    "objective": objective,
                    "best_epoch": int(evaluation["best_epoch"]),
                    "hard_roc_auc": _optional_metric(
                        validation, "hard_roc_auc", "roc_auc"
                    ),
                    "soft_roc_auc": validation.get("soft_roc_auc"),
                    "balanced_accuracy": validation[
                        "balanced_accuracy"
                    ],
                    "accuracy": validation["accuracy"],
                    "node_probability_mean": validation.get(
                        "node_probability_mean"
                    ),
                    "edge_probability_mean": validation.get(
                        "edge_probability_mean"
                    ),
                    "actual_node_ratio": validation.get(
                        "actual_node_ratio"
                    ),
                    "actual_edge_ratio": validation.get(
                        "actual_edge_ratio"
                    ),
                    "soft_hard_spectral": validation.get(
                        "soft_hard_spectral"
                    ),
                    "soft_hard_gw": validation.get("soft_hard_gw"),
                    "soft_hard_probability_difference": validation.get(
                        "soft_hard_probability_mean_absolute_difference"
                    ),
                }
            )
        probe_path = root / "fair_probe_{}".format(dataset) / (
            "probe/comparison.json"
        )
        if not probe_path.is_file():
            raise FileNotFoundError(str(probe_path))
        probe = _read_json(probe_path)
        if probe.get("test_used") is not False:
            raise ValueError("formal selector probe must not use test")
        rows = probe["rows"]
        by_name = {row["name"]: row for row in rows}
        required = {
            "current",
            "full_soft",
            "full_soft_hard",
            "random",
            "full",
        }
        if set(by_name) != required:
            raise ValueError("formal selector probe conditions mismatch")
        winner = max(
            rows,
            key=lambda row: (
                float(row["roc_auc"]),
                row["name"] == "full_soft_hard",
            ),
        )
        e3_auc = float(by_name["full_soft_hard"]["roc_auc"])
        datasets[dataset] = {
            "native_selector": native,
            "fair_probe": rows,
            "fair_probe_winner": winner["name"],
            "e3_minus_current_auc": (
                e3_auc - float(by_name["current"]["roc_auc"])
            ),
            "e3_minus_random_auc": (
                e3_auc - float(by_name["random"]["roc_auc"])
            ),
            "e3_beats_current_and_random": (
                e3_auc > float(by_name["current"]["roc_auc"])
                and e3_auc > float(by_name["random"]["roc_auc"])
            ),
        }
    return {
        "schema_version": 1,
        "artifact_type": "selector_transfer_formal_summary",
        "root": str(root),
        "test_used": False,
        "datasets": datasets,
        "e3_consistently_beats_current_and_random": all(
            value["e3_beats_current_and_random"]
            for value in datasets.values()
        ),
    }


def _value(value: Any) -> str:
    return "N/A" if value is None else "{:.6f}".format(float(value))


def selector_transfer_formal_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Selector Full–Soft–Hard 正式实验汇总",
        "",
        "- 数据集：WMRC、ADHD",
        "- Test 使用：否",
        "- 训练种子：42",
        "- 主判据：冻结硬图 44 维表示上的 train-only 平衡 Logistic "
        "Validation AUROC",
        "",
    ]
    for dataset in FORMAL_DATASETS:
        values = payload["datasets"][dataset]
        lines.extend(
            [
                "## {}".format(dataset.upper()),
                "",
                "### Selector 原生训练结果",
                "",
                "| 目标 | Best epoch | Hard AUC | Soft AUC | BA | "
                "节点概率 | 边概率 | Soft–Hard谱误差 | Soft–Hard GW代理 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in values["native_selector"]:
            lines.append(
                "| {objective} | {best_epoch} | {hard} | {soft} | "
                "{ba} | {node} | {edge} | {spectral} | {gw} |".format(
                    objective=row["objective"],
                    best_epoch=row["best_epoch"],
                    hard=_value(row["hard_roc_auc"]),
                    soft=_value(row["soft_roc_auc"]),
                    ba=_value(row["balanced_accuracy"]),
                    node=_value(row["node_probability_mean"]),
                    edge=_value(row["edge_probability_mean"]),
                    spectral=_value(row["soft_hard_spectral"]),
                    gw=_value(row["soft_hard_gw"]),
                )
            )
        lines.extend(
            [
                "",
                "### 公平冻结探针",
                "",
                "| 条件 | AUROC | ΔAUC vs current | Site-AUC | BA | "
                "Accuracy | F1 |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in values["fair_probe"]:
            lines.append(
                "| {name} | {auc:.6f} | {delta:+.6f} | {site} | "
                "{ba:.6f} | {accuracy:.6f} | {f1:.6f} |".format(
                    name=row["name"],
                    auc=float(row["roc_auc"]),
                    delta=float(row["delta_auc_vs_reference"]),
                    site=_value(row["site_stratified_roc_auc"]),
                    ba=float(row["balanced_accuracy"]),
                    accuracy=float(row["accuracy"]),
                    f1=float(row["f1"]),
                )
            )
        lines.extend(
            [
                "",
                "- 最佳探针条件：`{}`".format(
                    values["fair_probe_winner"]
                ),
                "- E3 − E0 AUROC：{:+.6f}".format(
                    values["e3_minus_current_auc"]
                ),
                "- E3 − Random AUROC：{:+.6f}".format(
                    values["e3_minus_random_auc"]
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## 冻结判定",
            "",
            "- E3 是否在两个数据集均优于 E0 与 Random：**{}**".format(
                "是"
                if payload[
                    "e3_consistently_beats_current_and_random"
                ]
                else "否"
            ),
            "",
            "> 该判定只回答上游硬图信息保留是否得到一致改善；不使用 "
            "test，也不把单次 validation 结果表述为最终泛化结论。",
            "",
        ]
    )
    return "\n".join(lines)


def write_selector_transfer_formal_summary(
    payload: Mapping[str, Any], output_dir: Path
) -> Dict[str, str]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "formal_summary.json"
    report_path = output_dir / "formal_summary.md"
    for path, content in (
        (json_path, payload),
        (report_path, selector_transfer_formal_markdown(payload)),
    ):
        temporary = path.with_suffix(path.suffix + ".tmp")
        if path.suffix == ".json":
            with temporary.open(
                "w", encoding="utf-8", newline="\n"
            ) as handle:
                json.dump(
                    content,
                    handle,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                handle.write("\n")
        else:
            with temporary.open(
                "w", encoding="utf-8", newline="\n"
            ) as handle:
                handle.write(content)
        os.replace(str(temporary), str(path))
    return {
        "summary_json": str(json_path),
        "summary_markdown": str(report_path),
    }
