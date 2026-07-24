"""Summarize paired Exact-STSE Coord/NoCoord runs across fixed seeds."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import math
from pathlib import Path


VARIANTS = ("exact_stse", "exact_stse_no_coord")
METRICS = ("balanced_accuracy", "roc_auc")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(42, 43, 44))
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def _read_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _mean_std(values):
    mean = sum(values) / float(len(values))
    if len(values) < 2:
        standard_deviation = 0.0
    else:
        standard_deviation = math.sqrt(
            sum((value - mean) ** 2 for value in values)
            / float(len(values) - 1)
        )
    return {"mean": mean, "standard_deviation": standard_deviation}


def main():
    args = parse_args()
    root = args.experiment_dir.resolve()
    rows = []
    for variant in VARIANTS:
        for seed in args.seeds:
            run_dir = root / "{}_seed{}".format(variant, seed)
            evaluation_path = run_dir / "best_evaluation.json"
            summary_path = run_dir / "run_summary.json"
            if not evaluation_path.is_file() or not summary_path.is_file():
                raise FileNotFoundError(
                    "incomplete Exact-STSE run: {}".format(run_dir)
                )
            evaluation = _read_json(evaluation_path)
            summary = _read_json(summary_path)
            row = {
                "variant": variant,
                "seed": int(seed),
                "best_epoch": int(evaluation["best_epoch"]),
                "parameter_count": int(summary["parameter_count"]),
                "mean_epoch_seconds": sum(
                    float(record["train"]["elapsed_seconds"])
                    + float(record["validation"]["elapsed_seconds"])
                    for record in _read_json(run_dir / "history.json")
                )
                / float(int(summary["epochs_completed"])),
                "train": {
                    name: evaluation["train"].get(name)
                    for name in METRICS
                },
                "validation": {
                    name: evaluation["validation"].get(name)
                    for name in METRICS
                },
            }
            rows.append(row)
    aggregates = {}
    for variant in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        aggregates[variant] = {
            partition: {
                metric: _mean_std(
                    [
                        float(row[partition][metric])
                        for row in selected
                        if row[partition][metric] is not None
                    ]
                )
                for metric in METRICS
            }
            for partition in ("train", "validation")
        }
        aggregates[variant]["parameter_count"] = selected[0][
            "parameter_count"
        ]
        aggregates[variant]["mean_epoch_seconds"] = _mean_std(
            [float(row["mean_epoch_seconds"]) for row in selected]
        )
    paired = []
    for seed in args.seeds:
        coord = next(
            row
            for row in rows
            if row["variant"] == "exact_stse" and row["seed"] == seed
        )
        no_coord = next(
            row
            for row in rows
            if row["variant"] == "exact_stse_no_coord"
            and row["seed"] == seed
        )
        paired.append(
            {
                "seed": int(seed),
                "validation_balanced_accuracy_delta": (
                    float(coord["validation"]["balanced_accuracy"])
                    - float(no_coord["validation"]["balanced_accuracy"])
                ),
                "validation_roc_auc_delta": (
                    float(coord["validation"]["roc_auc"])
                    - float(no_coord["validation"]["roc_auc"])
                ),
            }
        )
    auc_deltas = [
        row["validation_roc_auc_delta"] for row in paired
    ]
    result = {
        "experiment_dir": str(root),
        "seeds": [int(seed) for seed in args.seeds],
        "rows": rows,
        "aggregates": aggregates,
        "paired_coord_minus_no_coord": paired,
        "interpretation_gate": {
            "mean_validation_auc_delta": _mean_std(auc_deltas)["mean"],
            "seed_count_with_at_least_3pp_auc_gain": sum(
                value >= 0.03 for value in auc_deltas
            ),
            "supports_coordinate_dependence_rule": (
                _mean_std(auc_deltas)["mean"] >= 0.03
                and sum(value > 0.0 for value in auc_deltas) >= 2
            ),
        },
    }
    output_json = args.output_json.resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
    lines = [
        "# Exact-STSE 坐标消融汇总",
        "",
        "| 模型 | Train BA | Train AUC | Val BA | Val AUC | 参数量 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        aggregate = aggregates[variant]
        cells = []
        for partition, metric in (
            ("train", "balanced_accuracy"),
            ("train", "roc_auc"),
            ("validation", "balanced_accuracy"),
            ("validation", "roc_auc"),
        ):
            value = aggregate[partition][metric]
            cells.append(
                "{:.4f} ± {:.4f}".format(
                    value["mean"], value["standard_deviation"]
                )
            )
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                variant,
                cells[0],
                cells[1],
                cells[2],
                cells[3],
                aggregate["parameter_count"],
            )
        )
    lines.extend(
        [
            "",
            "## 配对差异（Coord − NoCoord）",
            "",
            "| Seed | Val BA Δ | Val AUC Δ |",
            "|---:|---:|---:|",
        ]
    )
    for row in paired:
        lines.append(
            "| {} | {:+.4f} | {:+.4f} |".format(
                row["seed"],
                row["validation_balanced_accuracy_delta"],
                row["validation_roc_auc_delta"],
            )
        )
    lines.extend(
        [
            "",
            "平均 Val AUC 差异：{:+.4f}。".format(
                result["interpretation_gate"]["mean_validation_auc_delta"]
            ),
            "",
            "是否满足预注册坐标依赖判据：{}。".format(
                "是"
                if result["interpretation_gate"][
                    "supports_coordinate_dependence_rule"
                ]
                else "否"
            ),
        ]
    )
    output_md = args.output_md.resolve()
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("summary json: {}".format(output_json))
    print("summary markdown: {}".format(output_md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
