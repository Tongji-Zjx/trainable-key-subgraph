"""Summarize D0--D4 evaluation JSON files into one Markdown table."""

from __future__ import absolute_import, print_function

import argparse
import json
from pathlib import Path


def _metrics(payload):
    if "metrics" in payload:
        return payload["metrics"]
    if "validation" in payload:
        return payload["validation"]
    raise ValueError("evaluation JSON contains no metrics")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("d0", "d1", "d2", "d3", "d4"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for name in ("d0", "d1", "d2", "d3", "d4"):
        path = getattr(args, name)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        metrics = _metrics(payload)
        rows.append(
            (
                name.upper(),
                metrics.get("balanced_accuracy"),
                metrics.get("roc_auc"),
                metrics.get("accuracy"),
                metrics.get("f1"),
                metrics.get("threshold"),
            )
        )
    lines = [
        "# Dual-STSE-HardSGW D0–D4 汇总",
        "",
        "| 模型 | BA | AUROC | Accuracy | F1 | Threshold |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                *[
                    (
                        "{:.6f}".format(value)
                        if isinstance(value, (int, float))
                        else str(value)
                    )
                    for value in row
                ]
            )
        )
    by_name = {row[0]: row for row in rows}
    d3_d2 = by_name["D3"][2] - by_name["D2"][2]
    d4_d0 = by_name["D4"][2] - by_name["D0"][2]
    lines.extend(
        (
            "",
            "- D3 − D2 AUROC：{:+.6f}".format(d3_d2),
            "- D4 − D0 AUROC：{:+.6f}".format(d4_d0),
            "- 学习型关键图有效：{}".format(d3_d2 > 0.0),
            "- 双通道带来增益：{}".format(d4_d0 > 0.0),
            "",
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(str(args.output.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
