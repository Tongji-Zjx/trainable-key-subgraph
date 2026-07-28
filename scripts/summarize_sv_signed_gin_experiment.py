"""Summarize SG0/SG1/SG2 validation or test evaluation artifacts."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("validation", "test"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--threshold-strategy",
        choices=("balanced_accuracy", "accuracy"),
        default="balanced_accuracy",
    )
    return parser.parse_args()


def _evaluation(run, split, strategy):
    if split == "validation":
        path = run / "best_evaluation.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload["metrics"][strategy]
    else:
        path = run / "test_evaluation.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload["metrics"]
    return path, metrics


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    rows = []
    for run in sorted(path for path in root.iterdir() if path.is_dir()):
        item = _evaluation(
            run, args.split, args.threshold_strategy
        )
        if item is None:
            continue
        path, metrics = item
        rows.append(
            {
                "run": run.name,
                "roc_auc": metrics.get("roc_auc"),
                "site_stratified_roc_auc": metrics.get(
                    "site_stratified_roc_auc"
                ),
                "balanced_accuracy": metrics.get(
                    "balanced_accuracy"
                ),
                "accuracy": metrics.get("accuracy"),
                "f1": metrics.get("f1"),
                "artifact": str(path),
            }
        )
    if not rows:
        raise ValueError("no SV Signed-GIN evaluations were found")
    lines = [
        "# SV-HardSGW Signed-GIN {}汇总".format(args.split),
        "",
        "| Run | AUROC | Site-stratified AUROC | BA | Accuracy | F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        values = []
        for key in (
            "roc_auc",
            "site_stratified_roc_auc",
            "balanced_accuracy",
            "accuracy",
            "f1",
        ):
            value = row[key]
            values.append(
                "N/A" if value is None else "{:.6f}".format(float(value))
            )
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                row["run"], *values
            )
        )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
