#!/usr/bin/env python3
"""Select S0-S3 using four disjoint development validation rotations only."""

from __future__ import absolute_import, division, print_function

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.background.safe_fusion import score_logits  # noqa: E402


STAGE_ORDER = ("s0", "s1", "s2", "s3")


def array_fingerprint(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        values = np.asarray(array)
        digest.update(str(values.shape).encode("ascii"))
        digest.update(b"\0")
        for value in values.tolist():
            digest.update(str(value).encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for stage in STAGE_ORDER:
        parser.add_argument(
            "--{}-fold-dir".format(stage),
            action="append",
            type=Path,
            required=True,
        )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stability-penalty", type=float, default=0.50)
    parser.add_argument("--simplicity-tolerance", type=float, default=0.002)
    return parser.parse_args()


def load_validation(directory):
    path = Path(directory) / "validation_features.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = np.load(str(path), allow_pickle=False)
    return {
        "path": str(path.resolve()),
        "sample_keys": payload["sample_keys"].astype(str),
        "sites": payload["sites"].astype(str),
        "labels": payload["labels"].astype(np.int64),
        "logits": payload["background_logits"].astype(np.float64),
    }


def summarize_stage(stage, directories, stability_penalty):
    if len(directories) != 4:
        raise ValueError("{} requires exactly four rotations".format(stage))
    folds = [load_validation(path) for path in directories]
    seen = set()
    rotation_rows = []
    for rotation, fold in enumerate(folds):
        keys = set(fold["sample_keys"].tolist())
        if seen.intersection(keys):
            raise ValueError("{} validation rotations overlap".format(stage))
        seen.update(keys)
        metrics = score_logits(fold["labels"], fold["sites"], fold["logits"])
        rotation_rows.append({"rotation": rotation, "metrics": metrics, "source": fold["path"]})
        rotation_rows[-1]["sample_set_sha256"] = array_fingerprint(
            np.sort(fold["sample_keys"])
        )
        order = np.argsort(fold["sample_keys"], kind="mergesort")
        rotation_rows[-1]["sample_label_site_sha256"] = array_fingerprint(
            fold["sample_keys"][order],
            fold["labels"][order],
            fold["sites"][order],
        )
    aucs = np.asarray([row["metrics"]["roc_auc"] for row in rotation_rows])
    labels = np.concatenate([fold["labels"] for fold in folds])
    sites = np.concatenate([fold["sites"] for fold in folds])
    logits = np.concatenate([fold["logits"] for fold in folds])
    development = score_logits(labels, sites, logits)
    return {
        "stage": stage,
        "development_sample_count": len(seen),
        "mean_rotation_roc_auc": float(np.mean(aucs)),
        "std_rotation_roc_auc": float(np.std(aucs)),
        "worst_rotation_roc_auc": float(np.min(aucs)),
        "stability_objective": float(
            np.mean(aucs) - float(stability_penalty) * np.std(aucs)
        ),
        "development_validation_rotation_metrics": development,
        "rotations": rotation_rows,
    }


def atomic_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main():
    args = parse_args()
    directories = {
        stage: getattr(args, "{}_fold_dir".format(stage)) for stage in STAGE_ORDER
    }
    stages = [
        summarize_stage(stage, directories[stage], args.stability_penalty)
        for stage in STAGE_ORDER
    ]
    reference = stages[0]["rotations"]
    for stage in stages[1:]:
        for expected, observed in zip(reference, stage["rotations"]):
            if (
                expected["sample_set_sha256"] != observed["sample_set_sha256"]
                or expected["sample_label_site_sha256"]
                != observed["sample_label_site_sha256"]
            ):
                raise ValueError(
                    "static stages do not use identical samples for rotation {}".format(
                        expected["rotation"]
                    )
                )
    best_objective = max(row["stability_objective"] for row in stages)
    eligible = [
        row for row in stages
        if row["stability_objective"] >= best_objective - args.simplicity_tolerance
    ]
    selected = min(eligible, key=lambda row: STAGE_ORDER.index(row["stage"]))
    report = {
        "artifact_type": "mokse_background_safe_stage_selection_v1",
        "selection_data": "four_disjoint_development_validation_rotations",
        "fixed_test_loaded": False,
        "stability_penalty": float(args.stability_penalty),
        "simplicity_tolerance": float(args.simplicity_tolerance),
        "selected_stage": selected["stage"],
        "stages": stages,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write(
        args.output_dir / "selection.json",
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    lines = [
        "# MoKSE-Net-BG-Safe 静态条件筛选",
        "",
        "固定test未加载；选择仅使用四次development validation轮换。",
        "",
        "| Stage | Mean AUC | Std | Worst | Stability objective |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in stages:
        lines.append(
            "| {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} |".format(
                row["stage"],
                row["mean_rotation_roc_auc"],
                row["std_rotation_roc_auc"],
                row["worst_rotation_roc_auc"],
                row["stability_objective"],
            )
        )
    lines.extend(("", "选择结果：`{}`。".format(selected["stage"])))
    atomic_write(args.output_dir / "selection.md", "\n".join(lines) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
