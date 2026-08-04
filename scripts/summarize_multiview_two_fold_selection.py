"""Freeze S/V/G choices from two validation folds without test access."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import math
import statistics
from pathlib import Path


def _read(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _fold_arg(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("fold must be NAME=SUMMARY_JSON")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("fold must be NAME=SUMMARY_JSON")
    return name, Path(path)


def _mask_arg(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("masking must be NAME=MASKING_JSON")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("masking must be NAME=MASKING_JSON")
    return name, Path(path)


def _mean(values):
    return float(statistics.mean(float(value) for value in values))


def _std(values):
    values = [float(value) for value in values]
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def _key(stage, row):
    if stage == "s":
        return str(row["static_mode"])
    if stage == "v":
        return str(row["v"])
    return "with_g" if bool(row["g"]) else "without_g"


def _collect(stage, folds):
    by_candidate = {}
    sources = {}
    for fold_name, path in folds:
        payload = _read(path)
        if payload.get("test_used") is not False:
            raise ValueError("two-fold selection requires validation-only summaries")
        expected = {"s": "stage1", "v": "stage2", "g": "stage3"}[stage]
        if payload.get("stage") != expected:
            raise ValueError("fold summary stage mismatch")
        sources[fold_name] = str(path.resolve())
        seen = set()
        for row in payload.get("conditions", ()):
            candidate = _key(stage, row)
            auc = row.get("roc_auc")
            if candidate in seen or auc is None:
                continue
            seen.add(candidate)
            by_candidate.setdefault(candidate, {})[fold_name] = float(auc)
    return by_candidate, sources


def _complete_rows(by_candidate, fold_names, official):
    rows = []
    required = set(fold_names)
    for candidate in sorted(by_candidate):
        values = by_candidate[candidate]
        complete = set(values) == required
        aucs = [values[name] for name in fold_names] if complete else []
        rows.append({
            "candidate": candidate,
            "officially_eligible": candidate in official,
            "complete_across_folds": complete,
            "fold_roc_auc": {name: values.get(name) for name in fold_names},
            "mean_roc_auc": _mean(aucs) if aucs else None,
            "std_roc_auc": _std(aucs) if aucs else None,
        })
    return rows


def _deltas(by_candidate, left, right, fold_names):
    if left not in by_candidate or right not in by_candidate:
        return None
    if any(name not in by_candidate[left] or name not in by_candidate[right]
           for name in fold_names):
        return None
    values = {
        name: float(by_candidate[left][name] - by_candidate[right][name])
        for name in fold_names
    }
    return {
        "fold_delta": values,
        "mean_delta": _mean([values[name] for name in fold_names]),
        "minimum_delta": min(values.values()),
    }


def _mask_delta(path):
    payload = _read(path)
    conditions = payload.get("conditions", {})
    row = conditions.get("mask_g", {})
    value = row.get("delta_auc_vs_all")
    if value is None:
        raise ValueError("G masking artifact lacks mask_g delta_auc_vs_all")
    return float(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("s", "v", "g"), required=True)
    parser.add_argument("--fold", action="append", type=_fold_arg, required=True)
    parser.add_argument("--masking", action="append", type=_mask_arg, default=[])
    parser.add_argument("--maximum-opposite-fold-drop", type=float, default=0.02)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.fold) < 2 or len({name for name, _ in args.fold}) != len(args.fold):
        parser.error("at least two uniquely named folds are required")
    fold_names = [name for name, _ in args.fold]
    by_candidate, sources = _collect(args.stage, args.fold)
    official = {
        "s": {"neural", "residual"},
        "v": {"none", "uot"},
        "g": {"without_g", "with_g"},
    }[args.stage]
    rows = _complete_rows(by_candidate, fold_names, official)
    available = {
        row["candidate"]: row for row in rows
        if row["officially_eligible"] and row["complete_across_folds"]
    }
    if not available:
        raise ValueError("no official candidate is complete across all folds")

    decision = {
        "selected": None,
        "selection_metric": "unweighted_mean_validation_roc_auc",
        "fold_count": len(fold_names),
        "test_used": False,
        "stable_is_control_only": args.stage == "s",
        "legacy_is_control_only": args.stage == "v",
    }
    comparisons = {}
    if args.stage == "s":
        selected = max(available.values(), key=lambda row: row["mean_roc_auc"])
        decision["selected"] = selected["candidate"]
        comparisons["residual_minus_neural"] = _deltas(
            by_candidate, "residual", "neural", fold_names
        )
    elif args.stage == "v":
        if "none" not in available:
            raise ValueError("V=none must be complete as the safe official baseline")
        uot_vs_none = _deltas(by_candidate, "uot", "none", fold_names)
        uot_vs_shuffled = _deltas(by_candidate, "uot", "shuffled", fold_names)
        comparisons["uot_minus_none"] = uot_vs_none
        comparisons["uot_minus_shuffled"] = uot_vs_shuffled
        tolerance = -abs(float(args.maximum_opposite_fold_drop))
        uot_passes = bool(
            "uot" in available
            and uot_vs_none is not None
            and uot_vs_shuffled is not None
            and uot_vs_none["mean_delta"] > 0.0
            and uot_vs_shuffled["mean_delta"] > 0.0
            and uot_vs_none["minimum_delta"] >= tolerance
        )
        decision["uot_gate_passes"] = uot_passes
        decision["selected"] = "uot" if uot_passes else "none"
    else:
        if "without_g" not in available:
            raise ValueError("without-G must be complete as the safe official baseline")
        g_delta = _deltas(by_candidate, "with_g", "without_g", fold_names)
        comparisons["with_g_minus_without_g"] = g_delta
        masking = dict(args.masking)
        if set(masking) != set(fold_names):
            raise ValueError("G selection requires one masking diagnostic per fold")
        masking_deltas = {name: _mask_delta(masking[name]) for name in fold_names}
        comparisons["mask_g_delta_vs_all"] = {
            "fold_delta": masking_deltas,
            "mean_delta": _mean([masking_deltas[name] for name in fold_names]),
            "maximum_delta": max(masking_deltas.values()),
        }
        tolerance = -abs(float(args.maximum_opposite_fold_drop))
        g_passes = bool(
            "with_g" in available
            and g_delta is not None
            and g_delta["mean_delta"] > 0.0
            and g_delta["minimum_delta"] >= tolerance
            and comparisons["mask_g_delta_vs_all"]["mean_delta"] < 0.0
        )
        decision["g_gate_passes"] = g_passes
        decision["selected"] = "with_g" if g_passes else "without_g"

    if decision["selected"] not in official:
        raise AssertionError("an excluded control reached the formal selection")
    payload = {
        "schema_version": 1,
        "artifact_type": "multiview_two_fold_frozen_selection",
        "stage": args.stage,
        "test_used": False,
        "folds": fold_names,
        "source_summaries": sources,
        "official_candidates": sorted(official),
        "excluded_controls": {
            "s": ["stable"], "v": ["legacy", "shuffled"], "g": []
        }[args.stage],
        "conditions": rows,
        "comparisons": comparisons,
        "decision": decision,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "selection.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# 多视图关键子图两折冻结选择：{}".format(args.stage.upper()),
        "",
        "- 主指标：两折 validation AUROC 等权平均",
        "- test 使用：否",
        "- 正式候选：{}".format(", ".join(sorted(official))),
        "- 排除对照：{}".format(", ".join(payload["excluded_controls"]) or "无"),
        "",
        "| 候选 | 正式候选 | 两折完整 | Mean AUROC | Std |",
        "|---|:---:|:---:|---:|---:|",
    ]
    for row in rows:
        mean = "N/A" if row["mean_roc_auc"] is None else "{:.6f}".format(row["mean_roc_auc"])
        std = "N/A" if row["std_roc_auc"] is None else "{:.6f}".format(row["std_roc_auc"])
        lines.append("| {} | {} | {} | {} | {} |".format(
            row["candidate"], "是" if row["officially_eligible"] else "否",
            "是" if row["complete_across_folds"] else "否", mean, std,
        ))
    lines.extend(("", "## 冻结决定", "", "```json",
                  json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True),
                  "```", ""))
    (args.output_dir / "selection.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
