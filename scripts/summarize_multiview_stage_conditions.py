"""Summarize frozen validation-only multi-view stage conditions."""

from __future__ import absolute_import, division, print_function

import argparse
import json
from pathlib import Path


def _read(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _number(value):
    return "N/A" if value is None else "{:.6f}".format(float(value))


def _formally_admissible(row, stage):
    """Return whether a validation condition may become the formal model.

    Stable S and legacy V remain useful controls, but the frozen experiment
    contract explicitly forbids promoting either one.  Shuffled
    correspondence is likewise a negative control only.
    """
    if row.get("static_mode") == "stable":
        return False
    if stage in ("stage2", "stage3") and row.get("v") in (
        "legacy", "shuffled"
    ):
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-dir", type=Path, action="append", required=True)
    parser.add_argument("--stage", choices=("stage1", "stage2", "stage3"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for directory in args.condition_dir:
        spec = _read(directory / "condition_spec.json")
        evaluation = _read(directory / "validation_evaluation.json")
        diagnostic = _read(directory / "validation_diagnostic.json")
        metrics = evaluation.get("metrics", evaluation)
        representation = diagnostic.get("representations", {}).get(
            "critical_final", {}
        )
        rows.append({
            "condition": spec["condition_name"],
            "static_mode": spec["static_mode"],
            "v": (
                "legacy" if spec.get("legacy_v") else
                (spec.get("correspondence", "uot") if spec.get("enable_v") else "none")
            ),
            "g": bool(spec.get("enable_g")),
            "roc_auc": metrics.get("roc_auc"),
            "balanced_accuracy": metrics.get("balanced_accuracy"),
            "accuracy": metrics.get("accuracy"),
            "f1": metrics.get("f1"),
            "q_standardized_mae": diagnostic.get("q_standardized_mae"),
            "delta_q_standardized_mae": diagnostic.get("delta_q_standardized_mae"),
            "representation_effective_rank": representation.get("effective_rank"),
            "representation_normalized_effective_rank": representation.get(
                "normalized_effective_rank"
            ),
            "gates": diagnostic.get("gates", {}),
        })
    rows.sort(key=lambda row: row["condition"])
    eligible = [row for row in rows if row["roc_auc"] is not None]
    admissible = [
        row for row in eligible if _formally_admissible(row, args.stage)
    ]
    best_observed = max(eligible, key=lambda row: row["roc_auc"]) if eligible else None
    best = max(admissible, key=lambda row: row["roc_auc"]) if admissible else None
    decision = {
        "best_validation_condition": None if best is None else best["condition"],
        "best_admissible_validation_condition": (
            None if best is None else best["condition"]
        ),
        "best_observed_condition_including_negative_control": (
            None if best_observed is None else best_observed["condition"]
        ),
        "negative_control_is_not_deployable": True,
        "formal_static_modes": ["neural", "residual"],
        "stable_static_is_control_only": True,
        "formal_v_modes": ["none", "uot"],
        "legacy_v_is_control_only": True,
    }
    if args.stage == "stage2":
        real = next((row for row in rows if row["v"] == "uot"), None)
        shuffled = next((row for row in rows if row["v"] == "shuffled"), None)
        no_v = next((row for row in rows if row["v"] == "none"), None)
        legacy = next((row for row in rows if row["v"] == "legacy"), None)
        decision["real_minus_shuffled_auc"] = (
            None if real is None or shuffled is None or
            real["roc_auc"] is None or shuffled["roc_auc"] is None
            else float(real["roc_auc"] - shuffled["roc_auc"])
        )
        decision["real_correspondence_beats_shuffled"] = (
            decision["real_minus_shuffled_auc"] is not None and
            decision["real_minus_shuffled_auc"] > 0.0
        )
        decision["real_minus_no_v_auc"] = (
            None if real is None or no_v is None or
            real["roc_auc"] is None or no_v["roc_auc"] is None
            else float(real["roc_auc"] - no_v["roc_auc"])
        )
        decision["legacy_minus_no_v_auc"] = (
            None if legacy is None or no_v is None or
            legacy["roc_auc"] is None or no_v["roc_auc"] is None
            else float(legacy["roc_auc"] - no_v["roc_auc"])
        )
        # This is deliberately labelled a validation screen rather than the
        # final paired-OOF gate required by the design document.
        decision["validation_screen_passes"] = bool(
            decision["real_correspondence_beats_shuffled"] and
            decision["real_minus_no_v_auc"] is not None and
            decision["real_minus_no_v_auc"] > 0.0
        )
        decision["paired_oof_gate_evaluated"] = False
    if args.stage == "stage3":
        with_g = next((row for row in rows if row["g"]), None)
        without_g = next((row for row in rows if not row["g"]), None)
        decision["g_auc_delta"] = (
            None if with_g is None or without_g is None or
            with_g["roc_auc"] is None or without_g["roc_auc"] is None
            else float(with_g["roc_auc"] - without_g["roc_auc"])
        )
    payload = {
        "schema_version": 1,
        "artifact_type": "multiview_stage_validation_summary",
        "stage": args.stage,
        "test_used": False,
        "conditions": rows,
        "decision": decision,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# 多视图关键子图 {} 验证汇总".format(args.stage.upper()),
        "",
        "- 仅使用 validation：是",
        "- test 使用：否",
        "",
        "| 条件 | S模式 | V模式 | G | AUROC | BA | Accuracy | Q MAE | ΔQ MAE | 表示有效秩 |",
        "|---|---|---|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {condition} | {static_mode} | {v} | {g} | {auc} | {ba} | {acc} | {q} | {dq} | {rank} |".format(
                condition=row["condition"], static_mode=row["static_mode"],
                v=row["v"], g="是" if row["g"] else "否",
                auc=_number(row["roc_auc"]), ba=_number(row["balanced_accuracy"]),
                acc=_number(row["accuracy"]), q=_number(row["q_standardized_mae"]),
                dq=_number(row["delta_q_standardized_mae"]),
                rank=_number(row["representation_effective_rank"]),
            )
        )
    lines.extend(("", "## 冻结决策", "", "```json", json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True), "```", ""))
    (args.output_dir / "summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
