"""Apply the preregistered cross-dataset Stage-1 upgrade gate."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import os
from pathlib import Path


def _write_json(path, payload):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adhd-summary", type=Path, required=True)
    parser.add_argument("--wmrc-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    with args.adhd_summary.open("r", encoding="utf-8") as handle:
        adhd = json.load(handle)
    with args.wmrc_summary.open("r", encoding="utf-8") as handle:
        wmrc = json.load(handle)
    primary = "N4_ema_center"
    candidates = {}
    for variant in adhd["comparisons_to_n0"]:
        left = adhd["comparisons_to_n0"][variant]
        right = wmrc["comparisons_to_n0"][variant]
        checks = {
            "auc_direction_consistent_positive": (
                left["pooled_auc_delta"] > 0.0 and right["pooled_auc_delta"] > 0.0
            ),
            "pooled_auc_not_clear_drop": (
                left["pooled_auc_delta"] >= -0.01
                and right["pooled_auc_delta"] >= -0.01
            ),
            "site_auc_not_clear_drop": (
                left["site_stratified_auc_delta"] >= -0.01
                and right["site_stratified_auc_delta"] >= -0.01
            ),
            "not_single_fold_only": (
                left["within_dataset_checks"]["not_single_fold_only"]
                and right["within_dataset_checks"]["not_single_fold_only"]
            ),
            "rank_fisher_consistent": (
                left["within_dataset_checks"]["rank_and_fisher_consistent"]
                and right["within_dataset_checks"]["rank_and_fisher_consistent"]
            ),
            "site_probe_not_obviously_stronger": (
                left["within_dataset_checks"]["site_probe_not_obviously_stronger"]
                and right["within_dataset_checks"]["site_probe_not_obviously_stronger"]
            ),
        }
        candidates[variant] = {
            "adhd": left, "wmrc": right, "checks": checks,
            "passes_all_checks": all(checks.values()),
        }
    primary_passes = bool(candidates[primary]["passes_all_checks"])
    payload = {
        "artifact_type": "theory_guided_neural_stage1_cross_dataset_gate",
        "primary_candidate": primary,
        "primary_was_preregistered": True,
        "outer_test_used_for_candidate_selection": False,
        "candidates": candidates,
        "stage2_allowed": primary_passes,
        "decision": "proceed_to_stage2" if primary_passes else "stop_after_stage1",
        "reason": (
            "preregistered N4 passed every cross-dataset gate"
            if primary_passes else
            "preregistered N4 did not pass every cross-dataset gate; do not cherry-pick N1-N3"
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "stage1_gate.json", payload)
    lines = [
        "# Stage 1 跨数据集升级闸门", "",
        "- 预注册主候选：`{}`".format(primary),
        "- 是否允许进入 Stage 2：{}".format("是" if primary_passes else "否"),
        "- 决策：`{}`".format(payload["decision"]), "",
        "| 候选 | ADHD ΔAUC | WMRC ΔAUC | 通过全部闸门 |", "|---|---:|---:|---|",
    ]
    for variant, value in candidates.items():
        lines.append("| {} | {:+.6f} | {:+.6f} | {} |".format(
            variant, value["adhd"]["pooled_auc_delta"],
            value["wmrc"]["pooled_auc_delta"],
            "是" if value["passes_all_checks"] else "否"
        ))
    lines.extend(["", "> N1–N3 仅作为机制消融，不依据 outer-test 结果替换预注册 N4。", ""])
    (args.output_dir / "stage1_gate.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
