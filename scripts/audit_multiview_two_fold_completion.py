"""Audit the frozen two-fold multi-view Stage-0 through Stage-4 workflow."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _add(checks, name, passed, evidence):
    checks.append({"name": name, "passed": bool(passed), "evidence": evidence})


def _file(checks, name, path):
    path = Path(path)
    passed = path.is_file()
    _add(checks, name, passed, str(path))
    return passed


def _manifest_keys(path):
    payload = _read(path)
    return payload, {row["sample_key"] for row in payload.get("records", ())}


def _stage0(root, checks):
    expected = {"train": 508, "validation": 117, "test": 313}
    manifests = {}
    keys = {}
    provenance = set()
    for split, count in expected.items():
        path = root / "full" / split / "manifest.json"
        if not _file(checks, "fold2_manifest_{}".format(split), path):
            continue
        payload, split_keys = _manifest_keys(path)
        manifests[split] = payload
        keys[split] = split_keys
        current = (
            payload.get("protocol_sha256"),
            payload.get("selector_checkpoint_sha256"),
            payload.get("feature_schema_sha256"),
        )
        provenance.add(current)
        labels = {int(row["label"]) for row in payload.get("records", ())}
        _add(
            checks,
            "fold2_manifest_{}_valid".format(split),
            payload.get("artifact_type")
            == "theory_guided_multiview_critical_manifest"
            and payload.get("split") == split
            and int(payload.get("sample_count", -1)) == count
            and len(split_keys) == count
            and labels == {0, 1},
            {"sample_count": len(split_keys), "labels": sorted(labels)},
        )
        audit = root / "full" / split / "audit.json"
        if _file(checks, "fold2_audit_{}".format(split), audit):
            report = _read(audit)
            _add(
                checks,
                "fold2_audit_{}_valid".format(split),
                report.get("artifact_type") == "multiview_critical_stage0_audit"
                and int(report.get("sample_count", -1)) == count,
                {"sample_count": report.get("sample_count")},
            )
    _add(
        checks,
        "fold2_stage0_provenance_consistent",
        len(provenance) == 1 and None not in next(iter(provenance), (None,)),
        [list(item) for item in provenance],
    )
    _add(
        checks,
        "fold2_splits_disjoint",
        set(keys) == set(expected)
        and not (keys["train"] & keys["validation"])
        and not (keys["train"] & keys["test"])
        and not (keys["validation"] & keys["test"]),
        {name: len(value) for name, value in keys.items()},
    )
    _file(checks, "fold2_train_only_scaler", root / "scaler.pt")
    _file(checks, "fold2_stage0_complete", root / "full" / "STAGE0_COMPLETE")
    return manifests, keys


def _selection(checks, path, stage, allowed, excluded):
    name = "two_fold_{}_selection".format(stage)
    if not _file(checks, name, path):
        return None
    payload = _read(path)
    decision = payload.get("decision", {})
    selected = decision.get("selected")
    _add(
        checks,
        name + "_valid",
        payload.get("artifact_type") == "multiview_two_fold_frozen_selection"
        and payload.get("stage") == stage
        and payload.get("test_used") is False
        and int(decision.get("fold_count", 0)) >= 2
        and selected in allowed
        and selected not in excluded
        and set(payload.get("official_candidates", ())) == set(allowed),
        {
            "selected": selected,
            "fold_count": decision.get("fold_count"),
            "official_candidates": payload.get("official_candidates"),
            "excluded_controls": payload.get("excluded_controls"),
        },
    )
    return payload


def _immutable_original(root, checks):
    path = root / "provenance" / "original_fold_immutable_snapshot.json"
    if not _file(checks, "original_fold_snapshot", path):
        return
    payload = _read(path)
    changed = []
    missing = []
    for relative, expected in payload.get("files", {}).items():
        current = PROJECT_ROOT / relative
        if not current.is_file():
            missing.append(relative)
        elif _sha256(current) != expected:
            changed.append(relative)
    _add(
        checks,
        "original_fold_results_immutable",
        not missing and not changed,
        {"file_count": payload.get("file_count"), "missing": missing, "changed": changed},
    )


def _stages(root, checks):
    s = _selection(
        checks, root / "two_fold" / "stage1" / "selection.json",
        "s", {"neural", "residual"}, {"stable"},
    )
    v = _selection(
        checks, root / "two_fold" / "stage2" / "selection.json",
        "v", {"none", "uot"}, {"legacy", "shuffled"},
    )
    g = _selection(
        checks, root / "two_fold" / "stage3" / "selection.json",
        "g", {"without_g", "with_g"}, set(),
    )
    for stage in ("STAGE1", "STAGE2", "STAGE3"):
        _file(
            checks, "two_fold_{}_complete".format(stage.lower()),
            root / "two_fold" / (stage + "_COMPLETE"),
        )
    architecture_path = root / "two_fold" / "frozen_architecture.json"
    if _file(checks, "two_fold_frozen_architecture", architecture_path):
        architecture = _read(architecture_path)
        expected = (
            None if s is None else s["decision"]["selected"],
            None if v is None else v["decision"]["selected"],
            None if g is None else g["decision"]["selected"],
        )
        actual = (
            architecture.get("static_mode"),
            architecture.get("v_mode"),
            architecture.get("g_mode"),
        )
        _add(
            checks,
            "two_fold_architecture_matches_selections",
            architecture.get("test_used") is False
            and actual == expected
            and actual[0] != "stable"
            and actual[1] != "legacy",
            {"expected": expected, "actual": actual},
        )


def _stage4(root, checks):
    summary_path = root / "stage4" / "summary.json"
    if _file(checks, "stage4_summary", summary_path):
        payload = _read(summary_path)
        models = payload.get("models", {})
        thresholds_frozen = True
        for row in models.values():
            if not isinstance(row, dict) or "threshold" not in row:
                thresholds_frozen = False
        _add(
            checks,
            "stage4_no_test_selection_or_refit",
            payload.get("artifact_type") == "multiview_stage4_frozen_summary"
            and payload.get("test_used_for_selection") is False
            and payload.get("test_threshold_refit") is False
            and set(models) == {"author", "critical", "fusion"}
            and thresholds_frozen,
            {
                "models": sorted(models),
                "test_used_for_selection": payload.get("test_used_for_selection"),
                "test_threshold_refit": payload.get("test_threshold_refit"),
            },
        )
    _file(checks, "stage4_complete", root / "stage4" / "COMPLETE")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    checks = []
    _stage0(root, checks)
    _immutable_original(root, checks)
    _stages(root, checks)
    _stage4(root, checks)
    failed = [row for row in checks if not row["passed"]]
    payload = {
        "schema_version": 1,
        "artifact_type": "multiview_two_fold_completion_audit",
        "root": str(root),
        "check_count": len(checks),
        "passed_count": len(checks) - len(failed),
        "failed_count": len(failed),
        "complete": not failed,
        "checks": checks,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "completion_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# 多视图关键子图两折 Stage 0–4 完成审计",
        "",
        "- 检查：{}".format(len(checks)),
        "- 通过：{}".format(len(checks) - len(failed)),
        "- 失败：{}".format(len(failed)),
        "- 完成：{}".format("是" if not failed else "否"),
        "",
        "| 检查 | 状态 |",
        "|---|:---:|",
    ]
    for row in checks:
        lines.append("| {} | {} |".format(row["name"], "通过" if row["passed"] else "失败"))
    (args.output_dir / "completion_audit.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if args.require_complete and failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
