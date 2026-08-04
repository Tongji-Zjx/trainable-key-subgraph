"""Audit Stage 0-4 multi-view artifacts against the frozen experiment gates."""

from __future__ import absolute_import, division, print_function

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch


def _read(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _trusted_load(path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def _finite(value):
    return value is not None and math.isfinite(float(value))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _check_file(checks, name, path):
    exists = Path(path).is_file()
    checks.append({"name": name, "passed": exists, "evidence": str(path)})
    return exists


def _append(checks, name, passed, evidence):
    checks.append(
        {"name": name, "passed": bool(passed), "evidence": evidence}
    )


def _manifest_inventory(root, checks):
    manifests = {}
    for split in ("train", "validation", "test"):
        path = root / "full" / split / "manifest.json"
        if not _check_file(checks, "stage0_manifest_{}".format(split), path):
            continue
        payload = _read(path)
        manifests[split] = (path, payload)
        labels = {int(row["label"]) for row in payload.get("records", ())}
        _append(
            checks,
            "stage0_manifest_{}_valid".format(split),
            payload.get("artifact_type")
            == "theory_guided_multiview_critical_manifest"
            and payload.get("split") == split
            and len(payload.get("records", ())) > 0
            and labels == {0, 1},
            {
                "sample_count": len(payload.get("records", ())),
                "labels": sorted(labels),
            },
        )
    if len(manifests) != 3:
        return manifests
    provenance = {
        (
            payload.get("protocol_sha256"),
            payload.get("selector_checkpoint_sha256"),
            payload.get("feature_schema_sha256"),
        )
        for _, payload in manifests.values()
    }
    key_sets = {
        split: {row["sample_key"] for row in payload[1].get("records", ())}
        for split, payload in manifests.items()
    }
    _append(
        checks,
        "stage0_provenance_consistent",
        len(provenance) == 1 and None not in next(iter(provenance)),
        [list(item) for item in provenance],
    )
    _append(
        checks,
        "stage0_splits_disjoint",
        not (key_sets["train"] & key_sets["validation"])
        and not (key_sets["train"] & key_sets["test"])
        and not (key_sets["validation"] & key_sets["test"]),
        {name: len(values) for name, values in key_sets.items()},
    )
    return manifests


def _stage0(root, manifests, checks, smoke_dir, overfit_dir):
    for split in ("train", "validation", "test"):
        path = root / "full" / split / "audit.json"
        if _check_file(checks, "stage0_audit_{}".format(split), path):
            audit = _read(path)
            _append(
                checks,
                "stage0_audit_{}_valid".format(split),
                audit.get("artifact_type") == "multiview_critical_stage0_audit"
                and audit.get("split") == split
                and int(audit.get("sample_count", 0)) > 0
                and _finite(audit.get("fgw_convergence_fraction"))
                and _finite(audit.get("singleton_object_fraction")),
                {
                    "sample_count": audit.get("sample_count"),
                    "fgw_convergence_fraction": audit.get(
                        "fgw_convergence_fraction"
                    ),
                },
            )
    scaler_path = root / "scaler.pt"
    if _check_file(checks, "stage0_train_only_scaler", scaler_path):
        payload = _trusted_load(scaler_path)
        scaler = payload.get("scaler")
        provenance = None
        if len(manifests) == 3:
            provenance = next(
                iter(
                    {
                        (
                            item[1].get("protocol_sha256"),
                            item[1].get("selector_checkpoint_sha256"),
                            item[1].get("feature_schema_sha256"),
                        )
                        for item in manifests.values()
                    }
                )
            )
        _append(
            checks,
            "stage0_scaler_provenance",
            payload.get("artifact_type") == "multiview_train_scaler"
            and scaler is not None
            and provenance is not None
            and (
                getattr(scaler, "protocol_sha256", None),
                getattr(scaler, "selector_checkpoint_sha256", None),
                getattr(scaler, "feature_schema_sha256", None),
            )
            == provenance
            and getattr(scaler, "train_manifest_sha256", None)
            == _sha256(manifests["train"][0]),
            "scaler provenance matches all three manifests and train manifest hash",
        )
    if smoke_dir is not None:
        _check_file(checks, "engineering_cuda_smoke", smoke_dir / "best_checkpoint.pt")
    if overfit_dir is not None:
        history_path = overfit_dir / "history.json"
        if _check_file(checks, "engineering_overfit_history", history_path):
            history = _read(history_path)
            first = history[0]["train"] if history else {}
            last = history[-1]["train"] if history else {}
            _append(
                checks,
                "engineering_overfit_losses_decrease",
                len(history) >= 2
                and float(last["classification_loss"])
                < float(first["classification_loss"])
                and float(last["q_loss"]) < float(first["q_loss"])
                and float(last["delta_q_loss"]) < float(first["delta_q_loss"]),
                {"first": first, "last": last},
            )


def _condition_set(root, stage, required, checks):
    summary_path = root / stage / "summary" / "summary.json"
    if not _check_file(checks, "{}_summary".format(stage), summary_path):
        return None
    summary = _read(summary_path)
    conditions = summary.get("conditions", ())
    names = {row.get("condition") for row in conditions}
    modes = {row.get("v") for row in conditions}
    if stage == "stage1":
        present = set(required).issubset(names)
    elif stage == "stage3":
        present = len(conditions) >= 2 and {False, True}.issubset(
            {bool(row.get("g")) for row in conditions}
        )
    else:
        present = set(required).issubset(modes)
    _append(
        checks,
        "{}_conditions_complete".format(stage),
        summary.get("test_used") is False and present,
        {"conditions": sorted(name for name in names if name), "v_modes": sorted(modes)},
    )
    return summary


def _stages(root, checks):
    stage1 = _condition_set(
        root, "stage1", ("S_stable", "S_neural", "S_residual"), checks
    )
    if stage1 is not None:
        _append(
            checks,
            "stage1_selection_frozen",
            stage1.get("decision", {}).get("best_validation_condition") is not None,
            stage1.get("decision", {}),
        )

    stage2 = _condition_set(
        root, "stage2", ("none", "legacy", "uot", "shuffled"), checks
    )
    stage2_selection_path = root / "stage2" / "frozen_selection.json"
    if _check_file(checks, "stage2_frozen_selection", stage2_selection_path):
        selection = _read(stage2_selection_path)
        decision = {} if stage2 is None else stage2.get("decision", {})
        safe_rejection = (
            decision.get("validation_screen_passes") is False
            and selection.get("v_mode") in ("none", "legacy")
        )
        paired_gate = decision.get("paired_oof_gate_evaluated") is True
        _append(
            checks,
            "stage2_gate_respected",
            selection.get("test_used") is False
            and selection.get("v_mode") != "shuffled"
            and (paired_gate or safe_rejection),
            {"selection": selection, "decision": decision},
        )

    stage3 = _condition_set(root, "stage3", (), checks)
    stage3_selection_path = root / "stage3" / "frozen_selection.json"
    masking = list((root / "stage3").glob("*/validation_channel_masking.json"))
    _append(
        checks,
        "stage3_channel_masking_present",
        len(masking) == 1,
        [str(path) for path in masking],
    )
    if _check_file(checks, "stage3_frozen_selection", stage3_selection_path):
        selection = _read(stage3_selection_path)
        _append(
            checks,
            "stage3_selection_validation_only",
            selection.get("test_used") is False
            and selection.get("source")
            == "validation_only_stage3_gate_and_masking",
            selection,
        )
    if stage3 is not None:
        _append(
            checks,
            "stage3_g_comparison_present",
            stage3.get("decision", {}).get("g_auc_delta") is not None,
            stage3.get("decision", {}),
        )

    stage4_summary = root / "stage4" / "summary.json"
    complete = root / "stage4" / "COMPLETE"
    if _check_file(checks, "stage4_summary", stage4_summary):
        summary = _read(stage4_summary)
        _append(
            checks,
            "stage4_test_leakage_gate",
            summary.get("test_used_for_selection") is False
            and summary.get("test_threshold_refit") is False
            and set(summary.get("models", {}))
            == {"author", "critical", "fusion"},
            {
                "test_used_for_selection": summary.get("test_used_for_selection"),
                "test_threshold_refit": summary.get("test_threshold_refit"),
            },
        )
    _check_file(checks, "stage4_complete_marker", complete)
    if stage2_selection_path.is_file() and stage3_selection_path.is_file() and stage4_summary.is_file():
        earliest_test = stage4_summary.stat().st_mtime
        _append(
            checks,
            "freeze_precedes_test_summary",
            stage2_selection_path.stat().st_mtime <= earliest_test
            and stage3_selection_path.stat().st_mtime <= earliest_test,
            {
                "stage2_selection_mtime": stage2_selection_path.stat().st_mtime,
                "stage3_selection_mtime": stage3_selection_path.stat().st_mtime,
                "stage4_summary_mtime": earliest_test,
            },
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--smoke-dir", type=Path)
    parser.add_argument("--overfit-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    checks = []
    manifests = _manifest_inventory(root, checks)
    _stage0(root, manifests, checks, args.smoke_dir, args.overfit_dir)
    _stages(root, checks)
    failed = [row for row in checks if not row["passed"]]
    payload = {
        "schema_version": 1,
        "artifact_type": "multiview_stage0_to_stage4_completion_audit",
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
        "# 多视图关键子图 Stage 0-4 完成性审计",
        "",
        "- 检查数：{}".format(len(checks)),
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
