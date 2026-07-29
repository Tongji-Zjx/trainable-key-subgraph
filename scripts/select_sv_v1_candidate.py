"""Freeze one SV V1 candidate using validation-only evidence."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import os
from pathlib import Path


EXPECTED_VARIANTS = {
    "v1a": "signed_gin_static_anchor_residual",
    "v1b": "signed_gin_static_anchor_residual_attention",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1a-evaluation", type=Path, required=True)
    parser.add_argument("--v1a-diagnostic", type=Path, required=True)
    parser.add_argument("--v1b-evaluation", type=Path, required=True)
    parser.add_argument("--v1b-diagnostic", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--maximum-fusion-regret", type=float, default=0.01
    )
    parser.add_argument(
        "--minimum-gin-normalized-rank", type=float, default=0.10
    )
    parser.add_argument(
        "--maximum-gin-projection-cosine",
        type=float,
        default=0.995,
    )
    parser.add_argument(
        "--maximum-attention-entropy", type=float, default=0.99
    )
    parser.add_argument(
        "--minimum-attention-ablation-gain",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--minimum-v1b-gin-auc-gain", type=float, default=0.0
    )
    return parser.parse_args()


def _read(path):
    with Path(path).resolve().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def _metrics(evaluation):
    metrics = evaluation.get("metrics", {}).get("balanced_accuracy")
    if not isinstance(metrics, dict):
        raise ValueError("candidate evaluation lacks validation metrics")
    for name in ("roc_auc", "composite_auc"):
        if metrics.get(name) is None:
            raise ValueError("candidate metric is missing: {}".format(name))
    return metrics


def _candidate(name, evaluation, diagnostic, args):
    if diagnostic.get("variant") != EXPECTED_VARIANTS[name]:
        raise ValueError("{} diagnostic variant is invalid".format(name))
    validation = diagnostic.get("splits", {}).get("validation", {})
    representations = validation.get("representations", {})
    gin = representations.get("gin_representation")
    projection = representations.get("gin_projection")
    if not isinstance(gin, dict) or not isinstance(projection, dict):
        raise ValueError("{} diagnostic lacks GIN statistics".format(name))
    metrics = _metrics(evaluation)
    branch_metrics = evaluation.get("branch_metrics", {})
    if "gin" not in branch_metrics or "static_spectral" not in branch_metrics:
        raise ValueError("{} evaluation lacks branch metrics".format(name))
    regret = evaluation.get("fusion_regret", {}).get("roc_auc")
    if regret is None:
        raise ValueError("{} evaluation lacks fusion regret".format(name))
    rank = float(gin["normalized_effective_rank"])
    cosine_value = projection.get("mean_pairwise_cosine")
    cosine = None if cosine_value is None else float(cosine_value)
    core_checks = {
        "fusion_regret_within_limit": (
            float(regret) <= float(args.maximum_fusion_regret)
        ),
        "gin_representation_not_low_rank": (
            rank >= float(args.minimum_gin_normalized_rank)
        ),
        "gin_projection_not_nearly_collinear": (
            cosine is None
            or cosine < float(args.maximum_gin_projection_cosine)
        ),
        "diagnostic_checks_pass": all(
            bool(value)
            for value in diagnostic.get("checks", {}).values()
        ),
    }
    return {
        "variant": EXPECTED_VARIANTS[name],
        "core_passed": all(core_checks.values()),
        "core_checks": core_checks,
        "observed": {
            "roc_auc": float(metrics["roc_auc"]),
            "composite_auc": float(metrics["composite_auc"]),
            "gin_branch_roc_auc": float(
                branch_metrics["gin"]["roc_auc"]
            ),
            "static_branch_roc_auc": float(
                branch_metrics["static_spectral"]["roc_auc"]
            ),
            "fusion_regret": float(regret),
            "gin_normalized_effective_rank": rank,
            "gin_projection_mean_pairwise_cosine": cosine,
        },
    }


def select_candidate(v1a_evaluation, v1a_diagnostic,
                     v1b_evaluation, v1b_diagnostic, args):
    provenance_a = v1a_diagnostic.get("provenance")
    provenance_b = v1b_diagnostic.get("provenance")
    if provenance_a != provenance_b:
        raise ValueError("V1 candidates do not share frozen provenance")
    v1a = _candidate(
        "v1a", v1a_evaluation, v1a_diagnostic, args
    )
    v1b = _candidate(
        "v1b", v1b_evaluation, v1b_diagnostic, args
    )
    validation_b = v1b_diagnostic["splits"]["validation"]
    attention = validation_b.get("attention")
    masked = validation_b.get("attention_ablation_metrics")
    if not isinstance(attention, dict) or not isinstance(masked, dict):
        raise ValueError("V1B diagnostic lacks attention ablation")
    entropy = float(attention["normalized_entropy"]["median"])
    masked_auc = float(masked["roc_auc"])
    final_auc = float(v1b["observed"]["roc_auc"])
    retention_checks = {
        "v1b_core_passed": bool(v1b["core_passed"]),
        "gin_branch_auc_improved": (
            v1b["observed"]["gin_branch_roc_auc"]
            - v1a["observed"]["gin_branch_roc_auc"]
            > float(args.minimum_v1b_gin_auc_gain)
        ),
        "final_not_below_v1a": (
            v1b["observed"]["composite_auc"]
            >= v1a["observed"]["composite_auc"]
        ),
        "gin_rank_not_below_v1a": (
            v1b["observed"]["gin_normalized_effective_rank"]
            >= v1a["observed"]["gin_normalized_effective_rank"]
        ),
        "attention_not_nearly_uniform": (
            entropy < float(args.maximum_attention_entropy)
        ),
        "attention_ablation_has_positive_gain": (
            final_auc - masked_auc
            > float(args.minimum_attention_ablation_gain)
        ),
    }
    retain_v1b = all(retention_checks.values())
    if retain_v1b:
        selected = "v1b"
        reason = "V1B passed every validation-only retention check."
    elif v1a["core_passed"]:
        selected = "v1a"
        reason = (
            "V1B added no validated attention benefit; freeze the "
            "simpler safe-residual candidate."
        )
    else:
        selected = None
        reason = "Neither V1 candidate passed the validation-only gate."
    return {
        "artifact_type": "sv_v1_validation_candidate_selection",
        "test_used": False,
        "parameter_updates": 0,
        "selected_candidate": selected,
        "selected_variant": (
            None if selected is None else EXPECTED_VARIANTS[selected]
        ),
        "reason": reason,
        "v1a": v1a,
        "v1b": v1b,
        "v1b_retention_checks": retention_checks,
        "v1b_attention": {
            "normalized_entropy_median": entropy,
            "final_roc_auc": final_auc,
            "attention_masked_roc_auc": masked_auc,
            "ablation_gain": final_auc - masked_auc,
        },
        "thresholds": {
            "maximum_fusion_regret": args.maximum_fusion_regret,
            "minimum_gin_normalized_rank": (
                args.minimum_gin_normalized_rank
            ),
            "maximum_gin_projection_cosine": (
                args.maximum_gin_projection_cosine
            ),
            "maximum_attention_entropy": (
                args.maximum_attention_entropy
            ),
            "minimum_attention_ablation_gain": (
                args.minimum_attention_ablation_gain
            ),
            "minimum_v1b_gin_auc_gain": (
                args.minimum_v1b_gin_auc_gain
            ),
        },
        "provenance": provenance_a,
    }


def _markdown(result):
    selected = result["selected_candidate"] or "none"
    lines = [
        "# SV-HardSGW V1 validation-only 候选冻结",
        "",
        "- Test 使用：否",
        "- 参数更新量：0",
        "- 冻结候选：`{}`".format(selected),
        "- 冻结变体：`{}`".format(result["selected_variant"]),
        "- 原因：{}".format(result["reason"]),
        "",
        "| 候选 | Final AUC | Composite AUC | GIN AUC | "
        "GIN归一化有效秩 | Fusion regret |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("v1a", "v1b"):
        observed = result[name]["observed"]
        lines.append(
            "| {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | "
            "{:+.6f} |".format(
                name,
                observed["roc_auc"],
                observed["composite_auc"],
                observed["gin_branch_roc_auc"],
                observed["gin_normalized_effective_rank"],
                observed["fusion_regret"],
            )
        )
    lines.extend(
        [
            "",
            "## V1B 保留条件",
            "",
        ]
    )
    for name, value in result["v1b_retention_checks"].items():
        lines.append(
            "- {}：{}".format(name, "通过" if value else "未通过")
        )
    lines.extend(
        [
            "",
            "> 此冻结决定只使用 train/validation 产物；唯一候选冻结后，"
            "方可运行 outer-fold OOF。",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    json_path = output_dir / "candidate_selection.json"
    markdown_path = output_dir / "summary.md"
    if json_path.exists() or markdown_path.exists():
        raise FileExistsError("SV V1 candidate selection output exists")
    result = select_candidate(
        _read(args.v1a_evaluation),
        _read(args.v1a_diagnostic),
        _read(args.v1b_evaluation),
        _read(args.v1b_diagnostic),
        args,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(json_path, result)
    markdown_path.write_text(_markdown(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "selected_candidate": result["selected_candidate"],
                "selected_variant": result["selected_variant"],
                "selection": str(json_path),
                "summary": str(markdown_path),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
