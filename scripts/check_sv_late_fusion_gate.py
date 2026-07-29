"""Apply the frozen validation-only gate for SV late-fusion experiments."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-evaluation", type=Path, required=True
    )
    parser.add_argument(
        "--improved-evaluation", type=Path, required=True
    )
    parser.add_argument(
        "--improved-diagnostic", type=Path, required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--minimum-composite-gain", type=float, default=0.01
    )
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
    return parser.parse_args()


def _read_json(path):
    with Path(path).resolve().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def _primary_metrics(evaluation):
    metrics = evaluation.get("metrics", {}).get("balanced_accuracy")
    if not isinstance(metrics, dict):
        raise ValueError(
            "evaluation has no balanced_accuracy validation metrics"
        )
    for key in ("roc_auc", "composite_auc"):
        if metrics.get(key) is None:
            raise ValueError("evaluation metric is missing: {}".format(key))
    return metrics


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    json_path = output_dir / "gate.json"
    markdown_path = output_dir / "summary.md"
    if json_path.exists() or markdown_path.exists():
        raise FileExistsError("SV late-fusion gate output exists")

    baseline = _read_json(args.baseline_evaluation)
    improved = _read_json(args.improved_evaluation)
    diagnostic = _read_json(args.improved_diagnostic)
    baseline_metrics = _primary_metrics(baseline)
    improved_metrics = _primary_metrics(improved)
    branch_metrics = improved.get("branch_metrics")
    if not isinstance(branch_metrics, dict) or not branch_metrics:
        raise ValueError("improved evaluation has no branch metrics")
    branch_auc = {
        str(name): float(values["roc_auc"])
        for name, values in branch_metrics.items()
        if values.get("roc_auc") is not None
    }
    if not branch_auc:
        raise ValueError("improved evaluation has no finite branch AUROC")

    validation_representations = diagnostic.get("splits", {}).get(
        "validation", {}
    ).get("representations", {})
    gin_representation = validation_representations.get(
        "gin_representation"
    )
    gin_projection = validation_representations.get("gin_projection")
    if not isinstance(gin_representation, dict) or not isinstance(
        gin_projection, dict
    ):
        raise ValueError("improved diagnostic lacks GIN representations")
    normalized_rank = float(
        gin_representation["normalized_effective_rank"]
    )
    projection_cosine_value = gin_projection.get(
        "mean_pairwise_cosine"
    )
    projection_cosine = (
        None
        if projection_cosine_value is None
        else float(projection_cosine_value)
    )

    baseline_composite = float(baseline_metrics["composite_auc"])
    improved_composite = float(improved_metrics["composite_auc"])
    improved_auc = float(improved_metrics["roc_auc"])
    best_branch_auc = max(branch_auc.values())
    checks = {
        "composite_auc_gain": (
            improved_composite - baseline_composite
            >= float(args.minimum_composite_gain)
        ),
        "fusion_not_worse_than_best_branch": (
            best_branch_auc - improved_auc
            <= float(args.maximum_fusion_regret)
        ),
        "gin_representation_not_low_rank": (
            normalized_rank >= float(args.minimum_gin_normalized_rank)
        ),
        "gin_projection_not_nearly_collinear": (
            projection_cosine is None
            or projection_cosine
            < float(args.maximum_gin_projection_cosine)
        ),
    }
    passed = all(checks.values())
    result = {
        "artifact_type": "sv_late_fusion_validation_gate",
        "test_used": False,
        "passed": passed,
        "checks": checks,
        "thresholds": {
            "minimum_composite_gain": args.minimum_composite_gain,
            "maximum_fusion_regret": args.maximum_fusion_regret,
            "minimum_gin_normalized_rank": (
                args.minimum_gin_normalized_rank
            ),
            "maximum_gin_projection_cosine": (
                args.maximum_gin_projection_cosine
            ),
        },
        "observed": {
            "baseline_composite_auc": baseline_composite,
            "improved_composite_auc": improved_composite,
            "composite_auc_gain": (
                improved_composite - baseline_composite
            ),
            "improved_roc_auc": improved_auc,
            "branch_roc_auc": branch_auc,
            "best_branch_roc_auc": best_branch_auc,
            "fusion_regret": best_branch_auc - improved_auc,
            "gin_normalized_effective_rank": normalized_rank,
            "gin_projection_mean_pairwise_cosine": projection_cosine,
        },
        "inputs": {
            "baseline_evaluation": str(
                Path(args.baseline_evaluation).resolve()
            ),
            "improved_evaluation": str(
                Path(args.improved_evaluation).resolve()
            ),
            "improved_diagnostic": str(
                Path(args.improved_diagnostic).resolve()
            ),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(json_path, result)
    lines = [
        "# SV SignedGIN 后期融合 Validation 闸门",
        "",
        "- Test 使用：否",
        "- 总体结论：{}".format("通过" if passed else "未通过"),
        "",
        "| 检查项 | 结果 |",
        "|---|---|",
    ]
    for name, value in checks.items():
        lines.append("| {} | {} |".format(name, "通过" if value else "未通过"))
    lines.extend(
        [
            "",
            "## 观测值",
            "",
            "```json",
            json.dumps(
                result["observed"],
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            "```",
        ]
    )
    markdown_path.write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "passed": passed,
                "gate": str(json_path),
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
