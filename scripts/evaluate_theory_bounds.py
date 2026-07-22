"""Aggregate strict-theory hard exports into traceable spectral--GW bounds."""

from __future__ import absolute_import, print_function

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.data_protocol import validate_data_protocol  # noqa: E402
from keysubgraph.data.data_split import file_sha256  # noqa: E402
from keysubgraph.theory import TheoryBoundEvaluator  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "configs" / "data_protocol_strict_theory.json",
    )
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--classification-evaluation", type=Path)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "theory_evaluation" / "strict_theory_test.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _mean(values):
    return sum(values) / len(values) if values else None


def main():
    args = parse_args()
    protocol = validate_data_protocol(args.protocol, PROJECT_ROOT)
    if protocol.get("experiment_mode") == "all_samples_exploratory":
        raise ValueError("theory bounds require strict train/validation/test partitions")
    export_root = args.export_dir.resolve() / args.split
    files = sorted(export_root.glob("*.json"))
    if not files:
        raise ValueError("no hard-export JSON files found for split")
    records = []
    timepoint_rows = []
    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("split") != args.split:
            raise ValueError("export split mismatch: {}".format(path))
        if payload.get("data_protocol_sha256") != file_sha256(args.protocol):
            raise ValueError("export protocol hash mismatch: {}".format(path))
        full = payload.get("H_SGW_full")
        hard = payload.get("H_SGW_hard")
        if not isinstance(full, list) or not isinstance(hard, list) or len(full) != len(hard):
            raise ValueError("missing or incompatible H_SGW vectors: {}".format(path))
        records.append((payload["sample_key"], int(payload["label"]), full, hard))
        for timepoint in payload.get("timepoints", []):
            if not timepoint.get("hard_union_available"):
                continue
            required = (
                "full_to_soft_laplacian_fro_error",
                "full_to_soft_laplacian_operator_error",
                "full_to_soft_gw_error",
                "soft_to_hard_spectral_winf",
                "soft_to_hard_gw_error",
                "full_to_hard_spectral_winf",
                "full_to_hard_gw_error",
                "gw_solver_error_proxy",
                "gw_solver_converged",
            )
            if any(name not in timepoint for name in required):
                raise ValueError("incomplete fidelity row: {}".format(path))
            row = {name: timepoint[name] for name in required}
            row["label"] = int(payload["label"])
            timepoint_rows.append(row)
    if not timepoint_rows:
        raise ValueError("no valid hard-union timepoints were exported")
    labels = [row[1] for row in records]
    bound = TheoryBoundEvaluator(args.bootstrap_repeats, args.seed).evaluate(
        [row[2] for row in records], [row[3] for row in records], labels
    )
    class_conditional = {}
    for label in (0, 1):
        rows = [row for row in timepoint_rows if row["label"] == label]
        if not rows:
            raise ValueError("theory evaluation requires both classes")
        class_conditional[str(label)] = {
            "timepoint_count": len(rows),
            "q_lambda": _mean([row["soft_to_hard_spectral_winf"] for row in rows]),
            "q_gw": _mean([row["soft_to_hard_gw_error"] for row in rows]),
        }
    classification = None
    if args.classification_evaluation is not None:
        with args.classification_evaluation.resolve().open("r", encoding="utf-8") as handle:
            classification = json.load(handle)
    payload = {
        "schema_version": 1,
        "protocol_name": "strict_theory",
        "split": args.split,
        "sample_count": len(records),
        "timepoint_count": len(timepoint_rows),
        "class_counts": {str(label): labels.count(label) for label in (0, 1)},
        "data_protocol_sha256": file_sha256(args.protocol),
        "classification_evaluation": classification,
        "full_to_soft_laplacian_fro_risk": _mean(
            [row["full_to_soft_laplacian_fro_error"] for row in timepoint_rows]
        ),
        "full_to_soft_laplacian_operator_risk": _mean(
            [row["full_to_soft_laplacian_operator_error"] for row in timepoint_rows]
        ),
        "full_to_soft_gw_risk": _mean(
            [row["full_to_soft_gw_error"] for row in timepoint_rows]
        ),
        "full_to_hard_spectral_winf_risk": _mean(
            [row["full_to_hard_spectral_winf"] for row in timepoint_rows]
        ),
        "full_to_hard_gw_risk": _mean(
            [row["full_to_hard_gw_error"] for row in timepoint_rows]
        ),
        "gw_solver_error_proxy": _mean(
            [row["gw_solver_error_proxy"] for row in timepoint_rows]
        ),
        "gw_solver_nonconverged_count": sum(
            not bool(row["gw_solver_converged"]) for row in timepoint_rows
        ),
        "class_conditional_soft_to_hard": class_conditional,
        "theory_bound": bound,
        "interpretation": (
            "theory sufficient condition verified"
            if bound["lower_bound_positive"]
            else "theory sufficient condition not verified"
        ),
    }
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(output))
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
