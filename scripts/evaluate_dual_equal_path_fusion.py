"""Fuse frozen D3 ProxyInput and ExactInput paths with equal logits."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.analysis.dual_frozen_logit_ensemble import (  # noqa: E402
    build_frozen_equal_logit_ensemble,
    write_frozen_equal_logit_ensemble_artifacts,
)
from keysubgraph.data.data_split import file_sha256  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation-alignment-csv", type=Path, required=True
    )
    parser.add_argument("--test-alignment-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _read_alignment(path):
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "sample_key",
        "label",
        "proxy_exact_probability",
        "exact_exact_probability",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError("alignment CSV lacks frozen D3 path columns")
    components = {
        "proxy_input_exact_head": {
            "sample_keys": [],
            "labels": [],
            "probabilities": [],
        },
        "exact_input_exact_head": {
            "sample_keys": [],
            "labels": [],
            "probabilities": [],
        },
    }
    for row in rows:
        for name, column in (
            ("proxy_input_exact_head", "proxy_exact_probability"),
            ("exact_input_exact_head", "exact_exact_probability"),
        ):
            components[name]["sample_keys"].append(row["sample_key"])
            components[name]["labels"].append(int(row["label"]))
            components[name]["probabilities"].append(float(row[column]))
    return components


def _read_summary(csv_path):
    path = Path(csv_path).resolve().parent / "summary.json"
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("artifact") != "dual_proxy_exact_alignment":
        raise ValueError("unexpected Proxy-Exact alignment summary")
    return path, payload


def _validate_provenance(validation, test):
    if (
        validation.get("provenance", {}).get("split") != "validation"
        or test.get("provenance", {}).get("split") != "test"
    ):
        raise ValueError("path fusion alignment splits are invalid")
    keys = (
        "protocol_sha256",
        "selector_checkpoint_sha256",
        "sgw_checkpoint_sha256",
        "scaler_sha256",
        "selection_mode",
        "selection_seed",
    )
    for key in keys:
        if (
            validation["provenance"].get(key)
            != test["provenance"].get(key)
        ):
            raise ValueError(
                "path fusion provenance disagrees on {}".format(key)
            )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("equal path-fusion output already exists")
    validation_summary_path, validation_summary = _read_summary(
        args.validation_alignment_csv
    )
    test_summary_path, test_summary = _read_summary(
        args.test_alignment_csv
    )
    _validate_provenance(validation_summary, test_summary)
    evaluation = build_frozen_equal_logit_ensemble(
        validation_components=_read_alignment(
            args.validation_alignment_csv
        ),
        test_components=_read_alignment(args.test_alignment_csv),
        ensemble_scope="within_seed_proxy_exact_plus_exact_exact",
    )
    provenance = {
        "read_only_frozen_predictions": True,
        "validation_alignment_csv": str(
            Path(args.validation_alignment_csv).resolve()
        ),
        "validation_alignment_csv_sha256": file_sha256(
            args.validation_alignment_csv
        ),
        "validation_summary": str(validation_summary_path),
        "validation_summary_sha256": file_sha256(
            validation_summary_path
        ),
        "test_alignment_csv": str(
            Path(args.test_alignment_csv).resolve()
        ),
        "test_alignment_csv_sha256": file_sha256(
            args.test_alignment_csv
        ),
        "test_summary": str(test_summary_path),
        "test_summary_sha256": file_sha256(test_summary_path),
        "source_provenance": validation_summary["provenance"],
    }
    paths = write_frozen_equal_logit_ensemble_artifacts(
        output_dir, evaluation, provenance
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "artifacts": {
                    name: str(path) for name, path in paths.items()
                },
                "thresholds": evaluation["thresholds"],
                "validation_metrics": evaluation["validation"]["metrics"],
                "test_metrics": evaluation["test"]["metrics"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
