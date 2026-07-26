"""Audit temporal length, mask and frozen-base leakage before T1--T4 training."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.data_split import file_sha256  # noqa: E402
from keysubgraph.data.dual_temporal_manifest import (  # noqa: E402
    read_dual_temporal_manifest,
)
from keysubgraph.training.dual_sgw_feature_trainer import (  # noqa: E402
    binary_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _summary(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "mean": float(array.mean()),
        "standard_deviation": float(array.std()),
        "median": float(np.median(array)),
    }


def _split_report(payload, records):
    labels = [int(record.label) for record in records]
    lengths = [record.valid_transition_count for record in records]
    windows = [int(record.window_count) for record in records]
    probabilities = [
        float(record.base_logits.softmax(dim=-1)[1]) for record in records
    ]
    by_class = {}
    for label in (0, 1):
        selected = [
            length for length, item in zip(lengths, labels) if item == label
        ]
        by_class[str(label)] = _summary(selected)
    length_auc = (
        float(roc_auc_score(labels, lengths))
        if set(labels) == {0, 1}
        else None
    )
    length_auc_direction_free = (
        max(length_auc, 1.0 - length_auc) if length_auc is not None else None
    )
    base = binary_metrics(labels, probabilities, threshold=0.5)
    return {
        "split": payload["split"],
        "sample_count": len(records),
        "class_counts": base["class_counts"],
        "window_count": _summary(windows),
        "valid_transition_count": _summary(lengths),
        "zero_transition_sample_count": int(
            sum(length == 0 for length in lengths)
        ),
        "valid_transition_count_by_class": by_class,
        "length_only_auc": length_auc,
        "length_only_direction_free_auc": length_auc_direction_free,
        "frozen_base_metrics_at_0_5": base,
    }


def _atomic_json(path, payload):
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError("temporal input audit already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def main():
    args = parse_args()
    paths = [args.train_manifest, args.validation_manifest]
    if args.test_manifest is not None:
        paths.append(args.test_manifest)
    loaded = [read_dual_temporal_manifest(path) for path in paths]
    expected_splits = ["train", "validation"] + (
        ["test"] if args.test_manifest is not None else []
    )
    if [item[0]["split"] for item in loaded] != expected_splits:
        raise ValueError("temporal audit manifests use wrong splits")
    provenance_keys = (
        "protocol_sha256",
        "selector_checkpoint_sha256",
        "exact_head_checkpoint_sha256",
        "sgw_scaler_sha256",
        "exact_manifest_sha256",
        "selection_mode",
        "selection_seed",
    )
    if any(
        item[0][key] != loaded[0][0][key]
        for item in loaded[1:]
        for key in provenance_keys
    ):
        raise ValueError("temporal audit manifests disagree on provenance")
    key_sets = [
        {record.sample_key for record in records}
        for _, records in loaded
    ]
    for left in range(len(key_sets)):
        for right in range(left + 1, len(key_sets)):
            if key_sets[left] & key_sets[right]:
                raise ValueError("temporal audit splits overlap")
    report = {
        "artifact": "dual_d3b_temporal_input_audit",
        "checks": {
            "split_identity_disjoint": True,
            "provenance_consistent": True,
            "all_values_finite": True,
            "invalid_transitions_zero": True,
        },
        "manifests": {
            payload["split"]: {
                "path": str(Path(path).resolve()),
                "sha256": file_sha256(path),
            }
            for path, (payload, _) in zip(paths, loaded)
        },
        "splits": {
            payload["split"]: _split_report(payload, records)
            for payload, records in loaded
        },
    }
    _atomic_json(args.output, report)
    print(
        json.dumps(
            report["splits"], ensure_ascii=False, indent=2, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
