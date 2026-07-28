"""Audit frozen SV Signed-GIN manifests and optional train-only scaler."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.sv_signed_gin_manifest import (  # noqa: E402
    read_sv_signed_gin_manifest,
)
from keysubgraph.data.sv_signed_gin_scaler import (  # noqa: E402
    load_sv_signed_gin_standardizers,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--test-manifest", type=Path)
    parser.add_argument("--scaler", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _summary(path):
    manifest, records = read_sv_signed_gin_manifest(path)
    node_counts = [
        int(window.node_features.shape[0])
        for record in records
        for window in record.windows
        if window is not None
    ]
    signed_counts = {
        "positive": sum(
            int((window.adjacency > 0.0).sum().item() // 2)
            for record in records
            for window in record.windows
            if window is not None
        ),
        "negative": sum(
            int((window.adjacency < 0.0).sum().item() // 2)
            for record in records
            for window in record.windows
            if window is not None
        ),
    }
    return manifest, records, {
        "sample_count": len(records),
        "class_counts": dict(
            sorted(Counter(record.label for record in records).items())
        ),
        "site_counts": dict(
            sorted(Counter(record.site for record in records).items())
        ),
        "valid_window_count": sum(
            record.valid_window_count for record in records
        ),
        "valid_transition_count": sum(
            record.valid_transition_count for record in records
        ),
        "nodes_per_window": {
            "minimum": min(node_counts),
            "maximum": max(node_counts),
            "mean": sum(node_counts) / float(len(node_counts)),
        },
        "signed_edge_counts": signed_counts,
        "all_finite": all(
            bool(torch.isfinite(record.static_features).all())
            and bool(torch.isfinite(record.variation).all())
            for record in records
        ),
    }


def _atomic_json(path, payload):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def main():
    args = parse_args()
    requested = (
        ("train", args.train_manifest),
        ("validation", args.validation_manifest),
        ("test", args.test_manifest),
    )
    summaries = {}
    manifests = {}
    records_by_split = {}
    all_keys = set()
    for expected_split, path in requested:
        if path is None:
            continue
        manifest, records, summary = _summary(path)
        if manifest["split"] != expected_split:
            raise ValueError("SV audit manifest split mismatch")
        keys = {record.sample_key for record in records}
        if all_keys.intersection(keys):
            raise ValueError("SV audit found sample overlap across splits")
        all_keys.update(keys)
        manifests[expected_split] = manifest
        records_by_split[expected_split] = records
        summaries[expected_split] = summary
    provenance = {
        (
            manifest["protocol_sha256"],
            manifest["selector_checkpoint_sha256"],
            manifest["selection_mode"],
            int(manifest["selection_seed"]),
        )
        for manifest in manifests.values()
    }
    if len(provenance) != 1:
        raise ValueError("SV audit manifests disagree on provenance")
    scaler_summary = None
    if args.scaler is not None:
        scaler = load_sv_signed_gin_standardizers(args.scaler)
        common = next(iter(provenance))
        if (
            scaler.protocol_sha256,
            scaler.selector_checkpoint_sha256,
            scaler.selection_mode,
            int(scaler.selection_seed),
        ) != common:
            raise ValueError("SV audit scaler provenance mismatch")
        scaler_summary = {
            "train_sample_count": scaler.train_sample_count,
            "train_node_count": scaler.train_node_count,
            "node_scale_minimum": float(scaler.node_scale.min()),
            "static_scale_minimum": float(scaler.static_scale.min()),
            "variation_scale_minimum": float(
                scaler.variation_scale.min()
            ),
        }
    result = {
        "artifact_type": "sv_hard_sgw_signed_gin_input_audit",
        "split_overlap_count": 0,
        "provenance_consistent": True,
        "summaries": summaries,
        "scaler": scaler_summary,
    }
    _atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "split_overlap_count": 0,
                "splits": {
                    key: value["sample_count"]
                    for key, value in summaries.items()
                },
                "signed_edges": {
                    key: value["signed_edge_counts"]
                    for key, value in summaries.items()
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
