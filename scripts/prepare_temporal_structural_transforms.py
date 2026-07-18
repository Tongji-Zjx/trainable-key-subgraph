"""Fit and freeze matched A/B/F/G/H temporal structural transforms."""

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

from keysubgraph.data.baseline_dataset import BaselineHardSubgraphDataset  # noqa: E402
from keysubgraph.data.data_split import file_sha256  # noqa: E402
from keysubgraph.features.structural_prior import (  # noqa: E402
    TEMPORAL_STRUCTURAL_GROUPS,
    fit_temporal_structural_transform,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--permutation-seed", type=int, default=42)
    return parser.parse_args()


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError("temporal transform output directory already exists")
    output_dir.mkdir(parents=True)
    dataset = BaselineHardSubgraphDataset(PROJECT_ROOT, args.train_manifest)
    if dataset.split != "train":
        raise ValueError("temporal transforms require a train manifest")
    records = {}
    reference = None
    for group in TEMPORAL_STRUCTURAL_GROUPS:
        transform = fit_temporal_structural_transform(
            dataset, group, permutation_seed=args.permutation_seed
        )
        identity = (
            transform["mean"], transform["std"],
            transform["valid_window_counts"], transform["train_sample_key_sha256"],
        )
        if reference is None:
            reference = identity
        elif identity != reference:
            raise RuntimeError("A/B/F/G/H transforms do not share normalization")
        path = output_dir / "group_{}.json".format(group)
        atomic_json(path, transform)
        records[group] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "use_structural_features": transform["use_structural_features"],
            "use_structural_deltas": transform["use_structural_deltas"],
            "structural_delta_order": transform["structural_delta_order"],
        }
        print("prepared temporal structural group {}".format(group), flush=True)
    manifest = {
        "schema_version": 2,
        "immutable": True,
        "purpose": "baseline_temporal_structural_delta_transforms",
        "fitted_on": "train_only",
        "normalization_delta_order": "ordered",
        "train_manifest": str(args.train_manifest.resolve()),
        "train_manifest_sha256": file_sha256(args.train_manifest.resolve()),
        "permutation_seed": int(args.permutation_seed),
        "groups": records,
    }
    manifest_path = output_dir / "temporal_structural_transforms_manifest.json"
    atomic_json(manifest_path, manifest)
    print(json.dumps({
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "sample_count": len(dataset),
        "groups": list(TEMPORAL_STRUCTURAL_GROUPS),
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
