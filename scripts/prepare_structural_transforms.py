"""Fit and freeze A-E structural transforms from one training manifest only."""

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
    STRUCTURAL_GROUPS,
    fit_structural_transform,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prior-beta", type=float, default=1.0)
    parser.add_argument("--prior-permutation-seed", type=int, default=42)
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
        raise FileExistsError("structural transform output directory already exists")
    output_dir.mkdir(parents=True)
    dataset = BaselineHardSubgraphDataset(PROJECT_ROOT, args.train_manifest)
    if dataset.split != "train":
        raise ValueError("structural transforms require a train manifest")
    records = {}
    reference = None
    for group in STRUCTURAL_GROUPS:
        transform = fit_structural_transform(
            dataset,
            group,
            beta=args.prior_beta,
            permutation_seed=args.prior_permutation_seed,
        )
        path = output_dir / "group_{}.json".format(group)
        atomic_json(path, transform)
        if group != "A":
            identity = (
                transform["mean"], transform["std"],
                transform["normalized_importance"],
                transform["train_sample_key_sha256"],
            )
            if reference is None:
                reference = identity
            elif identity != reference:
                raise RuntimeError("A-E transforms do not share training statistics")
        records[group] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "prior_mode": transform["prior_mode"],
            "use_structural_features": transform["use_structural_features"],
        }
        print("prepared structural group {}".format(group), flush=True)
    manifest = {
        "schema_version": 1,
        "immutable": True,
        "purpose": "baseline_structural_prior_transforms",
        "fitted_on": "train_only",
        "train_manifest": str(args.train_manifest.resolve()),
        "train_manifest_sha256": file_sha256(args.train_manifest.resolve()),
        "prior_beta": float(args.prior_beta),
        "prior_permutation_seed": int(args.prior_permutation_seed),
        "groups": records,
    }
    manifest_path = output_dir / "structural_transforms_manifest.json"
    atomic_json(manifest_path, manifest)
    print(json.dumps({
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "sample_count": len(dataset),
        "groups": list(STRUCTURAL_GROUPS),
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
