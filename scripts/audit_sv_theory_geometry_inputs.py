"""Audit SV theory sidecars, provenance, masks and train-only scaling."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.sv_theory_geometry import (  # noqa: E402
    SVTheoryAugmentedDataset,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--base-scaler", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--test-cache", type=Path, required=True)
    parser.add_argument("--theory-scaler", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _summary(dataset):
    direction = torch.stack(
        [sample.spectral_direction for sample in dataset.samples], dim=0
    )
    diffusion = torch.stack(
        [sample.diffusion_geometry for sample in dataset.samples], dim=0
    )
    return {
        "sample_count": len(dataset),
        "class_counts": {
            str(label): sum(
                int(value == label) for value in dataset.labels
            )
            for label in (0, 1)
        },
        "spectral_direction_shape": list(direction.shape),
        "diffusion_geometry_shape": list(diffusion.shape),
        "all_finite": bool(
            torch.isfinite(direction).all()
            and torch.isfinite(diffusion).all()
        ),
        "spectral_direction_mean_abs": float(
            direction.abs().mean()
        ),
        "diffusion_geometry_mean_abs": float(
            diffusion.abs().mean()
        ),
        "spectral_direction_feature_mean_max_abs": float(
            direction.mean(dim=0).abs().max()
        ),
        "diffusion_geometry_feature_mean_max_abs": float(
            diffusion.mean(dim=0).abs().max()
        ),
    }


def main():
    args = parse_args()
    output = Path(args.output).resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError("SV theory audit output already exists")
    inputs = (
        (
            "train",
            args.train_manifest,
            args.train_cache,
        ),
        (
            "validation",
            args.validation_manifest,
            args.validation_cache,
        ),
        ("test", args.test_manifest, args.test_cache),
    )
    datasets = {}
    summaries = {}
    for split, manifest, cache in inputs:
        dataset = SVTheoryAugmentedDataset(
            manifest,
            args.base_scaler,
            cache,
            args.theory_scaler,
            include_windows=False,
        )
        if dataset.split != split:
            raise ValueError("SV theory audit split mismatch")
        datasets[split] = dataset
        summaries[split] = _summary(dataset)
    key_sets = {
        split: set(dataset.sample_keys)
        for split, dataset in datasets.items()
    }
    overlap = {
        "train_validation": sorted(
            key_sets["train"] & key_sets["validation"]
        ),
        "train_test": sorted(
            key_sets["train"] & key_sets["test"]
        ),
        "validation_test": sorted(
            key_sets["validation"] & key_sets["test"]
        ),
    }
    if any(overlap.values()):
        raise ValueError("SV theory audit found split overlap")
    train_centering_passed = (
        summaries["train"][
            "spectral_direction_feature_mean_max_abs"
        ]
        <= 1.0e-4
        and summaries["train"][
            "diffusion_geometry_feature_mean_max_abs"
        ]
        <= 1.0e-4
    )
    if not train_centering_passed:
        raise ValueError(
            "SV theory train-only standardization is not centered"
        )
    payload = {
        "artifact_type": "sv_theory_geometry_input_audit",
        "splits": summaries,
        "overlap": overlap,
        "train_centering_passed": train_centering_passed,
        "passed": all(
            value["all_finite"] for value in summaries.values()
        )
        and train_centering_passed,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    os.replace(str(temporary), str(output))
    print(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
