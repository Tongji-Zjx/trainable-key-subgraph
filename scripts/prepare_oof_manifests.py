"""Freeze A-D baseline manifests from one fold-wide Key/Random control inventory."""

from __future__ import absolute_import, print_function

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.baseline_manifest import build_baseline_manifest  # noqa: E402


SPLIT_ROLE = {"train": "inner_train", "validation": "inner_validation", "test": "outer_test"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/crossfit/manifests")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    control_manifest = args.control_root / "key_random_control_manifest.json"
    outputs = []
    for source in ("key", "random"):
        for split, role in SPLIT_ROLE.items():
            output_dir = args.output_root / "fold_{}".format(args.fold) / "{}_{}".format(source, role)
            payload = build_baseline_manifest(
                PROJECT_ROOT, args.protocol, args.control_root / source, split,
                output_dir, checkpoint_path=args.checkpoint,
                evidence_level="confirmatory_cross_fitted",
                overwrite=args.overwrite,
                matched_control_manifest_path=control_manifest,
                subgraph_source=source,
            )
            outputs.append({
                "source": source, "split": split, "role": role,
                "manifest": str((output_dir / "baseline_manifest.json").resolve()),
                "sample_count": payload["sample_count"],
            })
    print(json.dumps({"fold": args.fold, "manifests": outputs}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
