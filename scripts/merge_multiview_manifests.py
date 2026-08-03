"""Merge disjoint multi-view cache shards with strict provenance checks."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.multiview_critical import (  # noqa: E402
    read_multiview_manifest,
    write_multiview_manifest,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    paths, keys, provenance, split = [], set(), None, None
    for manifest_path in args.manifest:
        payload, records = read_multiview_manifest(manifest_path, PROJECT_ROOT)
        current = (
            payload["protocol_sha256"],
            payload["selector_checkpoint_sha256"],
            payload["feature_schema_sha256"],
            payload.get("feature_config"),
            payload.get("git_commit"),
        )
        if provenance is None:
            provenance, split = current, payload["split"]
        if current != provenance or payload["split"] != split:
            raise ValueError("multi-view shard provenance mismatch")
        for row, record in zip(payload["records"], records):
            if record.features.sample_key in keys:
                raise ValueError("multi-view shards overlap")
            keys.add(record.features.sample_key)
            paths.append(PROJECT_ROOT / row["feature_path"])
    if len(paths) != int(args.expected_count):
        raise ValueError(
            "merged multi-view count mismatch: {} != {}".format(
                len(paths), args.expected_count
            )
        )
    output = write_multiview_manifest(
        paths, args.output, PROJECT_ROOT, overwrite=args.overwrite
    )
    print(json.dumps({
        "manifest": str(output), "split": split,
        "sample_count": len(paths), "shard_count": len(args.manifest),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
