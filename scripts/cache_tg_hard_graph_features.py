"""Recompute and cache Stage-C graph features from frozen hard exports."""

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

from keysubgraph.features import (  # noqa: E402
    HardExportFeatureAdapter,
    save_hard_graph_cache,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("max-samples must be positive")
    source_paths = sorted(
        path for path in args.export_dir.resolve().glob("*.json") if path.is_file()
    )
    if args.max_samples is not None:
        source_paths = source_paths[: args.max_samples]
    if not source_paths:
        raise ValueError("hard export directory contains no sample JSON files")
    adapter = HardExportFeatureAdapter()
    records = []
    exclusions = []
    for source_path in source_paths:
        with source_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        cache = adapter.build(payload)
        output_path = args.output_dir / (cache.sample_id + ".pt")
        save_hard_graph_cache(cache, output_path, overwrite=args.overwrite)
        record = {
            "sample_key": cache.sample_key,
            "sample_id": cache.sample_id,
            "label": cache.label,
            "split": cache.split,
            "num_valid_windows": cache.num_valid_windows,
            "eligible_for_stage_c": cache.eligible_for_stage_c,
            "cache_path": str(output_path.resolve()),
        }
        records.append(record)
        if not cache.eligible_for_stage_c:
            exclusions.append({
                "sample_key": cache.sample_key,
                "reason": cache.exclusion_reason,
            })
    manifest = {
        "schema_version": 1,
        "artifact_type": "tg_sgw_hard_graph_feature_cache_manifest",
        "sample_count": len(records),
        "eligible_sample_count": sum(item["eligible_for_stage_c"] for item in records),
        "excluded_sample_count": len(exclusions),
        "records": records,
        "exclusions": exclusions,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(manifest_path))
    print(json.dumps({
        "sample_count": manifest["sample_count"],
        "eligible_sample_count": manifest["eligible_sample_count"],
        "excluded_sample_count": manifest["excluded_sample_count"],
        "manifest": str(manifest_path.resolve()),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
