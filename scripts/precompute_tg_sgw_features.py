"""Precompute canonical 18-D/34-D SGW features from hard graph caches."""

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

from keysubgraph.features import load_hard_graph_cache  # noqa: E402
from keysubgraph.theory import (  # noqa: E402
    SGWFeatureExtractor,
    TGSGWFeatureArtifact,
    save_tg_sgw_feature_artifact,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--gw-max-iter", type=int, default=100)
    parser.add_argument("--gw-sinkhorn-iter", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    source_paths = sorted(args.cache_dir.resolve().glob("*.pt"))
    if args.max_samples is not None:
        if args.max_samples < 1:
            raise ValueError("max-samples must be positive")
        source_paths = source_paths[: args.max_samples]
    if not source_paths:
        raise ValueError("hard graph cache directory contains no .pt files")
    extractor = SGWFeatureExtractor(
        gw_max_iter=args.gw_max_iter,
        gw_sinkhorn_iter=args.gw_sinkhorn_iter,
    )
    records = []
    for source_path in source_paths:
        cache = load_hard_graph_cache(source_path)
        windows = tuple(
            item.graph if item is not None else None for item in cache.windows
        )
        features = extractor.compute_hard_graph_sequence(windows, cache.time_values)
        artifact = TGSGWFeatureArtifact(
            sample_key=cache.sample_key,
            sample_id=cache.sample_id,
            label=cache.label,
            split=cache.split,
            features=features,
            eligible_for_stage_c=cache.eligible_for_stage_c and bool(features.transition_mask.any()),
            data_protocol_sha256=cache.data_protocol_sha256,
            teacher_checkpoint_sha256=cache.teacher_checkpoint_sha256,
        )
        output_path = args.output_dir / (cache.sample_id + ".pt")
        save_tg_sgw_feature_artifact(artifact, output_path, overwrite=args.overwrite)
        records.append({
            "sample_key": cache.sample_key,
            "sample_id": cache.sample_id,
            "label": cache.label,
            "split": cache.split,
            "eligible_for_stage_c": artifact.eligible_for_stage_c,
            "valid_transition_count": int(features.transition_mask.sum()),
            "feature_path": str(output_path.resolve()),
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest = {
        "schema_version": 1,
        "artifact_type": "tg_sgw_theory_feature_manifest",
        "time_quantity": "speed",
        "core_dim": 18,
        "classification_dim": 34,
        "spectral_w1_grid": list(extractor.spectral_w1_grid),
        "records": records,
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(manifest_path))
    print(json.dumps({
        "sample_count": len(records),
        "eligible_sample_count": sum(item["eligible_for_stage_c"] for item in records),
        "manifest": str(manifest_path.resolve()),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
