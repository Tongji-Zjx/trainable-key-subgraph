"""Build and verify matched-source baseline manifests and downstream splits."""

from __future__ import absolute_import, print_function

import argparse
import hashlib
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.baseline_downstream_split import (  # noqa: E402
    create_baseline_downstream_splits,
)
from keysubgraph.data.baseline_manifest import build_baseline_manifest  # noqa: E402
from keysubgraph.data.data_split import SplitConfig, file_sha256  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "configs" / "data_protocol_all_samples.json",
    )
    parser.add_argument("--matched-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--search-attempts", type=int, default=256)
    parser.add_argument("--max-class-ratio-deviation", type=float, default=0.05)
    return parser.parse_args()


def _assignment_signature(assignments):
    normalized = sorted(
        (
            str(item["sample_key"]),
            str(item["split"]),
            int(item["label"]),
            str(item.get("group_id", "")),
        )
        for item in assignments
    )
    encoded = json.dumps(normalized, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), normalized


def main():
    args = parse_args()
    matched_root = args.matched_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError("source experiment output root already exists")
    matched_manifest = matched_root / "matched_control_manifest.json"
    if not matched_manifest.is_file():
        raise FileNotFoundError(str(matched_manifest))
    with matched_manifest.open("r", encoding="utf-8") as handle:
        matched_payload = json.load(handle)
    matched_sources = tuple(str(item) for item in matched_payload.get("sources", []))
    if not matched_sources or len(set(matched_sources)) != len(matched_sources):
        raise ValueError("matched-control manifest has invalid sources")
    split_config = SplitConfig(
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        search_attempts=args.search_attempts,
        max_class_ratio_deviation=args.max_class_ratio_deviation,
    )
    sources = {}
    reference_assignments = None
    assignment_hash = None
    for source in matched_sources:
        manifest_dir = output_root / "manifests" / source
        parent = build_baseline_manifest(
            project_root=PROJECT_ROOT,
            protocol_path=args.protocol,
            export_dir=matched_root / source,
            split="all",
            output_dir=manifest_dir,
            checkpoint_path=args.checkpoint,
            evidence_level="exploratory_in_sample",
            matched_control_manifest_path=matched_manifest,
            subgraph_source=source,
        )
        parent_path = manifest_dir / "baseline_manifest.json"
        split_dir = output_root / "splits" / source
        result = create_baseline_downstream_splits(
            PROJECT_ROOT,
            parent_path,
            split_dir,
            config=split_config,
        )
        with (split_dir / "baseline_splits.json").open("r", encoding="utf-8") as handle:
            split_payload = json.load(handle)
        current_hash, current_assignments = _assignment_signature(
            split_payload["assignments"]
        )
        if reference_assignments is None:
            reference_assignments = current_assignments
            assignment_hash = current_hash
        elif current_assignments != reference_assignments:
            raise RuntimeError("downstream assignments differ between sources")
        sources[source] = {
            "parent_manifest": str(parent_path),
            "parent_manifest_sha256": file_sha256(parent_path),
            "sample_count": parent["sample_count"],
            "subgraph_count": parent["subgraph_count"],
            "train_manifest": result["manifests"]["train"],
            "validation_manifest": result["manifests"]["validation"],
            "test_manifest": result["manifests"]["test"],
        }
    payload = {
        "schema_version": 1,
        "immutable": True,
        "purpose": "baseline_matched_source_experiment",
        "evidence_level": "exploratory_in_sample",
        "matched_control_manifest": str(matched_manifest),
        "matched_control_manifest_sha256": file_sha256(matched_manifest),
        "downstream_assignment_sha256": assignment_hash,
        "downstream_seed": args.seed,
        "sources": sources,
    }
    experiment_path = output_root / "source_experiment.json"
    with experiment_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({
        "output_root": str(output_root),
        "source_experiment": str(experiment_path),
        "source_count": len(sources),
        "sample_count": len(reference_assignments),
        "downstream_assignment_sha256": assignment_hash,
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
