"""Build and evaluate all 13 test-time perturbation conditions for one A checkpoint."""

from __future__ import absolute_import, print_function

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.baseline_manifest import build_baseline_manifest  # noqa: E402
from keysubgraph.data.edge_perturbation import build_edge_perturbation_exports  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--key-export-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base-seed", type=int, default=2026)
    args = parser.parse_args()
    conditions = []
    for repeat in range(5):
        repeat_root = args.output_root / "exports/repeat_{}".format(repeat)
        manifest_path = repeat_root / "matched_control_manifest.json"
        if not manifest_path.exists():
            build_edge_perturbation_exports(
                PROJECT_ROOT, args.protocol, args.key_export_root, "test", repeat_root,
                ratios=(0.0, 0.25, 0.50), perturbation_seed=args.base_seed + repeat,
            )
        sources = (
            ("key_edge_000", "none", 0.0),
            ("key_edge_targeted_025", "targeted", 0.25),
            ("key_edge_random_025", "random", 0.25),
            ("key_edge_targeted_050", "targeted", 0.50),
            ("key_edge_random_050", "random", 0.50),
        )
        for source, mode, dose in sources:
            if repeat > 0 and mode != "random":
                continue
            repeat_index = repeat if mode == "random" else None
            code = "{}_dose{:03d}_{}".format(mode, int(round(dose * 100)), "single" if repeat_index is None else "repeat{}".format(repeat_index))
            condition_root = args.output_root / code
            manifest_output = condition_root / "manifest"
            manifest_json = manifest_output / "baseline_manifest.json"
            if not manifest_json.exists():
                build_baseline_manifest(
                    PROJECT_ROOT, args.protocol, repeat_root / source, "test",
                    manifest_output, evidence_level="confirmatory_cross_fitted",
                    matched_control_manifest_path=manifest_path,
                    subgraph_source=source,
                )
            prediction_path = condition_root / "predictions.json"
            if not prediction_path.exists():
                subprocess.run([
                    "python", "-u", "scripts/evaluate_baseline.py",
                    "--manifest", str(manifest_json), "--checkpoint", str(args.checkpoint),
                    "--output", str(prediction_path), "--device", args.device,
                ], cwd=str(PROJECT_ROOT), check=True)
            conditions.append({
                "condition": code, "mode": mode, "dose": dose,
                "repeat_index": repeat_index, "predictions": str(prediction_path.resolve()),
            })
    summary = {
        "schema_version": 1, "outer_fold": args.fold, "model_seed": args.model_seed,
        "checkpoint": str(args.checkpoint.resolve()), "condition_count": len(conditions),
        "conditions": conditions,
    }
    summary_path = args.output_root / "perturbation_run_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"summary": str(summary_path.resolve()), "condition_count": len(conditions)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
