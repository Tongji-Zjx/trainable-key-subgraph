"""Run extractor, Key/Random controls, and A-D models for one OOF fold."""

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

from keysubgraph.crossfit.fold_runner import build_fold_commands  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--downstream-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--print-only", action="store_true")
    parser.add_argument("--force-stage", action="append", default=[])
    args = parser.parse_args()
    protocol = PROJECT_ROOT / "outputs/crossfit/fold_{}/protocol/data_protocol.json".format(args.fold)
    if not protocol.is_file():
        raise FileNotFoundError("prepare the fold protocol first: {}".format(protocol))
    commands = build_fold_commands(
        PROJECT_ROOT, args.fold, args.downstream_seed, args.device, args.smoke
    )
    if args.print_only:
        print(json.dumps([
            {"stage": stage, "command": command, "completion_artifact": str(artifact)}
            for stage, command, artifact in commands
        ], ensure_ascii=False, indent=2))
        return 0
    for stage, command, artifact in commands:
        artifact_path = PROJECT_ROOT / artifact
        if artifact_path.exists() and stage not in set(args.force_stage):
            print("SKIP {}: {} exists".format(stage, artifact), flush=True)
            continue
        print("START {}".format(stage), flush=True)
        subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)
        if not artifact_path.exists():
            raise RuntimeError("stage did not create completion artifact: {}".format(stage))
        print("FINISH {}".format(stage), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
