"""Run one resumable fold of promoted representation-level F2."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.crossfit.svg_short_term_representation_f2_runner import (  # noqa: E402
    build_svg_short_term_representation_f2_fold_commands,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-crossfit-root", type=Path, required=True)
    parser.add_argument("--g2-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--short-term-seed", type=int, required=True)
    parser.add_argument("--g2-seed", type=int, default=43)
    parser.add_argument("--fusion-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--print-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    commands = build_svg_short_term_representation_f2_fold_commands(
        PROJECT_ROOT,
        args.source_crossfit_root,
        args.g2_root,
        args.output_root,
        args.fold,
        args.short_term_seed,
        g2_seed=args.g2_seed,
        fusion_seed=args.fusion_seed,
        device=args.device,
        epochs=args.epochs,
        num_workers=args.num_workers,
    )
    missing_inputs = []
    for _, command, _ in commands[:3]:
        for flag in (
            "--protocol",
            "--short-term-checkpoint",
            "--g2-manifest",
            "--g2-scaler",
            "--g2-spectral-manifest",
            "--g2-spectral-scaler",
            "--g2-checkpoint",
        ):
            path = Path(command[command.index(flag) + 1])
            if not path.is_file():
                missing_inputs.append(str(path))
    if missing_inputs:
        raise FileNotFoundError(
            "representation F2 inputs are missing: {}".format(
                sorted(set(missing_inputs))
            )
        )
    if args.print_only:
        print(
            json.dumps(
                [
                    {
                        "stage": stage,
                        "command": command,
                        "completion_artifact": str(artifact),
                    }
                    for stage, command, artifact in commands
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    for stage, command, artifact in commands:
        if artifact.is_file():
            print("SKIP {}: {} exists".format(stage, artifact), flush=True)
            continue
        print("START {}".format(stage), flush=True)
        subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)
        if not artifact.is_file():
            raise RuntimeError(
                "representation F2 stage did not create {}".format(artifact)
            )
        print("FINISH {}".format(stage), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

