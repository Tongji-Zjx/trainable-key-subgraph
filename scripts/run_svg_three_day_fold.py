"""Run one resumable fold of the frozen three-day SVG study."""

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

from keysubgraph.crossfit.svg_three_day_runner import (  # noqa: E402
    SVG_THREE_DAY_ALL_CANDIDATES,
    SVG_THREE_DAY_SCREEN_CANDIDATES,
    build_svg_three_day_fold_commands,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-crossfit-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument(
        "--candidates",
        nargs="+",
        choices=SVG_THREE_DAY_ALL_CANDIDATES,
        default=list(SVG_THREE_DAY_SCREEN_CANDIDATES),
    )
    parser.add_argument(
        "--mode", choices=("screen", "confirmatory"), default="screen"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--selection-seed",
        type=int,
        default=42,
        help="frozen hard-selection seed; independent of classifier seed",
    )
    parser.add_argument("--model-epochs", type=int, default=60)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--print-only", action="store_true")
    return parser.parse_args()


def _validate_source(args):
    fold = args.source_crossfit_root / "fold_{}".format(args.fold)
    required = [
        fold / "protocol" / "data_protocol.json",
        fold / "selector" / "best_checkpoint.pt",
        fold / "scaler.json",
    ]
    splits = (
        ("train", "validation", "test")
        if args.mode == "confirmatory"
        else ("train", "validation")
    )
    required.extend(fold / "cache" / split / "manifest.json" for split in splits)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "three-day SVG source artifacts are missing: {}".format(missing)
        )


def main():
    args = parse_args()
    _validate_source(args)
    commands = build_svg_three_day_fold_commands(
        PROJECT_ROOT,
        args.source_crossfit_root,
        args.output_root,
        args.fold,
        candidates=args.candidates,
        mode=args.mode,
        device=args.device,
        seed=args.seed,
        selection_seed=args.selection_seed,
        model_epochs=args.model_epochs,
        num_workers=args.num_workers,
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
                "stage did not create completion artifact: {}".format(artifact)
            )
        print("FINISH {}".format(stage), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
