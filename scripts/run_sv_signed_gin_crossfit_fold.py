"""Run one resumable SV Signed-GIN cross-fit fold."""

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

from keysubgraph.crossfit.sv_signed_gin_runner import (  # noqa: E402
    SV_CROSSFIT_DEFAULT_VARIANT,
    SV_CROSSFIT_VARIANTS,
    build_sv_crossfit_fold_commands,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "sv_signed_gin_crossfit"
        / "wmrc_3fold_seed202607",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=SV_CROSSFIT_VARIANTS,
        default=[SV_CROSSFIT_DEFAULT_VARIANT],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selector-epochs", type=int, default=80)
    parser.add_argument("--model-epochs", type=int, default=80)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--print-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    protocol = (
        args.output_root
        / "fold_{}".format(args.fold)
        / "protocol"
        / "data_protocol.json"
    )
    if not protocol.is_file():
        raise FileNotFoundError(
            "prepare cross-fit protocols first: {}".format(protocol)
        )
    commands = build_sv_crossfit_fold_commands(
        PROJECT_ROOT,
        args.output_root,
        args.fold,
        variants=args.variants,
        device=args.device,
        seed=args.seed,
        selector_epochs=args.selector_epochs,
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
            print(
                "SKIP {}: {} exists".format(stage, artifact),
                flush=True,
            )
            continue
        print("START {}".format(stage), flush=True)
        subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)
        if not artifact.is_file():
            raise RuntimeError(
                "stage did not create completion artifact: {}".format(
                    artifact
                )
            )
        print("FINISH {}".format(stage), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
