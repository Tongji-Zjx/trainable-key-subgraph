"""Run or resume one Stage-1 N0--N4 cross-fit fold."""

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

from keysubgraph.crossfit.theory_guided_runner import (  # noqa: E402
    build_stage1_fold_commands,
)
from keysubgraph.models.theory_guided_neural import (  # noqa: E402
    THEORY_NEURAL_VARIANTS,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crossfit-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--variants", nargs="+", choices=THEORY_NEURAL_VARIANTS,
                        default=list(THEORY_NEURAL_VARIANTS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--gw-max-iter", type=int, default=100)
    parser.add_argument("--gw-sinkhorn-iter", type=int, default=100)
    parser.add_argument("--gw-tolerance", type=float, default=1.0e-7)
    parser.add_argument("--print-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    commands = build_stage1_fold_commands(
        PROJECT_ROOT, args.crossfit_root, args.output_root, args.fold,
        variants=args.variants, device=args.device, seed=args.seed,
        epochs=args.epochs, batch_size=args.batch_size,
        accumulation_steps=args.gradient_accumulation_steps,
        num_workers=args.num_workers, gw_max_iter=args.gw_max_iter,
        gw_sinkhorn_iter=args.gw_sinkhorn_iter, gw_tolerance=args.gw_tolerance,
    )
    if args.print_only:
        print(json.dumps([
            {"stage": stage, "command": command,
             "completion_artifact": str(artifact)}
            for stage, command, artifact in commands
        ], ensure_ascii=False, indent=2))
        return 0
    for stage, command, artifact in commands:
        if artifact.is_file():
            print("SKIP {}: {} exists".format(stage, artifact), flush=True)
            continue
        print("START {}".format(stage), flush=True)
        subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)
        if not artifact.is_file():
            raise RuntimeError("Stage-1 command created no completion artifact")
        print("FINISH {}".format(stage), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
