"""Run one resumable structured short-term cross-fit fold end to end."""

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

from keysubgraph.crossfit.structured_short_term_runner import (  # noqa: E402
    build_structured_short_term_crossfit_fold_commands,
)
from keysubgraph.models.structured_short_term import (  # noqa: E402
    PAPER_ALIGNED_VARIANT,
    PAPER_ALIGNED_PST_VARIANT,
    STRUCTURED_SAFE_VARIANT,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--evaluation-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--model-variant",
        choices=(
            STRUCTURED_SAFE_VARIANT,
            PAPER_ALIGNED_VARIANT,
            PAPER_ALIGNED_PST_VARIANT,
        ),
        default=STRUCTURED_SAFE_VARIANT,
    )
    parser.add_argument("--print-only", action="store_true")
    return parser.parse_args()


def _resume_command(stage, command, artifact):
    if stage != "train" or artifact.is_file():
        return command
    training_dir = artifact.parent
    last_checkpoint = training_dir / "last_checkpoint.pt"
    history = training_dir / "history.json"
    if last_checkpoint.is_file():
        print(
            "RESUME train: {}".format(last_checkpoint),
            flush=True,
        )
        return list(command) + ["--resume", str(last_checkpoint)]
    if history.exists():
        raise RuntimeError(
            "partial training output has history but no resumable checkpoint: "
            "{}".format(training_dir)
        )
    return command


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
            "prepare or copy the frozen fold protocol first: {}".format(
                protocol
            )
        )
    commands = build_structured_short_term_crossfit_fold_commands(
        PROJECT_ROOT,
        args.output_root,
        args.fold,
        device=args.device,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        evaluation_batch_size=args.evaluation_batch_size,
        num_workers=args.num_workers,
        model_variant=args.model_variant,
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
        command = _resume_command(stage, command, artifact)
        print("START {}".format(stage), flush=True)
        subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)
        if not artifact.is_file():
            raise RuntimeError(
                "stage did not create completion artifact: {}".format(
                    artifact
                )
            )
        print("FINISH {}".format(stage), flush=True)
    print(
        "FOLD {} COMPLETE: {}".format(
            args.fold,
            commands[-1][2],
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
