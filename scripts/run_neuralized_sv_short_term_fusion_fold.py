"""Fit validation-only ST + corrected neural S/V fusion for one outer fold."""

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

from keysubgraph.crossfit.neuralized_sv_runner import (  # noqa: E402
    build_neuralized_sv_short_term_fusion_command,
)
from keysubgraph.models.neuralized_sv import NEURALIZED_SV_VARIANTS  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-crossfit-root", type=Path, required=True)
    parser.add_argument("--neuralized-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--variant", choices=NEURALIZED_SV_VARIANTS, required=True)
    parser.add_argument("--short-term-seed", type=int, required=True)
    parser.add_argument("--neural-seed", type=int, default=42)
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()
    stage, command, artifact = build_neuralized_sv_short_term_fusion_command(
        PROJECT_ROOT,
        args.source_crossfit_root,
        args.neuralized_root,
        args.output_root,
        args.fold,
        args.variant,
        args.short_term_seed,
        neural_seed=args.neural_seed,
    )
    if args.print_only:
        print(
            json.dumps(
                {
                    "stage": stage,
                    "command": command,
                    "completion_artifact": str(artifact),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if artifact.is_file():
        print("SKIP {}: {} exists".format(stage, artifact), flush=True)
        return 0
    print("START {}".format(stage), flush=True)
    subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)
    if not artifact.is_file():
        raise RuntimeError("corrected neural S/V fusion created no artifact")
    print("FINISH {}".format(stage), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

