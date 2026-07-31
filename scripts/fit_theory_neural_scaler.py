"""Fit the Stage-1 node, Q and Gamma scaler on inner-train only."""

from __future__ import absolute_import, division, print_function

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.theory_neural_scaler import (  # noqa: E402
    fit_theory_neural_scaler,
    save_theory_neural_scaler,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    scaler = fit_theory_neural_scaler(args.train_manifest, PROJECT_ROOT)
    save_theory_neural_scaler(scaler, args.output, overwrite=args.overwrite)
    print("Stage-1 scaler: {}".format(Path(args.output).resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
