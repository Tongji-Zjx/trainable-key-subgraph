"""Fuse independently trained Full Graph and Hard Graph SV channels."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.analysis.sv_full_hard_late_fusion import (  # noqa: E402
    build_sv_full_hard_late_fusion,
    write_sv_full_hard_late_fusion,
)


def _read(path):
    with Path(path).resolve().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hard-validation", type=Path, required=True)
    parser.add_argument("--hard-test", type=Path, required=True)
    parser.add_argument("--full-validation", type=Path, required=True)
    parser.add_argument("--full-test", type=Path, required=True)
    parser.add_argument(
        "--alpha-grid",
        nargs="+",
        type=float,
        default=(0.0, 0.25, 0.5, 0.75, 1.0),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = build_sv_full_hard_late_fusion(
        _read(args.hard_validation),
        _read(args.hard_test),
        _read(args.full_validation),
        _read(args.full_test),
        alpha_grid=args.alpha_grid,
    )
    paths = write_sv_full_hard_late_fusion(
        payload, args.output_dir
    )
    print(
        json.dumps(
            {
                "artifacts": paths,
                "selected_hard_weight": payload[
                    "selected_hard_weight"
                ],
                "selected_full_weight": payload[
                    "selected_full_weight"
                ],
                "standalone_validation_auc": payload[
                    "standalone_validation_auc"
                ],
                "validation": payload["validation"]["metrics"],
                "test": payload["test"]["metrics"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
