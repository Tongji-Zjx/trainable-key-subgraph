"""Compare frozen selector outputs with the same low-capacity probe."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.analysis.selector_transfer_probe import (  # noqa: E402
    compare_selector_transfer_conditions,
    write_selector_transfer_probe_artifacts,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition",
        action="append",
        nargs=3,
        metavar=("NAME", "TRAIN_MANIFEST", "VALIDATION_MANIFEST"),
        required=True,
        help="repeat for every condition; first condition is reference",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    conditions = [
        (name, Path(train), Path(validation))
        for name, train, validation in args.condition
    ]
    payload = compare_selector_transfer_conditions(
        conditions, seed=args.seed
    )
    artifacts = write_selector_transfer_probe_artifacts(
        payload, args.output_dir
    )
    print(json.dumps(artifacts, ensure_ascii=False, indent=2))
    for row in payload["rows"]:
        print(
            "{name}: validation_auc={roc_auc:.6f} "
            "delta={delta_auc_vs_reference:+.6f} "
            "site_auc={site}".format(
                site=(
                    "N/A"
                    if row["site_stratified_roc_auc"] is None
                    else "{:.6f}".format(
                        row["site_stratified_roc_auc"]
                    )
                ),
                **row
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
