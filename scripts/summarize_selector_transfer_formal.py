"""Summarize formal selector-transfer training and fair probes."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.analysis.selector_transfer_summary import (  # noqa: E402
    summarize_selector_transfer_formal,
    write_selector_transfer_formal_summary,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    payload = summarize_selector_transfer_formal(args.root)
    paths = write_selector_transfer_formal_summary(payload, args.root)
    print(json.dumps(paths, ensure_ascii=False, indent=2))
    print(
        "E3 consistently beats current and random: {}".format(
            payload["e3_consistently_beats_current_and_random"]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
