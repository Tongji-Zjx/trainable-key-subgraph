"""Average three independently trained SGW classifier probabilities."""

from __future__ import absolute_import, print_function

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.data_split import file_sha256  # noqa: E402
from keysubgraph.training.dual_sgw_feature_ensemble import (  # noqa: E402
    build_dual_sgw_probability_ensemble,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation-evaluations",
        type=Path,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--test-evaluations", type=Path, nargs="+", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read(path):
    with Path(path).resolve().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _atomic_json(path, payload):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def main():
    args = parse_args()
    if len(args.validation_evaluations) != len(args.test_evaluations):
        raise ValueError("validation/test component counts differ")
    validation = [_read(path) for path in args.validation_evaluations]
    test = [_read(path) for path in args.test_evaluations]
    payload = build_dual_sgw_probability_ensemble(validation, test)
    payload["component_files"] = {
        "validation": [
            {
                "path": str(Path(path).resolve()),
                "sha256": file_sha256(path),
            }
            for path in args.validation_evaluations
        ],
        "test": [
            {
                "path": str(Path(path).resolve()),
                "sha256": file_sha256(path),
            }
            for path in args.test_evaluations
        ],
    }
    _atomic_json(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "classifier_type": payload["classifier_type"],
                "component_seeds": payload["component_seeds"],
                "validation_thresholds": payload[
                    "validation_thresholds"
                ],
                "validation_metrics": payload["validation"]["metrics"],
                "test_metrics": payload["test"]["metrics"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
