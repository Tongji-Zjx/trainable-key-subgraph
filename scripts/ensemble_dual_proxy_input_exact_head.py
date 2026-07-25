"""Ensemble multiple formal D3 ProxyInput-ExactHead seeds equally."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.analysis.dual_frozen_logit_ensemble import (  # noqa: E402
    build_frozen_equal_logit_ensemble,
    write_frozen_equal_logit_ensemble_artifacts,
)
from keysubgraph.data.data_split import file_sha256  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--component-dir",
        type=Path,
        action="append",
        required=True,
        help="Repeat for each formal ProxyInput-ExactHead seed directory.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _partition(payload, split):
    predictions = payload[split]["predictions"]
    return {
        "sample_keys": [
            str(item["sample_key"]) for item in predictions
        ],
        "labels": [int(item["label"]) for item in predictions],
        "probabilities": [
            float(item["positive_probability"]) for item in predictions
        ],
    }


def _read_component(path, index):
    directory = Path(path).resolve()
    evaluation_path = directory / "evaluation.json"
    with evaluation_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if (
        payload.get("artifact")
        != "dual_d3_proxy_input_exact_head_evaluation"
    ):
        raise ValueError(
            "{} is not a formal ProxyInput-ExactHead result".format(
                directory
            )
        )
    provenance = payload.get("provenance", {})
    selector_sha = provenance.get("selector_checkpoint_sha256")
    sgw_sha = provenance.get("sgw_checkpoint_sha256")
    if not selector_sha or not sgw_sha:
        raise ValueError("formal component lacks checkpoint provenance")
    name = "component_{:02d}_{}_{}".format(
        index + 1, selector_sha[:8], sgw_sha[:8]
    )
    return {
        "name": name,
        "directory": directory,
        "evaluation_path": evaluation_path,
        "evaluation_sha256": file_sha256(evaluation_path),
        "payload": payload,
        "protocol_sha256": provenance.get("protocol_sha256"),
        "selector_sha256": selector_sha,
        "sgw_sha256": sgw_sha,
    }


def _validate_components(components):
    if len(components) < 2:
        raise ValueError("multi-seed ensemble requires at least two seeds")
    if len({item["selector_sha256"] for item in components}) != len(
        components
    ):
        raise ValueError(
            "multi-seed ensemble contains repeated selector checkpoints"
        )
    protocols = {item["protocol_sha256"] for item in components}
    if len(protocols) != 1 or None in protocols:
        raise ValueError("multi-seed components use different protocols")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("multi-seed ensemble output already exists")
    components = [
        _read_component(path, index)
        for index, path in enumerate(args.component_dir)
    ]
    _validate_components(components)
    validation = {
        item["name"]: _partition(item["payload"], "validation")
        for item in components
    }
    test = {
        item["name"]: _partition(item["payload"], "test")
        for item in components
    }
    evaluation = build_frozen_equal_logit_ensemble(
        validation_components=validation,
        test_components=test,
        ensemble_scope="across_seed_proxy_input_exact_head",
    )
    provenance = {
        "read_only_frozen_predictions": True,
        "protocol_sha256": components[0]["protocol_sha256"],
        "component_count": len(components),
        "components": [
            {
                "name": item["name"],
                "directory": str(item["directory"]),
                "evaluation": str(item["evaluation_path"]),
                "evaluation_sha256": item["evaluation_sha256"],
                "selector_checkpoint_sha256": item[
                    "selector_sha256"
                ],
                "sgw_checkpoint_sha256": item["sgw_sha256"],
            }
            for item in components
        ],
    }
    paths = write_frozen_equal_logit_ensemble_artifacts(
        output_dir, evaluation, provenance
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "artifacts": {
                    name: str(path) for name, path in paths.items()
                },
                "component_names": evaluation["component_names"],
                "thresholds": evaluation["thresholds"],
                "validation_metrics": evaluation["validation"]["metrics"],
                "test_metrics": evaluation["test"]["metrics"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
