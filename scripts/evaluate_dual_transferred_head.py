"""Evaluate cached Exact-SGW with the frozen original D3 proxy head."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.analysis.dual_transferred_head import (  # noqa: E402
    build_transferred_head_evaluation,
    write_transferred_head_artifacts,
)
from keysubgraph.data.data_protocol import validate_data_protocol  # noqa: E402
from keysubgraph.data.data_split import file_sha256  # noqa: E402
from keysubgraph.data.dual_sgw_manifest import (  # noqa: E402
    read_dual_sgw_manifest,
)
from keysubgraph.models.dual_stse_hard_sgw import (  # noqa: E402
    DualSTSEHardSGWClassifier,
)
from keysubgraph.training.dual_stse_hard_sgw_trainer import (  # noqa: E402
    load_dual_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument(
        "--selector-checkpoint", type=Path, required=True
    )
    parser.add_argument(
        "--validation-manifest", type=Path, required=True
    )
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def _validate_manifests(validation, test, protocol_sha, selector_sha):
    if validation["split"] != "validation" or test["split"] != "test":
        raise ValueError("transferred-head manifests use wrong splits")
    keys = (
        "protocol_sha256",
        "selector_checkpoint_sha256",
        "selection_mode",
        "selection_seed",
    )
    for key in keys:
        if validation[key] != test[key]:
            raise ValueError(
                "transferred-head manifests disagree on {}".format(key)
            )
    if validation["protocol_sha256"] != protocol_sha:
        raise ValueError("transferred-head protocol hash mismatch")
    if validation["selector_checkpoint_sha256"] != selector_sha:
        raise ValueError(
            "transferred-head selector does not match Exact-SGW cache"
        )
    if validation["selection_mode"] != "learned":
        raise ValueError("transferred-head requires learned D3 features")


def _predict(records, head, device, batch_size):
    sample_keys = []
    labels = []
    probabilities = []
    head.eval()
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch_records = records[start : start + batch_size]
            features = torch.stack(
                [
                    record.representation.detach().to(torch.float32)
                    for record in batch_records
                ],
                dim=0,
            ).to(device)
            logits = head(features)
            positive = torch.softmax(logits, dim=-1)[:, 1]
            sample_keys.extend(
                record.sample_key for record in batch_records
            )
            labels.extend(int(record.label) for record in batch_records)
            probabilities.extend(
                float(value)
                for value in positive.detach().cpu().tolist()
            )
    return sample_keys, labels, probabilities


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("transferred-head batch size must be positive")
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("transferred-head output already exists")
    validate_data_protocol(args.protocol, PROJECT_ROOT)
    protocol_sha = file_sha256(args.protocol)
    selector_sha = file_sha256(args.selector_checkpoint)
    validation_payload, validation_records, _ = (
        read_dual_sgw_manifest(args.validation_manifest)
    )
    test_payload, test_records, _ = read_dual_sgw_manifest(
        args.test_manifest
    )
    _validate_manifests(
        validation_payload, test_payload, protocol_sha, selector_sha
    )
    validation_keys = {
        record.sample_key for record in validation_records
    }
    test_keys = {record.sample_key for record in test_records}
    if validation_keys & test_keys:
        raise ValueError("transferred-head validation and test overlap")
    device = torch.device(args.device)
    model = DualSTSEHardSGWClassifier().to(device)
    checkpoint = load_dual_checkpoint(
        args.selector_checkpoint,
        model,
        device,
        expected_stage="selector_proxy",
        expected_protocol_sha256=protocol_sha,
    )
    threshold = checkpoint.get("validation_threshold")
    if threshold is None:
        raise ValueError("selector checkpoint has no validation threshold")
    head = model.selector_proxy_head
    validation_keys_ordered, validation_labels, validation_probabilities = (
        _predict(validation_records, head, device, args.batch_size)
    )
    test_keys_ordered, test_labels, test_probabilities = _predict(
        test_records, head, device, args.batch_size
    )
    evaluation = build_transferred_head_evaluation(
        validation_sample_keys=validation_keys_ordered,
        validation_labels=validation_labels,
        validation_probabilities=validation_probabilities,
        test_sample_keys=test_keys_ordered,
        test_labels=test_labels,
        test_probabilities=test_probabilities,
        original_proxy_threshold=float(threshold),
    )
    evaluation["architecture"]["frozen_parameter_count"] = sum(
        parameter.numel() for parameter in head.parameters()
    )
    provenance = {
        "read_only": True,
        "protocol": str(Path(args.protocol).resolve()),
        "protocol_sha256": protocol_sha,
        "selector_checkpoint": str(
            Path(args.selector_checkpoint).resolve()
        ),
        "selector_checkpoint_sha256": selector_sha,
        "selector_best_epoch": int(checkpoint["best_epoch"]),
        "selector_best_validation_roc_auc": float(
            checkpoint["best_validation_roc_auc"]
        ),
        "validation_manifest": str(
            Path(args.validation_manifest).resolve()
        ),
        "validation_manifest_sha256": file_sha256(
            args.validation_manifest
        ),
        "test_manifest": str(Path(args.test_manifest).resolve()),
        "test_manifest_sha256": file_sha256(args.test_manifest),
        "selection_mode": validation_payload["selection_mode"],
        "selection_seed": int(validation_payload["selection_seed"]),
    }
    artifacts = write_transferred_head_artifacts(
        output_dir, evaluation, provenance
    )
    printable = {
        "output_dir": str(output_dir),
        "artifacts": {
            name: str(path) for name, path in artifacts.items()
        },
        "architecture": evaluation["architecture"],
        "thresholds": evaluation["thresholds"],
        "validation_metrics": evaluation["validation"]["metrics"],
        "test_metrics": evaluation["test"]["metrics"],
    }
    print(
        json.dumps(
            printable, ensure_ascii=False, indent=2, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
