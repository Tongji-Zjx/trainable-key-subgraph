"""Formally evaluate frozen D3 B-path Variation-Only Exact-Head."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.analysis.dual_frozen_feature_masking import (  # noqa: E402
    apply_frozen_feature_mask,
)
from keysubgraph.analysis.dual_variation_only_exact_head import (  # noqa: E402
    build_variation_only_exact_head_evaluation,
    write_variation_only_exact_head_artifacts,
)
from keysubgraph.data.data_protocol import (  # noqa: E402
    protocol_node_name_policy,
    validate_data_protocol,
)
from keysubgraph.data.data_split import file_sha256  # noqa: E402
from keysubgraph.data.dual_sgw_manifest import (  # noqa: E402
    read_dual_sgw_manifest,
)
from keysubgraph.data.dual_sgw_scaler import (  # noqa: E402
    load_dual_sgw_standardizer,
)
from keysubgraph.data.exact_stse_dataset import (  # noqa: E402
    ExactSTSEDataset,
    create_exact_stse_loader,
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
    parser.add_argument("--selector-checkpoint", type=Path, required=True)
    parser.add_argument("--sgw-checkpoint", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def _device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _dataset(protocol, split):
    paths = protocol["paths"]
    return ExactSTSEDataset(
        PROJECT_ROOT / paths["dataset_root"],
        PROJECT_ROOT / paths["sample_index_csv"],
        PROJECT_ROOT / paths["splits_csv"],
        split,
        protocol["edge_presence_threshold"],
        require_coordinates=False,
        node_name_policy=protocol_node_name_policy(protocol),
    )


def _validate_coverage(dataset, records, name):
    assignments = {
        item.sample_key: int(item.label) for item in dataset.assignments
    }
    cached = {record.sample_key: int(record.label) for record in records}
    if assignments != cached:
        raise ValueError(
            "{} manifest does not exactly cover frozen split".format(name)
        )


def _validate_provenance(
    protocol_sha,
    selector_sha,
    scaler_sha,
    validation_manifest,
    test_manifest,
    scaler,
    sgw_payload,
):
    if (
        validation_manifest["split"] != "validation"
        or test_manifest["split"] != "test"
    ):
        raise ValueError("Variation-Only manifests use wrong splits")
    keys = (
        "protocol_sha256",
        "selector_checkpoint_sha256",
        "selection_mode",
        "selection_seed",
    )
    for key in keys:
        if validation_manifest[key] != test_manifest[key]:
            raise ValueError(
                "Variation-Only manifests disagree on {}".format(key)
            )
    if validation_manifest["protocol_sha256"] != protocol_sha:
        raise ValueError("Variation-Only protocol hash mismatch")
    if validation_manifest["selector_checkpoint_sha256"] != selector_sha:
        raise ValueError("Variation-Only selector provenance mismatch")
    if validation_manifest["selection_mode"] != "learned":
        raise ValueError("Variation-Only requires learned D3 selection")
    if (
        scaler.protocol_sha256 != protocol_sha
        or scaler.selector_checkpoint_sha256 != selector_sha
        or scaler.selection_mode
        != validation_manifest["selection_mode"]
        or int(scaler.selection_seed)
        != int(validation_manifest["selection_seed"])
    ):
        raise ValueError("Variation-Only scaler provenance mismatch")
    provenance = sgw_payload.get("provenance", {})
    if (
        provenance.get("selector_checkpoint_sha256") != selector_sha
        or provenance.get("sgw_scaler_sha256") != scaler_sha
    ):
        raise ValueError("Variation-Only classifier provenance mismatch")


def _predict(
    dataset,
    selector_model,
    exact_model,
    scaler,
    device,
    selection_seed,
    batch_size,
    num_workers,
    split,
):
    loader = create_exact_stse_loader(
        dataset,
        batch_size,
        seed=selection_seed,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    sample_keys = []
    labels = []
    probabilities = []
    processed = 0
    started = time.perf_counter()
    selector_model.eval()
    exact_model.eval()
    scaler.eval()
    train_mean = scaler.mean.detach().cpu().numpy()
    with torch.no_grad():
        for cpu_batch in loader:
            batch = cpu_batch.to(device)
            selection = selector_model.selector(
                batch,
                selection_mode="learned",
                random_seed=selection_seed,
            )
            proxy_raw = selector_model.proxy(
                batch, selection.hard_windows
            ).representation.to(torch.float32)
            masked = apply_frozen_feature_mask(
                proxy_raw.detach().cpu().numpy(),
                train_mean,
                "B",
            )
            masked_tensor = torch.tensor(
                np.asarray(masked),
                dtype=torch.float32,
                device=device,
            )
            logits = exact_model.sgw_auxiliary_head(
                scaler(masked_tensor)
            )
            positive = torch.softmax(logits, dim=-1)[:, 1]
            sample_keys.extend(batch.sample_keys)
            labels.extend(int(sample.label) for sample in batch)
            probabilities.extend(
                float(value)
                for value in positive.detach().cpu().tolist()
            )
            processed += len(batch)
            print(
                "{} processed {}/{} elapsed={:.1f}s".format(
                    split,
                    processed,
                    len(dataset),
                    time.perf_counter() - started,
                ),
                flush=True,
            )
    return sample_keys, labels, probabilities


def main():
    args = parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("invalid Variation-Only loader settings")
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Variation-Only output already exists")
    protocol = validate_data_protocol(args.protocol, PROJECT_ROOT)
    protocol_sha = file_sha256(args.protocol)
    selector_sha = file_sha256(args.selector_checkpoint)
    scaler_sha = file_sha256(args.scaler)
    validation_manifest, validation_records, _ = (
        read_dual_sgw_manifest(args.validation_manifest)
    )
    test_manifest, test_records, _ = read_dual_sgw_manifest(
        args.test_manifest
    )
    validation_dataset = _dataset(protocol, "validation")
    test_dataset = _dataset(protocol, "test")
    _validate_coverage(
        validation_dataset, validation_records, "validation"
    )
    _validate_coverage(test_dataset, test_records, "test")
    validation_keys = {
        item.sample_key for item in validation_dataset.assignments
    }
    test_keys = {item.sample_key for item in test_dataset.assignments}
    if validation_keys & test_keys:
        raise ValueError("Variation-Only validation and test overlap")
    device = _device(args.device)
    selector_model = DualSTSEHardSGWClassifier().to(device)
    selector_payload = load_dual_checkpoint(
        args.selector_checkpoint,
        selector_model,
        device,
        expected_stage="selector_proxy",
        expected_protocol_sha256=protocol_sha,
    )
    exact_model = DualSTSEHardSGWClassifier().to(device)
    sgw_payload = load_dual_checkpoint(
        args.sgw_checkpoint,
        exact_model,
        device,
        expected_stage="sgw_classifier",
        expected_protocol_sha256=protocol_sha,
    )
    scaler = load_dual_sgw_standardizer(args.scaler).to(device)
    _validate_provenance(
        protocol_sha,
        selector_sha,
        scaler_sha,
        validation_manifest,
        test_manifest,
        scaler,
        sgw_payload,
    )
    selection_seed = int(validation_manifest["selection_seed"])
    started = time.perf_counter()
    validation = _predict(
        validation_dataset,
        selector_model,
        exact_model,
        scaler,
        device,
        selection_seed,
        args.batch_size,
        args.num_workers,
        "validation",
    )
    test = _predict(
        test_dataset,
        selector_model,
        exact_model,
        scaler,
        device,
        selection_seed,
        args.batch_size,
        args.num_workers,
        "test",
    )
    evaluation = build_variation_only_exact_head_evaluation(
        validation_sample_keys=validation[0],
        validation_labels=validation[1],
        validation_probabilities=validation[2],
        test_sample_keys=test[0],
        test_labels=test[1],
        test_probabilities=test[2],
    )
    provenance = {
        "read_only_frozen_models": True,
        "protocol": str(Path(args.protocol).resolve()),
        "protocol_sha256": protocol_sha,
        "selector_checkpoint": str(
            Path(args.selector_checkpoint).resolve()
        ),
        "selector_checkpoint_sha256": selector_sha,
        "selector_best_epoch": int(selector_payload["best_epoch"]),
        "sgw_checkpoint": str(Path(args.sgw_checkpoint).resolve()),
        "sgw_checkpoint_sha256": file_sha256(args.sgw_checkpoint),
        "sgw_best_epoch": int(sgw_payload["best_epoch"]),
        "scaler": str(Path(args.scaler).resolve()),
        "scaler_sha256": scaler_sha,
        "scaler_fit_split": "train",
        "validation_manifest": str(
            Path(args.validation_manifest).resolve()
        ),
        "validation_manifest_sha256": file_sha256(
            args.validation_manifest
        ),
        "test_manifest": str(Path(args.test_manifest).resolve()),
        "test_manifest_sha256": file_sha256(args.test_manifest),
        "selection_mode": validation_manifest["selection_mode"],
        "selection_seed": selection_seed,
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(device),
        "all_34_proxy_evaluator_unchanged": True,
    }
    artifacts = write_variation_only_exact_head_artifacts(
        output_dir, evaluation, provenance
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "artifacts": {
                    name: str(path) for name, path in artifacts.items()
                },
                "primary_threshold_policy": evaluation[
                    "primary_threshold_policy"
                ],
                "thresholds": evaluation["thresholds"],
                "validation_metrics": evaluation["validation"]["metrics"],
                "test_metrics": evaluation["test"]["metrics"],
                "all_34_proxy_evaluator_unchanged": True,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
