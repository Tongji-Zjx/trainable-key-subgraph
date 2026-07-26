"""Run cheap validation-only A-E masking with frozen D3 components."""

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
    FEATURE_MASK_CONDITIONS,
    apply_frozen_feature_mask,
    build_frozen_feature_mask_evaluation,
    write_frozen_feature_mask_artifacts,
)
from keysubgraph.data.data_protocol import validate_data_protocol  # noqa: E402
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


def _dataset(protocol):
    paths = protocol["paths"]
    return ExactSTSEDataset(
        PROJECT_ROOT / paths["dataset_root"],
        PROJECT_ROOT / paths["sample_index_csv"],
        PROJECT_ROOT / paths["splits_csv"],
        "validation",
        protocol["edge_presence_threshold"],
        require_coordinates=False,
    )


def _validate_coverage(dataset, records):
    assignments = {
        item.sample_key: int(item.label) for item in dataset.assignments
    }
    cached = {record.sample_key: int(record.label) for record in records}
    if assignments != cached:
        raise ValueError(
            "validation manifest does not exactly cover frozen validation"
        )


def _validate_provenance(
    protocol_sha,
    selector_sha,
    scaler_sha,
    manifest,
    scaler,
    sgw_payload,
):
    if manifest["split"] != "validation":
        raise ValueError("feature-mask diagnosis requires validation manifest")
    if manifest["protocol_sha256"] != protocol_sha:
        raise ValueError("feature-mask manifest protocol mismatch")
    if manifest["selector_checkpoint_sha256"] != selector_sha:
        raise ValueError("feature-mask selector provenance mismatch")
    if manifest["selection_mode"] != "learned":
        raise ValueError("feature-mask diagnosis requires learned D3")
    if (
        scaler.protocol_sha256 != protocol_sha
        or scaler.selector_checkpoint_sha256 != selector_sha
        or scaler.selection_mode != manifest["selection_mode"]
        or int(scaler.selection_seed) != int(manifest["selection_seed"])
    ):
        raise ValueError("feature-mask scaler provenance mismatch")
    provenance = sgw_payload.get("provenance", {})
    if (
        provenance.get("selector_checkpoint_sha256") != selector_sha
        or provenance.get("sgw_scaler_sha256") != scaler_sha
    ):
        raise ValueError("feature-mask classifier provenance mismatch")


def _collect_proxy_features(
    dataset,
    selector_model,
    device,
    selection_seed,
    batch_size,
    num_workers,
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
    features = []
    processed = 0
    started = time.perf_counter()
    selector_model.eval()
    with torch.no_grad():
        for cpu_batch in loader:
            batch = cpu_batch.to(device)
            selection = selector_model.selector(
                batch,
                selection_mode="learned",
                random_seed=selection_seed,
            )
            raw = selector_model.proxy(
                batch, selection.hard_windows
            ).representation.to(torch.float32)
            sample_keys.extend(batch.sample_keys)
            labels.extend(int(sample.label) for sample in batch)
            features.extend(raw.detach().cpu().tolist())
            processed += len(batch)
            print(
                "validation processed {}/{} elapsed={:.1f}s".format(
                    processed,
                    len(dataset),
                    time.perf_counter() - started,
                ),
                flush=True,
            )
    return (
        sample_keys,
        labels,
        np.asarray(features, dtype=np.float64),
    )


def _predict_conditions(raw, train_mean, exact_model, scaler, device):
    probabilities = {}
    exact_model.eval()
    scaler.eval()
    with torch.no_grad():
        for specification in FEATURE_MASK_CONDITIONS:
            code = specification["code"]
            masked = apply_frozen_feature_mask(raw, train_mean, code)
            tensor = torch.tensor(
                masked, dtype=torch.float32, device=device
            )
            logits = exact_model.sgw_auxiliary_head(scaler(tensor))
            probabilities[code] = (
                torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
            )
    return probabilities


def main():
    args = parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("invalid frozen feature-mask loader settings")
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("frozen feature-mask output exists")
    protocol = validate_data_protocol(args.protocol, PROJECT_ROOT)
    protocol_sha = file_sha256(args.protocol)
    selector_sha = file_sha256(args.selector_checkpoint)
    scaler_sha = file_sha256(args.scaler)
    manifest, records, _ = read_dual_sgw_manifest(
        args.validation_manifest
    )
    dataset = _dataset(protocol)
    _validate_coverage(dataset, records)
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
        manifest,
        scaler,
        sgw_payload,
    )
    selection_seed = int(manifest["selection_seed"])
    started = time.perf_counter()
    sample_keys, labels, raw = _collect_proxy_features(
        dataset,
        selector_model,
        device,
        selection_seed,
        args.batch_size,
        args.num_workers,
    )
    probabilities = _predict_conditions(
        raw,
        scaler.mean.detach().cpu().numpy(),
        exact_model,
        scaler,
        device,
    )
    evaluation = build_frozen_feature_mask_evaluation(
        sample_keys, labels, probabilities
    )
    provenance = {
        "read_only_frozen_models": True,
        "test_split_used": False,
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
        "selection_mode": manifest["selection_mode"],
        "selection_seed": selection_seed,
        "device": str(device),
        "elapsed_seconds": time.perf_counter() - started,
    }
    artifacts = write_frozen_feature_mask_artifacts(
        output_dir, evaluation, provenance
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "artifacts": {
                    name: str(path) for name, path in artifacts.items()
                },
                "shared_threshold": evaluation["shared_threshold"],
                "conditions": [
                    {
                        "code": row["code"],
                        "roc_auc": row["metrics"]["roc_auc"],
                        "auc_delta_vs_A": row["auc_delta_vs_A"],
                        "balanced_accuracy": row["metrics"][
                            "balanced_accuracy"
                        ],
                        "accuracy": row["metrics"]["accuracy"],
                        "f1": row["metrics"]["f1"],
                    }
                    for row in evaluation["conditions"]
                ],
                "contrasts": evaluation["contrasts"],
                "duplicate_condition_check": evaluation[
                    "duplicate_condition_check"
                ],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
