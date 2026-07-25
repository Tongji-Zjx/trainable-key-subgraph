"""Diagnose frozen D3 proxy versus cached Exact-SGW representations."""

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

from keysubgraph.analysis.dual_proxy_exact_alignment import (  # noqa: E402
    analyze_proxy_exact_alignment,
    write_proxy_exact_alignment_artifacts,
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
    parser.add_argument(
        "--selector-checkpoint", type=Path, required=True
    )
    parser.add_argument("--sgw-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="validation",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _validate_provenance(
    protocol_sha,
    manifest,
    scaler,
    selector_checkpoint,
    sgw_payload,
    scaler_path,
):
    selector_sha = file_sha256(selector_checkpoint)
    scaler_sha = file_sha256(scaler_path)
    if manifest["protocol_sha256"] != protocol_sha:
        raise ValueError("Exact-SGW manifest protocol hash mismatch")
    if manifest["selector_checkpoint_sha256"] != selector_sha:
        raise ValueError(
            "selector checkpoint does not match cached Exact-SGW features"
        )
    if (
        scaler.protocol_sha256 != protocol_sha
        or scaler.selector_checkpoint_sha256 != selector_sha
        or scaler.selection_mode != manifest["selection_mode"]
        or int(scaler.selection_seed) != int(manifest["selection_seed"])
    ):
        raise ValueError("Exact-SGW scaler provenance mismatch")
    provenance = sgw_payload.get("provenance", {})
    if provenance.get("selector_checkpoint_sha256") != selector_sha:
        raise ValueError("D3 classifier uses a different selector")
    if provenance.get("sgw_scaler_sha256") != scaler_sha:
        raise ValueError("D3 classifier uses a different Exact-SGW scaler")
    return selector_sha, scaler_sha


def main():
    args = parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("invalid diagnostic loader configuration")
    if args.max_samples is not None and args.max_samples < 2:
        raise ValueError("alignment diagnosis requires at least two samples")
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("alignment diagnostic output already exists")
    protocol = validate_data_protocol(args.protocol, PROJECT_ROOT)
    protocol_sha = file_sha256(args.protocol)
    manifest, records, lookup = read_dual_sgw_manifest(args.manifest)
    if manifest["split"] != args.split:
        raise ValueError("Exact-SGW manifest split mismatch")
    record_lookup = {record.sample_key: record for record in records}
    scaler = load_dual_sgw_standardizer(args.scaler)
    paths = protocol["paths"]
    dataset = ExactSTSEDataset(
        PROJECT_ROOT / paths["dataset_root"],
        PROJECT_ROOT / paths["sample_index_csv"],
        PROJECT_ROOT / paths["splits_csv"],
        args.split,
        protocol["edge_presence_threshold"],
        require_coordinates=False,
    )
    dataset_keys = {item.sample_key for item in dataset.assignments}
    if dataset_keys != set(lookup):
        raise ValueError("Exact-SGW cache does not cover the frozen split")
    device = _device(args.device)
    selector_model = DualSTSEHardSGWClassifier().to(device)
    selector_payload = load_dual_checkpoint(
        args.selector_checkpoint,
        selector_model,
        device,
        expected_stage="selector_proxy",
        expected_protocol_sha256=protocol_sha,
    )
    sgw_model = DualSTSEHardSGWClassifier().to(device)
    sgw_payload = load_dual_checkpoint(
        args.sgw_checkpoint,
        sgw_model,
        device,
        expected_stage="sgw_classifier",
        expected_protocol_sha256=protocol_sha,
    )
    selector_sha, scaler_sha = _validate_provenance(
        protocol_sha,
        manifest,
        scaler,
        args.selector_checkpoint,
        sgw_payload,
        args.scaler,
    )
    proxy_threshold = selector_payload.get("validation_threshold")
    exact_threshold = sgw_payload.get("validation_threshold")
    if proxy_threshold is None or exact_threshold is None:
        raise ValueError(
            "D3 checkpoints require frozen validation thresholds"
        )
    selector_model.eval()
    sgw_model.eval()
    scaler = scaler.to(device).eval()
    loader = create_exact_stse_loader(
        dataset,
        args.batch_size,
        seed=int(manifest["selection_seed"]),
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    sample_keys = []
    labels = []
    proxy_features = []
    exact_features = []
    proxy_standardized = []
    exact_standardized = []
    proxy_masks = []
    exact_masks = []
    probabilities = {
        "proxy_proxy": [],
        "exact_proxy": [],
        "exact_exact": [],
        "proxy_exact": [],
    }
    processed = 0
    started = time.perf_counter()
    with torch.no_grad():
        for cpu_batch in loader:
            if (
                args.max_samples is not None
                and processed >= args.max_samples
            ):
                break
            samples = cpu_batch.samples
            if args.max_samples is not None:
                samples = samples[: args.max_samples - processed]
            batch = type(cpu_batch)(tuple(samples)).to(device)
            selection = selector_model.selector(
                batch,
                selection_mode="learned",
                random_seed=int(manifest["selection_seed"]),
            )
            proxy_output = selector_model.proxy(
                batch, selection.hard_windows
            )
            exact_raw = torch.stack(
                [
                    lookup[sample_key].to(torch.float32)
                    for sample_key in batch.sample_keys
                ],
                dim=0,
            ).to(device)
            proxy_raw = proxy_output.representation.to(torch.float32)
            exact_scaled = scaler(exact_raw)
            proxy_scaled = scaler(proxy_raw)
            path_logits = {
                "proxy_proxy": selector_model.selector_proxy_head(
                    proxy_raw
                ),
                "exact_proxy": selector_model.selector_proxy_head(
                    exact_raw
                ),
                "exact_exact": sgw_model.sgw_auxiliary_head(
                    exact_scaled
                ),
                "proxy_exact": sgw_model.sgw_auxiliary_head(
                    proxy_scaled
                ),
            }
            sample_keys.extend(batch.sample_keys)
            labels.extend(int(sample.label) for sample in batch)
            proxy_features.extend(proxy_raw.detach().cpu().tolist())
            exact_features.extend(exact_raw.detach().cpu().tolist())
            proxy_standardized.extend(
                proxy_scaled.detach().cpu().tolist()
            )
            exact_standardized.extend(
                exact_scaled.detach().cpu().tolist()
            )
            for index, sample in enumerate(batch):
                transition_count = max(0, sample.num_timepoints - 1)
                proxy_masks.append(
                    proxy_output.transition_mask[
                        index, :transition_count
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                )
                exact_masks.append(
                    record_lookup[
                        sample.sample_key
                    ].transition_mask.detach().cpu().numpy()
                )
            for name, logits in path_logits.items():
                probabilities[name].extend(
                    torch.softmax(logits, dim=-1)[:, 1]
                    .detach()
                    .cpu()
                    .tolist()
                )
            processed += len(batch)
            print(
                "processed {}/{} elapsed={:.1f}s".format(
                    processed,
                    min(
                        len(dataset),
                        args.max_samples
                        if args.max_samples is not None
                        else len(dataset),
                    ),
                    time.perf_counter() - started,
                ),
                flush=True,
            )
    if processed < 2:
        raise ValueError("diagnostic processed too few samples")
    analysis = analyze_proxy_exact_alignment(
        sample_keys=sample_keys,
        labels=labels,
        proxy_features=np.asarray(proxy_features),
        exact_features=np.asarray(exact_features),
        probabilities=probabilities,
        proxy_threshold=float(proxy_threshold),
        exact_threshold=float(exact_threshold),
        proxy_transition_masks=proxy_masks,
        exact_transition_masks=exact_masks,
        proxy_standardized=np.asarray(proxy_standardized),
        exact_standardized=np.asarray(exact_standardized),
    )
    provenance = {
        "read_only": True,
        "split": args.split,
        "protocol": str(Path(args.protocol).resolve()),
        "protocol_sha256": protocol_sha,
        "selector_checkpoint": str(
            Path(args.selector_checkpoint).resolve()
        ),
        "selector_checkpoint_sha256": selector_sha,
        "sgw_checkpoint": str(Path(args.sgw_checkpoint).resolve()),
        "sgw_checkpoint_sha256": file_sha256(args.sgw_checkpoint),
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": file_sha256(args.manifest),
        "scaler": str(Path(args.scaler).resolve()),
        "scaler_sha256": scaler_sha,
        "selection_mode": manifest["selection_mode"],
        "selection_seed": int(manifest["selection_seed"]),
        "proxy_threshold": float(proxy_threshold),
        "exact_threshold": float(exact_threshold),
        "elapsed_seconds": time.perf_counter() - started,
    }
    artifact_paths = write_proxy_exact_alignment_artifacts(
        output_dir, analysis, provenance
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "sample_count": processed,
                "artifacts": {
                    key: str(value)
                    for key, value in artifact_paths.items()
                },
                "feature_blocks": analysis["summary"][
                    "feature_blocks"
                ],
                "classification_paths": analysis["summary"][
                    "classification_paths"
                ],
                "probability_alignment": analysis["summary"][
                    "probability_alignment"
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
