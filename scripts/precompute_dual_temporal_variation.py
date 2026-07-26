"""Cache exact per-transition variation and frozen D3-B base logits."""

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
from keysubgraph.data.data_protocol import validate_data_protocol  # noqa: E402
from keysubgraph.data.data_split import file_sha256  # noqa: E402
from keysubgraph.data.dual_sgw_manifest import (  # noqa: E402
    read_dual_sgw_manifest,
)
from keysubgraph.data.dual_sgw_scaler import (  # noqa: E402
    load_dual_sgw_standardizer,
)
from keysubgraph.data.dual_temporal_artifact import (  # noqa: E402
    DualTemporalVariationRecord,
    save_dual_temporal_record,
)
from keysubgraph.data.dual_temporal_manifest import (  # noqa: E402
    dual_temporal_filename,
    write_dual_temporal_manifest,
)
from keysubgraph.data.exact_stse_dataset import (  # noqa: E402
    ExactSTSEDataset,
    create_exact_stse_loader,
)
from keysubgraph.features.dual_temporal_variation import (  # noqa: E402
    DualTemporalVariationExtractor,
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
        "--split", choices=("train", "validation", "test"), required=True
    )
    parser.add_argument("--selector-checkpoint", type=Path, required=True)
    parser.add_argument("--sgw-checkpoint", type=Path, required=True)
    parser.add_argument("--sgw-scaler", type=Path, required=True)
    parser.add_argument("--exact-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--variation-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main():
    args = parse_args()
    if args.num_workers < 0 or args.variation_tolerance < 0.0:
        raise ValueError("invalid temporal precomputation arguments")
    protocol = validate_data_protocol(args.protocol, PROJECT_ROOT)
    protocol_sha = file_sha256(args.protocol)
    selector_sha = file_sha256(args.selector_checkpoint)
    exact_head_sha = file_sha256(args.sgw_checkpoint)
    scaler_sha = file_sha256(args.sgw_scaler)
    exact_manifest_sha = file_sha256(args.exact_manifest)
    exact_manifest, exact_records, _ = read_dual_sgw_manifest(
        args.exact_manifest
    )
    if (
        exact_manifest["split"] != args.split
        or exact_manifest["protocol_sha256"] != protocol_sha
        or exact_manifest["selector_checkpoint_sha256"] != selector_sha
        or exact_manifest["selection_mode"] != "learned"
        or int(exact_manifest["selection_seed"]) != args.selection_seed
    ):
        raise ValueError("exact manifest does not match temporal run")
    exact_lookup = {record.sample_key: record for record in exact_records}
    paths = protocol["paths"]
    dataset = ExactSTSEDataset(
        PROJECT_ROOT / paths["dataset_root"],
        PROJECT_ROOT / paths["sample_index_csv"],
        PROJECT_ROOT / paths["splits_csv"],
        args.split,
        protocol["edge_presence_threshold"],
        require_coordinates=False,
    )
    expected = {
        item.sample_key: int(item.label) for item in dataset.assignments
    }
    cached = {key: int(record.label) for key, record in exact_lookup.items()}
    if expected != cached:
        raise ValueError("exact manifest does not exactly cover frozen split")
    loader = create_exact_stse_loader(
        dataset,
        batch_size=1,
        seed=args.selection_seed,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=False,
    )
    device = _device(args.device)
    selector_model = DualSTSEHardSGWClassifier().to(device)
    load_dual_checkpoint(
        args.selector_checkpoint,
        selector_model,
        device,
        expected_stage="selector_proxy",
        expected_protocol_sha256=protocol_sha,
    )
    exact_model = DualSTSEHardSGWClassifier().to(device)
    exact_payload = load_dual_checkpoint(
        args.sgw_checkpoint,
        exact_model,
        device,
        expected_stage="sgw_classifier",
        expected_protocol_sha256=protocol_sha,
    )
    scaler = load_dual_sgw_standardizer(args.sgw_scaler).to(device)
    exact_provenance = exact_payload.get("provenance", {})
    if (
        exact_provenance.get("selector_checkpoint_sha256") != selector_sha
        or exact_provenance.get("sgw_scaler_sha256") != scaler_sha
    ):
        raise ValueError("exact head provenance mismatch")
    extractor = DualTemporalVariationExtractor(
        laplacian_eta=selector_model.config.laplacian_eta
    )
    selector_model.eval()
    exact_model.eval()
    scaler.eval()
    train_mean = scaler.mean.detach().cpu().numpy()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    maximum_alignment_error = 0.0
    started = time.perf_counter()
    with torch.no_grad():
        for index, cpu_batch in enumerate(loader):
            if args.max_samples is not None and index >= args.max_samples:
                break
            batch = cpu_batch.to(device)
            selection = selector_model.selector(
                batch,
                selection_mode="learned",
                random_seed=args.selection_seed,
            )
            proxy_raw = selector_model.proxy(
                batch, selection.hard_windows
            ).representation.to(torch.float32)
            masked = apply_frozen_feature_mask(
                proxy_raw.detach().cpu().numpy(), train_mean, "B"
            )
            masked_tensor = torch.tensor(
                np.asarray(masked), dtype=torch.float32, device=device
            )
            base_logits = exact_model.sgw_auxiliary_head(
                scaler(masked_tensor)
            )[0]
            sample = batch[0]
            windows = tuple(
                window.cropped_graph if window.window_valid else None
                for window in selection.hard_windows[0]
            )
            temporal = extractor.compute(windows)
            exact_record = exact_lookup[sample.sample_key]
            if bool(temporal.mask.any()):
                temporal_mean = temporal.values[temporal.mask].mean(dim=0)
            else:
                temporal_mean = temporal.values.new_zeros(16)
            error = float(
                (temporal_mean.cpu() - exact_record.variation).abs().max()
            )
            maximum_alignment_error = max(maximum_alignment_error, error)
            if error > args.variation_tolerance:
                raise ValueError(
                    "temporal/exact variation mismatch for {}: {}".format(
                        sample.sample_key, error
                    )
                )
            record = DualTemporalVariationRecord(
                sample_key=sample.sample_key,
                label=int(sample.label),
                split=sample.split,
                window_count=sample.num_timepoints,
                transition_values=temporal.values.detach().cpu(),
                transition_mask=temporal.mask.detach().cpu(),
                base_logits=base_logits.detach().cpu(),
                protocol_sha256=protocol_sha,
                selector_checkpoint_sha256=selector_sha,
                exact_head_checkpoint_sha256=exact_head_sha,
                sgw_scaler_sha256=scaler_sha,
                exact_manifest_sha256=exact_manifest_sha,
                selection_mode="learned",
                selection_seed=args.selection_seed,
            )
            feature_path = output_dir / dual_temporal_filename(
                sample.sample_key
            )
            save_dual_temporal_record(
                record, feature_path, overwrite=args.overwrite
            )
            records.append((record, feature_path))
            print(
                "{} processed {}/{} elapsed={:.1f}s".format(
                    args.split,
                    index + 1,
                    len(dataset),
                    time.perf_counter() - started,
                ),
                flush=True,
            )
    manifest = write_dual_temporal_manifest(
        records,
        output_dir / "manifest.json",
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest),
                "sample_count": len(records),
                "maximum_variation_alignment_error": (
                    maximum_alignment_error
                ),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
