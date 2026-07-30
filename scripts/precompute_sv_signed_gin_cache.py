"""Freeze selector outputs and precompute SV/Signed-GIN hard-graph records."""

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

from keysubgraph.data.data_protocol import (  # noqa: E402
    protocol_node_name_policy,
    validate_data_protocol,
)
from keysubgraph.data.data_split import file_sha256  # noqa: E402
from keysubgraph.data.exact_stse_dataset import (  # noqa: E402
    ExactSTSEDataset,
    create_exact_stse_loader,
)
from keysubgraph.data.sv_signed_gin_artifact import (  # noqa: E402
    SVSignedGINRecord,
    SVSignedGINWindowRecord,
    load_sv_signed_gin_record,
    save_sv_signed_gin_record,
)
from keysubgraph.data.sv_signed_gin_manifest import (  # noqa: E402
    sv_signed_gin_filename,
    write_sv_signed_gin_manifest_from_paths,
)
from keysubgraph.features.sv_hard_graph_features import (  # noqa: E402
    SVHardSampleFeatureBuilder,
)
from keysubgraph.models.dual_stse_hard_sgw import (  # noqa: E402
    DualSTSEHardSGWClassifier,
)
from keysubgraph.training.dual_stse_hard_sgw_trainer import (  # noqa: E402
    load_dual_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT
        / "configs"
        / "data_protocol_exact_stse_no_coord_full.json",
    )
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), required=True
    )
    parser.add_argument(
        "--selection-mode",
        choices=("learned", "full", "random"),
        default="learned",
    )
    parser.add_argument("--selector-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.selection_mode == "learned" and args.selector_checkpoint is None:
        raise ValueError("learned SV selection requires a checkpoint")
    if args.selection_mode != "learned" and args.selector_checkpoint is not None:
        raise ValueError("full/random SV selection must not load a checkpoint")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("SV max-samples must be positive")
    protocol = validate_data_protocol(args.protocol, PROJECT_ROOT)
    protocol_sha256 = file_sha256(args.protocol)
    paths = protocol["paths"]
    dataset = ExactSTSEDataset(
        PROJECT_ROOT / paths["dataset_root"],
        PROJECT_ROOT / paths["sample_index_csv"],
        PROJECT_ROOT / paths["splits_csv"],
        args.split,
        protocol["edge_presence_threshold"],
        require_coordinates=False,
        node_name_policy=protocol_node_name_policy(protocol),
    )
    loader = create_exact_stse_loader(
        dataset,
        batch_size=1,
        seed=args.selection_seed,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=False,
    )
    device = torch.device(args.device)
    model = DualSTSEHardSGWClassifier().to(device)
    selector_sha256 = "none"
    if args.selector_checkpoint is not None:
        selector_sha256 = file_sha256(args.selector_checkpoint)
        load_dual_checkpoint(
            args.selector_checkpoint,
            model,
            device,
            expected_stage="selector_proxy",
            expected_protocol_sha256=protocol_sha256,
        )
    model.eval()
    builder = SVHardSampleFeatureBuilder()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_paths = []
    with torch.no_grad():
        for index, cpu_batch in enumerate(loader):
            if args.max_samples is not None and index >= args.max_samples:
                break
            cpu_sample = cpu_batch[0].graph
            path = output_dir / sv_signed_gin_filename(
                cpu_sample.sample_key
            )
            if path.exists() and not args.overwrite:
                record = load_sv_signed_gin_record(path)
                checks = (
                    record.sample_key == cpu_sample.sample_key,
                    record.sample_id == cpu_sample.sample_id,
                    record.subject_id == cpu_sample.subject_id,
                    record.site == cpu_sample.site,
                    int(record.label) == int(cpu_sample.label),
                    record.split == cpu_sample.split == args.split,
                    record.protocol_sha256 == protocol_sha256,
                    record.selector_checkpoint_sha256
                    == selector_sha256,
                    record.selection_mode == args.selection_mode,
                    int(record.selection_seed)
                    == int(args.selection_seed),
                )
                if not all(checks):
                    raise ValueError(
                        "existing SV cache record provenance mismatch: "
                        + cpu_sample.sample_key
                    )
                feature_paths.append(path)
                print(
                    "reused {}/{} {} valid_windows={} "
                    "transitions={}".format(
                        index + 1,
                        min(
                            len(dataset),
                            args.max_samples
                            if args.max_samples is not None
                            else len(dataset),
                        ),
                        cpu_sample.sample_key,
                        record.valid_window_count,
                        record.valid_transition_count,
                    ),
                    flush=True,
                )
                del record
                continue
            batch = cpu_batch.to(device)
            selection = model.selector(
                batch,
                selection_mode=args.selection_mode,
                random_seed=args.selection_seed,
            )
            sample = batch[0].graph
            cropped = tuple(
                window.cropped_graph if window.window_valid else None
                for window in selection.hard_windows[0]
            )
            features = builder.build(cropped)
            window_records = tuple(
                (
                    SVSignedGINWindowRecord(
                        node_features=window.node_features.detach().cpu(),
                        adjacency=window.adjacency.detach().cpu(),
                        time_start=float(window.time_start),
                    )
                    if window is not None
                    else None
                )
                for window in features.windows
            )
            record = SVSignedGINRecord(
                sample_key=sample.sample_key,
                sample_id=sample.sample_id,
                subject_id=sample.subject_id,
                site=sample.site,
                label=sample.label,
                split=sample.split,
                windows=window_records,
                static_features=features.static_features.detach().cpu(),
                variation=features.variation.detach().cpu(),
                window_mask=features.window_mask.detach().cpu(),
                transition_mask=features.transition_mask.detach().cpu(),
                protocol_sha256=protocol_sha256,
                selector_checkpoint_sha256=selector_sha256,
                selection_mode=args.selection_mode,
                selection_seed=args.selection_seed,
            )
            save_sv_signed_gin_record(
                record, path, overwrite=args.overwrite
            )
            feature_paths.append(path)
            print(
                "processed {}/{} {} valid_windows={} transitions={}".format(
                    index + 1,
                    min(
                        len(dataset),
                        args.max_samples
                        if args.max_samples is not None
                        else len(dataset),
                    ),
                    sample.sample_key,
                    record.valid_window_count,
                    record.valid_transition_count,
                ),
                flush=True,
            )
            del record
            del window_records
            del features
            del selection
            del cropped
            del sample
            del batch
    manifest = write_sv_signed_gin_manifest_from_paths(
        feature_paths,
        output_dir / "manifest.json",
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest),
                "sample_count": len(feature_paths),
                "split": args.split,
                "selection_mode": args.selection_mode,
                "selector_checkpoint_sha256": selector_sha256,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
