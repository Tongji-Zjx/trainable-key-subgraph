"""Freeze selector outputs into Stage-1 edge-aware neural records."""

from __future__ import absolute_import, division, print_function

import argparse
import hashlib
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
from keysubgraph.data.theory_neural_artifact import (  # noqa: E402
    TheoryNeuralRecord,
    TheoryNeuralWindowRecord,
    load_theory_neural_record,
    save_theory_neural_record,
)
from keysubgraph.data.theory_neural_manifest import (  # noqa: E402
    theory_neural_filename,
    write_theory_neural_manifest,
)
from keysubgraph.features.theory_neural_features import (  # noqa: E402
    THEORY_EDGE_FEATURE_DIM,
    TheoryNeuralFeatureBuilder,
)
from keysubgraph.models.dual_stse_hard_sgw import (  # noqa: E402
    DualSTSEHardSGWClassifier,
)
from keysubgraph.theory.sgw_core_features import SGWCoreConfig  # noqa: E402
from keysubgraph.training.dual_stse_hard_sgw_trainer import (  # noqa: E402
    load_dual_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--selector-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--gw-max-iter", type=int, default=100)
    parser.add_argument("--gw-sinkhorn-iter", type=int, default=100)
    parser.add_argument("--gw-tolerance", type=float, default=1.0e-7)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _schema_hash(config):
    payload = {
        "artifact": "svg_theory_guided_neural_record",
        "schema_version": 1,
        "node_dim": 15,
        "edge_dim": THEORY_EDGE_FEATURE_DIM,
        "quantile_dim": 16,
        "transition_dim": 18,
        "sgw_core_schema_sha256": config.schema_sha256(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def main():
    args = parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("Stage-1 max-samples must be positive")
    protocol = validate_data_protocol(args.protocol, PROJECT_ROOT)
    protocol_sha256 = file_sha256(args.protocol)
    selector_sha256 = file_sha256(args.selector_checkpoint)
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
    selector = DualSTSEHardSGWClassifier().to(device)
    load_dual_checkpoint(
        args.selector_checkpoint,
        selector,
        device,
        expected_stage="selector_proxy",
        expected_protocol_sha256=protocol_sha256,
    )
    selector.eval()
    core_config = SGWCoreConfig(
        gw_max_iter=args.gw_max_iter,
        gw_sinkhorn_iter=args.gw_sinkhorn_iter,
        gw_tolerance=args.gw_tolerance,
    )
    schema_hash = _schema_hash(core_config)
    builder = TheoryNeuralFeatureBuilder(core_config)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_paths = []
    total = min(len(dataset), args.max_samples or len(dataset))
    with torch.no_grad():
        for index, cpu_batch in enumerate(loader):
            if index >= total:
                break
            cpu_sample = cpu_batch[0].graph
            path = output_dir / theory_neural_filename(cpu_sample.sample_key)
            if path.exists() and not args.overwrite:
                record = load_theory_neural_record(path)
                checks = (
                    record.sample_key == cpu_sample.sample_key,
                    record.split == args.split,
                    record.protocol_sha256 == protocol_sha256,
                    record.selector_checkpoint_sha256 == selector_sha256,
                    record.feature_schema_sha256 == schema_hash,
                    int(record.selection_seed) == int(args.selection_seed),
                )
                if not all(checks):
                    raise ValueError("Stage-1 existing record provenance mismatch")
                feature_paths.append(path)
                print("reused {}/{} {}".format(index + 1, total, record.sample_key), flush=True)
                continue
            batch = cpu_batch.to(device)
            selection = selector.selector(
                batch, selection_mode="learned", random_seed=args.selection_seed
            )
            sample = batch[0].graph
            cropped = tuple(
                window.cropped_graph if window.window_valid else None
                for window in selection.hard_windows[0]
            )
            features = builder.build(cropped, time_values=sample.window_starts)
            window_records = tuple(
                TheoryNeuralWindowRecord(
                    node_features=window.node_features.detach().cpu(),
                    adjacency=window.adjacency.detach().cpu(),
                    edge_features=window.edge_features.detach().cpu(),
                    spectral_quantiles=window.spectral_quantiles.detach().cpu(),
                    communities=window.communities.detach().cpu(),
                    node_ids=window.node_ids,
                    time_start=window.time_start,
                )
                if window is not None
                else None
                for window in features.windows
            )
            record = TheoryNeuralRecord(
                sample_key=sample.sample_key,
                sample_id=sample.sample_id,
                subject_id=sample.subject_id,
                site=sample.site,
                label=int(sample.label),
                split=sample.split,
                windows=window_records,
                window_mask=features.window_mask.detach().cpu(),
                transition_features=features.transition_features.detach().cpu(),
                transition_mask=features.transition_mask.detach().cpu(),
                gw_solver_converged=features.gw_solver_converged,
                protocol_sha256=protocol_sha256,
                selector_checkpoint_sha256=selector_sha256,
                selection_mode="learned",
                selection_seed=args.selection_seed,
                feature_schema_sha256=schema_hash,
            )
            save_theory_neural_record(record, path, overwrite=args.overwrite)
            feature_paths.append(path)
            print(
                "processed {}/{} {} windows={} transitions={}".format(
                    index + 1,
                    total,
                    sample.sample_key,
                    int(features.window_mask.sum()),
                    int(features.transition_mask.sum()),
                ),
                flush=True,
            )
    manifest = write_theory_neural_manifest(
        feature_paths,
        output_dir / "manifest.json",
        PROJECT_ROOT,
        overwrite=args.overwrite,
    )
    print(json.dumps({"manifest": str(manifest), "sample_count": len(feature_paths)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
