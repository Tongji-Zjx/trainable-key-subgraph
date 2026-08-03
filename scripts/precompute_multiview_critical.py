"""Precompute complete S/V/G artifacts for the revised multi-view model."""

from __future__ import absolute_import, division, print_function

import argparse
import hashlib
import json
import sys
import time
import subprocess
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.data_protocol import protocol_node_name_policy, validate_data_protocol  # noqa: E402
from keysubgraph.data.data_split import file_sha256  # noqa: E402
from keysubgraph.data.exact_stse_dataset import ExactSTSEDataset, create_exact_stse_loader  # noqa: E402
from keysubgraph.data.multiview_critical import (  # noqa: E402
    MultiViewCriticalRecord,
    load_multiview_record,
    multiview_filename,
    save_multiview_record,
    write_multiview_manifest,
)
from keysubgraph.features.hard_graph_cache import CachedHardWindow, HardGraphSampleCache  # noqa: E402
from keysubgraph.features.hard_graph_features import HardGraphWindow  # noqa: E402
from keysubgraph.features.multiview_critical import (  # noqa: E402
    MultiViewCriticalFeatureBuilder,
    hard_windows_from_graph_sequence_sample,
)
from keysubgraph.models.dual_stse_hard_sgw import DualSTSEHardSGWClassifier  # noqa: E402
from keysubgraph.theory.sgw_core_features import SGWCoreConfig  # noqa: E402
from keysubgraph.training.dual_stse_hard_sgw_trainer import load_dual_checkpoint  # noqa: E402


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
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--gw-max-iter", type=int, default=100)
    parser.add_argument("--gw-sinkhorn-iter", type=int, default=100)
    parser.add_argument("--object-uot-iterations", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _window_to(window, device):
    return HardGraphWindow(
        adjacency=window.adjacency.detach().to(device),
        communities=window.communities.detach().to(device),
        node_names=tuple(window.node_names),
        time_start=float(window.time_start),
        edge_presence_threshold=float(window.edge_presence_threshold),
        node_ids=tuple(window.node_ids) if window.node_ids is not None else None,
        window_valid=bool(window.window_valid),
    )


def _schema_hash(config, uot_iterations):
    payload = {
        "artifact": "theory_guided_multiview_critical",
        "schema_version": 1,
        "node_dim": 15,
        "edge_dim": 6,
        "spectral_dim": 9,
        "stable_dim": 28,
        "q_dim": 16,
        "delta_q_dim": 18,
        "decomposition": "community_connected_components_v1",
        "correspondence": "signed_diffusion_fgw_plus_uot_v1",
        "uot_iterations": int(uot_iterations),
        "sgw_schema": config.schema_sha256(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL, universal_newlines=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main():
    args = parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("multi-view max-samples must be positive")
    if args.shard_count < 1 or args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise ValueError("multi-view shard index/count are invalid")
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
        dataset, batch_size=1, seed=args.selection_seed,
        num_workers=args.num_workers, shuffle=False, pin_memory=False,
    )
    device = torch.device(args.device)
    selector = DualSTSEHardSGWClassifier().to(device)
    load_dual_checkpoint(
        args.selector_checkpoint, selector, device,
        expected_stage="selector_proxy", expected_protocol_sha256=protocol_sha256,
    )
    selector.eval()
    core = SGWCoreConfig(
        gw_max_iter=args.gw_max_iter,
        gw_sinkhorn_iter=args.gw_sinkhorn_iter,
    )
    schema_hash = _schema_hash(core, args.object_uot_iterations)
    feature_config_json = json.dumps(
        {
            "edge_presence_threshold": float(protocol["edge_presence_threshold"]),
            "object_decomposition": "community_connected_components_v1",
            "spectral": {
                "heat_times": [0.1, 1.0, 10.0],
                "projector_bands": 3,
                "chebyshev_order": 2,
            },
            "signed_diffusion_fgw": {
                "gw_max_iter": int(args.gw_max_iter),
                "gw_sinkhorn_iter": int(args.gw_sinkhorn_iter),
                "feature_weight": 0.25,
            },
            "uot": {
                "iterations": int(args.object_uot_iterations),
                "entropic_reg": 0.1,
                "mass_reg": 1.0,
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    git_commit = _git_commit()
    builder = MultiViewCriticalFeatureBuilder(
        core_config=core, uot_iterations=args.object_uot_iterations
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths_written = []
    selected_indices = [
        index for index in range(len(dataset))
        if index % int(args.shard_count) == int(args.shard_index)
    ]
    if args.max_samples is not None:
        selected_indices = selected_indices[: int(args.max_samples)]
    selected_set = set(selected_indices)
    total = len(selected_indices)
    processed = 0
    for index, cpu_batch in enumerate(loader):
        if processed >= total:
            break
        if index not in selected_set:
            continue
        processed += 1
        cpu_sample = cpu_batch[0].graph
        output_path = output_dir / multiview_filename(cpu_sample.sample_key)
        if output_path.exists() and not args.overwrite:
            record = load_multiview_record(output_path)
            if (
                record.features.sample_key != cpu_sample.sample_key
                or record.split != args.split
                or record.protocol_sha256 != protocol_sha256
                or record.selector_checkpoint_sha256 != selector_sha256
                or record.feature_schema_sha256 != schema_hash
            ):
                raise ValueError("existing multi-view artifact provenance mismatch")
            paths_written.append(output_path)
            print("reused {}/{} {}".format(processed, total, cpu_sample.sample_key), flush=True)
            continue
        with torch.no_grad():
            selection = selector.selector(
                cpu_batch.to(device), selection_mode="learned", random_seed=args.selection_seed
            )
        cropped = tuple(
            _window_to(item.cropped_graph, device) if item.window_valid else None
            for item in selection.hard_windows[0]
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        cached = tuple(
            CachedHardWindow(item, None, ()) if item is not None else None
            for item in cropped
        )
        cache = HardGraphSampleCache(
            sample_key=cpu_sample.sample_key,
            sample_id=cpu_sample.sample_id,
            label=int(cpu_sample.label),
            split=args.split,
            windows=cached,
            time_values=tuple(float(value) for value in cpu_sample.window_starts),
            time_mask=tuple(item is not None for item in cached),
            eligible_for_stage_c=sum(item is not None for item in cached) >= 2,
            exclusion_reason=None,
            data_protocol_sha256=protocol_sha256,
            teacher_checkpoint_sha256=selector_sha256,
        )
        full_windows = tuple(
            _window_to(item, device)
            for item in hard_windows_from_graph_sequence_sample(cpu_sample)
        )
        features = builder.build(cache, full_graph_windows=full_windows)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_memory_mib = float(torch.cuda.max_memory_allocated(device)) / (1024.0 ** 2)
        else:
            peak_memory_mib = 0.0
        elapsed = time.perf_counter() - started
        features = features.to(torch.device("cpu"))
        record = MultiViewCriticalRecord(
            sample_id=cpu_sample.sample_id,
            subject_id=cpu_sample.subject_id,
            site=cpu_sample.site,
            split=args.split,
            features=features,
            protocol_sha256=protocol_sha256,
            selector_checkpoint_sha256=selector_sha256,
            feature_schema_sha256=schema_hash,
            precompute_seconds=elapsed,
            peak_memory_mib=peak_memory_mib,
            feature_config_json=feature_config_json,
            git_commit=git_commit,
        )
        save_multiview_record(record, output_path, overwrite=args.overwrite)
        paths_written.append(output_path)
        object_count = sum(
            len(item.objects) for item in features.hard_windows if item is not None
        )
        print(
            "processed {}/{} {} windows={} transitions={} objects={}".format(
                processed, total, cpu_sample.sample_key,
                int(features.window_mask.sum()), int(features.transition_mask.sum()), object_count,
            ),
            flush=True,
        )
    manifest = write_multiview_manifest(
        paths_written, output_dir / "manifest.json", PROJECT_ROOT, overwrite=args.overwrite
    )
    print(json.dumps({
        "manifest": str(manifest),
        "sample_count": len(paths_written),
        "shard_count": int(args.shard_count),
        "shard_index": int(args.shard_index),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
