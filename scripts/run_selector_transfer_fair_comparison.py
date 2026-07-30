"""Extract all selector controls in one pass and run a fair frozen probe."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.analysis.selector_transfer_probe import (  # noqa: E402
    compare_selector_transfer_feature_sets,
    write_selector_transfer_probe_artifacts,
)
from keysubgraph.data.data_protocol import (  # noqa: E402
    protocol_node_name_policy,
    validate_data_protocol,
)
from keysubgraph.data.data_split import file_sha256  # noqa: E402
from keysubgraph.data.exact_stse_dataset import (  # noqa: E402
    ExactSTSEDataset,
    create_exact_stse_loader,
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
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument(
        "--learned-condition",
        action="append",
        nargs=2,
        metavar=("NAME", "CHECKPOINT"),
        default=[],
    )
    parser.add_argument("--include-random", action="store_true")
    parser.add_argument("--include-full", action="store_true")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--probe-seed", type=int, default=42)
    parser.add_argument("--max-samples-per-split", type=int)
    return parser.parse_args()


def _atomic_torch_save(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, str(temporary))
    os.replace(str(temporary), str(path))


def _make_dataset(protocol, split):
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


def _extract_split(
    protocol,
    split,
    condition_models,
    device,
    workers,
    selection_seed,
    max_samples,
):
    dataset = _make_dataset(protocol, split)
    loader = create_exact_stse_loader(
        dataset,
        batch_size=1,
        seed=selection_seed,
        num_workers=workers,
        shuffle=False,
        pin_memory=False,
    )
    builder = SVHardSampleFeatureBuilder()
    result = {
        name: {
            "split": split,
            "protocol_sha256": provenance["protocol_sha256"],
            "sample_keys": [],
            "labels": [],
            "sites": [],
            "features": [],
        }
        for name, _, _, provenance in condition_models
    }
    expected = min(
        len(dataset),
        max_samples if max_samples is not None else len(dataset),
    )
    with torch.no_grad():
        for index, cpu_batch in enumerate(loader):
            if max_samples is not None and index >= max_samples:
                break
            batch = cpu_batch.to(device)
            graph = batch[0].graph
            for name, model, selection_mode, _ in condition_models:
                selection = model.selector(
                    batch,
                    selection_mode=selection_mode,
                    random_seed=selection_seed,
                )
                cropped = tuple(
                    window.cropped_graph if window.window_valid else None
                    for window in selection.hard_windows[0]
                )
                features = builder.build(cropped)
                combined = torch.cat(
                    (features.static_features, features.variation), dim=0
                )
                result[name]["sample_keys"].append(graph.sample_key)
                result[name]["labels"].append(int(graph.label))
                result[name]["sites"].append(str(graph.site))
                result[name]["features"].append(
                    combined.detach().cpu().numpy()
                )
            print(
                "{} {}/{} {}".format(
                    split, index + 1, expected, graph.sample_key
                ),
                flush=True,
            )
    for name in result:
        result[name]["features"] = np.asarray(
            result[name]["features"], dtype=np.float64
        )
        result[name]["labels"] = np.asarray(
            result[name]["labels"], dtype=int
        )
    return result


def main():
    args = parse_args()
    if args.max_samples_per_split is not None:
        if args.max_samples_per_split < 4:
            raise ValueError("comparison max samples must be at least four")
    names = [name for name, _ in args.learned_condition]
    if args.include_random:
        names.append("random")
    if args.include_full:
        names.append("full")
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError(
            "comparison requires at least two uniquely named conditions"
        )
    if args.reference not in names:
        raise ValueError("comparison reference condition is absent")
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("selector-transfer comparison output exists")
    protocol = validate_data_protocol(args.protocol, PROJECT_ROOT)
    protocol_sha256 = file_sha256(args.protocol)
    device = torch.device(args.device)
    condition_models = []
    for name, checkpoint in args.learned_condition:
        checkpoint_path = Path(checkpoint).resolve()
        model = DualSTSEHardSGWClassifier().to(device)
        load_dual_checkpoint(
            checkpoint_path,
            model,
            device,
            expected_stage="selector_proxy",
            expected_protocol_sha256=protocol_sha256,
        )
        model.eval()
        condition_models.append(
            (
                name,
                model,
                "learned",
                {
                    "protocol_sha256": protocol_sha256,
                    "selection_mode": "learned",
                    "selection_seed": int(args.selection_seed),
                    "selector_checkpoint": str(checkpoint_path),
                    "selector_checkpoint_sha256": file_sha256(
                        checkpoint_path
                    ),
                },
            )
        )
    control_model = DualSTSEHardSGWClassifier().to(device)
    control_model.eval()
    if args.include_random:
        condition_models.append(
            (
                "random",
                control_model,
                "random",
                {
                    "protocol_sha256": protocol_sha256,
                    "selection_mode": "random",
                    "selection_seed": int(args.selection_seed),
                    "selector_checkpoint": None,
                    "selector_checkpoint_sha256": "none",
                },
            )
        )
    if args.include_full:
        condition_models.append(
            (
                "full",
                control_model,
                "full",
                {
                    "protocol_sha256": protocol_sha256,
                    "selection_mode": "full",
                    "selection_seed": int(args.selection_seed),
                    "selector_checkpoint": None,
                    "selector_checkpoint_sha256": "none",
                },
            )
        )
    order = [args.reference] + [
        name for name in names if name != args.reference
    ]
    by_name = {item[0]: item for item in condition_models}
    condition_models = [by_name[name] for name in order]
    train = _extract_split(
        protocol,
        "train",
        condition_models,
        device,
        args.num_workers,
        args.selection_seed,
        args.max_samples_per_split,
    )
    validation = _extract_split(
        protocol,
        "validation",
        condition_models,
        device,
        args.num_workers,
        args.selection_seed,
        args.max_samples_per_split,
    )
    conditions = [
        (name, train[name], validation[name], provenance)
        for name, _, _, provenance in condition_models
    ]
    payload = compare_selector_transfer_feature_sets(
        conditions, seed=args.probe_seed
    )
    output_dir.mkdir(parents=True)
    cache_payload = {
        "schema_version": 1,
        "artifact_type": "selector_transfer_compact_feature_cache",
        "test_used": False,
        "conditions": {},
    }
    for name, _, _, provenance in condition_models:
        cache_payload["conditions"][name] = {
            "provenance": provenance,
            "train": {
                "sample_keys": train[name]["sample_keys"],
                "labels": torch.as_tensor(train[name]["labels"]),
                "sites": train[name]["sites"],
                "features": torch.as_tensor(train[name]["features"]),
            },
            "validation": {
                "sample_keys": validation[name]["sample_keys"],
                "labels": torch.as_tensor(validation[name]["labels"]),
                "sites": validation[name]["sites"],
                "features": torch.as_tensor(
                    validation[name]["features"]
                ),
            },
        }
    cache_path = output_dir / "feature_cache.pt"
    _atomic_torch_save(cache_path, cache_payload)
    artifact_paths = write_selector_transfer_probe_artifacts(
        payload, output_dir / "probe"
    )
    print(
        json.dumps(
            {
                "feature_cache": str(cache_path),
                **artifact_paths
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    for row in payload["rows"]:
        print(
            "{name}: AUC={roc_auc:.6f} delta={delta_auc_vs_reference:+.6f}"
            .format(**row),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
