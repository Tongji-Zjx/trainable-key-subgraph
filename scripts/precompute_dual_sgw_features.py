"""Export hard graphs and precompute detached exact 34-D SGW features."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.data_protocol import validate_data_protocol  # noqa: E402
from keysubgraph.data.data_split import file_sha256  # noqa: E402
from keysubgraph.data.dual_sgw_manifest import (  # noqa: E402
    dual_feature_filename,
    write_dual_sgw_manifest,
)
from keysubgraph.data.exact_stse_dataset import (  # noqa: E402
    ExactSTSEDataset,
    create_exact_stse_loader,
)
from keysubgraph.models.dual_exact_sgw import (  # noqa: E402
    DualExactSGWBranch,
    DualSGWFeatureRecord,
    save_dual_sgw_feature_record,
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
        choices=("full", "random", "learned"),
        required=True,
    )
    parser.add_argument("--selector-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hard-cache-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args()


def _atomic_hard_record(path, payload, overwrite):
    path = Path(path).resolve()
    if path.exists() and not overwrite:
        raise FileExistsError("dual hard-graph cache already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, str(temporary))
    os.replace(str(temporary), str(path))


def main():
    args = parse_args()
    if args.selection_mode == "learned" and args.selector_checkpoint is None:
        raise ValueError("learned selection requires a selector checkpoint")
    if args.selection_mode != "learned" and args.selector_checkpoint is not None:
        raise ValueError("full/random selection must not load a selector")
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
    exact = DualExactSGWBranch().to(device).eval()
    records = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.hard_cache_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for index, cpu_batch in enumerate(loader):
            if args.max_samples is not None and index >= args.max_samples:
                break
            batch = cpu_batch.to(device)
            selected = model.selector(
                batch,
                selection_mode=args.selection_mode,
                random_seed=args.selection_seed,
            )
            exact_output = exact(batch, selected.hard_windows)
            sample = batch[0]
            filename = dual_feature_filename(sample.sample_key)
            feature_path = args.output_dir / filename
            record = DualSGWFeatureRecord(
                sample_key=sample.sample_key,
                label=sample.label,
                split=sample.split,
                selection_mode=args.selection_mode,
                selection_seed=args.selection_seed,
                core=exact_output.core[0].cpu(),
                variation=exact_output.variation[0].cpu(),
                representation=exact_output.representation[0].cpu(),
                transition_mask=exact_output.transition_mask[0].cpu(),
                protocol_sha256=protocol_sha256,
                selector_checkpoint_sha256=selector_sha256,
            )
            save_dual_sgw_feature_record(
                record, feature_path, overwrite=args.overwrite
            )
            hard_path = args.hard_cache_dir / filename
            _atomic_hard_record(
                hard_path,
                {
                    "schema_version": 1,
                    "artifact_type": "dual_stse_hard_graph_sequence",
                    "sample_key": sample.sample_key,
                    "label": sample.label,
                    "split": sample.split,
                    "selection_mode": args.selection_mode,
                    "selection_seed": args.selection_seed,
                    "protocol_sha256": protocol_sha256,
                    "selector_checkpoint_sha256": selector_sha256,
                    "hard_windows": tuple(
                        window.cropped_graph
                        if window.window_valid
                        else None
                        for window in selected.hard_windows[0]
                    ),
                    "selection_diagnostics": {
                        key: value
                        for key, value in selected.diagnostics.items()
                        if key != "selections"
                    },
                },
                args.overwrite,
            )
            records.append((record, feature_path))
            print(
                "processed {}/{} {}".format(
                    index + 1, len(dataset), sample.sample_key
                ),
                flush=True,
            )
    manifest = write_dual_sgw_manifest(
        records,
        args.output_dir / "manifest.json",
        protocol_sha256,
        selector_sha256,
        args.selection_mode,
        args.selection_seed,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest),
                "sample_count": len(records),
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

