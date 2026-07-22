"""Cache frozen soft-teacher logits and 192-D representations for Stage C."""

from __future__ import absolute_import, division, print_function

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.tg_student_dataset import (  # noqa: E402
    TGTeacherTarget,
    save_tg_teacher_target,
)
from keysubgraph.data.data_protocol import validate_data_protocol  # noqa: E402
from keysubgraph.data.data_split import file_sha256  # noqa: E402
from keysubgraph.data.graph_dataset import GraphSequenceDataset, create_data_loader  # noqa: E402
from keysubgraph.models import TGSoftTeacher, TGSoftTeacherConfig  # noqa: E402
from keysubgraph.training import load_tg_soft_teacher_checkpoint  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_payload(path):
    try:
        return torch.load(str(path.resolve()), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path.resolve()), map_location="cpu")


def main():
    args = parse_args()
    protocol = validate_data_protocol(args.protocol, PROJECT_ROOT)
    if protocol.get("protocol_name") != "strict_theory":
        raise ValueError("teacher targets require the strict_theory protocol")
    checkpoint_payload = _load_payload(args.teacher_checkpoint)
    model = TGSoftTeacher(TGSoftTeacherConfig(**checkpoint_payload["model_config"]))
    protocol_hash = file_sha256(args.protocol)
    teacher_hash = file_sha256(args.teacher_checkpoint)
    load_tg_soft_teacher_checkpoint(
        args.teacher_checkpoint, model, torch.device("cpu"),
        expected_protocol_sha256=protocol_hash,
    )
    paths = protocol["paths"]
    dataset = GraphSequenceDataset(
        PROJECT_ROOT / paths["dataset_root"],
        PROJECT_ROOT / paths["sample_index_csv"],
        PROJECT_ROOT / paths["splits_csv"],
        args.split,
        protocol["edge_presence_threshold"],
    )
    if args.max_samples is not None:
        if args.max_samples < 1:
            raise ValueError("max-samples must be positive")
        dataset.assignments = dataset.assignments[: args.max_samples]
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    loader = create_data_loader(
        dataset, args.batch_size, seed=0, num_workers=args.num_workers,
        shuffle=False, pin_memory=device.type == "cuda",
    )
    model.to(device).eval()
    records = []
    with torch.no_grad():
        for cpu_batch in loader:
            batch = cpu_batch.to(device)
            output = model(batch, return_details=False)
            for index, sample in enumerate(cpu_batch.samples):
                target = TGTeacherTarget(
                    sample.sample_key, sample.sample_id, sample.label, sample.split,
                    output.logits[index].detach().cpu(),
                    output.representation[index].detach().cpu(),
                    protocol_hash, teacher_hash,
                )
                suffix = hashlib.sha256(sample.sample_key.encode("utf-8")).hexdigest()[:12]
                output_path = args.output_dir / (sample.sample_id + "_" + suffix + ".pt")
                save_tg_teacher_target(target, output_path, overwrite=args.overwrite)
                records.append({
                    "sample_key": sample.sample_key,
                    "sample_id": sample.sample_id,
                    "label": sample.label,
                    "split": sample.split,
                    "target_path": str(output_path.resolve()),
                })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest = {
        "schema_version": 1,
        "artifact_type": "tg_sgw_teacher_target_manifest",
        "split": args.split,
        "data_protocol_sha256": protocol_hash,
        "teacher_checkpoint_sha256": teacher_hash,
        "records": records,
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(manifest_path))
    print(json.dumps({
        "sample_count": len(records), "split": args.split,
        "manifest": str(manifest_path.resolve()), "device": str(device),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
