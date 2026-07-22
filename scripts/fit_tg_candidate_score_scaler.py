"""Fit the frozen TG-SGW candidate-score scaler on the training split only."""

from __future__ import absolute_import, print_function

import argparse
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.data_protocol import validate_data_protocol  # noqa: E402
from keysubgraph.data.data_split import file_sha256  # noqa: E402
from keysubgraph.data.graph_dataset import GraphSequenceDataset  # noqa: E402
from keysubgraph.extraction import HardExtractionConfig, HardSubgraphExtractor  # noqa: E402
from keysubgraph.models import TGSoftTeacher, TGSoftTeacherConfig  # noqa: E402
from keysubgraph.theory import CandidateScoreStandardizer  # noqa: E402
from keysubgraph.training import load_tg_soft_teacher_checkpoint  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROJECT_ROOT / "configs" / "data_protocol_strict_theory.json")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
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
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("max-samples must be positive")
    protocol = validate_data_protocol(args.protocol, PROJECT_ROOT)
    if protocol.get("protocol_name") != "strict_theory":
        raise ValueError("TG-SGW candidate scaling requires strict_theory")
    protocol_sha256 = file_sha256(args.protocol)
    checkpoint_sha256 = file_sha256(args.checkpoint)
    payload = _load_payload(args.checkpoint)
    model = TGSoftTeacher(TGSoftTeacherConfig(**payload["model_config"]))
    load_tg_soft_teacher_checkpoint(
        args.checkpoint,
        model,
        device=torch.device("cpu"),
        expected_protocol_sha256=protocol_sha256,
    )
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    model.to(device)
    extractor = HardSubgraphExtractor(model, HardExtractionConfig())
    paths = protocol["paths"]
    dataset = GraphSequenceDataset(
        PROJECT_ROOT / paths["dataset_root"],
        PROJECT_ROOT / paths["sample_index_csv"],
        PROJECT_ROOT / paths["splits_csv"],
        "train",
        protocol["edge_presence_threshold"],
    )
    count = len(dataset) if args.max_samples is None else min(len(dataset), args.max_samples)
    candidates = []
    window_count = 0
    for sample_index in range(count):
        sample = dataset[sample_index].to(device)
        for time_index in range(sample.num_timepoints):
            _, _, _, pool = extractor.build_candidate_pool(sample, time_index)
            candidates.extend(pool)
            window_count += 1
    scaler = CandidateScoreStandardizer.fit(
        candidates,
        weights=(0.35, 0.35, 0.20, 0.10),
        fit_split="train",
        data_protocol_sha256=protocol_sha256,
        teacher_checkpoint_sha256=checkpoint_sha256,
    )
    scaler.save(args.output, overwrite=args.overwrite)
    print(json.dumps({
        "fit_split": "train",
        "sample_count": count,
        "window_count": window_count,
        "candidate_count": len(candidates),
        "output": str(args.output.resolve()),
        "data_protocol_sha256": protocol_sha256,
        "teacher_checkpoint_sha256": checkpoint_sha256,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
