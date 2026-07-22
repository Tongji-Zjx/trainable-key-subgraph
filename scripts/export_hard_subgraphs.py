"""Load a frozen checkpoint and export hard key subgraphs for one split."""

from __future__ import absolute_import, print_function

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
from keysubgraph.data.graph_dataset import GraphSequenceDataset  # noqa: E402
from keysubgraph.extraction.hard_extractor import (  # noqa: E402
    HardExtractionConfig,
    HardSubgraphExtractor,
    export_hard_sample,
)
from keysubgraph.models.soft_extractor import (  # noqa: E402
    SoftExtractorConfig,
    SoftGraphClassifier,
)
from keysubgraph.models import TGSoftTeacher, TGSoftTeacherConfig  # noqa: E402
from keysubgraph.models.tg_sgw_types import TG_SGW_MODEL_NAME  # noqa: E402
from keysubgraph.theory import CandidateScoreStandardizer  # noqa: E402
from keysubgraph.training import load_tg_soft_teacher_checkpoint  # noqa: E402
from keysubgraph.training.trainer import load_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROJECT_ROOT / "configs" / "data_protocol_strict_theory.json")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test", "all"), default="validation")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "hard_subgraphs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seeds-per-community", type=int, default=1)
    parser.add_argument("--neighborhood-hops", type=int, default=1)
    parser.add_argument("--max-nodes", type=int, default=20)
    parser.add_argument("--max-edges", type=int, default=80)
    parser.add_argument("--min-nodes", type=int, default=2)
    parser.add_argument("--min-edges", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--overlap-threshold", type=float, default=0.60)
    parser.add_argument("--beta-lambda", type=float, default=0.20)
    parser.add_argument("--beta-gw", type=float, default=0.10)
    parser.add_argument("--beta-overlap", type=float, default=0.10)
    parser.add_argument("--beta-size", type=float, default=0.05)
    parser.add_argument("--prefilter-r1", type=int, default=32)
    parser.add_argument("--prefilter-r2", type=int, default=8)
    parser.add_argument("--max-union-nodes", type=int)
    parser.add_argument("--max-union-edges", type=int)
    parser.add_argument("--candidate-score-scaler", type=Path)
    parser.add_argument("--min-export-gain", type=float, default=0.0)
    parser.add_argument("--eval-gw-entropic-reg", type=float, default=0.01)
    parser.add_argument("--eval-gw-max-iter", type=int, default=100)
    parser.add_argument("--eval-gw-sinkhorn-iter", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("max-samples must be positive")
    protocol = validate_data_protocol(args.protocol, PROJECT_ROOT)
    if protocol.get("protocol_name", "strict_theory") != "strict_theory":
        raise ValueError("strong theory export requires protocol_name=strict_theory")
    if args.split == "all" and protocol.get("experiment_mode") != "all_samples_exploratory":
        raise ValueError("--split all requires an all-sample protocol")
    try:
        checkpoint = torch.load(
            str(args.checkpoint.resolve()), map_location="cpu", weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(str(args.checkpoint.resolve()), map_location="cpu")
    protocol_sha256 = file_sha256(args.protocol)
    checkpoint_protocol_sha256 = (
        checkpoint.get("protocol_sha256")
        if checkpoint.get("model_name") == TG_SGW_MODEL_NAME
        else checkpoint.get("data_protocol_sha256")
    )
    if checkpoint_protocol_sha256 != protocol_sha256:
        raise ValueError("checkpoint does not match the current data protocol")
    if (
        checkpoint.get("model_name") != TG_SGW_MODEL_NAME
        and checkpoint.get("edge_presence_threshold") != protocol["edge_presence_threshold"]
    ):
        raise ValueError("checkpoint and protocol edge thresholds differ")
    if checkpoint.get("model_name") == TG_SGW_MODEL_NAME:
        model = TGSoftTeacher(TGSoftTeacherConfig(**checkpoint["model_config"]))
        load_tg_soft_teacher_checkpoint(
            args.checkpoint,
            model,
            device=torch.device("cpu"),
            expected_protocol_sha256=protocol_sha256,
        )
        if args.candidate_score_scaler is None:
            raise ValueError("TG-SGW hard export requires --candidate-score-scaler")
    else:
        model_config = SoftExtractorConfig(**checkpoint["model_config"])
        if not model_config.theory_alignment_enabled:
            raise ValueError(
                "spectral-GW hard export requires a strong theory-aligned checkpoint"
            )
        model = SoftGraphClassifier(model_config)
        load_checkpoint(args.checkpoint, model, device=torch.device("cpu"))
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model.to(device)
    config = HardExtractionConfig(
        seeds_per_community=args.seeds_per_community,
        neighborhood_hops=args.neighborhood_hops,
        max_nodes=args.max_nodes,
        max_edges=args.max_edges,
        min_nodes=args.min_nodes,
        min_edges=args.min_edges,
        top_k=args.top_k,
        overlap_threshold=args.overlap_threshold,
        beta_lambda=args.beta_lambda,
        beta_gw=args.beta_gw,
        beta_overlap=args.beta_overlap,
        beta_size=args.beta_size,
        prefilter_discriminative_top_r1=args.prefilter_r1,
        prefilter_spectral_top_r2=args.prefilter_r2,
        max_union_nodes=args.max_union_nodes,
        max_union_edges=args.max_union_edges,
        min_export_gain=args.min_export_gain,
        eval_gw_entropic_reg=args.eval_gw_entropic_reg,
        eval_gw_max_iter=args.eval_gw_max_iter,
        eval_gw_sinkhorn_iter=args.eval_gw_sinkhorn_iter,
    )
    candidate_scaler = None
    if args.candidate_score_scaler is not None:
        candidate_scaler = CandidateScoreStandardizer.load(args.candidate_score_scaler)
        if candidate_scaler.data_protocol_sha256 != protocol_sha256:
            raise ValueError("candidate scaler data protocol mismatch")
        if candidate_scaler.teacher_checkpoint_sha256 != file_sha256(args.checkpoint):
            raise ValueError("candidate scaler teacher checkpoint mismatch")
    extractor = HardSubgraphExtractor(model, config, candidate_scaler)
    paths = protocol["paths"]
    dataset = GraphSequenceDataset(
        PROJECT_ROOT / paths["dataset_root"],
        PROJECT_ROOT / paths["sample_index_csv"],
        PROJECT_ROOT / paths["splits_csv"],
        args.split,
        protocol["edge_presence_threshold"],
    )
    count = len(dataset) if args.max_samples is None else min(len(dataset), args.max_samples)
    exports = []
    total_timepoints = 0
    total_subgraphs = 0
    for index in range(count):
        sample = dataset[index].to(device)
        result = extractor.extract_sample(sample)
        output_path = args.output_dir / args.split / (sample.sample_id + ".json")
        export_hard_sample(
            result,
            output_path,
            config,
            args.checkpoint,
            file_sha256(args.protocol),
            overwrite=args.overwrite,
        )
        exports.append(str(output_path.resolve()))
        total_timepoints += len(result.timepoints)
        total_subgraphs += sum(item.num_valid_subgraphs for item in result.timepoints)
    completion_path = args.output_dir / "_completion" / "{}.json".format(args.split)
    completion_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = completion_path.with_suffix(completion_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": 1,
                "complete": True,
                "split": args.split,
                "sample_count": count,
                "timepoint_count": total_timepoints,
                "selected_subgraph_count": total_subgraphs,
                "checkpoint": str(args.checkpoint.resolve()),
                "data_protocol_sha256": file_sha256(args.protocol),
                "theory_alignment": checkpoint.get("theory_alignment"),
                "hard_extraction_config": config.__dict__,
            },
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    os.replace(str(temporary), str(completion_path))
    print(
        json.dumps(
            {
                "split": args.split,
                "device": str(device),
                "sample_count": count,
                "timepoint_count": total_timepoints,
                "selected_subgraph_count": total_subgraphs,
                "completion_marker": str(completion_path.resolve()),
                "exports": exports,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
