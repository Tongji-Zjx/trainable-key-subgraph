"""Audit sample alignment, graph inputs, and layer-wise representation collapse."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.data_protocol import validate_data_protocol  # noqa: E402
from keysubgraph.data.data_split import file_sha256  # noqa: E402
from keysubgraph.data.graph_dataset import (  # noqa: E402
    GraphSequenceBatch,
    GraphSequenceDataset,
    create_data_loader,
)
from keysubgraph.full_graph_diagnostics import (  # noqa: E402
    FullGraphRepresentationMonitor,
    summarize_full_graph_inputs,
    validate_full_graph_batch_alignment,
)
from keysubgraph.models import (  # noqa: E402
    FullGraphClassifierConfig,
    FullGraphSequenceClassifier,
)
from keysubgraph.training import load_full_graph_classifier_checkpoint  # noqa: E402


class _InMemoryDataset(Dataset):
    def __init__(self, samples, assignments, split):
        self.samples = tuple(samples)
        self.assignments = tuple(assignments)
        self.split = split

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def _trusted_load(path, device):
    try:
        return torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=device)


def _selected_indices(dataset, maximum):
    if maximum <= 0 or maximum >= len(dataset):
        return tuple(range(len(dataset)))
    by_label = {0: [], 1: []}
    for index, assignment in enumerate(dataset.assignments):
        by_label[int(assignment.label)].append((assignment.sample_key, index))
    target_zero = maximum // 2
    target_one = maximum - target_zero
    if len(by_label[0]) < target_zero or len(by_label[1]) < target_one:
        raise ValueError("requested balanced diagnostic cohort is unavailable")
    selected = (
        [index for _, index in sorted(by_label[0])[:target_zero]]
        + [index for _, index in sorted(by_label[1])[:target_one]]
    )
    return tuple(
        sorted(selected, key=lambda index: dataset.assignments[index].sample_key)
    )


def _class_representation_summary(representations, labels):
    values = torch.cat(representations, dim=0).to(torch.float64)
    targets = torch.cat(labels, dim=0).to(torch.long)
    result = {}
    centroids = {}
    for label in (0, 1):
        selected = values[targets == label]
        centroid = selected.mean(dim=0)
        centroids[label] = centroid
        result[str(label)] = {
            "sample_count": int(selected.shape[0]),
            "mean_within_class_squared_distance": float(
                (selected - centroid).square().sum(dim=-1).mean()
            ),
            "centroid_norm": float(torch.linalg.vector_norm(centroid)),
        }
    result["centroid_euclidean_distance"] = float(
        torch.linalg.vector_norm(centroids[0] - centroids[1])
    )
    result["centroid_cosine_similarity"] = float(
        torch.nn.functional.cosine_similarity(
            centroids[0].unsqueeze(0), centroids[1].unsqueeze(0)
        )[0]
    )
    return result


def _collapse_flags(layer_summaries):
    flags = []
    for name, values in layer_summaries.items():
        variance = values.get("mean_feature_variance")
        active = values.get("active_feature_fraction")
        cosine = values.get("mean_pairwise_cosine")
        reasons = []
        if variance is not None and variance < 1.0e-6:
            reasons.append("mean_feature_variance<1e-6")
        if active is not None and active < 0.10:
            reasons.append("active_feature_fraction<0.10")
        if cosine is not None and cosine > 0.995:
            reasons.append("mean_pairwise_cosine>0.995")
        if reasons:
            flags.append({"layer": name, "heuristic_reasons": reasons})
    return flags


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "configs" / "data_protocol_strict_theory.json",
    )
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=64,
        help="balanced deterministic cohort size; <=0 scans the complete split",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    protocol = validate_data_protocol(args.protocol, PROJECT_ROOT)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    header = _trusted_load(args.checkpoint.resolve(), torch.device("cpu"))
    model = FullGraphSequenceClassifier(
        FullGraphClassifierConfig(**header["model_config"])
    )
    payload = load_full_graph_classifier_checkpoint(
        args.checkpoint,
        model,
        device,
        expected_protocol_sha256=file_sha256(args.protocol),
    )
    model.to(device)
    model.eval()

    paths = protocol["paths"]
    source = GraphSequenceDataset(
        PROJECT_ROOT / paths["dataset_root"],
        PROJECT_ROOT / paths["sample_index_csv"],
        PROJECT_ROOT / paths["splits_csv"],
        args.split,
        protocol["edge_presence_threshold"],
    )
    indices = _selected_indices(source, args.max_samples)
    assignments = [source.assignments[index] for index in indices]
    samples = [source[index] for index in indices]
    dataset = _InMemoryDataset(samples, assignments, args.split)
    loader = create_data_loader(
        dataset,
        args.batch_size,
        seed=int(payload["training_config"]["seed"]),
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    assignment_by_key = {
        assignment.sample_key: assignment for assignment in assignments
    }

    first_batch = next(iter(loader))
    permutation_check = {
        "executed": len(first_batch) > 1,
        "maximum_logit_difference": None,
        "maximum_representation_difference": None,
        "passed": None,
    }
    if len(first_batch) > 1:
        reversed_batch = GraphSequenceBatch(tuple(reversed(first_batch.samples)))
        with torch.no_grad():
            original_output = model(first_batch.to(device))
            reversed_output = model(reversed_batch.to(device))
        logit_difference = (
            original_output.logits
            - reversed_output.logits.flip(0)
        ).abs().max()
        representation_difference = (
            original_output.representation
            - reversed_output.representation.flip(0)
        ).abs().max()
        permutation_check.update(
            {
                "maximum_logit_difference": float(logit_difference.cpu()),
                "maximum_representation_difference": float(
                    representation_difference.cpu()
                ),
                "passed": bool(
                    logit_difference <= 1.0e-5
                    and representation_difference <= 1.0e-5
                ),
            }
        )

    monitor = FullGraphRepresentationMonitor(model)
    alignment_records = []
    representations = []
    labels = []
    with torch.no_grad():
        for cpu_batch in loader:
            batch = cpu_batch.to(device)
            output = model(batch)
            monitor.add_model_output(output)
            alignment_records.extend(
                validate_full_graph_batch_alignment(
                    cpu_batch, output, assignment_by_key
                )
            )
            representations.append(output.representation.detach().cpu())
            labels.append(cpu_batch.labels)
    monitor.close()
    layer_summaries = monitor.summary()
    label_counts = {
        str(label): int(sum(sample.label == label for sample in samples))
        for label in (0, 1)
    }
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(payload["epoch"]),
        "best_epoch": int(payload["best_epoch"]),
        "encoder_type": model.config.encoder_type,
        "protocol": str(args.protocol.resolve()),
        "split": args.split,
        "cohort": {
            "selected_sample_count": len(samples),
            "full_split_sample_count": len(source),
            "selection": (
                "complete_split"
                if len(samples) == len(source)
                else "balanced_lexicographic_sample_key"
            ),
            "label_counts": label_counts,
        },
        "alignment": {
            "passed": len(alignment_records) == len(samples),
            "checked_sample_count": len(alignment_records),
            "unique_sample_key_count": len(
                set(record["sample_key"] for record in alignment_records)
            ),
            "batch_permutation_equivariance": permutation_check,
            "records": alignment_records,
        },
        "graph_inputs": summarize_full_graph_inputs(samples),
        "representations": layer_summaries,
        "class_representation": _class_representation_summary(
            representations, labels
        ),
        "collapse_flags": _collapse_flags(layer_summaries),
        "collapse_flag_note": (
            "Thresholds are diagnostic heuristics, not acceptance theorems; "
            "inspect trends across consecutive layers."
        ),
    }
    output = (
        args.output
        if args.output is not None
        else args.checkpoint.resolve().parent
        / "pipeline_diagnostic_{}.json".format(args.split)
    ).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(str(temporary), str(output))

    print("diagnostic output:", output)
    print("alignment passed:", result["alignment"]["passed"])
    print(
        "batch permutation passed:",
        result["alignment"]["batch_permutation_equivariance"]["passed"],
    )
    print(
        "graph validation failures:",
        result["graph_inputs"]["validation_failure_count"],
    )
    print("collapse flags:", len(result["collapse_flags"]))
    for flag in result["collapse_flags"]:
        print("  {}: {}".format(
            flag["layer"], ", ".join(flag["heuristic_reasons"])
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
