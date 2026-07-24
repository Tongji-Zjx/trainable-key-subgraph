"""Diagnose Exact-STSE alignment, permutation consistency and collapse."""

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

from keysubgraph.data.data_protocol import validate_data_protocol  # noqa: E402
from keysubgraph.data.data_split import file_sha256  # noqa: E402
from keysubgraph.data.exact_stse_dataset import (  # noqa: E402
    ExactSTSEBatch,
    ExactSTSEDataset,
    create_exact_stse_loader,
)
from keysubgraph.hard_stse_diagnostics import (  # noqa: E402
    representation_summary,
)
from keysubgraph.models.exact_stse import ExactSTSEClassifier  # noqa: E402
from keysubgraph.training import (  # noqa: E402
    exact_stse_config_from_dict,
    load_exact_stse_checkpoint,
)

PAIRWISE_COSINE_MAX_ROWS = 2048


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT
        / "configs"
        / "data_protocol_exact_stse_coords.json",
    )
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), default="validation"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _trusted_load(path, device):
    try:
        return torch.load(
            str(path.resolve()), map_location=device, weights_only=False
        )
    except TypeError:
        return torch.load(str(path.resolve()), map_location=device)


def _scalable_representation_summary(values):
    """Use every row for variance but bound the quadratic cosine diagnostic."""
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError("representation diagnostics require shape [B,D]")
    detached = values.detach().to(dtype=torch.float64, device="cpu")
    variance = detached.var(dim=0, unbiased=False)
    norms = detached.norm(dim=-1)
    if detached.shape[0] > 1:
        if detached.shape[0] > PAIRWISE_COSINE_MAX_ROWS:
            indices = torch.linspace(
                0,
                detached.shape[0] - 1,
                steps=PAIRWISE_COSINE_MAX_ROWS,
                dtype=torch.float64,
            ).round().to(dtype=torch.long)
            cosine_values = detached.index_select(0, indices)
        else:
            cosine_values = detached
        cosine_norms = cosine_values.norm(dim=-1)
        normalized = cosine_values / cosine_norms[:, None].clamp_min(
            1.0e-12
        )
        cosine = normalized.matmul(normalized.transpose(0, 1))
        upper = torch.triu(
            torch.ones_like(cosine, dtype=torch.bool), diagonal=1
        )
        mean_cosine = float(cosine[upper].mean())
    else:
        cosine_values = detached
        mean_cosine = None
    return {
        "sample_count": int(detached.shape[0]),
        "dimension": int(detached.shape[1]),
        "mean_feature_variance": float(variance.mean()),
        "maximum_feature_variance": float(variance.max()),
        "active_feature_fraction": float(
            (variance > 1.0e-6).double().mean()
        ),
        "mean_pairwise_cosine": mean_cosine,
        "pairwise_cosine_row_count": int(cosine_values.shape[0]),
        "representation_norm": {
            "mean": float(norms.mean()),
            "standard_deviation": float(
                norms.var(unbiased=False).clamp_min(0.0).sqrt()
            ),
        },
    }


def main():
    args = parse_args()
    if args.max_samples < 2:
        raise ValueError("diagnostic needs at least two samples")
    protocol = validate_data_protocol(args.protocol, PROJECT_ROOT)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    payload = _trusted_load(args.checkpoint, device)
    model = ExactSTSEClassifier(
        exact_stse_config_from_dict(payload["model_config"])
    ).to(device)
    load_exact_stse_checkpoint(
        args.checkpoint,
        model,
        device,
        expected_protocol_sha256=file_sha256(args.protocol),
    )
    model.eval()
    paths = protocol["paths"]
    dataset = ExactSTSEDataset(
        PROJECT_ROOT / paths["dataset_root"],
        PROJECT_ROOT / paths["sample_index_csv"],
        PROJECT_ROOT / paths["splits_csv"],
        args.split,
        protocol["edge_presence_threshold"],
        require_coordinates=model.config.use_coordinates,
    )
    loader = create_exact_stse_loader(
        dataset,
        args.batch_size,
        seed=int(payload["training_config"]["seed"]),
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    projected, encoded, windows, subjects, logits, probabilities = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    sample_keys = []
    labels = []
    sequence_lengths = []
    permutation_passed = True
    maximum_permutation_difference = 0.0
    processed = 0
    with torch.no_grad():
        for cpu_batch in loader:
            if processed >= args.max_samples:
                break
            remaining = args.max_samples - processed
            samples = tuple(cpu_batch.samples[:remaining])
            batch = ExactSTSEBatch(samples).to(device)
            output = model(batch)
            if len(batch) > 1:
                reverse = ExactSTSEBatch(tuple(reversed(batch.samples)))
                reverse_logits = model(reverse).logits.flip(0)
                difference = float(
                    (output.logits - reverse_logits).abs().max().cpu()
                )
                maximum_permutation_difference = max(
                    maximum_permutation_difference, difference
                )
                permutation_passed = (
                    permutation_passed and difference <= 1.0e-6
                )
            for sample_encodings in output.window_encodings:
                for encoding in sample_encodings:
                    projected.append(encoding.projected_nodes.detach().cpu())
                    encoded.append(encoding.encoded_nodes.detach().cpu())
            windows.extend(
                item.detach().cpu() for item in output.window_embeddings
            )
            subjects.append(output.subject_embedding.detach().cpu())
            logits.append(output.logits.detach().cpu())
            probabilities.append(
                torch.softmax(output.logits, dim=-1)[:, 1].detach().cpu()
            )
            sample_keys.extend(batch.sample_keys)
            labels.extend(int(value) for value in batch.labels.tolist())
            sequence_lengths.extend(output.diagnostics["sequence_lengths"])
            processed += len(batch)
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "variant": model.config.model_variant,
        "split": args.split,
        "sample_count": len(sample_keys),
        "alignment": {
            "passed": len(set(sample_keys)) == len(sample_keys),
            "sample_keys": sample_keys,
            "labels": labels,
            "sequence_lengths": sequence_lengths,
        },
        "batch_permutation": {
            "passed": permutation_passed,
            "maximum_absolute_logit_difference": maximum_permutation_difference,
        },
        "representations": {
            "projected_nodes": _scalable_representation_summary(
                torch.cat(projected, dim=0)
            ),
            "encoded_nodes": _scalable_representation_summary(
                torch.cat(encoded, dim=0)
            ),
            "window_embedding": representation_summary(
                torch.cat(windows, dim=0)
            ),
            "subject_embedding": representation_summary(
                torch.cat(subjects, dim=0)
            ),
            "logits": representation_summary(torch.cat(logits, dim=0)),
            "positive_probability": representation_summary(
                torch.cat(probabilities, dim=0)[:, None]
            ),
        },
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print("diagnostic output: {}".format(output_path))
    print("alignment passed: {}".format(result["alignment"]["passed"]))
    print(
        "batch permutation passed: {}".format(
            result["batch_permutation"]["passed"]
        )
    )
    for name, summary in result["representations"].items():
        print(
            "{} variance={:.6e} active={:.4f} cosine={}".format(
                name,
                summary["mean_feature_variance"],
                summary["active_feature_fraction"],
                summary["mean_pairwise_cosine"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
