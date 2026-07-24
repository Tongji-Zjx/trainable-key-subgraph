"""Check feature coverage, provenance, branch variance and fusion sensitivity."""

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
from keysubgraph.data.dual_sgw_manifest import read_dual_sgw_manifest  # noqa: E402
from keysubgraph.data.exact_stse_dataset import (  # noqa: E402
    ExactSTSEDataset,
    create_exact_stse_loader,
)
from keysubgraph.models.dual_stse_hard_sgw import (  # noqa: E402
    DualSTSEHardSGWClassifier,
)
from keysubgraph.training.dual_stse_hard_sgw_trainer import (  # noqa: E402
    load_dual_checkpoint,
)


def _summary(values):
    variance = values.detach().to(torch.float64).var(
        dim=0, unbiased=False
    )
    return {
        "sample_count": int(values.shape[0]),
        "dimension": int(values.shape[1]),
        "mean_feature_variance": float(variance.mean()),
        "active_feature_fraction": float(
            (variance > 1.0e-6).double().mean()
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), default="validation"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=64)
    args = parser.parse_args()
    protocol = validate_data_protocol(args.protocol, PROJECT_ROOT)
    protocol_sha = file_sha256(args.protocol)
    manifest, _, lookup = read_dual_sgw_manifest(args.manifest)
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
        args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    device = torch.device(args.device)
    model = DualSTSEHardSGWClassifier().to(device)
    payload = load_dual_checkpoint(
        args.checkpoint,
        model,
        device,
        expected_protocol_sha256=protocol_sha,
    )
    model.eval()
    stse_values = []
    sgw_values = []
    fusion_values = []
    probabilities = []
    zero_sgw_probabilities = []
    processed = 0
    with torch.no_grad():
        for cpu_batch in loader:
            if processed >= args.max_samples:
                break
            samples = tuple(
                cpu_batch.samples[
                    : args.max_samples - processed
                ]
            )
            batch = type(cpu_batch)(samples).to(device)
            features = torch.stack(
                [lookup[key] for key in batch.sample_keys]
            ).to(device)
            output = model(batch, exact_sgw_features=features)
            ablated = model(
                batch, exact_sgw_features=torch.zeros_like(features)
            )
            stse_values.append(output.stse_representation.cpu())
            sgw_values.append(output.sgw_representation.cpu())
            fusion_values.append(output.fusion_representation.cpu())
            probabilities.append(
                torch.softmax(output.fusion_logits, dim=-1)[:, 1].cpu()
            )
            zero_sgw_probabilities.append(
                torch.softmax(ablated.fusion_logits, dim=-1)[:, 1].cpu()
            )
            processed += len(batch)
    stse = torch.cat(stse_values)
    sgw = torch.cat(sgw_values)
    fusion = torch.cat(fusion_values)
    probability = torch.cat(probabilities)
    ablated_probability = torch.cat(zero_sgw_probabilities)
    result = {
        "passed": (
            manifest["split"] == args.split
            and manifest["protocol_sha256"] == protocol_sha
            and processed > 0
        ),
        "stage": payload["stage"],
        "split": args.split,
        "sample_count": processed,
        "stse_representation": _summary(stse),
        "sgw_representation": _summary(sgw),
        "fusion_representation": _summary(fusion),
        "mean_absolute_probability_change_when_sgw_zeroed": float(
            (probability - ablated_probability).abs().mean()
        ),
        "manifest_selection_mode": manifest["selection_mode"],
        "checkpoint_provenance": payload["provenance"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(
            result, handle, ensure_ascii=False, indent=2, sort_keys=True
        )
        handle.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
