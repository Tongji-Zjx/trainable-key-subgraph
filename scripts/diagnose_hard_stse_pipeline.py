"""Diagnose alignment, hard-graph invariants and representation collapse."""

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
from keysubgraph.data.graph_dataset import (  # noqa: E402
    GraphSequenceDataset,
    create_data_loader,
)
from keysubgraph.hard_stse_diagnostics import (  # noqa: E402
    audit_hard_stse_output,
    representation_summary,
)
from keysubgraph.models.hard_stse_temporal_sgw import (  # noqa: E402
    HardSTSETemporalSGWClassifier,
)
from keysubgraph.training import (  # noqa: E402
    hard_stse_config_from_dict,
    load_hard_stse_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "configs" / "data_protocol_strict_theory.json",
    )
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), default="validation"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _trusted_load(path, device):
    try:
        return torch.load(str(path.resolve()), map_location=device, weights_only=False)
    except TypeError:
        return torch.load(str(path.resolve()), map_location=device)


def main():
    args = parse_args()
    if args.max_samples < 1:
        raise ValueError("max samples must be positive")
    protocol = validate_data_protocol(args.protocol, PROJECT_ROOT)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    payload = _trusted_load(args.checkpoint, device)
    model = HardSTSETemporalSGWClassifier(
        hard_stse_config_from_dict(payload["model_config"])
    ).to(device)
    load_hard_stse_checkpoint(
        args.checkpoint,
        model,
        device,
        expected_protocol_sha256=file_sha256(args.protocol),
    )
    model.eval()
    paths = protocol["paths"]
    dataset = GraphSequenceDataset(
        PROJECT_ROOT / paths["dataset_root"],
        PROJECT_ROOT / paths["sample_index_csv"],
        PROJECT_ROOT / paths["splits_csv"],
        args.split,
        protocol["edge_presence_threshold"],
    )
    loader = create_data_loader(
        dataset,
        args.batch_size,
        seed=int(payload["training_config"]["seed"]),
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    neural, final, fusion_logits = [], [], []
    theory = []
    audits = []
    processed = 0
    with torch.no_grad():
        for cpu_batch in loader:
            if processed >= args.max_samples:
                break
            batch = cpu_batch.to(device)
            output = model(
                batch,
                epoch=int(payload["best_epoch"]),
                random_selection_seed=int(payload["training_config"]["seed"]),
                compute_theory_proxies=False,
            )
            audits.append(audit_hard_stse_output(batch, output))
            neural.append(output.neural_representation.detach().cpu())
            final.append(output.final_representation.detach().cpu())
            fusion_logits.append(output.fusion_logits.detach().cpu())
            if output.theory_representation is not None:
                theory.append(output.theory_representation.detach().cpu())
            processed += len(batch)
    failures = [
        failure
        for audit in audits
        for failure in audit["failures"]
    ]
    inventory = [
        item
        for audit in audits
        for item in audit["sample_inventory"]
    ]
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "variant": model.config.variant,
        "split": args.split,
        "sample_count": len(inventory),
        "alignment": {
            "passed": len({
                item["sample_key"] for item in inventory
            }) == len(inventory),
            "sample_inventory": inventory,
        },
        "hard_graph_invariants": {
            "passed": not failures,
            "failure_count": len(failures),
            "failures": failures,
            "total_window_count": sum(
                audit["total_window_count"] for audit in audits
            ),
            "valid_window_count": sum(
                audit["valid_window_count"] for audit in audits
            ),
        },
        "representations": {
            "neural": representation_summary(torch.cat(neural, dim=0)),
            "final": representation_summary(torch.cat(final, dim=0)),
            "fusion_logits": representation_summary(
                torch.cat(fusion_logits, dim=0)
            ),
            "theory": (
                representation_summary(torch.cat(theory, dim=0))
                if theory
                else None
            ),
        },
    }
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print("diagnostic output: {}".format(args.output))
    print("alignment passed: {}".format(result["alignment"]["passed"]))
    print("hard graph passed: {}".format(
        result["hard_graph_invariants"]["passed"]
    ))
    for name, summary in result["representations"].items():
        if summary is not None:
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
