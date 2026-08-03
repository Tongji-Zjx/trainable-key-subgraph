"""Evaluate the frozen Author-ST plus critical residual fusion checkpoint."""

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

from keysubgraph.data.data_protocol import protocol_node_name_policy, validate_data_protocol  # noqa: E402
from keysubgraph.data.data_split import file_sha256  # noqa: E402
from keysubgraph.data.graph_dataset import GraphSequenceDataset  # noqa: E402
from keysubgraph.data.multiview_critical import (  # noqa: E402
    MultiViewCriticalDataset,
    PairedMultiViewAuthorDataset,
    create_multiview_author_loader,
)
from keysubgraph.models.multiview_critical import (  # noqa: E402
    MultiViewCriticalClassifier,
    MultiViewCriticalConfig,
    MultiViewCriticalShortTermFusion,
)
from keysubgraph.training.author_short_term_trainer import model_from_author_short_term_checkpoint  # noqa: E402
from keysubgraph.training.dual_sgw_feature_trainer import binary_metrics  # noqa: E402
from keysubgraph.training.multiview_critical_trainer import load_multiview_checkpoint  # noqa: E402


def _trusted_load(path, device):
    try:
        return torch.load(str(Path(path).resolve()), map_location=device, weights_only=False)
    except TypeError:
        return torch.load(str(Path(path).resolve()), map_location=device)


def _graph_dataset(protocol, split):
    paths = protocol["paths"]
    return GraphSequenceDataset(
        PROJECT_ROOT / paths["dataset_root"],
        PROJECT_ROOT / paths["sample_index_csv"],
        PROJECT_ROOT / paths["splits_csv"],
        split,
        protocol["edge_presence_threshold"],
        node_name_policy=protocol_node_name_policy(protocol),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument("--critical-checkpoint", type=Path, required=True)
    parser.add_argument("--short-term-checkpoint", type=Path, required=True)
    parser.add_argument("--fusion-checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()
    device = torch.device(args.device)
    protocol = validate_data_protocol(args.protocol, PROJECT_ROOT)
    protocol_sha256 = file_sha256(args.protocol)
    critical_payload = _trusted_load(args.critical_checkpoint, "cpu")
    critical = MultiViewCriticalClassifier(
        MultiViewCriticalConfig(**critical_payload["model_config"])
    ).to(device)
    load_multiview_checkpoint(args.critical_checkpoint, critical, device)
    short_term, short_payload = model_from_author_short_term_checkpoint(
        args.short_term_checkpoint, device
    )
    fusion_payload = _trusted_load(args.fusion_checkpoint, device)
    expected = {
        "protocol_sha256": protocol_sha256,
        "critical_checkpoint_sha256": file_sha256(args.critical_checkpoint),
        "short_term_checkpoint_sha256": file_sha256(args.short_term_checkpoint),
        "scaler_sha256": file_sha256(args.scaler),
    }
    for name, value in expected.items():
        if fusion_payload.get(name) != value:
            raise ValueError("fusion checkpoint {} mismatch".format(name))
    if short_payload.get("protocol_sha256") != protocol_sha256:
        raise ValueError("short-term checkpoint/protocol mismatch")
    model = MultiViewCriticalShortTermFusion(critical, short_term).to(device)
    model.load_state_dict(fusion_payload["model_state_dict"])
    model.eval()
    critical_dataset = MultiViewCriticalDataset(PROJECT_ROOT, args.manifest, args.scaler)
    if critical_dataset.split != args.split:
        raise ValueError("fusion manifest/split mismatch")
    paired = PairedMultiViewAuthorDataset(
        critical_dataset, _graph_dataset(protocol, args.split)
    )
    loader = create_multiview_author_loader(
        paired, args.batch_size, 0, False, args.num_workers,
        args.device.startswith("cuda"),
    )
    labels, probabilities, predictions = [], [], []
    with torch.no_grad():
        for critical_batch, author_batch in loader:
            critical_batch = critical_batch.to(device)
            author_batch = author_batch.to(device)
            output = model(critical_batch, author_batch)
            current = torch.softmax(output.logits, dim=-1)[:, 1].cpu().tolist()
            current_labels = critical_batch.labels.tolist()
            labels.extend(int(value) for value in current_labels)
            probabilities.extend(float(value) for value in current)
            predictions.extend(
                {
                    "sample_key": key,
                    "label": int(label),
                    "positive_probability": float(probability),
                }
                for key, label, probability in zip(
                    critical_batch.sample_keys, current_labels, current
                )
            )
    threshold = float(fusion_payload["threshold"])
    metrics = binary_metrics(labels, probabilities, threshold)
    result = {
        "schema_version": 1,
        "artifact_type": "multiview_author_short_term_fusion_evaluation",
        "split": args.split,
        "threshold": threshold,
        "metrics": metrics,
        "predictions": predictions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
