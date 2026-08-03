"""Train the final frozen Author-ST anchor plus critical representation residual."""

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
from keysubgraph.training.dual_sgw_feature_trainer import binary_metrics, fit_binary_threshold  # noqa: E402
from keysubgraph.training.multiview_critical_trainer import load_multiview_checkpoint  # noqa: E402
from keysubgraph.training.trainer import class_weights_from_labels, set_reproducible_seed  # noqa: E402


def _atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def _atomic_torch(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, str(temporary))
    os.replace(str(temporary), str(path))


def _run(model, loader, device, weights, optimizer=None, threshold=0.5, gradient_clip=1.0):
    training = optimizer is not None
    model.train(training)
    # Frozen encoders must stay in evaluation mode even while the residual trains.
    model.critical_model.eval()
    model.author_short_term_model.eval()
    labels_all, probabilities_all = [], []
    loss_total, count_total = 0.0, 0
    for critical, author in loader:
        critical, author = critical.to(device), author.to(device)
        labels = critical.labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(critical, author)
            per_sample = torch.nn.functional.cross_entropy(output.logits, labels, reduction="none")
            loss = (per_sample * weights.index_select(0, labels)).mean()
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [item for item in model.parameters() if item.requires_grad], float(gradient_clip)
                )
                optimizer.step()
        probabilities = torch.softmax(output.logits, dim=-1)[:, 1]
        count = int(labels.numel())
        loss_total += float(loss.detach().cpu()) * count
        count_total += count
        labels_all.extend(int(value) for value in labels.detach().cpu().tolist())
        probabilities_all.extend(float(value) for value in probabilities.detach().cpu().tolist())
    metrics = binary_metrics(labels_all, probabilities_all, threshold)
    metrics["loss"] = loss_total / count_total
    return metrics, labels_all, probabilities_all


def _graph_dataset(protocol, split):
    paths = protocol["paths"]
    return GraphSequenceDataset(
        PROJECT_ROOT / paths["dataset_root"],
        PROJECT_ROOT / paths["sample_index_csv"],
        PROJECT_ROOT / paths["splits_csv"], split,
        protocol["edge_presence_threshold"],
        node_name_policy=protocol_node_name_policy(protocol),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument("--critical-checkpoint", type=Path, required=True)
    parser.add_argument("--short-term-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    args = parser.parse_args()

    set_reproducible_seed(args.seed)
    protocol = validate_data_protocol(args.protocol, PROJECT_ROOT)
    protocol_sha256 = file_sha256(args.protocol)
    device = torch.device(args.device)
    try:
        raw = torch.load(str(args.critical_checkpoint), map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(str(args.critical_checkpoint), map_location="cpu")
    critical_model = MultiViewCriticalClassifier(MultiViewCriticalConfig(**raw["model_config"])).to(device)
    load_multiview_checkpoint(args.critical_checkpoint, critical_model, device)
    if raw.get("protocol_sha256") != protocol_sha256:
        raise ValueError("critical checkpoint/protocol mismatch")
    short_term_model, short_payload = model_from_author_short_term_checkpoint(
        args.short_term_checkpoint, device
    )
    if short_payload.get("protocol_sha256") != protocol_sha256:
        raise ValueError("short-term checkpoint/protocol mismatch")
    model = MultiViewCriticalShortTermFusion(critical_model, short_term_model).to(device)
    model.freeze_base_encoders()

    train_critical = MultiViewCriticalDataset(PROJECT_ROOT, args.train_manifest, args.scaler)
    validation_critical = MultiViewCriticalDataset(PROJECT_ROOT, args.validation_manifest, args.scaler)
    train = PairedMultiViewAuthorDataset(train_critical, _graph_dataset(protocol, "train"))
    validation = PairedMultiViewAuthorDataset(validation_critical, _graph_dataset(protocol, "validation"))
    train_loader = create_multiview_author_loader(
        train, args.batch_size, args.seed, True, args.num_workers, args.device.startswith("cuda")
    )
    validation_loader = create_multiview_author_loader(
        validation, args.batch_size, args.seed, False, args.num_workers, args.device.startswith("cuda")
    )
    weights = class_weights_from_labels(train.labels).to(device)
    optimizer = torch.optim.AdamW(
        [item for item in model.parameters() if item.requires_grad],
        lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    history, best, stale = [], float("-inf"), 0
    for epoch in range(1, args.epochs + 1):
        train_metrics, _, _ = _run(
            model, train_loader, device, weights, optimizer,
            gradient_clip=args.gradient_clip,
        )
        validation_raw, labels, probabilities = _run(model, validation_loader, device, weights)
        threshold = fit_binary_threshold(labels, probabilities, "balanced_accuracy")
        validation_metrics = binary_metrics(labels, probabilities, threshold)
        validation_metrics["loss"] = validation_raw["loss"]
        score = validation_metrics["roc_auc"] if validation_metrics["roc_auc"] is not None else float("-inf")
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
            "critical_gate": float(torch.tanh(model.gate).detach().cpu()),
            "short_term_adapter_gate": float(torch.tanh(model.short_term_gate).detach().cpu()),
        }
        history.append(row)
        payload = {
            "schema_version": 1,
            "model_name": model.model_name,
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "threshold": threshold,
            "validation": validation_metrics,
            "critical_checkpoint_sha256": file_sha256(args.critical_checkpoint),
            "short_term_checkpoint_sha256": file_sha256(args.short_term_checkpoint),
            "protocol_sha256": protocol_sha256,
            "train_manifest_sha256": file_sha256(args.train_manifest),
            "validation_manifest_sha256": file_sha256(args.validation_manifest),
            "scaler_sha256": file_sha256(args.scaler),
            "model_seed": int(args.seed),
        }
        _atomic_torch(output_dir / "last_checkpoint.pt", payload)
        _atomic_json(output_dir / "history.json", history)
        if score > best:
            best, stale = float(score), 0
            _atomic_torch(output_dir / "best_checkpoint.pt", payload)
            _atomic_json(output_dir / "best_evaluation.json", {"epoch": epoch, "validation": validation_metrics})
        else:
            stale += 1
        print(
            "epoch {}/{} train_auc={} validation_auc={} validation_ba={:.6f} gate={:.6f}".format(
                epoch, args.epochs, train_metrics["roc_auc"], validation_metrics["roc_auc"],
                validation_metrics["balanced_accuracy"], row["critical_gate"],
            ), flush=True,
        )
        if args.early_stopping_patience and stale >= args.early_stopping_patience:
            break
    print(json.dumps({"best_auc": best, "epochs_completed": len(history)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
