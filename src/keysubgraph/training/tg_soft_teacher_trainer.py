"""Epoch runner for the TG-SGW soft teacher; checkpointing is kept separate."""

from __future__ import absolute_import, division, print_function

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import torch

from keysubgraph.models.tg_soft_teacher_loss import (
    TGSoftTeacherLossConfig,
    compute_tg_soft_teacher_loss,
)
from keysubgraph.training.trainer import set_reproducible_seed

from .tg_soft_teacher_checkpoint import (
    load_tg_soft_teacher_checkpoint,
    save_tg_soft_teacher_checkpoint,
)


@dataclass(frozen=True)
class TGSoftTeacherTrainingConfig:
    epochs: int = 150
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    gradient_clip_norm: float = 5.0
    early_stopping_patience: int = 20
    selection_metric: str = "balanced_accuracy"
    seed: int = 42
    max_train_batches: Optional[int] = None
    max_validation_batches: Optional[int] = None

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("invalid TG soft-teacher optimization configuration")
        if self.gradient_clip_norm <= 0.0 or self.early_stopping_patience < 1:
            raise ValueError("gradient clip and patience must be positive")
        if self.selection_metric not in ("balanced_accuracy", "roc_auc", "loss"):
            raise ValueError("unsupported TG soft-teacher selection metric")


def _class_weights(labels: Sequence[int]) -> torch.Tensor:
    counts = [sum(1 for value in labels if int(value) == label) for label in (0, 1)]
    if min(counts) < 1:
        raise ValueError("TG soft-teacher training requires both classes")
    total = float(sum(counts))
    return torch.tensor([total / (2.0 * count) for count in counts], dtype=torch.float32)


def _selection_value(metrics: Dict[str, Any], metric: str) -> float:
    value = metrics.get(metric)
    if value is None:
        return float("-inf")
    return -float(value) if metric == "loss" else float(value)


def _atomic_json(path: Path, payload) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def train_tg_soft_teacher(
    model,
    train_loader,
    validation_loader,
    train_labels: Sequence[int],
    device: torch.device,
    loss_config: TGSoftTeacherLossConfig,
    training_config: TGSoftTeacherTrainingConfig,
    output_dir: Path,
    protocol_path: Path,
    protocol_sha256: str,
    resume_checkpoint: Optional[Path] = None,
) -> Dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    history_path = output_dir / "history.json"
    if history_path.exists() and resume_checkpoint is None:
        raise FileExistsError("TG soft-teacher output exists; resume or choose a new directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    set_reproducible_seed(training_config.seed)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    weights = _class_weights(train_labels).to(device)
    history = []
    best_epoch = 0
    best_value = float("-inf")
    start_epoch = 1
    if resume_checkpoint is not None:
        checkpoint = load_tg_soft_teacher_checkpoint(
            resume_checkpoint,
            model,
            device,
            optimizer=optimizer,
            expected_protocol_sha256=protocol_sha256,
            data_loader=train_loader,
            restore_rng=True,
        )
        if checkpoint.get("loss_config") != asdict(loss_config):
            raise ValueError("resume loss configuration differs from checkpoint")
        history = list(checkpoint["history"])
        best_epoch = int(checkpoint["best_epoch"])
        best_value = float(checkpoint["best_selection_value"])
        start_epoch = int(checkpoint["epoch"]) + 1
    epochs_without_improvement = max(0, start_epoch - 1 - best_epoch)
    for epoch in range(start_epoch, training_config.epochs + 1):
        train_metrics = run_tg_soft_teacher_epoch(
            model,
            train_loader,
            device,
            epoch,
            loss_config,
            optimizer=optimizer,
            class_weights=weights,
            gradient_clip_norm=training_config.gradient_clip_norm,
            max_batches=training_config.max_train_batches,
        )
        validation_metrics = run_tg_soft_teacher_epoch(
            model,
            validation_loader,
            device,
            epoch,
            loss_config,
            optimizer=None,
            class_weights=None,
            gradient_clip_norm=training_config.gradient_clip_norm,
            max_batches=training_config.max_validation_batches,
        )
        record = {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        history.append(record)
        value = _selection_value(validation_metrics, training_config.selection_metric)
        # A one-batch smoke validation may contain one class, making AUROC
        # unavailable. The first epoch must still create a usable checkpoint.
        improved = best_epoch == 0 or value > best_value
        if improved:
            best_value = value
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        checkpoint_kwargs = dict(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            history=history,
            loss_config=loss_config,
            training_config=training_config,
            protocol_path=protocol_path,
            protocol_sha256=protocol_sha256,
            best_epoch=best_epoch,
            best_selection_value=best_value,
            data_loader=train_loader,
        )
        save_tg_soft_teacher_checkpoint(output_dir / "last_checkpoint.pt", **checkpoint_kwargs)
        if improved:
            save_tg_soft_teacher_checkpoint(output_dir / "best_checkpoint.pt", **checkpoint_kwargs)
        _atomic_json(history_path, history)
        print(
            "epoch {}/{} train_loss={:.6f} validation_loss={:.6f} "
            "validation_balanced_accuracy={:.6f} validation_auc={} "
            "train_node_score={:.4f}+/-{:.4f} "
            "train_edge_score={:.4f}+/-{:.4f}".format(
                epoch,
                training_config.epochs,
                train_metrics["loss"],
                validation_metrics["loss"],
                validation_metrics["balanced_accuracy"],
                validation_metrics["roc_auc"],
                train_metrics["node_score_mean"],
                train_metrics["node_score_std"],
                train_metrics["edge_score_mean"],
                train_metrics["edge_score_std"],
            ),
            flush=True,
        )
        if epochs_without_improvement >= training_config.early_stopping_patience:
            break
    return {
        "best_checkpoint": output_dir / "best_checkpoint.pt",
        "last_checkpoint": output_dir / "last_checkpoint.pt",
        "history": history_path,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "selection_metric": training_config.selection_metric,
    }


def run_tg_soft_teacher_epoch(
    model,
    data_loader: Iterable,
    device: torch.device,
    epoch: int,
    loss_config: TGSoftTeacherLossConfig,
    optimizer: Optional[torch.optim.Optimizer] = None,
    class_weights: Optional[torch.Tensor] = None,
    gradient_clip_norm: float = 5.0,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    totals = {
        "loss": 0.0,
        "classification_loss": 0.0,
        "budget_loss": 0.0,
        "laplacian_fidelity": 0.0,
        "gw_identity_upper_bound": 0.0,
        "laplacian_operator_error": 0.0,
    }
    sample_count = 0
    correct = 0
    gradient_norms = []
    all_labels = []
    all_predictions = []
    all_probabilities = []
    score_accumulators = {
        "node": {
            "count": 0, "total": 0.0, "squared_total": 0.0,
            "minimum": float("inf"), "maximum": float("-inf"),
            "above_half_count": 0.0, "entropy_total": 0.0,
        },
        "edge": {
            "count": 0, "total": 0.0, "squared_total": 0.0,
            "minimum": float("inf"), "maximum": float("-inf"),
            "above_half_count": 0.0, "entropy_total": 0.0,
        },
    }
    for batch_index, cpu_batch in enumerate(data_loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = cpu_batch.to(device)
        labels = batch.labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(batch, return_details=False)
            loss = compute_tg_soft_teacher_loss(
                output, labels, epoch, loss_config, class_weights=class_weights
            )
            if training:
                loss.total.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                gradient_norms.append(float(norm.detach().cpu()))
                optimizer.step()
        count = int(labels.numel())
        sample_count += count
        correct += int((output.logits.argmax(dim=-1) == labels).sum().detach().cpu())
        probabilities = torch.softmax(output.logits, dim=-1)[:, 1]
        all_labels.extend(int(value) for value in labels.detach().cpu().tolist())
        all_predictions.extend(int(value) for value in output.logits.argmax(dim=-1).detach().cpu().tolist())
        all_probabilities.extend(float(value) for value in probabilities.detach().cpu().tolist())
        totals["loss"] += float(loss.total.detach().cpu()) * count
        totals["classification_loss"] += float(loss.classification.detach().cpu()) * count
        totals["budget_loss"] += float(loss.budget.detach().cpu()) * count
        totals["laplacian_fidelity"] += float(loss.laplacian_fidelity.detach().cpu()) * count
        totals["gw_identity_upper_bound"] += float(loss.gw_identity_upper_bound.detach().cpu()) * count
        totals["laplacian_operator_error"] += float(output.laplacian_operator_norms.mean().detach().cpu()) * count
        named_statistics = (
            ("node", output.node_score_statistics),
            ("edge", output.edge_score_statistics),
        )
        # Pack all diagnostic scalars into one transfer. Calling .cpu() for
        # every scalar separately forces many CUDA synchronizations and is
        # especially costly for batch_size=1.
        packed = []
        for _, statistics in named_statistics:
            dtype = statistics.total.dtype
            packed.extend((
                statistics.total,
                statistics.squared_total,
                statistics.minimum,
                statistics.maximum,
                statistics.above_half_count.to(dtype=dtype),
                statistics.entropy_total,
            ))
        packed_values = torch.stack(packed).detach().cpu().tolist()
        for statistics_index, (name, statistics) in enumerate(named_statistics):
            accumulator = score_accumulators[name]
            accumulator["count"] += int(statistics.count)
            offset = 6 * statistics_index
            total, squared_total, minimum, maximum, above_half, entropy = (
                packed_values[offset : offset + 6]
            )
            accumulator["total"] += float(total)
            accumulator["squared_total"] += float(squared_total)
            accumulator["minimum"] = min(
                accumulator["minimum"], float(minimum)
            )
            accumulator["maximum"] = max(
                accumulator["maximum"], float(maximum)
            )
            accumulator["above_half_count"] += float(above_half)
            accumulator["entropy_total"] += float(entropy)
    if sample_count == 0:
        raise ValueError("TG soft-teacher epoch processed no samples")
    metrics = {name: value / sample_count for name, value in totals.items()}
    metrics.update(
        {
            "sample_count": sample_count,
            "accuracy": correct / float(sample_count),
            "mean_gradient_norm": (
                sum(gradient_norms) / len(gradient_norms) if gradient_norms else None
            ),
            "effective_laplacian_weight": _effective_weight(
                loss_config.laplacian_max_weight, epoch, loss_config.theory_warmup_epochs
            ),
            "effective_gw_weight": _effective_weight(
                loss_config.gw_identity_max_weight, epoch, loss_config.theory_warmup_epochs
            ),
        }
    )
    metrics.update(_classification_metrics(all_labels, all_predictions, all_probabilities))
    for name, accumulator in score_accumulators.items():
        score_count = accumulator["count"]
        if score_count < 1:
            raise ValueError("TG soft-teacher epoch has no {} scores".format(name))
        score_mean = accumulator["total"] / float(score_count)
        score_variance = max(
            0.0,
            accumulator["squared_total"] / float(score_count)
            - score_mean * score_mean,
        )
        metrics.update({
            "{}_score_count".format(name): score_count,
            "{}_score_mean".format(name): score_mean,
            "{}_score_std".format(name): math.sqrt(score_variance),
            "{}_score_min".format(name): accumulator["minimum"],
            "{}_score_max".format(name): accumulator["maximum"],
            "{}_score_fraction_ge_0_5".format(name): (
                accumulator["above_half_count"] / float(score_count)
            ),
            "{}_score_entropy".format(name): (
                accumulator["entropy_total"] / float(score_count)
            ),
        })
    return metrics


def _effective_weight(target: float, epoch: int, warmup_epochs: int) -> float:
    if warmup_epochs == 0:
        return float(target)
    return float(target) * min(1.0, float(epoch) / float(warmup_epochs))


def _classification_metrics(labels, predictions, probabilities):
    true_negative = sum(1 for y, p in zip(labels, predictions) if y == 0 and p == 0)
    false_positive = sum(1 for y, p in zip(labels, predictions) if y == 0 and p == 1)
    false_negative = sum(1 for y, p in zip(labels, predictions) if y == 1 and p == 0)
    true_positive = sum(1 for y, p in zip(labels, predictions) if y == 1 and p == 1)
    specificity = true_negative / float(max(1, true_negative + false_positive))
    sensitivity = true_positive / float(max(1, true_positive + false_negative))
    precision = true_positive / float(max(1, true_positive + false_positive))
    f1 = (
        2.0 * precision * sensitivity / (precision + sensitivity)
        if precision + sensitivity > 0.0
        else 0.0
    )
    roc_auc = None
    if len(set(labels)) == 2:
        from sklearn.metrics import roc_auc_score

        roc_auc = float(roc_auc_score(labels, probabilities))
    return {
        "balanced_accuracy": 0.5 * (specificity + sensitivity),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": [[true_negative, false_positive], [false_negative, true_positive]],
    }
