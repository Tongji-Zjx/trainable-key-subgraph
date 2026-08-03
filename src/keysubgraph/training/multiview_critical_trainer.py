"""Training and frozen-threshold evaluation for the revised critical channel."""

from __future__ import absolute_import, division, print_function

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from keysubgraph.models.multiview_critical import (
    MultiViewCriticalClassifier,
    MultiViewCriticalConfig,
    multiview_critical_loss,
)
from keysubgraph.training.dual_sgw_feature_trainer import (
    binary_metrics,
    fit_binary_threshold,
)
from keysubgraph.training.trainer import class_weights_from_labels, set_reproducible_seed


@dataclass(frozen=True)
class MultiViewTrainingConfig:
    epochs: int = 80
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 15
    lambda_q: float = 0.1
    lambda_delta_q: float = 0.1
    seed: int = 42
    max_train_batches: object = None
    max_validation_batches: object = None

    def __post_init__(self):
        if self.epochs < 1 or self.learning_rate <= 0.0:
            raise ValueError("multi-view epochs/learning rate must be positive")
        if self.weight_decay < 0.0 or self.gradient_clip_norm <= 0.0:
            raise ValueError("invalid multi-view optimizer settings")
        if self.early_stopping_patience < 0 or self.lambda_q < 0.0 or self.lambda_delta_q < 0.0:
            raise ValueError("invalid multi-view loss/patience settings")


def _atomic_json(path, payload):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def _atomic_torch(path, payload):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, str(temporary))
    os.replace(str(temporary), str(path))


def run_multiview_epoch(
    model,
    loader,
    device,
    class_weights,
    optimizer=None,
    gradient_clip_norm=1.0,
    lambda_q=0.1,
    lambda_delta_q=0.1,
    threshold=0.5,
    max_batches=None,
    include_predictions=False,
):
    training = optimizer is not None
    model.train(training)
    labels_all, probabilities_all, keys_all = [], [], []
    totals = {"loss": 0.0, "classification_loss": 0.0, "q_loss": 0.0, "delta_q_loss": 0.0}
    count_total = 0
    gradients = []
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= int(max_batches):
            break
        batch = batch.to(device)
        labels = batch.labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(batch)
            losses = multiview_critical_loss(
                output, labels, lambda_q=lambda_q,
                lambda_delta_q=lambda_delta_q,
                class_weights=class_weights.to(device),
            )
            if training:
                losses["loss"].backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip_norm))
                gradients.append(float(norm.detach().cpu()))
                optimizer.step()
        probabilities = torch.softmax(output.logits, dim=-1)[:, 1]
        current = int(labels.numel())
        count_total += current
        for name in totals:
            totals[name] += float(losses[name].detach().cpu()) * current
        labels_all.extend(int(value) for value in labels.detach().cpu().tolist())
        probabilities_all.extend(float(value) for value in probabilities.detach().cpu().tolist())
        keys_all.extend(batch.sample_keys)
    if count_total < 1:
        raise ValueError("multi-view epoch processed no samples")
    metrics = binary_metrics(labels_all, probabilities_all, threshold)
    metrics.update({name: value / count_total for name, value in totals.items()})
    metrics["mean_gradient_norm"] = sum(gradients) / len(gradients) if gradients else None
    metrics["elapsed_seconds"] = time.perf_counter() - started
    if include_predictions:
        metrics["sample_keys"] = keys_all
        metrics["labels"] = labels_all
        metrics["probabilities"] = probabilities_all
    return metrics


def train_multiview_critical(
    model, train_loader, validation_loader, device, output_dir, config=None,
    checkpoint_metadata=None,
):
    config = config or MultiViewTrainingConfig()
    set_reproducible_seed(config.seed)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    class_weights = class_weights_from_labels(train_loader.dataset.labels).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1.0e-5
    )
    history, best_auc, stale = [], float("-inf"), 0
    for epoch in range(1, config.epochs + 1):
        train = run_multiview_epoch(
            model, train_loader, device, class_weights, optimizer,
            config.gradient_clip_norm, config.lambda_q, config.lambda_delta_q,
            max_batches=config.max_train_batches,
        )
        validation = run_multiview_epoch(
            model, validation_loader, device, class_weights,
            lambda_q=config.lambda_q, lambda_delta_q=config.lambda_delta_q,
            max_batches=config.max_validation_batches, include_predictions=True,
        )
        threshold = fit_binary_threshold(
            validation["labels"], validation["probabilities"], "balanced_accuracy"
        )
        validation_metrics = binary_metrics(
            validation["labels"], validation["probabilities"], threshold
        )
        for name in ("loss", "classification_loss", "q_loss", "delta_q_loss"):
            validation_metrics[name] = validation[name]
        auc = validation_metrics["roc_auc"]
        score = float(auc) if auc is not None else float("-inf")
        scheduler.step(score if score != float("-inf") else 0.0)
        row = {"epoch": epoch, "train": train, "validation": validation_metrics}
        history.append(row)
        checkpoint = {
            "schema_version": 1,
            "model_name": model.model_name,
            "model_config": model.config_dict(),
            "training_config": asdict(config),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "threshold": threshold,
            "validation": validation_metrics,
        }
        checkpoint.update(dict(checkpoint_metadata or {}))
        _atomic_torch(output_dir / "last_checkpoint.pt", checkpoint)
        _atomic_json(output_dir / "history.json", history)
        if score > best_auc:
            best_auc, stale = score, 0
            _atomic_torch(output_dir / "best_checkpoint.pt", checkpoint)
            _atomic_json(output_dir / "best_evaluation.json", {"epoch": epoch, "validation": validation_metrics})
        else:
            stale += 1
        print(
            "epoch {}/{} train_loss={:.6f} train_auc={} validation_loss={:.6f} validation_auc={} q={:.6f} dq={:.6f}".format(
                epoch, config.epochs, train["loss"], train["roc_auc"],
                validation_metrics["loss"], validation_metrics["roc_auc"],
                validation_metrics["q_loss"], validation_metrics["delta_q_loss"],
            ), flush=True,
        )
        if config.early_stopping_patience and stale >= config.early_stopping_patience:
            break
    return {"best_checkpoint": str(output_dir / "best_checkpoint.pt"), "epochs_completed": len(history), "best_auc": best_auc}


def load_multiview_checkpoint(path, model, device):
    try:
        payload = torch.load(str(Path(path).resolve()), map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(str(Path(path).resolve()), map_location=device)
    if payload.get("model_name") != model.model_name or payload.get("model_config") != model.config_dict():
        raise ValueError("multi-view checkpoint/model mismatch")
    model.load_state_dict(payload["model_state_dict"])
    return payload
