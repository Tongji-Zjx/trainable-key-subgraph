"""Training and validation for the controlled full-graph encoder comparison."""

from __future__ import absolute_import, division, print_function

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from keysubgraph.models.full_graph_classifier import FullGraphSequenceClassifier
from keysubgraph.training.trainer import class_weights_from_labels, set_reproducible_seed


FULL_GRAPH_CHECKPOINT_SCHEMA_VERSION = 1
FULL_GRAPH_MODEL_NAME = "full_graph_encoder_comparison"


@dataclass(frozen=True)
class FullGraphTrainingConfig:
    epochs: int = 60
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 10
    scheduler_factor: float = 0.5
    scheduler_patience: int = 4
    minimum_learning_rate: float = 1.0e-5
    seed: int = 42
    max_train_batches: Optional[int] = None
    max_validation_batches: Optional[int] = None
    memorization_mode: bool = False

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("optimizer configuration is invalid")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient clip must be positive")
        if self.early_stopping_patience < 0 or self.scheduler_patience < 0:
            raise ValueError("patience values must be non-negative")
        if self.scheduler_factor <= 0.0 or self.scheduler_factor >= 1.0:
            raise ValueError("scheduler factor must lie in (0, 1)")
        if self.minimum_learning_rate <= 0.0:
            raise ValueError("minimum learning rate must be positive")
        for value in (self.max_train_batches, self.max_validation_batches):
            if value is not None and value < 1:
                raise ValueError("batch limits must be positive")
        if self.memorization_mode:
            if self.weight_decay != 0.0:
                raise ValueError("memorization mode requires zero weight decay")
            if self.early_stopping_patience != 0:
                raise ValueError("memorization mode disables early stopping")


def _atomic_json(path: Path, payload: Any) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(str(temporary), str(path))


def _atomic_torch_save(path: Path, payload: Dict[str, Any]) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, str(temporary))
    os.replace(str(temporary), str(path))


def _trusted_torch_load(path: Path, device: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(
            str(Path(path).resolve()), map_location=device, weights_only=False
        )
    except TypeError:
        return torch.load(str(Path(path).resolve()), map_location=device)


def _classification_metrics(
    labels: List[int], probabilities: List[float], predictions: List[int]
) -> Dict[str, Any]:
    unique = set(labels)
    accuracy = float(accuracy_score(labels, predictions))
    probability_mean = sum(probabilities) / float(len(probabilities))
    probability_variance = sum(
        (value - probability_mean) ** 2 for value in probabilities
    ) / float(len(probabilities))
    return {
        "sample_count": len(labels),
        "class_counts": {
            str(label): int(sum(value == label for value in labels))
            for label in (0, 1)
        },
        "accuracy": accuracy,
        "balanced_accuracy": (
            float(balanced_accuracy_score(labels, predictions))
            if unique == {0, 1}
            else accuracy
        ),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": (
            float(roc_auc_score(labels, probabilities))
            if unique == {0, 1}
            else None
        ),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=[0, 1]
        ).astype(int).tolist(),
        "positive_probability": {
            "minimum": min(probabilities),
            "maximum": max(probabilities),
            "mean": probability_mean,
            "standard_deviation": math.sqrt(probability_variance),
        },
        "predicted_positive_ratio": (
            sum(predictions) / float(len(predictions))
        ),
    }


def run_full_graph_classifier_epoch(
    model: FullGraphSequenceClassifier,
    data_loader: Iterable,
    device: torch.device,
    class_weights: torch.Tensor,
    optimizer: Optional[torch.optim.Optimizer] = None,
    gradient_clip_norm: float = 1.0,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    weights = class_weights.to(device=device, dtype=torch.float32)
    sample_count = 0
    weighted_loss_total = 0.0
    unweighted_loss_total = 0.0
    labels_all, probabilities_all, predictions_all = [], [], []
    gradient_norms = []
    prototype_sum = None
    prototype_entropy_total = 0.0
    prototype_max_total = 0.0
    prototype_sample_count = 0
    started = time.perf_counter()

    for batch_index, cpu_batch in enumerate(data_loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = cpu_batch.to(device)
        labels = batch.labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(batch)
            per_sample = torch.nn.functional.cross_entropy(
                output.logits, labels, reduction="none"
            )
            # An ordinary mean after per-sample multiplication is required:
            # weighted CE's normalized mean cancels the target weight for B=1.
            weighted_loss = (
                per_sample * weights.index_select(0, labels)
            ).mean()
            if training:
                weighted_loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), gradient_clip_norm
                )
                gradient_norms.append(float(norm.detach().cpu()))
                optimizer.step()

        count = int(labels.numel())
        sample_count += count
        weighted_loss_total += float(weighted_loss.detach().cpu()) * count
        unweighted_loss_total += float(per_sample.detach().sum().cpu())
        probabilities = torch.softmax(output.logits, dim=-1)[:, 1]
        predictions = output.logits.argmax(dim=-1)
        labels_all.extend(int(value) for value in labels.detach().cpu().tolist())
        probabilities_all.extend(
            float(value) for value in probabilities.detach().cpu().tolist()
        )
        predictions_all.extend(
            int(value) for value in predictions.detach().cpu().tolist()
        )

        if output.prototype_attention is not None:
            attention = output.prototype_attention.detach()
            bounded = attention.clamp_min(torch.finfo(attention.dtype).eps)
            batch_sum = attention.sum(dim=0).cpu()
            prototype_sum = (
                batch_sum
                if prototype_sum is None
                else prototype_sum + batch_sum
            )
            prototype_entropy_total += float(
                (-(bounded * bounded.log()).sum(dim=-1)).sum().cpu()
            )
            prototype_max_total += float(attention.max(dim=-1).values.sum().cpu())
            prototype_sample_count += count

    if sample_count < 1:
        raise ValueError("full-graph epoch processed no samples")
    metrics = {
        "weighted_loss": weighted_loss_total / float(sample_count),
        "unweighted_log_loss": unweighted_loss_total / float(sample_count),
        "mean_gradient_norm": (
            sum(gradient_norms) / len(gradient_norms)
            if gradient_norms
            else None
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    metrics.update(
        _classification_metrics(
            labels_all, probabilities_all, predictions_all
        )
    )
    if prototype_sample_count:
        usage = prototype_sum / float(prototype_sample_count)
        metrics.update(
            {
                "prototype_attention_entropy": (
                    prototype_entropy_total / float(prototype_sample_count)
                ),
                "prototype_mean_max_attention": (
                    prototype_max_total / float(prototype_sample_count)
                ),
                "prototype_usage": [float(value) for value in usage.tolist()],
                "prototype_max_usage": float(usage.max()),
            }
        )
    else:
        metrics.update(
            {
                "prototype_attention_entropy": None,
                "prototype_mean_max_attention": None,
                "prototype_usage": None,
                "prototype_max_usage": None,
            }
        )
    return metrics


def _selection_key(metrics: Dict[str, Any]) -> Tuple[float, float]:
    auc = metrics.get("roc_auc")
    return (
        float(auc) if auc is not None and math.isfinite(float(auc)) else float("-inf"),
        -float(metrics["unweighted_log_loss"]),
    )


def _checkpoint_payload(
    model,
    optimizer,
    scheduler,
    epoch,
    history,
    training_config,
    class_weights,
    protocol_path,
    protocol_sha256,
    best_epoch,
    best_selection_key,
):
    return {
        "model_name": FULL_GRAPH_MODEL_NAME,
        "schema_version": FULL_GRAPH_CHECKPOINT_SCHEMA_VERSION,
        "stage": "full_graph_encoder_comparison",
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "model_config": asdict(model.config),
        "training_config": asdict(training_config),
        "class_weights": class_weights.detach().cpu(),
        "protocol_path": str(Path(protocol_path).resolve()),
        "protocol_sha256": str(protocol_sha256),
        "history": list(history),
        "best_epoch": int(best_epoch),
        "best_selection_key": [
            float(best_selection_key[0]),
            float(best_selection_key[1]),
        ],
        "torch_version": str(torch.__version__),
    }


def load_full_graph_classifier_checkpoint(
    path: Path,
    model: FullGraphSequenceClassifier,
    device: torch.device,
    expected_protocol_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    payload = _trusted_torch_load(path, device)
    if payload.get("model_name") != FULL_GRAPH_MODEL_NAME:
        raise ValueError("not a full-graph encoder checkpoint")
    if payload.get("schema_version") != FULL_GRAPH_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported full-graph checkpoint schema")
    if payload.get("model_config") != asdict(model.config):
        raise ValueError("full-graph model configuration mismatch")
    if (
        expected_protocol_sha256 is not None
        and payload.get("protocol_sha256") != expected_protocol_sha256
    ):
        raise ValueError("full-graph checkpoint protocol hash mismatch")
    model.load_state_dict(payload["model_state_dict"])
    return payload


def train_full_graph_classifier(
    model: FullGraphSequenceClassifier,
    train_loader: Iterable,
    validation_loader: Iterable,
    train_labels: Iterable[int],
    device: torch.device,
    training_config: FullGraphTrainingConfig,
    output_dir: Path,
    protocol_path: Path,
    protocol_sha256: str,
) -> Dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.json"
    if history_path.exists():
        raise FileExistsError("full-graph training output already exists")
    if training_config.memorization_mode:
        if model.config.encoder_type != "signed_gnn_tcn":
            raise ValueError(
                "memorization mode is restricted to the controlled baseline"
            )
        dropout_values = (
            model.config.baseline_dropout,
            model.config.gated_gnn_dropout,
            model.config.classifier_dropout,
        )
        if any(value != 0.0 for value in dropout_values):
            raise ValueError("memorization mode requires all dropout to be zero")
        train_dataset = getattr(train_loader, "dataset", None)
        replay_dataset = getattr(validation_loader, "dataset", None)
        if (
            train_dataset is not None
            and replay_dataset is not None
            and train_dataset is not replay_dataset
        ):
            raise ValueError(
                "memorization mode must replay the identical training dataset"
            )
    set_reproducible_seed(training_config.seed)
    model.to(device)
    class_weights = class_weights_from_labels(train_labels)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=training_config.scheduler_factor,
        patience=training_config.scheduler_patience,
        min_lr=training_config.minimum_learning_rate,
    )
    history = []
    best_epoch = 0
    best_key = (float("-inf"), float("-inf"))
    epochs_without_improvement = 0
    started = time.perf_counter()

    for epoch in range(1, training_config.epochs + 1):
        train_metrics = run_full_graph_classifier_epoch(
            model,
            train_loader,
            device,
            class_weights,
            optimizer=optimizer,
            gradient_clip_norm=training_config.gradient_clip_norm,
            max_batches=training_config.max_train_batches,
        )
        evaluation_metrics = run_full_graph_classifier_epoch(
            model,
            validation_loader,
            device,
            class_weights,
            optimizer=None,
            gradient_clip_norm=training_config.gradient_clip_norm,
            max_batches=training_config.max_validation_batches,
        )
        scheduler.step(evaluation_metrics["unweighted_log_loss"])
        learning_rate = float(optimizer.param_groups[0]["lr"])
        evaluation_key = (
            "memorization_train_replay"
            if training_config.memorization_mode
            else "validation"
        )
        record = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train": train_metrics,
            evaluation_key: evaluation_metrics,
        }
        history.append(record)
        candidate_key = _selection_key(evaluation_metrics)
        improved = best_epoch == 0 or candidate_key > best_key
        if improved:
            best_epoch = epoch
            best_key = candidate_key
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        payload = _checkpoint_payload(
            model,
            optimizer,
            scheduler,
            epoch,
            history,
            training_config,
            class_weights,
            protocol_path,
            protocol_sha256,
            best_epoch,
            best_key,
        )
        _atomic_torch_save(output_dir / "last_checkpoint.pt", payload)
        if improved:
            _atomic_torch_save(output_dir / "best_checkpoint.pt", payload)
        _atomic_json(history_path, history)
        print(
            "epoch {}/{} train_loss={:.6f} train_ba={:.6f} train_auc={} "
            "{}_loss={:.6f} {}_ba={:.6f} {}_auc={} "
            "{}_probability_std={:.8f} lr={:.8f}".format(
                epoch,
                training_config.epochs,
                train_metrics["unweighted_log_loss"],
                train_metrics["balanced_accuracy"],
                train_metrics["roc_auc"],
                evaluation_key,
                evaluation_metrics["unweighted_log_loss"],
                evaluation_key,
                evaluation_metrics["balanced_accuracy"],
                evaluation_key,
                evaluation_metrics["roc_auc"],
                evaluation_key,
                evaluation_metrics["positive_probability"][
                    "standard_deviation"
                ],
                learning_rate,
            ),
            flush=True,
        )
        if (
            training_config.early_stopping_patience > 0
            and epochs_without_improvement
            >= training_config.early_stopping_patience
        ):
            break

    load_full_graph_classifier_checkpoint(
        output_dir / "best_checkpoint.pt",
        model,
        device,
        expected_protocol_sha256=protocol_sha256,
    )
    best_train = run_full_graph_classifier_epoch(
        model,
        train_loader,
        device,
        class_weights,
        optimizer=None,
        max_batches=training_config.max_train_batches,
    )
    best_evaluation = run_full_graph_classifier_epoch(
        model,
        validation_loader,
        device,
        class_weights,
        optimizer=None,
        max_batches=training_config.max_validation_batches,
    )
    evaluation_key = (
        "memorization_train_replay"
        if training_config.memorization_mode
        else "validation"
    )
    evaluation = {
        "diagnostic_only": bool(training_config.memorization_mode),
        "run_mode": (
            "full_training_set_memorization"
            if training_config.memorization_mode
            else "controlled_validation"
        ),
        "best_epoch": best_epoch,
        "selection": {
            "primary": "{}_roc_auc".format(evaluation_key),
            "tie_breaker": "{}_unweighted_log_loss".format(evaluation_key),
            "value": [best_key[0], -best_key[1]],
        },
        "train": best_train,
        evaluation_key: best_evaluation,
    }
    _atomic_json(output_dir / "best_evaluation.json", evaluation)
    return {
        "best_checkpoint": output_dir / "best_checkpoint.pt",
        "last_checkpoint": output_dir / "last_checkpoint.pt",
        "history": history_path,
        "best_evaluation": output_dir / "best_evaluation.json",
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "elapsed_seconds": time.perf_counter() - started,
        "diagnostic_only": bool(training_config.memorization_mode),
        "run_mode": evaluation["run_mode"],
    }
