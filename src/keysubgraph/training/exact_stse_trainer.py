"""Training and versioned checkpoints for the isolated Exact-STSE baseline."""

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

from keysubgraph.models.exact_stse import (
    ExactSTSEClassifier,
    ExactSTSEConfig,
)
from keysubgraph.training.trainer import (
    class_weights_from_labels,
    set_reproducible_seed,
)


EXACT_STSE_CHECKPOINT_SCHEMA_VERSION = 1
EXACT_STSE_MODEL_NAME = "document_specified_exact_stse"


@dataclass(frozen=True)
class ExactSTSETrainingConfig:
    epochs: int = 80
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 15
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    minimum_learning_rate: float = 1.0e-5
    seed: int = 42
    max_train_batches: Optional[int] = None
    max_validation_batches: Optional[int] = None

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.learning_rate <= 0.0:
            raise ValueError("training epochs and learning rate must be positive")
        if self.weight_decay < 0.0 or self.gradient_clip_norm <= 0.0:
            raise ValueError("optimizer configuration is invalid")
        if self.early_stopping_patience < 0 or self.scheduler_patience < 0:
            raise ValueError("patience values cannot be negative")
        if not 0.0 < self.scheduler_factor < 1.0:
            raise ValueError("scheduler factor must lie in (0,1)")
        if self.minimum_learning_rate <= 0.0:
            raise ValueError("minimum learning rate must be positive")
        for limit in (self.max_train_batches, self.max_validation_batches):
            if limit is not None and limit < 1:
                raise ValueError("batch limits must be positive")


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


def exact_stse_config_from_dict(payload: Dict[str, Any]) -> ExactSTSEConfig:
    values = dict(payload)
    if "classifier_hidden_dims" in values:
        values["classifier_hidden_dims"] = tuple(
            int(value) for value in values["classifier_hidden_dims"]
        )
    return ExactSTSEConfig(**values)


def _classification_metrics(
    labels: List[int],
    probabilities: List[float],
    predictions: List[int],
) -> Dict[str, Any]:
    unique = set(labels)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1]).astype(int)
    true_negative, false_positive = matrix[0]
    false_negative, true_positive = matrix[1]
    sensitivity_denominator = true_positive + false_negative
    specificity_denominator = true_negative + false_positive
    probability_mean = sum(probabilities) / float(len(probabilities))
    probability_variance = sum(
        (value - probability_mean) ** 2 for value in probabilities
    ) / float(len(probabilities))
    accuracy = float(accuracy_score(labels, predictions))
    return {
        "sample_count": len(labels),
        "class_counts": {
            str(label): int(sum(value == label for value in labels))
            for label in (0, 1)
        },
        "threshold": 0.5,
        "accuracy": accuracy,
        "balanced_accuracy": (
            float(balanced_accuracy_score(labels, predictions))
            if unique == {0, 1}
            else accuracy
        ),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "sensitivity": (
            float(true_positive) / float(sensitivity_denominator)
            if sensitivity_denominator
            else None
        ),
        "specificity": (
            float(true_negative) / float(specificity_denominator)
            if specificity_denominator
            else None
        ),
        "roc_auc": (
            float(roc_auc_score(labels, probabilities))
            if unique == {0, 1}
            else None
        ),
        "confusion_matrix": matrix.tolist(),
        "positive_probability": {
            "minimum": min(probabilities),
            "maximum": max(probabilities),
            "mean": probability_mean,
            "standard_deviation": math.sqrt(probability_variance),
        },
    }


def run_exact_stse_epoch(
    model: ExactSTSEClassifier,
    data_loader: Iterable,
    device: torch.device,
    class_weights: torch.Tensor,
    optimizer: Optional[torch.optim.Optimizer] = None,
    gradient_clip_norm: float = 1.0,
    max_batches: Optional[int] = None,
    include_predictions: bool = False,
) -> Dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    count_total = 0
    weighted_loss_total = 0.0
    unweighted_loss_total = 0.0
    labels_all: List[int] = []
    probabilities_all: List[float] = []
    predictions_all: List[int] = []
    sample_keys_all: List[str] = []
    gradient_norms: List[float] = []
    started = time.perf_counter()
    weights = class_weights.to(device=device, dtype=torch.float32)
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
            # Do not use weighted cross_entropy(reduction="mean"): with B=1
            # PyTorch divides by the sole class weight and cancels weighting.
            weighted_loss = (per_sample * weights[labels]).mean()
            if training:
                weighted_loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), gradient_clip_norm
                )
                gradient_norms.append(float(norm.detach().cpu()))
                optimizer.step()
        count = int(labels.numel())
        count_total += count
        weighted_loss_total += float(weighted_loss.detach().cpu()) * count
        unweighted_loss_total += float(per_sample.sum().detach().cpu())
        probabilities = torch.softmax(output.logits, dim=-1)[:, 1]
        predictions = output.logits.argmax(dim=-1)
        labels_all.extend(int(value) for value in labels.detach().cpu().tolist())
        probabilities_all.extend(
            float(value) for value in probabilities.detach().cpu().tolist()
        )
        predictions_all.extend(
            int(value) for value in predictions.detach().cpu().tolist()
        )
        sample_keys_all.extend(batch.sample_keys)
    if count_total < 1:
        raise ValueError("Exact-STSE epoch processed no samples")
    metrics = {
        "loss": weighted_loss_total / float(count_total),
        "weighted_loss": weighted_loss_total / float(count_total),
        "unweighted_log_loss": unweighted_loss_total / float(count_total),
        "mean_gradient_norm": (
            sum(gradient_norms) / float(len(gradient_norms))
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
    if include_predictions:
        metrics["predictions"] = [
            {
                "sample_key": sample_key,
                "label": label,
                "positive_probability": probability,
                "prediction": prediction,
            }
            for sample_key, label, probability, prediction in zip(
                sample_keys_all,
                labels_all,
                probabilities_all,
                predictions_all,
            )
        ]
    return metrics


def _selection_key(metrics: Dict[str, Any]) -> Tuple[float, float, float]:
    return (
        float(metrics["balanced_accuracy"]),
        (
            float(metrics["roc_auc"])
            if metrics.get("roc_auc") is not None
            else float("-inf")
        ),
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
    best_key,
):
    return {
        "model_name": EXACT_STSE_MODEL_NAME,
        "schema_version": EXACT_STSE_CHECKPOINT_SCHEMA_VERSION,
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
        "best_selection_key": [float(value) for value in best_key],
        "torch_version": str(torch.__version__),
    }


def load_exact_stse_checkpoint(
    path: Path,
    model: ExactSTSEClassifier,
    device: torch.device,
    expected_protocol_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    payload = _trusted_torch_load(path, device)
    if payload.get("model_name") != EXACT_STSE_MODEL_NAME:
        raise ValueError("not an Exact-STSE checkpoint")
    if payload.get("schema_version") != EXACT_STSE_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported Exact-STSE checkpoint schema")
    if payload.get("model_config") != asdict(model.config):
        raise ValueError("Exact-STSE model configuration mismatch")
    if (
        expected_protocol_sha256 is not None
        and payload.get("protocol_sha256") != expected_protocol_sha256
    ):
        raise ValueError("Exact-STSE checkpoint protocol hash mismatch")
    model.load_state_dict(payload["model_state_dict"])
    return payload


def train_exact_stse(
    model: ExactSTSEClassifier,
    train_loader: Iterable,
    validation_loader: Iterable,
    train_labels: Iterable[int],
    device: torch.device,
    training_config: ExactSTSETrainingConfig,
    output_dir: Path,
    protocol_path: Path,
    protocol_sha256: str,
    resume_checkpoint: Optional[Path] = None,
) -> Dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.json"
    if history_path.exists() and resume_checkpoint is None:
        raise FileExistsError("Exact-STSE training output already exists")
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
        mode="max",
        factor=training_config.scheduler_factor,
        patience=training_config.scheduler_patience,
        min_lr=training_config.minimum_learning_rate,
    )
    history = []
    best_epoch = 0
    best_key = (float("-inf"), float("-inf"), float("-inf"))
    start_epoch = 1
    if resume_checkpoint is not None:
        payload = load_exact_stse_checkpoint(
            resume_checkpoint, model, device, protocol_sha256
        )
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        history = list(payload["history"])
        best_epoch = int(payload["best_epoch"])
        best_key = tuple(
            float(value) for value in payload["best_selection_key"]
        )
        start_epoch = int(payload["epoch"]) + 1
    epochs_without_improvement = (
        0
        if not history
        else int(history[-1].get("epochs_without_improvement", 0))
    )
    for epoch in range(start_epoch, training_config.epochs + 1):
        train_metrics = run_exact_stse_epoch(
            model,
            train_loader,
            device,
            class_weights,
            optimizer=optimizer,
            gradient_clip_norm=training_config.gradient_clip_norm,
            max_batches=training_config.max_train_batches,
        )
        validation_metrics = run_exact_stse_epoch(
            model,
            validation_loader,
            device,
            class_weights,
            optimizer=None,
            max_batches=training_config.max_validation_batches,
        )
        key = _selection_key(validation_metrics)
        scheduler.step(key[0])
        improved = best_epoch == 0 or key > best_key
        if improved:
            best_epoch, best_key = epoch, key
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        record = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train": train_metrics,
            "validation": validation_metrics,
            "epochs_without_improvement": epochs_without_improvement,
        }
        history.append(record)
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
            "validation_loss={:.6f} validation_ba={:.6f} "
            "validation_auc={} lr={:.8f}".format(
                epoch,
                training_config.epochs,
                train_metrics["loss"],
                train_metrics["balanced_accuracy"],
                train_metrics["roc_auc"],
                validation_metrics["loss"],
                validation_metrics["balanced_accuracy"],
                validation_metrics["roc_auc"],
                float(optimizer.param_groups[0]["lr"]),
            ),
            flush=True,
        )
        if (
            training_config.early_stopping_patience > 0
            and epochs_without_improvement
            >= training_config.early_stopping_patience
        ):
            break
    if best_epoch == 0:
        raise RuntimeError("Exact-STSE training produced no eligible checkpoint")
    load_exact_stse_checkpoint(
        output_dir / "best_checkpoint.pt",
        model,
        device,
        expected_protocol_sha256=protocol_sha256,
    )
    best_train = run_exact_stse_epoch(
        model,
        train_loader,
        device,
        class_weights,
        optimizer=None,
        max_batches=training_config.max_train_batches,
    )
    best_validation = run_exact_stse_epoch(
        model,
        validation_loader,
        device,
        class_weights,
        optimizer=None,
        max_batches=training_config.max_validation_batches,
        include_predictions=True,
    )
    evaluation_path = output_dir / "best_evaluation.json"
    _atomic_json(
        evaluation_path,
        {
            "best_epoch": best_epoch,
            "selection": {
                "primary": "validation_balanced_accuracy",
                "tie_breaker": "validation_roc_auc",
                "second_tie_breaker": "negative_validation_log_loss",
            },
            "train": best_train,
            "validation": best_validation,
        },
    )
    return {
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_checkpoint": output_dir / "best_checkpoint.pt",
        "last_checkpoint": output_dir / "last_checkpoint.pt",
        "history": history_path,
        "best_evaluation": evaluation_path,
    }
