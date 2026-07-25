"""Training and evaluation for low-capacity exact-SGW feature classifiers."""

from __future__ import absolute_import, division, print_function

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from keysubgraph.models.dual_sgw_feature_classifier import (
    DUAL_SGW_FEATURE_CLASSIFIER_MODEL_NAME,
    DUAL_SGW_FEATURE_CLASSIFIER_SCHEMA_VERSION,
    DualSGWFeatureClassifier,
)
from keysubgraph.training.trainer import (
    class_weights_from_labels,
    set_reproducible_seed,
)


@dataclass(frozen=True)
class DualSGWFeatureTrainingConfig:
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
            raise ValueError("feature-classifier epochs and LR must be positive")
        if self.weight_decay < 0.0 or self.gradient_clip_norm <= 0.0:
            raise ValueError("invalid feature-classifier optimizer settings")
        if self.early_stopping_patience < 0 or self.scheduler_patience < 0:
            raise ValueError("feature-classifier patience cannot be negative")
        if not 0.0 < self.scheduler_factor < 1.0:
            raise ValueError("scheduler factor must lie in (0,1)")
        if self.minimum_learning_rate <= 0.0:
            raise ValueError("minimum learning rate must be positive")


def _atomic_json(path: Path, payload: Any) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def _atomic_torch_save(path: Path, payload: Dict[str, Any]) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, str(temporary))
    os.replace(str(temporary), str(path))


def _trusted_load(path: Path, device: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(
            str(Path(path).resolve()),
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(str(Path(path).resolve()), map_location=device)


def fit_binary_threshold(
    labels: List[int],
    probabilities: List[float],
    metric: str,
) -> float:
    """Fit a deterministic threshold using validation labels only."""
    if metric not in ("balanced_accuracy", "accuracy"):
        raise ValueError("unsupported threshold metric")
    if len(labels) != len(probabilities) or not labels:
        raise ValueError("threshold fitting requires aligned predictions")
    if set(labels) != {0, 1}:
        return 0.5
    unique = sorted(set(float(value) for value in probabilities))
    candidates = [0.5] + unique
    candidates.extend(
        0.5 * (left + right)
        for left, right in zip(unique[:-1], unique[1:])
    )
    best_key = (float("-inf"), float("-inf"))
    best_threshold = 0.5
    for threshold in candidates:
        predictions = [
            int(probability >= threshold) for probability in probabilities
        ]
        if metric == "balanced_accuracy":
            score = float(balanced_accuracy_score(labels, predictions))
        else:
            score = float(accuracy_score(labels, predictions))
        key = (score, -abs(float(threshold) - 0.5))
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def binary_metrics(
    labels: List[int],
    probabilities: List[float],
    threshold: float,
) -> Dict[str, Any]:
    if len(labels) != len(probabilities) or not labels:
        raise ValueError("metrics require non-empty aligned predictions")
    predictions = [
        int(probability >= threshold) for probability in probabilities
    ]
    matrix = confusion_matrix(labels, predictions, labels=[0, 1]).astype(int)
    unique = set(labels)
    probability_mean = sum(probabilities) / float(len(probabilities))
    variance = sum(
        (value - probability_mean) ** 2 for value in probabilities
    ) / float(len(probabilities))
    true_negative, false_positive = matrix[0]
    false_negative, true_positive = matrix[1]
    return {
        "sample_count": len(labels),
        "class_counts": {
            str(label): int(sum(item == label for item in labels))
            for label in (0, 1)
        },
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(labels, predictions))
            if unique == {0, 1}
            else float(accuracy_score(labels, predictions))
        ),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": (
            float(roc_auc_score(labels, probabilities))
            if unique == {0, 1}
            else None
        ),
        "sensitivity": (
            float(true_positive) / float(true_positive + false_negative)
            if true_positive + false_negative
            else None
        ),
        "specificity": (
            float(true_negative) / float(true_negative + false_positive)
            if true_negative + false_positive
            else None
        ),
        "confusion_matrix": matrix.tolist(),
        "positive_probability": {
            "minimum": min(probabilities),
            "maximum": max(probabilities),
            "mean": probability_mean,
            "standard_deviation": math.sqrt(variance),
        },
    }


def run_dual_sgw_feature_epoch(
    model: DualSGWFeatureClassifier,
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
    labels_all: List[int] = []
    probabilities_all: List[float] = []
    keys_all: List[str] = []
    loss_total = 0.0
    sample_total = 0
    gradient_norms: List[float] = []
    started = time.perf_counter()
    for batch_index, batch in enumerate(data_loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        features = batch["features"].to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        labels = batch["label"].to(
            device=device, dtype=torch.long, non_blocking=True
        )
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(features)
            loss = torch.nn.functional.cross_entropy(
                logits, labels, weight=class_weights.to(device)
            )
            if training:
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), gradient_clip_norm
                )
                gradient_norms.append(float(norm.detach().cpu()))
                optimizer.step()
        probabilities = torch.softmax(logits, dim=-1)[:, 1]
        count = int(labels.numel())
        loss_total += float(loss.detach().cpu()) * count
        sample_total += count
        labels_all.extend(
            int(value) for value in labels.detach().cpu().tolist()
        )
        probabilities_all.extend(
            float(value) for value in probabilities.detach().cpu().tolist()
        )
        keys_all.extend(str(value) for value in batch["sample_key"])
    if sample_total < 1:
        raise ValueError("feature-classifier epoch processed no samples")
    metrics = binary_metrics(labels_all, probabilities_all, threshold=0.5)
    metrics.update(
        {
            "loss": loss_total / float(sample_total),
            "elapsed_seconds": time.perf_counter() - started,
            "mean_gradient_norm": (
                sum(gradient_norms) / float(len(gradient_norms))
                if gradient_norms
                else None
            ),
        }
    )
    if include_predictions:
        metrics["predictions"] = [
            {
                "sample_key": key,
                "label": label,
                "positive_probability": probability,
            }
            for key, label, probability in zip(
                keys_all, labels_all, probabilities_all
            )
        ]
    return metrics


def _prediction_vectors(
    result: Mapping[str, Any],
) -> Tuple[List[str], List[int], List[float]]:
    predictions = result.get("predictions", [])
    return (
        [str(item["sample_key"]) for item in predictions],
        [int(item["label"]) for item in predictions],
        [float(item["positive_probability"]) for item in predictions],
    )


def _checkpoint_payload(
    model: DualSGWFeatureClassifier,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    history: List[Dict[str, Any]],
    config: DualSGWFeatureTrainingConfig,
    class_weights: torch.Tensor,
    provenance: Mapping[str, Any],
    best_epoch: int,
    best_auc: float,
) -> Dict[str, Any]:
    return {
        "schema_version": DUAL_SGW_FEATURE_CLASSIFIER_SCHEMA_VERSION,
        "model_name": DUAL_SGW_FEATURE_CLASSIFIER_MODEL_NAME,
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "model_config": model.config_dict(),
        "training_config": asdict(config),
        "class_weights": class_weights.detach().cpu(),
        "provenance": dict(provenance),
        "history": list(history),
        "best_epoch": int(best_epoch),
        "best_validation_roc_auc": float(best_auc),
        "selection_metric": "validation_roc_auc",
        "validation_thresholds": None,
        "threshold_fit_split": "validation",
    }


def load_dual_sgw_feature_checkpoint(
    path: Path,
    model: DualSGWFeatureClassifier,
    device: torch.device,
    expected_provenance: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    payload = _trusted_load(path, device)
    if (
        payload.get("schema_version")
        != DUAL_SGW_FEATURE_CLASSIFIER_SCHEMA_VERSION
    ):
        raise ValueError("unsupported SGW feature checkpoint schema")
    if payload.get("model_name") != DUAL_SGW_FEATURE_CLASSIFIER_MODEL_NAME:
        raise ValueError("not an SGW feature-classifier checkpoint")
    if payload.get("model_config") != model.config_dict():
        raise ValueError("SGW feature-classifier configuration mismatch")
    if (
        expected_provenance is not None
        and payload.get("provenance") != dict(expected_provenance)
    ):
        raise ValueError("SGW feature-classifier provenance mismatch")
    model.load_state_dict(payload["model_state_dict"])
    return payload


def train_dual_sgw_feature_classifier(
    model: DualSGWFeatureClassifier,
    train_loader: Iterable,
    validation_loader: Iterable,
    train_labels: Iterable[int],
    device: torch.device,
    config: DualSGWFeatureTrainingConfig,
    output_dir: Path,
    provenance: Mapping[str, Any],
) -> Dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.json"
    if history_path.exists():
        raise FileExistsError("feature-classifier output already exists")
    set_reproducible_seed(config.seed)
    model.to(device)
    class_weights = class_weights_from_labels(train_labels)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.minimum_learning_rate,
    )
    history: List[Dict[str, Any]] = []
    best_epoch = 0
    best_auc = float("-inf")
    without_improvement = 0
    for epoch in range(1, config.epochs + 1):
        train = run_dual_sgw_feature_epoch(
            model,
            train_loader,
            device,
            class_weights,
            optimizer=optimizer,
            gradient_clip_norm=config.gradient_clip_norm,
            max_batches=config.max_train_batches,
        )
        validation = run_dual_sgw_feature_epoch(
            model,
            validation_loader,
            device,
            class_weights,
            max_batches=config.max_validation_batches,
        )
        selection_value = (
            float(validation["roc_auc"])
            if validation["roc_auc"] is not None
            else -float(validation["loss"])
        )
        scheduler.step(selection_value)
        improved = best_epoch == 0 or selection_value > best_auc
        if improved:
            best_epoch = epoch
            best_auc = selection_value
            without_improvement = 0
        else:
            without_improvement += 1
        record = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train": train,
            "validation": validation,
            "epochs_without_improvement": without_improvement,
        }
        history.append(record)
        payload = _checkpoint_payload(
            model,
            optimizer,
            scheduler,
            epoch,
            history,
            config,
            class_weights,
            provenance,
            best_epoch,
            best_auc,
        )
        _atomic_torch_save(output_dir / "last_checkpoint.pt", payload)
        if improved:
            _atomic_torch_save(output_dir / "best_checkpoint.pt", payload)
        _atomic_json(history_path, history)
        print(
            "epoch {}/{} classifier={} train_loss={:.6f} train_auc={} "
            "validation_loss={:.6f} validation_auc={}".format(
                epoch,
                config.epochs,
                model.config.classifier_type,
                train["loss"],
                train["roc_auc"],
                validation["loss"],
                validation["roc_auc"],
            ),
            flush=True,
        )
        if (
            config.early_stopping_patience > 0
            and without_improvement >= config.early_stopping_patience
        ):
            break
    payload = load_dual_sgw_feature_checkpoint(
        output_dir / "best_checkpoint.pt",
        model,
        device,
        expected_provenance=provenance,
    )
    validation = run_dual_sgw_feature_epoch(
        model,
        validation_loader,
        device,
        class_weights,
        max_batches=config.max_validation_batches,
        include_predictions=True,
    )
    keys, labels, probabilities = _prediction_vectors(validation)
    thresholds = {
        "balanced_accuracy": fit_binary_threshold(
            labels, probabilities, "balanced_accuracy"
        ),
        "accuracy": fit_binary_threshold(labels, probabilities, "accuracy"),
    }
    payload["validation_thresholds"] = thresholds
    _atomic_torch_save(output_dir / "best_checkpoint.pt", payload)
    evaluation = {
        "best_epoch": int(payload["best_epoch"]),
        "selection_metric": "validation_roc_auc",
        "validation_thresholds": thresholds,
        "metrics": {
            name: binary_metrics(labels, probabilities, threshold)
            for name, threshold in thresholds.items()
        },
        "predictions": [
            {
                "sample_key": key,
                "label": label,
                "positive_probability": probability,
            }
            for key, label, probability in zip(keys, labels, probabilities)
        ],
    }
    evaluation_path = output_dir / "best_evaluation.json"
    _atomic_json(evaluation_path, evaluation)
    return {
        "classifier_type": model.config.classifier_type,
        "epochs_completed": len(history),
        "best_epoch": int(payload["best_epoch"]),
        "best_validation_roc_auc": float(
            payload["best_validation_roc_auc"]
        ),
        "validation_thresholds": thresholds,
        "best_checkpoint": output_dir / "best_checkpoint.pt",
        "last_checkpoint": output_dir / "last_checkpoint.pt",
        "history": history_path,
        "best_evaluation": evaluation_path,
    }
