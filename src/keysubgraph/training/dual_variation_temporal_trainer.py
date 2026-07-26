"""Training, checkpointing and frozen-threshold evaluation for T1--T4."""

from __future__ import absolute_import, division, print_function

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import torch

from keysubgraph.models.dual_variation_temporal import (
    DualVariationTemporalClassifier,
)
from keysubgraph.training.dual_sgw_feature_trainer import (
    binary_metrics,
    fit_binary_threshold,
)
from keysubgraph.training.trainer import (
    class_weights_from_labels,
    set_reproducible_seed,
)


DUAL_TEMPORAL_CHECKPOINT_SCHEMA_VERSION = 1
DUAL_TEMPORAL_MODEL_NAME = "dual_d3b_variation_temporal_residual"


@dataclass(frozen=True)
class DualTemporalTrainingConfig:
    epochs: int = 60
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 10
    scheduler_factor: float = 0.5
    scheduler_patience: int = 4
    minimum_learning_rate: float = 1.0e-5
    temporal_auxiliary_weight: float = 0.30
    seed: int = 42
    max_train_batches: Optional[int] = None
    max_validation_batches: Optional[int] = None

    def __post_init__(self):
        if self.epochs < 1 or self.learning_rate <= 0.0:
            raise ValueError("temporal epochs and learning rate must be positive")
        if self.weight_decay < 0.0 or self.gradient_clip_norm <= 0.0:
            raise ValueError("invalid temporal optimizer settings")
        if self.early_stopping_patience < 0 or self.scheduler_patience < 0:
            raise ValueError("temporal patience cannot be negative")
        if not 0.0 < self.scheduler_factor < 1.0:
            raise ValueError("temporal scheduler factor must lie in (0,1)")
        if self.minimum_learning_rate <= 0.0:
            raise ValueError("temporal minimum learning rate must be positive")
        if self.temporal_auxiliary_weight < 0.0:
            raise ValueError("temporal auxiliary weight cannot be negative")


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


def _prediction_vectors(
    result: Mapping[str, Any],
) -> Tuple[List[str], List[int], List[float]]:
    predictions = result.get("predictions", [])
    return (
        [str(item["sample_key"]) for item in predictions],
        [int(item["label"]) for item in predictions],
        [float(item["positive_probability"]) for item in predictions],
    )


def run_dual_temporal_epoch(
    model: DualVariationTemporalClassifier,
    data_loader: Iterable,
    device: torch.device,
    class_weights: torch.Tensor,
    auxiliary_weight: float = 0.30,
    optimizer: Optional[torch.optim.Optimizer] = None,
    gradient_clip_norm: float = 1.0,
    max_batches: Optional[int] = None,
    include_predictions: bool = False,
) -> Dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    labels_all: List[int] = []
    probabilities_all: List[float] = []
    temporal_probabilities_all: List[float] = []
    keys_all: List[str] = []
    loss_total = 0.0
    classification_total = 0.0
    auxiliary_total = 0.0
    sample_total = 0
    gradient_norms: List[float] = []
    alpha_values: List[float] = []
    started = time.perf_counter()
    weights = class_weights.to(device)
    for batch_index, batch in enumerate(data_loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = batch.to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(batch)
            classification = torch.nn.functional.cross_entropy(
                output.final_logits, batch.labels, weight=weights
            )
            nonempty = batch.sequence_lengths > 0
            if (
                model.is_residual
                and auxiliary_weight > 0.0
                and bool(nonempty.any())
            ):
                auxiliary = torch.nn.functional.cross_entropy(
                    output.temporal_logits[nonempty],
                    batch.labels[nonempty],
                    weight=weights,
                )
            else:
                auxiliary = classification.new_zeros(())
            loss = classification + float(auxiliary_weight) * auxiliary
            if training:
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), gradient_clip_norm
                )
                gradient_norms.append(float(norm.detach().cpu()))
                optimizer.step()
        probabilities = torch.softmax(output.final_logits, dim=-1)[:, 1]
        temporal_probabilities = torch.softmax(
            output.temporal_logits, dim=-1
        )[:, 1]
        count = int(batch.labels.numel())
        sample_total += count
        loss_total += float(loss.detach().cpu()) * count
        classification_total += (
            float(classification.detach().cpu()) * count
        )
        auxiliary_total += float(auxiliary.detach().cpu()) * count
        labels_all.extend(
            int(value) for value in batch.labels.detach().cpu().tolist()
        )
        probabilities_all.extend(
            float(value) for value in probabilities.detach().cpu().tolist()
        )
        temporal_probabilities_all.extend(
            float(value)
            for value in temporal_probabilities.detach().cpu().tolist()
        )
        keys_all.extend(str(value) for value in batch.sample_keys)
        if output.alpha is not None:
            alpha_values.append(float(output.alpha.detach().cpu()))
    if sample_total < 1:
        raise ValueError("temporal epoch processed no samples")
    metrics = binary_metrics(labels_all, probabilities_all, threshold=0.5)
    temporal_metrics = binary_metrics(
        labels_all, temporal_probabilities_all, threshold=0.5
    )
    metrics.update(
        {
            "loss": loss_total / float(sample_total),
            "classification_loss": classification_total / float(sample_total),
            "temporal_auxiliary_loss": auxiliary_total / float(sample_total),
            "temporal_roc_auc": temporal_metrics["roc_auc"],
            "alpha": (
                sum(alpha_values) / float(len(alpha_values))
                if alpha_values
                else None
            ),
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
                "temporal_positive_probability": temporal_probability,
            }
            for key, label, probability, temporal_probability in zip(
                keys_all,
                labels_all,
                probabilities_all,
                temporal_probabilities_all,
            )
        ]
    return metrics


def _checkpoint_payload(
    model: DualVariationTemporalClassifier,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    history: List[Dict[str, Any]],
    config: DualTemporalTrainingConfig,
    class_weights: torch.Tensor,
    provenance: Mapping[str, Any],
    best_epoch: int,
    best_auc: float,
) -> Dict[str, Any]:
    return {
        "schema_version": DUAL_TEMPORAL_CHECKPOINT_SCHEMA_VERSION,
        "model_name": DUAL_TEMPORAL_MODEL_NAME,
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


def load_dual_temporal_checkpoint(
    path: Path,
    model: DualVariationTemporalClassifier,
    device: torch.device,
    expected_provenance: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    payload = _trusted_load(path, device)
    if payload.get("schema_version") != DUAL_TEMPORAL_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported dual temporal checkpoint schema")
    if payload.get("model_name") != DUAL_TEMPORAL_MODEL_NAME:
        raise ValueError("not a dual temporal checkpoint")
    if payload.get("model_config") != model.config_dict():
        raise ValueError("dual temporal model configuration mismatch")
    if (
        expected_provenance is not None
        and payload.get("provenance") != dict(expected_provenance)
    ):
        raise ValueError("dual temporal checkpoint provenance mismatch")
    model.load_state_dict(payload["model_state_dict"])
    return payload


def train_dual_temporal_classifier(
    model: DualVariationTemporalClassifier,
    train_loader: Iterable,
    validation_loader: Iterable,
    train_labels: Iterable[int],
    device: torch.device,
    config: DualTemporalTrainingConfig,
    output_dir: Path,
    provenance: Mapping[str, Any],
) -> Dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.json"
    if history_path.exists():
        raise FileExistsError("dual temporal output already exists")
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
        train = run_dual_temporal_epoch(
            model,
            train_loader,
            device,
            class_weights,
            auxiliary_weight=config.temporal_auxiliary_weight,
            optimizer=optimizer,
            gradient_clip_norm=config.gradient_clip_norm,
            max_batches=config.max_train_batches,
        )
        validation = run_dual_temporal_epoch(
            model,
            validation_loader,
            device,
            class_weights,
            auxiliary_weight=config.temporal_auxiliary_weight,
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
            "epoch {}/{} variant={} train_loss={:.6f} train_auc={} "
            "validation_loss={:.6f} validation_auc={} alpha={}".format(
                epoch,
                config.epochs,
                model.config.variant,
                train["loss"],
                train["roc_auc"],
                validation["loss"],
                validation["roc_auc"],
                validation["alpha"],
            ),
            flush=True,
        )
        if (
            config.early_stopping_patience > 0
            and without_improvement >= config.early_stopping_patience
        ):
            break
    payload = load_dual_temporal_checkpoint(
        output_dir / "best_checkpoint.pt",
        model,
        device,
        expected_provenance=provenance,
    )
    validation = run_dual_temporal_epoch(
        model,
        validation_loader,
        device,
        class_weights,
        auxiliary_weight=config.temporal_auxiliary_weight,
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
        "predictions": validation["predictions"],
    }
    evaluation_path = output_dir / "best_evaluation.json"
    _atomic_json(evaluation_path, evaluation)
    return {
        "variant": model.config.variant,
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
