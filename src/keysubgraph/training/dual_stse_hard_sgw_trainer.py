"""Stage-aware training and checkpoints for Dual-STSE-HardSGW."""

from __future__ import absolute_import, division, print_function

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from keysubgraph.models.dual_stse_hard_sgw import (
    DualSTSEHardSGWClassifier,
)
from keysubgraph.models.dual_stse_hard_sgw_loss import (
    DualSTSEHardSGWCriterion,
    DualSTSEHardSGWLossConfig,
)
from keysubgraph.training.trainer import (
    class_weights_from_labels,
    set_reproducible_seed,
)


DUAL_CHECKPOINT_SCHEMA_VERSION = 1
DUAL_MODEL_NAME = "dual_stse_hard_sgw"


@dataclass(frozen=True)
class DualTrainingConfig:
    stage: str
    epochs: int = 80
    learning_rate: float = 1.0e-3
    stse_learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 15
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    minimum_learning_rate: float = 1.0e-5
    seed: int = 42
    fine_tune_stse: bool = False
    max_train_batches: Optional[int] = None
    max_validation_batches: Optional[int] = None

    def __post_init__(self) -> None:
        if self.stage not in (
            "selector_proxy",
            "sgw_classifier",
            "fusion",
        ):
            raise ValueError("unsupported dual training stage")
        if self.epochs < 1 or self.learning_rate <= 0.0:
            raise ValueError("dual epochs and learning rate must be positive")
        if self.stse_learning_rate <= 0.0:
            raise ValueError("dual STSE learning rate must be positive")
        if self.weight_decay < 0.0 or self.gradient_clip_norm <= 0.0:
            raise ValueError("invalid dual optimizer configuration")
        if self.early_stopping_patience < 0 or self.scheduler_patience < 0:
            raise ValueError("dual patience values cannot be negative")
        if not 0.0 < self.scheduler_factor < 1.0:
            raise ValueError("dual scheduler factor must lie in (0,1)")
        if self.minimum_learning_rate <= 0.0:
            raise ValueError("dual minimum learning rate must be positive")
        if self.fine_tune_stse and self.stage != "fusion":
            raise ValueError("STSE fine-tuning is only valid in fusion")


def _atomic_json(path: Path, payload: Any) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(
            payload, handle, ensure_ascii=False, indent=2, sort_keys=True
        )
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
        return torch.load(
            str(Path(path).resolve()), map_location=device
        )


def _features_for_batch(
    sample_keys,
    feature_lookup: Optional[Mapping[str, torch.Tensor]],
    device: torch.device,
) -> Optional[torch.Tensor]:
    if feature_lookup is None:
        return None
    missing = [key for key in sample_keys if key not in feature_lookup]
    if missing:
        raise KeyError(
            "exact SGW feature is missing for {}".format(missing[0])
        )
    return torch.stack(
        [
            feature_lookup[key].detach().to(
                device=device, dtype=torch.float32
            )
            for key in sample_keys
        ],
        dim=0,
    )


def _fit_threshold(labels: List[int], probabilities: List[float]) -> float:
    if set(labels) != {0, 1}:
        return 0.5
    unique = sorted(set(float(value) for value in probabilities))
    candidates = [0.5] + unique
    candidates.extend(
        0.5 * (left + right)
        for left, right in zip(unique[:-1], unique[1:])
    )
    best = (float("-inf"), float("-inf"))
    best_threshold = 0.5
    for threshold in candidates:
        predictions = [
            int(probability >= threshold)
            for probability in probabilities
        ]
        balanced = float(
            balanced_accuracy_score(labels, predictions)
        )
        key = (balanced, -abs(float(threshold) - 0.5))
        if key > best:
            best = key
            best_threshold = float(threshold)
    return best_threshold


def _metrics(
    labels: List[int],
    probabilities: List[float],
    threshold: float,
) -> Dict[str, Any]:
    predictions = [
        int(probability >= threshold) for probability in probabilities
    ]
    matrix = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).astype(int)
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
            float(true_positive)
            / float(true_positive + false_negative)
            if true_positive + false_negative
            else None
        ),
        "specificity": (
            float(true_negative)
            / float(true_negative + false_positive)
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


def _stage_logits(output, stage):
    if stage == "selector_proxy":
        return output.selector_proxy_logits
    if stage == "sgw_classifier":
        return output.sgw_logits
    return output.fusion_logits


def run_dual_epoch(
    model: DualSTSEHardSGWClassifier,
    data_loader: Iterable,
    device: torch.device,
    criterion: DualSTSEHardSGWCriterion,
    stage: str,
    class_weights: torch.Tensor,
    feature_lookup: Optional[Mapping[str, torch.Tensor]] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    gradient_clip_norm: float = 1.0,
    max_batches: Optional[int] = None,
    threshold: float = 0.5,
    include_predictions: bool = False,
) -> Dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    labels_all: List[int] = []
    probabilities_all: List[float] = []
    keys_all: List[str] = []
    component_totals: Dict[str, float] = {}
    sample_total = 0
    gradient_norms: List[float] = []
    started = time.perf_counter()
    for batch_index, cpu_batch in enumerate(data_loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = cpu_batch.to(device)
        labels = batch.labels.to(device)
        exact_features = _features_for_batch(
            batch.sample_keys, feature_lookup, device
        )
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(
                batch,
                exact_sgw_features=exact_features,
                compute_selector_proxy=stage == "selector_proxy",
            )
            loss = criterion(
                output, labels, stage, class_weights.to(device)
            )
            if training:
                loss.total.backward()
                trainable = [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ]
                norm = torch.nn.utils.clip_grad_norm_(
                    trainable, gradient_clip_norm
                )
                gradient_norms.append(float(norm.detach().cpu()))
                optimizer.step()
        logits = _stage_logits(output, stage)
        if logits is None:
            raise RuntimeError("dual stage produced no logits")
        probabilities = torch.softmax(logits, dim=-1)[:, 1]
        count = int(labels.numel())
        sample_total += count
        for name in (
            "total",
            "fusion_ce",
            "stse_ce",
            "sgw_ce",
            "selector_proxy_ce",
            "node_budget",
            "edge_budget",
            "laplacian",
            "gw_proxy",
        ):
            value = getattr(loss, name)
            component_totals[name] = component_totals.get(
                name, 0.0
            ) + float(value.detach().cpu()) * count
        labels_all.extend(
            int(value) for value in labels.detach().cpu().tolist()
        )
        probabilities_all.extend(
            float(value)
            for value in probabilities.detach().cpu().tolist()
        )
        keys_all.extend(batch.sample_keys)
    if sample_total < 1:
        raise ValueError("dual epoch processed no samples")
    result = {
        name: value / float(sample_total)
        for name, value in component_totals.items()
    }
    result["loss"] = result["total"]
    result["mean_gradient_norm"] = (
        sum(gradient_norms) / float(len(gradient_norms))
        if gradient_norms
        else None
    )
    result["elapsed_seconds"] = time.perf_counter() - started
    result.update(_metrics(labels_all, probabilities_all, threshold))
    if include_predictions:
        result["predictions"] = [
            {
                "sample_key": key,
                "label": label,
                "positive_probability": probability,
                "prediction": int(probability >= threshold),
            }
            for key, label, probability in zip(
                keys_all, labels_all, probabilities_all
            )
        ]
    return result


def _build_optimizer(
    model: DualSTSEHardSGWClassifier,
    config: DualTrainingConfig,
):
    model.set_stage_trainability(config.stage)
    parameter_groups = []
    base = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if base:
        parameter_groups.append(
            {"params": base, "lr": config.learning_rate}
        )
    if config.stage == "fusion" and config.fine_tune_stse:
        model.stse_channel.set_trainable(
            encoder=True, classifier=True
        )
        stse_parameters = list(model.stse_channel.parameters())
        existing = {
            id(parameter)
            for group in parameter_groups
            for parameter in group["params"]
        }
        stse_parameters = [
            parameter
            for parameter in stse_parameters
            if id(parameter) not in existing
        ]
        parameter_groups.append(
            {
                "params": stse_parameters,
                "lr": config.stse_learning_rate,
            }
        )
    if not parameter_groups:
        raise ValueError("dual stage has no trainable parameters")
    return torch.optim.AdamW(
        parameter_groups, weight_decay=config.weight_decay
    )


def _checkpoint_payload(
    model,
    optimizer,
    scheduler,
    epoch,
    history,
    training_config,
    loss_config,
    class_weights,
    protocol_sha256,
    provenance,
    best_epoch,
    best_auc,
    validation_threshold=None,
):
    return {
        "schema_version": DUAL_CHECKPOINT_SCHEMA_VERSION,
        "model_name": DUAL_MODEL_NAME,
        "stage": training_config.stage,
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "model_config": asdict(model.config),
        "training_config": asdict(training_config),
        "loss_config": asdict(loss_config),
        "class_weights": class_weights.detach().cpu(),
        "protocol_sha256": str(protocol_sha256),
        "provenance": dict(provenance),
        "history": list(history),
        "best_epoch": int(best_epoch),
        "best_validation_roc_auc": float(best_auc),
        "validation_threshold": validation_threshold,
        "selection_metric": "validation_roc_auc",
        "threshold_fit_split": "validation",
    }


def load_dual_checkpoint(
    path: Path,
    model: DualSTSEHardSGWClassifier,
    device: torch.device,
    expected_stage: Optional[str] = None,
    expected_protocol_sha256: Optional[str] = None,
    expected_provenance: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    payload = _trusted_load(path, device)
    if payload.get("schema_version") != DUAL_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported dual checkpoint schema")
    if payload.get("model_name") != DUAL_MODEL_NAME:
        raise ValueError("not a Dual-STSE-HardSGW checkpoint")
    if expected_stage is not None and payload.get("stage") != expected_stage:
        raise ValueError("dual checkpoint stage mismatch")
    if (
        expected_protocol_sha256 is not None
        and payload.get("protocol_sha256") != expected_protocol_sha256
    ):
        raise ValueError("dual checkpoint protocol hash mismatch")
    if expected_provenance is not None and payload.get(
        "provenance"
    ) != dict(expected_provenance):
        raise ValueError("dual checkpoint provenance mismatch")
    model.load_state_dict(payload["model_state_dict"])
    return payload


def train_dual_stage(
    model: DualSTSEHardSGWClassifier,
    train_loader: Iterable,
    validation_loader: Iterable,
    train_labels: Iterable[int],
    device: torch.device,
    training_config: DualTrainingConfig,
    loss_config: DualSTSEHardSGWLossConfig,
    output_dir: Path,
    protocol_sha256: str,
    provenance: Mapping[str, str],
    train_feature_lookup: Optional[Mapping[str, torch.Tensor]] = None,
    validation_feature_lookup: Optional[
        Mapping[str, torch.Tensor]
    ] = None,
) -> Dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.json"
    if history_path.exists():
        raise FileExistsError("dual training output already exists")
    set_reproducible_seed(training_config.seed)
    model.to(device)
    class_weights = class_weights_from_labels(train_labels)
    criterion = DualSTSEHardSGWCriterion(loss_config)
    optimizer = _build_optimizer(model, training_config)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=training_config.scheduler_factor,
        patience=training_config.scheduler_patience,
        min_lr=training_config.minimum_learning_rate,
    )
    history = []
    best_epoch = 0
    best_auc = float("-inf")
    without_improvement = 0
    for epoch in range(1, training_config.epochs + 1):
        train_metrics = run_dual_epoch(
            model,
            train_loader,
            device,
            criterion,
            training_config.stage,
            class_weights,
            feature_lookup=train_feature_lookup,
            optimizer=optimizer,
            gradient_clip_norm=training_config.gradient_clip_norm,
            max_batches=training_config.max_train_batches,
        )
        validation_metrics = run_dual_epoch(
            model,
            validation_loader,
            device,
            criterion,
            training_config.stage,
            class_weights,
            feature_lookup=validation_feature_lookup,
            max_batches=training_config.max_validation_batches,
        )
        auc = validation_metrics.get("roc_auc")
        selection_value = (
            float(auc) if auc is not None else -validation_metrics["loss"]
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
            "stage": training_config.stage,
            "learning_rates": [
                float(group["lr"]) for group in optimizer.param_groups
            ],
            "train": train_metrics,
            "validation": validation_metrics,
            "epochs_without_improvement": without_improvement,
        }
        history.append(record)
        payload = _checkpoint_payload(
            model,
            optimizer,
            scheduler,
            epoch,
            history,
            training_config,
            loss_config,
            class_weights,
            protocol_sha256,
            provenance,
            best_epoch,
            best_auc,
        )
        _atomic_torch_save(
            output_dir / "last_checkpoint.pt", payload
        )
        if improved:
            _atomic_torch_save(
                output_dir / "best_checkpoint.pt", payload
            )
        _atomic_json(history_path, history)
        print(
            "epoch {}/{} stage={} train_loss={:.6f} train_auc={} "
            "validation_loss={:.6f} validation_auc={}".format(
                epoch,
                training_config.epochs,
                training_config.stage,
                train_metrics["loss"],
                train_metrics["roc_auc"],
                validation_metrics["loss"],
                validation_metrics["roc_auc"],
            ),
            flush=True,
        )
        if (
            training_config.early_stopping_patience > 0
            and without_improvement
            >= training_config.early_stopping_patience
        ):
            break
    payload = load_dual_checkpoint(
        output_dir / "best_checkpoint.pt",
        model,
        device,
        expected_stage=training_config.stage,
        expected_protocol_sha256=protocol_sha256,
        expected_provenance=provenance,
    )
    validation = run_dual_epoch(
        model,
        validation_loader,
        device,
        criterion,
        training_config.stage,
        class_weights,
        feature_lookup=validation_feature_lookup,
        max_batches=training_config.max_validation_batches,
        include_predictions=True,
    )
    labels = [
        int(item["label"]) for item in validation["predictions"]
    ]
    probabilities = [
        float(item["positive_probability"])
        for item in validation["predictions"]
    ]
    threshold = _fit_threshold(labels, probabilities)
    validation.update(_metrics(labels, probabilities, threshold))
    payload["validation_threshold"] = threshold
    _atomic_torch_save(
        output_dir / "best_checkpoint.pt", payload
    )
    evaluation = {
        "best_epoch": int(payload["best_epoch"]),
        "selection_metric": "validation_roc_auc",
        "validation_threshold": threshold,
        "validation": validation,
    }
    evaluation_path = output_dir / "best_evaluation.json"
    _atomic_json(evaluation_path, evaluation)
    return {
        "stage": training_config.stage,
        "epochs_completed": len(history),
        "best_epoch": int(payload["best_epoch"]),
        "best_validation_roc_auc": float(
            payload["best_validation_roc_auc"]
        ),
        "validation_threshold": threshold,
        "best_checkpoint": output_dir / "best_checkpoint.pt",
        "last_checkpoint": output_dir / "last_checkpoint.pt",
        "history": history_path,
        "best_evaluation": evaluation_path,
    }

