"""Training, checkpointing and evaluation for the revised short-term branch."""

from __future__ import absolute_import, division, print_function

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from sklearn.metrics import roc_auc_score

from keysubgraph.features.structured_short_term_features import (
    StructuredShortTermStandardizer,
)
from keysubgraph.models.structured_short_term import (
    StructuredShortTermClassifier,
    StructuredShortTermConfig,
)
from keysubgraph.training.dual_sgw_feature_trainer import (
    binary_metrics,
    fit_binary_threshold,
)
from keysubgraph.training.trainer import (
    class_weights_from_labels,
    set_reproducible_seed,
)


STRUCTURED_SHORT_TERM_CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class StructuredShortTermTrainingConfig:
    epochs: int = 80
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 15
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    minimum_learning_rate: float = 1.0e-5
    selection_metric: str = "roc_auc"
    seed: int = 42
    max_train_batches: Optional[int] = None
    max_validation_batches: Optional[int] = None

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.learning_rate <= 0.0:
            raise ValueError("short-term epochs and learning rate must be positive")
        if self.weight_decay < 0.0 or self.gradient_clip_norm <= 0.0:
            raise ValueError("short-term optimizer configuration is invalid")
        if self.early_stopping_patience < 0 or self.scheduler_patience < 0:
            raise ValueError("short-term patience values cannot be negative")
        if not 0.0 < self.scheduler_factor < 1.0:
            raise ValueError("short-term scheduler factor must lie in (0,1)")
        if self.minimum_learning_rate <= 0.0:
            raise ValueError("short-term minimum learning rate must be positive")
        if self.selection_metric not in (
            "roc_auc",
            "balanced_accuracy",
            "loss",
        ):
            raise ValueError("unsupported short-term selection metric")
        for limit in (self.max_train_batches, self.max_validation_batches):
            if limit is not None and limit < 1:
                raise ValueError("short-term batch limits must be positive")


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


def _site_stratified_auc(
    labels: List[int],
    probabilities: List[float],
    sites: List[str],
) -> Optional[float]:
    numerator = 0.0
    denominator = 0.0
    for site in sorted(set(sites)):
        indices = [
            index for index, value in enumerate(sites) if value == site
        ]
        site_labels = [labels[index] for index in indices]
        if set(site_labels) != {0, 1}:
            continue
        site_probabilities = [probabilities[index] for index in indices]
        positive = sum(value == 1 for value in site_labels)
        negative = sum(value == 0 for value in site_labels)
        pair_count = float(positive * negative)
        numerator += float(
            roc_auc_score(site_labels, site_probabilities)
        ) * pair_count
        denominator += pair_count
    return numerator / denominator if denominator > 0.0 else None


def run_structured_short_term_epoch(
    model: StructuredShortTermClassifier,
    data_loader: Iterable,
    device: torch.device,
    class_weights: torch.Tensor,
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
    sample_keys_all: List[str] = []
    sites_all: List[str] = []
    weighted_loss_total = 0.0
    unweighted_loss_total = 0.0
    sample_total = 0
    gradient_norms: List[float] = []
    memory_entropies: List[float] = []
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
                output.logits,
                labels,
                reduction="none",
            )
            weighted_loss = (per_sample * weights[labels]).mean()
            if training:
                weighted_loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    gradient_clip_norm,
                )
                gradient_norms.append(float(gradient_norm.detach().cpu()))
                optimizer.step()
        count = int(labels.numel())
        sample_total += count
        weighted_loss_total += float(weighted_loss.detach().cpu()) * count
        unweighted_loss_total += float(per_sample.detach().sum().cpu())
        probabilities = torch.softmax(output.logits, dim=-1)[:, 1]
        labels_all.extend(int(value) for value in labels.detach().cpu().tolist())
        probabilities_all.extend(
            float(value) for value in probabilities.detach().cpu().tolist()
        )
        sample_keys_all.extend(batch.sample_keys)
        sites_all.extend(str(sample.site) for sample in batch.samples)
        entropy = -(
            output.memory_attention
            * output.memory_attention.clamp_min(1.0e-12).log()
        ).sum(dim=1)
        slot_count = float(output.memory_attention.shape[1])
        entropy_denominator = max(1.0, float(torch.log(
            output.memory_attention.new_tensor(slot_count)
        ).item()))
        entropy = entropy / entropy_denominator
        memory_entropies.extend(
            float(value) for value in entropy.detach().cpu().tolist()
        )

    if sample_total <= 0:
        raise ValueError("short-term epoch processed no samples")
    metrics = binary_metrics(labels_all, probabilities_all, threshold)
    metrics.update(
        {
            "loss": weighted_loss_total / float(sample_total),
            "weighted_loss": weighted_loss_total / float(sample_total),
            "unweighted_log_loss": unweighted_loss_total / float(sample_total),
            "site_stratified_roc_auc": _site_stratified_auc(
                labels_all,
                probabilities_all,
                sites_all,
            ),
            "mean_gradient_norm": (
                sum(gradient_norms) / float(len(gradient_norms))
                if gradient_norms
                else None
            ),
            "mean_normalized_memory_entropy": (
                sum(memory_entropies) / float(len(memory_entropies))
            ),
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    if include_predictions:
        metrics["predictions"] = [
            {
                "sample_key": key,
                "site": site,
                "label": label,
                "positive_probability": probability,
                "prediction": int(probability >= threshold),
            }
            for key, site, label, probability in zip(
                sample_keys_all,
                sites_all,
                labels_all,
                probabilities_all,
            )
        ]
    return metrics


def _selection_key(
    metrics: Dict[str, Any],
    selection_metric: str,
) -> Tuple[float, float, float]:
    auc = (
        float(metrics["roc_auc"])
        if metrics.get("roc_auc") is not None
        else float("-inf")
    )
    balanced_accuracy = float(metrics["balanced_accuracy"])
    negative_loss = -float(metrics["unweighted_log_loss"])
    if selection_metric == "roc_auc":
        return auc, balanced_accuracy, negative_loss
    if selection_metric == "balanced_accuracy":
        return balanced_accuracy, auc, negative_loss
    return negative_loss, auc, balanced_accuracy


def _checkpoint_payload(
    model: StructuredShortTermClassifier,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    history: List[Dict[str, Any]],
    training_config: StructuredShortTermTrainingConfig,
    class_weights: torch.Tensor,
    protocol_path: Path,
    protocol_sha256: str,
    standardizer_path: Path,
    standardizer_sha256: str,
    best_epoch: int,
    best_key: Tuple[float, float, float],
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    return {
        "model_name": model.model_name,
        "schema_version": STRUCTURED_SHORT_TERM_CHECKPOINT_SCHEMA_VERSION,
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "model_config": model.config.to_dict(),
        "standardizer": model.standardizer.to_dict(),
        "standardizer_path": str(Path(standardizer_path).resolve()),
        "standardizer_sha256": str(standardizer_sha256),
        "training_config": asdict(training_config),
        "class_weights": class_weights.detach().cpu(),
        "protocol_path": str(Path(protocol_path).resolve()),
        "protocol_sha256": str(protocol_sha256),
        "history": list(history),
        "best_epoch": int(best_epoch),
        "best_selection_key": [float(value) for value in best_key],
        "validation_thresholds": dict(thresholds or {}),
        "torch_version": str(torch.__version__),
    }


def load_structured_short_term_checkpoint(
    path: Path,
    model: StructuredShortTermClassifier,
    device: torch.device,
    expected_protocol_sha256: Optional[str] = None,
    expected_standardizer_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    payload = _trusted_load(path, device)
    if payload.get("model_name") != model.model_name:
        raise ValueError("not a structured short-term checkpoint")
    if (
        payload.get("schema_version")
        != STRUCTURED_SHORT_TERM_CHECKPOINT_SCHEMA_VERSION
    ):
        raise ValueError("unsupported structured short-term checkpoint schema")
    if payload.get("model_config") != model.config.to_dict():
        raise ValueError("structured short-term model configuration mismatch")
    if payload.get("standardizer") != model.standardizer.to_dict():
        raise ValueError("structured short-term standardizer mismatch")
    if (
        expected_protocol_sha256 is not None
        and payload.get("protocol_sha256") != expected_protocol_sha256
    ):
        raise ValueError("structured short-term protocol hash mismatch")
    if (
        expected_standardizer_sha256 is not None
        and payload.get("standardizer_sha256")
        != expected_standardizer_sha256
    ):
        raise ValueError("structured short-term standardizer hash mismatch")
    model.load_state_dict(payload["model_state_dict"])
    return payload


def model_from_structured_short_term_checkpoint(
    path: Path,
    device: torch.device,
) -> Tuple[StructuredShortTermClassifier, Dict[str, Any]]:
    payload = _trusted_load(path, device)
    if payload.get("model_name") != StructuredShortTermClassifier.model_name:
        raise ValueError("not a structured short-term checkpoint")
    config = StructuredShortTermConfig.from_dict(payload["model_config"])
    standardizer = StructuredShortTermStandardizer.from_dict(
        payload["standardizer"]
    )
    model = StructuredShortTermClassifier(config, standardizer).to(device)
    model.load_state_dict(payload["model_state_dict"])
    return model, payload


def train_structured_short_term(
    model: StructuredShortTermClassifier,
    train_loader: Iterable,
    validation_loader: Iterable,
    train_labels: Iterable[int],
    device: torch.device,
    training_config: StructuredShortTermTrainingConfig,
    output_dir: Path,
    protocol_path: Path,
    protocol_sha256: str,
    standardizer_path: Path,
    standardizer_sha256: str,
    resume_checkpoint: Optional[Path] = None,
) -> Dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.json"
    if history_path.exists() and resume_checkpoint is None:
        raise FileExistsError("structured short-term training output exists")
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
        mode=("min" if training_config.selection_metric == "loss" else "max"),
        factor=training_config.scheduler_factor,
        patience=training_config.scheduler_patience,
        min_lr=training_config.minimum_learning_rate,
    )
    history: List[Dict[str, Any]] = []
    best_epoch = 0
    best_key = (float("-inf"), float("-inf"), float("-inf"))
    start_epoch = 1
    epochs_without_improvement = 0
    if resume_checkpoint is not None:
        payload = load_structured_short_term_checkpoint(
            resume_checkpoint,
            model,
            device,
            protocol_sha256,
            standardizer_sha256,
        )
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        history = list(payload["history"])
        best_epoch = int(payload["best_epoch"])
        best_key = tuple(
            float(value) for value in payload["best_selection_key"]
        )
        start_epoch = int(payload["epoch"]) + 1
        if history:
            epochs_without_improvement = int(
                history[-1].get("epochs_without_improvement", 0)
            )

    for epoch in range(start_epoch, training_config.epochs + 1):
        train_metrics = run_structured_short_term_epoch(
            model,
            train_loader,
            device,
            class_weights,
            optimizer=optimizer,
            gradient_clip_norm=training_config.gradient_clip_norm,
            max_batches=training_config.max_train_batches,
        )
        validation_metrics = run_structured_short_term_epoch(
            model,
            validation_loader,
            device,
            class_weights,
            max_batches=training_config.max_validation_batches,
        )
        key = _selection_key(
            validation_metrics,
            training_config.selection_metric,
        )
        scheduler_value = (
            validation_metrics["unweighted_log_loss"]
            if training_config.selection_metric == "loss"
            else key[0]
        )
        scheduler.step(scheduler_value)
        improved = best_epoch == 0 or key > best_key
        if improved:
            best_epoch = epoch
            best_key = key
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
        checkpoint = _checkpoint_payload(
            model,
            optimizer,
            scheduler,
            epoch,
            history,
            training_config,
            class_weights,
            protocol_path,
            protocol_sha256,
            standardizer_path,
            standardizer_sha256,
            best_epoch,
            best_key,
        )
        _atomic_torch_save(output_dir / "last_checkpoint.pt", checkpoint)
        if improved:
            _atomic_torch_save(output_dir / "best_checkpoint.pt", checkpoint)
        _atomic_json(history_path, history)
        print(
            "epoch {}/{} train_loss={:.6f} train_ba={:.6f} train_auc={} "
            "validation_loss={:.6f} validation_ba={:.6f} "
            "validation_auc={} memory_entropy={:.6f} lr={:.8f}".format(
                epoch,
                training_config.epochs,
                train_metrics["loss"],
                train_metrics["balanced_accuracy"],
                train_metrics["roc_auc"],
                validation_metrics["loss"],
                validation_metrics["balanced_accuracy"],
                validation_metrics["roc_auc"],
                validation_metrics["mean_normalized_memory_entropy"],
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
        raise RuntimeError("structured short-term training saved no checkpoint")

    best_path = output_dir / "best_checkpoint.pt"
    best_payload = load_structured_short_term_checkpoint(
        best_path,
        model,
        device,
        protocol_sha256,
        standardizer_sha256,
    )
    best_train = run_structured_short_term_epoch(
        model,
        train_loader,
        device,
        class_weights,
        max_batches=training_config.max_train_batches,
    )
    validation_raw = run_structured_short_term_epoch(
        model,
        validation_loader,
        device,
        class_weights,
        max_batches=training_config.max_validation_batches,
        include_predictions=True,
    )
    labels = [
        int(record["label"]) for record in validation_raw["predictions"]
    ]
    probabilities = [
        float(record["positive_probability"])
        for record in validation_raw["predictions"]
    ]
    thresholds = {
        metric: fit_binary_threshold(labels, probabilities, metric)
        for metric in ("balanced_accuracy", "accuracy")
    }
    validation_by_threshold = {
        metric: binary_metrics(labels, probabilities, threshold)
        for metric, threshold in thresholds.items()
    }
    best_payload["validation_thresholds"] = thresholds
    _atomic_torch_save(best_path, best_payload)
    evaluation_path = output_dir / "best_evaluation.json"
    _atomic_json(
        evaluation_path,
        {
            "best_epoch": best_epoch,
            "selection_metric": training_config.selection_metric,
            "train": best_train,
            "validation_raw_threshold_0_5": validation_raw,
            "validation_thresholds": thresholds,
            "validation": validation_by_threshold,
        },
    )
    return {
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_checkpoint": best_path,
        "last_checkpoint": output_dir / "last_checkpoint.pt",
        "history": history_path,
        "best_evaluation": evaluation_path,
        "validation_thresholds": thresholds,
    }


def evaluate_structured_short_term(
    model: StructuredShortTermClassifier,
    data_loader: Iterable,
    device: torch.device,
    class_weights: torch.Tensor,
    threshold: float,
) -> Dict[str, Any]:
    return run_structured_short_term_epoch(
        model,
        data_loader,
        device,
        class_weights,
        threshold=threshold,
        include_predictions=True,
    )
