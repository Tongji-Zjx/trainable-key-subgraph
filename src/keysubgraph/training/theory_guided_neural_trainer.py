"""Stage-1 training and frozen-threshold evaluation utilities."""

from __future__ import absolute_import, division, print_function

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import torch

from keysubgraph.models.theory_guided_neural import (
    TheoryGuidedNeuralClassifier,
)
from keysubgraph.training.dual_sgw_feature_trainer import (
    binary_metrics,
    fit_binary_threshold,
)
from keysubgraph.training.sv_signed_gin_trainer import (
    balanced_classification_loss,
    site_stratified_roc_auc,
)
from keysubgraph.training.trainer import (
    class_weights_from_labels,
    set_reproducible_seed,
)


THEORY_NEURAL_CHECKPOINT_SCHEMA_VERSION = 1
THEORY_NEURAL_MODEL_NAME = "svg_theory_guided_neural_stage1"


@dataclass(frozen=True)
class TheoryNeuralTrainingConfig:
    epochs: int = 80
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 2
    early_stopping_patience: int = 15
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    minimum_learning_rate: float = 1.0e-5
    selection_metric: str = "composite_auc"
    quantile_loss_weight: float = 0.05
    transition_loss_weight: float = 0.05
    center_loss_weight: float = 0.02
    auxiliary_warmup_epochs: int = 5
    auxiliary_ramp_epochs: int = 10
    center_momentum: float = 0.90
    seed: int = 42
    max_train_batches: Optional[int] = None
    max_validation_batches: Optional[int] = None

    def __post_init__(self):
        if self.epochs < 1 or self.learning_rate <= 0.0:
            raise ValueError("Stage-1 epochs and learning rate must be positive")
        if self.weight_decay < 0.0 or self.gradient_clip_norm <= 0.0:
            raise ValueError("invalid Stage-1 optimizer configuration")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("Stage-1 accumulation steps must be positive")
        if self.early_stopping_patience < 0 or self.scheduler_patience < 0:
            raise ValueError("Stage-1 patience cannot be negative")
        if not 0.0 < self.scheduler_factor < 1.0:
            raise ValueError("Stage-1 scheduler factor must lie in (0,1)")
        if self.minimum_learning_rate <= 0.0:
            raise ValueError("Stage-1 minimum learning rate must be positive")
        if self.selection_metric not in ("roc_auc", "composite_auc"):
            raise ValueError("unsupported Stage-1 selection metric")
        if min(
            self.quantile_loss_weight,
            self.transition_loss_weight,
            self.center_loss_weight,
        ) < 0.0:
            raise ValueError("Stage-1 auxiliary weights cannot be negative")
        if self.auxiliary_warmup_epochs < 0 or self.auxiliary_ramp_epochs < 1:
            raise ValueError("invalid Stage-1 auxiliary schedule")
        if not 0.0 <= self.center_momentum < 1.0:
            raise ValueError("Stage-1 center momentum must lie in [0,1)")


@dataclass
class EMAClassCenters:
    centers: torch.Tensor
    initialized: torch.Tensor
    momentum: float

    @classmethod
    def create(cls, hidden_dim, device, momentum):
        return cls(
            centers=torch.zeros(2, hidden_dim, device=device),
            initialized=torch.zeros(2, dtype=torch.bool, device=device),
            momentum=float(momentum),
        )

    def loss(self, representations, labels):
        available = self.initialized.index_select(0, labels)
        if not bool(available.any()):
            return representations.new_zeros(())
        targets = self.centers.index_select(0, labels).detach()
        return (representations[available] - targets[available]).square().mean()

    def update(self, representations, labels):
        values = representations.detach()
        with torch.no_grad():
            for label in (0, 1):
                mask = labels == label
                if not bool(mask.any()):
                    continue
                current = values[mask].mean(dim=0)
                if bool(self.initialized[label]):
                    self.centers[label].mul_(self.momentum).add_(
                        current, alpha=1.0 - self.momentum
                    )
                else:
                    self.centers[label].copy_(current)
                    self.initialized[label] = True

    def state_dict(self):
        return {
            "centers": self.centers.detach().cpu(),
            "initialized": self.initialized.detach().cpu(),
            "momentum": self.momentum,
        }


def _atomic_json(path, payload):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def _atomic_torch_save(path, payload):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, str(temporary))
    os.replace(str(temporary), str(path))


def _trusted_load(path, device):
    try:
        return torch.load(
            str(Path(path).resolve()), map_location=device, weights_only=False
        )
    except TypeError:
        return torch.load(str(Path(path).resolve()), map_location=device)


def _site_lookup(loader):
    keys = getattr(loader.dataset, "sample_keys", ())
    sites = getattr(loader.dataset, "sites", ())
    return {str(key): str(site) for key, site in zip(keys, sites)}


def auxiliary_scale(epoch, config):
    if epoch <= config.auxiliary_warmup_epochs:
        return 0.0
    progress = epoch - config.auxiliary_warmup_epochs
    return min(1.0, float(progress) / float(config.auxiliary_ramp_epochs))


def _reconstruction_losses(output):
    q_values = []
    gamma_values = []
    for sample in output.samples:
        if sample.q_predictions is not None and sample.q_predictions.numel():
            q_values.append(
                torch.nn.functional.smooth_l1_loss(
                    sample.q_predictions, sample.q_targets
                )
            )
        if (
            sample.gamma_predictions is not None
            and sample.gamma_predictions.numel()
        ):
            gamma_values.append(
                torch.nn.functional.smooth_l1_loss(
                    sample.gamma_predictions, sample.gamma_targets
                )
            )
    zero = output.logits.new_zeros(())
    return (
        torch.stack(q_values).mean() if q_values else zero,
        torch.stack(gamma_values).mean() if gamma_values else zero,
    )


def run_theory_neural_epoch(
    model: TheoryGuidedNeuralClassifier,
    data_loader: Iterable,
    device: torch.device,
    class_weights: torch.Tensor,
    optimizer=None,
    gradient_clip_norm=1.0,
    gradient_accumulation_steps=1,
    max_batches=None,
    threshold=0.5,
    include_predictions=False,
    quantile_loss_weight=0.0,
    transition_loss_weight=0.0,
    center_loss_weight=0.0,
    class_centers=None,
):
    training = optimizer is not None
    model.train(training)
    labels_all: List[int] = []
    probabilities_all: List[float] = []
    keys_all: List[str] = []
    sites_all: List[str] = []
    sites_by_key = _site_lookup(data_loader)
    loss_sums = {"loss": 0.0, "classification_loss": 0.0,
                 "quantile_loss": 0.0, "transition_loss": 0.0,
                 "center_loss": 0.0}
    sample_count = 0
    gradient_norms = []
    started = time.perf_counter()
    total_batches = len(data_loader)
    if max_batches is not None:
        total_batches = min(total_batches, int(max_batches))
    if training:
        optimizer.zero_grad(set_to_none=True)
    accumulated_samples = 0
    for batch_index, batch in enumerate(data_loader):
        if batch_index >= total_batches:
            break
        batch = batch.to(device)
        labels = batch.labels.to(device)
        with torch.set_grad_enabled(training):
            output = model(batch)
            classification = balanced_classification_loss(
                output.logits, labels, class_weights.to(device)
            )
            quantile, transition = _reconstruction_losses(output)
            center = (
                class_centers.loss(output.representations, labels)
                if class_centers is not None
                else output.logits.new_zeros(())
            )
            loss = (
                classification
                + float(quantile_loss_weight) * quantile
                + float(transition_loss_weight) * transition
                + float(center_loss_weight) * center
            )
            if training:
                count = int(labels.numel())
                (loss * float(count)).backward()
                accumulated_samples += count
                end_group = (
                    (batch_index + 1) % int(gradient_accumulation_steps) == 0
                    or batch_index + 1 == total_batches
                )
                if end_group:
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.div_(float(accumulated_samples))
                    norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(gradient_clip_norm)
                    )
                    gradient_norms.append(float(norm.detach().cpu()))
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    accumulated_samples = 0
                if class_centers is not None:
                    class_centers.update(output.representations, labels)
        probabilities = torch.softmax(output.logits, dim=-1)[:, 1]
        count = int(labels.numel())
        sample_count += count
        for name, value in (
            ("loss", loss),
            ("classification_loss", classification),
            ("quantile_loss", quantile),
            ("transition_loss", transition),
            ("center_loss", center),
        ):
            loss_sums[name] += float(value.detach().cpu()) * count
        labels_all.extend(int(value) for value in labels.detach().cpu().tolist())
        probabilities_all.extend(
            float(value) for value in probabilities.detach().cpu().tolist()
        )
        keys_all.extend(str(value) for value in batch.sample_keys)
        sites_all.extend(sites_by_key.get(str(value), "") for value in batch.sample_keys)
    if sample_count < 1:
        raise ValueError("Stage-1 epoch processed no samples")
    metrics = binary_metrics(labels_all, probabilities_all, threshold)
    site_auc = site_stratified_roc_auc(labels_all, probabilities_all, sites_all)
    metrics.update({
        name: value / float(sample_count) for name, value in loss_sums.items()
    })
    metrics.update({
        "site_stratified_roc_auc": site_auc,
        "composite_auc": (
            0.5 * (float(metrics["roc_auc"]) + float(site_auc))
            if metrics["roc_auc"] is not None and site_auc is not None else None
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "mean_gradient_norm": (
            sum(gradient_norms) / float(len(gradient_norms))
            if gradient_norms else None
        ),
    })
    if include_predictions:
        metrics["predictions"] = [
            {"sample_key": key, "site": site, "label": label,
             "positive_probability": probability}
            for key, site, label, probability in zip(
                keys_all, sites_all, labels_all, probabilities_all
            )
        ]
    return metrics


def _selection_value(metrics, name):
    value = metrics.get(name)
    return float(value) if value is not None else -float(metrics["loss"])


def _checkpoint_payload(model, optimizer, scheduler, epoch, history, config,
                        class_weights, provenance, best_epoch, best_value,
                        centers):
    return {
        "schema_version": THEORY_NEURAL_CHECKPOINT_SCHEMA_VERSION,
        "model_name": THEORY_NEURAL_MODEL_NAME,
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
        "best_selection_value": float(best_value),
        "selection_metric": config.selection_metric,
        "class_centers": centers.state_dict() if centers is not None else None,
        "validation_thresholds": None,
        "threshold_fit_split": "validation",
    }


def load_theory_neural_checkpoint(path, model, device, expected_provenance=None):
    payload = _trusted_load(path, device)
    if (payload.get("schema_version") != THEORY_NEURAL_CHECKPOINT_SCHEMA_VERSION
            or payload.get("model_name") != THEORY_NEURAL_MODEL_NAME):
        raise ValueError("not a Stage-1 theory-neural checkpoint")
    expected = model.config.__class__(**payload["model_config"])
    if asdict(expected) != model.config_dict():
        raise ValueError("Stage-1 model configuration mismatch")
    if expected_provenance is not None and payload.get("provenance") != dict(expected_provenance):
        raise ValueError("Stage-1 checkpoint provenance mismatch")
    model.load_state_dict(payload["model_state_dict"])
    return payload


def train_theory_neural_classifier(model, train_loader, validation_loader,
                                   train_labels, device, config, output_dir,
                                   provenance):
    set_reproducible_seed(config.seed)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError("Stage-1 training output exists")
    model.to(device)
    class_weights = class_weights_from_labels(train_labels)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=config.scheduler_factor,
        patience=config.scheduler_patience, min_lr=config.minimum_learning_rate
    )
    centers = (
        EMAClassCenters.create(model.config.hidden_dim, device, config.center_momentum)
        if model.config.uses_center_loss else None
    )
    history = []
    best_epoch = 0
    best_value = float("-inf")
    without_improvement = 0
    for epoch in range(1, config.epochs + 1):
        scale = auxiliary_scale(epoch, config)
        train = run_theory_neural_epoch(
            model, train_loader, device, class_weights, optimizer=optimizer,
            gradient_clip_norm=config.gradient_clip_norm,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            max_batches=config.max_train_batches,
            quantile_loss_weight=config.quantile_loss_weight * scale,
            transition_loss_weight=config.transition_loss_weight * scale,
            center_loss_weight=config.center_loss_weight * scale,
            class_centers=centers,
        )
        validation = run_theory_neural_epoch(
            model, validation_loader, device, class_weights,
            max_batches=config.max_validation_batches,
            quantile_loss_weight=config.quantile_loss_weight * scale,
            transition_loss_weight=config.transition_loss_weight * scale,
            center_loss_weight=config.center_loss_weight * scale,
            class_centers=centers,
        )
        value = _selection_value(validation, config.selection_metric)
        scheduler.step(value)
        improved = best_epoch == 0 or value > best_value
        if improved:
            best_epoch, best_value, without_improvement = epoch, value, 0
        else:
            without_improvement += 1
        record = {
            "epoch": epoch, "auxiliary_scale": scale,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train": train, "validation": validation,
            "epochs_without_improvement": without_improvement,
        }
        history.append(record)
        payload = _checkpoint_payload(
            model, optimizer, scheduler, epoch, history, config, class_weights,
            provenance, best_epoch, best_value, centers
        )
        _atomic_torch_save(output_dir / "last_checkpoint.pt", payload)
        if improved:
            _atomic_torch_save(output_dir / "best_checkpoint.pt", payload)
        _atomic_json(output_dir / "history.json", history)
        print(
            "epoch {}/{} variant={} train_auc={} validation_auc={} "
            "validation_site_auc={} selection={:.6f} aux_scale={:.3f}".format(
                epoch, config.epochs, model.config.variant, train["roc_auc"],
                validation["roc_auc"], validation["site_stratified_roc_auc"],
                value, scale
            ), flush=True
        )
        if (config.early_stopping_patience > 0
                and without_improvement >= config.early_stopping_patience):
            break
    checkpoint = load_theory_neural_checkpoint(
        output_dir / "best_checkpoint.pt", model, device,
        expected_provenance=provenance
    )
    validation = run_theory_neural_epoch(
        model, validation_loader, device, class_weights,
        max_batches=config.max_validation_batches, include_predictions=True
    )
    labels = [int(item["label"]) for item in validation["predictions"]]
    probabilities = [
        float(item["positive_probability"]) for item in validation["predictions"]
    ]
    thresholds = {
        name: fit_binary_threshold(labels, probabilities, name)
        for name in ("balanced_accuracy", "accuracy")
    }
    checkpoint["validation_thresholds"] = thresholds
    checkpoint["threshold_fit_split"] = "validation"
    _atomic_torch_save(output_dir / "best_checkpoint.pt", checkpoint)
    evaluation = {
        "best_epoch": int(checkpoint["best_epoch"]),
        "selection_metric": config.selection_metric,
        "best_selection_value": float(checkpoint["best_selection_value"]),
        "validation_thresholds": thresholds,
        "metrics": validation,
    }
    _atomic_json(output_dir / "best_evaluation.json", evaluation)
    return evaluation


def evaluate_theory_neural_classifier(model, loader, checkpoint_path, device,
                                      output_path, threshold_strategy,
                                      expected_provenance=None):
    checkpoint = load_theory_neural_checkpoint(
        checkpoint_path, model, device, expected_provenance=expected_provenance
    )
    thresholds = checkpoint.get("validation_thresholds")
    if not isinstance(thresholds, dict) or threshold_strategy not in thresholds:
        raise ValueError("checkpoint has no frozen validation threshold")
    threshold = float(thresholds[threshold_strategy])
    metrics = run_theory_neural_epoch(
        model, loader, device, checkpoint["class_weights"], threshold=threshold,
        include_predictions=True
    )
    result = {
        "artifact_type": "theory_guided_neural_evaluation",
        "variant": model.config.variant,
        "split": str(getattr(loader.dataset, "split", "unknown")),
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_epoch": int(checkpoint["best_epoch"]),
        "threshold": threshold,
        "threshold_strategy": threshold_strategy,
        "threshold_fit_split": "validation",
        "metrics": metrics,
        "predictions": metrics["predictions"],
    }
    _atomic_json(output_path, result)
    return result
