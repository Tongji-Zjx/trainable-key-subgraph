"""Training and frozen-threshold evaluation for the spectral evolution branch."""

from __future__ import absolute_import, division, print_function

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from sklearn.metrics import roc_auc_score

from keysubgraph.models.sv_spectral_evolution import (
    SVSpectralEvolutionClassifier,
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


SV_SPECTRAL_EVOLUTION_CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SVSpectralEvolutionTrainingConfig:
    epochs: int = 80
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 15
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    minimum_learning_rate: float = 1.0e-5
    selection_metric: str = "composite_auc"
    auxiliary_loss_weight: float = 0.25
    seed: int = 42
    max_train_batches: Optional[int] = None
    max_validation_batches: Optional[int] = None

    def __post_init__(self) -> None:
        if (
            self.epochs < 1
            or self.learning_rate <= 0.0
            or self.weight_decay < 0.0
            or self.gradient_clip_norm <= 0.0
            or self.early_stopping_patience < 0
            or self.scheduler_patience < 0
            or not 0.0 < self.scheduler_factor < 1.0
            or self.minimum_learning_rate <= 0.0
            or self.auxiliary_loss_weight < 0.0
        ):
            raise ValueError("invalid spectral evolution training config")
        if self.selection_metric not in ("roc_auc", "composite_auc"):
            raise ValueError("unsupported spectral evolution selection metric")


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


def _site_lookup(data_loader: Iterable) -> Dict[str, str]:
    keys = getattr(data_loader.dataset, "sample_keys", ())
    sites = getattr(data_loader.dataset, "sites", ())
    if len(keys) != len(sites):
        return {}
    return {str(key): str(site) for key, site in zip(keys, sites)}


def _safe_auc(labels: List[int], probabilities: List[float]):
    return (
        float(roc_auc_score(labels, probabilities))
        if len(set(labels)) == 2
        else None
    )


def run_sv_spectral_evolution_epoch(
    model: SVSpectralEvolutionClassifier,
    data_loader: Iterable,
    device: torch.device,
    class_weights: torch.Tensor,
    optimizer: Optional[torch.optim.Optimizer] = None,
    gradient_clip_norm: float = 1.0,
    max_batches: Optional[int] = None,
    threshold: float = 0.5,
    include_predictions: bool = False,
    auxiliary_loss_weight: float = 0.25,
) -> Dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    labels_all: List[int] = []
    probabilities_all: List[float] = []
    anchor_probabilities: List[float] = []
    dynamic_probabilities: List[float] = []
    keys_all: List[str] = []
    sites_all: List[str] = []
    loss_total = 0.0
    main_total = 0.0
    auxiliary_total = 0.0
    sample_total = 0
    gate_values: List[float] = []
    transition_counts: List[int] = []
    sites_by_key = _site_lookup(data_loader)
    started = time.perf_counter()
    for batch_index, batch in enumerate(data_loader):
        if max_batches is not None and batch_index >= int(max_batches):
            break
        batch = batch.to(device)
        labels = batch.labels.to(device)
        with torch.set_grad_enabled(training):
            output = model(batch)
            main = balanced_classification_loss(
                output.logits, labels, class_weights.to(device)
            )
            auxiliary = balanced_classification_loss(
                output.dynamic_logits,
                labels,
                class_weights.to(device),
            )
            loss = main + float(auxiliary_loss_weight) * auxiliary
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ],
                float(gradient_clip_norm),
            )
            optimizer.step()
        count = int(labels.shape[0])
        probabilities = torch.softmax(output.logits.detach(), dim=-1)[:, 1]
        anchor = torch.softmax(
            output.anchor_logits.detach(), dim=-1
        )[:, 1]
        dynamic = torch.softmax(
            output.dynamic_logits.detach(), dim=-1
        )[:, 1]
        labels_all.extend(int(value) for value in labels.detach().cpu())
        probabilities_all.extend(
            float(value) for value in probabilities.cpu()
        )
        anchor_probabilities.extend(float(value) for value in anchor.cpu())
        dynamic_probabilities.extend(
            float(value) for value in dynamic.cpu()
        )
        keys = list(batch.sample_keys)
        keys_all.extend(keys)
        sites_all.extend(sites_by_key.get(key, "") for key in keys)
        transition_counts.extend(
            int(value) for value in output.transition_counts.detach().cpu()
        )
        gate_values.append(
            float(output.residual_gate.detach().cpu().item())
        )
        loss_total += float(loss.detach().cpu().item()) * count
        main_total += float(main.detach().cpu().item()) * count
        auxiliary_total += (
            float(auxiliary.detach().cpu().item()) * count
        )
        sample_total += count
    if sample_total < 1:
        raise ValueError("spectral evolution epoch processed no samples")
    metrics = binary_metrics(labels_all, probabilities_all, threshold)
    site_auc = site_stratified_roc_auc(
        labels_all, probabilities_all, sites_all
    )
    roc_auc = metrics["roc_auc"]
    composite = (
        0.5 * (float(roc_auc) + float(site_auc))
        if roc_auc is not None and site_auc is not None
        else roc_auc
    )
    result = {
        **metrics,
        "loss": loss_total / sample_total,
        "main_loss": main_total / sample_total,
        "dynamic_auxiliary_loss": auxiliary_total / sample_total,
        "anchor_roc_auc": _safe_auc(labels_all, anchor_probabilities),
        "dynamic_roc_auc": _safe_auc(labels_all, dynamic_probabilities),
        "site_stratified_roc_auc": site_auc,
        "composite_auc": composite,
        "residual_gate": sum(gate_values) / len(gate_values),
        "mean_transition_count": (
            sum(transition_counts) / float(len(transition_counts))
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    if include_predictions:
        result["predictions"] = [
            {
                "sample_key": key,
                "site": site,
                "label": label,
                "positive_probability": probability,
                "anchor_positive_probability": anchor_probability,
                "dynamic_positive_probability": dynamic_probability,
            }
            for (
                key,
                site,
                label,
                probability,
                anchor_probability,
                dynamic_probability,
            ) in zip(
                keys_all,
                sites_all,
                labels_all,
                probabilities_all,
                anchor_probabilities,
                dynamic_probabilities,
            )
        ]
    return result


def _selection_value(metrics: Dict[str, Any], name: str) -> float:
    value = metrics.get(name)
    return float(value) if value is not None else float("-inf")


def _checkpoint_payload(
    model,
    optimizer,
    scheduler,
    epoch,
    history,
    config,
    class_weights,
    provenance,
    best_epoch,
    best_value,
):
    return {
        "schema_version": SV_SPECTRAL_EVOLUTION_CHECKPOINT_SCHEMA_VERSION,
        "model_name": model.model_name,
        "model_config": model.config_dict(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": int(epoch),
        "history": history,
        "training_config": asdict(config),
        "class_weights": class_weights.detach().cpu(),
        "provenance": dict(provenance),
        "best_epoch": int(best_epoch),
        "best_selection_value": float(best_value),
    }


def load_sv_spectral_evolution_checkpoint(
    path: Path,
    model: SVSpectralEvolutionClassifier,
    device: torch.device,
    expected_provenance: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = _trusted_load(path, device)
    if (
        payload.get("schema_version")
        != SV_SPECTRAL_EVOLUTION_CHECKPOINT_SCHEMA_VERSION
        or payload.get("model_name") != model.model_name
        or payload.get("model_config") != model.config_dict()
    ):
        raise ValueError("unsupported spectral evolution checkpoint")
    if expected_provenance is not None and payload.get(
        "provenance"
    ) != expected_provenance:
        raise ValueError("spectral evolution checkpoint provenance mismatch")
    model.load_state_dict(payload["model_state_dict"])
    return payload


def train_sv_spectral_evolution_classifier(
    model: SVSpectralEvolutionClassifier,
    train_loader: Iterable,
    validation_loader: Iterable,
    train_labels: Iterable[int],
    device: torch.device,
    config: SVSpectralEvolutionTrainingConfig,
    output_dir: Path,
    provenance: Dict[str, Any],
) -> Dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("spectral evolution output exists")
    output_dir.mkdir(parents=True, exist_ok=True)
    set_reproducible_seed(config.seed)
    model.to(device)
    class_weights = class_weights_from_labels(train_labels).to(torch.float32)
    trainable = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if not trainable:
        raise ValueError("spectral evolution has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable,
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
    history = []
    best_epoch = 0
    best_value = float("-inf")
    without_improvement = 0
    for epoch in range(1, config.epochs + 1):
        train = run_sv_spectral_evolution_epoch(
            model,
            train_loader,
            device,
            class_weights,
            optimizer=optimizer,
            gradient_clip_norm=config.gradient_clip_norm,
            max_batches=config.max_train_batches,
            auxiliary_loss_weight=config.auxiliary_loss_weight,
        )
        validation = run_sv_spectral_evolution_epoch(
            model,
            validation_loader,
            device,
            class_weights,
            max_batches=config.max_validation_batches,
            auxiliary_loss_weight=config.auxiliary_loss_weight,
        )
        value = _selection_value(validation, config.selection_metric)
        scheduler.step(value)
        improved = best_epoch == 0 or value > best_value
        if improved:
            best_epoch = epoch
            best_value = value
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
            best_value,
        )
        _atomic_torch_save(output_dir / "last_checkpoint.pt", payload)
        if improved:
            _atomic_torch_save(
                output_dir / "best_checkpoint.pt", payload
            )
        _atomic_json(output_dir / "history.json", history)
        print(
            "epoch {}/{} train_loss={:.6f} train_auc={} "
            "validation_loss={:.6f} validation_auc={} "
            "validation_site_auc={} anchor_auc={} dynamic_auc={} "
            "gate={:.6f} selection={:.6f}".format(
                epoch,
                config.epochs,
                train["loss"],
                train["roc_auc"],
                validation["loss"],
                validation["roc_auc"],
                validation["site_stratified_roc_auc"],
                validation["anchor_roc_auc"],
                validation["dynamic_roc_auc"],
                validation["residual_gate"],
                value,
            ),
            flush=True,
        )
        if (
            config.early_stopping_patience > 0
            and without_improvement >= config.early_stopping_patience
        ):
            break

    payload = load_sv_spectral_evolution_checkpoint(
        output_dir / "best_checkpoint.pt",
        model,
        device,
        expected_provenance=provenance,
    )
    validation = run_sv_spectral_evolution_epoch(
        model,
        validation_loader,
        device,
        class_weights,
        include_predictions=True,
        auxiliary_loss_weight=config.auxiliary_loss_weight,
    )
    labels = [int(row["label"]) for row in validation["predictions"]]
    probabilities = [
        float(row["positive_probability"])
        for row in validation["predictions"]
    ]
    thresholds = {
        "balanced_accuracy": fit_binary_threshold(
            labels, probabilities, "balanced_accuracy"
        ),
        "accuracy": fit_binary_threshold(
            labels, probabilities, "accuracy"
        ),
    }
    payload["validation_thresholds"] = thresholds
    _atomic_torch_save(output_dir / "best_checkpoint.pt", payload)
    evaluation = {
        "best_epoch": int(payload["best_epoch"]),
        "selection_metric": config.selection_metric,
        "best_selection_value": float(payload["best_selection_value"]),
        "validation_thresholds": thresholds,
        "residual_gate": validation["residual_gate"],
        "metrics": {
            name: {
                **binary_metrics(labels, probabilities, threshold),
                "site_stratified_roc_auc": validation[
                    "site_stratified_roc_auc"
                ],
                "composite_auc": validation["composite_auc"],
            }
            for name, threshold in thresholds.items()
        },
        "predictions": validation["predictions"],
    }
    _atomic_json(output_dir / "best_evaluation.json", evaluation)
    return {
        "epochs_completed": len(history),
        "best_epoch": int(payload["best_epoch"]),
        "best_selection_value": float(payload["best_selection_value"]),
        "residual_gate": validation["residual_gate"],
        "best_checkpoint": str(output_dir / "best_checkpoint.pt"),
        "best_evaluation": str(output_dir / "best_evaluation.json"),
    }
