"""Training, checkpointing and frozen-threshold evaluation for SV Signed-GIN."""

from __future__ import absolute_import, division, print_function

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import torch
from sklearn.metrics import roc_auc_score

from keysubgraph.models.sv_signed_gin import SVSignedGINClassifier
from keysubgraph.training.dual_sgw_feature_trainer import (
    binary_metrics,
    fit_binary_threshold,
)
from keysubgraph.training.trainer import (
    class_weights_from_labels,
    set_reproducible_seed,
)


SV_SIGNED_GIN_CHECKPOINT_SCHEMA_VERSION = 1
SV_SIGNED_GIN_MODEL_NAME = "sv_hard_sgw_signed_gin"


@dataclass(frozen=True)
class SVSignedGINTrainingConfig:
    epochs: int = 80
    static_anchor_epochs: int = 80
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    early_stopping_patience: int = 15
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    minimum_learning_rate: float = 1.0e-5
    selection_metric: str = "composite_auc"
    auxiliary_loss_weight: float = 0.0
    residual_gate_penalty_weight: float = 0.01
    seed: int = 42
    max_train_batches: Optional[int] = None
    max_validation_batches: Optional[int] = None

    def __post_init__(self) -> None:
        if (
            self.epochs < 1
            or self.static_anchor_epochs < 1
            or self.learning_rate <= 0.0
        ):
            raise ValueError("SV epochs and learning rate must be positive")
        if self.weight_decay < 0.0 or self.gradient_clip_norm <= 0.0:
            raise ValueError("invalid SV optimizer configuration")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("SV accumulation steps must be positive")
        if self.early_stopping_patience < 0 or self.scheduler_patience < 0:
            raise ValueError("SV patience values cannot be negative")
        if not 0.0 < self.scheduler_factor < 1.0:
            raise ValueError("SV scheduler factor must lie in (0,1)")
        if self.minimum_learning_rate <= 0.0:
            raise ValueError("SV minimum learning rate must be positive")
        if self.auxiliary_loss_weight < 0.0:
            raise ValueError("SV auxiliary loss weight cannot be negative")
        if self.residual_gate_penalty_weight < 0.0:
            raise ValueError(
                "SV residual gate penalty weight cannot be negative"
            )
        if self.selection_metric not in ("roc_auc", "composite_auc"):
            raise ValueError("unsupported SV selection metric")


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


def balanced_classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    """Balanced empirical risk whose weights do not cancel at batch size 1."""
    if logits.ndim != 2 or logits.shape[-1] != 2:
        raise ValueError("SV logits must have shape [B,2]")
    if tuple(labels.shape) != (logits.shape[0],):
        raise ValueError("SV labels do not align with logits")
    if tuple(class_weights.shape) != (2,):
        raise ValueError("SV class weights must have shape [2]")
    per_sample = torch.nn.functional.cross_entropy(
        logits, labels, reduction="none"
    )
    weights = class_weights.to(logits).index_select(0, labels)
    # class_weights_from_labels returns N/(2*n_c), whose expectation under the
    # train distribution is exactly one.  Do not divide by the current
    # microbatch weight sum: that would erase weighting for batch size 1.
    return (per_sample * weights).mean()


def site_stratified_roc_auc(
    labels: List[int],
    probabilities: List[float],
    sites: List[str],
) -> Optional[float]:
    if not (len(labels) == len(probabilities) == len(sites)):
        raise ValueError("SV site-stratified vectors are misaligned")
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
        numerator += (
            float(roc_auc_score(site_labels, site_probabilities))
            * pair_count
        )
        denominator += pair_count
    return numerator / denominator if denominator > 0.0 else None


def _site_lookup(data_loader: Iterable) -> Dict[str, str]:
    dataset = data_loader.dataset
    keys = getattr(dataset, "sample_keys", ())
    sites = getattr(dataset, "sites", ())
    if len(keys) != len(sites):
        return {}
    return {str(key): str(site) for key, site in zip(keys, sites)}


def run_sv_signed_gin_epoch(
    model: SVSignedGINClassifier,
    data_loader: Iterable,
    device: torch.device,
    class_weights: torch.Tensor,
    optimizer: Optional[torch.optim.Optimizer] = None,
    gradient_clip_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    max_batches: Optional[int] = None,
    threshold: float = 0.5,
    include_predictions: bool = False,
    auxiliary_loss_weight: float = 0.0,
    residual_gate_penalty_weight: float = 0.0,
) -> Dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    if gradient_accumulation_steps < 1:
        raise ValueError("SV accumulation steps must be positive")
    labels_all: List[int] = []
    probabilities_all: List[float] = []
    keys_all: List[str] = []
    sites_all: List[str] = []
    branch_probabilities: Dict[str, List[float]] = {}
    fusion_weight_values: List[List[float]] = []
    residual_gate_values: Dict[str, List[float]] = {}
    loss_total = 0.0
    main_loss_total = 0.0
    auxiliary_loss_total = 0.0
    gate_penalty_total = 0.0
    sample_total = 0
    gradient_norms: List[float] = []
    started = time.perf_counter()
    sites_by_key = _site_lookup(data_loader)
    total_batches = len(data_loader)
    if max_batches is not None:
        total_batches = min(total_batches, int(max_batches))
    if training:
        optimizer.zero_grad(set_to_none=True)
    accumulated_sample_count = 0
    for batch_index, batch in enumerate(data_loader):
        if batch_index >= total_batches:
            break
        batch = batch.to(device)
        labels = batch.labels.to(device)
        with torch.set_grad_enabled(training):
            output = model(batch)
            main_loss = balanced_classification_loss(
                output.logits, labels, class_weights.to(device)
            )
            auxiliary_loss = output.logits.new_zeros(())
            if output.branch_logits:
                auxiliary_names = (
                    tuple(
                        name
                        for name in output.residual_gates
                        if name in output.branch_logits
                    )
                    if output.residual_gates
                    else tuple(output.branch_logits)
                )
                auxiliary_values = [
                    balanced_classification_loss(
                        output.branch_logits[name],
                        labels,
                        class_weights.to(device),
                    )
                    for name in auxiliary_names
                ]
                auxiliary_loss = torch.stack(
                    auxiliary_values
                ).mean()
            gate_penalty = output.logits.new_zeros(())
            if output.residual_gates:
                gate_penalty = torch.stack(
                    list(output.residual_gates.values())
                ).sum()
            loss = (
                main_loss
                + float(auxiliary_loss_weight) * auxiliary_loss
                + float(residual_gate_penalty_weight) * gate_penalty
            )
            if training:
                count = int(labels.numel())
                # Accumulate a sample sum first, then divide gradients by the
                # exact number of samples in this optimizer step.  This stays
                # equivalent to one effective batch even when the final
                # physical batch is smaller.
                (loss * float(count)).backward()
                accumulated_sample_count += count
                end_group = (
                    (batch_index + 1) % gradient_accumulation_steps == 0
                    or batch_index + 1 == total_batches
                )
                if end_group:
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.div_(
                                float(accumulated_sample_count)
                            )
                    norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), gradient_clip_norm
                    )
                    gradient_norms.append(float(norm.detach().cpu()))
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    accumulated_sample_count = 0
        probabilities = torch.softmax(output.logits, dim=-1)[:, 1]
        if output.branch_logits:
            for name, branch_logits in output.branch_logits.items():
                branch_probabilities.setdefault(name, []).extend(
                    float(value)
                    for value in torch.softmax(
                        branch_logits, dim=-1
                    )[:, 1].detach().cpu().tolist()
                )
        if output.fusion_weights is not None:
            fusion_weight_values.append(
                [
                    float(value)
                    for value in output.fusion_weights.detach().cpu().tolist()
                ]
            )
        if output.residual_gates:
            for name, value in output.residual_gates.items():
                residual_gate_values.setdefault(name, []).append(
                    float(value.detach().cpu())
                )
        count = int(labels.numel())
        sample_total += count
        loss_total += float(loss.detach().cpu()) * count
        main_loss_total += float(main_loss.detach().cpu()) * count
        auxiliary_loss_total += (
            float(auxiliary_loss.detach().cpu()) * count
        )
        gate_penalty_total += (
            float(gate_penalty.detach().cpu()) * count
        )
        batch_labels = [
            int(value) for value in labels.detach().cpu().tolist()
        ]
        batch_probabilities = [
            float(value)
            for value in probabilities.detach().cpu().tolist()
        ]
        labels_all.extend(batch_labels)
        probabilities_all.extend(batch_probabilities)
        keys_all.extend(str(value) for value in batch.sample_keys)
        sites_all.extend(
            sites_by_key.get(str(value), "") for value in batch.sample_keys
        )
    if sample_total < 1:
        raise ValueError("SV epoch processed no samples")
    metrics = binary_metrics(labels_all, probabilities_all, threshold)
    stratified = (
        site_stratified_roc_auc(
            labels_all, probabilities_all, sites_all
        )
        if sites_by_key
        else None
    )
    metrics.update(
        {
            "loss": loss_total / float(sample_total),
            "main_loss": main_loss_total / float(sample_total),
            "auxiliary_loss": (
                auxiliary_loss_total / float(sample_total)
            ),
            "residual_gate_penalty": (
                gate_penalty_total / float(sample_total)
            ),
            "site_stratified_roc_auc": stratified,
            "composite_auc": (
                0.5 * (float(metrics["roc_auc"]) + float(stratified))
                if metrics["roc_auc"] is not None
                and stratified is not None
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
    if branch_probabilities:
        branch_metrics = {}
        for name, values in sorted(branch_probabilities.items()):
            current = binary_metrics(labels_all, values, threshold)
            current["site_stratified_roc_auc"] = (
                site_stratified_roc_auc(labels_all, values, sites_all)
                if sites_by_key
                else None
            )
            branch_metrics[name] = current
        metrics["branch_metrics"] = branch_metrics
    if fusion_weight_values:
        names = ("gin", "static_spectral", "variation")
        metrics["fusion_weights"] = {
            name: sum(values[index] for values in fusion_weight_values)
            / float(len(fusion_weight_values))
            for index, name in enumerate(names)
        }
    if residual_gate_values:
        metrics["residual_gates"] = {
            name: sum(values) / float(len(values))
            for name, values in sorted(residual_gate_values.items())
        }
    if include_predictions:
        predictions = []
        for index, (key, site, label, probability) in enumerate(
            zip(keys_all, sites_all, labels_all, probabilities_all)
        ):
            item = {
                "sample_key": key,
                "site": site,
                "label": label,
                "positive_probability": probability,
            }
            if branch_probabilities:
                item["branch_positive_probabilities"] = {
                    name: values[index]
                    for name, values in sorted(
                        branch_probabilities.items()
                    )
                }
            predictions.append(item)
        metrics["predictions"] = predictions
    return metrics


def _selection_value(
    metrics: Mapping[str, Any], selection_metric: str
) -> float:
    value = metrics.get(selection_metric)
    if value is None:
        if selection_metric == "composite_auc":
            raise ValueError(
                "SV composite AUC requires eligible within-site class pairs"
            )
        return -float(metrics["loss"])
    return float(value)


def _checkpoint_payload(
    model: SVSignedGINClassifier,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    history: List[Dict[str, Any]],
    config: SVSignedGINTrainingConfig,
    class_weights: torch.Tensor,
    provenance: Mapping[str, Any],
    best_epoch: int,
    best_value: float,
) -> Dict[str, Any]:
    return {
        "schema_version": SV_SIGNED_GIN_CHECKPOINT_SCHEMA_VERSION,
        "model_name": SV_SIGNED_GIN_MODEL_NAME,
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
        "training_stage": model.training_stage,
        "validation_thresholds": None,
        "threshold_fit_split": "validation",
    }


def load_sv_signed_gin_checkpoint(
    path: Path,
    model: SVSignedGINClassifier,
    device: torch.device,
    expected_provenance: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    payload = _trusted_load(path, device)
    if payload.get("schema_version") != (
        SV_SIGNED_GIN_CHECKPOINT_SCHEMA_VERSION
    ) or payload.get("model_name") != SV_SIGNED_GIN_MODEL_NAME:
        raise ValueError("not an SV Signed-GIN checkpoint")
    checkpoint_config = payload.get("model_config")
    if not isinstance(checkpoint_config, dict):
        raise ValueError("SV Signed-GIN checkpoint has no model config")
    normalized_config = model.config.__class__(
        **checkpoint_config
    )
    if asdict(normalized_config) != model.config_dict():
        raise ValueError("SV Signed-GIN model configuration mismatch")
    if expected_provenance is not None and payload.get(
        "provenance"
    ) != dict(expected_provenance):
        raise ValueError("SV Signed-GIN checkpoint provenance mismatch")
    model.load_state_dict(payload["model_state_dict"])
    return payload


def _trainable_optimizer(
    model: SVSignedGINClassifier,
    config: SVSignedGINTrainingConfig,
):
    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("SV training stage has no trainable parameters")
    optimizer = torch.optim.AdamW(
        parameters,
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
    return optimizer, scheduler


def _thresholds_and_evaluation(
    model: SVSignedGINClassifier,
    validation_loader: Iterable,
    device: torch.device,
    class_weights: torch.Tensor,
    config: SVSignedGINTrainingConfig,
) -> Dict[str, Any]:
    validation = run_sv_signed_gin_epoch(
        model,
        validation_loader,
        device,
        class_weights,
        max_batches=config.max_validation_batches,
        include_predictions=True,
    )
    labels = [
        int(item["label"]) for item in validation["predictions"]
    ]
    probabilities = [
        float(item["positive_probability"])
        for item in validation["predictions"]
    ]
    thresholds = {
        "balanced_accuracy": fit_binary_threshold(
            labels, probabilities, "balanced_accuracy"
        ),
        "accuracy": fit_binary_threshold(
            labels, probabilities, "accuracy"
        ),
    }
    return {
        "thresholds": thresholds,
        "validation": validation,
        "labels": labels,
        "probabilities": probabilities,
    }


def _train_static_anchor_residual_classifier(
    model: SVSignedGINClassifier,
    train_loader: Iterable,
    validation_loader: Iterable,
    train_labels: Iterable[int],
    device: torch.device,
    config: SVSignedGINTrainingConfig,
    output_dir: Path,
    provenance: Mapping[str, Any],
) -> Dict[str, Any]:
    """Train V1A without allowing residual experts to damage the anchor."""

    set_reproducible_seed(config.seed)
    model.reset_residual_fusion_parameters(config.seed)
    model.to(device)
    class_weights = class_weights_from_labels(train_labels)
    history: List[Dict[str, Any]] = []

    model.set_training_stage("static_anchor")
    optimizer, scheduler = _trainable_optimizer(model, config)
    anchor_best_epoch = 0
    anchor_best_value = float("-inf")
    without_improvement = 0
    for epoch in range(1, config.static_anchor_epochs + 1):
        train = run_sv_signed_gin_epoch(
            model,
            train_loader,
            device,
            class_weights,
            optimizer=optimizer,
            gradient_clip_norm=config.gradient_clip_norm,
            gradient_accumulation_steps=(
                config.gradient_accumulation_steps
            ),
            max_batches=config.max_train_batches,
        )
        validation = run_sv_signed_gin_epoch(
            model,
            validation_loader,
            device,
            class_weights,
            max_batches=config.max_validation_batches,
        )
        value = _selection_value(
            validation, config.selection_metric
        )
        scheduler.step(value)
        improved = anchor_best_epoch == 0 or value > anchor_best_value
        if improved:
            anchor_best_epoch = epoch
            anchor_best_value = value
            without_improvement = 0
        else:
            without_improvement += 1
        record = {
            "phase": "static_anchor",
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
            anchor_best_epoch,
            anchor_best_value,
        )
        payload["best_phase"] = "static_anchor"
        _atomic_torch_save(
            output_dir / "static_anchor_last_checkpoint.pt", payload
        )
        if improved:
            _atomic_torch_save(
                output_dir / "static_anchor_checkpoint.pt", payload
            )
        _atomic_json(output_dir / "history.json", history)
        print(
            "phase=static_anchor epoch {}/{} train_auc={} "
            "validation_auc={} validation_site_auc={} "
            "selection={:.6f} lr={:.8f}".format(
                epoch,
                config.static_anchor_epochs,
                train["roc_auc"],
                validation["roc_auc"],
                validation["site_stratified_roc_auc"],
                value,
                optimizer.param_groups[0]["lr"],
            ),
            flush=True,
        )
        if (
            config.early_stopping_patience > 0
            and without_improvement >= config.early_stopping_patience
        ):
            break

    load_sv_signed_gin_checkpoint(
        output_dir / "static_anchor_checkpoint.pt",
        model,
        device,
        expected_provenance=provenance,
    )
    model.set_training_stage("residual_experts")
    anchor_reference = _thresholds_and_evaluation(
        model,
        validation_loader,
        device,
        class_weights,
        config,
    )
    anchor_validation = anchor_reference["validation"]
    _atomic_json(
        output_dir / "static_anchor_evaluation.json",
        {
            "best_epoch": anchor_best_epoch,
            "selection_metric": config.selection_metric,
            "best_selection_value": anchor_best_value,
            "validation_thresholds": anchor_reference["thresholds"],
            "metrics": anchor_validation,
        },
    )

    optimizer, scheduler = _trainable_optimizer(model, config)
    best_epoch = 0
    best_value = _selection_value(
        anchor_validation, config.selection_metric
    )
    without_improvement = 0
    anchor_payload = _checkpoint_payload(
        model,
        optimizer,
        scheduler,
        0,
        history,
        config,
        class_weights,
        provenance,
        best_epoch,
        best_value,
    )
    anchor_payload["best_phase"] = "static_anchor"
    anchor_payload["static_anchor_best_epoch"] = anchor_best_epoch
    _atomic_torch_save(
        output_dir / "best_checkpoint.pt", anchor_payload
    )

    for epoch in range(1, config.epochs + 1):
        train = run_sv_signed_gin_epoch(
            model,
            train_loader,
            device,
            class_weights,
            optimizer=optimizer,
            gradient_clip_norm=config.gradient_clip_norm,
            gradient_accumulation_steps=(
                config.gradient_accumulation_steps
            ),
            max_batches=config.max_train_batches,
            auxiliary_loss_weight=config.auxiliary_loss_weight,
            residual_gate_penalty_weight=(
                config.residual_gate_penalty_weight
            ),
        )
        validation = run_sv_signed_gin_epoch(
            model,
            validation_loader,
            device,
            class_weights,
            max_batches=config.max_validation_batches,
            auxiliary_loss_weight=config.auxiliary_loss_weight,
            residual_gate_penalty_weight=(
                config.residual_gate_penalty_weight
            ),
        )
        value = _selection_value(
            validation, config.selection_metric
        )
        scheduler.step(value)
        improved = value > best_value
        if improved:
            best_epoch = epoch
            best_value = value
            without_improvement = 0
        else:
            without_improvement += 1
        record = {
            "phase": "residual_experts",
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
        payload["best_phase"] = (
            "residual_experts" if best_epoch > 0 else "static_anchor"
        )
        payload["static_anchor_best_epoch"] = anchor_best_epoch
        _atomic_torch_save(
            output_dir / "last_checkpoint.pt", payload
        )
        if improved:
            _atomic_torch_save(
                output_dir / "best_checkpoint.pt", payload
            )
        _atomic_json(output_dir / "history.json", history)
        gates = validation.get("residual_gates", {})
        print(
            "phase=residual_experts epoch {}/{} train_auc={} "
            "validation_auc={} validation_site_auc={} "
            "selection={:.6f} "
            "gates=gin:{:.6f},variation:{:.6f},attention:{} "
            "lr={:.8f}".format(
                epoch,
                config.epochs,
                train["roc_auc"],
                validation["roc_auc"],
                validation["site_stratified_roc_auc"],
                value,
                float(gates.get("gin", 0.0)),
                float(gates.get("variation", 0.0)),
                (
                    "{:.6f}".format(float(gates["attention"]))
                    if "attention" in gates
                    else "N/A"
                ),
                optimizer.param_groups[0]["lr"],
            ),
            flush=True,
        )
        if (
            config.early_stopping_patience > 0
            and without_improvement >= config.early_stopping_patience
        ):
            break

    payload = load_sv_signed_gin_checkpoint(
        output_dir / "best_checkpoint.pt",
        model,
        device,
        expected_provenance=provenance,
    )
    model.set_training_stage("residual_experts")
    final = _thresholds_and_evaluation(
        model,
        validation_loader,
        device,
        class_weights,
        config,
    )
    thresholds = final["thresholds"]
    validation = final["validation"]
    labels = final["labels"]
    probabilities = final["probabilities"]
    payload["validation_thresholds"] = thresholds
    payload["history"] = list(history)
    payload["residual_epochs_completed"] = sum(
        row["phase"] == "residual_experts" for row in history
    )
    _atomic_torch_save(output_dir / "best_checkpoint.pt", payload)
    evaluation = {
        "best_epoch": int(payload["best_epoch"]),
        "best_phase": payload["best_phase"],
        "static_anchor_best_epoch": anchor_best_epoch,
        "selection_metric": config.selection_metric,
        "best_selection_value": float(
            payload["best_selection_value"]
        ),
        "validation_thresholds": thresholds,
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
        "branch_metrics": validation.get("branch_metrics"),
        "residual_gates": validation.get("residual_gates"),
        "fusion_regret": {
            "roc_auc": (
                float(
                    validation["branch_metrics"][
                        "static_spectral"
                    ]["roc_auc"]
                )
                - float(validation["roc_auc"])
                if validation.get("branch_metrics")
                and validation["roc_auc"] is not None
                and validation["branch_metrics"][
                    "static_spectral"
                ]["roc_auc"]
                is not None
                else None
            ),
            "site_stratified_roc_auc": (
                float(
                    validation["branch_metrics"][
                        "static_spectral"
                    ]["site_stratified_roc_auc"]
                )
                - float(validation["site_stratified_roc_auc"])
                if validation.get("branch_metrics")
                and validation["site_stratified_roc_auc"] is not None
                and validation["branch_metrics"][
                    "static_spectral"
                ]["site_stratified_roc_auc"]
                is not None
                else None
            ),
        },
        "static_anchor_metrics": {
            key: anchor_validation.get(key)
            for key in (
                "roc_auc",
                "site_stratified_roc_auc",
                "composite_auc",
            )
        },
    }
    _atomic_json(output_dir / "best_evaluation.json", evaluation)
    return {
        "variant": model.config.variant,
        "epochs_completed": len(history),
        "static_anchor_epochs_completed": sum(
            row["phase"] == "static_anchor" for row in history
        ),
        "residual_epochs_completed": sum(
            row["phase"] == "residual_experts" for row in history
        ),
        "best_phase": payload["best_phase"],
        "best_epoch": int(payload["best_epoch"]),
        "best_selection_value": float(
            payload["best_selection_value"]
        ),
        "static_anchor_checkpoint": str(
            output_dir / "static_anchor_checkpoint.pt"
        ),
        "best_checkpoint": str(output_dir / "best_checkpoint.pt"),
        "last_checkpoint": str(output_dir / "last_checkpoint.pt"),
        "history": str(output_dir / "history.json"),
        "best_evaluation": str(
            output_dir / "best_evaluation.json"
        ),
    }


def train_sv_signed_gin_classifier(
    model: SVSignedGINClassifier,
    train_loader: Iterable,
    validation_loader: Iterable,
    train_labels: Iterable[int],
    device: torch.device,
    config: SVSignedGINTrainingConfig,
    output_dir: Path,
    provenance: Mapping[str, Any],
) -> Dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.json"
    if history_path.exists():
        raise FileExistsError("SV Signed-GIN output already exists")
    if model.config.uses_residual_fusion:
        return _train_static_anchor_residual_classifier(
            model,
            train_loader,
            validation_loader,
            train_labels,
            device,
            config,
            output_dir,
            provenance,
        )
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
    best_value = float("-inf")
    without_improvement = 0
    for epoch in range(1, config.epochs + 1):
        train = run_sv_signed_gin_epoch(
            model,
            train_loader,
            device,
            class_weights,
            optimizer=optimizer,
            gradient_clip_norm=config.gradient_clip_norm,
            gradient_accumulation_steps=(
                config.gradient_accumulation_steps
            ),
            max_batches=config.max_train_batches,
            auxiliary_loss_weight=config.auxiliary_loss_weight,
        )
        validation = run_sv_signed_gin_epoch(
            model,
            validation_loader,
            device,
            class_weights,
            max_batches=config.max_validation_batches,
            auxiliary_loss_weight=config.auxiliary_loss_weight,
        )
        selection_value = _selection_value(
            validation, config.selection_metric
        )
        scheduler.step(selection_value)
        improved = best_epoch == 0 or selection_value > best_value
        if improved:
            best_epoch = epoch
            best_value = selection_value
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
        _atomic_torch_save(
            output_dir / "last_checkpoint.pt", payload
        )
        if improved:
            _atomic_torch_save(
                output_dir / "best_checkpoint.pt", payload
            )
        _atomic_json(history_path, history)
        branch_text = ""
        if validation.get("branch_metrics"):
            branch_text = " branch_auc={}".format(
                ",".join(
                    "{}:{:.4f}".format(
                        name,
                        float(values["roc_auc"]),
                    )
                    for name, values in sorted(
                        validation["branch_metrics"].items()
                    )
                    if values["roc_auc"] is not None
                )
            )
        fusion_text = ""
        if validation.get("fusion_weights"):
            fusion_text = " fusion={}".format(
                ",".join(
                    "{}:{:.3f}".format(name, value)
                    for name, value in sorted(
                        validation["fusion_weights"].items()
                    )
                )
            )
        print(
            "epoch {}/{} variant={} train_loss={:.6f} train_auc={} "
            "validation_loss={:.6f} validation_auc={} "
            "validation_site_auc={} selection={:.6f} lr={:.8f}{}{}".format(
                epoch,
                config.epochs,
                model.config.variant,
                train["loss"],
                train["roc_auc"],
                validation["loss"],
                validation["roc_auc"],
                validation["site_stratified_roc_auc"],
                selection_value,
                optimizer.param_groups[0]["lr"],
                branch_text,
                fusion_text,
            ),
            flush=True,
        )
        if (
            config.early_stopping_patience > 0
            and without_improvement >= config.early_stopping_patience
        ):
            break

    payload = load_sv_signed_gin_checkpoint(
        output_dir / "best_checkpoint.pt",
        model,
        device,
        expected_provenance=provenance,
    )
    validation = run_sv_signed_gin_epoch(
        model,
        validation_loader,
        device,
        class_weights,
        max_batches=config.max_validation_batches,
        include_predictions=True,
        auxiliary_loss_weight=config.auxiliary_loss_weight,
    )
    labels = [
        int(item["label"]) for item in validation["predictions"]
    ]
    probabilities = [
        float(item["positive_probability"])
        for item in validation["predictions"]
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
        "best_selection_value": float(
            payload["best_selection_value"]
        ),
        "validation_thresholds": thresholds,
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
        "branch_metrics": validation.get("branch_metrics"),
        "fusion_weights": validation.get("fusion_weights"),
    }
    _atomic_json(output_dir / "best_evaluation.json", evaluation)
    return {
        "variant": model.config.variant,
        "epochs_completed": len(history),
        "best_epoch": int(payload["best_epoch"]),
        "best_selection_value": float(
            payload["best_selection_value"]
        ),
        "best_checkpoint": str(
            output_dir / "best_checkpoint.pt"
        ),
        "last_checkpoint": str(
            output_dir / "last_checkpoint.pt"
        ),
        "history": str(history_path),
        "best_evaluation": str(
            output_dir / "best_evaluation.json"
        ),
    }
