"""Training, evaluation and versioned checkpoints for Hard-STSE-Temporal-SGW."""

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

from keysubgraph.models.hard_stse_loss import (
    HardSTSECriterion,
    HardSTSELossConfig,
)
from keysubgraph.models.hard_stse_temporal_sgw import (
    HardSTSETemporalSGWClassifier,
)
from keysubgraph.models.hard_stse_types import (
    HardSelectionSchedule,
    HardSTSEConfig,
)
from keysubgraph.training.trainer import (
    class_weights_from_labels,
    set_reproducible_seed,
)


HARD_STSE_CHECKPOINT_SCHEMA_VERSION = 1
HARD_STSE_MODEL_NAME = "hard_stse_temporal_sgw"


@dataclass(frozen=True)
class HardSTSETrainingConfig:
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
            raise ValueError("scheduler factor must lie in (0, 1)")
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


def hard_stse_config_from_dict(payload: Dict[str, Any]) -> HardSTSEConfig:
    values = dict(payload)
    schedule = values.get("selection_schedule")
    if isinstance(schedule, dict):
        values["selection_schedule"] = HardSelectionSchedule(**schedule)
    return HardSTSEConfig(**values)


def fit_hard_stse_standardizers(
    model: HardSTSETemporalSGWClassifier,
    data_loader: Iterable,
    device: torch.device,
    max_batches: Optional[int] = None,
    selection_seed: int = 42,
) -> Dict[str, int]:
    """Fit all transforms from the training partition only."""
    was_training = model.training
    model.eval()
    statistic_values = [[] for _ in range(model.config.graph_statistic_dim)]
    theory_fixed = []
    sample_count = 0
    with torch.no_grad():
        for batch_index, cpu_batch in enumerate(data_loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = cpu_batch.to(device)
            output = model(
                batch,
                epoch=1,
                random_selection_seed=selection_seed,
                compute_theory_proxies=False,
            )
            sample_count += len(batch)
            for encoding in output.diagnostics["window_encodings"]:
                values = encoding.raw_statistics.detach().cpu()
                mask = encoding.graph_statistic_mask.detach().cpu()
                for index in range(model.config.graph_statistic_dim):
                    if bool(mask[index]):
                        statistic_values[index].append(float(values[index]))
            fixed_raw = output.diagnostics["theory_diagnostics"].get(
                "fixed_raw"
            )
            if fixed_raw is not None:
                theory_fixed.append(fixed_raw.detach().cpu())
    if sample_count < 1:
        raise ValueError("standardizer fitting processed no training samples")
    means, scales = [], []
    for values in statistic_values:
        if not values:
            means.append(0.0)
            scales.append(1.0)
            continue
        tensor = torch.tensor(values, dtype=torch.float32)
        mean = tensor.mean()
        scale = torch.sqrt(
            (tensor - mean).square().mean() + model.config.epsilon
        )
        means.append(float(mean))
        scales.append(float(scale))
    model.window_encoder.set_graph_statistic_transform(
        torch.tensor(means, device=device),
        torch.tensor(scales, device=device),
    )
    if model.sgw_branch is not None:
        if not theory_fixed:
            raise ValueError("M3 training produced no theory features to fit")
        model.sgw_branch.fit_fixed_standardizer(
            torch.cat(theory_fixed, dim=0).to(device)
        )
    model.train(was_training)
    return {
        "training_samples": sample_count,
        "graph_statistic_observations": sum(
            len(values) for values in statistic_values
        ),
        "theory_samples": (
            sum(item.shape[0] for item in theory_fixed)
            if theory_fixed
            else 0
        ),
    }


def _classification_metrics(
    labels: List[int], probabilities: List[float], predictions: List[int]
) -> Dict[str, Any]:
    unique = set(labels)
    accuracy = float(accuracy_score(labels, predictions))
    mean = sum(probabilities) / float(len(probabilities))
    variance = sum((value - mean) ** 2 for value in probabilities) / float(
        len(probabilities)
    )
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
            "mean": mean,
            "standard_deviation": math.sqrt(variance),
        },
    }


def run_hard_stse_epoch(
    model: HardSTSETemporalSGWClassifier,
    data_loader: Iterable,
    device: torch.device,
    criterion: HardSTSECriterion,
    class_weights: torch.Tensor,
    epoch: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    gradient_clip_norm: float = 1.0,
    max_batches: Optional[int] = None,
    selection_seed: int = 42,
) -> Dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    count_total = 0
    accumulators = {
        key: 0.0
        for key in (
            "loss",
            "classification_loss",
            "fusion_ce",
            "neural_ce",
            "theory_ce",
            "node_budget",
            "edge_budget",
            "laplacian",
            "gw_proxy",
            "node_probability_mean",
            "edge_probability_mean",
            "actual_node_ratio",
            "actual_edge_candidate_ratio",
            "actual_edge_original_ratio",
        )
    }
    labels_all, probabilities_all, predictions_all = [], [], []
    gradient_norms = []
    started = time.perf_counter()
    compute_proxies = (
        model.config.selection_mode == "learned"
        and epoch > model.config.selection_schedule.anneal_end_epoch
        and (
            criterion.loss_config.laplacian_weight_max > 0.0
            or criterion.loss_config.gw_proxy_weight_max > 0.0
        )
    )
    for batch_index, cpu_batch in enumerate(data_loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = cpu_batch.to(device)
        labels = batch.labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(
                batch,
                epoch=epoch,
                random_selection_seed=selection_seed,
                compute_theory_proxies=compute_proxies,
            )
            losses = criterion(output, labels, epoch, class_weights.to(device))
            if training:
                losses.total.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), gradient_clip_norm
                )
                gradient_norms.append(float(norm.detach().cpu()))
                optimizer.step()
        count = int(labels.numel())
        count_total += count
        values = {
            "loss": losses.total,
            "classification_loss": losses.classification,
            "fusion_ce": losses.fusion_ce,
            "neural_ce": losses.neural_ce,
            "theory_ce": losses.theory_ce,
            "node_budget": losses.node_budget,
            "edge_budget": losses.edge_budget,
            "laplacian": losses.laplacian,
            "gw_proxy": losses.gw_proxy,
            "node_probability_mean": output.diagnostics[
                "node_probability_mean"
            ],
            "edge_probability_mean": output.diagnostics[
                "edge_probability_mean"
            ],
            "actual_node_ratio": output.fusion_logits.new_tensor(
                output.diagnostics["actual_node_ratio"]
            ),
            "actual_edge_candidate_ratio": output.fusion_logits.new_tensor(
                output.diagnostics["actual_edge_candidate_ratio"]
            ),
            "actual_edge_original_ratio": output.fusion_logits.new_tensor(
                output.diagnostics["actual_edge_original_ratio"]
            ),
        }
        for key, value in values.items():
            accumulators[key] += float(value.detach().cpu()) * count
        probabilities = torch.softmax(output.fusion_logits, dim=-1)[:, 1]
        predictions = output.fusion_logits.argmax(dim=-1)
        labels_all.extend(int(value) for value in labels.detach().cpu().tolist())
        probabilities_all.extend(
            float(value) for value in probabilities.detach().cpu().tolist()
        )
        predictions_all.extend(
            int(value) for value in predictions.detach().cpu().tolist()
        )
    if count_total < 1:
        raise ValueError("Hard-STSE epoch processed no samples")
    metrics = {
        key: value / float(count_total)
        for key, value in accumulators.items()
    }
    metrics.update(
        {
            "mean_gradient_norm": (
                sum(gradient_norms) / len(gradient_norms)
                if gradient_norms
                else None
            ),
            "elapsed_seconds": time.perf_counter() - started,
            "curriculum_weights": criterion._curriculum_weights(epoch),
        }
    )
    metrics.update(
        _classification_metrics(
            labels_all, probabilities_all, predictions_all
        )
    )
    return metrics


def _selection_key(metrics: Dict[str, Any]) -> Tuple[float, float]:
    return (
        float(metrics["balanced_accuracy"]),
        (
            float(metrics["roc_auc"])
            if metrics.get("roc_auc") is not None
            else float("-inf")
        ),
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
    protocol_path,
    protocol_sha256,
    best_epoch,
    best_key,
):
    return {
        "model_name": HARD_STSE_MODEL_NAME,
        "schema_version": HARD_STSE_CHECKPOINT_SCHEMA_VERSION,
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "model_config": asdict(model.config),
        "training_config": asdict(training_config),
        "loss_config": asdict(loss_config),
        "class_weights": class_weights.detach().cpu(),
        "protocol_path": str(Path(protocol_path).resolve()),
        "protocol_sha256": str(protocol_sha256),
        "history": list(history),
        "best_epoch": int(best_epoch),
        "best_selection_key": [float(best_key[0]), float(best_key[1])],
        "torch_version": str(torch.__version__),
    }


def load_hard_stse_checkpoint(
    path: Path,
    model: HardSTSETemporalSGWClassifier,
    device: torch.device,
    expected_protocol_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    payload = _trusted_torch_load(path, device)
    if payload.get("model_name") != HARD_STSE_MODEL_NAME:
        raise ValueError("not a Hard-STSE-Temporal-SGW checkpoint")
    if payload.get("schema_version") != HARD_STSE_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported Hard-STSE checkpoint schema")
    if payload.get("model_config") != asdict(model.config):
        raise ValueError("Hard-STSE model configuration mismatch")
    if (
        expected_protocol_sha256 is not None
        and payload.get("protocol_sha256") != expected_protocol_sha256
    ):
        raise ValueError("Hard-STSE checkpoint protocol hash mismatch")
    model.load_state_dict(payload["model_state_dict"])
    return payload


def train_hard_stse(
    model: HardSTSETemporalSGWClassifier,
    train_loader: Iterable,
    validation_loader: Iterable,
    train_labels: Iterable[int],
    device: torch.device,
    training_config: HardSTSETrainingConfig,
    loss_config: HardSTSELossConfig,
    output_dir: Path,
    protocol_path: Path,
    protocol_sha256: str,
    resume_checkpoint: Optional[Path] = None,
) -> Dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.json"
    if history_path.exists() and resume_checkpoint is None:
        raise FileExistsError("Hard-STSE training output already exists")
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
    criterion = HardSTSECriterion(model.config, loss_config)
    history = []
    best_epoch = 0
    best_key = (float("-inf"), float("-inf"))
    start_epoch = 1
    if resume_checkpoint is not None:
        payload = load_hard_stse_checkpoint(
            resume_checkpoint, model, device, protocol_sha256
        )
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        history = list(payload["history"])
        best_epoch = int(payload["best_epoch"])
        best_key = tuple(float(value) for value in payload["best_selection_key"])
        start_epoch = int(payload["epoch"]) + 1
    else:
        standardizer_summary = fit_hard_stse_standardizers(
            model,
            train_loader,
            device,
            max_batches=training_config.max_train_batches,
            selection_seed=training_config.seed,
        )
        _atomic_json(
            output_dir / "standardizer_fit_summary.json",
            standardizer_summary,
        )
    epochs_without_improvement = (
        0 if not history else int(history[-1].get("epochs_without_improvement", 0))
    )
    require_completed_curriculum = (
        model.config.selection_mode == "learned"
        and training_config.epochs
        >= model.config.selection_schedule.anneal_end_epoch
    )
    for epoch in range(start_epoch, training_config.epochs + 1):
        train_metrics = run_hard_stse_epoch(
            model,
            train_loader,
            device,
            criterion,
            class_weights,
            epoch,
            optimizer=optimizer,
            gradient_clip_norm=training_config.gradient_clip_norm,
            max_batches=training_config.max_train_batches,
            selection_seed=training_config.seed,
        )
        validation_metrics = run_hard_stse_epoch(
            model,
            validation_loader,
            device,
            criterion,
            class_weights,
            epoch,
            optimizer=None,
            gradient_clip_norm=training_config.gradient_clip_norm,
            max_batches=training_config.max_validation_batches,
            selection_seed=training_config.seed,
        )
        key = _selection_key(validation_metrics)
        scheduler.step(key[0])
        checkpoint_eligible = (
            not require_completed_curriculum
            or epoch >= model.config.selection_schedule.anneal_end_epoch
        )
        improved = checkpoint_eligible and (
            best_epoch == 0 or key > best_key
        )
        curriculum_in_progress = (
            model.config.selection_mode == "learned"
            and epoch <= model.config.selection_schedule.anneal_end_epoch
        )
        if improved:
            best_epoch, best_key = epoch, key
            epochs_without_improvement = 0
        elif curriculum_in_progress:
            # M2/M3 are intentionally different models before the retention
            # schedule reaches its target.  Early stopping during this period
            # would prevent the joint-stability stage from running at all.
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
            loss_config,
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
            "validation_auc={} node_p={:.4f} edge_p={:.4f} lr={:.8f}".format(
                epoch,
                training_config.epochs,
                train_metrics["loss"],
                train_metrics["balanced_accuracy"],
                train_metrics["roc_auc"],
                validation_metrics["loss"],
                validation_metrics["balanced_accuracy"],
                validation_metrics["roc_auc"],
                train_metrics["node_probability_mean"],
                train_metrics["edge_probability_mean"],
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
    load_hard_stse_checkpoint(
        output_dir / "best_checkpoint.pt",
        model,
        device,
        expected_protocol_sha256=protocol_sha256,
    )
    best_train = run_hard_stse_epoch(
        model,
        train_loader,
        device,
        criterion,
        class_weights,
        best_epoch,
        optimizer=None,
        max_batches=training_config.max_train_batches,
        selection_seed=training_config.seed,
    )
    best_validation = run_hard_stse_epoch(
        model,
        validation_loader,
        device,
        criterion,
        class_weights,
        best_epoch,
        optimizer=None,
        max_batches=training_config.max_validation_batches,
        selection_seed=training_config.seed,
    )
    best_evaluation_path = output_dir / "best_evaluation.json"
    _atomic_json(
        best_evaluation_path,
        {
            "best_epoch": best_epoch,
            "selection": {
                "primary": "validation_balanced_accuracy",
                "tie_breaker": "validation_roc_auc",
                "value": [best_key[0], best_key[1]],
            },
            "train": best_train,
            "validation": best_validation,
        },
    )
    return {
        "best_checkpoint": output_dir / "best_checkpoint.pt",
        "last_checkpoint": output_dir / "last_checkpoint.pt",
        "history": history_path,
        "best_evaluation": best_evaluation_path,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
    }
