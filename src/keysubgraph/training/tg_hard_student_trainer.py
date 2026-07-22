"""Two-stage optimization, validation thresholding and early stopping for Stage C."""

from __future__ import absolute_import, division, print_function

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import torch

from keysubgraph.models.tg_hard_student_loss import (
    TGHardStudentLossConfig,
    compute_tg_hard_student_loss,
)
from keysubgraph.training.baseline_trainer import (
    baseline_class_weights,
    baseline_metrics,
    select_balanced_accuracy_threshold,
)
from keysubgraph.training.trainer import set_reproducible_seed

from .tg_hard_student_checkpoint import (
    load_tg_hard_student_checkpoint,
    save_tg_hard_student_checkpoint,
)


@dataclass(frozen=True)
class TGHardStudentTrainingConfig:
    epochs: int = 150
    frozen_graph_epochs: int = 15
    head_learning_rate: float = 5.0e-4
    graph_encoder_learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    gradient_clip_norm: float = 5.0
    early_stopping_patience: int = 20
    selection_metric: str = "balanced_accuracy"
    seed: int = 42
    max_train_batches: Optional[int] = None
    max_validation_batches: Optional[int] = None

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.frozen_graph_epochs < 0:
            raise ValueError("invalid TG hard-student epoch configuration")
        if self.frozen_graph_epochs >= self.epochs:
            raise ValueError("frozen graph stage must end before training ends")
        if self.head_learning_rate <= 0.0 or self.graph_encoder_learning_rate <= 0.0:
            raise ValueError("TG hard-student learning rates must be positive")
        if self.weight_decay < 0.0 or self.gradient_clip_norm <= 0.0:
            raise ValueError("invalid TG hard-student optimizer configuration")
        if self.early_stopping_patience < 1:
            raise ValueError("early-stopping patience must be positive")
        if self.selection_metric not in (
            "balanced_accuracy", "roc_auc", "unweighted_log_loss"
        ):
            raise ValueError("unsupported TG hard-student selection metric")
        for value in (self.max_train_batches, self.max_validation_batches):
            if value is not None and value < 1:
                raise ValueError("Stage-C batch limits must be positive")


def initialize_student_graph_encoder(student, teacher) -> None:
    student_state = student.graph_encoder.state_dict()
    teacher_state = teacher.graph_encoder.state_dict()
    if set(student_state) != set(teacher_state):
        raise ValueError("teacher/student Signed GNN parameter names differ")
    for name in student_state:
        if student_state[name].shape != teacher_state[name].shape:
            raise ValueError("teacher/student Signed GNN shape mismatch: {}".format(name))
    student.graph_encoder.load_state_dict(teacher_state)


def set_student_graph_encoder_trainable(model, trainable: bool) -> None:
    for parameter in model.graph_encoder.parameters():
        parameter.requires_grad = bool(trainable)


def build_tg_hard_student_optimizer(model, config: TGHardStudentTrainingConfig):
    graph_parameters = list(model.graph_encoder.parameters())
    graph_ids = {id(parameter) for parameter in graph_parameters}
    head_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in graph_ids
    ]
    return torch.optim.AdamW(
        [
            {"params": graph_parameters, "lr": config.graph_encoder_learning_rate,
             "group_name": "graph_encoder"},
            {"params": head_parameters, "lr": config.head_learning_rate,
             "group_name": "student_head"},
        ],
        weight_decay=config.weight_decay,
    )


def _batch_tensors(batch, device):
    hard = tuple(item.hard_cache for item in batch)
    theory = torch.stack(
        [item.standardized_theory_features for item in batch]
    ).to(device)
    labels = torch.tensor([item.label for item in batch], dtype=torch.long, device=device)
    teacher_logits = torch.stack(
        [item.teacher_target.logits for item in batch]
    ).to(device)
    teacher_representation = torch.stack(
        [item.teacher_target.representation for item in batch]
    ).to(device)
    return hard, theory, labels, teacher_logits, teacher_representation


def run_tg_hard_student_epoch(
    model,
    data_loader: Iterable,
    device: torch.device,
    loss_config: TGHardStudentLossConfig,
    optimizer: Optional[torch.optim.Optimizer] = None,
    class_weights: Optional[torch.Tensor] = None,
    gradient_clip_norm: float = 5.0,
    max_batches: Optional[int] = None,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    names = (
        "loss", "classification_loss", "knowledge_distillation_loss",
        "representation_distillation_loss", "supervised_contrastive_loss",
    )
    totals = {name: 0.0 for name in names}
    labels_all = []
    probabilities_all = []
    gradient_norms = []
    sample_count = 0
    for batch_index, batch in enumerate(data_loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        hard, theory, labels, teacher_logits, teacher_representation = _batch_tensors(
            batch, device
        )
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(hard, theory)
            loss = compute_tg_hard_student_loss(
                output, labels, teacher_logits, teacher_representation,
                loss_config, class_weights=class_weights,
            )
            if training:
                loss.total.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), gradient_clip_norm
                )
                gradient_norms.append(float(norm.detach().cpu()))
                optimizer.step()
        count = int(labels.numel())
        sample_count += count
        probabilities = torch.softmax(output.logits, dim=-1)[:, 1]
        labels_all.extend(int(value) for value in labels.detach().cpu().tolist())
        probabilities_all.extend(
            float(value) for value in probabilities.detach().cpu().tolist()
        )
        values = (
            loss.total, loss.classification, loss.knowledge_distillation,
            loss.representation_distillation, loss.supervised_contrastive,
        )
        for name, value in zip(names, values):
            totals[name] += float(value.detach().cpu()) * count
    if sample_count == 0:
        raise ValueError("TG hard-student epoch processed no samples")
    metrics = baseline_metrics(labels_all, probabilities_all, threshold)
    metrics.update({name: value / sample_count for name, value in totals.items()})
    metrics["mean_gradient_norm"] = (
        sum(gradient_norms) / len(gradient_norms) if gradient_norms else None
    )
    metrics["labels"] = labels_all
    metrics["probabilities"] = probabilities_all
    return metrics


def _selection_value(metrics: Dict[str, Any], name: str) -> float:
    value = metrics.get(name)
    if value is None:
        return float("-inf")
    return -float(value) if name == "unweighted_log_loss" else float(value)


def _atomic_json(path: Path, payload) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def _public_metrics(metrics):
    return {key: value for key, value in metrics.items() if key not in ("labels", "probabilities")}


def train_tg_hard_student(
    model,
    train_loader,
    validation_loader,
    train_labels: Sequence[int],
    device: torch.device,
    loss_config: TGHardStudentLossConfig,
    training_config: TGHardStudentTrainingConfig,
    output_dir: Path,
    theory_standardizer,
    protocol_sha256: str,
    teacher_checkpoint_sha256: str,
    candidate_scaler_sha256: str,
    theory_scaler_sha256: str,
    teacher_model=None,
    resume_checkpoint: Optional[Path] = None,
) -> Dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    history_path = output_dir / "history.json"
    if history_path.exists() and resume_checkpoint is None:
        raise FileExistsError("TG hard-student output exists; resume or choose a new directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    set_reproducible_seed(training_config.seed)
    model.to(device)
    initialized = False
    if resume_checkpoint is None:
        if teacher_model is None:
            raise ValueError("fresh Stage-C training requires a loaded teacher model")
        initialize_student_graph_encoder(model, teacher_model)
        initialized = True
    optimizer = build_tg_hard_student_optimizer(model, training_config)
    class_weights = baseline_class_weights(train_labels).to(device)
    history = []
    best_epoch = 0
    best_value = float("-inf")
    best_threshold = 0.5
    start_epoch = 1
    if resume_checkpoint is not None:
        checkpoint = load_tg_hard_student_checkpoint(
            resume_checkpoint, model, device, optimizer=optimizer,
            expected_protocol_sha256=protocol_sha256,
            expected_teacher_checkpoint_sha256=teacher_checkpoint_sha256,
            data_loader=train_loader, restore_rng=True,
        )
        if checkpoint.get("loss_config") != asdict(loss_config):
            raise ValueError("resume loss configuration differs from checkpoint")
        if checkpoint.get("training_config") != asdict(training_config):
            raise ValueError("resume training configuration differs from checkpoint")
        history = list(checkpoint["history"])
        best_epoch = int(checkpoint["best_epoch"])
        best_value = float(checkpoint["best_selection_value"])
        best_threshold = float(checkpoint["classification_threshold"])
        initialized = bool(checkpoint.get("teacher_encoder_initialized"))
        start_epoch = int(checkpoint["epoch"]) + 1
    epochs_without_improvement = max(0, start_epoch - 1 - best_epoch)
    for epoch in range(start_epoch, training_config.epochs + 1):
        graph_trainable = epoch > training_config.frozen_graph_epochs
        set_student_graph_encoder_trainable(model, graph_trainable)
        train_metrics = run_tg_hard_student_epoch(
            model, train_loader, device, loss_config, optimizer=optimizer,
            class_weights=class_weights,
            gradient_clip_norm=training_config.gradient_clip_norm,
            max_batches=training_config.max_train_batches,
        )
        validation_raw = run_tg_hard_student_epoch(
            model, validation_loader, device, loss_config, optimizer=None,
            gradient_clip_norm=training_config.gradient_clip_norm,
            max_batches=training_config.max_validation_batches,
        )
        threshold = select_balanced_accuracy_threshold(
            validation_raw["labels"], validation_raw["probabilities"]
        )
        validation_metrics = baseline_metrics(
            validation_raw["labels"], validation_raw["probabilities"], threshold
        )
        for key in (
            "loss", "classification_loss", "knowledge_distillation_loss",
            "representation_distillation_loss", "supervised_contrastive_loss",
            "mean_gradient_norm",
        ):
            validation_metrics[key] = validation_raw[key]
        record = {
            "epoch": epoch,
            "stage": "fine_tune" if graph_trainable else "frozen_graph_encoder",
            "train": _public_metrics(train_metrics),
            "validation": validation_metrics,
        }
        history.append(record)
        value = _selection_value(validation_metrics, training_config.selection_metric)
        improved = value > best_value
        if improved:
            best_epoch, best_value, best_threshold = epoch, value, threshold
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        kwargs = dict(
            model=model, theory_standardizer=theory_standardizer, epoch=epoch,
            protocol_sha256=protocol_sha256,
            teacher_checkpoint_sha256=teacher_checkpoint_sha256,
            candidate_scaler_sha256=candidate_scaler_sha256,
            theory_scaler_sha256=theory_scaler_sha256,
            optimizer=optimizer, history=history, loss_config=loss_config,
            training_config=training_config, best_epoch=best_epoch,
            best_selection_value=best_value,
            classification_threshold=(threshold if improved else best_threshold),
            data_loader=train_loader, teacher_encoder_initialized=initialized,
        )
        save_tg_hard_student_checkpoint(output_dir / "last_checkpoint.pt", **kwargs)
        if improved:
            save_tg_hard_student_checkpoint(output_dir / "best_checkpoint.pt", **kwargs)
        _atomic_json(history_path, history)
        print(
            "epoch {}/{} stage={} train_loss={:.6f} validation_loss={:.6f} "
            "validation_balanced_accuracy={:.6f} validation_auc={} threshold={:.6f}".format(
                epoch, training_config.epochs, record["stage"], train_metrics["loss"],
                validation_metrics["loss"], validation_metrics["balanced_accuracy"],
                validation_metrics["roc_auc"], threshold,
            ), flush=True,
        )
        if epochs_without_improvement >= training_config.early_stopping_patience:
            break
    return {
        "best_checkpoint": output_dir / "best_checkpoint.pt",
        "last_checkpoint": output_dir / "last_checkpoint.pt",
        "history": history_path,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "classification_threshold": best_threshold,
        "selection_metric": training_config.selection_metric,
    }
