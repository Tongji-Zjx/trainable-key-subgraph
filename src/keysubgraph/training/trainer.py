"""Training and validation for the verified soft_graph baseline."""

from __future__ import absolute_import, division, print_function

import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from torch.nn.utils import clip_grad_norm_

from keysubgraph.data.data_split import file_sha256
from keysubgraph.models.losses import compute_soft_graph_loss
from keysubgraph.models.soft_extractor import SoftGraphClassifier


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    target_node_ratio: float = 0.30
    target_edge_ratio: float = 0.30
    budget_weight: float = 1.0
    gradient_clip_norm: float = 5.0
    seed: int = 42
    selection_metric: str = "roc_auc"
    max_train_batches: Optional[int] = None
    max_validation_batches: Optional[int] = None

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("optimizer parameters are invalid")
        if not 0.0 <= self.target_node_ratio <= 1.0:
            raise ValueError("target_node_ratio must be in [0, 1]")
        if not 0.0 <= self.target_edge_ratio <= 1.0:
            raise ValueError("target_edge_ratio must be in [0, 1]")
        if self.budget_weight < 0.0 or self.gradient_clip_norm <= 0.0:
            raise ValueError("loss weight and gradient clip must be positive")
        if self.selection_metric not in ("roc_auc", "balanced_accuracy", "loss"):
            raise ValueError("unsupported selection_metric")
        for value in (self.max_train_batches, self.max_validation_batches):
            if value is not None and value < 1:
                raise ValueError("batch limits must be positive when provided")


def set_reproducible_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def class_weights_from_labels(labels: Iterable[int]) -> torch.Tensor:
    counts = np.bincount(np.asarray(list(labels), dtype=np.int64), minlength=2)
    if bool((counts == 0).any()):
        raise ValueError("training data must contain both classes")
    weights = counts.sum() / (2.0 * counts.astype(np.float64))
    return torch.tensor(weights, dtype=torch.float32)


def _metrics(labels: List[int], probabilities: List[float], predictions: List[int]) -> Dict[str, Any]:
    unique = set(labels)
    roc_auc = float(roc_auc_score(labels, probabilities)) if unique == {0, 1} else None
    accuracy = float(accuracy_score(labels, predictions))
    return {
        "sample_count": len(labels),
        "accuracy": accuracy,
        "balanced_accuracy": (
            float(balanced_accuracy_score(labels, predictions))
            if unique == {0, 1}
            else accuracy
        ),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": roc_auc,
        "class_counts": {str(label): labels.count(label) for label in (0, 1)},
    }


def _run_epoch(
    model: SoftGraphClassifier,
    loader: Iterable,
    device: torch.device,
    config: TrainingConfig,
    class_weights: torch.Tensor,
    optimizer: Optional[torch.optim.Optimizer],
    max_batches: Optional[int],
    include_predictions: bool = False,
) -> Dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    classification_loss = 0.0
    budget_loss = 0.0
    sample_count = 0
    labels_all: List[int] = []
    probabilities_all: List[float] = []
    predictions_all: List[int] = []
    gradient_norms = []
    prediction_rows = []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = batch.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            loss = compute_soft_graph_loss(
                output,
                batch.labels,
                target_node_ratio=config.target_node_ratio,
                target_edge_ratio=config.target_edge_ratio,
                budget_weight=config.budget_weight,
                class_weights=class_weights,
            )
            if not bool(torch.isfinite(loss.total)):
                raise RuntimeError("non-finite training loss")
            if training:
                loss.total.backward()
                gradient_norm = clip_grad_norm_(
                    model.parameters(), config.gradient_clip_norm
                )
                if not bool(torch.isfinite(gradient_norm)):
                    raise RuntimeError("non-finite gradient norm")
                optimizer.step()
                gradient_norms.append(float(gradient_norm.detach().cpu()))

            current_size = len(batch)
            total_loss += float(loss.total.detach().cpu()) * current_size
            classification_loss += float(loss.classification.detach().cpu()) * current_size
            budget_loss += float(loss.budget.detach().cpu()) * current_size
            sample_count += current_size
            probabilities = torch.softmax(output.logits.detach(), dim=-1)[:, 1].cpu()
            predictions = output.logits.detach().argmax(dim=-1).cpu()
            labels = batch.labels.detach().cpu()
            probabilities_all.extend(float(item) for item in probabilities.tolist())
            predictions_all.extend(int(item) for item in predictions.tolist())
            labels_all.extend(int(item) for item in labels.tolist())
            if include_predictions:
                for sample, label, prediction, probability in zip(
                    batch,
                    labels.tolist(),
                    predictions.tolist(),
                    probabilities.tolist(),
                ):
                    prediction_rows.append(
                        {
                            "sample_key": sample.sample_key,
                            "sample_id": sample.sample_id,
                            "site": sample.site,
                            "subject_id": sample.subject_id,
                            "label": int(label),
                            "prediction": int(prediction),
                            "class_1_probability": float(probability),
                        }
                    )

    if sample_count == 0:
        raise RuntimeError("epoch processed no samples")
    result = _metrics(labels_all, probabilities_all, predictions_all)
    result.update(
        {
            "loss": total_loss / sample_count,
            "classification_loss": classification_loss / sample_count,
            "budget_loss": budget_loss / sample_count,
            "mean_gradient_norm": (
                sum(gradient_norms) / len(gradient_norms) if gradient_norms else None
            ),
        }
    )
    if include_predictions:
        result["predictions"] = prediction_rows
    return result


def evaluate_model(
    model: SoftGraphClassifier,
    loader: Iterable,
    device: torch.device,
    config: TrainingConfig,
    class_weights: torch.Tensor,
    max_batches: Optional[int] = None,
    include_predictions: bool = False,
) -> Dict[str, Any]:
    return _run_epoch(
        model,
        loader,
        device,
        config,
        class_weights,
        optimizer=None,
        max_batches=max_batches,
        include_predictions=include_predictions,
    )


def _atomic_torch_save(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, str(temporary))
    os.replace(str(temporary), str(path))


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def _checkpoint_payload(
    model: SoftGraphClassifier,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    training_config: TrainingConfig,
    train_metrics: Dict[str, Any],
    selection_metrics: Dict[str, Any],
    selection_partition: str,
    protocol_path: Path,
    protocol: Dict[str, Any],
    class_weights: torch.Tensor,
    train_loader: Iterable,
) -> Dict[str, Any]:
    loader_generator = getattr(train_loader, "generator", None)
    rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "train_loader_generator": (
            loader_generator.get_state() if loader_generator is not None else None
        ),
    }
    payload = {
        "schema_version": 1,
        "epoch": epoch,
        "training_mode": model.training_mode,
        "model_config": asdict(model.config),
        "training_config": asdict(training_config),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_metrics": train_metrics,
        "selection_partition": selection_partition,
        "selection_metrics": selection_metrics,
        "data_protocol_sha256": file_sha256(protocol_path),
        "data_artifact_sha256": protocol["sha256"],
        "edge_presence_threshold": protocol["edge_presence_threshold"],
        "class_weights": class_weights.tolist(),
        "rng_state": rng_state,
    }
    if selection_partition == "validation":
        payload["validation_metrics"] = selection_metrics
    elif selection_partition == "cohort":
        payload["cohort_metrics"] = selection_metrics
    return payload


def _restore_rng_state(checkpoint: Dict[str, Any], train_loader: Iterable) -> None:
    """Restore stochastic state when resuming; older checkpoints remain loadable."""

    state = checkpoint.get("rng_state")
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])
    loader_generator = getattr(train_loader, "generator", None)
    loader_state = state.get("train_loader_generator")
    if loader_generator is not None and loader_state is not None:
        loader_generator.set_state(loader_state.cpu())


def _selection_value(metrics: Dict[str, Any], metric: str) -> float:
    if metric == "loss":
        return -float(metrics["loss"])
    value = metrics.get(metric)
    return -math.inf if value is None else float(value)


def train_model(
    model: SoftGraphClassifier,
    train_loader: Iterable,
    validation_loader: Iterable,
    train_labels: Iterable[int],
    device: torch.device,
    config: TrainingConfig,
    output_dir: Path,
    protocol_path: Path,
    protocol: Dict[str, Any],
    resume_checkpoint: Optional[Path] = None,
    selection_partition: str = "validation",
) -> Dict[str, Any]:
    """Train and select a checkpoint on an explicitly named evaluation partition.

    ``validation`` retains the strict predictive workflow. ``cohort`` is reserved
    for the explicitly exploratory all-sample workflow and is not an estimate of
    out-of-sample performance.
    """

    if selection_partition not in ("validation", "cohort"):
        raise ValueError("selection_partition must be validation or cohort")

    set_reproducible_seed(config.seed)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best_checkpoint.pt"
    last_path = output_dir / "last_checkpoint.pt"
    history_path = output_dir / "history.json"
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    class_weights = class_weights_from_labels(train_labels).to(device)
    start_epoch = 1
    history: List[Dict[str, Any]] = []
    best_value = -math.inf

    if resume_checkpoint is not None:
        checkpoint = load_checkpoint(resume_checkpoint, model, optimizer, device)
        if checkpoint["data_protocol_sha256"] != file_sha256(protocol_path):
            raise ValueError("resume checkpoint uses a different data protocol")
        checkpoint_partition = checkpoint.get("selection_partition", "validation")
        if checkpoint_partition != selection_partition:
            raise ValueError("resume checkpoint uses a different selection partition")
        start_epoch = int(checkpoint["epoch"]) + 1
        _restore_rng_state(checkpoint, train_loader)
        if history_path.exists():
            with history_path.open("r", encoding="utf-8") as handle:
                history = json.load(handle)
        for row in history:
            best_value = max(
                best_value,
                _selection_value(row[selection_partition], config.selection_metric),
            )

    for epoch in range(start_epoch, config.epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            device,
            config,
            class_weights,
            optimizer,
            config.max_train_batches,
            False,
        )
        selection_metrics = evaluate_model(
            model,
            validation_loader,
            device,
            config,
            class_weights,
            config.max_validation_batches,
            False,
        )
        row = {"epoch": epoch, "train": train_metrics, selection_partition: selection_metrics}
        history.append(row)
        payload = _checkpoint_payload(
            model,
            optimizer,
            epoch,
            config,
            train_metrics,
            selection_metrics,
            selection_partition,
            protocol_path,
            protocol,
            class_weights.detach().cpu(),
            train_loader,
        )
        _atomic_torch_save(last_path, payload)
        selection_value = _selection_value(selection_metrics, config.selection_metric)
        if not best_path.exists() or selection_value > best_value:
            best_value = selection_value
            _atomic_torch_save(best_path, payload)
        _atomic_json(history_path, history)
        print(
            "epoch {}/{} train_loss={:.6f} {}_loss={:.6f} {}={}".format(
                epoch,
                config.epochs,
                float(train_metrics["loss"]),
                selection_partition,
                float(selection_metrics["loss"]),
                config.selection_metric,
                selection_metrics.get(config.selection_metric),
            ),
            flush=True,
        )

    return {
        "best_checkpoint": best_path,
        "last_checkpoint": last_path,
        "history": history_path,
        "epochs_completed": len(history),
        "best_validation_value": best_value if selection_partition == "validation" else None,
        "best_cohort_value": (
            (-best_value if config.selection_metric == "loss" else best_value)
            if selection_partition == "cohort"
            else None
        ),
        "selection_partition": selection_partition,
        "selection_metric": config.selection_metric,
    }


def load_checkpoint(
    path: Path,
    model: SoftGraphClassifier,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    device = device or torch.device("cpu")
    checkpoint = torch.load(
        str(Path(path).resolve()), map_location=device, weights_only=False
    )
    if checkpoint.get("schema_version") != 1:
        raise ValueError("unsupported checkpoint schema")
    if checkpoint.get("training_mode") != "soft_graph":
        raise ValueError("checkpoint is not a soft_graph model")
    if checkpoint.get("model_config") != asdict(model.config):
        raise ValueError("checkpoint model configuration does not match")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
