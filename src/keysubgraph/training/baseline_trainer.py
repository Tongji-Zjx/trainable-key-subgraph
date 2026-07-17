"""Training, threshold selection, metrics, and checkpoints for the baseline."""

from __future__ import absolute_import, division, print_function

import json
import hashlib
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.nn.utils import clip_grad_norm_

from keysubgraph.data.baseline_manifest import read_baseline_manifest
from keysubgraph.data.data_split import file_sha256
from keysubgraph.features.structural_prior import STATIC_WINDOW_STRUCTURAL_FEATURES
from keysubgraph.models.baseline_classifier import SignedSequenceBaseline


@dataclass(frozen=True)
class BaselineTrainingConfig:
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 5.0
    seed: int = 42
    early_stopping_patience: int = 15
    selection_metric: str = "unweighted_log_loss"
    max_train_batches: Optional[int] = None
    max_validation_batches: Optional[int] = None

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("invalid baseline optimizer configuration")
        if self.gradient_clip_norm <= 0.0 or self.early_stopping_patience < 1:
            raise ValueError("gradient clip and patience must be positive")
        if self.selection_metric not in ("unweighted_log_loss", "roc_auc"):
            raise ValueError("unsupported baseline selection metric")
        for value in (self.max_train_batches, self.max_validation_batches):
            if value is not None and value < 1:
                raise ValueError("batch limits must be positive")


def set_baseline_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def baseline_class_weights(labels: Iterable[int]) -> torch.Tensor:
    values = np.asarray(list(labels), dtype=np.int64)
    counts = np.bincount(values, minlength=2)
    if bool((counts == 0).any()):
        raise ValueError("baseline training data must contain both classes")
    weights = counts.sum() / (2.0 * counts.astype(np.float64))
    return torch.tensor(weights, dtype=torch.float32)


def _unweighted_log_loss(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    labels_array = np.asarray(labels, dtype=np.float64)
    probabilities_array = np.clip(
        np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0 - 1e-12
    )
    values = -(
        labels_array * np.log(probabilities_array)
        + (1.0 - labels_array) * np.log(1.0 - probabilities_array)
    )
    return float(values.mean())


def select_balanced_accuracy_threshold(
    labels: Sequence[int], probabilities: Sequence[float]
) -> float:
    if len(labels) != len(probabilities) or not labels:
        raise ValueError("threshold inputs must be non-empty and aligned")
    probability_array = np.asarray(probabilities, dtype=np.float64)
    candidates = set(float(value) for value in probability_array.tolist())
    candidates.update((0.0, 0.5, 1.0))
    maximum = float(probability_array.max())
    if maximum < 1.0:
        candidates.add(float(np.nextafter(maximum, 1.0)))
    unique = set(int(value) for value in labels)
    best = None
    for threshold in sorted(candidates):
        predictions = (probability_array >= threshold).astype(np.int64)
        score = (
            float(balanced_accuracy_score(labels, predictions))
            if unique == {0, 1}
            else float(accuracy_score(labels, predictions))
        )
        key = (score, -abs(threshold - 0.5), -threshold)
        if best is None or key > best[0]:
            best = (key, threshold)
    return float(best[1])


def baseline_metrics(
    labels: Sequence[int], probabilities: Sequence[float], threshold: float
) -> Dict[str, Any]:
    if len(labels) != len(probabilities) or not labels:
        raise ValueError("metric inputs must be non-empty and aligned")
    predictions = [int(value >= threshold) for value in probabilities]
    unique = set(int(value) for value in labels)
    accuracy = float(accuracy_score(labels, predictions))
    return {
        "sample_count": len(labels),
        "class_counts": {str(label): list(labels).count(label) for label in (0, 1)},
        "threshold": float(threshold),
        "unweighted_log_loss": _unweighted_log_loss(labels, probabilities),
        "roc_auc": (
            float(roc_auc_score(labels, probabilities)) if unique == {0, 1} else None
        ),
        "accuracy": accuracy,
        "balanced_accuracy": (
            float(balanced_accuracy_score(labels, predictions))
            if unique == {0, 1}
            else accuracy
        ),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=[0, 1]
        ).tolist(),
    }


def _run_baseline_epoch(
    model: SignedSequenceBaseline,
    loader: Iterable,
    device: torch.device,
    class_weights: torch.Tensor,
    gradient_clip_norm: float,
    optimizer: Optional[torch.optim.Optimizer] = None,
    max_batches: Optional[int] = None,
    threshold: float = 0.5,
    include_predictions: bool = False,
) -> Dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    weighted_loss_sum = 0.0
    sample_count = 0
    gradient_norms = []
    labels_all: List[int] = []
    probabilities_all: List[float] = []
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
            weighted_loss = F.cross_entropy(
                output.logits, batch.labels, weight=class_weights
            )
            if not bool(torch.isfinite(weighted_loss)):
                raise RuntimeError("non-finite baseline loss")
            if training:
                weighted_loss.backward()
                gradient_norm = clip_grad_norm_(model.parameters(), gradient_clip_norm)
                if not bool(torch.isfinite(gradient_norm)):
                    raise RuntimeError("non-finite baseline gradient norm")
                optimizer.step()
                gradient_norms.append(float(gradient_norm.detach().cpu()))
            current_size = batch.batch_size
            weighted_loss_sum += float(weighted_loss.detach().cpu()) * current_size
            sample_count += current_size
            probabilities = torch.softmax(output.logits.detach(), dim=-1)[:, 1].cpu().tolist()
            labels = batch.labels.detach().cpu().tolist()
            labels_all.extend(int(value) for value in labels)
            probabilities_all.extend(float(value) for value in probabilities)
            if include_predictions:
                for index, (label, probability) in enumerate(zip(labels, probabilities)):
                    prediction_rows.append(
                        {
                            "sample_key": batch.sample_keys[index],
                            "sample_id": batch.sample_ids[index],
                            "subject_id": batch.subject_ids[index],
                            "site": batch.sites[index],
                            "label": int(label),
                            "class_1_probability": float(probability),
                        }
                    )
    if sample_count == 0:
        raise RuntimeError("baseline epoch processed no samples")
    result = baseline_metrics(labels_all, probabilities_all, threshold)
    result.update(
        {
            "weighted_loss": weighted_loss_sum / sample_count,
            "mean_gradient_norm": (
                sum(gradient_norms) / len(gradient_norms) if gradient_norms else None
            ),
        }
    )
    result["labels"] = labels_all
    result["probabilities"] = probabilities_all
    if include_predictions:
        for row in prediction_rows:
            row["prediction"] = int(row["class_1_probability"] >= threshold)
        result["predictions"] = prediction_rows
    return result


def evaluate_baseline(
    model: SignedSequenceBaseline,
    loader: Iterable,
    device: torch.device,
    class_weights: torch.Tensor,
    threshold: float,
    max_batches: Optional[int] = None,
    include_predictions: bool = False,
) -> Dict[str, Any]:
    return _run_baseline_epoch(
        model,
        loader,
        device,
        class_weights,
        gradient_clip_norm=1.0,
        optimizer=None,
        max_batches=max_batches,
        threshold=threshold,
        include_predictions=include_predictions,
    )


def _clean_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if key not in ("labels", "probabilities", "predictions")
    }


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def _atomic_torch(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, str(temporary))
    os.replace(str(temporary), str(path))


def _trusted_torch_load(path: Path, device: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=device)


def _selection_value(metrics: Dict[str, Any], name: str) -> float:
    if name == "unweighted_log_loss":
        return -float(metrics[name])
    value = metrics.get(name)
    return -math.inf if value is None else float(value)


def _checkpoint_payload(
    model: SignedSequenceBaseline,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    training_config: BaselineTrainingConfig,
    train_metrics: Dict[str, Any],
    validation_metrics: Dict[str, Any],
    threshold: float,
    class_weights: torch.Tensor,
    train_manifest_path: Path,
    validation_manifest_path: Path,
    train_manifest_payload: Dict[str, Any],
    structural_transform: Optional[Dict[str, Any]],
    structural_transform_sha256: str,
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "training_mode": "signed_sequence_baseline",
        "epoch": int(epoch),
        "model_config": asdict(model.config),
        "training_config": asdict(training_config),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_metrics": _clean_metrics(train_metrics),
        "validation_metrics": _clean_metrics(validation_metrics),
        "classification_threshold": float(threshold),
        "class_weights": class_weights.detach().cpu().tolist(),
        "train_manifest_sha256": file_sha256(train_manifest_path),
        "validation_manifest_sha256": file_sha256(validation_manifest_path),
        "data_protocol_sha256": train_manifest_payload["data_protocol_sha256"],
        "extractor_checkpoint_sha256": train_manifest_payload["checkpoint_sha256"],
        "evidence_level": train_manifest_payload["evidence_level"],
        "parent_manifest_sha256": train_manifest_payload.get(
            "parent_manifest_sha256"
        ),
        "downstream_splits_json_sha256": train_manifest_payload.get(
            "downstream_splits_json_sha256"
        ),
        "subgraph_source": train_manifest_payload.get("subgraph_source", "key"),
        "matched_control_manifest_sha256": train_manifest_payload.get(
            "matched_control_manifest_sha256", ""
        ),
        "structural_transform": structural_transform,
        "structural_transform_sha256": structural_transform_sha256,
    }


def train_baseline(
    model: SignedSequenceBaseline,
    train_loader: Iterable,
    validation_loader: Iterable,
    train_labels: Iterable[int],
    device: torch.device,
    config: BaselineTrainingConfig,
    output_dir: Path,
    train_manifest_path: Path,
    validation_manifest_path: Path,
    project_root: Path,
    structural_transform: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    set_baseline_seed(config.seed)
    project_root = Path(project_root).resolve()
    train_manifest_payload, train_records = read_baseline_manifest(
        train_manifest_path, project_root, verify_exports=False
    )
    validation_manifest_payload, validation_records = read_baseline_manifest(
        validation_manifest_path, project_root, verify_exports=False
    )
    if train_manifest_payload["split"] != "train":
        raise ValueError("training manifest must use split='train'")
    if validation_manifest_payload["split"] != "validation":
        raise ValueError("validation manifest must use split='validation'")
    for name in ("data_protocol_sha256", "checkpoint_sha256", "evidence_level"):
        if train_manifest_payload[name] != validation_manifest_payload[name]:
            raise ValueError("train and validation manifests differ in {}".format(name))
    for name in ("subgraph_source", "matched_control_manifest_sha256"):
        if train_manifest_payload.get(name, "key" if name == "subgraph_source" else "") != validation_manifest_payload.get(
            name, "key" if name == "subgraph_source" else ""
        ):
            raise ValueError("train and validation manifests differ in {}".format(name))
    train_parent = train_manifest_payload.get("parent_manifest_sha256")
    validation_parent = validation_manifest_payload.get("parent_manifest_sha256")
    if train_parent != validation_parent:
        raise ValueError("train and validation manifests have different parents")
    train_split_artifact = train_manifest_payload.get(
        "downstream_splits_json_sha256"
    )
    validation_split_artifact = validation_manifest_payload.get(
        "downstream_splits_json_sha256"
    )
    if train_split_artifact != validation_split_artifact:
        raise ValueError("train and validation manifests use different downstream splits")
    train_keys = {record.sample_key for record in train_records}
    validation_keys = {record.sample_key for record in validation_records}
    if train_keys & validation_keys:
        raise ValueError("train and validation baseline samples overlap")

    def record_group(record):
        if record.subject_id:
            return "{}::{}".format(record.site, record.subject_id)
        return "sample::{}".format(record.sample_key)

    train_groups = {record_group(record) for record in train_records}
    validation_groups = {record_group(record) for record in validation_records}
    if train_groups & validation_groups:
        raise ValueError("subject groups overlap between train and validation")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.json"
    best_path = output_dir / "best_checkpoint.pt"
    last_path = output_dir / "last_checkpoint.pt"
    structural_transform_path = output_dir / "structural_transform.json"
    if any(
        path.exists()
        for path in (history_path, best_path, last_path, structural_transform_path)
    ):
        raise FileExistsError("baseline training outputs already exist")
    structural_transform_hash = ""
    if model.config.structural_interface_version == 1:
        if not isinstance(structural_transform, dict):
            raise ValueError("structural interface v1 requires a fitted transform")
        if structural_transform.get("structural_group") != model.config.structural_group:
            raise ValueError("structural transform group differs from model")
        if structural_transform.get("prior_mode") != model.config.prior_mode:
            raise ValueError("structural transform prior differs from model")
        if bool(structural_transform.get("use_structural_features")) != bool(
            model.config.use_structural_features
        ):
            raise ValueError("structural transform feature flag differs from model")
        if structural_transform.get("fitted_on") != "train_only":
            raise ValueError("structural transform was not fit on train only")
        if structural_transform.get("feature_names") != list(
            STATIC_WINDOW_STRUCTURAL_FEATURES
        ):
            raise ValueError("structural transform feature schema differs")
        if len(structural_transform["feature_names"]) != model.config.structural_feature_dim:
            raise ValueError("structural transform feature dimension differs")
        if float(structural_transform.get("beta")) != model.config.prior_beta:
            raise ValueError("structural transform beta differs from model")
        if int(structural_transform.get("permutation_seed")) != model.config.prior_permutation_seed:
            raise ValueError("structural transform permutation seed differs from model")
        expected_key_hash = hashlib.sha256(
            json.dumps(
                sorted(record.sample_key for record in train_records),
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if structural_transform.get("train_sample_key_sha256") != expected_key_hash:
            raise ValueError("structural transform was fit on a different train cohort")
        if int(structural_transform.get("sample_count", -1)) != len(train_records):
            raise ValueError("structural transform train sample count differs")
        _atomic_json(structural_transform_path, structural_transform)
        structural_transform_hash = file_sha256(structural_transform_path)
        model.configure_structural_transform(
            torch.tensor(structural_transform["mean"], dtype=torch.float32),
            torch.tensor(structural_transform["std"], dtype=torch.float32),
            torch.tensor(structural_transform["prior_scale"], dtype=torch.float32),
        )
    elif structural_transform is not None:
        raise ValueError("legacy baseline cannot accept a structural transform")
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    class_weights = baseline_class_weights(train_labels).to(device)
    history = []
    best_value = -math.inf
    epochs_without_improvement = 0
    for epoch in range(1, config.epochs + 1):
        train_metrics = _run_baseline_epoch(
            model,
            train_loader,
            device,
            class_weights,
            config.gradient_clip_norm,
            optimizer=optimizer,
            max_batches=config.max_train_batches,
            threshold=0.5,
        )
        validation_raw = evaluate_baseline(
            model,
            validation_loader,
            device,
            class_weights,
            threshold=0.5,
            max_batches=config.max_validation_batches,
        )
        threshold = select_balanced_accuracy_threshold(
            validation_raw["labels"], validation_raw["probabilities"]
        )
        validation_metrics = baseline_metrics(
            validation_raw["labels"], validation_raw["probabilities"], threshold
        )
        validation_metrics["weighted_loss"] = validation_raw["weighted_loss"]
        validation_metrics["mean_gradient_norm"] = None
        row = {
            "epoch": epoch,
            "train": _clean_metrics(train_metrics),
            "validation": _clean_metrics(validation_metrics),
        }
        history.append(row)
        checkpoint = _checkpoint_payload(
            model,
            optimizer,
            epoch,
            config,
            train_metrics,
            validation_metrics,
            threshold,
            class_weights,
            train_manifest_path,
            validation_manifest_path,
            train_manifest_payload,
            structural_transform,
            structural_transform_hash,
        )
        _atomic_torch(last_path, checkpoint)
        value = _selection_value(validation_metrics, config.selection_metric)
        if not best_path.exists() or value > best_value:
            best_value = value
            epochs_without_improvement = 0
            _atomic_torch(best_path, checkpoint)
        else:
            epochs_without_improvement += 1
        _atomic_json(history_path, history)
        print(
            "epoch {}/{} train_weighted_loss={:.6f} validation_log_loss={:.6f} "
            "validation_auc={} threshold={:.6f}".format(
                epoch,
                config.epochs,
                float(train_metrics["weighted_loss"]),
                float(validation_metrics["unweighted_log_loss"]),
                validation_metrics["roc_auc"],
                threshold,
            ),
            flush=True,
        )
        if epochs_without_improvement >= config.early_stopping_patience:
            break
    return {
        "best_checkpoint": best_path,
        "last_checkpoint": last_path,
        "history": history_path,
        "epochs_completed": len(history),
        "selection_metric": config.selection_metric,
        "structural_transform": (
            structural_transform_path
            if model.config.structural_interface_version == 1
            else None
        ),
    }


def load_baseline_checkpoint(
    path: Path,
    model: SignedSequenceBaseline,
    device: Optional[torch.device] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, Any]:
    device = device or torch.device("cpu")
    checkpoint = _trusted_torch_load(Path(path).resolve(), device)
    if checkpoint.get("schema_version") != 1 or checkpoint.get(
        "training_mode"
    ) != "signed_sequence_baseline":
        raise ValueError("checkpoint is not a signed sequence baseline")
    checkpoint_model_config = dict(checkpoint.get("model_config", {}))
    checkpoint_model_config.setdefault("history_mode", "full")
    checkpoint_model_config.setdefault("history_keep_ratio", 1.0)
    checkpoint_model_config.setdefault("temporal_order", "ordered")
    checkpoint_model_config.setdefault("permutation_seed", 42)
    checkpoint_model_config.setdefault("structural_interface_version", 0)
    checkpoint_model_config.setdefault("structural_group", "neutral")
    checkpoint_model_config.setdefault("structural_feature_dim", 11)
    checkpoint_model_config.setdefault("structural_hidden_dim", 32)
    checkpoint_model_config.setdefault("prior_mode", "none")
    checkpoint_model_config.setdefault("prior_beta", 1.0)
    checkpoint_model_config.setdefault("prior_permutation_seed", 42)
    if checkpoint_model_config != asdict(model.config):
        raise ValueError("baseline checkpoint model configuration differs")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def read_baseline_checkpoint_payload(
    path: Path, device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Read trusted local checkpoint metadata before model construction."""

    device = device or torch.device("cpu")
    checkpoint = _trusted_torch_load(Path(path).resolve(), device)
    if checkpoint.get("schema_version") != 1 or checkpoint.get(
        "training_mode"
    ) != "signed_sequence_baseline":
        raise ValueError("checkpoint is not a signed sequence baseline")
    return checkpoint
