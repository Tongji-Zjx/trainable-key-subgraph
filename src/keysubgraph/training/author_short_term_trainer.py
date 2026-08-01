"""Source-faithful training for the coordinate-free author short-term branch."""

from __future__ import absolute_import, division, print_function

import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Sampler

from keysubgraph.data.graph_dataset import list_batch_collate
from keysubgraph.features.paper_short_term_pst import (
    PaperShortTermCommunityFrequency,
)
from keysubgraph.models.author_short_term import (
    AUTHOR_SHORT_TERM_MODEL_NAME,
    AUTHOR_SHORT_TERM_PROFILES,
    AuthorNoCoordinateShortTermClassifier,
    AuthorShortTermConfig,
)
from keysubgraph.training.dual_sgw_feature_trainer import binary_metrics
from keysubgraph.training.trainer import set_reproducible_seed


AUTHOR_SHORT_TERM_CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class AuthorShortTermTrainingConfig:
    profile: str
    epochs: int = 1000
    learning_rate: float = 5.0e-5
    weight_decay: float = 1.0e-5
    gradient_clip_norm: float = 5.0
    label_smoothing: float = 0.10
    early_stopping_minimum_epochs: int = 30
    early_stopping_patience: int = 25
    scheduler_minimum_learning_rate: float = 1.0e-6
    initial_positive_probability: float = 0.75
    seed: int = 42
    max_train_batches: Optional[int] = None
    max_validation_batches: Optional[int] = None

    def __post_init__(self) -> None:
        if self.profile not in AUTHOR_SHORT_TERM_PROFILES:
            raise ValueError("unsupported author short-term profile")
        if self.epochs < 1 or self.learning_rate <= 0.0:
            raise ValueError("invalid author short-term training duration")
        if self.weight_decay < 0.0 or self.gradient_clip_norm <= 0.0:
            raise ValueError("invalid author short-term optimizer settings")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label smoothing must lie in [0,1)")
        if (
            self.early_stopping_minimum_epochs < 0
            or self.early_stopping_patience < 0
        ):
            raise ValueError("early-stopping values cannot be negative")
        if self.scheduler_minimum_learning_rate <= 0.0:
            raise ValueError("minimum learning rate must be positive")
        if not 0.0 < self.initial_positive_probability < 1.0:
            raise ValueError("initial class prior must lie in (0,1)")
        for value in (self.max_train_batches, self.max_validation_batches):
            if value is not None and value < 1:
                raise ValueError("batch limits must be positive")


def author_short_term_training_config(
    profile: str,
    epochs: int = 1000,
    seed: Optional[int] = None,
) -> AuthorShortTermTrainingConfig:
    """Return exact dataset-specific optimizer values from author wrappers."""

    if profile == "adhd":
        return AuthorShortTermTrainingConfig(
            profile=profile,
            epochs=epochs,
            learning_rate=6.413547853662974e-5,
            weight_decay=1.031701067121726e-5,
            seed=784341473 if seed is None else int(seed),
        )
    if profile == "wmrc":
        return AuthorShortTermTrainingConfig(
            profile=profile,
            epochs=epochs,
            learning_rate=4.2321370614349516e-5,
            weight_decay=4.880987598860477e-6,
            seed=1196888311 if seed is None else int(seed),
        )
    raise ValueError("unsupported author short-term profile")


class AuthorBalancedBatchSampler(Sampler):
    """The authors' deterministic half-positive/half-negative sampler."""

    def __init__(
        self, labels: Iterable[int], batch_size: int, seed: int
    ) -> None:
        self.labels = [int(value) for value in labels]
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch size must be positive")
        self.positive = [
            index for index, value in enumerate(self.labels) if value == 1
        ]
        self.negative = [
            index for index, value in enumerate(self.labels) if value == 0
        ]
        if not self.positive or not self.negative:
            raise ValueError("balanced sampler requires both classes")
        self.num_batches = int(
            math.ceil(len(self.labels) / float(self.batch_size))
        )
        self.random = random.Random(int(seed))

    def __iter__(self):
        for _ in range(self.num_batches):
            positive_count = self.batch_size // 2
            negative_count = self.batch_size - positive_count
            if len(self.positive) >= positive_count:
                positives = self.random.sample(
                    self.positive, positive_count
                )
            else:
                positives = [
                    self.random.choice(self.positive)
                    for _ in range(positive_count)
                ]
            if len(self.negative) >= negative_count:
                negatives = self.random.sample(
                    self.negative, negative_count
                )
            else:
                negatives = [
                    self.random.choice(self.negative)
                    for _ in range(negative_count)
                ]
            batch = positives + negatives
            self.random.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.num_batches


def create_author_short_term_train_loader(
    dataset,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    sampler = AuthorBalancedBatchSampler(
        [assignment.label for assignment in dataset.assignments],
        batch_size,
        seed,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=list_batch_collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def create_author_short_term_evaluation_loader(
    dataset,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=list_batch_collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )


def fit_author_threshold(
    labels: List[int], probabilities: List[float], metric: str = "balanced"
) -> float:
    """Reproduce the author's fixed 0.01..0.99 threshold grid."""

    if metric not in ("balanced", "accuracy"):
        raise ValueError("unsupported author threshold metric")
    best_threshold = 0.5
    best_value = -1.0
    for value in np.linspace(0.01, 0.99, 99):
        result = binary_metrics(labels, probabilities, float(value))
        score = float(
            result[
                "balanced_accuracy" if metric == "balanced" else "accuracy"
            ]
        )
        if score > best_value or (
            abs(score - best_value) < 1.0e-8
            and float(value) < best_threshold
        ):
            best_value = score
            best_threshold = float(value)
    return best_threshold


def _site_stratified_auc(
    labels: List[int], probabilities: List[float], sites: List[str]
) -> Optional[float]:
    numerator = 0.0
    denominator = 0.0
    for site in sorted(set(sites)):
        indices = [index for index, value in enumerate(sites) if value == site]
        site_labels = [labels[index] for index in indices]
        if set(site_labels) != {0, 1}:
            continue
        site_probabilities = [probabilities[index] for index in indices]
        positive = sum(value == 1 for value in site_labels)
        negative = sum(value == 0 for value in site_labels)
        pairs = float(positive * negative)
        numerator += float(
            roc_auc_score(site_labels, site_probabilities)
        ) * pairs
        denominator += pairs
    return numerator / denominator if denominator > 0.0 else None


def run_author_short_term_epoch(
    model: AuthorNoCoordinateShortTermClassifier,
    data_loader: Iterable,
    device: torch.device,
    positive_class_weight: float,
    label_smoothing: float,
    optimizer: Optional[torch.optim.Optimizer] = None,
    gradient_clip_norm: float = 5.0,
    max_batches: Optional[int] = None,
    threshold: float = 0.5,
    include_predictions: bool = False,
) -> Dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    labels_all: List[int] = []
    probabilities_all: List[float] = []
    sample_keys: List[str] = []
    sites: List[str] = []
    loss_sum = 0.0
    sample_count = 0
    gradient_norms: List[float] = []
    memory_entropies: List[float] = []
    started = time.perf_counter()
    positive_weight = torch.tensor(
        [float(positive_class_weight)],
        device=device,
        dtype=torch.float32,
    )

    for batch_index, cpu_batch in enumerate(data_loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = cpu_batch.to(device)
        labels = batch.labels.to(device=device, dtype=torch.float32)
        targets = labels * (1.0 - label_smoothing) + 0.5 * label_smoothing
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(batch)
            per_sample = torch.nn.functional.binary_cross_entropy_with_logits(
                output.logits,
                targets,
                pos_weight=positive_weight,
                reduction="none",
            )
            loss = per_sample.mean()
            if training:
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), gradient_clip_norm
                )
                gradient_norms.append(float(norm.detach().cpu()))
                optimizer.step()
        count = int(labels.numel())
        sample_count += count
        loss_sum += float(per_sample.detach().sum().cpu())
        probabilities = torch.sigmoid(output.logits)
        labels_all.extend(int(value) for value in labels.detach().cpu().tolist())
        probabilities_all.extend(
            float(value) for value in probabilities.detach().cpu().tolist()
        )
        sample_keys.extend(batch.sample_keys)
        sites.extend(str(sample.site) for sample in batch.samples)
        entropy = -(
            output.memory_attention
            * output.memory_attention.clamp_min(1.0e-12).log()
        ).sum(dim=1)
        denominator = math.log(float(output.memory_attention.shape[1]))
        memory_entropies.extend(
            float(value)
            for value in (entropy / max(denominator, 1.0)).detach().cpu().tolist()
        )
    if sample_count <= 0:
        raise ValueError("author short-term epoch processed no samples")
    metrics = binary_metrics(labels_all, probabilities_all, threshold)
    metrics.update(
        {
            "loss": loss_sum / float(sample_count),
            "site_stratified_roc_auc": _site_stratified_auc(
                labels_all, probabilities_all, sites
            ),
            "mean_gradient_norm": (
                sum(gradient_norms) / float(len(gradient_norms))
                if gradient_norms
                else None
            ),
            "mean_normalized_memory_entropy": sum(memory_entropies)
            / float(len(memory_entropies)),
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
                sample_keys, sites, labels_all, probabilities_all
            )
        ]
    return metrics


def _atomic_json(path: Path, payload: Any) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def _atomic_torch(path: Path, payload: Dict[str, Any]) -> None:
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


def _checkpoint_payload(
    model: AuthorNoCoordinateShortTermClassifier,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    history: List[Dict[str, Any]],
    training_config: AuthorShortTermTrainingConfig,
    positive_class_weight: float,
    protocol_path: Path,
    protocol_sha256: str,
    community_frequency_path: Path,
    community_frequency_sha256: str,
    best_records: Dict[str, Dict[str, Any]],
    validation_thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    return {
        "model_name": AUTHOR_SHORT_TERM_MODEL_NAME,
        "schema_version": AUTHOR_SHORT_TERM_CHECKPOINT_SCHEMA_VERSION,
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "model_config": model.config.to_dict(),
        "community_frequency": model.community_frequency.to_dict(),
        "training_config": asdict(training_config),
        "positive_class_weight": float(positive_class_weight),
        "protocol_path": str(Path(protocol_path).resolve()),
        "protocol_sha256": str(protocol_sha256),
        "community_frequency_path": str(
            Path(community_frequency_path).resolve()
        ),
        "community_frequency_sha256": str(community_frequency_sha256),
        "history": list(history),
        "best_records": best_records,
        "primary_checkpoint_policy": "author_validation_acc_auc_composite",
        "validation_thresholds": dict(validation_thresholds or {}),
        "torch_version": str(torch.__version__),
    }


def model_from_author_short_term_checkpoint(
    path: Path, device: torch.device
) -> Tuple[AuthorNoCoordinateShortTermClassifier, Dict[str, Any]]:
    payload = _trusted_load(path, device)
    if (
        payload.get("model_name") != AUTHOR_SHORT_TERM_MODEL_NAME
        or int(payload.get("schema_version", 0))
        != AUTHOR_SHORT_TERM_CHECKPOINT_SCHEMA_VERSION
    ):
        raise ValueError("not an author short-term checkpoint")
    config = AuthorShortTermConfig.from_dict(payload["model_config"])
    frequency = PaperShortTermCommunityFrequency.from_dict(
        payload["community_frequency"]
    )
    training = AuthorShortTermTrainingConfig(
        **dict(payload["training_config"])
    )
    model = AuthorNoCoordinateShortTermClassifier(
        config,
        frequency,
        initial_positive_probability=training.initial_positive_probability,
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    return model, payload


def train_author_short_term(
    model: AuthorNoCoordinateShortTermClassifier,
    train_loader: Iterable,
    validation_loader: Iterable,
    train_labels: Iterable[int],
    device: torch.device,
    training_config: AuthorShortTermTrainingConfig,
    output_dir: Path,
    protocol_path: Path,
    protocol_sha256: str,
    community_frequency_path: Path,
    community_frequency_sha256: str,
    resume_checkpoint: Optional[Path] = None,
) -> Dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.json"
    if history_path.exists() and resume_checkpoint is None:
        raise FileExistsError("author short-term output already exists")
    labels = [int(value) for value in train_labels]
    positive = sum(value == 1 for value in labels)
    negative = sum(value == 0 for value in labels)
    if positive <= 0 or negative <= 0:
        raise ValueError("training data must contain both classes")
    positive_class_weight = float(negative) / float(positive)
    set_reproducible_seed(training_config.seed)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=training_config.epochs,
        eta_min=training_config.scheduler_minimum_learning_rate,
    )
    history: List[Dict[str, Any]] = []
    best_records = {
        name: {"score": -1.0, "epoch": 0, "threshold": 0.5}
        for name in ("accuracy", "roc_auc", "balanced")
    }
    early_best = -1.0
    epochs_without_improvement = 0
    start_epoch = 1
    if resume_checkpoint is not None:
        payload = _trusted_load(resume_checkpoint, device)
        if payload.get("protocol_sha256") != protocol_sha256:
            raise ValueError("resume protocol hash mismatch")
        if (
            payload.get("community_frequency_sha256")
            != community_frequency_sha256
        ):
            raise ValueError("resume frequency hash mismatch")
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        history = list(payload["history"])
        best_records = dict(payload["best_records"])
        early_best = max(
            (
                float(row["validation"]["author_balanced_composite"])
                for row in history
            ),
            default=-1.0,
        )
        epochs_without_improvement = int(
            history[-1].get("epochs_without_improvement", 0)
        ) if history else 0
        start_epoch = int(payload["epoch"]) + 1

    for epoch in range(start_epoch, training_config.epochs + 1):
        train_metrics = run_author_short_term_epoch(
            model,
            train_loader,
            device,
            positive_class_weight,
            training_config.label_smoothing,
            optimizer=optimizer,
            gradient_clip_norm=training_config.gradient_clip_norm,
            max_batches=training_config.max_train_batches,
        )
        validation_raw = run_author_short_term_epoch(
            model,
            validation_loader,
            device,
            positive_class_weight,
            training_config.label_smoothing,
            max_batches=training_config.max_validation_batches,
            include_predictions=True,
        )
        validation_labels = [
            int(row["label"]) for row in validation_raw["predictions"]
        ]
        validation_probabilities = [
            float(row["positive_probability"])
            for row in validation_raw["predictions"]
        ]
        threshold = fit_author_threshold(
            validation_labels, validation_probabilities, "balanced"
        )
        validation_metrics = binary_metrics(
            validation_labels, validation_probabilities, threshold
        )
        validation_metrics.update(
            {
                "loss": validation_raw["loss"],
                "site_stratified_roc_auc": validation_raw[
                    "site_stratified_roc_auc"
                ],
                "threshold": threshold,
                "mean_normalized_memory_entropy": validation_raw[
                    "mean_normalized_memory_entropy"
                ],
            }
        )
        auc = (
            float(validation_metrics["roc_auc"])
            if validation_metrics.get("roc_auc") is not None
            else -1.0
        )
        composite = (
            float(validation_metrics["accuracy"]) + auc
        ) / 2.0
        validation_metrics["author_balanced_composite"] = composite
        scores = {
            "accuracy": float(validation_metrics["accuracy"]),
            "roc_auc": auc,
            "balanced": composite,
        }
        improved_names = []
        for name, score in scores.items():
            if score > float(best_records[name]["score"]):
                best_records[name] = {
                    "score": score,
                    "epoch": epoch,
                    "threshold": threshold,
                }
                improved_names.append(name)
        if composite > early_best:
            early_best = composite
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        scheduler.step()
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
            positive_class_weight,
            protocol_path,
            protocol_sha256,
            community_frequency_path,
            community_frequency_sha256,
            best_records,
        )
        _atomic_torch(output_dir / "last_checkpoint.pt", checkpoint)
        for name in improved_names:
            target = output_dir / "best_{}_checkpoint.pt".format(name)
            _atomic_torch(target, checkpoint)
            if name == "balanced":
                _atomic_torch(output_dir / "best_checkpoint.pt", checkpoint)
        _atomic_json(history_path, history)
        print(
            "epoch {}/{} train_loss={:.6f} train_auc={} "
            "validation_loss={:.6f} validation_auc={} validation_ba={:.6f} "
            "author_composite={:.6f} threshold={:.2f} lr={:.8f}".format(
                epoch,
                training_config.epochs,
                train_metrics["loss"],
                train_metrics["roc_auc"],
                validation_metrics["loss"],
                validation_metrics["roc_auc"],
                validation_metrics["balanced_accuracy"],
                composite,
                threshold,
                float(optimizer.param_groups[0]["lr"]),
            ),
            flush=True,
        )
        if (
            epoch >= training_config.early_stopping_minimum_epochs
            and training_config.early_stopping_patience > 0
            and epochs_without_improvement
            >= training_config.early_stopping_patience
        ):
            break
    best_path = output_dir / "best_checkpoint.pt"
    if not best_path.is_file():
        raise RuntimeError("author short-term training saved no primary checkpoint")
    model, best_payload = model_from_author_short_term_checkpoint(
        best_path, device
    )
    validation_raw = run_author_short_term_epoch(
        model,
        validation_loader,
        device,
        positive_class_weight,
        training_config.label_smoothing,
        max_batches=training_config.max_validation_batches,
        include_predictions=True,
    )
    validation_labels = [
        int(row["label"]) for row in validation_raw["predictions"]
    ]
    validation_probabilities = [
        float(row["positive_probability"])
        for row in validation_raw["predictions"]
    ]
    thresholds = {
        "balanced_accuracy": fit_author_threshold(
            validation_labels, validation_probabilities, "balanced"
        ),
        "accuracy": fit_author_threshold(
            validation_labels, validation_probabilities, "accuracy"
        ),
    }
    best_payload["validation_thresholds"] = thresholds
    _atomic_torch(best_path, best_payload)
    validation_by_threshold = {
        name: binary_metrics(
            validation_labels, validation_probabilities, threshold
        )
        for name, threshold in thresholds.items()
    }
    best_evaluation = {
        "best_epoch": int(best_payload["epoch"]),
        "primary_checkpoint_policy": (
            "author_validation_acc_auc_composite"
        ),
        "validation_thresholds": thresholds,
        "validation": validation_by_threshold,
    }
    _atomic_json(output_dir / "best_evaluation.json", best_evaluation)
    return {
        "epochs_completed": len(history),
        "best_epoch": int(best_payload["epoch"]),
        "best_checkpoint": best_path,
        "last_checkpoint": output_dir / "last_checkpoint.pt",
        "history": history_path,
        "best_evaluation": output_dir / "best_evaluation.json",
        "validation_thresholds": thresholds,
    }


def evaluate_author_short_term(
    model: AuthorNoCoordinateShortTermClassifier,
    data_loader: Iterable,
    device: torch.device,
    positive_class_weight: float,
    label_smoothing: float,
    threshold: float,
) -> Dict[str, Any]:
    return run_author_short_term_epoch(
        model,
        data_loader,
        device,
        positive_class_weight,
        label_smoothing,
        threshold=threshold,
        include_predictions=True,
    )

