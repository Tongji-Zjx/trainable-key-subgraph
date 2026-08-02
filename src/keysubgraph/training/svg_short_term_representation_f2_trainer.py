"""Training utilities for promoted representation-level F2 fusion."""

from __future__ import absolute_import, division, print_function

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from keysubgraph.data.svg_short_term_representation_f2 import (
    collate_svg_short_term_representation_f2,
)
from keysubgraph.models.svg_short_term_representation_f2 import (
    SVG_SHORT_TERM_REPRESENTATION_F2_MODEL_NAME,
    SVG_SHORT_TERM_REPRESENTATION_F2_SCHEMA_VERSION,
    SVGShortTermRepresentationF2,
    SVGShortTermRepresentationF2Config,
)
from keysubgraph.training.dual_sgw_feature_trainer import (
    binary_metrics,
    fit_binary_threshold,
)
from keysubgraph.training.sv_signed_gin_trainer import (
    site_stratified_roc_auc,
)
from keysubgraph.training.trainer import (
    class_weights_from_labels,
    set_reproducible_seed,
)


@dataclass(frozen=True)
class SVGShortTermRepresentationF2TrainingConfig:
    epochs: int = 80
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 15
    minimum_epochs: int = 5
    residual_auxiliary_weight: float = 0.25
    gate_penalty_weight: float = 1.0e-3
    seed: int = 42
    max_train_batches: Optional[int] = None
    max_validation_batches: Optional[int] = None

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.learning_rate <= 0.0:
            raise ValueError("representation F2 epochs/LR must be positive")
        if self.weight_decay < 0.0 or self.gradient_clip_norm <= 0.0:
            raise ValueError("invalid representation F2 optimizer settings")
        if self.early_stopping_patience < 0 or self.minimum_epochs < 0:
            raise ValueError("representation F2 patience cannot be negative")
        if (
            self.residual_auxiliary_weight < 0.0
            or self.gate_penalty_weight < 0.0
        ):
            raise ValueError("representation F2 loss weights cannot be negative")


def create_svg_short_term_representation_f2_loader(
    dataset,
    batch_size: int,
    seed: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
):
    if batch_size < 1 or num_workers < 0:
        raise ValueError("invalid representation F2 loader configuration")
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=bool(shuffle),
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_svg_short_term_representation_f2,
        generator=generator,
    )


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


def _trusted_load(path: Path, device: torch.device):
    try:
        return torch.load(
            str(Path(path).resolve()), map_location=device, weights_only=False
        )
    except TypeError:
        return torch.load(str(Path(path).resolve()), map_location=device)


def run_svg_short_term_representation_f2_epoch(
    model: SVGShortTermRepresentationF2,
    loader: Iterable,
    device: torch.device,
    class_weights: torch.Tensor,
    residual_auxiliary_weight: float,
    gate_penalty_weight: float,
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
    anchor_probabilities_all: List[float] = []
    keys_all: List[str] = []
    sites_all: List[str] = []
    total_loss = 0.0
    total_classification = 0.0
    total_auxiliary = 0.0
    total_count = 0
    gates: List[float] = []
    residual_magnitudes: List[float] = []
    started = time.perf_counter()
    class_weights = class_weights.to(device=device, dtype=torch.float32)
    for batch_index, cpu_batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = cpu_batch.to(device)
        labels = batch.labels.to(dtype=torch.long)
        targets = labels.to(dtype=torch.float32)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(
                batch.g2_anchor_logits,
                batch.short_term_representations,
            )
            per_sample = F.binary_cross_entropy_with_logits(
                output.logits, targets, reduction="none"
            )
            classification = (
                per_sample * class_weights[labels]
            ).mean()
            residual_target = targets - torch.sigmoid(
                batch.g2_anchor_logits.detach()
            )
            auxiliary = F.mse_loss(
                output.residual_probability_prediction,
                residual_target,
            )
            gate_penalty = output.gate
            loss = (
                classification
                + float(residual_auxiliary_weight) * auxiliary
                + float(gate_penalty_weight) * gate_penalty
            )
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), gradient_clip_norm
                )
                optimizer.step()
        probabilities = torch.sigmoid(output.logits.detach())
        anchor_probabilities = torch.sigmoid(
            batch.g2_anchor_logits.detach()
        )
        count = int(labels.numel())
        total_count += count
        total_loss += float(loss.detach().cpu()) * count
        total_classification += float(classification.detach().cpu()) * count
        total_auxiliary += float(auxiliary.detach().cpu()) * count
        labels_all.extend(int(value) for value in labels.cpu().tolist())
        probabilities_all.extend(float(value) for value in probabilities.cpu().tolist())
        anchor_probabilities_all.extend(
            float(value) for value in anchor_probabilities.cpu().tolist()
        )
        keys_all.extend(batch.sample_keys)
        sites_all.extend(batch.sites)
        gates.append(float(output.gate.detach().cpu()))
        residual_magnitudes.extend(
            float(value)
            for value in output.residual_logits.detach().abs().cpu().tolist()
        )
    if total_count < 1:
        raise ValueError("representation F2 epoch processed no samples")
    metrics = binary_metrics(labels_all, probabilities_all, threshold)
    metrics.update(
        {
            "loss": total_loss / float(total_count),
            "classification_loss": total_classification / float(total_count),
            "residual_auxiliary_loss": total_auxiliary / float(total_count),
            "site_stratified_roc_auc": site_stratified_roc_auc(
                labels_all, probabilities_all, sites_all
            ),
            "anchor_roc_auc": binary_metrics(
                labels_all, anchor_probabilities_all, 0.5
            )["roc_auc"],
            "gate": sum(gates) / float(len(gates)),
            "mean_absolute_residual_logit": sum(residual_magnitudes)
            / float(len(residual_magnitudes)),
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
                "anchor_positive_probability": anchor_probability,
            }
            for key, site, label, probability, anchor_probability in zip(
                keys_all,
                sites_all,
                labels_all,
                probabilities_all,
                anchor_probabilities_all,
            )
        ]
    return metrics


def _checkpoint_payload(
    model,
    optimizer,
    epoch,
    history,
    config,
    class_weights,
    provenance,
    best_epoch,
    best_auc,
):
    return {
        "model_name": SVG_SHORT_TERM_REPRESENTATION_F2_MODEL_NAME,
        "schema_version": SVG_SHORT_TERM_REPRESENTATION_F2_SCHEMA_VERSION,
        "model_config": model.config.to_dict(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "history": list(history),
        "training_config": asdict(config),
        "class_weights": class_weights.detach().cpu(),
        "provenance": dict(provenance),
        "best_epoch": int(best_epoch),
        "best_validation_roc_auc": float(best_auc),
        "validation_thresholds": None,
        "threshold_fit_split": "validation",
        "frozen_base_encoders": True,
        "fusion_level": "short_term_representation_to_g2_logit_residual",
    }


def train_svg_short_term_representation_f2(
    model,
    train_loader,
    validation_loader,
    train_labels,
    device,
    config,
    output_dir,
    provenance,
):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "history.json").exists():
        raise FileExistsError("representation F2 training output exists")
    set_reproducible_seed(config.seed)
    model.to(device)
    class_weights = class_weights_from_labels(train_labels).to(torch.float32)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    history = []
    best_auc = -float("inf")
    best_epoch = 0
    stale = 0
    for epoch in range(1, config.epochs + 1):
        train_metrics = run_svg_short_term_representation_f2_epoch(
            model,
            train_loader,
            device,
            class_weights,
            config.residual_auxiliary_weight,
            config.gate_penalty_weight,
            optimizer=optimizer,
            gradient_clip_norm=config.gradient_clip_norm,
            max_batches=config.max_train_batches,
        )
        validation_metrics = run_svg_short_term_representation_f2_epoch(
            model,
            validation_loader,
            device,
            class_weights,
            config.residual_auxiliary_weight,
            config.gate_penalty_weight,
            max_batches=config.max_validation_batches,
        )
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation_metrics,
            }
        )
        auc = validation_metrics.get("roc_auc")
        value = -float("inf") if auc is None else float(auc)
        if value > best_auc + 1.0e-12:
            best_auc = value
            best_epoch = epoch
            stale = 0
            _atomic_torch(
                output_dir / "best_checkpoint.pt",
                _checkpoint_payload(
                    model,
                    optimizer,
                    epoch,
                    history,
                    config,
                    class_weights,
                    provenance,
                    best_epoch,
                    best_auc,
                ),
            )
        else:
            stale += 1
        _atomic_json(output_dir / "history.json", history)
        print(
            "epoch {}/{} train_auc={} validation_auc={} anchor_auc={} gate={:.6f}".format(
                epoch,
                config.epochs,
                train_metrics["roc_auc"],
                validation_metrics["roc_auc"],
                validation_metrics["anchor_roc_auc"],
                validation_metrics["gate"],
            ),
            flush=True,
        )
        if (
            epoch >= config.minimum_epochs
            and stale >= config.early_stopping_patience
        ):
            break
    checkpoint = _trusted_load(output_dir / "best_checkpoint.pt", device)
    model.load_state_dict(checkpoint["model_state_dict"])
    validation = run_svg_short_term_representation_f2_epoch(
        model,
        validation_loader,
        device,
        class_weights,
        config.residual_auxiliary_weight,
        config.gate_penalty_weight,
        include_predictions=True,
    )
    predictions = validation.pop("predictions")
    labels = [int(row["label"]) for row in predictions]
    probabilities = [float(row["positive_probability"]) for row in predictions]
    thresholds = {
        name: fit_binary_threshold(labels, probabilities, name)
        for name in ("balanced_accuracy", "accuracy")
    }
    checkpoint["validation_thresholds"] = thresholds
    checkpoint["history"] = history
    _atomic_torch(output_dir / "best_checkpoint.pt", checkpoint)
    evaluation = {
        "artifact_type": "svg_short_term_representation_f2_best_evaluation",
        "best_epoch": int(best_epoch),
        "validation_thresholds": thresholds,
        "validation": {
            name: binary_metrics(labels, probabilities, threshold)
            for name, threshold in thresholds.items()
        },
        "gate": validation["gate"],
        "mean_absolute_residual_logit": validation[
            "mean_absolute_residual_logit"
        ],
        "anchor_roc_auc": validation["anchor_roc_auc"],
    }
    _atomic_json(output_dir / "best_evaluation.json", evaluation)
    return {
        "best_checkpoint": output_dir / "best_checkpoint.pt",
        "best_evaluation": output_dir / "best_evaluation.json",
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "validation_thresholds": thresholds,
    }


def load_svg_short_term_representation_f2_checkpoint(path, device):
    payload = _trusted_load(path, device)
    if (
        payload.get("model_name")
        != SVG_SHORT_TERM_REPRESENTATION_F2_MODEL_NAME
        or int(payload.get("schema_version", 0))
        != SVG_SHORT_TERM_REPRESENTATION_F2_SCHEMA_VERSION
    ):
        raise ValueError("not a representation F2 checkpoint")
    config = SVGShortTermRepresentationF2Config(**payload["model_config"])
    model = SVGShortTermRepresentationF2(config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    return model, payload

