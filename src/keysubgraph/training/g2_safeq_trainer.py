"""Training and leakage-safe validation selection for G2-SafeQ."""

from __future__ import absolute_import, division, print_function

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from keysubgraph.data.g2_safeq import collate_g2_safeq
from keysubgraph.models.g2_safeq import (
    G2_SAFEQ_MODEL_NAME,
    G2_SAFEQ_SCHEMA_VERSION,
    G2SafeQConfig,
    G2SafeQResidual,
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


G2_SAFEQ_GRID = (0.0, 0.25, 0.50, 0.75, 1.0)


@dataclass(frozen=True)
class G2SafeQTrainingConfig:
    epochs: int = 60
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 12
    minimum_epochs: int = 5
    seed: int = 42
    minimum_composite_gain: float = 0.005
    maximum_component_drop: float = 0.002
    max_train_batches: Optional[int] = None
    max_validation_batches: Optional[int] = None

    def __post_init__(self) -> None:
        if int(self.epochs) < 1 or float(self.learning_rate) <= 0.0:
            raise ValueError("SafeQ epochs/LR must be positive")
        if float(self.weight_decay) < 0.0:
            raise ValueError("SafeQ weight decay cannot be negative")
        if float(self.gradient_clip_norm) <= 0.0:
            raise ValueError("SafeQ gradient clipping must be positive")
        if int(self.early_stopping_patience) < 0:
            raise ValueError("SafeQ patience cannot be negative")
        if int(self.minimum_epochs) < 0:
            raise ValueError("SafeQ minimum epochs cannot be negative")
        if float(self.minimum_composite_gain) < 0.0:
            raise ValueError("SafeQ minimum gain cannot be negative")
        if float(self.maximum_component_drop) < 0.0:
            raise ValueError("SafeQ maximum component drop cannot be negative")


def create_g2_safeq_loader(
    dataset,
    batch_size: int,
    seed: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
):
    if int(batch_size) < 1 or int(num_workers) < 0:
        raise ValueError("invalid SafeQ loader configuration")
    if getattr(dataset, "split", None) != "train" and shuffle:
        raise ValueError("SafeQ validation/test loaders cannot shuffle")
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        collate_fn=collate_g2_safeq,
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


def _composite_auc(global_auc, site_auc) -> Optional[float]:
    if global_auc is None:
        return None
    if site_auc is None:
        return float(global_auc)
    return 0.5 * (float(global_auc) + float(site_auc))


def run_g2_safeq_epoch(
    model: G2SafeQResidual,
    loader: Iterable,
    device: torch.device,
    class_weights: torch.Tensor,
    optimizer: Optional[torch.optim.Optimizer] = None,
    gradient_clip_norm: float = 1.0,
    max_batches: Optional[int] = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    threshold: float = 0.5,
    include_predictions: bool = False,
) -> Dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    class_weights = class_weights.to(device=device, dtype=torch.float32)
    labels_all: List[int] = []
    probabilities_all: List[float] = []
    base_probabilities_all: List[float] = []
    static_probabilities_all: List[float] = []
    base_logits_all: List[float] = []
    static_logits_all: List[float] = []
    sites_all: List[str] = []
    keys_all: List[str] = []
    residuals_all: List[float] = []
    valid_transition_count = 0
    total_loss = 0.0
    total_count = 0
    started = time.perf_counter()
    for batch_index, cpu_batch in enumerate(loader):
        if max_batches is not None and batch_index >= int(max_batches):
            break
        batch = cpu_batch.to(device)
        labels = batch.labels.to(dtype=torch.long)
        targets = labels.to(dtype=torch.float32)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(
                batch.base_logits,
                batch.static_logits,
                batch.transition_summaries,
                batch.has_valid_transition,
                alpha=alpha,
                beta=beta,
            )
            per_sample = F.binary_cross_entropy_with_logits(
                output.logits, targets, reduction="none"
            )
            loss = (per_sample * class_weights[labels]).mean()
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(gradient_clip_norm)
                )
                optimizer.step()
        probabilities = torch.sigmoid(output.logits.detach())
        base_probabilities = torch.sigmoid(output.base_logits.detach())
        static_probabilities = torch.sigmoid(output.static_logits.detach())
        count = int(labels.numel())
        total_count += count
        total_loss += float(loss.detach().cpu()) * count
        labels_all.extend(int(value) for value in labels.cpu().tolist())
        probabilities_all.extend(
            float(value) for value in probabilities.cpu().tolist()
        )
        base_probabilities_all.extend(
            float(value) for value in base_probabilities.cpu().tolist()
        )
        static_probabilities_all.extend(
            float(value) for value in static_probabilities.cpu().tolist()
        )
        base_logits_all.extend(
            float(value)
            for value in output.base_logits.detach().cpu().tolist()
        )
        static_logits_all.extend(
            float(value)
            for value in output.static_logits.detach().cpu().tolist()
        )
        sites_all.extend(str(value) for value in batch.sites)
        keys_all.extend(str(value) for value in batch.sample_keys)
        residuals_all.extend(
            float(value)
            for value in output.residual_logits.detach().cpu().tolist()
        )
        valid_transition_count += int(batch.has_valid_transition.sum().cpu())
    if total_count < 1:
        raise ValueError("SafeQ epoch processed no samples")
    metrics = binary_metrics(labels_all, probabilities_all, threshold)
    metrics.update(
        {
            "loss": total_loss / float(total_count),
            "site_stratified_roc_auc": site_stratified_roc_auc(
                labels_all, probabilities_all, sites_all
            ),
            "base_roc_auc": binary_metrics(
                labels_all, base_probabilities_all, 0.5
            )["roc_auc"],
            "static_roc_auc": binary_metrics(
                labels_all, static_probabilities_all, 0.5
            )["roc_auc"],
            "mean_absolute_residual_logit": sum(
                abs(value) for value in residuals_all
            )
            / float(len(residuals_all)),
            "valid_transition_fraction": float(valid_transition_count)
            / float(total_count),
            "alpha": float(alpha),
            "beta": float(beta),
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    metrics["composite_auc"] = _composite_auc(
        metrics["roc_auc"], metrics["site_stratified_roc_auc"]
    )
    if include_predictions:
        metrics["predictions"] = [
            {
                "sample_key": key,
                "site": site,
                "label": label,
                "positive_probability": probability,
                "base_positive_probability": base_probability,
                "static_positive_probability": static_probability,
                "base_logit": float_base_logit,
                "static_logit": float_static_logit,
                "residual_logit": residual,
            }
            for (
                key,
                site,
                label,
                probability,
                base_probability,
                static_probability,
                float_base_logit,
                float_static_logit,
                residual,
            ) in zip(
                keys_all,
                sites_all,
                labels_all,
                probabilities_all,
                base_probabilities_all,
                static_probabilities_all,
                base_logits_all,
                static_logits_all,
                residuals_all,
            )
        ]
    return metrics


def select_g2_safeq_mixing(
    labels: Sequence[int],
    sites: Sequence[str],
    base_logits: Sequence[float],
    static_logits: Sequence[float],
    residual_logits: Sequence[float],
    split: str,
    grid: Sequence[float] = G2_SAFEQ_GRID,
    minimum_composite_gain: float = 0.005,
    maximum_component_drop: float = 0.002,
) -> Dict[str, Any]:
    """Fit alpha/beta on validation only, with an exact G2 fallback."""

    if str(split) != "validation":
        raise ValueError("SafeQ mixing may only be fit on validation")
    count = len(labels)
    if not (
        count > 0
        and len(sites) == count
        and len(base_logits) == count
        and len(static_logits) == count
        and len(residual_logits) == count
    ):
        raise ValueError("SafeQ mixing vectors are misaligned")
    grid_values = tuple(sorted(set(float(value) for value in grid)))
    if not grid_values or any(value < 0.0 or value > 1.0 for value in grid_values):
        raise ValueError("SafeQ mixing grid must lie in [0,1]")
    labels_list = [int(value) for value in labels]
    sites_list = [str(value) for value in sites]

    def candidate(alpha, beta):
        logits = [
            float(base)
            + float(beta) * (float(static) - float(base))
            + float(alpha) * float(residual)
            for base, static, residual in zip(
                base_logits, static_logits, residual_logits
            )
        ]
        probabilities = [
            float(torch.sigmoid(torch.tensor(value)).item())
            for value in logits
        ]
        metric = binary_metrics(labels_list, probabilities, 0.5)
        site_auc = site_stratified_roc_auc(
            labels_list, probabilities, sites_list
        )
        return {
            "alpha": float(alpha),
            "beta": float(beta),
            "roc_auc": metric["roc_auc"],
            "site_stratified_roc_auc": site_auc,
            "composite_auc": _composite_auc(metric["roc_auc"], site_auc),
            "probabilities": probabilities,
        }

    baseline = candidate(0.0, 0.0)
    candidates = []
    eligible = []
    for alpha in grid_values:
        for beta in grid_values:
            value = candidate(alpha, beta)
            value["eligible"] = False
            if alpha != 0.0 or beta != 0.0:
                global_ok = (
                    value["roc_auc"]
                    >= baseline["roc_auc"] - float(maximum_component_drop)
                )
                if baseline["site_stratified_roc_auc"] is None:
                    site_ok = True
                else:
                    site_ok = (
                        value["site_stratified_roc_auc"] is not None
                        and value["site_stratified_roc_auc"]
                        >= baseline["site_stratified_roc_auc"]
                        - float(maximum_component_drop)
                    )
                gain_ok = (
                    value["composite_auc"]
                    >= baseline["composite_auc"]
                    + float(minimum_composite_gain)
                )
                value["eligible"] = bool(global_ok and site_ok and gain_ok)
            candidates.append(value)
            if value["eligible"]:
                eligible.append(value)
    selected = baseline
    fallback = True
    if eligible:
        selected = sorted(
            eligible,
            key=lambda value: (
                -float(value["composite_auc"]),
                -float(value["roc_auc"]),
                float(value["alpha"]) + float(value["beta"]),
                float(value["alpha"]),
                float(value["beta"]),
            ),
        )[0]
        fallback = False
    thresholds = {
        strategy: fit_binary_threshold(
            labels_list, selected["probabilities"], strategy
        )
        for strategy in ("balanced_accuracy", "accuracy")
    }
    public_candidates = [
        {key: value for key, value in item.items() if key != "probabilities"}
        for item in candidates
    ]
    return {
        "fit_split": "validation",
        "grid": list(grid_values),
        "minimum_composite_gain": float(minimum_composite_gain),
        "maximum_component_drop": float(maximum_component_drop),
        "fallback_to_frozen_g2": fallback,
        "baseline": {
            key: value
            for key, value in baseline.items()
            if key != "probabilities"
        },
        "selected": {
            key: value
            for key, value in selected.items()
            if key != "probabilities"
        },
        "validation_thresholds": thresholds,
        "candidates": public_candidates,
    }


def _checkpoint_payload(
    model,
    optimizer,
    epoch,
    history,
    config,
    class_weights,
    provenance,
    best_epoch,
    best_composite_auc,
):
    return {
        "model_name": G2_SAFEQ_MODEL_NAME,
        "schema_version": G2_SAFEQ_SCHEMA_VERSION,
        "model_config": model.config.to_dict(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "history": list(history),
        "training_config": asdict(config),
        "class_weights": class_weights.detach().cpu(),
        "provenance": dict(provenance),
        "best_epoch": int(best_epoch),
        "best_validation_composite_auc": float(best_composite_auc),
        "mixing_selection": None,
        "validation_thresholds": None,
        "threshold_fit_split": "validation",
        "mixing_fit_split": "validation",
        "frozen_g2": True,
    }


def train_g2_safeq(
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
        raise FileExistsError("SafeQ training output exists")
    set_reproducible_seed(config.seed)
    model.to(device)
    class_weights = class_weights_from_labels(train_labels).to(torch.float32)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    history = []
    best_value = -float("inf")
    best_epoch = 0
    stale = 0
    for epoch in range(1, int(config.epochs) + 1):
        train_metrics = run_g2_safeq_epoch(
            model,
            train_loader,
            device,
            class_weights,
            optimizer=optimizer,
            gradient_clip_norm=config.gradient_clip_norm,
            max_batches=config.max_train_batches,
            alpha=1.0,
            beta=0.0,
        )
        validation_metrics = run_g2_safeq_epoch(
            model,
            validation_loader,
            device,
            class_weights,
            max_batches=config.max_validation_batches,
            alpha=1.0,
            beta=0.0,
        )
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation_metrics,
            }
        )
        current = validation_metrics.get("composite_auc")
        value = -float("inf") if current is None else float(current)
        if best_epoch == 0 or value > best_value + 1.0e-12:
            best_value = value
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
                    best_value,
                ),
            )
        else:
            stale += 1
        _atomic_json(output_dir / "history.json", history)
        print(
            "epoch {}/{} train_auc={} validation_auc={} site_auc={} "
            "base_auc={} residual={:.6f}".format(
                epoch,
                config.epochs,
                train_metrics["roc_auc"],
                validation_metrics["roc_auc"],
                validation_metrics["site_stratified_roc_auc"],
                validation_metrics["base_roc_auc"],
                validation_metrics["mean_absolute_residual_logit"],
            ),
            flush=True,
        )
        if (
            epoch >= int(config.minimum_epochs)
            and stale >= int(config.early_stopping_patience)
        ):
            break

    checkpoint = _trusted_load(output_dir / "best_checkpoint.pt", device)
    model.load_state_dict(checkpoint["model_state_dict"])
    validation = run_g2_safeq_epoch(
        model,
        validation_loader,
        device,
        class_weights,
        alpha=1.0,
        beta=0.0,
        include_predictions=True,
    )
    predictions = validation.pop("predictions")
    labels = [int(row["label"]) for row in predictions]
    sites = [str(row["site"]) for row in predictions]
    base_logits = [float(row["base_logit"]) for row in predictions]
    static_logits = [float(row["static_logit"]) for row in predictions]
    residual_logits = [float(row["residual_logit"]) for row in predictions]
    mixing = select_g2_safeq_mixing(
        labels,
        sites,
        base_logits,
        static_logits,
        residual_logits,
        split="validation",
        minimum_composite_gain=config.minimum_composite_gain,
        maximum_component_drop=config.maximum_component_drop,
    )
    checkpoint["mixing_selection"] = mixing
    checkpoint["validation_thresholds"] = mixing["validation_thresholds"]
    checkpoint["history"] = history
    _atomic_torch(output_dir / "best_checkpoint.pt", checkpoint)
    _atomic_json(
        output_dir / "best_evaluation.json",
        {
            "artifact_type": "g2_safeq_best_evaluation",
            "best_epoch": int(best_epoch),
            "mixing_selection": mixing,
            "frozen_g2": True,
        },
    )
    return {
        "best_checkpoint": output_dir / "best_checkpoint.pt",
        "best_evaluation": output_dir / "best_evaluation.json",
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "mixing_selection": mixing,
    }


def load_g2_safeq_checkpoint(path, device):
    payload = _trusted_load(path, device)
    if (
        payload.get("model_name") != G2_SAFEQ_MODEL_NAME
        or int(payload.get("schema_version", 0)) != G2_SAFEQ_SCHEMA_VERSION
    ):
        raise ValueError("not a SafeQ checkpoint")
    config = G2SafeQConfig(**payload["model_config"])
    model = G2SafeQResidual(config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    return model, payload
