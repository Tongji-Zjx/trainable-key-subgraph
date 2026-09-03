"""Deterministic training and export for the MoKSE global background branch."""

from __future__ import absolute_import, division, print_function

import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from keysubgraph.tge.trainer import classification_metrics, site_stratified_roc_auc

from .data import BackgroundFeatureScaler, GlobalStaticGraphRecord
from .model import MoKSEBackgroundModel, StaticBackgroundConfig


@dataclass(frozen=True)
class BackgroundTrainingConfig:
    epochs: int = 120
    batch_size: int = 16
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    gradient_clip: float = 5.0
    patience: int = 15
    seed: int = 43
    lambda_background: float = 0.20
    lambda_rank: float = 0.05
    lambda_gate: float = 1.0e-3
    strict_deterministic: bool = True

    def __post_init__(self):
        if min(self.epochs, self.batch_size, self.patience) < 1:
            raise ValueError("training integer settings must be positive")
        if min(self.learning_rate, self.gradient_clip) <= 0.0:
            raise ValueError("learning rate and gradient clip must be positive")
        if min(self.weight_decay, self.lambda_background, self.lambda_rank,
               self.lambda_gate) < 0.0:
            raise ValueError("loss weights and weight decay must be non-negative")


class BackgroundFusionDataset:
    def __init__(
        self,
        records: Sequence[GlobalStaticGraphRecord],
        evolution_split: Mapping[str, object],
        scaler: BackgroundFeatureScaler,
    ):
        rows = tuple(evolution_split["rows"])
        if len(records) != len(rows):
            raise ValueError("background/evolution row count mismatch")
        self.items = []
        evolution_logits = np.asarray(evolution_split["base_logits"], dtype=np.float32)
        evolution_repr = np.asarray(
            evolution_split["sample_embeddings"], dtype=np.float32
        )
        labels = np.asarray(evolution_split["labels"], dtype=np.int64)
        for index, (record, row) in enumerate(zip(records, rows)):
            if record.sample_key != str(row["sample_key"]):
                raise ValueError("background/evolution sample order mismatch")
            if record.label != int(labels[index]) or record.site != str(row["site"]):
                raise ValueError("background/evolution provenance mismatch")
            self.items.append(
                {
                    "sample_key": record.sample_key,
                    "sample_id": record.sample_id,
                    "site": record.site,
                    "label": record.label,
                    "node_features": scaler.transform(record.node_features),
                    "positive_adjacency": record.positive_adjacency,
                    "negative_adjacency": record.negative_adjacency,
                    "evolution_logit": float(evolution_logits[index]),
                    "evolution_representation": torch.from_numpy(
                        evolution_repr[index].copy()
                    ),
                }
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate_background_batch(items: Sequence[Mapping[str, object]], device):
    count = len(items)
    maximum = max(int(item["node_features"].shape[0]) for item in items)
    feature_dim = int(items[0]["node_features"].shape[1])
    features = torch.zeros(count, maximum, feature_dim, dtype=torch.float32)
    positive = torch.zeros(count, maximum, maximum, dtype=torch.float32)
    negative = torch.zeros_like(positive)
    mask = torch.zeros(count, maximum, dtype=torch.bool)
    for index, item in enumerate(items):
        nodes = int(item["node_features"].shape[0])
        features[index, :nodes] = item["node_features"]
        positive[index, :nodes, :nodes] = item["positive_adjacency"]
        negative[index, :nodes, :nodes] = item["negative_adjacency"]
        mask[index, :nodes] = True
    return {
        "sample_keys": tuple(str(item["sample_key"]) for item in items),
        "sites": tuple(str(item["site"]) for item in items),
        "labels": torch.tensor([int(item["label"]) for item in items], dtype=torch.float32, device=device),
        "node_features": features.to(device),
        "positive_adjacency": positive.to(device),
        "negative_adjacency": negative.to(device),
        "node_mask": mask.to(device),
        "evolution_logit": torch.tensor(
            [float(item["evolution_logit"]) for item in items],
            dtype=torch.float32,
            device=device,
        ),
        "evolution_representation": torch.stack(
            [item["evolution_representation"] for item in items], dim=0
        ).to(device),
    }


def set_background_seed(seed: int, strict: bool = True):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if strict:
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def pairwise_rank_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    positive = logits[labels > 0.5]
    negative = logits[labels <= 0.5]
    if positive.numel() == 0 or negative.numel() == 0:
        return logits.sum() * 0.0
    differences = positive[:, None] - negative[None, :]
    return torch.nn.functional.softplus(-differences).mean()


def _iter_batches(dataset, batch_size, order, device):
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        yield collate_background_batch([dataset[int(i)] for i in indices], device)


def _forward(model, batch):
    return model(
        batch["node_features"],
        batch["positive_adjacency"],
        batch["negative_adjacency"],
        batch["node_mask"],
        batch["evolution_logit"],
    )


def evaluate_background_model(model, dataset, device, batch_size, mode):
    model.eval()
    collected = {
        "sample_keys": [], "sites": [], "labels": [],
        "evolution_representations": [], "evolution_logits": [],
        "background_representations": [], "background_logits": [],
        "fused_logits": [], "fusion_alpha": [], "background_residual": [],
    }
    with torch.no_grad():
        order = list(range(len(dataset)))
        for batch in _iter_batches(dataset, batch_size, order, device):
            output = _forward(model, batch)
            collected["sample_keys"].extend(batch["sample_keys"])
            collected["sites"].extend(batch["sites"])
            collected["labels"].append(batch["labels"].cpu().numpy())
            collected["evolution_representations"].append(
                batch["evolution_representation"].cpu().numpy()
            )
            collected["evolution_logits"].append(batch["evolution_logit"].cpu().numpy())
            collected["background_representations"].append(
                output["background_representation"].cpu().numpy()
            )
            collected["background_logits"].append(output["background_logit"].cpu().numpy())
            collected["fused_logits"].append(output["fused_logit"].cpu().numpy())
            collected["background_residual"].append(output["background_residual"].cpu().numpy())
            collected["fusion_alpha"].append(float(output["fusion_alpha"].cpu()))
    for name in (
        "labels", "evolution_representations", "evolution_logits",
        "background_representations", "background_logits", "fused_logits",
        "background_residual",
    ):
        collected[name] = np.concatenate(collected[name], axis=0)
    primary = (
        collected["background_logits"] if mode == "background_only"
        else collected["fused_logits"]
    )
    labels = collected["labels"].astype(np.int64)
    def score(logits):
        probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
        result = classification_metrics(labels, probability, 0.5)
        result["site_stratified_roc_auc"] = site_stratified_roc_auc(
            labels.tolist(), probability.tolist(), collected["sites"]
        )
        result["logit_mean"] = float(np.mean(logits))
        result["logit_standard_deviation"] = float(np.std(logits))
        return result
    branch_metrics = {
        "evolution": score(collected["evolution_logits"]),
        "background": score(collected["background_logits"]),
        "fusion": score(collected["fused_logits"]),
    }
    metrics = score(primary)
    metrics["fusion_alpha"] = float(np.mean(collected["fusion_alpha"]))
    collected["metrics"] = metrics
    collected["branch_metrics"] = branch_metrics
    return collected


def _atomic_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _save_export(path: Path, result):
    np.savez_compressed(
        str(path),
        sample_keys=np.asarray(result["sample_keys"], dtype=str),
        sites=np.asarray(result["sites"], dtype=str),
        labels=result["labels"].astype(np.int64),
        evolution_representations=result["evolution_representations"].astype(np.float32),
        evolution_logits=result["evolution_logits"].astype(np.float32),
        background_representations=result["background_representations"].astype(np.float32),
        background_logits=result["background_logits"].astype(np.float32),
        fused_logits=result["fused_logits"].astype(np.float32),
        background_residual=result["background_residual"].astype(np.float32),
    )


def train_background_model(
    train_dataset,
    validation_dataset,
    test_dataset,
    scaler,
    output_dir: Path,
    device,
    mode: str,
    model_config=StaticBackgroundConfig(),
    training_config=BackgroundTrainingConfig(),
):
    if mode not in ("background_only", "fusion"):
        raise ValueError("unsupported background training mode")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_background_seed(training_config.seed, training_config.strict_deterministic)
    model = MoKSEBackgroundModel(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    bce = nn.BCEWithLogitsLoss()
    generator = np.random.RandomState(training_config.seed)
    history = []
    best = None
    stale = 0
    checkpoint_path = output_dir / "best_checkpoint.pt"

    for epoch in range(1, training_config.epochs + 1):
        model.train()
        losses = []
        order = generator.permutation(len(train_dataset)).tolist()
        for batch in _iter_batches(
            train_dataset, training_config.batch_size, order, device
        ):
            optimizer.zero_grad(set_to_none=True)
            output = _forward(model, batch)
            if mode == "background_only":
                primary = output["background_logit"]
                loss = bce(primary, batch["labels"])
                loss = loss + training_config.lambda_rank * pairwise_rank_loss(
                    primary, batch["labels"]
                )
            else:
                primary = output["fused_logit"]
                loss = bce(primary, batch["labels"])
                loss = loss + training_config.lambda_background * bce(
                    output["background_logit"], batch["labels"]
                )
                loss = loss + training_config.lambda_rank * pairwise_rank_loss(
                    primary, batch["labels"]
                )
                loss = loss + training_config.lambda_gate * output["fusion_alpha"].pow(2)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite background training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.gradient_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation = evaluate_background_model(
            model, validation_dataset, device, training_config.batch_size, mode
        )
        metric = validation["metrics"]
        key = (
            -float("inf") if metric.get("roc_auc") is None else float(metric["roc_auc"]),
            float(metric["accuracy"]),
            -float(np.mean(losses)),
        )
        improved = best is None or key > best
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation": metric,
            "improved": bool(improved),
        }
        history.append(row)
        print(
            "epoch {}/{} mode={} loss={:.6f} val_auc={} val_acc={:.6f} alpha={:.6f}".format(
                epoch, training_config.epochs, mode, row["train_loss"],
                metric.get("roc_auc"), metric["accuracy"], metric["fusion_alpha"],
            ), flush=True,
        )
        if improved:
            best = key
            stale = 0
            torch.save(
                {
                    "artifact_type": "mokse_background_checkpoint_v1",
                    "epoch": epoch,
                    "mode": mode,
                    "model_config": asdict(model_config),
                    "training_config": asdict(training_config),
                    "feature_scaler": scaler.as_dict(),
                    "model_state_dict": model.state_dict(),
                    "validation_metrics": metric,
                    "selection_rule": "validation_AUROC_then_ACC@0.5_then_loss",
                },
                str(checkpoint_path),
            )
        else:
            stale += 1
        if stale >= training_config.patience:
            break

    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    results = {}
    for split, dataset in (
        ("train", train_dataset), ("validation", validation_dataset), ("test", test_dataset)
    ):
        result = evaluate_background_model(
            model, dataset, device, training_config.batch_size, mode
        )
        results[split] = result
        _save_export(output_dir / (split + "_features.npz"), result)
    report = {
        "artifact_type": "mokse_background_evaluation_v1",
        "mode": mode,
        "best_epoch": int(checkpoint["epoch"]),
        "selection_rule": checkpoint["selection_rule"],
        "test_used_for_selection": False,
        "metrics": {name: result["metrics"] for name, result in results.items()},
        "branch_metrics": {
            name: result["branch_metrics"] for name, result in results.items()
        },
    }
    _atomic_json(output_dir / "evaluation.json", report)
    _atomic_json(output_dir / "history.json", {"history": history})
    _atomic_json(output_dir / "feature_scaler.json", scaler.as_dict())
    return report
