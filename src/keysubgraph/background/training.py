"""Deterministic training and export for the MoKSE global background branch."""

from __future__ import absolute_import, division, print_function

import json
import hashlib
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
    signed_dropedge_probability: float = 0.0
    lambda_consistency: float = 0.0
    checkpoint_ensemble_top_k: int = 1
    strict_deterministic: bool = True

    def __post_init__(self):
        if min(self.epochs, self.batch_size, self.patience) < 1:
            raise ValueError("training integer settings must be positive")
        if min(self.learning_rate, self.gradient_clip) <= 0.0:
            raise ValueError("learning rate and gradient clip must be positive")
        if min(self.weight_decay, self.lambda_background, self.lambda_rank,
               self.lambda_gate, self.lambda_consistency) < 0.0:
            raise ValueError("loss weights and weight decay must be non-negative")
        if not 0.0 <= self.signed_dropedge_probability < 1.0:
            raise ValueError("signed DropEdge probability must be in [0,1)")
        if self.lambda_consistency > 0.0 and self.signed_dropedge_probability <= 0.0:
            raise ValueError("consistency loss requires a non-zero DropEdge probability")
        if self.checkpoint_ensemble_top_k < 1:
            raise ValueError("checkpoint ensemble size must be positive")


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
                    "community_labels": record.community_labels,
                    "raw_positive_adjacency": record.raw_positive_adjacency,
                    "raw_negative_adjacency": record.raw_negative_adjacency,
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
    raw_positive = torch.zeros_like(positive)
    raw_negative = torch.zeros_like(positive)
    mask = torch.zeros(count, maximum, dtype=torch.bool)
    communities = torch.full((count, maximum), -1, dtype=torch.long)
    for index, item in enumerate(items):
        nodes = int(item["node_features"].shape[0])
        features[index, :nodes] = item["node_features"]
        positive[index, :nodes, :nodes] = item["positive_adjacency"]
        negative[index, :nodes, :nodes] = item["negative_adjacency"]
        raw_positive[index, :nodes, :nodes] = item["raw_positive_adjacency"]
        raw_negative[index, :nodes, :nodes] = item["raw_negative_adjacency"]
        communities[index, :nodes] = item["community_labels"]
        mask[index, :nodes] = True
    return {
        "sample_keys": tuple(str(item["sample_key"]) for item in items),
        "sites": tuple(str(item["site"]) for item in items),
        "labels": torch.tensor([int(item["label"]) for item in items], dtype=torch.float32, device=device),
        "node_features": features.to(device),
        "positive_adjacency": positive.to(device),
        "negative_adjacency": negative.to(device),
        "raw_positive_adjacency": raw_positive.to(device),
        "raw_negative_adjacency": raw_negative.to(device),
        "community_labels": communities.to(device),
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


def _stable_view_seed(seed, epoch, sample_key, view_id, sign_name):
    payload = "{}|{}|{}|{}|{}".format(
        int(seed), int(epoch), sample_key, int(view_id), sign_name
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2 ** 63 - 1)


def normalize_raw_signed_batch(channel, node_mask, epsilon=1.0e-12):
    """Symmetrically degree-normalize a padded non-negative edge channel."""

    valid = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
    values = channel * valid.to(channel.dtype)
    degree = values.sum(dim=-1)
    inverse = torch.where(
        degree > epsilon, degree.clamp_min(epsilon).rsqrt(), torch.zeros_like(degree)
    )
    return inverse.unsqueeze(-1) * values * inverse.unsqueeze(-2)


def deterministic_signed_balanced_dropedge(
    raw_positive,
    raw_negative,
    node_mask,
    sample_keys,
    probability,
    seed,
    epoch,
    view_id=1,
):
    """Drop an equal fraction per signed channel and renormalize the graph.

    Sampling happens on upper-triangular real edges and is mirrored, so the
    result remains symmetric, never invents an edge, and preserves every
    retained raw weight.  Seeds are sample based rather than batch based.
    """

    if not 0.0 <= float(probability) < 1.0:
        raise ValueError("DropEdge probability must be in [0,1)")
    positive = raw_positive.clone()
    negative = raw_negative.clone()
    for batch_index, sample_key in enumerate(sample_keys):
        nodes = int(node_mask[batch_index].sum().item())
        for sign_name, channel in (("positive", positive), ("negative", negative)):
            upper = torch.nonzero(
                torch.triu(channel[batch_index, :nodes, :nodes] > 0.0, diagonal=1),
                as_tuple=False,
            )
            edge_count = int(upper.shape[0])
            drop_count = int(math.floor(float(probability) * edge_count))
            if drop_count < 1:
                continue
            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                _stable_view_seed(seed, epoch, sample_key, view_id, sign_name)
            )
            chosen = torch.randperm(edge_count, generator=generator)[:drop_count]
            selected = upper.detach().cpu().index_select(0, chosen)
            rows = selected[:, 0].to(channel.device)
            columns = selected[:, 1].to(channel.device)
            channel[batch_index, rows, columns] = 0.0
            channel[batch_index, columns, rows] = 0.0
    return (
        normalize_raw_signed_batch(positive, node_mask),
        normalize_raw_signed_batch(negative, node_mask),
        positive,
        negative,
    )


def background_consistency_loss(first, second):
    cosine = torch.nn.functional.cosine_similarity(
        first["background_representation"],
        second["background_representation"],
        dim=-1,
        eps=1.0e-8,
    )
    logit_difference = first["background_logit"] - second["background_logit"]
    return (1.0 - cosine + 0.1 * logit_difference.pow(2)).mean()


def _iter_batches(dataset, batch_size, order, device):
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        yield collate_background_batch([dataset[int(i)] for i in indices], device)


def _forward(model, batch, positive=None, negative=None):
    return model(
        batch["node_features"],
        batch["positive_adjacency"] if positive is None else positive,
        batch["negative_adjacency"] if negative is None else negative,
        batch["node_mask"],
        batch["evolution_logit"],
        community_labels=batch["community_labels"],
    )


def _score_logits(labels, logits, sites):
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
    result = classification_metrics(labels, probability, 0.5)
    result["site_stratified_roc_auc"] = site_stratified_roc_auc(
        labels.tolist(), probability.tolist(), sites
    )
    result["logit_mean"] = float(np.mean(logits))
    result["logit_standard_deviation"] = float(np.std(logits))
    return result


def _finalize_collected(collected, mode):
    primary = (
        collected["background_logits"] if mode == "background_only"
        else collected["fused_logits"]
    )
    labels = collected["labels"].astype(np.int64)
    branch_metrics = {
        "evolution": _score_logits(
            labels, collected["evolution_logits"], collected["sites"]
        ),
        "background": _score_logits(
            labels, collected["background_logits"], collected["sites"]
        ),
        "fusion": _score_logits(
            labels, collected["fused_logits"], collected["sites"]
        ),
    }
    metrics = _score_logits(labels, primary, collected["sites"])
    metrics["fusion_alpha"] = float(np.mean(collected["fusion_alpha"]))
    collected["metrics"] = metrics
    collected["branch_metrics"] = branch_metrics
    return collected


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
    return _finalize_collected(collected, mode)


def evaluate_background_ensemble(models, dataset, device, batch_size, mode):
    """Average same-run checkpoint outputs without touching evaluation labels."""

    if not models:
        raise ValueError("background checkpoint ensemble cannot be empty")
    results = [
        evaluate_background_model(model, dataset, device, batch_size, mode)
        for model in models
    ]
    reference = results[0]
    for result in results[1:]:
        if result["sample_keys"] != reference["sample_keys"]:
            raise ValueError("background ensemble sample order mismatch")
        if not np.array_equal(result["labels"], reference["labels"]):
            raise ValueError("background ensemble labels mismatch")
    averaged = {
        "sample_keys": list(reference["sample_keys"]),
        "sites": list(reference["sites"]),
        "labels": reference["labels"].copy(),
        "evolution_representations": reference["evolution_representations"].copy(),
        "evolution_logits": reference["evolution_logits"].copy(),
        "fusion_alpha": [],
    }
    for name in (
        "background_representations",
        "background_logits",
        "fused_logits",
        "background_residual",
    ):
        averaged[name] = np.mean(
            np.stack([result[name] for result in results], axis=0), axis=0
        )
    averaged["fusion_alpha"] = [
        float(np.mean(result["fusion_alpha"])) for result in results
    ]
    return _finalize_collected(averaged, mode)


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


def _cpu_state_dict(model):
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


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
    top_checkpoints = []
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
            augmented = None
            if training_config.signed_dropedge_probability > 0.0:
                drop_positive, drop_negative, _, _ = (
                    deterministic_signed_balanced_dropedge(
                        batch["raw_positive_adjacency"],
                        batch["raw_negative_adjacency"],
                        batch["node_mask"],
                        batch["sample_keys"],
                        training_config.signed_dropedge_probability,
                        training_config.seed,
                        epoch,
                        view_id=1,
                    )
                )
                augmented = _forward(
                    model, batch, positive=drop_positive, negative=drop_negative
                )
            if mode == "background_only":
                primary = output["background_logit"]
                if augmented is None:
                    averaged = primary
                    loss = bce(primary, batch["labels"])
                else:
                    augmented_primary = augmented["background_logit"]
                    averaged = 0.5 * (primary + augmented_primary)
                    loss = 0.5 * (
                        bce(primary, batch["labels"])
                        + bce(augmented_primary, batch["labels"])
                    )
                loss = loss + training_config.lambda_rank * pairwise_rank_loss(
                    averaged, batch["labels"]
                )
            else:
                primary = output["fused_logit"]
                if augmented is None:
                    averaged = primary
                    loss = bce(primary, batch["labels"])
                    background_bce = bce(
                        output["background_logit"], batch["labels"]
                    )
                else:
                    augmented_primary = augmented["fused_logit"]
                    averaged = 0.5 * (primary + augmented_primary)
                    loss = 0.5 * (
                        bce(primary, batch["labels"])
                        + bce(augmented_primary, batch["labels"])
                    )
                    background_bce = 0.5 * (
                        bce(output["background_logit"], batch["labels"])
                        + bce(augmented["background_logit"], batch["labels"])
                    )
                loss = loss + training_config.lambda_background * background_bce
                loss = loss + training_config.lambda_rank * pairwise_rank_loss(
                    averaged, batch["labels"]
                )
                loss = loss + training_config.lambda_gate * output["fusion_alpha"].pow(2)
            if augmented is not None and training_config.lambda_consistency > 0.0:
                loss = loss + training_config.lambda_consistency * (
                    background_consistency_loss(output, augmented)
                )
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
        checkpoint_candidate = {
            "key": key,
            "epoch": epoch,
            "state_dict": _cpu_state_dict(model),
            "validation_metrics": metric,
            "train_loss": float(np.mean(losses)),
        }
        top_checkpoints.append(checkpoint_candidate)
        top_checkpoints.sort(key=lambda item: item["key"], reverse=True)
        top_checkpoints = top_checkpoints[: training_config.checkpoint_ensemble_top_k]
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
    top_checkpoints.sort(key=lambda item: item["key"], reverse=True)
    ensemble_models = []
    for rank, item in enumerate(top_checkpoints, start=1):
        member = MoKSEBackgroundModel(model_config).to(device)
        member.load_state_dict(item["state_dict"])
        member.eval()
        ensemble_models.append(member)
        member_payload = {
            "artifact_type": "mokse_background_checkpoint_v2",
            "rank": int(rank),
            "epoch": int(item["epoch"]),
            "mode": mode,
            "model_config": asdict(model_config),
            "training_config": asdict(training_config),
            "feature_scaler": scaler.as_dict(),
            "model_state_dict": item["state_dict"],
            "validation_metrics": item["validation_metrics"],
            "selection_rule": "validation_AUROC_then_ACC@0.5_then_loss",
        }
        torch.save(member_payload, str(output_dir / "checkpoint_top_{}.pt".format(rank)))
    model.load_state_dict(checkpoint["model_state_dict"])
    results = {}
    evaluation_datasets = [
        ("train", train_dataset),
        ("validation", validation_dataset),
    ]
    if test_dataset is not None:
        evaluation_datasets.append(("test", test_dataset))
    for split, dataset in evaluation_datasets:
        result = evaluate_background_ensemble(
            ensemble_models, dataset, device, training_config.batch_size, mode
        )
        results[split] = result
        _save_export(output_dir / (split + "_features.npz"), result)
    report = {
        "artifact_type": "mokse_background_evaluation_v1",
        "mode": mode,
        "best_epoch": int(checkpoint["epoch"]),
        "checkpoint_ensemble_size": len(ensemble_models),
        "checkpoint_ensemble_epochs": [
            int(item["epoch"]) for item in top_checkpoints
        ],
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
