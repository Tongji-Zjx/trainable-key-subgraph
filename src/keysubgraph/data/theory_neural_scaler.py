"""Train-only scaling for Stage-1 neural inputs and auxiliary targets."""

from __future__ import absolute_import, division, print_function

import json
import os
from pathlib import Path

import torch

from keysubgraph.data.data_split import file_sha256
from keysubgraph.data.theory_neural_manifest import read_theory_neural_manifest
from keysubgraph.features.sv_hard_graph_features import SV_NODE_FEATURE_DIM
from keysubgraph.theory.sgw_core_features import SGW_CORE_DIM, SGW_QUANTILE_DIM


class TheoryNeuralScaler(torch.nn.Module):
    def __init__(
        self,
        node_mean,
        node_scale,
        quantile_mean,
        quantile_scale,
        transition_mean,
        transition_scale,
        train_manifest_sha256,
        protocol_sha256,
        selector_checkpoint_sha256,
        feature_schema_sha256,
        epsilon=1.0e-8,
    ):
        super().__init__()
        expected = (
            (node_mean, SV_NODE_FEATURE_DIM),
            (node_scale, SV_NODE_FEATURE_DIM),
            (quantile_mean, SGW_QUANTILE_DIM),
            (quantile_scale, SGW_QUANTILE_DIM),
            (transition_mean, SGW_CORE_DIM),
            (transition_scale, SGW_CORE_DIM),
        )
        if any(tuple(value.shape) != (dimension,) for value, dimension in expected):
            raise ValueError("Stage-1 scaler dimension mismatch")
        if any(bool((value <= 0.0).any()) for value in (
            node_scale, quantile_scale, transition_scale
        )):
            raise ValueError("Stage-1 scaler scale must be positive")
        self.register_buffer("node_mean", node_mean.to(torch.float32))
        self.register_buffer("node_scale", node_scale.to(torch.float32))
        self.register_buffer("quantile_mean", quantile_mean.to(torch.float32))
        self.register_buffer("quantile_scale", quantile_scale.to(torch.float32))
        self.register_buffer("transition_mean", transition_mean.to(torch.float32))
        self.register_buffer("transition_scale", transition_scale.to(torch.float32))
        self.train_manifest_sha256 = str(train_manifest_sha256)
        self.protocol_sha256 = str(protocol_sha256)
        self.selector_checkpoint_sha256 = str(selector_checkpoint_sha256)
        self.feature_schema_sha256 = str(feature_schema_sha256)
        self.epsilon = float(epsilon)

    def standardize_nodes(self, value):
        return (value - self.node_mean.to(value)) / self.node_scale.to(value)

    def standardize_quantiles(self, value):
        return (value - self.quantile_mean.to(value)) / self.quantile_scale.to(value)

    def standardize_transitions(self, value):
        return (
            value - self.transition_mean.to(value)
        ) / self.transition_scale.to(value)


def _mean_scale(values, epsilon):
    values = torch.cat(values, dim=0).to(torch.float64)
    mean = values.mean(dim=0)
    scale = torch.sqrt((values - mean).square().mean(dim=0) + float(epsilon))
    return mean.to(torch.float32), scale.to(torch.float32)


def fit_theory_neural_scaler(manifest_path, project_root, epsilon=1.0e-8):
    if epsilon <= 0.0:
        raise ValueError("Stage-1 scaler epsilon must be positive")
    manifest, records = read_theory_neural_manifest(manifest_path, project_root)
    if manifest["split"] != "train":
        raise ValueError("Stage-1 scaler must fit the inner-train split")
    nodes = []
    quantiles = []
    transitions = []
    for record in records:
        nodes.extend(
            window.node_features
            for window in record.windows
            if window is not None
        )
        quantiles.append(
            torch.stack(
                [
                    window.spectral_quantiles
                    for window in record.windows
                    if window is not None
                ],
                dim=0,
            )
        )
        if bool(record.transition_mask.any()):
            transitions.append(record.transition_features[record.transition_mask])
    if not nodes or not quantiles or not transitions:
        raise ValueError("Stage-1 train scaler has insufficient valid values")
    node_mean, node_scale = _mean_scale(nodes, epsilon)
    quantile_mean, quantile_scale = _mean_scale(quantiles, epsilon)
    transition_mean, transition_scale = _mean_scale(transitions, epsilon)
    return TheoryNeuralScaler(
        node_mean,
        node_scale,
        quantile_mean,
        quantile_scale,
        transition_mean,
        transition_scale,
        file_sha256(manifest_path),
        manifest["protocol_sha256"],
        manifest["selector_checkpoint_sha256"],
        manifest["feature_schema_sha256"],
        epsilon,
    )


def save_theory_neural_scaler(scaler, path, overwrite=False):
    path = Path(path).resolve()
    if path.exists() and not overwrite:
        raise FileExistsError("Stage-1 scaler already exists")
    payload = {
        "schema_version": 1,
        "artifact_type": "svg_theory_guided_neural_scaler",
        "fit_split": "train",
        "train_manifest_sha256": scaler.train_manifest_sha256,
        "protocol_sha256": scaler.protocol_sha256,
        "selector_checkpoint_sha256": scaler.selector_checkpoint_sha256,
        "feature_schema_sha256": scaler.feature_schema_sha256,
        "epsilon": scaler.epsilon,
        "node_mean": scaler.node_mean.tolist(),
        "node_scale": scaler.node_scale.tolist(),
        "quantile_mean": scaler.quantile_mean.tolist(),
        "quantile_scale": scaler.quantile_scale.tolist(),
        "transition_mean": scaler.transition_mean.tolist(),
        "transition_scale": scaler.transition_scale.tolist(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))
    return path


def load_theory_neural_scaler(path):
    with Path(path).resolve().open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("artifact_type") != "svg_theory_guided_neural_scaler":
        raise ValueError("unsupported Stage-1 scaler")
    return TheoryNeuralScaler(
        torch.tensor(payload["node_mean"]),
        torch.tensor(payload["node_scale"]),
        torch.tensor(payload["quantile_mean"]),
        torch.tensor(payload["quantile_scale"]),
        torch.tensor(payload["transition_mean"]),
        torch.tensor(payload["transition_scale"]),
        payload["train_manifest_sha256"],
        payload["protocol_sha256"],
        payload["selector_checkpoint_sha256"],
        payload["feature_schema_sha256"],
        payload["epsilon"],
    )
