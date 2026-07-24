"""Document-specified Exact-STSE coordinate-ablation baseline."""

from __future__ import absolute_import, division, print_function

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from keysubgraph.data.exact_stse_dataset import ExactSTSEBatch
from keysubgraph.features.graph_features import align_current_to_previous


@dataclass(frozen=True)
class ExactSTSEConfig:
    use_coordinates: bool = True
    community_vocab_size: int = 117
    community_embedding_dim: int = 16
    hidden_dim: int = 64
    ffn_dim: int = 128
    classifier_hidden_dims: Tuple[int, int] = (64, 32)
    dropout: float = 0.20
    epsilon: float = 1.0e-8
    subject_pooling: str = "window_mean_reproduction_assumption"

    def __post_init__(self) -> None:
        dimensions = (
            self.community_vocab_size,
            self.community_embedding_dim,
            self.hidden_dim,
            self.ffn_dim,
        ) + tuple(self.classifier_hidden_dims)
        if any(int(value) < 1 for value in dimensions):
            raise ValueError("Exact-STSE dimensions must be positive")
        if len(self.classifier_hidden_dims) != 2:
            raise ValueError("Exact-STSE classifier requires two hidden layers")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0,1)")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        if self.subject_pooling != "window_mean_reproduction_assumption":
            raise ValueError("Exact-STSE subject pooling must remain mean")

    @property
    def input_dim(self) -> int:
        return self.community_embedding_dim + (
            8 if self.use_coordinates else 2
        )

    @property
    def model_variant(self) -> str:
        return "exact_stse" if self.use_coordinates else "exact_stse_no_coord"


@dataclass(frozen=True)
class ExactSTSEWindowFeatures:
    values: torch.Tensor
    degree: torch.Tensor
    previous_degree: torch.Tensor
    delta_degree: torch.Tensor
    community_indices: torch.Tensor
    neighbor_coordinates: Optional[torch.Tensor]


@dataclass(frozen=True)
class ExactSTSEWindowEncoding:
    features: ExactSTSEWindowFeatures
    normalized_features: torch.Tensor
    projected_nodes: torch.Tensor
    residual_nodes: torch.Tensor
    encoded_nodes: torch.Tensor
    window_embedding: torch.Tensor


@dataclass(frozen=True)
class ExactSTSEOutput:
    logits: torch.Tensor
    subject_embedding: torch.Tensor
    window_embeddings: Tuple[torch.Tensor, ...]
    window_encodings: Tuple[Tuple[ExactSTSEWindowEncoding, ...], ...]
    diagnostics: Dict[str, Any]


class ExactSTSEFeatureBuilder(object):
    def __init__(self, config: ExactSTSEConfig) -> None:
        self.config = config

    def build(
        self,
        current_adjacency: torch.Tensor,
        previous_adjacency: torch.Tensor,
        current_names: Tuple[str, ...],
        previous_names: Tuple[str, ...],
        coordinates: torch.Tensor,
        communities: torch.Tensor,
    ) -> ExactSTSEWindowFeatures:
        node_count = int(current_adjacency.shape[0])
        if tuple(current_adjacency.shape) != (node_count, node_count):
            raise ValueError("current Exact-STSE adjacency must be square")
        if previous_adjacency.ndim != 2:
            raise ValueError("previous Exact-STSE adjacency must be square")
        if len(current_names) != node_count:
            raise ValueError("current node names do not align with adjacency")
        if tuple(communities.shape) != (node_count,):
            raise ValueError("community labels do not align with nodes")

        degree = current_adjacency.abs().sum(dim=-1)
        previous_raw = previous_adjacency.abs().sum(dim=-1)
        previous_indices_cpu, present_cpu = align_current_to_previous(
            current_names, previous_names
        )
        previous_indices = previous_indices_cpu.to(
            device=current_adjacency.device
        )
        present = present_cpu.to(device=current_adjacency.device)
        safe_indices = previous_indices.clamp_min(0)
        previous_degree = previous_raw.index_select(0, safe_indices)
        # The paper has no node-birth rule.  A node absent from the previous
        # window receives delta zero rather than a fabricated large change.
        previous_degree = torch.where(present, previous_degree, degree)
        delta_degree = degree - previous_degree

        community_indices = communities.to(dtype=torch.long) + 1
        if bool((community_indices < 0).any()) or bool(
            (community_indices >= self.config.community_vocab_size).any()
        ):
            raise ValueError("community id is outside the Exact-STSE vocabulary")

        neighbor_coordinates = None
        numeric_parts = [degree[:, None]]
        if self.config.use_coordinates:
            if tuple(coordinates.shape) != (node_count, 3):
                raise ValueError("Exact-STSE coordinates must have shape [N,3]")
            normalized_adjacency = current_adjacency / (
                degree[:, None] + self.config.epsilon
            )
            neighbor_coordinates = normalized_adjacency.matmul(coordinates)
            numeric_parts.extend((coordinates, neighbor_coordinates))
        numeric_parts.append(delta_degree[:, None])
        numeric = torch.cat(tuple(numeric_parts), dim=-1)
        return ExactSTSEWindowFeatures(
            values=numeric,
            degree=degree,
            previous_degree=previous_degree,
            delta_degree=delta_degree,
            community_indices=community_indices,
            neighbor_coordinates=neighbor_coordinates,
        )


class ExactSTSEWindowEncoder(nn.Module):
    def __init__(self, config: ExactSTSEConfig) -> None:
        super().__init__()
        self.config = config
        self.feature_builder = ExactSTSEFeatureBuilder(config)
        self.community_embedding = nn.Embedding(
            config.community_vocab_size,
            config.community_embedding_dim,
        )
        self.input_norm = nn.LayerNorm(config.input_dim)
        self.input_projection = nn.Linear(
            config.input_dim, config.hidden_dim
        )
        self.ffn_linear1 = nn.Linear(config.hidden_dim, config.ffn_dim)
        self.ffn_linear2 = nn.Linear(config.ffn_dim, config.hidden_dim)
        self.output_norm = nn.LayerNorm(config.hidden_dim)

    def forward(
        self,
        current_adjacency: torch.Tensor,
        previous_adjacency: torch.Tensor,
        current_names: Tuple[str, ...],
        previous_names: Tuple[str, ...],
        coordinates: torch.Tensor,
        communities: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
    ) -> ExactSTSEWindowEncoding:
        features = self.feature_builder.build(
            current_adjacency=current_adjacency,
            previous_adjacency=previous_adjacency,
            current_names=current_names,
            previous_names=previous_names,
            coordinates=coordinates,
            communities=communities,
        )
        community = self.community_embedding(features.community_indices)
        parts = [features.degree[:, None]]
        if self.config.use_coordinates:
            parts.extend((coordinates, features.neighbor_coordinates))
        parts.extend((community, features.delta_degree[:, None]))
        node_features = torch.cat(parts, dim=-1)
        if node_features.shape[-1] != self.config.input_dim:
            raise RuntimeError("Exact-STSE input feature dimension is invalid")
        normalized = self.input_norm(node_features)
        projected = self.input_projection(normalized)
        residual = self.ffn_linear2(F.gelu(self.ffn_linear1(projected)))
        encoded = self.output_norm(projected + residual)
        if node_mask is None:
            window = encoded.mean(dim=0)
        else:
            if tuple(node_mask.shape) != (encoded.shape[0],):
                raise ValueError("Exact-STSE node mask has invalid shape")
            valid = node_mask.to(device=encoded.device, dtype=encoded.dtype)
            if not bool(node_mask.to(dtype=torch.bool).any()):
                raise ValueError("Exact-STSE cannot pool an empty node set")
            window = (encoded * valid[:, None]).sum(dim=0)
            window = window / valid.sum().clamp_min(1.0)
        return ExactSTSEWindowEncoding(
            features=features,
            normalized_features=normalized,
            projected_nodes=projected,
            residual_nodes=residual,
            encoded_nodes=encoded,
            window_embedding=window,
        )


def _classifier(config: ExactSTSEConfig) -> nn.Sequential:
    first, second = config.classifier_hidden_dims
    return nn.Sequential(
        nn.Linear(config.hidden_dim, first),
        nn.ReLU(),
        nn.Dropout(config.dropout),
        nn.Linear(first, second),
        nn.ReLU(),
        nn.Dropout(config.dropout),
        nn.Linear(second, 2),
    )


class ExactSTSEClassifier(nn.Module):
    model_name = "document_specified_exact_stse"

    def __init__(self, config: ExactSTSEConfig) -> None:
        super().__init__()
        self.config = config
        self.window_encoder = ExactSTSEWindowEncoder(config)
        self.classifier = _classifier(config)

    def reset_parameters_with_seed(self, seed: int) -> None:
        """Give shared-shape modules paired initialization across ablations."""
        for name, module in self.named_modules():
            if module is self or not hasattr(module, "reset_parameters"):
                continue
            material = "{}\0{}".format(int(seed), name).encode("utf-8")
            module_seed = int.from_bytes(
                hashlib.sha256(material).digest()[:8], byteorder="big"
            ) % (2 ** 31)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(module_seed)
                module.reset_parameters()

    def forward(self, batch: ExactSTSEBatch) -> ExactSTSEOutput:
        if len(batch) < 1:
            raise ValueError("Exact-STSE cannot process an empty batch")
        subjects = []
        sample_windows = []
        all_encodings = []
        for sample in batch:
            graph = sample.graph
            windows = []
            encodings = []
            for time_index in range(graph.num_timepoints):
                previous_index = max(0, time_index - 1)
                encoding = self.window_encoder(
                    current_adjacency=graph.adjacency[time_index],
                    previous_adjacency=graph.adjacency[previous_index],
                    current_names=graph.node_names[time_index],
                    previous_names=graph.node_names[previous_index],
                    coordinates=sample.coordinates[time_index],
                    communities=graph.communities[time_index],
                )
                windows.append(encoding.window_embedding)
                encodings.append(encoding)
            sequence = torch.stack(windows, dim=0)
            subject = sequence.mean(dim=0)
            subjects.append(subject)
            sample_windows.append(sequence)
            all_encodings.append(tuple(encodings))
        subject_embedding = torch.stack(subjects, dim=0)
        logits = self.classifier(subject_embedding)
        return ExactSTSEOutput(
            logits=logits,
            subject_embedding=subject_embedding,
            window_embeddings=tuple(sample_windows),
            window_encodings=tuple(all_encodings),
            diagnostics={
                "variant": self.config.model_variant,
                "subject_pooling": self.config.subject_pooling,
                "input_dim": self.config.input_dim,
                "uses_coordinates": self.config.use_coordinates,
                "sequence_lengths": tuple(
                    int(item.shape[0]) for item in sample_windows
                ),
            },
        )
