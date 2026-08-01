"""Coordinate-free reproduction of the authors' local short-term branch.

The neural operations in this module follow ``model_fusion_multi_gnn_commu``.
The only intentional architecture change is the removal of raw coordinates
and signed neighbour-coordinate aggregation.  Dataset identity and labels are
provided by the project's frozen protocol rather than filename basenames.
"""

from __future__ import absolute_import, division, print_function

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import torch
from torch import nn

from keysubgraph.data.graph_dataset import GraphSequenceBatch
from keysubgraph.features.paper_short_term_pst import (
    PaperShortTermCommunityFrequency,
)


AUTHOR_SHORT_TERM_MODEL_NAME = "author_no_coordinate_short_term"
AUTHOR_SHORT_TERM_PROFILES = ("adhd", "wmrc")


@dataclass(frozen=True)
class AuthorShortTermConfig:
    """Architecture values frozen from the authors' supplied programs."""

    window_embedding_dim: int
    transformer_layers: int
    transformer_heads: int = 8
    community_vocab_size: int = 128
    community_embedding_dim: int = 32
    memory_slots: int = 32
    memory_dim: int = 128
    transformer_dropout: float = 0.27
    window_dropout: float = 0.10
    maximum_windows: int = 200
    maximum_nodes: int = 116

    def __post_init__(self) -> None:
        dimensions = (
            self.window_embedding_dim,
            self.transformer_layers,
            self.transformer_heads,
            self.community_vocab_size,
            self.community_embedding_dim,
            self.memory_slots,
            self.memory_dim,
            self.maximum_windows,
            self.maximum_nodes,
        )
        if any(int(value) <= 0 for value in dimensions):
            raise ValueError("author short-term dimensions must be positive")
        if self.window_embedding_dim % self.transformer_heads != 0:
            raise ValueError("window embedding must be divisible by heads")
        for value in (self.transformer_dropout, self.window_dropout):
            if not 0.0 <= float(value) < 1.0:
                raise ValueError("dropout must lie in [0,1)")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AuthorShortTermConfig":
        return cls(**dict(payload))


def author_short_term_config(profile: str) -> AuthorShortTermConfig:
    """Return the dataset-specific architecture in the author wrappers."""

    if profile == "adhd":
        return AuthorShortTermConfig(
            window_embedding_dim=192,
            transformer_layers=3,
        )
    if profile == "wmrc":
        return AuthorShortTermConfig(
            window_embedding_dim=96,
            transformer_layers=2,
        )
    raise ValueError("unsupported author short-term profile")


class AuthorPositionalEncoding(nn.Module):
    """Sinusoidal encoding copied from the supplied author model."""

    def __init__(self, dimension: int, maximum_length: int = 200) -> None:
        super().__init__()
        values = torch.zeros(maximum_length, dimension)
        position = torch.arange(
            0, maximum_length, dtype=torch.float32
        ).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, dimension, 2, dtype=torch.float32)
            * (-math.log(10000.0) / float(dimension))
        )
        values[:, 0::2] = torch.sin(position * divisor)
        values[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("pe", values.unsqueeze(0))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[1] > self.pe.shape[1]:
            raise ValueError("author positional encoding length exceeded")
        return values + self.pe[:, : values.shape[1], :]


class AuthorNoCoordinateWindowEncoder(nn.Module):
    """Author WindowEncoder with only the two coordinate blocks removed."""

    def __init__(self, config: AuthorShortTermConfig) -> None:
        super().__init__()
        self.community_embedding = nn.Embedding(
            config.community_vocab_size + 1,
            config.community_embedding_dim,
            padding_idx=0,
        )
        self.node_feature_dim = 2 + config.community_embedding_dim
        self.input_norm = nn.LayerNorm(self.node_feature_dim)
        self.input_projection = nn.Linear(
            self.node_feature_dim,
            config.window_embedding_dim,
        )
        self.ffn = nn.Sequential(
            nn.Linear(
                config.window_embedding_dim,
                config.window_embedding_dim * 2,
            ),
            nn.GELU(),
            nn.Linear(
                config.window_embedding_dim * 2,
                config.window_embedding_dim,
            ),
        )
        self.output_norm = nn.LayerNorm(config.window_embedding_dim)
        self.dropout = nn.Dropout(config.window_dropout)

    def forward(
        self,
        current_adjacency: torch.Tensor,
        previous_adjacency: torch.Tensor,
        communities: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        current_degree = current_adjacency.abs().sum(dim=-1)
        previous_degree = previous_adjacency.abs().sum(dim=-1)
        delta_degree = current_degree - previous_degree
        community_indices = (communities + 1).clamp(
            min=0,
            max=self.community_embedding.num_embeddings - 1,
        )
        community_features = self.community_embedding(community_indices)
        node_features = torch.cat(
            (
                current_degree.unsqueeze(-1),
                community_features,
                delta_degree.unsqueeze(-1),
            ),
            dim=-1,
        )
        projected = self.input_projection(self.input_norm(node_features))
        encoded = self.output_norm(projected + self.ffn(projected))
        encoded = self.dropout(encoded)
        return encoded.mean(dim=1), {
            "absolute_degree": current_degree,
            "delta_absolute_degree": delta_degree,
            "community_indices": community_indices,
            "node_features": node_features,
            "encoded_nodes": encoded,
        }


class AuthorBrainFunctionMemory(nn.Module):
    """Exact differentiable read path of the supplied MemoryModule."""

    def __init__(self, slots: int, memory_dim: int, input_dim: int) -> None:
        super().__init__()
        self.slots = int(slots)
        self.memory_dim = int(memory_dim)
        self.memory = nn.Parameter(torch.randn(1, slots, memory_dim))
        self.input_projection = nn.Linear(input_dim, memory_dim)
        self.input_norm = nn.LayerNorm(input_dim)
        # Retained for state compatibility with the author implementation.
        # The supplied ADHD/WMRC wrappers do not enable explicit writes.
        self.write_gate = nn.Sequential(
            nn.Linear(memory_dim * 2, memory_dim),
            nn.Sigmoid(),
        )

    def read(self, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        projected = self.input_projection(self.input_norm(values))
        memory = self.memory.expand(values.shape[0], -1, -1)
        attention = torch.softmax(
            torch.matmul(
                projected.unsqueeze(1), memory.transpose(1, 2)
            ),
            dim=-1,
        )
        readout = torch.matmul(attention, memory).squeeze(1)
        return readout, attention.squeeze(1)


@dataclass(frozen=True)
class AuthorShortTermOutput:
    logits: torch.Tensor
    final_representation: torch.Tensor
    cls_representation: torch.Tensor
    memory_representation: torch.Tensor
    memory_attention: torch.Tensor
    graph_statistics: torch.Tensor
    window_embeddings: torch.Tensor
    time_mask: torch.Tensor
    diagnostics: Dict[str, Any]


class AuthorNoCoordinateShortTermClassifier(nn.Module):
    """Standalone local short-term classifier from the author full model."""

    model_name = AUTHOR_SHORT_TERM_MODEL_NAME

    def __init__(
        self,
        config: AuthorShortTermConfig,
        community_frequency: PaperShortTermCommunityFrequency,
        initial_positive_probability: float = 0.75,
    ) -> None:
        super().__init__()
        if not 0.0 < initial_positive_probability < 1.0:
            raise ValueError("initial positive probability must lie in (0,1)")
        self.config = config
        self.community_frequency = community_frequency
        self.window_encoder = AuthorNoCoordinateWindowEncoder(config)
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, config.window_embedding_dim)
        )
        self.position_encoding = AuthorPositionalEncoding(
            config.window_embedding_dim,
            config.maximum_windows,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.window_embedding_dim,
            nhead=config.transformer_heads,
            dim_feedforward=config.window_embedding_dim * 4,
            dropout=config.transformer_dropout,
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.transformer_layers,
        )
        self.memory = AuthorBrainFunctionMemory(
            config.memory_slots,
            config.memory_dim,
            config.window_embedding_dim,
        )
        self.anomaly_projection = nn.Linear(
            1, config.window_embedding_dim
        )
        maximum_label = max(
            (label for label, _ in community_frequency.counts),
            default=0,
        )
        counts = torch.zeros(maximum_label + 1, dtype=torch.float32)
        for label, count in community_frequency.counts:
            counts[int(label)] = float(count)
        epsilon = 1.0e-8
        self.register_buffer(
            "community_log_probabilities",
            torch.log(
                (counts + epsilon)
                / (float(community_frequency.total_count) + epsilon)
            ),
        )
        self.graph_feature_projection = nn.Sequential(
            nn.LayerNorm(3),
            nn.Linear(3, config.memory_dim),
            nn.GELU(),
        )
        classifier_dim = (
            config.window_embedding_dim + config.memory_dim * 2
        )
        self.classifier_norm = nn.LayerNorm(classifier_dim)
        self.classifier = nn.Linear(classifier_dim, 1)
        initial_bias = math.log(
            initial_positive_probability
            / (1.0 - initial_positive_probability + 1.0e-8)
        )
        with torch.no_grad():
            self.classifier.bias.fill_(initial_bias)

    def _pad_batch(
        self, batch: GraphSequenceBatch
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(batch) <= 0:
            raise ValueError("author short-term batch is empty")
        lengths = torch.tensor(
            [sample.num_timepoints for sample in batch.samples],
            dtype=torch.long,
            device=batch.samples[0].adjacency[0].device,
        )
        maximum_time = int(lengths.max().item())
        if maximum_time > self.config.maximum_windows:
            raise ValueError("sample exceeds author maximum window count")
        maximum_nodes = self.config.maximum_nodes
        reference = batch.samples[0].adjacency[0]
        adjacency = reference.new_zeros(
            (len(batch), maximum_time, maximum_nodes, maximum_nodes)
        )
        communities = torch.full(
            (len(batch), maximum_time, maximum_nodes),
            -1,
            dtype=torch.long,
            device=reference.device,
        )
        for batch_index, sample in enumerate(batch.samples):
            for time_index, (graph, labels) in enumerate(
                zip(sample.adjacency, sample.communities)
            ):
                nodes = int(graph.shape[0])
                if nodes > maximum_nodes:
                    raise ValueError(
                        "graph exceeds author maximum node count; truncation is forbidden"
                    )
                adjacency[
                    batch_index, time_index, :nodes, :nodes
                ] = graph
                communities[
                    batch_index, time_index, :nodes
                ] = labels
        return adjacency, communities, lengths

    def _community_anomaly(
        self, communities: torch.Tensor
    ) -> torch.Tensor:
        lookup = self.community_log_probabilities
        indices = (communities + 1).clamp(
            min=0, max=int(lookup.numel()) - 1
        )
        log_probability = lookup[indices]
        valid = (communities >= 0).to(dtype=log_probability.dtype)
        count = valid.sum(dim=-1).clamp(min=1.0)
        return -((log_probability * valid).sum(dim=-1) / count)

    def forward(self, batch: GraphSequenceBatch) -> AuthorShortTermOutput:
        adjacency, communities, lengths = self._pad_batch(batch)
        batch_size, maximum_time, _, _ = adjacency.shape
        previous = torch.cat(
            (torch.zeros_like(adjacency[:, :1]), adjacency[:, :-1]),
            dim=1,
        )
        with torch.no_grad():
            anomalies = self._community_anomaly(communities)
        windows: List[torch.Tensor] = []
        first_delta = None
        for time_index in range(maximum_time):
            embedding, diagnostics = self.window_encoder(
                adjacency[:, time_index],
                previous[:, time_index],
                communities[:, time_index],
            )
            if time_index == 0:
                first_delta = diagnostics["delta_absolute_degree"]
            embedding = embedding + self.anomaly_projection(
                anomalies[:, time_index].unsqueeze(-1)
            )
            windows.append(embedding.unsqueeze(1))
        window_embeddings = torch.cat(tuple(windows), dim=1)
        encoded_windows = self.position_encoding(window_embeddings)
        cls = self.cls_token.expand(batch_size, -1, -1)
        temporal_input = torch.cat((cls, encoded_windows), dim=1)
        padding_mask = (
            torch.arange(
                maximum_time + 1, device=adjacency.device
            ).unsqueeze(0)
            >= (lengths + 1).unsqueeze(1)
        )
        temporal_output = self.temporal_encoder(
            temporal_input,
            src_key_padding_mask=padding_mask,
        )
        cls_representation = temporal_output[:, 0]
        memory_representation, memory_attention = self.memory.read(
            cls_representation
        )
        with torch.no_grad():
            absolute = adjacency.abs()
            mean_absolute = absolute.mean(dim=(1, 2, 3))
            std_absolute = absolute.reshape(batch_size, -1).std(dim=1)
            mean_degree = absolute.sum(dim=(2, 3)).mean(dim=1)
            graph_statistics = torch.stack(
                (mean_absolute, std_absolute, mean_degree), dim=1
            )
        projected_statistics = self.graph_feature_projection(
            graph_statistics
        )
        final_representation = torch.cat(
            (
                cls_representation,
                memory_representation,
                projected_statistics,
            ),
            dim=1,
        )
        logits = self.classifier(
            self.classifier_norm(final_representation)
        ).squeeze(-1)
        time_mask = (
            torch.arange(maximum_time, device=adjacency.device).unsqueeze(0)
            < lengths.unsqueeze(1)
        )
        return AuthorShortTermOutput(
            logits=logits,
            final_representation=final_representation,
            cls_representation=cls_representation,
            memory_representation=memory_representation,
            memory_attention=memory_attention,
            graph_statistics=graph_statistics,
            window_embeddings=window_embeddings,
            time_mask=time_mask,
            diagnostics={
                "uses_coordinates": False,
                "node_feature_dim": self.window_encoder.node_feature_dim,
                "node_feature_order": (
                    "absolute_degree",
                    "community_embedding_32",
                    "delta_absolute_degree",
                ),
                "memory_explicit_write_enabled": False,
                "first_window_delta_absolute_degree": first_delta,
                "sequence_lengths": tuple(int(value) for value in lengths),
                "author_padding_semantics": True,
            },
        )

