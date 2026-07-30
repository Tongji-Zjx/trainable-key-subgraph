"""Complete coordinate-free, community-structured short-term branch."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from keysubgraph.data.graph_dataset import GraphSequenceBatch, GraphSequenceSample
from keysubgraph.features.structured_short_term_features import (
    NODE_FEATURE_NAMES,
    StructuredShortTermFeatureBuilder,
    StructuredShortTermStandardizer,
    StructuredWindowFeatures,
)


@dataclass(frozen=True)
class StructuredShortTermConfig:
    """Configuration for the revised complete short-term branch."""

    node_feature_dim: int = len(NODE_FEATURE_NAMES)
    hidden_dim: int = 64
    node_ffn_dim: int = 128
    transformer_layers: int = 2
    transformer_heads: int = 4
    transformer_ffn_dim: int = 128
    memory_slots: int = 8
    statistics_embedding_dim: int = 16
    classifier_hidden_dims: Tuple[int, int] = (64, 32)
    dropout: float = 0.10
    epsilon: float = 1.0e-8

    def __post_init__(self) -> None:
        dimensions = (
            self.node_feature_dim,
            self.hidden_dim,
            self.node_ffn_dim,
            self.transformer_layers,
            self.transformer_heads,
            self.transformer_ffn_dim,
            self.memory_slots,
            self.statistics_embedding_dim,
        ) + tuple(self.classifier_hidden_dims)
        if any(int(value) <= 0 for value in dimensions):
            raise ValueError("short-term dimensions must be positive")
        if self.node_feature_dim != len(NODE_FEATURE_NAMES):
            raise ValueError("short-term node feature schema is fixed")
        if self.hidden_dim % self.transformer_heads != 0:
            raise ValueError("hidden_dim must be divisible by transformer_heads")
        if len(self.classifier_hidden_dims) != 2:
            raise ValueError("short-term classifier requires two hidden layers")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0,1)")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["classifier_hidden_dims"] = list(self.classifier_hidden_dims)
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "StructuredShortTermConfig":
        values = dict(payload)
        values["classifier_hidden_dims"] = tuple(values["classifier_hidden_dims"])
        return cls(**values)


@dataclass(frozen=True)
class StructuredWindowEncoding:
    features: StructuredWindowFeatures
    normalized_node_features: torch.Tensor
    encoded_nodes: torch.Tensor
    window_embedding: torch.Tensor


@dataclass(frozen=True)
class StructuredShortTermOutput:
    logits: torch.Tensor
    cls_representation: torch.Tensor
    memory_representation: torch.Tensor
    memory_attention: torch.Tensor
    sequence_statistics: torch.Tensor
    statistics_representation: torch.Tensor
    final_representation: torch.Tensor
    window_embeddings: torch.Tensor
    time_mask: torch.Tensor
    window_encodings: Tuple[Tuple[StructuredWindowEncoding, ...], ...]
    diagnostics: Dict[str, Any]


class SinusoidalPositionEncoding(nn.Module):
    """Parameter-free sinusoidal positions with dynamic sequence length."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = int(dimension)

    def forward(self, length: int, reference: torch.Tensor) -> torch.Tensor:
        if length <= 0:
            raise ValueError("position sequence must be non-empty")
        position = torch.arange(
            length,
            device=reference.device,
            dtype=reference.dtype,
        ).unsqueeze(1)
        even_count = (self.dimension + 1) // 2
        divisor = torch.exp(
            torch.arange(
                even_count,
                device=reference.device,
                dtype=reference.dtype,
            )
            * (-math.log(10000.0) / float(self.dimension))
            * 2.0
        )
        angles = position * divisor.unsqueeze(0)
        output = reference.new_zeros((length, self.dimension))
        output[:, 0::2] = torch.sin(angles[:, : output[:, 0::2].shape[1]])
        if self.dimension > 1:
            output[:, 1::2] = torch.cos(
                angles[:, : output[:, 1::2].shape[1]]
            )
        return output


class StructuredWindowEncoder(nn.Module):
    """Paper-style node encoder using revised invariant node features."""

    def __init__(self, config: StructuredShortTermConfig) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(config.node_feature_dim)
        self.input_projection = nn.Linear(
            config.node_feature_dim,
            config.hidden_dim,
        )
        self.ffn_linear1 = nn.Linear(config.hidden_dim, config.node_ffn_dim)
        self.ffn_linear2 = nn.Linear(config.node_ffn_dim, config.hidden_dim)
        self.output_norm = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        features: StructuredWindowFeatures,
        normalized_node_features: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
    ) -> StructuredWindowEncoding:
        values = self.input_norm(normalized_node_features)
        projected = self.input_projection(values)
        residual = self.ffn_linear2(
            self.dropout(F.gelu(self.ffn_linear1(projected)))
        )
        encoded = self.output_norm(projected + self.dropout(residual))
        if node_mask is None:
            pooled = encoded.mean(dim=0)
        else:
            if tuple(node_mask.shape) != (encoded.shape[0],):
                raise ValueError("short-term node mask has invalid shape")
            valid = node_mask.to(device=encoded.device, dtype=encoded.dtype)
            if not bool(node_mask.to(dtype=torch.bool).any()):
                raise ValueError("short-term branch cannot pool empty nodes")
            pooled = (encoded * valid.unsqueeze(1)).sum(dim=0)
            pooled = pooled / valid.sum().clamp_min(1.0)
        return StructuredWindowEncoding(
            features=features,
            normalized_node_features=normalized_node_features,
            encoded_nodes=encoded,
            window_embedding=pooled,
        )


class PrototypeMemoryReadout(nn.Module):
    """Differentiable prototype memory updated only by optimizer gradients."""

    def __init__(self, hidden_dim: int, memory_slots: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.memory = nn.Parameter(torch.empty(memory_slots, hidden_dim))
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        nn.init.normal_(self.memory, mean=0.0, std=1.0 / math.sqrt(hidden_dim))

    def forward(
        self,
        representation: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query = self.query(representation)
        scores = query.matmul(self.memory.transpose(0, 1))
        scores = scores / math.sqrt(float(self.hidden_dim))
        attention = torch.softmax(scores, dim=-1)
        readout = attention.matmul(self.memory)
        gate = torch.sigmoid(
            self.gate(torch.cat((representation, readout), dim=-1))
        )
        fused = gate * readout + (1.0 - gate) * representation
        return self.output_norm(fused), attention


class StructuredShortTermClassifier(nn.Module):
    """Full revised short-term network.

    Graph windows -> structured node features -> window encoder -> CLS
    Transformer -> prototype memory -> sequence statistics -> classifier.
    """

    model_name = "coordinate_free_community_structured_short_term"

    def __init__(
        self,
        config: StructuredShortTermConfig,
        standardizer: StructuredShortTermStandardizer,
    ) -> None:
        super().__init__()
        self.config = config
        self.standardizer = standardizer
        self.feature_builder = StructuredShortTermFeatureBuilder(
            edge_presence_threshold=standardizer.edge_presence_threshold,
            epsilon=config.epsilon,
        )
        self.window_encoder = StructuredWindowEncoder(config)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        self.position_encoding = SinusoidalPositionEncoding(config.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.transformer_heads,
            dim_feedforward=config.transformer_ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.transformer_layers,
            norm=nn.LayerNorm(config.hidden_dim),
        )
        self.memory_readout = PrototypeMemoryReadout(
            config.hidden_dim,
            config.memory_slots,
        )
        self.statistics_norm = nn.LayerNorm(6)
        self.statistics_projection = nn.Linear(
            6,
            config.statistics_embedding_dim,
        )
        first, second = config.classifier_hidden_dims
        classifier_input = (
            config.hidden_dim * 2 + config.statistics_embedding_dim
        )
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, first),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(first, second),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(second, 2),
        )

    @staticmethod
    def _sequence_statistics(
        degrees: torch.Tensor,
        anomalies: torch.Tensor,
        window_count: int,
    ) -> torch.Tensor:
        degree_std = degrees.std(unbiased=False)
        anomaly_std = anomalies.std(unbiased=False)
        return torch.stack(
            (
                degrees.mean(),
                degree_std,
                degrees.max(),
                anomalies.mean(),
                anomaly_std,
                degrees.new_tensor(math.log1p(float(window_count))),
            )
        )

    def _encode_sample(
        self,
        sample: GraphSequenceSample,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Tuple[StructuredWindowEncoding, ...],
    ]:
        windows = self.feature_builder.build_sample(sample)
        encodings: List[StructuredWindowEncoding] = []
        anomalies: List[torch.Tensor] = []
        degrees: List[torch.Tensor] = []
        for features in windows:
            normalized = self.standardizer.normalize_nodes(
                features.node_features
            )
            encodings.append(self.window_encoder(features, normalized))
            anomalies.append(
                self.standardizer.community_anomaly(
                    features.community_summary
                )
            )
            degrees.append(features.mean_absolute_degree)
        embeddings = torch.stack(
            tuple(encoding.window_embedding for encoding in encodings),
            dim=0,
        )
        statistics = self._sequence_statistics(
            torch.stack(tuple(degrees)),
            torch.stack(tuple(anomalies)),
            len(windows),
        )
        return embeddings, statistics, tuple(encodings)

    def forward(self, batch: GraphSequenceBatch) -> StructuredShortTermOutput:
        if len(batch) <= 0:
            raise ValueError("short-term batch is empty")
        sample_embeddings: List[torch.Tensor] = []
        sample_statistics: List[torch.Tensor] = []
        all_encodings: List[Tuple[StructuredWindowEncoding, ...]] = []
        lengths: List[int] = []
        for sample in batch.samples:
            embeddings, statistics, encodings = self._encode_sample(sample)
            sample_embeddings.append(embeddings)
            sample_statistics.append(statistics)
            all_encodings.append(encodings)
            lengths.append(int(embeddings.shape[0]))

        padded = pad_sequence(
            sample_embeddings,
            batch_first=True,
            padding_value=0.0,
        )
        batch_size, maximum_windows, _ = padded.shape
        length_tensor = torch.tensor(lengths, device=padded.device)
        time_mask = (
            torch.arange(maximum_windows, device=padded.device).unsqueeze(0)
            < length_tensor.unsqueeze(1)
        )
        cls = self.cls_token.expand(batch_size, -1, -1)
        temporal_input = torch.cat((cls, padded), dim=1)
        temporal_input = temporal_input + self.position_encoding(
            maximum_windows + 1,
            temporal_input,
        ).unsqueeze(0)
        padding_mask = torch.cat(
            (
                torch.zeros(
                    (batch_size, 1),
                    dtype=torch.bool,
                    device=padded.device,
                ),
                ~time_mask,
            ),
            dim=1,
        )
        temporal = self.temporal_encoder(
            temporal_input,
            src_key_padding_mask=padding_mask,
        )
        cls_representation = temporal[:, 0]
        memory_representation, memory_attention = self.memory_readout(
            cls_representation
        )
        sequence_statistics = torch.stack(tuple(sample_statistics), dim=0)
        statistics_representation = F.gelu(
            self.statistics_projection(
                self.statistics_norm(sequence_statistics)
            )
        )
        final_representation = torch.cat(
            (
                cls_representation,
                memory_representation,
                statistics_representation,
            ),
            dim=-1,
        )
        logits = self.classifier(final_representation)
        return StructuredShortTermOutput(
            logits=logits,
            cls_representation=cls_representation,
            memory_representation=memory_representation,
            memory_attention=memory_attention,
            sequence_statistics=sequence_statistics,
            statistics_representation=statistics_representation,
            final_representation=final_representation,
            window_embeddings=padded,
            time_mask=time_mask,
            window_encodings=tuple(all_encodings),
            diagnostics={
                "sequence_lengths": tuple(lengths),
                "node_feature_names": NODE_FEATURE_NAMES,
                "uses_coordinates": False,
                "uses_community_embedding": False,
                "memory_update": "optimizer_gradient_only",
            },
        )

