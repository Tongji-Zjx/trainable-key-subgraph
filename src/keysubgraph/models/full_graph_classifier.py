"""Controlled full-graph classifiers for the Stage-A encoder comparison."""

from __future__ import absolute_import, division, print_function

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from keysubgraph.data.graph_dataset import GraphSequenceBatch
from keysubgraph.features.graph_features import GraphFeatureBuilder

from .masked_pooling import MaskedGraphPooling
from .masked_tcn import MaskedTCNEncoder
from .signed_graph_encoder import SignedGraphEncoder


FULL_GRAPH_ENCODERS = ("signed_gnn_tcn", "sgg_bigru_proto")


@dataclass(frozen=True)
class FullGraphClassifierConfig:
    encoder_type: str = "signed_gnn_tcn"
    node_feature_dim: int = 13
    edge_feature_dim: int = 4
    node_hidden_dim: int = 64
    edge_hidden_dim: int = 32
    graph_embedding_dim: int = 96
    signed_gnn_layers: int = 3
    temporal_hidden_dim: int = 96
    temporal_kernel_size: int = 3
    temporal_dilations: Tuple[int, ...] = (1, 2, 4)
    bigru_hidden_per_direction: int = 48
    num_prototypes: int = 16
    prototype_dim: int = 64
    classifier_hidden_dims: Tuple[int, ...] = (128, 64)
    baseline_dropout: float = 0.2
    gated_gnn_dropout: float = 0.15
    classifier_dropout: float = 0.2
    epsilon: float = 1.0e-8

    def __post_init__(self) -> None:
        if self.encoder_type not in FULL_GRAPH_ENCODERS:
            raise ValueError("unsupported full-graph encoder type")
        dimensions = (
            self.node_feature_dim,
            self.edge_feature_dim,
            self.node_hidden_dim,
            self.edge_hidden_dim,
            self.graph_embedding_dim,
            self.signed_gnn_layers,
            self.temporal_hidden_dim,
            self.bigru_hidden_per_direction,
            self.num_prototypes,
            self.prototype_dim,
        ) + tuple(self.classifier_hidden_dims)
        if any(value < 1 for value in dimensions):
            raise ValueError("full-graph classifier dimensions must be positive")
        if self.node_feature_dim != 13 or self.edge_feature_dim != 4:
            raise ValueError("full-graph classifiers require the verified 13-D/4-D schema")
        if self.graph_embedding_dim != 96:
            raise ValueError("full-graph classifiers require 96-D window embeddings")
        if self.temporal_hidden_dim != 96:
            raise ValueError("the controlled TCN baseline requires a 96-D hidden state")
        if self.bigru_hidden_per_direction * 2 != self.graph_embedding_dim:
            raise ValueError("BiGRU output must match the 96-D window representation")
        for value in (
            self.baseline_dropout,
            self.gated_gnn_dropout,
            self.classifier_dropout,
        ):
            if value < 0.0 or value >= 1.0:
                raise ValueError("dropout must lie in [0, 1)")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")


@dataclass(frozen=True)
class FullGraphEncoderOutput:
    representation: torch.Tensor
    sequence_lengths: torch.Tensor
    prototype_attention: Optional[torch.Tensor]


@dataclass(frozen=True)
class FullGraphClassifierOutput:
    logits: torch.Tensor
    representation: torch.Tensor
    sequence_lengths: torch.Tensor
    prototype_attention: Optional[torch.Tensor]


def _masked_adjacency(sample, time_index: int) -> torch.Tensor:
    adjacency = sample.adjacency[time_index]
    mask = sample.edge_mask[time_index].to(device=adjacency.device, dtype=torch.bool)
    if tuple(mask.shape) != tuple(adjacency.shape):
        raise ValueError("edge mask must match adjacency")
    mask = mask & mask.transpose(0, 1)
    mask = mask.clone()
    mask.fill_diagonal_(False)
    return adjacency * mask.to(adjacency.dtype)


class SignedGNNTCNFullGraphEncoder(nn.Module):
    """Exact full-adjacency bypass for the current Signed GNN + TCN stack."""

    output_dim = 192

    def __init__(self, config: FullGraphClassifierConfig) -> None:
        super().__init__()
        self.config = config
        self.feature_builder = GraphFeatureBuilder(config.epsilon)
        self.graph_encoder = SignedGraphEncoder(
            input_dim=config.node_feature_dim,
            hidden_dim=config.node_hidden_dim,
            num_layers=config.signed_gnn_layers,
            dropout=config.baseline_dropout,
            epsilon=config.epsilon,
        )
        self.graph_pooling = MaskedGraphPooling(
            input_dim=config.node_hidden_dim,
            output_dim=config.graph_embedding_dim,
            dropout=config.baseline_dropout,
            epsilon=config.epsilon,
        )
        self.temporal_encoder = MaskedTCNEncoder(
            input_dim=config.graph_embedding_dim,
            hidden_dim=config.temporal_hidden_dim,
            kernel_size=config.temporal_kernel_size,
            dilations=config.temporal_dilations,
            dropout=config.baseline_dropout,
            epsilon=config.epsilon,
        )

    def forward(self, batch: GraphSequenceBatch) -> FullGraphEncoderOutput:
        if len(batch) == 0:
            raise ValueError("full-graph encoder cannot process an empty batch")
        sequences = []
        lengths = []
        for sample in batch:
            windows = []
            for time_index in range(sample.num_timepoints):
                features = self.feature_builder.build_timepoint(sample, time_index)
                hidden = self.graph_encoder(
                    features.node_features,
                    _masked_adjacency(sample, time_index),
                )
                windows.append(self.graph_pooling(hidden))
            sequences.append(torch.stack(windows, dim=0))
            lengths.append(sample.num_timepoints)
        representation, _, _ = self.temporal_encoder.forward_list(sequences)
        return FullGraphEncoderOutput(
            representation=representation,
            sequence_lengths=torch.tensor(
                lengths, dtype=torch.long, device=representation.device
            ),
            prototype_attention=None,
        )


class SymmetricSignedEdgeGatedLayer(nn.Module):
    """Residual signed message passing with an undirected edge-feature gate."""

    def __init__(
        self,
        hidden_dim: int,
        edge_hidden_dim: int,
        dropout: float,
        epsilon: float,
    ) -> None:
        super().__init__()
        self.epsilon = float(epsilon)
        gate_input_dim = 2 * int(hidden_dim) + int(edge_hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.positive_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.negative_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.self_projection = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.normalization = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        hidden: torch.Tensor,
        adjacency: torch.Tensor,
        edge_mask: torch.Tensor,
        edge_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        node_count, hidden_dim = hidden.shape
        if tuple(adjacency.shape) != (node_count, node_count):
            raise ValueError("adjacency must match node hidden states")
        if tuple(edge_mask.shape) != tuple(adjacency.shape):
            raise ValueError("edge mask must match adjacency")
        if tuple(edge_embeddings.shape[:2]) != tuple(adjacency.shape):
            raise ValueError("edge embeddings must match adjacency")

        valid = edge_mask.to(device=hidden.device, dtype=torch.bool)
        valid = valid & valid.transpose(0, 1)
        valid = valid.clone()
        valid.fill_diagonal_(False)
        upper_indices = torch.nonzero(torch.triu(valid, diagonal=1), as_tuple=False)
        gates = hidden.new_zeros((node_count, node_count))
        if upper_indices.numel() > 0:
            source = upper_indices[:, 0]
            target = upper_indices[:, 1]
            gate_input = torch.cat(
                (
                    hidden.index_select(0, source) + hidden.index_select(0, target),
                    (
                        hidden.index_select(0, source)
                        - hidden.index_select(0, target)
                    ).abs(),
                    edge_embeddings[source, target],
                ),
                dim=-1,
            )
            values = torch.sigmoid(self.gate(gate_input).squeeze(-1))
            gates[source, target] = values
            gates[target, source] = values

        signed = adjacency * valid.to(adjacency.dtype)
        positive_weights = signed.clamp_min(0.0) * gates
        negative_weights = (-signed.clamp_max(0.0)) * gates
        positive_denominator = positive_weights.sum(dim=-1, keepdim=True)
        negative_denominator = negative_weights.sum(dim=-1, keepdim=True)
        positive_message = positive_weights.matmul(
            self.positive_projection(hidden)
        ) / positive_denominator.clamp_min(self.epsilon)
        negative_message = negative_weights.matmul(
            self.negative_projection(hidden)
        ) / negative_denominator.clamp_min(self.epsilon)
        candidate = torch.nn.functional.gelu(
            self.self_projection(hidden) + positive_message + negative_message
        )
        output = self.normalization(hidden + self.dropout(candidate))
        return output, gates


class SignedEdgeGatedGraphEncoder(nn.Module):
    def __init__(self, config: FullGraphClassifierConfig) -> None:
        super().__init__()
        self.node_projection = nn.Sequential(
            nn.Linear(config.node_feature_dim, config.node_hidden_dim),
            nn.GELU(),
        )
        self.edge_projection = nn.Sequential(
            nn.Linear(config.edge_feature_dim, config.edge_hidden_dim),
            nn.GELU(),
        )
        self.layers = nn.ModuleList(
            [
                SymmetricSignedEdgeGatedLayer(
                    config.node_hidden_dim,
                    config.edge_hidden_dim,
                    config.gated_gnn_dropout,
                    config.epsilon,
                )
                for _ in range(config.signed_gnn_layers)
            ]
        )
        self.output_dim = int(config.node_hidden_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        adjacency: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if node_features.ndim != 2:
            raise ValueError("node features must have shape [N, F]")
        if edge_features.ndim != 3:
            raise ValueError("edge features must have shape [N, N, F]")
        hidden = self.node_projection(node_features)
        edge_embeddings = self.edge_projection(edge_features)
        gates = []
        for layer in self.layers:
            hidden, layer_gates = layer(
                hidden, adjacency, edge_mask, edge_embeddings
            )
            gates.append(layer_gates)
        return hidden, tuple(gates)


class PackedBiGRUEncoder(nn.Module):
    output_dim = 192

    def __init__(self, input_dim: int, hidden_per_direction: int) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_per_direction,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )

    def forward(
        self, sequences: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not sequences:
            raise ValueError("BiGRU requires at least one sequence")
        lengths = torch.tensor(
            [int(item.shape[0]) for item in sequences],
            dtype=torch.long,
            device=sequences[0].device,
        )
        if bool((lengths < 1).any()):
            raise ValueError("BiGRU sequences cannot be empty")
        padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        packed = pack_padded_sequence(
            padded,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        encoded_packed, _ = self.gru(packed)
        encoded, _ = pad_packed_sequence(
            encoded_packed,
            batch_first=True,
            total_length=padded.shape[1],
        )
        positions = torch.arange(
            encoded.shape[1], device=encoded.device
        ).unsqueeze(0)
        mask = positions < lengths.unsqueeze(1)
        weights = mask.to(encoded.dtype)
        mean = (encoded * weights.unsqueeze(-1)).sum(dim=1)
        mean = mean / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        maximum = encoded.masked_fill(~mask.unsqueeze(-1), -torch.inf).max(dim=1).values
        return torch.cat((mean, maximum), dim=-1), lengths


class PrototypeCodebook(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_prototypes: int,
        prototype_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.query = nn.Linear(input_dim, prototype_dim)
        self.prototypes = nn.Parameter(torch.empty(num_prototypes, prototype_dim))
        nn.init.xavier_uniform_(self.prototypes)
        self.fusion = nn.Linear(input_dim + prototype_dim, output_dim)
        self.normalization = nn.LayerNorm(output_dim)
        self.scale = math.sqrt(float(prototype_dim))

    def forward(self, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        query = self.query(values)
        attention = torch.softmax(
            query.matmul(self.prototypes.transpose(0, 1)) / self.scale,
            dim=-1,
        )
        read = attention.matmul(self.prototypes)
        fused = self.normalization(self.fusion(torch.cat((values, read), dim=-1)))
        return fused, attention


class SignedGatedBiGRUPrototypeEncoder(nn.Module):
    output_dim = 192

    def __init__(self, config: FullGraphClassifierConfig) -> None:
        super().__init__()
        self.config = config
        self.feature_builder = GraphFeatureBuilder(config.epsilon)
        self.graph_encoder = SignedEdgeGatedGraphEncoder(config)
        self.graph_pooling = MaskedGraphPooling(
            input_dim=config.node_hidden_dim,
            output_dim=config.graph_embedding_dim,
            dropout=config.gated_gnn_dropout,
            epsilon=config.epsilon,
        )
        # The verified Scheme-A design normalizes every pooled window before
        # it enters the recurrent encoder.  Keep this separate from the
        # generic pooling module so the controlled baseline remains unchanged.
        self.graph_pooling_normalization = nn.LayerNorm(
            config.graph_embedding_dim
        )
        self.temporal_encoder = PackedBiGRUEncoder(
            config.graph_embedding_dim,
            config.bigru_hidden_per_direction,
        )
        self.prototype_codebook = PrototypeCodebook(
            input_dim=PackedBiGRUEncoder.output_dim,
            num_prototypes=config.num_prototypes,
            prototype_dim=config.prototype_dim,
            output_dim=self.output_dim,
        )

    def forward(self, batch: GraphSequenceBatch) -> FullGraphEncoderOutput:
        if len(batch) == 0:
            raise ValueError("full-graph encoder cannot process an empty batch")
        sequences = []
        for sample in batch:
            windows = []
            for time_index in range(sample.num_timepoints):
                features = self.feature_builder.build_timepoint(sample, time_index)
                adjacency = _masked_adjacency(sample, time_index)
                hidden, _ = self.graph_encoder(
                    features.node_features,
                    features.edge_features,
                    adjacency,
                    features.edge_mask,
                )
                pooled = self.graph_pooling(hidden)
                windows.append(self.graph_pooling_normalization(pooled))
            sequences.append(torch.stack(windows, dim=0))
        sequence_representation, lengths = self.temporal_encoder(tuple(sequences))
        representation, attention = self.prototype_codebook(sequence_representation)
        return FullGraphEncoderOutput(
            representation=representation,
            sequence_lengths=lengths,
            prototype_attention=attention,
        )


def _classification_head(
    input_dim: int,
    hidden_dims: Tuple[int, ...],
    dropout: float,
) -> nn.Module:
    modules = []
    current = int(input_dim)
    for hidden in hidden_dims:
        modules.extend(
            (
                nn.Linear(current, int(hidden)),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        )
        current = int(hidden)
    modules.append(nn.Linear(current, 2))
    return nn.Sequential(*modules)


class FullGraphSequenceClassifier(nn.Module):
    """Shared classifier head over either controlled full-graph encoder."""

    training_stage = "full_graph_encoder_comparison"

    def __init__(self, config: Optional[FullGraphClassifierConfig] = None) -> None:
        super().__init__()
        self.config = config or FullGraphClassifierConfig()
        if self.config.encoder_type == "signed_gnn_tcn":
            self.encoder = SignedGNNTCNFullGraphEncoder(self.config)
        else:
            self.encoder = SignedGatedBiGRUPrototypeEncoder(self.config)
        self.classifier = _classification_head(
            self.encoder.output_dim,
            self.config.classifier_hidden_dims,
            self.config.classifier_dropout,
        )

    def forward(self, batch: GraphSequenceBatch) -> FullGraphClassifierOutput:
        encoded = self.encoder(batch)
        logits = self.classifier(encoded.representation)
        return FullGraphClassifierOutput(
            logits=logits,
            representation=encoded.representation,
            sequence_lengths=encoded.sequence_lengths,
            prototype_attention=encoded.prototype_attention,
        )
