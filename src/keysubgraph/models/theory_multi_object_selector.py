"""Theory-guided signed, multi-object, temporal selector scorer.

The module deliberately separates three concerns that the legacy scalar MLP
conflates: signed graph context, decomposition of one global soft graph into
K objects, and propagation of object state through time.  Coordinates and ROI
identities are intentionally absent; they remain correspondence-only metadata.
"""

from __future__ import absolute_import, division, print_function

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


def _symmetric_normalize(adjacency: torch.Tensor, epsilon: float) -> torch.Tensor:
    degree = adjacency.sum(dim=-1).clamp_min(float(epsilon))
    inverse = degree.rsqrt()
    return inverse[:, None] * adjacency * inverse[None, :]


def signed_laplacian_eigenvectors(
    adjacency: torch.Tensor,
    dimension: int,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """Return sign-canonicalized low-frequency signed-Laplacian vectors."""

    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("signed spectral initialization requires a square graph")
    count = int(adjacency.shape[0])
    if count < 1 or dimension < 1:
        raise ValueError("signed spectral dimensions must be positive")
    symmetric = 0.5 * (adjacency + adjacency.transpose(0, 1))
    symmetric = symmetric.clone()
    symmetric.fill_diagonal_(0.0)
    degree = symmetric.abs().sum(dim=-1).clamp_min(float(epsilon))
    inverse = degree.rsqrt()
    normalized = inverse[:, None] * symmetric * inverse[None, :]
    laplacian = torch.eye(count, device=adjacency.device, dtype=adjacency.dtype)
    laplacian = laplacian - normalized
    # Graph input is data rather than a trainable parameter.  Detaching the
    # eigensystem avoids unstable eigenvector gradients while retaining all
    # gradients through the neural selector.
    _, vectors = torch.linalg.eigh(laplacian.detach())
    take = min(count, int(dimension))
    vectors = vectors[:, :take]
    if take:
        pivot = vectors.abs().argmax(dim=0)
        columns = torch.arange(take, device=vectors.device)
        signs = torch.sign(vectors[pivot, columns])
        signs = torch.where(signs == 0.0, torch.ones_like(signs), signs)
        vectors = vectors * signs[None, :]
    if take < int(dimension):
        vectors = F.pad(vectors, (0, int(dimension) - take))
    return vectors


class SignedSpectralGCNIILayer(nn.Module):
    """Positive/negative GCNII propagation with initial-feature connection."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.normalization = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        hidden: torch.Tensor,
        initial: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        positive_message = positive @ hidden
        negative_message = negative @ hidden
        update = self.fusion(
            torch.cat(
                (hidden, positive_message, negative_message, initial), dim=-1
            )
        )
        return self.normalization(hidden + update)


class SignedSpectralGCNIIEncoder(nn.Module):
    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int,
        spectral_dim: int,
        layer_count: int,
        dropout: float,
        epsilon: float,
    ) -> None:
        super().__init__()
        if layer_count < 1:
            raise ValueError("signed spectral GCNII requires at least one layer")
        self.spectral_dim = int(spectral_dim)
        self.epsilon = float(epsilon)
        self.input = nn.Sequential(
            nn.LayerNorm(node_feature_dim + spectral_dim),
            nn.Linear(node_feature_dim + spectral_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.layers = nn.ModuleList(
            SignedSpectralGCNIILayer(hidden_dim, dropout)
            for _ in range(int(layer_count))
        )

    def spectral_features(
        self,
        adjacency: torch.Tensor,
        edge_presence_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the fixed spectral positional encoding for one window."""

        count = int(adjacency.shape[0])
        if tuple(adjacency.shape) != (count, count):
            raise ValueError("selector adjacency must be square")
        valid = edge_presence_mask.to(
            device=adjacency.device, dtype=torch.bool
        )
        valid = valid & valid.transpose(0, 1)
        valid = valid.clone()
        valid.fill_diagonal_(False)
        signed = 0.5 * (adjacency + adjacency.transpose(0, 1))
        signed = signed * valid.to(signed.dtype)
        return signed_laplacian_eigenvectors(
            signed, self.spectral_dim, self.epsilon
        )

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        edge_presence_mask: torch.Tensor,
        spectral_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        count = int(node_features.shape[0])
        if tuple(adjacency.shape) != (count, count):
            raise ValueError("selector adjacency must align with node features")
        valid = edge_presence_mask.to(device=adjacency.device, dtype=torch.bool)
        valid = valid & valid.transpose(0, 1)
        valid = valid.clone()
        valid.fill_diagonal_(False)
        signed = 0.5 * (adjacency + adjacency.transpose(0, 1))
        signed = signed * valid.to(signed.dtype)
        positive = _symmetric_normalize(signed.clamp_min(0.0), self.epsilon)
        negative = _symmetric_normalize((-signed).clamp_min(0.0), self.epsilon)
        if spectral_features is None:
            spectral = signed_laplacian_eigenvectors(
                signed, self.spectral_dim, self.epsilon
            ).to(node_features)
        else:
            if tuple(spectral_features.shape) != (
                count,
                self.spectral_dim,
            ):
                raise ValueError(
                    "cached selector spectrum has an invalid shape"
                )
            spectral = spectral_features.to(node_features)
        initial = self.input(torch.cat((node_features, spectral), dim=-1))
        hidden = initial
        for layer in self.layers:
            hidden = layer(hidden, initial, positive, negative)
        return hidden


@dataclass(frozen=True)
class MultiObjectRegularization:
    overlap: torch.Tensor
    reconstruction: torch.Tensor
    coverage: torch.Tensor
    temporal: torch.Tensor
    pairwise_soft_iou: torch.Tensor


@dataclass(frozen=True)
class TheoryMultiObjectScoreOutput:
    node_hidden: torch.Tensor
    edge_hidden: torch.Tensor
    node_probabilities: torch.Tensor
    edge_probabilities: torch.Tensor
    object_node_probabilities: torch.Tensor
    object_edge_probabilities: torch.Tensor
    object_representations: torch.Tensor
    next_object_states: torch.Tensor
    regularization: MultiObjectRegularization


class TheoryGuidedMultiObjectScorer(nn.Module):
    """Signed graph encoder plus global and temporally conditioned object heads."""

    def __init__(
        self,
        node_feature_dim: int = 15,
        edge_feature_dim: int = 6,
        hidden_dim: int = 64,
        edge_hidden_dim: int = 32,
        object_count: int = 3,
        spectral_dim: int = 8,
        graph_layers: int = 2,
        dropout: float = 0.10,
        overlap_minimum: float = 0.05,
        overlap_maximum: float = 0.30,
        target_object_ratio: float = 0.10,
        temporal_confidence_threshold: float = 0.25,
        epsilon: float = 1.0e-8,
    ) -> None:
        super().__init__()
        if object_count < 2:
            raise ValueError("multi-object selector requires at least two objects")
        if not 0.0 <= overlap_minimum <= overlap_maximum <= 1.0:
            raise ValueError("soft overlap interval is invalid")
        if not 0.0 < target_object_ratio <= 1.0:
            raise ValueError("target object ratio must lie in (0,1]")
        self.object_count = int(object_count)
        self.hidden_dim = int(hidden_dim)
        self.overlap_minimum = float(overlap_minimum)
        self.overlap_maximum = float(overlap_maximum)
        self.target_object_ratio = float(target_object_ratio)
        self.temporal_confidence_threshold = float(temporal_confidence_threshold)
        self.epsilon = float(epsilon)
        self.encoder = SignedSpectralGCNIIEncoder(
            node_feature_dim,
            hidden_dim,
            spectral_dim,
            graph_layers,
            dropout,
            epsilon,
        )
        self.global_node_head = nn.Linear(hidden_dim, 1)
        global_edge_input = edge_feature_dim + 3 * hidden_dim
        self.global_edge_encoder = nn.Sequential(
            nn.LayerNorm(global_edge_input),
            nn.Linear(global_edge_input, edge_hidden_dim),
            nn.GELU(),
            nn.Linear(edge_hidden_dim, edge_hidden_dim),
            nn.GELU(),
        )
        self.global_edge_head = nn.Linear(edge_hidden_dim, 1)
        self.object_queries = nn.Parameter(torch.empty(object_count, hidden_dim))
        nn.init.orthogonal_(self.object_queries)
        self.query_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.node_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.object_edge_head = nn.Sequential(
            nn.Linear(edge_hidden_dim + hidden_dim, edge_hidden_dim),
            nn.GELU(),
            nn.Linear(edge_hidden_dim, 1),
        )
        self.temporal_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.state_normalization = nn.LayerNorm(hidden_dim)

    def initial_states(self, reference: torch.Tensor) -> torch.Tensor:
        return self.object_queries.to(reference)

    @staticmethod
    def _valid_edges(mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        valid = mask.to(device=reference.device, dtype=torch.bool)
        valid = valid & valid.transpose(0, 1)
        valid = valid.clone()
        valid.fill_diagonal_(False)
        return valid

    def forward(
        self,
        node_features: torch.Tensor,
        edge_base_features: torch.Tensor,
        edge_presence_mask: torch.Tensor,
        adjacency: torch.Tensor,
        previous_object_states: Optional[torch.Tensor] = None,
        spectral_features: Optional[torch.Tensor] = None,
    ) -> TheoryMultiObjectScoreOutput:
        count = int(node_features.shape[0])
        if tuple(edge_base_features.shape[:2]) != (count, count):
            raise ValueError("selector edge features must align with nodes")
        valid = self._valid_edges(edge_presence_mask, node_features)
        hidden = self.encoder(
            node_features,
            adjacency,
            valid,
            spectral_features=spectral_features,
        )
        global_nodes = torch.sigmoid(self.global_node_head(hidden).squeeze(-1))
        left = hidden[:, None, :].expand(-1, count, -1)
        right = hidden[None, :, :].expand(count, -1, -1)
        edge_input = torch.cat(
            (edge_base_features, left + right, (left - right).abs(), left * right),
            dim=-1,
        )
        edge_hidden = self.global_edge_encoder(edge_input)
        global_edges = torch.sigmoid(self.global_edge_head(edge_hidden).squeeze(-1))
        global_edges = global_edges * valid.to(global_edges.dtype)

        prior = (
            self.initial_states(hidden)
            if previous_object_states is None
            else previous_object_states.to(hidden)
        )
        if tuple(prior.shape) != (self.object_count, self.hidden_dim):
            raise ValueError("previous object states have an invalid shape")
        query = self.state_normalization(prior + self.object_queries.to(prior))
        logits = self.query_projection(query) @ self.node_projection(hidden).transpose(0, 1)
        logits = logits / float(self.hidden_dim) ** 0.5
        global_node_logits = torch.logit(global_nodes.clamp(1.0e-6, 1.0 - 1.0e-6))
        object_nodes = torch.sigmoid(logits + global_node_logits[None, :])

        object_edge_input = torch.cat(
            (
                edge_hidden[None, :, :, :].expand(self.object_count, -1, -1, -1),
                query[:, None, None, :].expand(-1, count, count, -1),
            ),
            dim=-1,
        )
        object_edges = torch.sigmoid(self.object_edge_head(object_edge_input).squeeze(-1))
        object_edges = object_edges * global_edges[None, :, :]
        object_edges = 0.5 * (object_edges + object_edges.transpose(1, 2))
        object_edges = object_edges * valid[None, :, :].to(object_edges.dtype)

        denominator = object_nodes.sum(dim=-1, keepdim=True).clamp_min(self.epsilon)
        pooled = object_nodes @ hidden / denominator
        next_states = self.temporal_cell(
            pooled.reshape(-1, self.hidden_dim), prior.reshape(-1, self.hidden_dim)
        ).reshape(self.object_count, self.hidden_dim)
        next_states = self.state_normalization(next_states)

        intersection = object_nodes[:, None, :] * object_nodes[None, :, :]
        union = (
            object_nodes[:, None, :]
            + object_nodes[None, :, :]
            - intersection
        )
        soft_iou = intersection.sum(dim=-1) / union.sum(dim=-1).clamp_min(self.epsilon)
        upper = torch.triu(
            torch.ones_like(soft_iou, dtype=torch.bool), diagonal=1
        )
        pair_values = soft_iou[upper]
        overlap = (
            F.relu(pair_values - self.overlap_maximum).square()
            + F.relu(self.overlap_minimum - pair_values).square()
        ).mean()
        reconstructed_nodes = 1.0 - torch.prod(1.0 - object_nodes, dim=0)
        reconstructed_edges = 1.0 - torch.prod(1.0 - object_edges, dim=0)
        reconstruction = F.l1_loss(reconstructed_nodes, global_nodes)
        if bool(valid.any()):
            reconstruction = reconstruction + F.l1_loss(
                reconstructed_edges[valid], global_edges[valid]
            )
        coverage = (object_nodes.mean(dim=-1) - self.target_object_ratio).abs().mean()
        if previous_object_states is None:
            temporal = hidden.new_zeros(())
        else:
            confidence = object_nodes.max(dim=-1).values.detach()
            gate = (confidence >= self.temporal_confidence_threshold).to(hidden.dtype)
            distance = 1.0 - F.cosine_similarity(next_states, prior, dim=-1)
            temporal = (distance * gate).sum() / gate.sum().clamp_min(1.0)
        regularization = MultiObjectRegularization(
            overlap=overlap,
            reconstruction=reconstruction,
            coverage=coverage,
            temporal=temporal,
            pairwise_soft_iou=soft_iou,
        )
        return TheoryMultiObjectScoreOutput(
            node_hidden=hidden,
            edge_hidden=edge_hidden,
            node_probabilities=global_nodes,
            edge_probabilities=global_edges,
            object_node_probabilities=object_nodes,
            object_edge_probabilities=object_edges,
            object_representations=pooled,
            next_object_states=next_states,
            regularization=regularization,
        )
