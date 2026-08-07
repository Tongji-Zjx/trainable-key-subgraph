"""Theory-guided signed, multi-object, temporal selector scorer.

The module deliberately separates three concerns that the legacy scalar MLP
conflates: signed graph context, decomposition of one global soft graph into
K objects, and propagation of object state through time.  Coordinates and ROI
identities never enter importance scoring; when available, they are used only
as correspondence metadata for transporting and aligning object slots.
"""

from __future__ import absolute_import, division, print_function

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

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
    node_continuity: torch.Tensor
    edge_continuity: torch.Tensor
    pairwise_soft_iou: torch.Tensor


@dataclass(frozen=True)
class MultiObjectTemporalMemory:
    """ROI-aligned differentiable object memory carried between windows.

    Hard masks and seeds are filled by the outer hardening stage.  Keeping
    them in the same object makes the next window's soft scoring and hard
    hysteresis consume one consistent history rather than two unrelated
    post-hoc tracking states.
    """

    object_states: torch.Tensor
    node_probabilities: torch.Tensor
    edge_probabilities: torch.Tensor
    node_ids: Tuple[str, ...]
    signed_edge_values: Optional[torch.Tensor] = None
    object_representations: Optional[torch.Tensor] = None
    coordinate_centroids: Optional[torch.Tensor] = None
    spectral_descriptors: Optional[torch.Tensor] = None
    hard_node_masks: Optional[torch.Tensor] = None
    hard_edge_masks: Optional[torch.Tensor] = None
    seed_indices: Optional[torch.Tensor] = None


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
    next_memory: Optional[MultiObjectTemporalMemory]
    slot_alignment: torch.Tensor
    memory_update_gate: torch.Tensor
    transported_node_probabilities: Optional[torch.Tensor]
    transported_edge_probabilities: Optional[torch.Tensor]
    alignment_components: Dict[str, torch.Tensor]
    regularization: MultiObjectRegularization


def _soft_dice_matrix(
    left: torch.Tensor,
    right: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Pairwise soft-Dice similarity for two [K,N] object fields."""

    intersection = torch.einsum("in,jn->ij", left, right)
    denominator = (
        left.square().sum(dim=-1)[:, None]
        + right.square().sum(dim=-1)[None, :]
    )
    return (2.0 * intersection + float(epsilon)) / (
        denominator + float(epsilon)
    )


def _sinkhorn_assignment(
    similarity: torch.Tensor,
    temperature: float,
    iterations: int,
) -> torch.Tensor:
    """Return a differentiable doubly-stochastic previous-to-current map."""

    log_weights = similarity / float(temperature)
    for _ in range(int(iterations)):
        log_weights = log_weights - torch.logsumexp(
            log_weights, dim=1, keepdim=True
        )
        log_weights = log_weights - torch.logsumexp(
            log_weights, dim=0, keepdim=True
        )
    return log_weights.exp()


def _pairwise_cosine_similarity(
    left: torch.Tensor,
    right: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Pairwise cosine mapped to [0,1], with zero vectors treated neutrally."""

    numerator = left @ right.transpose(0, 1)
    denominator = (
        left.norm(dim=-1)[:, None] * right.norm(dim=-1)[None, :]
    )
    cosine = numerator / denominator.clamp_min(float(epsilon))
    similarity = 0.5 * (cosine.clamp(-1.0, 1.0) + 1.0)
    neutral = denominator <= float(epsilon)
    return torch.where(neutral, similarity.new_full((), 0.5), similarity)


def _object_weighted_mean(
    memberships: torch.Tensor,
    features: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    denominator = memberships.sum(dim=-1, keepdim=True).clamp_min(
        float(epsilon)
    )
    return memberships @ features / denominator


def _coordinate_similarity(
    previous: torch.Tensor,
    current: torch.Tensor,
    coordinates: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    finite = torch.isfinite(coordinates).all(dim=-1)
    nonzero = coordinates.abs().sum(dim=-1) > 0.0
    usable = coordinates[finite & nonzero]
    if usable.shape[0] < 2:
        return previous.new_ones((previous.shape[0], current.shape[0]))
    scale = (usable.max(dim=0).values - usable.min(dim=0).values).norm()
    scale = scale.clamp_min(float(epsilon))
    distance = torch.cdist(previous, current)
    return torch.exp(-distance / scale)


def composite_slot_similarity(
    node_similarity: torch.Tensor,
    signed_edge_similarity: Optional[torch.Tensor] = None,
    latent_similarity: Optional[torch.Tensor] = None,
    coordinate_similarity: Optional[torch.Tensor] = None,
    spectral_similarity: Optional[torch.Tensor] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Combine complementary correspondence evidence before Sinkhorn.

    Missing metadata simply removes that component and renormalizes the
    remaining weights.  This keeps coordinate-free datasets valid while ROI,
    coordinates and signed structure can strengthen correspondence when they
    are available.
    """

    configured = {
        "node": 0.40,
        "signed_edge": 0.25,
        "latent": 0.15,
        "coordinate": 0.10,
        "spectral": 0.10,
    }
    if weights is not None:
        configured.update({key: float(value) for key, value in weights.items()})
    components = {
        "node": node_similarity,
        "signed_edge": signed_edge_similarity,
        "latent": latent_similarity,
        "coordinate": coordinate_similarity,
        "spectral": spectral_similarity,
    }
    active = {
        key: value
        for key, value in components.items()
        if value is not None and configured[key] > 0.0
    }
    if not active:
        raise ValueError("slot alignment requires at least one component")
    shape = tuple(node_similarity.shape)
    if any(tuple(value.shape) != shape for value in active.values()):
        raise ValueError("slot-alignment component shapes do not agree")
    denominator = sum(configured[key] for key in active)
    combined = sum(
        configured[key] * value for key, value in active.items()
    ) / float(denominator)
    return combined, active


def _transport_temporal_memory(
    memory: MultiObjectTemporalMemory,
    current_node_ids: Sequence[str],
    adjacency: torch.Tensor,
    valid_edges: torch.Tensor,
    diffusion: float,
) -> Tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    torch.Tensor,
]:
    """Transport previous fields by stable ROI identity into current order."""

    current = tuple(str(value) for value in current_node_ids)
    if len(current) != len(set(current)):
        raise ValueError("current selector node identities must be unique")
    previous = tuple(str(value) for value in memory.node_ids)
    if len(previous) != len(set(previous)):
        raise ValueError("previous selector node identities must be unique")
    lookup = {value: index for index, value in enumerate(previous)}
    pairs = [
        (index, lookup[value])
        for index, value in enumerate(current)
        if value in lookup
    ]
    support = torch.zeros(
        len(current), dtype=torch.bool, device=adjacency.device
    )
    if not pairs:
        return None, None, None, support
    current_indices = torch.tensor(
        [item[0] for item in pairs], dtype=torch.long, device=adjacency.device
    )
    previous_indices = torch.tensor(
        [item[1] for item in pairs], dtype=torch.long, device=adjacency.device
    )
    object_count = int(memory.node_probabilities.shape[0])
    nodes = adjacency.new_zeros((object_count, len(current)))
    nodes[:, current_indices] = memory.node_probabilities.to(adjacency).index_select(
        1, previous_indices
    )
    edges = adjacency.new_zeros((object_count, len(current), len(current)))
    previous_edges = memory.edge_probabilities.to(adjacency)
    selected = previous_edges.index_select(1, previous_indices).index_select(
        2, previous_indices
    )
    edges[:, current_indices[:, None], current_indices[None, :]] = selected
    signed_edges = None
    if memory.signed_edge_values is not None:
        signed_edges = adjacency.new_zeros(
            (object_count, len(current), len(current))
        )
        previous_signed = memory.signed_edge_values.to(adjacency)
        selected_signed = previous_signed.index_select(
            1, previous_indices
        ).index_select(2, previous_indices)
        signed_edges[
            :, current_indices[:, None], current_indices[None, :]
        ] = selected_signed
    support[current_indices] = True

    if float(diffusion) > 0.0:
        weights = adjacency.abs() * valid_edges.to(adjacency.dtype)
        weights = weights + torch.eye(
            len(current), device=adjacency.device, dtype=adjacency.dtype
        )
        transition = weights / weights.sum(dim=-1, keepdim=True).clamp_min(
            1.0e-8
        )
        diffused = (transition @ nodes.transpose(0, 1)).transpose(0, 1)
        nodes = (1.0 - float(diffusion)) * nodes + float(diffusion) * diffused
    edges = edges * valid_edges[None, :, :].to(edges.dtype)
    if signed_edges is not None:
        signed_edges = signed_edges * valid_edges[None, :, :].to(
            signed_edges.dtype
        )
    return nodes, edges, signed_edges, support


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
        structural_memory_enabled: bool = False,
        memory_diffusion: float = 0.15,
        sinkhorn_temperature: float = 0.10,
        sinkhorn_iterations: int = 8,
        alignment_node_weight: float = 0.40,
        alignment_signed_edge_weight: float = 0.25,
        alignment_latent_weight: float = 0.15,
        alignment_coordinate_weight: float = 0.10,
        alignment_spectral_weight: float = 0.10,
        epsilon: float = 1.0e-8,
    ) -> None:
        super().__init__()
        if object_count < 2:
            raise ValueError("multi-object selector requires at least two objects")
        if not 0.0 <= overlap_minimum <= overlap_maximum <= 1.0:
            raise ValueError("soft overlap interval is invalid")
        if not 0.0 < target_object_ratio <= 1.0:
            raise ValueError("target object ratio must lie in (0,1]")
        if not 0.0 <= memory_diffusion <= 1.0:
            raise ValueError("memory diffusion must lie in [0,1]")
        if sinkhorn_temperature <= 0.0 or sinkhorn_iterations < 1:
            raise ValueError("Sinkhorn controls must be positive")
        alignment_weights = {
            "node": float(alignment_node_weight),
            "signed_edge": float(alignment_signed_edge_weight),
            "latent": float(alignment_latent_weight),
            "coordinate": float(alignment_coordinate_weight),
            "spectral": float(alignment_spectral_weight),
        }
        if any(value < 0.0 for value in alignment_weights.values()):
            raise ValueError("slot-alignment weights cannot be negative")
        if sum(alignment_weights.values()) <= 0.0:
            raise ValueError("at least one slot-alignment weight is required")
        self.object_count = int(object_count)
        self.hidden_dim = int(hidden_dim)
        self.overlap_minimum = float(overlap_minimum)
        self.overlap_maximum = float(overlap_maximum)
        self.target_object_ratio = float(target_object_ratio)
        self.temporal_confidence_threshold = float(temporal_confidence_threshold)
        self.structural_memory_enabled = bool(structural_memory_enabled)
        self.memory_diffusion = float(memory_diffusion)
        self.sinkhorn_temperature = float(sinkhorn_temperature)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        self.alignment_weights = alignment_weights
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
        if self.structural_memory_enabled:
            self.memory_gate = nn.Linear(2 * hidden_dim, 1)
            nn.init.zeros_(self.memory_gate.weight)
            nn.init.zeros_(self.memory_gate.bias)
        else:
            self.memory_gate = None

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
        previous_memory: Optional[MultiObjectTemporalMemory] = None,
        current_node_ids: Optional[Sequence[str]] = None,
        current_coordinates: Optional[torch.Tensor] = None,
    ) -> TheoryMultiObjectScoreOutput:
        count = int(node_features.shape[0])
        if tuple(edge_base_features.shape[:2]) != (count, count):
            raise ValueError("selector edge features must align with nodes")
        if current_coordinates is not None and tuple(
            current_coordinates.shape
        ) != (count, 3):
            raise ValueError("selector coordinates must have shape [N,3]")
        valid = self._valid_edges(edge_presence_mask, node_features)
        spectrum = (
            spectral_features
            if spectral_features is not None
            else self.encoder.spectral_features(adjacency, valid)
        )
        hidden = self.encoder(
            node_features,
            adjacency,
            valid,
            spectral_features=spectrum,
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

        if previous_memory is not None and not self.structural_memory_enabled:
            raise ValueError("structural memory was supplied to a legacy scorer")
        memory_states = (
            previous_memory.object_states
            if previous_memory is not None
            else previous_object_states
        )
        prior = (
            self.initial_states(hidden)
            if memory_states is None
            else memory_states.to(hidden)
        )
        if tuple(prior.shape) != (self.object_count, self.hidden_dim):
            raise ValueError("previous object states have an invalid shape")
        query = self.state_normalization(prior + self.object_queries.to(prior))
        logits = self.query_projection(query) @ self.node_projection(hidden).transpose(0, 1)
        logits = logits / float(self.hidden_dim) ** 0.5
        global_node_logits = torch.logit(global_nodes.clamp(1.0e-6, 1.0 - 1.0e-6))
        raw_object_nodes = torch.sigmoid(
            logits + global_node_logits[None, :]
        )

        object_edge_input = torch.cat(
            (
                edge_hidden[None, :, :, :].expand(self.object_count, -1, -1, -1),
                query[:, None, None, :].expand(-1, count, count, -1),
            ),
            dim=-1,
        )
        raw_object_edges = torch.sigmoid(
            self.object_edge_head(object_edge_input).squeeze(-1)
        )
        raw_object_edges = raw_object_edges * global_edges[None, :, :]
        raw_object_edges = 0.5 * (
            raw_object_edges + raw_object_edges.transpose(1, 2)
        )
        raw_object_edges = raw_object_edges * valid[None, :, :].to(
            raw_object_edges.dtype
        )
        raw_denominator = raw_object_nodes.sum(
            dim=-1, keepdim=True
        ).clamp_min(self.epsilon)
        raw_representations = raw_object_nodes @ hidden / raw_denominator
        raw_spectral = _object_weighted_mean(
            raw_object_nodes, spectrum.to(hidden), self.epsilon
        )
        coordinates = (
            torch.nan_to_num(current_coordinates.to(hidden), nan=0.0)
            if current_coordinates is not None
            else None
        )
        raw_centroids = (
            _object_weighted_mean(
                raw_object_nodes, coordinates, self.epsilon
            )
            if coordinates is not None
            else None
        )
        raw_signed_edges = raw_object_edges * adjacency[None, :, :]

        identity = torch.eye(
            self.object_count, device=hidden.device, dtype=hidden.dtype
        )
        slot_alignment = identity
        update_gate = hidden.new_ones((self.object_count,))
        transported_nodes = None
        transported_edges = None
        transported_signed_edges = None
        alignment_components: Dict[str, torch.Tensor] = {}
        support = torch.zeros(count, dtype=torch.bool, device=hidden.device)
        if previous_memory is not None:
            if current_node_ids is None or len(current_node_ids) != count:
                raise ValueError(
                    "structural memory requires one stable identity per node"
                )
            (
                transported_nodes,
                transported_edges,
                transported_signed_edges,
                support,
            ) = (
                _transport_temporal_memory(
                    previous_memory,
                    current_node_ids,
                    adjacency,
                    valid,
                    self.memory_diffusion,
                )
            )
        if transported_nodes is not None:
            node_similarity = _soft_dice_matrix(
                transported_nodes[:, support],
                raw_object_nodes[:, support],
                self.epsilon,
            )
            edge_support = valid & support[:, None] & support[None, :]
            signed_edge_similarity = None
            if (
                transported_signed_edges is not None
                and bool(edge_support.any())
            ):
                signed_edge_similarity = _pairwise_cosine_similarity(
                    transported_signed_edges[:, edge_support],
                    raw_signed_edges[:, edge_support],
                    self.epsilon,
                )
            latent_similarity = None
            if previous_memory.object_representations is not None:
                latent_similarity = _pairwise_cosine_similarity(
                    previous_memory.object_representations.to(hidden),
                    raw_representations,
                    self.epsilon,
                )
            coordinate_component = None
            if (
                raw_centroids is not None
                and previous_memory.coordinate_centroids is not None
            ):
                coordinate_component = _coordinate_similarity(
                    previous_memory.coordinate_centroids.to(hidden),
                    raw_centroids,
                    coordinates,
                    self.epsilon,
                )
            spectral_similarity = None
            if previous_memory.spectral_descriptors is not None:
                spectral_similarity = _pairwise_cosine_similarity(
                    previous_memory.spectral_descriptors.to(hidden),
                    raw_spectral,
                    self.epsilon,
                )
            slot_similarity, alignment_components = composite_slot_similarity(
                node_similarity,
                signed_edge_similarity=signed_edge_similarity,
                latent_similarity=latent_similarity,
                coordinate_similarity=coordinate_component,
                spectral_similarity=spectral_similarity,
                weights=self.alignment_weights,
            )
            slot_alignment = _sinkhorn_assignment(
                slot_similarity,
                self.sinkhorn_temperature,
                self.sinkhorn_iterations,
            )
            aligned_nodes = slot_alignment @ raw_object_nodes
            aligned_edges = torch.einsum(
                "ij,jmn->imn", slot_alignment, raw_object_edges
            )
            aligned_denominator = aligned_nodes.sum(
                dim=-1, keepdim=True
            ).clamp_min(self.epsilon)
            provisional = aligned_nodes @ hidden / aligned_denominator
            update_gate = torch.sigmoid(
                self.memory_gate(torch.cat((prior, provisional), dim=-1))
            ).squeeze(-1)
            object_nodes = (
                update_gate[:, None] * aligned_nodes
                + (1.0 - update_gate[:, None]) * transported_nodes
            )
            object_edges = (
                update_gate[:, None, None] * aligned_edges
                + (1.0 - update_gate[:, None, None]) * transported_edges
            )
            object_edges = object_edges * valid[None, :, :].to(
                object_edges.dtype
            )
        else:
            object_nodes = raw_object_nodes
            object_edges = raw_object_edges

        denominator = object_nodes.sum(dim=-1, keepdim=True).clamp_min(self.epsilon)
        pooled = object_nodes @ hidden / denominator
        final_spectral = _object_weighted_mean(
            object_nodes, spectrum.to(hidden), self.epsilon
        )
        final_centroids = (
            _object_weighted_mean(object_nodes, coordinates, self.epsilon)
            if coordinates is not None
            else None
        )
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
        if memory_states is None:
            temporal = hidden.new_zeros(())
        else:
            confidence = object_nodes.max(dim=-1).values.detach()
            gate = (confidence >= self.temporal_confidence_threshold).to(hidden.dtype)
            distance = 1.0 - F.cosine_similarity(next_states, prior, dim=-1)
            temporal = (distance * gate).sum() / gate.sum().clamp_min(1.0)
        if transported_nodes is None:
            node_continuity = hidden.new_zeros(())
            edge_continuity = hidden.new_zeros(())
        else:
            confidence = slot_alignment.max(dim=-1).values.detach()
            confidence = (
                confidence >= self.temporal_confidence_threshold
            ).to(hidden.dtype)
            node_similarity = torch.diagonal(
                _soft_dice_matrix(
                    object_nodes[:, support],
                    transported_nodes[:, support],
                    self.epsilon,
                )
            )
            node_continuity = (
                (1.0 - node_similarity) * confidence
            ).sum() / confidence.sum().clamp_min(1.0)
            edge_support = valid & support[:, None] & support[None, :]
            if bool(edge_support.any()):
                current_edge = object_edges[:, edge_support]
                previous_edge = transported_edges[:, edge_support]
                numerator = 2.0 * (current_edge * previous_edge).sum(dim=-1)
                denominator = (
                    current_edge.square().sum(dim=-1)
                    + previous_edge.square().sum(dim=-1)
                )
                edge_similarity = (numerator + self.epsilon) / (
                    denominator + self.epsilon
                )
                edge_continuity = (
                    (1.0 - edge_similarity) * confidence
                ).sum() / confidence.sum().clamp_min(1.0)
            else:
                edge_continuity = hidden.new_zeros(())
        regularization = MultiObjectRegularization(
            overlap=overlap,
            reconstruction=reconstruction,
            coverage=coverage,
            temporal=temporal,
            node_continuity=node_continuity,
            edge_continuity=edge_continuity,
            pairwise_soft_iou=soft_iou,
        )
        next_memory = None
        if self.structural_memory_enabled:
            if current_node_ids is None or len(current_node_ids) != count:
                raise ValueError(
                    "structural memory requires current stable node identities"
                )
            next_memory = MultiObjectTemporalMemory(
                object_states=next_states,
                node_probabilities=object_nodes,
                edge_probabilities=object_edges,
                node_ids=tuple(str(value) for value in current_node_ids),
                signed_edge_values=object_edges * adjacency[None, :, :],
                object_representations=pooled,
                coordinate_centroids=final_centroids,
                spectral_descriptors=final_spectral,
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
            next_memory=next_memory,
            slot_alignment=slot_alignment,
            memory_update_gate=update_gate,
            transported_node_probabilities=transported_nodes,
            transported_edge_probabilities=transported_edges,
            alignment_components=alignment_components,
            regularization=regularization,
        )
