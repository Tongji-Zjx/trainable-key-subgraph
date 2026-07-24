"""Residual STSE scorers and hard selection for signed graph windows."""

from __future__ import absolute_import, division, print_function

import hashlib
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from .hard_stse_types import HardSelectionOutput, HardSTSEConfig


class ResidualFeatureScorer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.input_normalization = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.residual = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.output_normalization = nn.LayerNorm(hidden_dim)
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.input_projection(self.input_normalization(features))
        hidden = self.output_normalization(hidden + self.residual(hidden))
        probabilities = torch.sigmoid(self.score(hidden).squeeze(-1))
        return hidden, probabilities


@dataclass(frozen=True)
class HardSTSEScoreOutput:
    node_hidden: torch.Tensor
    edge_hidden: torch.Tensor
    node_probabilities: torch.Tensor
    edge_probabilities: torch.Tensor


class HardSTSEScorer(nn.Module):
    """STSE-style node/edge scoring without message-passing layers."""

    def __init__(self, config: Optional[HardSTSEConfig] = None) -> None:
        super().__init__()
        self.config = config or HardSTSEConfig(
            variant="M2", selection_mode="learned", use_sgw=False
        )
        self.node_scorer = ResidualFeatureScorer(
            self.config.node_extractor_feature_dim,
            self.config.selector_node_hidden_dim,
            self.config.dropout,
        )
        edge_input_dim = (
            self.config.edge_extractor_base_dim
            + 2 * self.config.selector_node_hidden_dim
        )
        self.edge_scorer = ResidualFeatureScorer(
            edge_input_dim,
            self.config.selector_edge_hidden_dim,
            self.config.dropout,
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_base_features: torch.Tensor,
        edge_presence_mask: torch.Tensor,
    ) -> HardSTSEScoreOutput:
        if node_features.ndim != 2:
            raise ValueError("selector node features must have shape [N, F]")
        node_count = node_features.shape[0]
        if tuple(edge_base_features.shape[:2]) != (node_count, node_count):
            raise ValueError("selector edge features must align with nodes")
        if tuple(edge_presence_mask.shape) != (node_count, node_count):
            raise ValueError("selector edge mask must align with nodes")
        node_hidden, node_probabilities = self.node_scorer(node_features)
        left = node_hidden[:, None, :].expand(-1, node_count, -1)
        right = node_hidden[None, :, :].expand(node_count, -1, -1)
        edge_features = torch.cat(
            (
                edge_base_features,
                left + right,
                (left - right).abs(),
            ),
            dim=-1,
        )
        edge_hidden, edge_probabilities = self.edge_scorer(edge_features)
        valid = edge_presence_mask.to(device=node_features.device, dtype=torch.bool)
        valid = valid & valid.transpose(0, 1)
        valid = valid.clone()
        valid.fill_diagonal_(False)
        edge_probabilities = edge_probabilities * valid.to(
            dtype=edge_probabilities.dtype
        )
        edge_hidden = edge_hidden * valid[:, :, None].to(edge_hidden.dtype)
        return HardSTSEScoreOutput(
            node_hidden=node_hidden,
            edge_hidden=edge_hidden,
            node_probabilities=node_probabilities,
            edge_probabilities=edge_probabilities,
        )


def _stable_random_scores(
    count: int,
    sample_key: str,
    time_index: int,
    seed: int,
    stream: str,
    reference: torch.Tensor,
) -> torch.Tensor:
    material = "{}\0{}\0{}\0{}".format(
        int(seed), str(sample_key), int(time_index), str(stream)
    ).encode("utf-8")
    stable_seed = int.from_bytes(
        hashlib.sha256(material).digest()[:8], byteorder="big"
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(stable_seed)
    values = torch.rand(count, generator=generator, dtype=torch.float64)
    return values.to(device=reference.device, dtype=reference.dtype)


def _community_aware_nodes(
    scores: torch.Tensor,
    communities: torch.Tensor,
    target_count: int,
) -> torch.Tensor:
    count = int(scores.numel())
    if tuple(communities.shape) != (count,):
        raise ValueError("community labels must align with node scores")
    selected = []
    for community in torch.unique(communities, sorted=True):
        members = torch.nonzero(communities == community, as_tuple=False).flatten()
        best = members[scores.index_select(0, members).argmax()]
        selected.append(int(best))
    selected_set = set(selected)
    remaining = [
        index
        for index in torch.argsort(scores, descending=True).tolist()
        if index not in selected_set
    ]
    selected.extend(remaining[: max(0, target_count - len(selected))])
    mask = torch.zeros(count, dtype=torch.bool, device=scores.device)
    if selected:
        mask[torch.tensor(selected[:target_count], device=scores.device)] = True
    return mask


def select_hard_stse_window(
    node_probabilities: torch.Tensor,
    edge_probabilities: torch.Tensor,
    communities: torch.Tensor,
    edge_presence_mask: torch.Tensor,
    node_ratio: float,
    edge_ratio: float,
    node_minimum: int,
    edge_minimum: int,
    selection_mode: str,
    sample_key: str,
    time_index: int,
    random_seed: int = 42,
) -> HardSelectionOutput:
    """Select one community-covered hard union graph and attach STE masks."""

    if selection_mode not in ("full", "random", "learned"):
        raise ValueError("unsupported hard-selection mode")
    if node_ratio <= 0.0 or node_ratio > 1.0:
        raise ValueError("node ratio must lie in (0, 1]")
    if edge_ratio <= 0.0 or edge_ratio > 1.0:
        raise ValueError("edge ratio must lie in (0, 1]")
    node_count = int(node_probabilities.numel())
    if node_count < 1 or tuple(communities.shape) != (node_count,):
        raise ValueError("node probabilities and communities are invalid")
    if tuple(edge_probabilities.shape) != (node_count, node_count):
        raise ValueError("edge probabilities must be square")
    if tuple(edge_presence_mask.shape) != (node_count, node_count):
        raise ValueError("edge presence mask must be square")
    valid_edges = edge_presence_mask.to(
        device=node_probabilities.device, dtype=torch.bool
    )
    valid_edges = valid_edges & valid_edges.transpose(0, 1)
    valid_edges = valid_edges.clone()
    valid_edges.fill_diagonal_(False)

    community_count = int(torch.unique(communities).numel())
    if selection_mode == "full":
        requested_nodes = node_count
        node_scores = node_probabilities.new_ones(node_count)
        selected_nodes = torch.ones(
            node_count, dtype=torch.bool, device=node_probabilities.device
        )
    else:
        requested_nodes = min(
            node_count,
            max(
                int(node_minimum),
                int(math.ceil(float(node_ratio) * node_count)),
                community_count,
            ),
        )
        node_scores = (
            _stable_random_scores(
                node_count,
                sample_key,
                time_index,
                random_seed,
                "nodes",
                node_probabilities,
            )
            if selection_mode == "random"
            else node_probabilities
        )
        selected_nodes = _community_aware_nodes(
            node_scores, communities, requested_nodes
        )

    candidate = (
        valid_edges
        & selected_nodes[:, None]
        & selected_nodes[None, :]
    )
    candidate_upper = torch.triu(candidate, diagonal=1)
    edge_indices = torch.nonzero(candidate_upper, as_tuple=False)
    candidate_count = int(edge_indices.shape[0])
    if selection_mode == "full":
        requested_edges = candidate_count
    elif candidate_count:
        requested_edges = min(
            candidate_count,
            max(
                int(edge_minimum),
                int(math.ceil(float(edge_ratio) * candidate_count)),
            ),
        )
    else:
        requested_edges = 0

    hard_edge_mask = torch.zeros_like(valid_edges)
    if requested_edges:
        if selection_mode == "random":
            edge_scores = _stable_random_scores(
                candidate_count,
                sample_key,
                time_index,
                random_seed,
                "edges",
                node_probabilities,
            )
        elif selection_mode == "full":
            edge_scores = node_probabilities.new_ones(candidate_count)
        else:
            left = edge_indices[:, 0]
            right = edge_indices[:, 1]
            edge_scores = edge_probabilities[left, right] * torch.sqrt(
                (
                    node_probabilities.index_select(0, left)
                    * node_probabilities.index_select(0, right)
                ).clamp_min(0.0)
            )
        chosen = torch.topk(
            edge_scores, k=requested_edges, largest=True, sorted=False
        ).indices
        selected_edges = edge_indices.index_select(0, chosen)
        hard_edge_mask[selected_edges[:, 0], selected_edges[:, 1]] = True
        hard_edge_mask[selected_edges[:, 1], selected_edges[:, 0]] = True

    hard_node_mask = hard_edge_mask.any(dim=-1)
    hard_node_float = hard_node_mask.to(node_probabilities.dtype)
    hard_edge_float = hard_edge_mask.to(edge_probabilities.dtype)
    straight_node = node_probabilities + (
        hard_node_float - node_probabilities
    ).detach()
    straight_edge = edge_probabilities + (
        hard_edge_float - edge_probabilities
    ).detach()
    straight_edge = straight_edge * valid_edges.to(straight_edge.dtype)
    return HardSelectionOutput(
        node_probabilities=node_probabilities,
        edge_probabilities=edge_probabilities,
        hard_node_mask=hard_node_mask,
        hard_edge_mask=hard_edge_mask,
        candidate_node_mask=selected_nodes,
        straight_through_node_mask=straight_node,
        straight_through_edge_mask=straight_edge,
        requested_node_count=requested_nodes,
        original_edge_count=int(
            torch.triu(valid_edges, diagonal=1).sum()
        ),
        candidate_edge_count=candidate_count,
        requested_edge_count=requested_edges,
        actual_node_count=int(hard_node_mask.sum()),
        actual_edge_count=int(torch.triu(hard_edge_mask, diagonal=1).sum()),
        selection_mode=selection_mode,
    )
