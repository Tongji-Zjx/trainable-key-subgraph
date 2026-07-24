"""Dual differentiable/cropped hard-graph views for Hard-STSE."""

from __future__ import absolute_import, division, print_function

from typing import Optional

import torch

from keysubgraph.data.graph_dataset import GraphSequenceSample
from keysubgraph.models.hard_stse_types import (
    HardSelectionOutput,
    HardWindowOutput,
)
from .hard_graph_features import HardGraphWindow


def build_hard_stse_window(
    sample: GraphSequenceSample,
    time_index: int,
    selection: HardSelectionOutput,
) -> HardWindowOutput:
    if time_index < 0 or time_index >= sample.num_timepoints:
        raise IndexError("time index is outside the graph sequence")
    adjacency = sample.adjacency[time_index]
    node_count = adjacency.shape[0]
    if tuple(selection.hard_node_mask.shape) != (node_count,):
        raise ValueError("hard node mask does not align with adjacency")
    if tuple(selection.hard_edge_mask.shape) != (node_count, node_count):
        raise ValueError("hard edge mask does not align with adjacency")
    node_st = selection.straight_through_node_mask.to(
        device=adjacency.device, dtype=adjacency.dtype
    )
    edge_st = selection.straight_through_edge_mask.to(
        device=adjacency.device, dtype=adjacency.dtype
    )
    adjacency_st = adjacency * node_st[:, None] * node_st[None, :] * edge_st
    adjacency_st = 0.5 * (adjacency_st + adjacency_st.transpose(0, 1))
    adjacency_st = adjacency_st.clone()
    adjacency_st.fill_diagonal_(0.0)

    hard_nodes = selection.hard_node_mask.to(
        device=adjacency.device, dtype=torch.bool
    )
    hard_edges = selection.hard_edge_mask.to(
        device=adjacency.device, dtype=torch.bool
    )
    expected_forward = adjacency * hard_edges.to(adjacency.dtype)
    expected_forward = expected_forward * (
        hard_nodes[:, None] & hard_nodes[None, :]
    ).to(adjacency.dtype)
    if not torch.allclose(
        adjacency_st.detach(), expected_forward, atol=1.0e-7, rtol=0.0
    ):
        raise RuntimeError("STE adjacency forward value is not the hard graph")

    valid = bool(selection.actual_node_count >= 2 and selection.actual_edge_count >= 1)
    cropped: Optional[HardGraphWindow] = None
    if valid:
        indices = torch.nonzero(hard_nodes, as_tuple=False).flatten()
        cropped_adjacency = expected_forward.index_select(0, indices).index_select(
            1, indices
        )
        cropped_communities = sample.communities[time_index].to(
            device=adjacency.device, dtype=torch.long
        ).index_select(0, indices)
        source_names = sample.node_names[time_index]
        names = tuple(source_names[int(index)] for index in indices.tolist())
        cropped = HardGraphWindow(
            adjacency=cropped_adjacency,
            communities=cropped_communities,
            node_names=names,
            node_ids=names,
            time_start=float(sample.window_starts[time_index]),
            edge_presence_threshold=float(sample.edge_presence_threshold),
            window_valid=True,
        )
    return HardWindowOutput(
        adjacency_st=adjacency_st,
        hard_node_mask=hard_nodes,
        hard_edge_mask=hard_edges,
        straight_through_node_mask=node_st,
        straight_through_edge_mask=edge_st,
        cropped_graph=cropped,
        window_valid=valid,
        selection=selection,
    )

