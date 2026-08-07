"""Feature contracts for the revised theory-guided multi-view model.

The implementation is deliberately list based.  A sample may have a variable
number of windows, nodes and critical objects.  Expensive spectral/GW work is
performed by this builder and can therefore be cached outside the training
loop.
"""

from __future__ import absolute_import, division, print_function

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import torch

from keysubgraph.features.hard_graph_cache import HardGraphSampleCache
from keysubgraph.features.hard_graph_features import HardGraphWindow
from keysubgraph.features.graph_features import align_current_to_previous
from keysubgraph.features.sv_hard_graph_features import (
    SVHardNodeFeatureBuilder,
    SVHardSampleFeatureBuilder,
)
from keysubgraph.features.theory_neural_features import TheoryNeuralFeatureBuilder
from keysubgraph.theory.sgw_core_features import SGWCoreConfig
from keysubgraph.models.dynamic_subgraph_tracking import (
    DynamicTrackingConfig,
    build_dynamic_trajectories_from_costs,
)


MULTIVIEW_SPECTRAL_FEATURE_DIM = 9
MULTIVIEW_Q_DIM = 16
MULTIVIEW_DELTA_Q_DIM = 18
MULTIVIEW_STABLE_STATIC_DIM = 28


@dataclass(frozen=True)
class CriticalObjectFeatures:
    node_features: torch.Tensor
    spectral_features: torch.Tensor
    adjacency: torch.Tensor
    edge_features: torch.Tensor
    communities: torch.Tensor
    union_node_indices: torch.Tensor
    mass: float
    roi_ids: Tuple[str, ...] = ()
    coordinates: Optional[torch.Tensor] = None
    coordinate_mask: Optional[torch.Tensor] = None

    def to(self, device):
        return CriticalObjectFeatures(
            self.node_features.to(device),
            self.spectral_features.to(device),
            self.adjacency.to(device),
            self.edge_features.to(device),
            self.communities.to(device),
            self.union_node_indices.to(device),
            float(self.mass),
            self.roi_ids,
            self.coordinates.to(device) if self.coordinates is not None else None,
            self.coordinate_mask.to(device) if self.coordinate_mask is not None else None,
        )


@dataclass(frozen=True)
class CriticalWindowFeatures:
    node_features: torch.Tensor
    spectral_features: torch.Tensor
    adjacency: torch.Tensor
    edge_features: torch.Tensor
    communities: torch.Tensor
    q_target: torch.Tensor
    objects: Tuple[CriticalObjectFeatures, ...]
    object_coupling: torch.Tensor
    time_start: float

    def to(self, device):
        return CriticalWindowFeatures(
            self.node_features.to(device),
            self.spectral_features.to(device),
            self.adjacency.to(device),
            self.edge_features.to(device),
            self.communities.to(device),
            self.q_target.to(device),
            tuple(item.to(device) for item in self.objects),
            self.object_coupling.to(device),
            float(self.time_start),
        )


@dataclass(frozen=True)
class CriticalTransitionFeatures:
    source_index: int
    target_index: int
    object_cost: torch.Tensor
    transport_plan: torch.Tensor
    delta_q_target: torch.Tensor
    delta_time: float
    solver_converged: bool

    def to(self, device):
        return CriticalTransitionFeatures(
            int(self.source_index),
            int(self.target_index),
            self.object_cost.to(device),
            self.transport_plan.to(device),
            self.delta_q_target.to(device),
            float(self.delta_time),
            bool(self.solver_converged),
        )


@dataclass(frozen=True)
class MultiViewCriticalSampleFeatures:
    sample_key: str
    label: int
    stable_static: torch.Tensor
    hard_windows: Tuple[Optional[CriticalWindowFeatures], ...]
    full_windows: Tuple[Optional[CriticalWindowFeatures], ...]
    transitions: Tuple[Optional[CriticalTransitionFeatures], ...]
    window_mask: torch.Tensor
    transition_mask: torch.Tensor
    # Derived by the train-only data adapter from the cached spectral states.
    # It is deliberately absent from the expensive immutable record schema so
    # the Stage-2 legacy-Variation control does not require cache regeneration.
    legacy_variation: Optional[torch.Tensor] = None
    trajectory_set: Optional[Any] = None

    def to(self, device):
        return MultiViewCriticalSampleFeatures(
            self.sample_key,
            int(self.label),
            self.stable_static.to(device),
            tuple(item.to(device) if item is not None else None for item in self.hard_windows),
            tuple(item.to(device) if item is not None else None for item in self.full_windows),
            tuple(item.to(device) if item is not None else None for item in self.transitions),
            self.window_mask.to(device),
            self.transition_mask.to(device),
            self.legacy_variation.to(device) if self.legacy_variation is not None else None,
            getattr(self, "trajectory_set", None),
        )


@dataclass(frozen=True)
class MultiViewCriticalBatch:
    samples: Tuple[MultiViewCriticalSampleFeatures, ...]

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    @property
    def labels(self):
        return torch.tensor([item.label for item in self.samples], dtype=torch.long)

    @property
    def sample_keys(self):
        return tuple(item.sample_key for item in self.samples)

    def to(self, device):
        return MultiViewCriticalBatch(tuple(item.to(device) for item in self.samples))


class SignedSpectralInvariantBuilder(object):
    """Nine node-wise signed spectral features without eigenvector identity."""

    output_dim = MULTIVIEW_SPECTRAL_FEATURE_DIM

    def __init__(
        self,
        heat_times=(0.1, 1.0, 10.0),
        projector_bands=3,
        chebyshev_order=2,
        epsilon=1.0e-8,
    ):
        if len(tuple(heat_times)) != 3 or int(projector_bands) != 3:
            raise ValueError("the verified spectral schema uses three heat times/bands")
        if int(chebyshev_order) != 2 or float(epsilon) <= 0.0:
            raise ValueError("the verified spectral schema uses Chebyshev order two")
        self.heat_times = tuple(float(value) for value in heat_times)
        self.projector_bands = int(projector_bands)
        self.chebyshev_order = int(chebyshev_order)
        self.epsilon = float(epsilon)

    def __call__(self, adjacency, edge_presence_threshold=0.0):
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("spectral adjacency must be square")
        if adjacency.shape[0] < 1 or not bool(torch.isfinite(adjacency).all()):
            raise ValueError("spectral adjacency must be non-empty and finite")
        adjacency = 0.5 * (adjacency + adjacency.transpose(0, 1))
        adjacency = adjacency.clone()
        adjacency.fill_diagonal_(0.0)
        edge_mask = adjacency.abs() > float(edge_presence_threshold)
        edge_mask.fill_diagonal_(False)
        adjacency = adjacency * edge_mask.to(adjacency.dtype)
        degree = adjacency.abs().sum(dim=-1)
        laplacian = torch.diag(degree) - adjacency
        eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)

        hks = []
        squared = eigenvectors.square()
        for time_value in self.heat_times:
            hks.append((squared * torch.exp(-time_value * eigenvalues)[None, :]).sum(dim=1))

        projector = []
        count = int(adjacency.shape[0])
        for band in range(self.projector_bands):
            left = (band * count) // self.projector_bands
            right = ((band + 1) * count) // self.projector_bands
            if right <= left:
                projector.append(adjacency.new_zeros((count,)))
            else:
                projector.append(squared[:, left:right].sum(dim=1))

        maximum = eigenvalues.max().clamp_min(self.epsilon)
        scaled = (2.0 / maximum) * laplacian - torch.eye(
            count, dtype=adjacency.dtype, device=adjacency.device
        )
        signal = degree / degree.mean().clamp_min(self.epsilon)
        t0 = signal
        t1 = scaled.matmul(t0)
        t2 = 2.0 * scaled.matmul(t1) - t0
        result = torch.stack(tuple(hks + projector + [t0, t1, t2]), dim=1)
        if tuple(result.shape) != (count, self.output_dim) or not bool(torch.isfinite(result).all()):
            raise RuntimeError("signed spectral invariant schema is invalid")
        return result


def _connected_components(edge_mask, indices):
    remaining = set(int(value) for value in indices.tolist())
    output = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack = [start]
        component = [start]
        while stack:
            current = stack.pop()
            neighbours = torch.nonzero(edge_mask[current], as_tuple=False).flatten().tolist()
            for neighbour in neighbours:
                neighbour = int(neighbour)
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    stack.append(neighbour)
                    component.append(neighbour)
        output.append(tuple(sorted(component)))
    return tuple(output)


def decompose_critical_objects(
    node_features,
    spectral_features,
    adjacency,
    edge_features,
    communities,
    edge_presence_threshold,
):
    """Community-induced decomposition followed by connected components."""

    count = int(adjacency.shape[0])
    if tuple(node_features.shape[:1]) != (count,) or tuple(spectral_features.shape[:1]) != (count,):
        raise ValueError("critical object node features do not align")
    if tuple(communities.shape) != (count,):
        raise ValueError("critical object communities do not align")
    edge_mask = adjacency.abs() > float(edge_presence_threshold)
    edge_mask = edge_mask.clone()
    edge_mask.fill_diagonal_(False)
    groups = []
    for label in torch.unique(communities, sorted=True).tolist():
        indices = torch.nonzero(communities == int(label), as_tuple=False).flatten()
        community_mask = edge_mask & (communities[:, None] == int(label)) & (
            communities[None, :] == int(label)
        )
        groups.extend(_connected_components(community_mask, indices))
    if not groups:
        raise ValueError("critical window produced no objects")

    objects = []
    for values in groups:
        indices = torch.tensor(values, dtype=torch.long, device=adjacency.device)
        local_adjacency = adjacency.index_select(0, indices).index_select(1, indices)
        objects.append(
            CriticalObjectFeatures(
                node_features=node_features.index_select(0, indices),
                spectral_features=spectral_features.index_select(0, indices),
                adjacency=local_adjacency,
                edge_features=edge_features.index_select(0, indices).index_select(1, indices),
                communities=communities.index_select(0, indices),
                union_node_indices=indices,
                mass=float(indices.numel()) / float(count),
            )
        )

    coupling = adjacency.new_zeros((len(objects), len(objects), 2))
    for left, first in enumerate(objects):
        for right, second in enumerate(objects):
            if left == right:
                continue
            values = adjacency.index_select(0, first.union_node_indices).index_select(
                1, second.union_node_indices
            )
            possible = float(max(1, values.numel()))
            coupling[left, right, 0] = values.clamp_min(0.0).sum() / possible
            coupling[left, right, 1] = (-values.clamp_max(0.0)).sum() / possible
    return tuple(objects), coupling


def unbalanced_sinkhorn(
    cost,
    source_mass,
    target_mass,
    entropic_reg=0.1,
    mass_reg=1.0,
    iterations=100,
    epsilon=1.0e-8,
):
    """Deterministic unbalanced entropic OT for object correspondence."""

    if cost.ndim != 2 or cost.shape[0] < 1 or cost.shape[1] < 1:
        raise ValueError("UOT cost must be a non-empty matrix")
    if tuple(source_mass.shape) != (cost.shape[0],) or tuple(target_mass.shape) != (cost.shape[1],):
        raise ValueError("UOT masses do not align with cost")
    if entropic_reg <= 0.0 or mass_reg <= 0.0 or iterations < 1:
        raise ValueError("UOT parameters must be positive")
    source = source_mass.clamp_min(epsilon)
    target = target_mass.clamp_min(epsilon)
    source = source / source.sum()
    target = target / target.sum()
    positive = cost[cost > 0.0]
    scale = positive.median() if positive.numel() else cost.new_ones(())
    kernel = torch.exp(-(cost / scale.clamp_min(epsilon)) / float(entropic_reg)).clamp_min(epsilon)
    power = float(mass_reg) / float(mass_reg + entropic_reg)
    u = torch.ones_like(source)
    v = torch.ones_like(target)
    for _ in range(int(iterations)):
        u = (source / kernel.matmul(v).clamp_min(epsilon)).pow(power)
        v = (target / kernel.transpose(0, 1).matmul(u).clamp_min(epsilon)).pow(power)
    plan = u[:, None] * kernel * v[None, :]
    if not bool(torch.isfinite(plan).all()) or float(plan.sum()) <= 0.0:
        raise RuntimeError("UOT solver returned an invalid plan")
    return plan


class MultiViewCriticalFeatureBuilder(object):
    """Build complete S/V inputs from a frozen hard-graph sample cache."""

    def __init__(
        self,
        core_config=None,
        spectral_builder=None,
        fgw_feature_weight=0.25,
        uot_entropic_reg=0.1,
        uot_mass_reg=1.0,
        uot_iterations=100,
        tracking_config=None,
    ):
        self.core_config = core_config or SGWCoreConfig()
        self.theory = TheoryNeuralFeatureBuilder(self.core_config)
        self.hard = SVHardSampleFeatureBuilder(
            laplacian_eta=self.core_config.laplacian_eta
        )
        self.node = SVHardNodeFeatureBuilder()
        self.spectral = spectral_builder or SignedSpectralInvariantBuilder()
        self.fgw = self.core_config.build_extractor()
        self.fgw_feature_weight = float(fgw_feature_weight)
        self.uot_entropic_reg = float(uot_entropic_reg)
        self.uot_mass_reg = float(uot_mass_reg)
        self.uot_iterations = int(uot_iterations)
        self.tracking_config = tracking_config or DynamicTrackingConfig()

    @staticmethod
    def _graphs(cache):
        return tuple(item.graph if item is not None else None for item in cache.windows)

    def _selected_objects(
        self,
        source,
        node_window,
        theory_window,
        selected_graphs,
    ):
        """Use selector-provided fixed-K objects without decomposing the union."""

        union_ids = tuple(
            str(value)
            for value in (
                source.node_ids
                if source.node_ids is not None
                else source.node_names
            )
        )
        union_local = {value: index for index, value in enumerate(union_ids)}
        objects = []
        for graph in selected_graphs:
            if graph is None or not graph.window_valid:
                continue
            object_ids = tuple(
                str(value)
                for value in (
                    graph.node_ids
                    if graph.node_ids is not None
                    else graph.node_names
                )
            )
            if any(value not in union_local for value in object_ids):
                raise ValueError("fixed-K object is absent from its hard union")
            indices = torch.tensor(
                [union_local[value] for value in object_ids],
                dtype=torch.long,
                device=node_window.adjacency.device,
            )
            edge_features = theory_window.edge_features.index_select(
                0, indices
            ).index_select(1, indices).clone()
            edge_features[:, :, 0] = graph.adjacency
            edge_features[:, :, 1] = graph.adjacency.abs()
            spectral = self.spectral(
                graph.adjacency, graph.edge_presence_threshold
            )
            coordinates = graph.coordinates
            if coordinates is None:
                coordinates = graph.adjacency.new_zeros(
                    (graph.adjacency.shape[0], 3)
                )
            coordinate_mask = torch.isfinite(coordinates).all(dim=-1) & (
                coordinates.abs().sum(dim=-1) > 0.0
            )
            objects.append(
                CriticalObjectFeatures(
                    node_features=node_window.node_features.index_select(0, indices),
                    spectral_features=spectral,
                    adjacency=graph.adjacency,
                    edge_features=edge_features,
                    communities=graph.communities,
                    union_node_indices=indices,
                    mass=float(indices.numel()) / float(max(1, len(union_ids))),
                    roi_ids=tuple(str(value) for value in graph.node_names),
                    coordinates=coordinates,
                    coordinate_mask=coordinate_mask,
                )
            )
        if not objects:
            raise ValueError("fixed-K window contains no valid objects")
        coupling = node_window.adjacency.new_zeros((len(objects), len(objects), 2))
        for left, first in enumerate(objects):
            for right, second in enumerate(objects):
                if left == right:
                    continue
                values = node_window.adjacency.index_select(
                    0, first.union_node_indices
                ).index_select(1, second.union_node_indices)
                possible = float(max(1, values.numel()))
                coupling[left, right, 0] = values.clamp_min(0.0).sum() / possible
                coupling[left, right, 1] = (-values.clamp_max(0.0)).sum() / possible
        return tuple(objects), coupling

    def _window(self, source, node_window, theory_window, selected_graphs=None):
        spectral = self.spectral(
            node_window.adjacency, source.edge_presence_threshold
        )
        if selected_graphs is None:
            objects, coupling = decompose_critical_objects(
                node_window.node_features,
                spectral,
                node_window.adjacency,
                theory_window.edge_features,
                node_window.communities,
                source.edge_presence_threshold,
            )
        else:
            objects, coupling = self._selected_objects(
                source, node_window, theory_window, selected_graphs
            )
        return CriticalWindowFeatures(
            node_features=node_window.node_features,
            spectral_features=spectral,
            adjacency=node_window.adjacency,
            edge_features=theory_window.edge_features,
            communities=node_window.communities,
            q_target=theory_window.spectral_quantiles,
            objects=objects,
            object_coupling=coupling,
            time_start=float(source.time_start),
        )

    def _object_state(self, item, threshold):
        count = int(item.adjacency.shape[0])
        window = HardGraphWindow(
            adjacency=item.adjacency,
            communities=item.communities,
            node_names=tuple(str(index) for index in range(count)),
            node_ids=tuple(str(index) for index in range(count)),
            time_start=0.0,
            edge_presence_threshold=float(threshold),
            window_valid=True,
        )
        return self.fgw.compute_window_state(window)

    @staticmethod
    def _object_attribute_cost(first, second, threshold, epsilon=1.0e-8):
        """Pairwise semantic cost used inside signed diffusion-FGW.

        The cost contains coordinate-free node/community structure, invariant
        spectral features, and explicit positive/negative degree and incidence
        profiles.  Pair-local symmetric standardization avoids fitting any
        cohort statistic before the train-only scaler exists.
        """

        def attributes(item):
            adjacency = item.adjacency
            mask = adjacency.abs() > float(threshold)
            mask = mask.clone()
            mask.fill_diagonal_(False)
            positive = adjacency.clamp_min(0.0) * mask.to(adjacency.dtype)
            negative = (-adjacency.clamp_max(0.0)) * mask.to(adjacency.dtype)
            possible = float(max(1, int(adjacency.shape[0]) - 1))
            signed_profile = torch.stack((
                positive.sum(dim=-1),
                negative.sum(dim=-1),
                (positive > 0.0).to(adjacency.dtype).sum(dim=-1) / possible,
                (negative > 0.0).to(adjacency.dtype).sum(dim=-1) / possible,
            ), dim=-1)
            return torch.cat(
                (item.node_features, item.spectral_features, signed_profile), dim=-1
            )

        first_attributes = attributes(first)
        second_attributes = attributes(second)
        pooled = torch.cat((first_attributes, second_attributes), dim=0)
        mean = pooled.mean(dim=0)
        scale = pooled.std(dim=0, unbiased=False).clamp_min(float(epsilon))
        first_normalized = (first_attributes - mean) / scale
        second_normalized = (second_attributes - mean) / scale
        feature_cost = torch.cdist(first_normalized, second_normalized).square()
        feature_cost = feature_cost / float(max(1, first_normalized.shape[1]))
        terms = [feature_cost]
        if (
            len(first.roi_ids) == first.adjacency.shape[0]
            and len(second.roi_ids) == second.adjacency.shape[0]
        ):
            roi_cost = feature_cost.new_tensor(
                [
                    [0.0 if str(left) == str(right) else 1.0 for right in second.roi_ids]
                    for left in first.roi_ids
                ]
            )
            terms.append(roi_cost)
        if (
            first.coordinates is not None
            and second.coordinates is not None
            and first.coordinate_mask is not None
            and second.coordinate_mask is not None
        ):
            first_coordinates = first.coordinates.to(feature_cost)
            second_coordinates = second.coordinates.to(feature_cost)
            coordinate_cost = torch.cdist(
                first_coordinates, second_coordinates
            ).square()
            valid = first.coordinate_mask.to(feature_cost.device)[:, None] & (
                second.coordinate_mask.to(feature_cost.device)[None, :]
            )
            positive = coordinate_cost[valid]
            scale = (
                positive.median().clamp_min(float(epsilon))
                if positive.numel()
                else coordinate_cost.new_ones(())
            )
            coordinate_cost = torch.where(
                valid,
                (coordinate_cost / scale).clamp_max(4.0),
                torch.zeros_like(coordinate_cost),
            )
            # Missing coordinates contribute neither evidence nor a penalty.
            if bool(valid.any()):
                terms.append(coordinate_cost)
        feature_cost = sum(terms) / float(len(terms))
        if not bool(torch.isfinite(feature_cost).all()):
            raise RuntimeError("object attribute cost is not finite")
        return feature_cost

    def _transition(self, index, left, right, target, threshold):
        left_states = tuple(self._object_state(item, threshold) for item in left.objects)
        right_states = tuple(self._object_state(item, threshold) for item in right.objects)
        structural_cost = left.adjacency.new_zeros((len(left_states), len(right_states)))
        roi_cost = structural_cost.clone()
        coordinate_cost = structural_cost.clone()
        converged = True
        with torch.no_grad():
            for row, first in enumerate(left_states):
                for column, second in enumerate(right_states):
                    feature_cost = self._object_attribute_cost(
                        left.objects[row], right.objects[column], threshold
                    )
                    gw = self.fgw.gw(
                        first.diffusion_distance,
                        second.diffusion_distance,
                        first.node_measure,
                        second.node_measure,
                        feature_cost=feature_cost,
                        feature_weight=self.fgw_feature_weight,
                    )
                    # The fused node cost already contains invariant multiscale
                    # spectral features.  A small object-level spectrum term
                    # retains global spectral differences that node coupling
                    # alone cannot express.
                    spectral = (first.spectral_quantiles - second.spectral_quantiles).abs().mean()
                    structural_cost[row, column] = (
                        gw.distance + self.fgw_feature_weight * spectral
                    )
                    first_roi = set(str(value) for value in left.objects[row].roi_ids)
                    second_roi = set(str(value) for value in right.objects[column].roi_ids)
                    union_roi = first_roi | second_roi
                    roi_cost[row, column] = 1.0 - float(
                        len(first_roi & second_roi)
                    ) / float(max(1, len(union_roi)))
                    first_object = left.objects[row]
                    second_object = right.objects[column]
                    first_valid = (
                        first_object.coordinates is not None
                        and first_object.coordinate_mask is not None
                        and bool(first_object.coordinate_mask.any())
                    )
                    second_valid = (
                        second_object.coordinates is not None
                        and second_object.coordinate_mask is not None
                        and bool(second_object.coordinate_mask.any())
                    )
                    if first_valid and second_valid:
                        first_centroid = first_object.coordinates[
                            first_object.coordinate_mask
                        ].mean(dim=0)
                        second_centroid = second_object.coordinates[
                            second_object.coordinate_mask
                        ].mean(dim=0)
                        magnitude = max(
                            1.0,
                            float(first_centroid.norm()),
                            float(second_centroid.norm()),
                        )
                        coordinate_cost[row, column] = min(
                            2.0,
                            float((first_centroid - second_centroid).norm()) / magnitude,
                        )
                    converged = converged and bool(gw.converged)
            positive = structural_cost[structural_cost > 0.0]
            scale = (
                positive.median().clamp_min(1.0e-8)
                if positive.numel()
                else structural_cost.new_ones(())
            )
            cost = (
                0.50 * (structural_cost / scale).clamp_max(4.0)
                + 0.35 * roi_cost
                + 0.15 * coordinate_cost
            )
            incompatible = (roi_cost >= 1.0) & (coordinate_cost >= 0.50)
            cost = cost + incompatible.to(cost.dtype)
            source_mass = cost.new_tensor([item.mass for item in left.objects])
            target_mass = cost.new_tensor([item.mass for item in right.objects])
            plan = unbalanced_sinkhorn(
                cost,
                source_mass,
                target_mass,
                self.uot_entropic_reg,
                self.uot_mass_reg,
                self.uot_iterations,
            )
        delta_time = float(right.time_start) - float(left.time_start)
        if delta_time <= 0.0:
            raise ValueError("critical transition time must increase")
        target_rate = target.clone()
        # The canonical 18-D transition schema stores raw signed spectral
        # quantile deltas in 0:16 and two already time-normalized speeds in
        # 16:18.  D_V predicts a uniform rate target, so normalize only the raw
        # delta block exactly once.
        target_rate[:MULTIVIEW_Q_DIM] = (
            target_rate[:MULTIVIEW_Q_DIM] / float(delta_time)
        )
        return CriticalTransitionFeatures(
            source_index=index,
            target_index=index + 1,
            object_cost=cost,
            transport_plan=plan,
            delta_q_target=target_rate,
            delta_time=delta_time,
            solver_converged=converged,
        )

    @staticmethod
    def _edge_features(full_graph_windows, node_windows):
        output = []
        previous_source, previous_adjacency = None, None
        for source, current in zip(full_graph_windows, node_windows):
            if source is None or current is None:
                output.append(None)
                previous_source, previous_adjacency = None, None
                continue
            adjacency = current.adjacency
            delta = torch.zeros_like(adjacency)
            delta_mask = torch.zeros_like(adjacency, dtype=torch.bool)
            if previous_source is not None:
                indices_cpu, present_cpu = align_current_to_previous(
                    tuple(str(value) for value in source.node_names),
                    tuple(str(value) for value in previous_source.node_names),
                )
                indices = indices_cpu.to(adjacency.device)
                present = present_cpu.to(adjacency.device)
                safe = indices.clamp_min(0)
                previous = previous_adjacency.index_select(0, safe).index_select(1, safe)
                delta_mask = present[:, None] & present[None, :]
                delta_mask.fill_diagonal_(False)
                delta = torch.where(delta_mask, adjacency - previous, torch.zeros_like(adjacency))
            same = current.communities[:, None] == current.communities[None, :]
            output.append(torch.stack((
                adjacency, adjacency.abs(), delta, delta.abs(),
                delta_mask.to(adjacency.dtype), same.to(adjacency.dtype),
            ), dim=-1))
            previous_source, previous_adjacency = source, adjacency
        return tuple(output)

    def _full_windows(self, full_graph_windows, q_targets):
        if full_graph_windows is None:
            return ()
        node_windows = self.node.build_sequence(full_graph_windows)
        edge_windows = self._edge_features(full_graph_windows, node_windows)
        output = []
        for index, (source, node_window, edge_features) in enumerate(
            zip(full_graph_windows, node_windows, edge_windows)
        ):
            if source is None or node_window is None:
                output.append(None)
                continue
            spectral = self.spectral(node_window.adjacency, source.edge_presence_threshold)
            output.append(
                CriticalWindowFeatures(
                    node_features=node_window.node_features,
                    spectral_features=spectral,
                    adjacency=node_window.adjacency,
                    edge_features=edge_features,
                    communities=node_window.communities,
                    q_target=q_targets[index],
                    objects=(),
                    object_coupling=node_window.adjacency.new_zeros((0, 0, 2)),
                    time_start=float(source.time_start),
                )
            )
        return tuple(output)

    def build(
        self,
        cache,
        full_graph_windows=None,
        selected_object_windows=None,
        trajectory_set=None,
    ):
        if not isinstance(cache, HardGraphSampleCache):
            raise ValueError("multi-view builder requires HardGraphSampleCache")
        graphs = self._graphs(cache)
        base = self.hard.build(graphs)
        theory = self.theory.build(graphs, cache.time_values)
        hard_windows = []
        if selected_object_windows is not None and len(selected_object_windows) != len(graphs):
            raise ValueError("selected fixed-K windows do not align with hard unions")
        for index, (source, node_window, theory_window) in enumerate(
            zip(graphs, base.windows, theory.windows)
        ):
            if source is None or node_window is None or theory_window is None:
                hard_windows.append(None)
            else:
                hard_windows.append(
                    self._window(
                        source,
                        node_window,
                        theory_window,
                        None
                        if selected_object_windows is None
                        else selected_object_windows[index],
                    )
                )
        transitions = []
        threshold = next(
            float(item.edge_presence_threshold) for item in graphs if item is not None
        )
        for index in range(max(0, len(hard_windows) - 1)):
            if not bool(theory.transition_mask[index]):
                transitions.append(None)
                continue
            transitions.append(
                self._transition(
                    index,
                    hard_windows[index],
                    hard_windows[index + 1],
                    theory.transition_features[index],
                    threshold,
                )
            )
        if selected_object_windows is not None:
            trajectory_set = build_dynamic_trajectories_from_costs(
                [
                    len(item.objects) if item is not None else 0
                    for item in hard_windows
                ],
                [
                    item.object_cost if item is not None else None
                    for item in transitions
                ],
                max(
                    len(item) for item in selected_object_windows
                ),
                self.tracking_config,
            )
        return MultiViewCriticalSampleFeatures(
            sample_key=cache.sample_key,
            label=int(cache.label),
            stable_static=base.static_features,
            hard_windows=tuple(hard_windows),
            full_windows=self._full_windows(full_graph_windows, theory.transition_features.new_zeros((len(graphs), MULTIVIEW_Q_DIM)) if full_graph_windows is not None else ()),
            transitions=tuple(transitions),
            window_mask=theory.window_mask,
            transition_mask=theory.transition_mask,
            trajectory_set=trajectory_set,
        )


def hard_windows_from_graph_sequence_sample(sample):
    """Adapt a raw GraphSequenceSample to the verified full-graph window type."""

    output = []
    for index, (adjacency, communities, names) in enumerate(
        zip(sample.adjacency, sample.communities, sample.node_names)
    ):
        node_ids = tuple(str(value) for value in names)
        if len(set(node_ids)) != len(node_ids):
            node_ids = tuple("{}:{}".format(index, node) for node in range(adjacency.shape[0]))
        output.append(
            HardGraphWindow(
                adjacency=adjacency,
                communities=communities,
                node_names=tuple(str(value) for value in names),
                node_ids=node_ids,
                time_start=float(sample.window_starts[index]),
                edge_presence_threshold=float(sample.edge_presence_threshold),
                window_valid=True,
            )
        )
    return tuple(output)
