"""Identity-anchored partial matching for dynamic fixed-K subgraphs.

Exactly K objects may be active in every valid window, while the number of
global trajectories is dynamic.  A real-to-dummy assignment ends a track and
a dummy-to-real assignment starts a new one.  ROI identity and coordinates are
used only here; they never enter node/edge importance scorers.
"""

from __future__ import absolute_import, division, print_function

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .hard_stse_types import HardWindowOutput


@dataclass(frozen=True)
class SubgraphTrackDescriptor:
    roi_weights: Tuple[Tuple[str, float], ...]
    centroid: torch.Tensor
    coordinate_valid: bool
    spectral_signature: torch.Tensor


@dataclass(frozen=True)
class WindowTrackAssignment:
    window_index: int
    track_ids: torch.Tensor
    birth_mask: torch.Tensor
    continuation_from: torch.Tensor
    match_confidence: torch.Tensor
    death_track_ids: Tuple[int, ...]


@dataclass(frozen=True)
class DynamicSubgraphTrajectory:
    track_id: int
    birth_window: int
    death_window: int
    window_indices: Tuple[int, ...]
    object_indices: Tuple[int, ...]

    @property
    def length(self) -> int:
        return len(self.window_indices)


@dataclass(frozen=True)
class DynamicTrajectorySet:
    assignments: Tuple[WindowTrackAssignment, ...]
    trajectories: Tuple[DynamicSubgraphTrajectory, ...]
    active_subgraphs_per_valid_window: int
    total_birth_count: int

    @property
    def trajectory_count(self) -> int:
        return len(self.trajectories)


@dataclass(frozen=True)
class DynamicTrackingConfig:
    roi_weight: float = 0.55
    coordinate_weight: float = 0.25
    spectral_weight: float = 0.20
    history_weight: float = 0.25
    birth_cost: float = 0.45
    death_cost: float = 0.45
    history_length: int = 3
    epsilon: float = 1.0e-8

    def __post_init__(self) -> None:
        weights = (
            self.roi_weight,
            self.coordinate_weight,
            self.spectral_weight,
            self.history_weight,
            self.birth_cost,
            self.death_cost,
        )
        if any(float(value) < 0.0 for value in weights):
            raise ValueError("tracking weights and event costs cannot be negative")
        if self.roi_weight + self.coordinate_weight + self.spectral_weight <= 0.0:
            raise ValueError("tracking requires at least one correspondence term")
        if self.history_length < 1 or self.epsilon <= 0.0:
            raise ValueError("tracking history and epsilon must be positive")


def _descriptor(
    output: HardWindowOutput,
    coordinates: torch.Tensor,
) -> SubgraphTrackDescriptor:
    if output.cropped_graph is None or not output.window_valid:
        raise ValueError("cannot describe an invalid hard subgraph")
    indices = torch.nonzero(
        output.hard_node_mask, as_tuple=False
    ).flatten()
    if tuple(coordinates.shape) != (output.hard_node_mask.numel(), 3):
        raise ValueError("tracking coordinates do not align with source nodes")
    selected_coordinates = coordinates.index_select(0, indices).detach().cpu()
    coordinate_valid = bool(
        torch.isfinite(selected_coordinates).all()
        and selected_coordinates.numel()
        and (selected_coordinates.abs().sum() > 0.0)
    )
    centroid = (
        selected_coordinates.mean(dim=0)
        if coordinate_valid
        else torch.zeros(3, dtype=torch.float32)
    )
    probabilities = output.selection.node_probabilities.index_select(
        0, indices.to(output.selection.node_probabilities.device)
    ).detach().cpu().clamp_min(0.0)
    names = tuple(str(value) for value in output.cropped_graph.node_names)
    if len(names) != probabilities.numel():
        raise ValueError("tracking ROI identities do not align with hard nodes")
    totals: Dict[str, float] = {}
    for name, value in zip(names, probabilities.tolist()):
        totals[name] = totals.get(name, 0.0) + float(value)
    denominator = sum(totals.values())
    if denominator <= 0.0:
        denominator = float(max(1, len(totals)))
        totals = {name: 1.0 for name in totals}
    roi_weights = tuple(
        sorted((name, value / denominator) for name, value in totals.items())
    )

    adjacency = output.cropped_graph.adjacency.detach().cpu()
    degree = adjacency.abs().sum(dim=-1)
    laplacian = torch.diag(degree) - adjacency
    eigenvalues = torch.linalg.eigvalsh(laplacian)
    quantiles = torch.linspace(0.0, 1.0, 8)
    spectral = torch.quantile(eigenvalues, quantiles).to(torch.float32)
    return SubgraphTrackDescriptor(
        roi_weights=roi_weights,
        centroid=centroid.to(torch.float32),
        coordinate_valid=coordinate_valid,
        spectral_signature=spectral,
    )


def _roi_distance(
    left: SubgraphTrackDescriptor,
    right: SubgraphTrackDescriptor,
    epsilon: float,
) -> float:
    first = dict(left.roi_weights)
    second = dict(right.roi_weights)
    keys = set(first) | set(second)
    intersection = sum(min(first.get(key, 0.0), second.get(key, 0.0)) for key in keys)
    union = sum(max(first.get(key, 0.0), second.get(key, 0.0)) for key in keys)
    return 1.0 - intersection / max(float(epsilon), union)


def _coordinate_distance(
    left: SubgraphTrackDescriptor,
    right: SubgraphTrackDescriptor,
) -> Optional[float]:
    if not left.coordinate_valid or not right.coordinate_valid:
        return None
    magnitude = max(
        1.0,
        float(left.centroid.norm()),
        float(right.centroid.norm()),
    )
    return min(2.0, float((left.centroid - right.centroid).norm()) / magnitude)


def _spectral_distance(
    left: SubgraphTrackDescriptor,
    right: SubgraphTrackDescriptor,
    epsilon: float,
) -> float:
    scale = torch.cat(
        (left.spectral_signature.abs(), right.spectral_signature.abs())
    ).median().clamp_min(float(epsilon))
    return min(
        2.0,
        float(
            (left.spectral_signature - right.spectral_signature)
            .abs()
            .mean()
            / scale
        ),
    )


def descriptor_cost(
    left: SubgraphTrackDescriptor,
    right: SubgraphTrackDescriptor,
    config: DynamicTrackingConfig,
) -> float:
    values = [
        (float(config.roi_weight), _roi_distance(left, right, config.epsilon)),
        (
            float(config.spectral_weight),
            _spectral_distance(left, right, config.epsilon),
        ),
    ]
    coordinate = _coordinate_distance(left, right)
    if coordinate is not None:
        values.append((float(config.coordinate_weight), coordinate))
    denominator = sum(weight for weight, _ in values)
    return sum(weight * value for weight, value in values) / max(
        float(config.epsilon), denominator
    )


def _partial_assignment(
    costs: np.ndarray,
    birth_cost: float,
    death_cost: float,
) -> Tuple[Dict[int, int], Tuple[int, ...], Tuple[int, ...]]:
    previous_count, current_count = costs.shape
    size = previous_count + current_count
    large = 1.0e6
    augmented = np.full((size, size), large, dtype=np.float64)
    augmented[:previous_count, :current_count] = costs
    for index in range(previous_count):
        augmented[index, current_count + index] = float(death_cost)
    for index in range(current_count):
        augmented[previous_count + index, index] = float(birth_cost)
    augmented[previous_count:, current_count:] = 0.0
    rows, columns = linear_sum_assignment(augmented)
    matches = {}
    deaths = []
    births = []
    for row, column in zip(rows.tolist(), columns.tolist()):
        if row < previous_count and column < current_count:
            matches[int(column)] = int(row)
        elif row < previous_count and column >= current_count:
            deaths.append(int(row))
        elif row >= previous_count and column < current_count:
            births.append(int(column))
    return matches, tuple(sorted(deaths)), tuple(sorted(births))


def build_dynamic_trajectories(
    windows: Sequence[Sequence[Optional[HardWindowOutput]]],
    coordinates: Sequence[torch.Tensor],
    subgraph_count: int,
    config: Optional[DynamicTrackingConfig] = None,
) -> DynamicTrajectorySet:
    """Build variable-count contiguous tracks from fixed-K window objects."""

    config = config or DynamicTrackingConfig()
    if len(windows) != len(coordinates):
        raise ValueError("tracking windows and coordinate sequence differ in length")
    next_track_id = 0
    assignments: List[WindowTrackAssignment] = []
    track_observations: Dict[int, List[Tuple[int, int]]] = {}
    histories: Dict[int, List[SubgraphTrackDescriptor]] = {}
    previous_objects: List[Tuple[int, int, SubgraphTrackDescriptor]] = []

    for window_index, (objects, coordinate) in enumerate(zip(windows, coordinates)):
        if len(objects) != int(subgraph_count):
            raise ValueError("tracking requires exactly K object slots per window")
        valid = []
        for object_index, output in enumerate(objects):
            if output is None or not output.window_valid:
                continue
            valid.append((object_index, _descriptor(output, coordinate)))
        track_ids = torch.full((int(subgraph_count),), -1, dtype=torch.long)
        births = torch.zeros(int(subgraph_count), dtype=torch.bool)
        continuation = torch.full((int(subgraph_count),), -1, dtype=torch.long)
        confidence = torch.zeros(int(subgraph_count), dtype=torch.float32)
        death_track_ids: List[int] = []

        if not previous_objects:
            for object_index, descriptor in valid:
                track_id = next_track_id
                next_track_id += 1
                track_ids[object_index] = track_id
                births[object_index] = True
                track_observations[track_id] = [(window_index, object_index)]
                histories[track_id] = [descriptor]
        elif valid:
            cost_matrix = np.zeros((len(previous_objects), len(valid)), dtype=np.float64)
            for row, (_, track_id, previous_descriptor) in enumerate(previous_objects):
                history = histories[track_id][-int(config.history_length):]
                for column, (_, current_descriptor) in enumerate(valid):
                    local = descriptor_cost(previous_descriptor, current_descriptor, config)
                    historical = sum(
                        descriptor_cost(item, current_descriptor, config)
                        for item in history
                    ) / float(len(history))
                    cost_matrix[row, column] = local + float(config.history_weight) * historical
            matches, dead_rows, born_columns = _partial_assignment(
                cost_matrix, config.birth_cost, config.death_cost
            )
            death_track_ids.extend(previous_objects[row][1] for row in dead_rows)
            for column, (object_index, descriptor) in enumerate(valid):
                if column in matches:
                    row = matches[column]
                    previous_object_index, track_id, _ = previous_objects[row]
                    track_ids[object_index] = track_id
                    continuation[object_index] = previous_object_index
                    ordered = np.sort(cost_matrix[:, column])
                    second = ordered[1] if ordered.size > 1 else (
                        float(config.birth_cost) + float(config.death_cost)
                    )
                    confidence[object_index] = max(
                        0.0, float(second - cost_matrix[row, column])
                    )
                    track_observations[track_id].append((window_index, object_index))
                    histories[track_id].append(descriptor)
                else:
                    track_id = next_track_id
                    next_track_id += 1
                    track_ids[object_index] = track_id
                    births[object_index] = True
                    track_observations[track_id] = [(window_index, object_index)]
                    histories[track_id] = [descriptor]
            # Defensive check: every unmatched current column must be decoded as birth.
            if set(born_columns) != {
                column for column in range(len(valid)) if column not in matches
            }:
                raise RuntimeError("partial assignment birth decoding is inconsistent")
        else:
            death_track_ids.extend(item[1] for item in previous_objects)

        assignments.append(
            WindowTrackAssignment(
                window_index=window_index,
                track_ids=track_ids,
                birth_mask=births,
                continuation_from=continuation,
                match_confidence=confidence,
                death_track_ids=tuple(sorted(death_track_ids)),
            )
        )
        previous_objects = []
        for object_index, descriptor in valid:
            track_id = int(track_ids[object_index])
            previous_objects.append((object_index, track_id, descriptor))

    final_window = max(0, len(windows) - 1)
    trajectories = []
    for track_id in sorted(track_observations):
        observations = track_observations[track_id]
        windows_for_track = tuple(item[0] for item in observations)
        objects_for_track = tuple(item[1] for item in observations)
        if any(
            right != left + 1
            for left, right in zip(windows_for_track[:-1], windows_for_track[1:])
        ):
            raise RuntimeError("tracking currently forbids trajectory gaps")
        trajectories.append(
            DynamicSubgraphTrajectory(
                track_id=track_id,
                birth_window=windows_for_track[0],
                death_window=windows_for_track[-1],
                window_indices=windows_for_track,
                object_indices=objects_for_track,
            )
        )
    initial_births = int(assignments[0].birth_mask.sum()) if assignments else 0
    return DynamicTrajectorySet(
        assignments=tuple(assignments),
        trajectories=tuple(trajectories),
        active_subgraphs_per_valid_window=int(subgraph_count),
        total_birth_count=max(0, len(trajectories) - initial_births),
    )


def build_dynamic_trajectories_from_costs(
    window_object_counts: Sequence[int],
    transition_costs: Sequence[Optional[torch.Tensor]],
    subgraph_count: int,
    config: Optional[DynamicTrackingConfig] = None,
) -> DynamicTrajectorySet:
    """Decode exact FGW/identity transition costs into birth/death tracks."""

    config = config or DynamicTrackingConfig()
    if len(transition_costs) != max(0, len(window_object_counts) - 1):
        raise ValueError("transition costs do not align with object windows")
    next_track_id = 0
    assignments = []
    observations: Dict[int, List[Tuple[int, int]]] = {}
    previous_track_ids: List[int] = []
    for window_index, object_count in enumerate(window_object_counts):
        object_count = int(object_count)
        if object_count < 0 or object_count > int(subgraph_count):
            raise ValueError("window object count exceeds fixed-K slots")
        track_ids = torch.full((int(subgraph_count),), -1, dtype=torch.long)
        births = torch.zeros(int(subgraph_count), dtype=torch.bool)
        continuation = torch.full((int(subgraph_count),), -1, dtype=torch.long)
        confidence = torch.zeros(int(subgraph_count), dtype=torch.float32)
        death_ids = []
        if window_index == 0 or not previous_track_ids:
            for object_index in range(object_count):
                track_id = next_track_id
                next_track_id += 1
                track_ids[object_index] = track_id
                births[object_index] = True
                observations[track_id] = [(window_index, object_index)]
        else:
            tensor = transition_costs[window_index - 1]
            if tensor is None:
                death_ids.extend(previous_track_ids)
                matches = {}
                costs = np.zeros((len(previous_track_ids), object_count), dtype=np.float64)
            else:
                if tuple(tensor.shape) != (len(previous_track_ids), object_count):
                    raise ValueError("exact tracking cost matrix has an invalid shape")
                costs = tensor.detach().cpu().numpy().astype(np.float64)
                matches, dead_rows, _ = _partial_assignment(
                    costs, config.birth_cost, config.death_cost
                )
                death_ids.extend(previous_track_ids[row] for row in dead_rows)
            for object_index in range(object_count):
                if object_index in matches:
                    row = matches[object_index]
                    track_id = previous_track_ids[row]
                    track_ids[object_index] = track_id
                    continuation[object_index] = row
                    ordered = np.sort(costs[:, object_index])
                    second = ordered[1] if ordered.size > 1 else (
                        float(config.birth_cost) + float(config.death_cost)
                    )
                    confidence[object_index] = max(
                        0.0, float(second - costs[row, object_index])
                    )
                    observations[track_id].append((window_index, object_index))
                else:
                    track_id = next_track_id
                    next_track_id += 1
                    track_ids[object_index] = track_id
                    births[object_index] = True
                    observations[track_id] = [(window_index, object_index)]
        assignments.append(
            WindowTrackAssignment(
                window_index=window_index,
                track_ids=track_ids,
                birth_mask=births,
                continuation_from=continuation,
                match_confidence=confidence,
                death_track_ids=tuple(sorted(death_ids)),
            )
        )
        previous_track_ids = [int(track_ids[index]) for index in range(object_count)]

    trajectories = []
    for track_id in sorted(observations):
        current = observations[track_id]
        window_indices = tuple(item[0] for item in current)
        trajectories.append(
            DynamicSubgraphTrajectory(
                track_id=track_id,
                birth_window=window_indices[0],
                death_window=window_indices[-1],
                window_indices=window_indices,
                object_indices=tuple(item[1] for item in current),
            )
        )
    initial = int(assignments[0].birth_mask.sum()) if assignments else 0
    return DynamicTrajectorySet(
        assignments=tuple(assignments),
        trajectories=tuple(trajectories),
        active_subgraphs_per_valid_window=int(subgraph_count),
        total_birth_count=max(0, len(trajectories) - initial),
    )
