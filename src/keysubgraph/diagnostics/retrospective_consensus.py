"""Pure helpers for the retrospective exploration-consensus diagnostic."""

from __future__ import absolute_import, division, print_function

import itertools
import statistics
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


NodeSet = Set[int]
EdgeSet = Set[Tuple[int, int]]
ObjectSet = Tuple[NodeSet, EdgeSet]


def jaccard(left: Set, right: Set) -> float:
    union = left | right
    return len(left & right) / float(len(union)) if union else 1.0


def summarize(values: Iterable[float]) -> Dict[str, Optional[float]]:
    current = [float(value) for value in values]
    if not current:
        return {
            "count": 0,
            "minimum": None,
            "maximum": None,
            "mean": None,
            "median": None,
            "standard_deviation": None,
        }
    return {
        "count": len(current),
        "minimum": min(current),
        "maximum": max(current),
        "mean": statistics.mean(current),
        "median": statistics.median(current),
        "standard_deviation": statistics.pstdev(current),
    }


def transition_phase(
    right_window: int,
    exploration_windows: int,
    ramp_windows: int,
) -> str:
    """Assign transition ``right_window-1 -> right_window`` to one phase."""

    right_window = int(right_window)
    exploration_windows = max(0, int(exploration_windows))
    ramp_windows = max(1, int(ramp_windows))
    if right_window < exploration_windows:
        return "exploration_internal"
    if right_window == exploration_windows:
        return "exploration_boundary"
    if right_window < exploration_windows + ramp_windows:
        return "history_ramp"
    return "steady_state"


def best_object_assignment(
    previous: Sequence[ObjectSet],
    current: Sequence[ObjectSet],
) -> Dict[str, object]:
    """Exact small-K assignment maximizing mean node Jaccard.

    The selector currently uses K=3.  Exhaustive assignment avoids adding a
    SciPy dependency to the diagnostic helper and is exact for any small K.
    """

    count = min(len(previous), len(current))
    if count < 1:
        return {
            "permutation": (),
            "mean_node_jaccard": 0.0,
            "mean_edge_jaccard": 0.0,
        }
    best = None
    for permutation in itertools.permutations(range(len(current)), count):
        node_values = [
            jaccard(previous[index][0], current[target][0])
            for index, target in enumerate(permutation)
        ]
        edge_values = [
            jaccard(previous[index][1], current[target][1])
            for index, target in enumerate(permutation)
        ]
        candidate = (
            sum(node_values) / float(count),
            sum(edge_values) / float(count),
            tuple(int(value) for value in permutation),
        )
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    return {
        "permutation": best[2],
        "mean_node_jaccard": best[0],
        "mean_edge_jaccard": best[1],
    }


def accepted_assignment_metrics(
    previous: Sequence[ObjectSet],
    current: Sequence[ObjectSet],
    continuation_from: Sequence[int],
) -> Dict[str, float]:
    """Measure tracker-accepted matches, counting rejected slots as zero."""

    node_values: List[float] = []
    edge_values: List[float] = []
    accepted = 0
    for current_index, previous_index in enumerate(continuation_from):
        previous_index = int(previous_index)
        if (
            previous_index < 0
            or previous_index >= len(previous)
            or current_index >= len(current)
        ):
            continue
        accepted += 1
        node_values.append(
            jaccard(previous[previous_index][0], current[current_index][0])
        )
        edge_values.append(
            jaccard(previous[previous_index][1], current[current_index][1])
        )
    denominator = float(max(1, min(len(previous), len(current))))
    return {
        "accepted_count": float(accepted),
        "acceptance_rate": accepted / denominator,
        "accepted_mean_node_jaccard": (
            sum(node_values) / float(accepted) if accepted else 0.0
        ),
        "accepted_mean_edge_jaccard": (
            sum(edge_values) / float(accepted) if accepted else 0.0
        ),
        "coverage_adjusted_node_jaccard": sum(node_values) / denominator,
        "coverage_adjusted_edge_jaccard": sum(edge_values) / denominator,
    }


def aggregate_records(
    records: Sequence[Mapping[str, float]],
) -> Dict[str, Dict[str, Optional[float]]]:
    keys = sorted({key for record in records for key in record})
    return {
        key: summarize(record[key] for record in records if key in record)
        for key in keys
    }
