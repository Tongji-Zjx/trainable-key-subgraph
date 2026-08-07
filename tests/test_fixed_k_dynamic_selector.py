from __future__ import absolute_import, division, print_function

import unittest
from dataclasses import replace

import torch

from keysubgraph.data.exact_stse_dataset import ExactSTSEBatch, ExactSTSESample
from keysubgraph.data.graph_dataset import GraphSequenceSample
from keysubgraph.data.multiview_critical import (
    MultiViewCriticalRecord,
    validate_multiview_record,
)
from keysubgraph.features.hard_graph_features import HardGraphWindow
from keysubgraph.features.hard_graph_cache import CachedHardWindow, HardGraphSampleCache
from keysubgraph.features.multiview_critical import (
    MultiViewCriticalFeatureBuilder,
    hard_windows_from_graph_sequence_sample,
)
from keysubgraph.models.dual_hard_sgw_selector import DualHardSGWSelector
from keysubgraph.models.dynamic_subgraph_tracking import (
    DynamicTrackingConfig,
    build_dynamic_trajectories,
    build_dynamic_trajectories_from_costs,
)
from keysubgraph.models.hard_stse_types import HardSelectionOutput, HardWindowOutput
from keysubgraph.models.fixed_k_subgraph_selector import select_fixed_k_subgraphs


def _sample(count=20, windows=3):
    adjacency = []
    masks = []
    names = []
    communities = []
    coordinates = []
    for time_index in range(windows):
        graph = torch.zeros(count, count)
        for left in range(count):
            for right in range(left + 1, count):
                value = (0.1 + 0.01 * ((left + right + time_index) % 7))
                if (left + right) % 3 == 0:
                    value = -value
                graph[left, right] = value
                graph[right, left] = value
        adjacency.append(graph)
        masks.append(graph.abs() > 0.0)
        names.append(tuple("ROI-{:02d}".format(index) for index in range(count)))
        communities.append(torch.arange(count, dtype=torch.long) % 5)
        coordinates.append(
            torch.stack(
                (
                    torch.arange(count, dtype=torch.float32),
                    torch.arange(count, dtype=torch.float32) % 4,
                    torch.ones(count),
                ),
                dim=1,
            )
        )
    graph = GraphSequenceSample(
        sample_key="site/sample",
        sample_id="sample",
        site="site",
        subject_id="subject",
        session_id="session",
        label=1,
        split="train",
        relative_path="sample.pt",
        adjacency=tuple(adjacency),
        edge_mask=tuple(masks),
        node_names=tuple(names),
        communities=tuple(communities),
        window_starts=torch.arange(windows, dtype=torch.float32),
        source_global_threshold=0.0,
        repetition_time=2.0,
        edge_presence_threshold=0.0,
    )
    return ExactSTSESample(graph=graph, coordinates=tuple(coordinates))


def _object(source_names, source_coordinates, selected, track_name):
    count = len(source_names)
    node_mask = torch.zeros(count, dtype=torch.bool)
    node_mask[list(selected)] = True
    edge_mask = torch.zeros(count, count, dtype=torch.bool)
    edge_mask[selected[0], selected[1]] = True
    edge_mask[selected[1], selected[0]] = True
    node_probability = torch.full((count,), 0.7)
    edge_probability = edge_mask.to(torch.float32) * 0.8
    selection = HardSelectionOutput(
        node_probability,
        edge_probability,
        node_mask,
        edge_mask,
        node_mask,
        node_probability + (node_mask.to(torch.float32) - node_probability).detach(),
        edge_probability + (edge_mask.to(torch.float32) - edge_probability).detach(),
        2,
        1,
        1,
        1,
        2,
        1,
        "test",
    )
    local_names = tuple(source_names[index] for index in selected)
    local_adjacency = torch.tensor([[0.0, -0.6], [-0.6, 0.0]])
    graph = HardGraphWindow(
        adjacency=local_adjacency,
        communities=torch.tensor((0, 0), dtype=torch.long),
        node_names=local_names,
        node_ids=local_names,
        time_start=0.0,
        edge_presence_threshold=0.0,
        window_valid=True,
    )
    return HardWindowOutput(
        adjacency_st=local_adjacency,
        hard_node_mask=node_mask,
        hard_edge_mask=edge_mask,
        straight_through_node_mask=selection.straight_through_node_mask,
        straight_through_edge_mask=selection.straight_through_edge_mask,
        cropped_graph=graph,
        window_valid=True,
        selection=selection,
    )


class FixedKDynamicSelectorTest(unittest.TestCase):
    def test_diverse_k3_enforces_spatial_seeds_and_nonoverlapping_objects(self):
        count = 30
        node = torch.linspace(1.0, 0.1, count)
        raw = (
            torch.arange(count * count, dtype=torch.float32)
            .reshape(count, count)
            .remainder(17)
            / 20.0
            + 0.1
        )
        edge = 0.5 * (raw + raw.transpose(0, 1))
        mask = torch.ones(count, count, dtype=torch.bool)
        mask.fill_diagonal_(False)
        communities = torch.arange(count, dtype=torch.long) % 5
        coordinates = torch.stack(
            (
                torch.arange(count, dtype=torch.float32),
                torch.arange(count, dtype=torch.float32) % 3,
                torch.ones(count),
            ),
            dim=1,
        )
        result = select_fixed_k_subgraphs(
            node,
            edge,
            communities,
            mask,
            subgraph_count=3,
            candidate_multiplier=6,
            coordinates=coordinates,
            diversity_enabled=True,
            per_object_node_ratio=0.10,
            node_reuse_decay=0.25,
            edge_reuse_decay=0.10,
            max_node_overlap=0.40,
            max_edge_overlap=0.25,
            min_unique_node_fraction=0.50,
            quality_floor_ratio=0.80,
            min_seed_distance=0.15,
        )
        self.assertEqual(len(result.subgraphs), 3)
        self.assertEqual(
            [item.actual_node_count for item in result.subgraphs],
            [3, 3, 3],
        )
        upper = torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1)
        self.assertLessEqual(
            float(result.pairwise_node_overlap[upper].max()), 0.40
        )
        self.assertLessEqual(
            float(result.pairwise_edge_overlap[upper].max()), 0.25
        )
        self.assertGreaterEqual(
            float(result.unique_node_fractions[1:].min()), 0.50
        )
        self.assertGreaterEqual(
            float(result.seed_distance_matrix[upper].min()), 0.15
        )
        self.assertGreaterEqual(result.union_efficiency, 0.60)
        self.assertFalse(result.diversity_constraint_relaxed)
        node_union = torch.zeros(count, dtype=torch.bool)
        edge_union = torch.zeros(count, count, dtype=torch.bool)
        for item in result.subgraphs:
            node_union |= item.hard_node_mask
            edge_union |= item.hard_edge_mask
        self.assertTrue(torch.equal(node_union, result.union.hard_node_mask))
        self.assertTrue(torch.equal(edge_union, result.union.hard_edge_mask))

    def test_cpu_snapshot_hardening_matches_preoptimization_golden_masks(self):
        generator = torch.Generator().manual_seed(7)
        count = 12
        node = (
            torch.rand(count, generator=generator) * 0.8
            + 0.1
            + torch.arange(count) * 1.0e-5
        )
        raw = torch.rand(count, count, generator=generator)
        edge = 0.5 * (raw + raw.transpose(0, 1))
        mask = edge > 0.42
        for index in range(count):
            mask[index, (index + 1) % count] = True
            mask[(index + 1) % count, index] = True
        mask.fill_diagonal_(False)
        communities = torch.arange(count, dtype=torch.long) % 4
        result = select_fixed_k_subgraphs(
            node,
            edge,
            communities,
            mask,
            subgraph_count=5,
            candidate_multiplier=2,
            total_node_ratio=0.5,
            edge_ratio=0.3,
            node_minimum=2,
            edge_minimum=1,
            overlap_penalty=0.25,
        )
        self.assertEqual(result.seed_indices.tolist(), [7, 10, 9, 11, 5])
        expected_nodes = (
            (2, 7),
            (3, 10),
            (2, 9),
            (10, 11),
            (3, 5),
        )
        for selection, expected in zip(result.subgraphs, expected_nodes):
            actual = tuple(
                torch.nonzero(
                    selection.hard_node_mask, as_tuple=False
                ).flatten().tolist()
            )
            self.assertEqual(actual, expected)
            actual_edges = tuple(
                tuple(value)
                for value in torch.nonzero(
                    torch.triu(selection.hard_edge_mask, diagonal=1),
                    as_tuple=False,
                ).tolist()
            )
            self.assertEqual(actual_edges, (expected,))

    def test_cpu_snapshot_preserves_straight_through_gradients(self):
        torch.manual_seed(709)
        count = 16
        node = torch.rand(count, requires_grad=True)
        edge = torch.rand(count, count, requires_grad=True)
        mask = torch.ones(count, count, dtype=torch.bool)
        mask.fill_diagonal_(False)
        communities = torch.arange(count, dtype=torch.long) % 5
        result = select_fixed_k_subgraphs(
            node, edge, communities, mask, subgraph_count=5
        )
        loss = result.union.straight_through_node_mask.sum()
        loss = loss + result.union.straight_through_edge_mask.sum()
        for selection in result.subgraphs:
            loss = loss + selection.straight_through_node_mask.sum()
            loss = loss + selection.straight_through_edge_mask.sum()
        loss.backward()
        self.assertIsNotNone(node.grad)
        self.assertIsNotNone(edge.grad)
        self.assertGreater(float(node.grad.abs().sum()), 0.0)
        self.assertGreater(float(edge.grad.abs().sum()), 0.0)

    def test_each_window_has_five_connected_objects_and_union_is_exact_or(self):
        torch.manual_seed(901)
        sample = _sample()
        selector = DualHardSGWSelector().eval()
        result = selector(ExactSTSEBatch((sample,)), selection_mode="learned")
        self.assertEqual(len(result.hard_subgraphs[0]), sample.num_timepoints)
        for time_index, (union, objects) in enumerate(
            zip(result.hard_windows[0], result.hard_subgraphs[0])
        ):
            self.assertEqual(len(objects), 5)
            self.assertTrue(all(item is not None and item.window_valid for item in objects))
            node_union = torch.zeros_like(union.hard_node_mask)
            edge_union = torch.zeros_like(union.hard_edge_mask)
            for item in objects:
                node_union |= item.hard_node_mask
                edge_union |= item.hard_edge_mask
                self.assertGreaterEqual(item.selection.actual_node_count, 2)
                self.assertGreaterEqual(item.selection.actual_edge_count, 1)
            self.assertTrue(torch.equal(node_union, union.hard_node_mask))
            self.assertTrue(torch.equal(edge_union, union.hard_edge_mask))
            source = sample.graph.adjacency[time_index]
            kept = union.adjacency_st.detach() != 0.0
            self.assertTrue(torch.equal(union.adjacency_st.detach()[kept], source[kept]))

    def test_death_and_birth_create_a_new_global_track(self):
        names0 = ("A1", "A2", "C1", "C2")
        names1 = ("B1", "B2", "C1", "C2")
        coords0 = torch.tensor(
            [[0.0, 1.0, 1.0], [0.0, 2.0, 1.0], [10.0, 1.0, 1.0], [10.0, 2.0, 1.0]]
        )
        coords1 = torch.tensor(
            [[30.0, 1.0, 1.0], [30.0, 2.0, 1.0], [10.0, 1.0, 1.0], [10.0, 2.0, 1.0]]
        )
        first = (
            _object(names0, coords0, (0, 1), "A"),
            _object(names0, coords0, (2, 3), "C"),
        )
        second = (
            _object(names1, coords1, (0, 1), "B"),
            _object(names1, coords1, (2, 3), "C"),
        )
        tracks = build_dynamic_trajectories(
            (first, second),
            (coords0, coords1),
            2,
            DynamicTrackingConfig(birth_cost=0.20, death_cost=0.20),
        )
        self.assertEqual(tracks.trajectory_count, 3)
        self.assertEqual(tracks.total_birth_count, 1)
        first_ids = tracks.assignments[0].track_ids.tolist()
        second_ids = tracks.assignments[1].track_ids.tolist()
        self.assertNotEqual(first_ids[0], second_ids[0])
        self.assertEqual(first_ids[1], second_ids[1])
        self.assertEqual(tracks.assignments[1].death_track_ids, (first_ids[0],))
        lengths = sorted(item.length for item in tracks.trajectories)
        self.assertEqual(lengths, [1, 1, 2])

    def test_stage0_uses_selector_objects_without_redecomposition(self):
        torch.manual_seed(907)
        sample = _sample(count=20, windows=2)
        selected = DualHardSGWSelector().eval()(
            ExactSTSEBatch((sample,)),
            selection_mode="learned",
            track_subgraphs=True,
        )
        union_windows = []
        object_windows = []
        for time_index, (union, objects) in enumerate(
            zip(selected.hard_windows[0], selected.hard_subgraphs[0])
        ):
            union_indices = torch.nonzero(
                union.hard_node_mask, as_tuple=False
            ).flatten()
            union_windows.append(
                replace(
                    union.cropped_graph,
                    coordinates=sample.coordinates[time_index].index_select(
                        0, union_indices
                    ),
                )
            )
            current = []
            for item in objects:
                indices = torch.nonzero(
                    item.hard_node_mask, as_tuple=False
                ).flatten()
                current.append(
                    replace(
                        item.cropped_graph,
                        coordinates=sample.coordinates[time_index].index_select(
                            0, indices
                        ),
                    )
                )
            object_windows.append(tuple(current))
        cache = HardGraphSampleCache(
            sample_key=sample.sample_key,
            sample_id=sample.graph.sample_id,
            label=sample.label,
            split="train",
            windows=tuple(
                CachedHardWindow(item, None, ()) for item in union_windows
            ),
            time_values=tuple(float(value) for value in sample.graph.window_starts),
            time_mask=(True, True),
            eligible_for_stage_c=True,
            exclusion_reason=None,
            data_protocol_sha256="protocol",
            teacher_checkpoint_sha256="selector",
        )
        built = MultiViewCriticalFeatureBuilder(uot_iterations=5).build(
            cache,
            full_graph_windows=hard_windows_from_graph_sequence_sample(sample.graph),
            selected_object_windows=tuple(object_windows),
            trajectory_set=selected.trajectory_sets[0],
        )
        self.assertEqual(len(built.hard_windows[0].objects), 5)
        self.assertEqual(len(built.hard_windows[1].objects), 5)
        self.assertIsNotNone(built.trajectory_set)
        self.assertGreaterEqual(built.trajectory_set.trajectory_count, 5)
        for item in built.hard_windows[0].objects:
            self.assertEqual(len(item.roi_ids), item.adjacency.shape[0])
            self.assertTrue(bool(item.coordinate_mask.all()))
        validate_multiview_record(
            MultiViewCriticalRecord(
                sample_id=sample.graph.sample_id,
                subject_id=sample.graph.subject_id,
                site=sample.graph.site,
                split="train",
                features=built,
                protocol_sha256="protocol",
                selector_checkpoint_sha256="selector",
                feature_schema_sha256="fixed-k-v1",
            )
        )

    def test_exact_fgw_cost_decoder_allows_partial_correspondence(self):
        tracks = build_dynamic_trajectories_from_costs(
            (2, 2),
            (torch.tensor([[1.5, 2.0], [2.0, 0.0]]),),
            2,
            DynamicTrackingConfig(birth_cost=0.20, death_cost=0.20),
        )
        self.assertEqual(tracks.trajectory_count, 3)
        self.assertEqual(tracks.total_birth_count, 1)
        first = tracks.assignments[0].track_ids.tolist()
        second = tracks.assignments[1].track_ids.tolist()
        self.assertNotEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])


if __name__ == "__main__":
    unittest.main()
