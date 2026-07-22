from __future__ import absolute_import, division, print_function

import sys
import tempfile
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.extraction import (  # noqa: E402
    HardCandidatePoolBuilder,
    HardExtractionConfig,
    HardSubgraphCandidate,
    HardSubgraphExtractor,
)
from keysubgraph.data.graph_dataset import GraphSequenceSample  # noqa: E402
from keysubgraph.models import TGSoftTeacher, TGSoftTeacherConfig  # noqa: E402
from keysubgraph.theory import (  # noqa: E402
    CandidateScoreStandardizer,
    HardExportFidelityEvaluator,
    HardUnionGraphBuilder,
    SpectralGWGreedyExporter,
)


def _candidate(seed, nodes, edges, adjacency, score=1.0):
    return HardSubgraphCandidate(
        seed_node=seed,
        node_ids=tuple(nodes),
        node_names=tuple("n{}".format(node) for node in nodes),
        edge_index=tuple(edges),
        original_edge_weights=tuple(
            float(adjacency[left, right]) for left, right in edges
        ),
        node_scores=tuple(0.8 for _ in nodes),
        edge_scores=tuple(0.7 for _ in edges),
        community_labels=tuple(0 for _ in nodes),
        delta_degree=tuple(0.0 for _ in nodes),
        delta_degree_mask=tuple(False for _ in nodes),
        delta_edge_weight=tuple(0.0 for _ in edges),
        delta_edge_mask=tuple(False for _ in edges),
        score_node=score,
        score_edge=score,
        score_connectivity=1.0,
        score_dynamic=0.0,
        score_local_confidence=0.0,
        candidate_score=score,
    )


class TGHardStageBEightTest(unittest.TestCase):
    def setUp(self):
        self.adjacency = torch.tensor(
            [
                [0.0, -0.8, 0.4, 0.0],
                [-0.8, 0.0, 0.6, 0.0],
                [0.4, 0.6, 0.0, -0.5],
                [0.0, 0.0, -0.5, 0.0],
            ]
        )
        self.mask = self.adjacency.abs() > 0.0
        self.names = ("a", "b", "c", "d")
        self.communities = torch.tensor([0, 0, 1, 1])

    def test_union_is_non_induced_signed_and_inherits_metadata(self):
        first = _candidate(0, (0, 1), ((0, 1),), self.adjacency)
        second = _candidate(2, (1, 2), ((1, 2),), self.adjacency)
        builder = HardUnionGraphBuilder(max_union_nodes=3, max_union_edges=2)
        union, valid = builder.build(
            self.adjacency,
            self.names,
            self.mask,
            (first, second),
            communities=self.communities,
            edge_presence_threshold=0.0,
        )
        self.assertTrue(valid)
        self.assertEqual(union.edge_index, ((0, 1), (1, 2)))
        self.assertLess(float(union.adjacency[0, 1]), 0.0)
        self.assertEqual(float(union.adjacency[0, 2]), 0.0)
        self.assertEqual(union.node_names, ("a", "b", "c"))
        self.assertEqual(union.community_labels, (0, 0, 1))

    def test_empty_union_and_hard_budgets(self):
        builder = HardUnionGraphBuilder(max_union_nodes=2, max_union_edges=1)
        union, valid = builder.build(
            self.adjacency, self.names, self.mask, (), self.communities
        )
        self.assertIsNone(union)
        self.assertFalse(valid)
        too_large = _candidate(
            0, (0, 1, 2), ((0, 1), (1, 2)), self.adjacency
        )
        with self.assertRaises(ValueError):
            builder.build(
                self.adjacency,
                self.names,
                self.mask,
                (too_large,),
                self.communities,
            )

    def test_candidate_pool_rejects_invalid_and_deduplicates(self):
        high = _candidate(0, (0, 1), ((0, 1),), self.adjacency, score=0.9)
        duplicate = _candidate(1, (1, 0), ((0, 1),), self.adjacency, score=0.5)
        disconnected = _candidate(
            2, (0, 1, 2), ((0, 1),), self.adjacency, score=1.0
        )
        pool = HardCandidatePoolBuilder(2, 1, 4, 4).finalize(
            (duplicate, disconnected, high), self.adjacency, self.mask
        )
        self.assertTrue(pool.window_valid)
        self.assertEqual(tuple(item.seed_node for item in pool.candidates), (0,))
        self.assertEqual(pool.rejected_invalid, 1)
        self.assertEqual(pool.removed_duplicates, 1)
        empty = HardCandidatePoolBuilder(2, 1, 4, 4).finalize(
            (disconnected,), self.adjacency, self.mask
        )
        self.assertFalse(empty.window_valid)
        self.assertEqual(empty.candidates, ())


class TGHardStageBNineTest(unittest.TestCase):
    def setUp(self):
        self.adjacency = torch.tensor(
            [
                [0.0, 0.8, -0.3, 0.0],
                [0.8, 0.0, 0.5, 0.0],
                [-0.3, 0.5, 0.0, -0.6],
                [0.0, 0.0, -0.6, 0.0],
            ],
            dtype=torch.float64,
        )
        self.mask = self.adjacency.abs() > 0.0
        self.candidates = (
            _candidate(0, (0, 1), ((0, 1),), self.adjacency, 4.0),
            _candidate(1, (1, 2), ((1, 2),), self.adjacency, 3.0),
            _candidate(2, (2, 3), ((2, 3),), self.adjacency, 2.0),
            _candidate(3, (0, 2), ((0, 2),), self.adjacency, 1.0),
        )
        self.fidelity = HardExportFidelityEvaluator(
            laplacian_eta=1.0e-3,
            heat_kernel_t=1.0,
            spectral_quantile_grid=(0.25, 0.5, 0.75),
            train_max_iter=2,
            train_sinkhorn_iter=2,
            eval_max_iter=2,
            eval_sinkhorn_iter=2,
        )

    def test_candidate_score_scaler_is_train_only_and_round_trips(self):
        with self.assertRaises(ValueError):
            CandidateScoreStandardizer.fit(
                self.candidates, (1.0, 1.0, 1.0, 1.0), fit_split="validation"
            )
        scaler = CandidateScoreStandardizer.fit(
            self.candidates,
            (0.35, 0.35, 0.20, 0.10),
            fit_split="train",
            data_protocol_sha256="protocol",
            teacher_checkpoint_sha256="teacher",
        )
        self.assertEqual(scaler.fit_split, "train")
        self.assertGreaterEqual(min(scaler.scale), scaler.standard_deviation_floor)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "candidate_scaler.json"
            scaler.save(path)
            loaded = CandidateScoreStandardizer.load(path)
        self.assertEqual(loaded, scaler)
        self.assertAlmostEqual(loaded.score(self.candidates[0]), scaler.score(self.candidates[0]))

    def test_three_stage_filter_limits_gw_and_allows_empty(self):
        scaler = CandidateScoreStandardizer(
            feature_names=CandidateScoreStandardizer.DEFAULT_FEATURE_NAMES,
            mean=(0.0, 0.0, 0.0, 0.0),
            scale=(1.0, 1.0, 1.0, 1.0),
            weights=(1.0, 0.0, 0.0, 0.0),
            fit_split="train",
            standard_deviation_floor=1.0e-6,
        )
        calls = {"spectral": 0, "gw": 0}

        def spectral(soft_adjacency, edge_mask, union):
            calls["spectral"] += 1
            return float(union.edge_index[0][0])

        def gw(soft_adjacency, edge_mask, union):
            calls["gw"] += 1
            return 0.0, 0.0, True, 1, 0.0

        self.fidelity.fast_spectral_soft_to_hard = spectral
        self.fidelity.fast_soft_to_hard = gw
        exporter = SpectralGWGreedyExporter(
            self.fidelity,
            beta_lambda=0.0,
            beta_gw=0.0,
            beta_overlap=0.0,
            beta_size=0.0,
            candidate_score_scaler=scaler,
            prefilter_r1=3,
            prefilter_r2=2,
        )
        selected, union, trace = exporter.select(
            self.candidates,
            self.adjacency,
            ("a", "b", "c", "d"),
            self.mask,
            self.adjacency * 0.9,
            max_k=1,
        )
        self.assertEqual(len(selected), 1)
        self.assertIsNotNone(union)
        self.assertEqual(len(trace), 1)
        self.assertEqual(calls, {"spectral": 3, "gw": 2})
        self.assertEqual(exporter.last_prefilter_summary["discriminative_r1"], 3)
        self.assertEqual(exporter.last_prefilter_summary["spectral_r2"], 2)

        negative = CandidateScoreStandardizer(
            feature_names=CandidateScoreStandardizer.DEFAULT_FEATURE_NAMES,
            mean=(10.0, 0.0, 0.0, 0.0),
            scale=(1.0, 1.0, 1.0, 1.0),
            weights=(1.0, 0.0, 0.0, 0.0),
            fit_split="train",
            standard_deviation_floor=1.0e-6,
        )
        empty_exporter = SpectralGWGreedyExporter(
            self.fidelity,
            beta_lambda=0.0,
            beta_gw=0.0,
            beta_overlap=0.0,
            candidate_score_scaler=negative,
            prefilter_r1=3,
            prefilter_r2=2,
        )
        selected, union, trace = empty_exporter.select(
            self.candidates,
            self.adjacency,
            ("a", "b", "c", "d"),
            self.mask,
            self.adjacency,
            max_k=1,
        )
        self.assertEqual(selected, ())
        self.assertIsNone(union)
        self.assertEqual(trace, ())

    def test_candidate_generation_accepts_tg_soft_teacher(self):
        graph = self.adjacency.to(dtype=torch.float32)
        sample = GraphSequenceSample(
            sample_key="SITE/sample",
            sample_id="sample",
            site="SITE",
            subject_id="subject",
            session_id="1",
            label=0,
            split="train",
            relative_path="SITE/0/sample.pt",
            adjacency=(graph,),
            edge_mask=(graph.abs() > 0.0,),
            node_names=(("a", "b", "c", "d"),),
            communities=(torch.tensor([0, 0, 1, 1]),),
            window_starts=torch.tensor([0.0]),
            source_global_threshold=0.0,
            repetition_time=1.0,
            edge_presence_threshold=0.0,
        )
        teacher = TGSoftTeacher(
            TGSoftTeacherConfig(
                node_score_hidden_dim=4,
                edge_score_hidden_dim=4,
                signed_gnn_hidden_dim=4,
                signed_gnn_layers=1,
                dropout=0.0,
            )
        )
        extractor = HardSubgraphExtractor(
            teacher,
            HardExtractionConfig(
                max_nodes=4,
                max_edges=4,
                top_k=1,
                prefilter_spectral_top_r2=1,
                prefilter_discriminative_top_r1=2,
            ),
        )
        _, selection, pool, candidates = extractor.build_candidate_pool(sample, 0)
        self.assertEqual(tuple(selection.soft_adjacency.shape), (4, 4))
        self.assertGreaterEqual(len(candidates), 1)
        self.assertTrue(all(not parameter.requires_grad for parameter in teacher.parameters()))
        self.assertTrue(pool.window_valid)
        self.assertTrue(set(candidates).issubset(set(pool.candidates)))


if __name__ == "__main__":
    unittest.main()
