from __future__ import absolute_import, division, print_function

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from keysubgraph.theory.class_margin_diagnostics import (
    apply_standardizer,
    class_margin_metrics,
    exact_uniform_wasserstein,
    fit_train_only_standardizer,
    stratified_paired_bootstrap,
)
from keysubgraph.theory.sgw_core_features import (
    SGWCoreConfig,
    compute_sgw_core_sequence,
    load_stage0_sample_artifact,
    save_stage0_sample_artifact,
)


class TheoryStage0Test(unittest.TestCase):
    def test_identical_graph_has_zero_one_step_evolution(self):
        adjacency = torch.tensor(
            [
                [0.0, 0.5, -0.2],
                [0.5, 0.0, 0.3],
                [-0.2, 0.3, 0.0],
            ],
            dtype=torch.float32,
        )
        result = compute_sgw_core_sequence(
            (adjacency, adjacency.clone()),
            (0.0, 1.0),
            edge_presence_threshold=0.0,
            config=SGWCoreConfig(
                gw_max_iter=5,
                gw_sinkhorn_iter=10,
                gw_tolerance=1.0e-5,
            ),
        )
        self.assertEqual(result.valid_transition_count, 1)
        self.assertTrue(
            torch.allclose(result.core, torch.zeros(18), atol=1.0e-6)
        )

    def test_invalid_transitions_are_excluded(self):
        adjacency = torch.tensor(
            [[0.0, -0.4], [-0.4, 0.0]], dtype=torch.float32
        )
        result = compute_sgw_core_sequence(
            (adjacency, None, adjacency),
            (0.0, 1.0, 2.0),
            edge_presence_threshold=0.0,
            config=SGWCoreConfig(
                gw_max_iter=5,
                gw_sinkhorn_iter=10,
                gw_tolerance=1.0e-5,
            ),
        )
        self.assertEqual(result.transition_mask.tolist(), [False, False])
        self.assertEqual(result.valid_transition_count, 0)
        self.assertTrue(torch.equal(result.core, torch.zeros(18)))

    def test_full_equals_hard_has_zero_radii_and_valid_bounds(self):
        full = np.asarray(
            [[0.0, 0.0], [0.2, 0.1], [1.0, 1.0], [1.2, 0.9]]
        )
        metrics = class_margin_metrics(full, full.copy(), [0, 0, 1, 1])
        self.assertAlmostEqual(metrics["eta_0_pair"], 0.0)
        self.assertAlmostEqual(metrics["eta_1_pair"], 0.0)
        self.assertAlmostEqual(metrics["eta_0_ot"], 0.0)
        self.assertAlmostEqual(metrics["eta_1_ot"], 0.0)
        self.assertAlmostEqual(
            metrics["delta_full"], metrics["delta_hard"]
        )
        self.assertTrue(metrics["checks"]["eta_ot_not_above_pair"])
        self.assertTrue(
            metrics["checks"]["hard_margin_not_below_ot_lower_bound"]
        )

    def test_empirical_ot_is_not_above_identity_pairing(self):
        full = np.asarray([[0.0], [2.0], [10.0], [12.0]])
        hard = np.asarray([[2.0], [0.0], [12.0], [10.0]])
        metrics = class_margin_metrics(full, hard, [0, 0, 1, 1])
        self.assertAlmostEqual(metrics["eta_0_ot"], 0.0)
        self.assertAlmostEqual(metrics["eta_1_ot"], 0.0)
        self.assertGreater(metrics["eta_0_pair"], 0.0)
        self.assertGreater(metrics["eta_1_pair"], 0.0)
        self.assertTrue(metrics["checks"]["eta_ot_not_above_pair"])

    def test_unequal_uniform_ot_uses_exact_masses(self):
        first = np.asarray([[0.0], [2.0]])
        second = np.asarray([[0.0], [1.0], [2.0]])
        # Each target atom carries 1/3.  The middle atom receives 1/6 from
        # each source, yielding total transport cost 1/3.
        self.assertAlmostEqual(
            exact_uniform_wasserstein(first, second), 1.0 / 3.0, places=7
        )

    def test_bootstrap_is_reproducible(self):
        full = np.asarray(
            [[0.0], [0.3], [0.7], [1.0], [1.3]], dtype=np.float64
        )
        hard = full + np.asarray([[0.1], [-0.1], [0.0], [0.1], [-0.1]])
        labels = [0, 0, 1, 1, 1]
        first = stratified_paired_bootstrap(
            full, hard, labels, repeats=12, seed=77
        )
        second = stratified_paired_bootstrap(
            full, hard, labels, repeats=12, seed=77
        )
        self.assertEqual(first, second)

    def test_standardizer_is_fitted_from_train_full_only(self):
        train = np.asarray([[0.0, 2.0], [2.0, 4.0]])
        scaler = fit_train_only_standardizer(train)
        self.assertEqual(scaler["fit_split"], "train")
        self.assertEqual(scaler["fit_source"], "full_core_only")
        before = apply_standardizer(np.asarray([[10.0, 20.0]]), scaler)
        after = apply_standardizer(np.asarray([[-99.0, 99.0]]), scaler)
        self.assertTrue(np.allclose(scaler["mean"], [1.0, 3.0]))
        self.assertFalse(np.allclose(before, after))

    def test_sample_artifact_round_trip(self):
        side = {
            "core": torch.arange(18, dtype=torch.float32),
            "window_quantiles": torch.zeros((2, 16)),
            "window_mask": torch.ones(2, dtype=torch.bool),
            "transition_features": torch.zeros((1, 18)),
            "transition_mask": torch.ones(1, dtype=torch.bool),
            "gw_solver_converged": (True,),
            "valid_transition_count": 1,
        }
        payload = {
            "sample_key": "site/sample",
            "label": 0,
            "split": "test",
            "full": side,
            "hard": side,
            "provenance": {"hash": "abc"},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.pt"
            save_stage0_sample_artifact(path, payload)
            loaded = load_stage0_sample_artifact(path)
        self.assertEqual(loaded["sample_key"], "site/sample")
        self.assertTrue(torch.equal(loaded["full"]["core"], side["core"]))


if __name__ == "__main__":
    unittest.main()
