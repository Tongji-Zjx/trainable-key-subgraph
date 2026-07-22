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

from keysubgraph.features import HardGraphWindow  # noqa: E402
from keysubgraph.theory import (  # noqa: E402
    SGWFeatureExtractor,
    TGSGWFeatureArtifact,
    load_tg_sgw_feature_artifact,
    save_tg_sgw_feature_artifact,
)


def _window(adjacency, start, names=None):
    count = adjacency.shape[0]
    return HardGraphWindow(
        adjacency=adjacency,
        communities=torch.arange(count) % 2,
        node_names=tuple(names or ("n{}".format(i) for i in range(count))),
        node_ids=tuple(str(i) for i in range(count)),
        time_start=float(start),
        edge_presence_threshold=0.0,
        window_valid=True,
    )


class TGSGWFeatureTest(unittest.TestCase):
    def setUp(self):
        self.first = torch.tensor(
            [[0.0, 0.8, 0.0], [0.8, 0.0, -0.4], [0.0, -0.4, 0.0]],
            dtype=torch.float64,
        )
        self.second = torch.tensor(
            [[0.0, 0.3, -0.2], [0.3, 0.0, -0.7], [-0.2, -0.7, 0.0]],
            dtype=torch.float64,
        )
        self.extractor = SGWFeatureExtractor(
            gw_max_iter=10, gw_sinkhorn_iter=10
        )

    def test_identical_windows_are_zero_and_dimensions_are_fixed(self):
        window = _window(self.first, 0.0)
        result = self.extractor.compute_hard_graph_sequence(
            (window, _window(self.first.clone(), 1.0)), (0.0, 1.0)
        )
        self.assertEqual(tuple(result.h_core.shape), (18,))
        self.assertEqual(tuple(result.h_variation.shape), (16,))
        self.assertEqual(tuple(result.h_classification.shape), (34,))
        self.assertLess(float(result.h_classification.abs().max()), 1.0e-8)
        self.assertEqual(result.time_quantity, "speed")

    def test_spectral_and_gw_components_use_speed(self):
        first = self.extractor.compute_window_state(_window(self.first, 0.0))
        second = self.extractor.compute_window_state(_window(self.second, 1.0))
        fast = self.extractor.compute_sequence_feature((first, second), (0.0, 1.0))
        slow = self.extractor.compute_sequence_feature((first, second), (0.0, 2.0))
        self.assertTrue(torch.allclose(
            fast.transition_features[0, :16], slow.transition_features[0, :16]
        ))
        self.assertTrue(torch.allclose(
            fast.transition_features[0, 16:],
            2.0 * slow.transition_features[0, 16:],
            rtol=1.0e-5,
            atol=1.0e-8,
        ))

    def test_round_trip_reveals_variation_without_net_direction(self):
        first = self.extractor.compute_window_state(_window(self.first, 0.0))
        second = self.extractor.compute_window_state(_window(self.second, 1.0))
        result = self.extractor.compute_sequence_feature(
            (first, second, first), (0.0, 1.0, 2.0)
        )
        self.assertLess(float(result.h_core[:16].abs().max()), 1.0e-8)
        self.assertGreater(float(result.h_variation.max()), 1.0e-4)

    def test_invalid_windows_mask_transitions_and_never_decompose(self):
        result = self.extractor.compute_hard_graph_sequence(
            (_window(self.first, 0.0), None, _window(self.second, 2.0)),
            (0.0, 1.0, 2.0),
        )
        self.assertEqual(tuple(result.transition_mask.tolist()), (False, False))
        self.assertEqual(float(result.h_classification.abs().sum()), 0.0)

    def test_feature_artifact_round_trip(self):
        result = self.extractor.compute_hard_graph_sequence(
            (_window(self.first, 0.0), _window(self.second, 1.0)),
            (0.0, 1.0),
        )
        artifact = TGSGWFeatureArtifact(
            sample_key="SITE/sample",
            sample_id="sample",
            label=1,
            split="train",
            features=result,
            eligible_for_stage_c=True,
            data_protocol_sha256="protocol",
            teacher_checkpoint_sha256="teacher",
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "theory.pt"
            save_tg_sgw_feature_artifact(artifact, path)
            loaded = load_tg_sgw_feature_artifact(path)
        self.assertTrue(torch.equal(
            loaded.features.h_classification, artifact.features.h_classification
        ))


if __name__ == "__main__":
    unittest.main()
