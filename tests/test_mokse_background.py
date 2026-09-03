import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from keysubgraph.background.data import (
    build_global_static_record,
    build_static_node_features,
    fit_background_feature_scaler,
    signed_laplacian_encoding,
    signed_normalized_channels,
)
from keysubgraph.background.model import (
    GlobalBackgroundGCN,
    MoKSEBackgroundFusion,
    StaticBackgroundConfig,
)


class MoKSEBackgroundTest(unittest.TestCase):
    def adjacency(self):
        return torch.tensor(
            [
                [0.0, 0.7, -0.3, 0.0],
                [0.7, 0.0, 0.4, -0.2],
                [-0.3, 0.4, 0.0, 0.5],
                [0.0, -0.2, 0.5, 0.0],
            ],
            dtype=torch.float32,
        )

    def test_static_features_and_signed_spectrum_are_finite(self):
        adjacency = self.adjacency()
        communities = torch.tensor([0, 0, 1, 1])
        static = build_static_node_features(adjacency, communities)
        spectral, eigenvalues = signed_laplacian_encoding(adjacency, 3)
        self.assertEqual(tuple(static.shape), (4, 12))
        self.assertEqual(tuple(spectral.shape), (4, 3))
        self.assertTrue(torch.isfinite(static).all())
        self.assertTrue(torch.all(eigenvalues[1:] >= eigenvalues[:-1]))
        for column in range(spectral.shape[1]):
            pivot = int(spectral[:, column].abs().argmax())
            self.assertGreaterEqual(float(spectral[pivot, column]), 0.0)

    def test_manifest_matching_uses_site_and_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "SLD" / "1" / "sub-03.pt"
            path.parent.mkdir(parents=True)
            torch.save(
                {
                    "adjacency": self.adjacency(),
                    "community_labels": torch.tensor([0, 0, 1, 1]),
                    "coords": torch.zeros(4, 3),
                    "node_names": ["a", "b", "c", "d"],
                },
                path,
            )
            record = build_global_static_record(
                root,
                {"sample_key": "SLD/sub-03", "sample_id": "sub-03", "site": "SLD", "label": 1},
                spectral_dimensions=3,
            )
            self.assertEqual(record.sample_key, "SLD/sub-03")
            self.assertEqual(tuple(record.node_features.shape), (4, 15))

    def test_padding_does_not_change_graph_representation(self):
        torch.manual_seed(7)
        adjacency = self.adjacency()
        positive, negative = signed_normalized_channels(adjacency)
        features = torch.randn(1, 4, 15)
        config = StaticBackgroundConfig(
            input_dim=15, hidden_dim=8, representation_dim=5,
            layers=2, dropout=0.0,
        )
        model = GlobalBackgroundGCN(config).eval()
        direct = model(features, positive[None], negative[None], torch.ones(1, 4, dtype=torch.bool))
        padded_features = torch.zeros(1, 7, 15)
        padded_features[:, :4] = features
        padded_positive = torch.zeros(1, 7, 7)
        padded_negative = torch.zeros(1, 7, 7)
        padded_positive[:, :4, :4] = positive
        padded_negative[:, :4, :4] = negative
        mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0]], dtype=torch.bool)
        padded = model(padded_features, padded_positive, padded_negative, mask)
        self.assertTrue(
            torch.allclose(
                direct["background_representation"],
                padded["background_representation"],
                atol=1.0e-6,
            )
        )

    def test_fusion_residual_is_genuinely_bounded(self):
        fusion = MoKSEBackgroundFusion(alpha_max=0.5, alpha_initial=0.1)
        output = fusion(torch.zeros(3), torch.tensor([-1.0e6, 0.0, 1.0e6]))
        self.assertLessEqual(float(output["background_residual"].abs().max()), 0.5)
        self.assertAlmostEqual(float(output["fusion_alpha"]), 0.1, places=6)


if __name__ == "__main__":
    unittest.main()
