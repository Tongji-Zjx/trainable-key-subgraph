from __future__ import absolute_import, division, print_function

import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.models.baseline_classifier import (  # noqa: E402
    BaselineModelConfig,
    SignedSequenceBaseline,
)
from keysubgraph.models.node_only_subgraph_encoder import (  # noqa: E402
    NodeOnlySubgraphEncoder,
)
from tests.test_baseline_model import _sequence_batch  # noqa: E402


class NodeOnlySubgraphEncoderTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.encoder = NodeOnlySubgraphEncoder(12, 16, 2, dropout=0.0)
        self.encoder.eval()

    def test_padding_and_permutation_do_not_change_embedding(self):
        features = torch.randn(1, 3, 12)
        mask = torch.ones(1, 3, dtype=torch.bool)
        expected = self.encoder(features, mask)
        permutation = torch.tensor([2, 0, 1])
        actual = self.encoder(features.index_select(1, permutation), mask)
        self.assertTrue(torch.allclose(expected, actual, atol=1e-6, rtol=0.0))

        padded = torch.randn(1, 6, 12) * 100.0
        padded[:, :3] = features
        padded_mask = torch.zeros(1, 6, dtype=torch.bool)
        padded_mask[:, :3] = True
        actual = self.encoder(padded, padded_mask)
        self.assertTrue(torch.allclose(expected, actual, atol=1e-6, rtol=0.0))

    def test_features_affect_output_and_receive_gradients(self):
        features = torch.randn(2, 4, 12, requires_grad=True)
        mask = torch.ones(2, 4, dtype=torch.bool)
        original = self.encoder(features, mask)
        changed = features.detach().clone()
        changed[0, 0, 0] += 4.0
        self.assertFalse(torch.allclose(original, self.encoder(changed, mask)))
        original.sum().backward()
        self.assertTrue(bool(torch.isfinite(features.grad).all()))
        self.assertGreater(float(features.grad.abs().sum()), 0.0)


class NodeOnlySequenceBaselineTest(unittest.TestCase):
    def test_adjacency_cannot_affect_node_only_model(self):
        torch.manual_seed(72)
        model = SignedSequenceBaseline(
            BaselineModelConfig(
                encoder_type="node_only", signed_gnn_dropout=0.0,
                classifier_dropout=0.0, history_mode="independent_bag"
            )
        )
        model.eval()
        batch = _sequence_batch()
        expected = model(batch).logits
        batch.adjacency.copy_(torch.randn_like(batch.adjacency) * 50.0)
        actual = model(batch).logits
        self.assertTrue(torch.equal(expected, actual))

    def test_forward_backward_and_time_padding(self):
        torch.manual_seed(73)
        model = SignedSequenceBaseline(
            BaselineModelConfig(
                encoder_type="node_only", signed_gnn_dropout=0.0,
                classifier_dropout=0.0, history_mode="independent_bag"
            )
        )
        model.eval()
        expected = model(_sequence_batch()).logits
        padded = model(_sequence_batch(extra_time_padding=True)).logits
        self.assertTrue(torch.allclose(expected, padded, atol=1e-6, rtol=0.0))
        padded.sum().backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad]
        self.assertTrue(all(g is not None and bool(torch.isfinite(g).all()) for g in gradients))

    def test_invalid_encoder_type_is_rejected(self):
        with self.assertRaises(ValueError):
            BaselineModelConfig(encoder_type="unknown")


if __name__ == "__main__":
    unittest.main()
