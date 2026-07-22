from __future__ import absolute_import, division, print_function

import unittest

import torch

from keysubgraph.models import MaskedTCNEncoder, pad_temporal_sequences


class MaskedTCNTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.encoder = MaskedTCNEncoder(
            input_dim=4,
            hidden_dim=6,
            kernel_size=3,
            dilations=(1, 2, 4),
            dropout=0.0,
        )
        self.encoder.eval()

    def test_list_batch_preserves_lengths_and_returns_12d_representation(self):
        sequences = [torch.randn(3, 4), torch.randn(7, 4), torch.randn(2, 4)]
        representation, encoded, mask = self.encoder.forward_list(sequences)
        self.assertEqual(tuple(representation.shape), (3, 12))
        self.assertEqual(tuple(encoded.shape), (3, 7, 6))
        self.assertEqual(mask.sum(dim=1).tolist(), [3, 7, 2])
        self.assertEqual(float(encoded[~mask].abs().sum()), 0.0)

    def test_arbitrary_padding_values_cannot_change_output(self):
        sequence = torch.randn(1, 4, 4)
        mask = torch.tensor([[True, True, False, False]])
        first = sequence.clone()
        second = sequence.clone()
        first[:, 2:] = 1000.0
        second[:, 2:] = -999.0
        left, left_encoded = self.encoder(first, mask)
        right, right_encoded = self.encoder(second, mask)
        self.assertTrue(torch.allclose(left, right, atol=1.0e-6))
        self.assertTrue(torch.allclose(left_encoded, right_encoded, atol=1.0e-6))

    def test_padding_helper_rejects_empty_sequences(self):
        with self.assertRaises(ValueError):
            pad_temporal_sequences([])
        with self.assertRaises(ValueError):
            pad_temporal_sequences([torch.empty(0, 4)])


if __name__ == "__main__":
    unittest.main()
