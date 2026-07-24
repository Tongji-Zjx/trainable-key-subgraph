from __future__ import absolute_import, division, print_function

import unittest

import torch

from keysubgraph.models.hard_stse_temporal import HardSTSETemporalEncoder
from keysubgraph.models.hard_stse_types import HardSTSEConfig


class HardSTSETemporalTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(113)
        self.encoder = HardSTSETemporalEncoder(HardSTSEConfig(dropout=0.0))
        self.encoder.eval()
        self.short = torch.randn(2, 128)
        self.long = torch.randn(4, 128)

    def test_variable_lengths_masks_and_output_dimensions(self):
        output = self.encoder((self.short, self.long))
        self.assertEqual(tuple(output.padded_states.shape), (2, 4, 128))
        self.assertEqual(tuple(output.time_mask.shape), (2, 4))
        self.assertEqual(output.sequence_lengths.tolist(), [2, 4])
        self.assertEqual(tuple(output.representation.shape), (2, 192))
        self.assertTrue(torch.equal(
            output.padded_states[0, 2:],
            torch.zeros_like(output.padded_states[0, 2:]),
        ))
        self.assertAlmostEqual(float(output.attention[0].sum()), 1.0, places=6)
        self.assertEqual(float(output.attention[0, 2:].sum()), 0.0)

    def test_other_sample_padding_does_not_change_representation(self):
        with torch.no_grad():
            alone = self.encoder((self.short,)).representation[0]
            batched = self.encoder((self.short, self.long)).representation[0]
        self.assertTrue(torch.allclose(alone, batched, atol=1.0e-6))

    def test_temporal_modules_receive_gradients(self):
        self.encoder.train()
        output = self.encoder((self.short, self.long))
        weights = torch.linspace(0.1, 1.0, 192)
        (output.representation * weights).sum().backward()
        for module in (
            self.encoder.delta_projection,
            self.encoder.bigru,
            self.encoder.attention,
            self.encoder.output,
        ):
            gradient = sum(
                float(parameter.grad.abs().sum())
                for parameter in module.parameters()
                if parameter.grad is not None
            )
            self.assertGreater(gradient, 0.0)


if __name__ == "__main__":
    unittest.main()
