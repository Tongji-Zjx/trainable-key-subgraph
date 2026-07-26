from __future__ import absolute_import, division, print_function

import unittest

import torch

from keysubgraph.data.dual_temporal_dataset import DualTemporalBatch
from keysubgraph.models.dual_variation_temporal import (
    DUAL_TEMPORAL_VARIANTS,
    DualVariationTemporalClassifier,
    DualVariationTemporalConfig,
)


def _batch(padding_value=0.0):
    values = torch.randn(3, 4, 16)
    mask = torch.tensor(
        [
            [True, True, True, True],
            [True, True, False, False],
            [False, False, False, False],
        ]
    )
    values[~mask] = float(padding_value)
    return DualTemporalBatch(
        sample_keys=("a", "b", "c"),
        labels=torch.tensor([0, 1, 0]),
        transition_values=values,
        time_mask=mask,
        sequence_lengths=torch.tensor([4, 2, 0]),
        base_logits=torch.tensor(
            [[0.2, -0.1], [-0.3, 0.4], [0.7, -0.2]]
        ),
    )


class DualVariationTemporalModelTest(unittest.TestCase):
    def test_all_variants_have_expected_shapes(self):
        batch = _batch()
        for variant in DUAL_TEMPORAL_VARIANTS:
            model = DualVariationTemporalClassifier(
                DualVariationTemporalConfig(variant=variant)
            ).eval()
            output = model(batch)
            self.assertEqual(tuple(output.final_logits.shape), (3, 2))
            self.assertEqual(
                tuple(output.temporal_representation.shape), (3, 32)
            )
            self.assertTrue(bool(torch.isfinite(output.final_logits).all()))

    def test_residual_alpha_and_empty_sequence_identity(self):
        model = DualVariationTemporalClassifier().eval()
        batch = _batch()
        output = model(batch)
        self.assertAlmostEqual(float(output.alpha), 0.1, places=6)
        torch.testing.assert_close(
            output.temporal_logits[2], torch.zeros(2)
        )
        torch.testing.assert_close(
            output.final_logits[2], batch.base_logits[2]
        )

    def test_padding_values_do_not_change_outputs(self):
        torch.manual_seed(7)
        model = DualVariationTemporalClassifier().eval()
        first = _batch(padding_value=0.0)
        second = DualTemporalBatch(
            sample_keys=first.sample_keys,
            labels=first.labels,
            transition_values=first.transition_values.clone(),
            time_mask=first.time_mask,
            sequence_lengths=first.sequence_lengths,
            base_logits=first.base_logits,
        )
        second.transition_values[~second.time_mask] = 999.0
        left = model(first).final_logits
        right = model(second).final_logits
        torch.testing.assert_close(left, right)

    def test_batch_permutation_is_equivariant(self):
        torch.manual_seed(11)
        model = DualVariationTemporalClassifier().eval()
        batch = _batch()
        order = torch.tensor([2, 0, 1])
        permuted = DualTemporalBatch(
            sample_keys=tuple(batch.sample_keys[index] for index in order),
            labels=batch.labels.index_select(0, order),
            transition_values=batch.transition_values.index_select(0, order),
            time_mask=batch.time_mask.index_select(0, order),
            sequence_lengths=batch.sequence_lengths.index_select(0, order),
            base_logits=batch.base_logits.index_select(0, order),
        )
        expected = model(batch).final_logits.index_select(0, order)
        actual = model(permuted).final_logits
        torch.testing.assert_close(expected, actual)

    def test_gradient_reaches_temporal_branch_not_base_logits(self):
        torch.manual_seed(13)
        model = DualVariationTemporalClassifier()
        batch = _batch()
        batch.base_logits.requires_grad_(True)
        output = model(batch)
        output.final_logits.sum().backward()
        self.assertIsNone(batch.base_logits.grad)
        self.assertIsNotNone(model.alpha_logit.grad)
        self.assertTrue(
            any(
                parameter.grad is not None
                for name, parameter in model.named_parameters()
                if name != "alpha_logit"
            )
        )


if __name__ == "__main__":
    unittest.main()
