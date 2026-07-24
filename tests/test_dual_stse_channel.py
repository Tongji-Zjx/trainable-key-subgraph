from __future__ import absolute_import, division, print_function

import unittest

import torch

from keysubgraph.data.exact_stse_dataset import (
    ExactSTSEBatch,
    ExactSTSESample,
)
from keysubgraph.models.dual_stse_channel import ExistingNoCoordSTSEChannel
from keysubgraph.models.exact_stse import (
    ExactSTSEClassifier,
    ExactSTSEConfig,
)
from tests.test_exact_stse_model import _exact_sample


class DualSTSEChannelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(307)
        self.sample = _exact_sample("dual-stse", 1, 3)
        self.batch = ExactSTSEBatch((self.sample,))

    def test_adapter_exactly_reuses_existing_no_coord_model(self):
        model = ExactSTSEClassifier(
            ExactSTSEConfig(use_coordinates=False, dropout=0.0)
        ).eval()
        channel = ExistingNoCoordSTSEChannel(model).eval()
        with torch.no_grad():
            expected = model(self.batch)
            actual = channel(self.batch)
        self.assertTrue(torch.equal(actual.logits, expected.logits))
        self.assertTrue(
            torch.equal(actual.representation, expected.subject_embedding)
        )
        self.assertEqual(actual.representation.shape[-1], 64)
        self.assertIs(channel.model, model)

    def test_coordinates_cannot_affect_adapter_output(self):
        changed = ExactSTSESample(
            graph=self.sample.graph,
            coordinates=tuple(
                coordinates * 91.0 - 13.0
                for coordinates in self.sample.coordinates
            ),
        )
        channel = ExistingNoCoordSTSEChannel().eval()
        with torch.no_grad():
            first = channel(self.batch).logits
            second = channel(ExactSTSEBatch((changed,))).logits
        self.assertTrue(torch.equal(first, second))

    def test_trainability_is_controlled_without_replacing_modules(self):
        channel = ExistingNoCoordSTSEChannel()
        channel.set_trainable(encoder=False, classifier=False)
        self.assertFalse(
            any(
                parameter.requires_grad
                for parameter in channel.parameters()
            )
        )
        channel.set_trainable(encoder=True, classifier=False)
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in channel.model.window_encoder.parameters()
            )
        )
        self.assertFalse(
            any(
                parameter.requires_grad
                for parameter in channel.model.classifier.parameters()
            )
        )

    def test_coordinate_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must not use coordinates"):
            ExistingNoCoordSTSEChannel(
                ExactSTSEClassifier(
                    ExactSTSEConfig(use_coordinates=True)
                )
            )


if __name__ == "__main__":
    unittest.main()
