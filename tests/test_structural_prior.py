from __future__ import absolute_import, division, print_function

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.features.structural_prior import (  # noqa: E402
    STATIC_WINDOW_STRUCTURAL_FEATURES,
    compute_static_subgraph_features,
    fit_structural_transform,
)
from keysubgraph.models.baseline_classifier import (  # noqa: E402
    BaselineModelConfig,
    SignedSequenceBaseline,
)
from tests.test_baseline_model import _sequence_batch  # noqa: E402


class _Dataset(object):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def _dataset(split="train"):
    samples = []
    base = torch.arange(1, 12, dtype=torch.float32)
    for index, label in enumerate((0, 0, 1, 1)):
        values = base + float(index) + label * torch.linspace(0.0, 2.0, 11)
        mask = torch.ones(11, dtype=torch.bool)
        mask[5] = index != 0
        subgraph = SimpleNamespace(structural_features=values, structural_mask=mask)
        window = SimpleNamespace(subgraphs=(subgraph,))
        samples.append(SimpleNamespace(
            split=split,
            label=label,
            sample_key="SITE/sample_{}".format(index),
            windows=(window,),
        ))
    return _Dataset(samples)


class StructuralPriorTest(unittest.TestCase):
    def test_static_signed_features_and_missing_sign_masks(self):
        adjacency = torch.tensor([
            [0.0, 0.5, -0.3],
            [0.5, 0.0, 0.2],
            [-0.3, 0.2, 0.0],
        ])
        values, mask = compute_static_subgraph_features(
            adjacency, torch.tensor([0, 0, 1]), 0.0
        )
        self.assertEqual(len(STATIC_WINDOW_STRUCTURAL_FEATURES), 11)
        self.assertAlmostEqual(float(values[6]), 0.7, places=6)
        self.assertAlmostEqual(float(values[8]), 0.3, places=6)
        self.assertAlmostEqual(float(values[9]), 0.5, places=6)
        self.assertAlmostEqual(float(values[10]), 0.0, places=6)
        self.assertTrue(bool(mask.all()))

        positive_only = adjacency.abs()
        _, positive_mask = compute_static_subgraph_features(
            positive_only, torch.tensor([0, 0, 1]), 0.0
        )
        self.assertFalse(bool(positive_mask[7]))
        self.assertFalse(bool(positive_mask[10]))
        self.assertTrue(bool(positive_mask[8]))

    def test_train_only_standardization_and_prior_controls(self):
        dataset = _dataset()
        transforms = {
            group: fit_structural_transform(dataset, group, beta=1.0, permutation_seed=7)
            for group in ("A", "B", "C", "D", "E")
        }
        self.assertFalse(transforms["A"]["use_structural_features"])
        self.assertEqual(transforms["A"]["prior_scale"], [1.0] * 11)
        for group in ("B", "C", "D", "E"):
            self.assertEqual(transforms[group]["fitted_on"], "train_only")
            self.assertEqual(transforms[group]["sample_count"], 4)
            self.assertEqual(transforms[group]["window_count"], 4)
        self.assertEqual(transforms["B"]["prior_scale"], [1.0] * 11)
        self.assertEqual(len(set(round(value, 8) for value in transforms["C"]["prior_scale"])), 1)
        self.assertEqual(
            sorted(transforms["D"]["prior_scale"]),
            sorted(transforms["E"]["prior_scale"]),
        )
        self.assertNotEqual(
            transforms["D"]["prior_scale"], transforms["E"]["prior_scale"]
        )
        with self.assertRaisesRegex(ValueError, "train"):
            fit_structural_transform(_dataset(split="validation"), "B")

    def test_groups_have_equal_parameters_and_zero_group_ignores_features(self):
        models = []
        for group, use_features, prior in (
            ("A", False, "none"),
            ("B", True, "none"),
            ("C", True, "uniform"),
            ("D", True, "real"),
            ("E", True, "permuted"),
        ):
            torch.manual_seed(31)
            model = SignedSequenceBaseline(BaselineModelConfig(
                node_hidden_dim=16,
                fusion_dim=24,
                gru_hidden_dim=20,
                classifier_hidden_dim=10,
                signed_gnn_dropout=0.0,
                classifier_dropout=0.0,
                structural_interface_version=1,
                structural_group=group,
                use_structural_features=use_features,
                prior_mode=prior,
            ))
            model.configure_structural_transform(
                torch.zeros(11), torch.ones(11), torch.ones(11)
            )
            model.eval()
            models.append(model)
        counts = [sum(parameter.numel() for parameter in model.parameters()) for model in models]
        self.assertEqual(len(set(counts)), 1)

        batch = _sequence_batch()
        changed = replace(
            batch, window_structural_features=batch.window_structural_features + 100.0
        )
        self.assertTrue(torch.allclose(models[0](batch).logits, models[0](changed).logits))
        self.assertFalse(torch.allclose(models[1](batch).logits, models[1](changed).logits))

        masked = batch.window_structural_mask.clone()
        masked[0, 0] = False
        first = replace(batch, window_structural_mask=masked)
        changed_values = batch.window_structural_features.clone()
        changed_values[0, 0] = 1e9
        second = replace(
            batch,
            window_structural_features=changed_values,
            window_structural_mask=masked,
        )
        self.assertTrue(torch.allclose(models[1](first).logits, models[1](second).logits))


if __name__ == "__main__":
    unittest.main()
