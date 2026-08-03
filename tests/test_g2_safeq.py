from __future__ import absolute_import, division, print_function

import json
import tempfile
import unittest
from pathlib import Path

import torch

from keysubgraph.data.data_split import file_sha256
from keysubgraph.data.g2_safeq import G2SafeQDataset
from keysubgraph.models.g2_safeq import (
    G2SafeQConfig,
    G2SafeQResidual,
    aggregate_transition_hidden,
)
from keysubgraph.training.g2_safeq_trainer import (
    G2SafeQTrainingConfig,
    create_g2_safeq_loader,
    select_g2_safeq_mixing,
    train_g2_safeq,
)


def _write_manifest(root, split, count=12):
    directory = Path(root) / split
    directory.mkdir(parents=True, exist_ok=True)
    feature = directory / "features.pt"
    labels = torch.tensor([index % 2 for index in range(count)])
    summary = torch.randn(count, 8)
    has_transition = torch.tensor(
        [index % 4 != 0 for index in range(count)], dtype=torch.bool
    )
    summary[~has_transition] = 0.0
    torch.save(
        {
            "artifact_type": "g2_safeq_features",
            "schema_version": 1,
            "split": split,
            "sample_keys": tuple(
                "site_{}/sample_{}".format(index % 2, index)
                for index in range(count)
            ),
            "sites": tuple("site_{}".format(index % 2) for index in range(count)),
            "subject_ids": tuple("subject_{}".format(index) for index in range(count)),
            "labels": labels,
            "base_logits": torch.zeros(count),
            "static_logits": torch.linspace(-0.2, 0.2, count),
            "transition_summaries": summary,
            "has_valid_transition": has_transition,
        },
        str(feature),
    )
    manifest = directory / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifact_type": "g2_safeq_manifest",
                "schema_version": 1,
                "split": split,
                "sample_count": count,
                "transition_hidden_dim": 4,
                "summary_dim": 8,
                "feature_file": feature.name,
                "feature_sha256": file_sha256(feature),
                "protocol_sha256": "protocol",
                "selector_checkpoint_sha256": "selector",
                "selection_mode": "learned",
                "selection_seed": 42,
                "g2_checkpoint_sha256": "g2",
                "g2_scaler_sha256": "scaler",
                "g2_spectral_scaler_sha256": "spectral",
                "g2_variant": "svg_v2_g2_signed_delta_q",
                "frozen_g2": True,
                "train_only_scalers": True,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return manifest


class G2SafeQTest(unittest.TestCase):
    def test_zero_zero_is_exact_frozen_g2_identity(self):
        model = G2SafeQResidual(
            G2SafeQConfig(
                transition_hidden_dim=3,
                residual_hidden_dim=4,
                dropout=0.0,
            )
        )
        with torch.no_grad():
            model.residual_head[-1].weight.fill_(0.5)
            model.residual_head[-1].bias.fill_(1.0)
        base = torch.tensor((-0.7, 0.2, 1.1))
        static = torch.tensor((0.5, -0.4, 0.8))
        summary = torch.randn(3, 6)
        valid = torch.tensor((True, True, True))
        output = model(base, static, summary, valid, alpha=0.0, beta=0.0)
        self.assertTrue(torch.equal(output.logits, base))

    def test_variable_transition_aggregation_and_empty_sample_are_safe(self):
        hidden = torch.tensor(
            (
                (1.0, 3.0),
                (3.0, 7.0),
                (5.0, 11.0),
            )
        )
        indices = torch.tensor((0, 0, 2), dtype=torch.long)
        summary, valid = aggregate_transition_hidden(hidden, indices, 4, 2)
        self.assertTrue(
            torch.allclose(summary[0], torch.tensor((2.0, 5.0, 1.0, 2.0)))
        )
        self.assertTrue(
            torch.allclose(summary[2], torch.tensor((5.0, 11.0, 0.0, 0.0)))
        )
        self.assertTrue(torch.equal(valid, torch.tensor((True, False, True, False))))
        self.assertTrue(torch.equal(summary[1], torch.zeros(4)))
        self.assertTrue(torch.equal(summary[3], torch.zeros(4)))

        model = G2SafeQResidual(
            G2SafeQConfig(transition_hidden_dim=2, residual_hidden_dim=3, dropout=0.0)
        )
        with torch.no_grad():
            model.residual_head[-1].weight.fill_(1.0)
            model.residual_head[-1].bias.fill_(2.0)
        base = torch.zeros(4)
        output = model(base, base, summary, valid, alpha=1.0, beta=0.0)
        self.assertEqual(float(output.residual_logits[1]), 0.0)
        self.assertEqual(float(output.residual_logits[3]), 0.0)

    def test_frozen_inputs_receive_no_gradient_and_only_residual_updates(self):
        model = G2SafeQResidual(
            G2SafeQConfig(transition_hidden_dim=2, residual_hidden_dim=3, dropout=0.0)
        )
        base = torch.randn(4, requires_grad=True)
        static = torch.randn(4, requires_grad=True)
        summary = torch.randn(4, 4, requires_grad=True)
        output = model(
            base,
            static,
            summary,
            torch.ones(4, dtype=torch.bool),
            alpha=1.0,
            beta=0.5,
        )
        torch.nn.functional.binary_cross_entropy_with_logits(
            output.logits, torch.tensor((0.0, 1.0, 0.0, 1.0))
        ).backward()
        self.assertIsNone(base.grad)
        self.assertIsNone(static.grad)
        self.assertIsNone(summary.grad)
        self.assertGreater(
            float(model.residual_head[-1].weight.grad.abs().sum()), 0.0
        )

    def test_mixing_is_validation_only_and_falls_back_without_gain(self):
        labels = (0, 0, 1, 1)
        sites = ("a", "b", "a", "b")
        with self.assertRaises(ValueError):
            select_g2_safeq_mixing(
                labels,
                sites,
                (0.0,) * 4,
                (0.0,) * 4,
                (-1.0, -1.0, 1.0, 1.0),
                split="test",
            )
        fallback = select_g2_safeq_mixing(
            labels,
            sites,
            (-1.0, -0.5, 0.5, 1.0),
            (-1.0, -0.5, 0.5, 1.0),
            (0.0, 0.0, 0.0, 0.0),
            split="validation",
        )
        self.assertTrue(fallback["fallback_to_frozen_g2"])
        self.assertEqual(fallback["selected"]["alpha"], 0.0)
        self.assertEqual(fallback["selected"]["beta"], 0.0)

        accepted = select_g2_safeq_mixing(
            (0, 1, 0, 1, 0, 1, 0, 1),
            ("a", "a", "b", "b", "a", "a", "b", "b"),
            (0.0,) * 8,
            (0.0,) * 8,
            (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
            split="validation",
        )
        self.assertFalse(accepted["fallback_to_frozen_g2"])
        self.assertEqual(accepted["selected"]["alpha"], 0.25)
        self.assertEqual(accepted["selected"]["beta"], 0.0)

    def test_artifact_provenance_and_smoke_training(self):
        with tempfile.TemporaryDirectory(dir=".") as temporary:
            train_path = _write_manifest(temporary, "train")
            validation_path = _write_manifest(temporary, "validation")
            train = G2SafeQDataset(train_path)
            validation = G2SafeQDataset(validation_path)
            self.assertTrue(train.manifest["train_only_scalers"])
            model = G2SafeQResidual(
                G2SafeQConfig(transition_hidden_dim=4, residual_hidden_dim=4)
            )
            output = Path(temporary) / "model"
            result = train_g2_safeq(
                model,
                create_g2_safeq_loader(train, 4, 42, True),
                create_g2_safeq_loader(validation, 4, 42, False),
                train.labels,
                torch.device("cpu"),
                G2SafeQTrainingConfig(
                    epochs=1,
                    early_stopping_patience=0,
                    minimum_epochs=0,
                ),
                output,
                {
                    "protocol_sha256": "protocol",
                    "outer_test_used": False,
                },
            )
            self.assertTrue(Path(result["best_checkpoint"]).is_file())
            checkpoint = torch.load(
                str(result["best_checkpoint"]), map_location="cpu"
            )
            self.assertEqual(checkpoint["mixing_fit_split"], "validation")
            self.assertFalse(checkpoint["provenance"]["outer_test_used"])


if __name__ == "__main__":
    unittest.main()
