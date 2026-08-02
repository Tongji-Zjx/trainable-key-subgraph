from __future__ import absolute_import, division, print_function

import json
import tempfile
import unittest
from pathlib import Path

import torch

from keysubgraph.crossfit.svg_short_term_representation_f2_runner import (
    build_svg_short_term_representation_f2_fold_commands,
)
from keysubgraph.data.data_split import file_sha256
from keysubgraph.data.svg_short_term_representation_f2 import (
    SVGShortTermRepresentationF2Dataset,
)
from keysubgraph.models.svg_short_term_representation_f2 import (
    SVGShortTermRepresentationF2,
    SVGShortTermRepresentationF2Config,
)
from keysubgraph.training.svg_short_term_representation_f2_trainer import (
    SVGShortTermRepresentationF2TrainingConfig,
    create_svg_short_term_representation_f2_loader,
    train_svg_short_term_representation_f2,
)


def _write_manifest(root, split, count=8):
    root = Path(root) / split
    root.mkdir(parents=True, exist_ok=True)
    feature = root / "features.pt"
    keys = tuple("site/sample_{}".format(index) for index in range(count))
    torch.save(
        {
            "artifact_type": "svg_short_term_representation_f2_features",
            "schema_version": 1,
            "split": split,
            "sample_keys": keys,
            "sites": tuple("site" for _ in keys),
            "subject_ids": tuple("subject_{}".format(i) for i in range(count)),
            "labels": torch.tensor([index % 2 for index in range(count)]),
            "g2_anchor_logits": torch.linspace(-0.4, 0.4, count),
            "g2_representations": torch.randn(count, 6),
            "short_term_representations": torch.randn(count, 10),
        },
        str(feature),
    )
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifact_type": "svg_short_term_representation_f2_manifest",
                "schema_version": 1,
                "split": split,
                "sample_count": count,
                "g2_representation_dim": 6,
                "short_term_representation_dim": 10,
                "feature_file": "features.pt",
                "feature_sha256": file_sha256(feature),
                "protocol_sha256": "protocol",
                "short_term_checkpoint_sha256": "short",
                "g2_checkpoint_sha256": "g2",
                "g2_scaler_sha256": "scaler",
                "g2_spectral_scaler_sha256": "spectral",
                "g2_variant": "svg_v2_g2_signed_delta_q",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return manifest


class SVGShortTermRepresentationF2Test(unittest.TestCase):
    def test_zero_initialized_model_is_exactly_the_g2_anchor(self):
        model = SVGShortTermRepresentationF2(
            SVGShortTermRepresentationF2Config(10, 6)
        )
        anchor = torch.tensor([-0.7, 0.2, 1.1])
        short = torch.randn(3, 10)
        output = model(anchor, short)
        self.assertTrue(torch.equal(output.logits, anchor))
        self.assertTrue(torch.equal(output.residual_logits, torch.zeros(3)))
        self.assertAlmostEqual(float(output.gate), 0.01, places=6)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            output.logits, torch.tensor([0.0, 1.0, 0.0])
        )
        loss.backward()
        self.assertGreater(
            float(model.residual_head[-1].weight.grad.abs().sum()), 0.0
        )

    def test_artifact_round_trip_and_smoke_training(self):
        with tempfile.TemporaryDirectory(dir=".") as temporary:
            train_path = _write_manifest(temporary, "train")
            validation_path = _write_manifest(temporary, "validation")
            train = SVGShortTermRepresentationF2Dataset(train_path)
            validation = SVGShortTermRepresentationF2Dataset(validation_path)
            self.assertEqual(train.short_term_representation_dim, 10)
            model = SVGShortTermRepresentationF2(
                SVGShortTermRepresentationF2Config(10, 6, residual_hidden_dim=8)
            )
            train_loader = create_svg_short_term_representation_f2_loader(
                train, 4, 42, True
            )
            validation_loader = create_svg_short_term_representation_f2_loader(
                validation, 4, 42, False
            )
            output = Path(temporary) / "model"
            result = train_svg_short_term_representation_f2(
                model,
                train_loader,
                validation_loader,
                train.labels,
                torch.device("cpu"),
                SVGShortTermRepresentationF2TrainingConfig(
                    epochs=1,
                    early_stopping_patience=0,
                    minimum_epochs=0,
                ),
                output,
                {"protocol_sha256": "protocol"},
            )
            self.assertTrue(Path(result["best_checkpoint"]).is_file())
            self.assertTrue((output / "best_evaluation.json").is_file())

    def test_runner_builds_resumable_cache_train_and_evaluation_stages(self):
        commands = build_svg_short_term_representation_f2_fold_commands(
            Path("project"),
            Path("source"),
            Path("g2"),
            Path("output"),
            1,
            784341473,
        )
        self.assertEqual(
            [stage for stage, _, _ in commands],
            [
                "cache_train",
                "cache_validation",
                "cache_test",
                "train_f2",
                "evaluate_validation",
                "evaluate_test",
            ],
        )
        train_command = commands[3][1]
        self.assertIn("--initial-gate", train_command)
        self.assertIn("0.01", train_command)


if __name__ == "__main__":
    unittest.main()

