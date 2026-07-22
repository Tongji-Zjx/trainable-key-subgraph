from __future__ import absolute_import, division, print_function

import tempfile
import unittest
from pathlib import Path

import torch

from keysubgraph.models import TGSoftTeacher, TGSoftTeacherConfig, TGSoftTeacherLossConfig
from keysubgraph.training import (
    TGSoftTeacherTrainingConfig,
    load_tg_soft_teacher_checkpoint,
    save_tg_soft_teacher_checkpoint,
)


class TGSoftTeacherCheckpointTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        self.config = TGSoftTeacherConfig(
            node_score_hidden_dim=8,
            edge_score_hidden_dim=8,
            signed_gnn_hidden_dim=8,
            signed_gnn_layers=1,
            classifier_hidden_dims=(8,),
            dropout=0.0,
        )
        self.model = TGSoftTeacher(self.config)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1.0e-3)
        self.loss_config = TGSoftTeacherLossConfig()
        self.training_config = TGSoftTeacherTrainingConfig(epochs=1)

    def test_versioned_checkpoint_roundtrip_and_legacy_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_tg_soft_teacher_checkpoint(
                path,
                model=self.model,
                optimizer=self.optimizer,
                epoch=1,
                history=[{"epoch": 1}],
                loss_config=self.loss_config,
                training_config=self.training_config,
                protocol_path=Path(directory) / "protocol.json",
                protocol_sha256="abc",
                best_epoch=1,
                best_selection_value=0.5,
            )
            restored = TGSoftTeacher(self.config)
            payload = load_tg_soft_teacher_checkpoint(
                path,
                restored,
                torch.device("cpu"),
                expected_protocol_sha256="abc",
            )
            self.assertEqual(payload["stage"], "soft_teacher")
            for left, right in zip(self.model.parameters(), restored.parameters()):
                self.assertTrue(torch.equal(left, right))

            legacy = Path(directory) / "legacy.pt"
            torch.save({"model_state_dict": self.model.state_dict()}, str(legacy))
            with self.assertRaises(ValueError):
                load_tg_soft_teacher_checkpoint(legacy, restored, torch.device("cpu"))

    def test_protocol_and_model_mismatch_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_tg_soft_teacher_checkpoint(
                path,
                model=self.model,
                optimizer=self.optimizer,
                epoch=1,
                history=[],
                loss_config=self.loss_config,
                training_config=self.training_config,
                protocol_path=Path(directory) / "protocol.json",
                protocol_sha256="abc",
                best_epoch=1,
                best_selection_value=0.5,
            )
            with self.assertRaises(ValueError):
                load_tg_soft_teacher_checkpoint(
                    path,
                    self.model,
                    torch.device("cpu"),
                    expected_protocol_sha256="different",
                )
            incompatible = TGSoftTeacher(
                TGSoftTeacherConfig(
                    node_score_hidden_dim=16,
                    edge_score_hidden_dim=8,
                    signed_gnn_hidden_dim=8,
                    signed_gnn_layers=1,
                    classifier_hidden_dims=(8,),
                    dropout=0.0,
                )
            )
            with self.assertRaises(ValueError):
                load_tg_soft_teacher_checkpoint(path, incompatible, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
