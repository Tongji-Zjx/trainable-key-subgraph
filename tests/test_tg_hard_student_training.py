from __future__ import absolute_import, division, print_function

import dataclasses
import sys
import tempfile
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.tg_student_dataset import (  # noqa: E402
    TGHardStudentSample,
    TGTeacherTarget,
)
from keysubgraph.features import HardExportFeatureAdapter, TGTheoryFeatureStandardizer  # noqa: E402
from keysubgraph.models import (  # noqa: E402
    TGHardClassifierConfig,
    TGHardSGWClassifier,
    TGHardStudentLossConfig,
    TGSoftTeacher,
    TGSoftTeacherConfig,
)
from keysubgraph.theory import SGWSequenceFeatures, TGSGWFeatureArtifact  # noqa: E402
from keysubgraph.training import (  # noqa: E402
    TGHardStudentTrainingConfig,
    build_tg_hard_student_optimizer,
    initialize_student_graph_encoder,
    run_tg_hard_student_epoch,
    set_student_graph_encoder_trainable,
    train_tg_hard_student,
)


def _sample(index, label, split):
    def window(time_index, scale):
        return {
            "time_index": time_index,
            "time_start": float(time_index),
            "window_valid": True,
            "union_node_ids": [0, 1, 2],
            "union_node_names": ["a", "b", "c"],
            "union_community_labels": [0, 0, 1],
            "union_edge_index": [[0, 1], [1, 2]],
            "union_original_edge_weights": [0.8 * scale, -0.4 * scale],
            "subgraphs": [{
                "node_ids": [0, 1], "edge_index": [[0, 1]],
                "original_edge_weights": [0.8 * scale],
                "candidate_score": 0.7, "seed_node": 0,
            }],
        }
    cache = HardExportFeatureAdapter().build({
        "sample_key": "SITE/sample{}".format(index),
        "sample_id": "sample{}".format(index),
        "label": label, "split": split, "edge_presence_threshold": 0.0,
        "data_protocol_sha256": "protocol", "checkpoint_sha256": "teacher",
        "timepoints": [window(0, 1.0 + 0.05 * index), window(1, 0.8)],
    })
    theory_vector = torch.arange(34, dtype=torch.float32) * (index + 1.0)
    features = SGWSequenceFeatures(
        h_core=theory_vector[:18], h_variation=theory_vector[18:],
        h_classification=theory_vector,
        transition_features=torch.zeros(1, 18),
        transition_mask=torch.ones(1, dtype=torch.bool),
        gw_solver_converged=(True,),
    )
    artifact = TGSGWFeatureArtifact(
        cache.sample_key, cache.sample_id, label, split, features, True,
        "protocol", "teacher",
    )
    target = TGTeacherTarget(
        cache.sample_key, cache.sample_id, label, split,
        torch.tensor([0.2, -0.2]) if label == 0 else torch.tensor([-0.2, 0.2]),
        torch.linspace(0.0, 1.0, 192) * (index + 1.0), "protocol", "teacher",
    )
    return cache, artifact, target


class TGHardStudentTrainingTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.student_config = TGHardClassifierConfig(
            signed_gnn_hidden_dim=8, signed_gnn_layers=1,
            classifier_hidden_dims=(16,), dropout=0.0,
        )
        self.teacher_config = TGSoftTeacherConfig(
            signed_gnn_hidden_dim=8, signed_gnn_layers=1, dropout=0.0,
        )
        raw = [_sample(index, index % 2, "train") for index in range(4)]
        self.scaler = TGTheoryFeatureStandardizer.fit(
            [item[1].features.h_classification for item in raw],
            data_protocol_sha256="protocol", teacher_checkpoint_sha256="teacher",
        )
        self.train_samples = tuple(
            TGHardStudentSample(
                cache, artifact, target,
                self.scaler.transform(artifact.features.h_classification),
            ) for cache, artifact, target in raw
        )
        validation_raw = [_sample(index + 10, index % 2, "validation") for index in range(2)]
        self.validation_samples = tuple(
            TGHardStudentSample(
                cache, artifact, target,
                self.scaler.transform(artifact.features.h_classification),
            ) for cache, artifact, target in validation_raw
        )

    def test_teacher_initialization_and_freeze_then_unfreeze(self):
        teacher = TGSoftTeacher(self.teacher_config)
        student = TGHardSGWClassifier(self.student_config)
        initialize_student_graph_encoder(student, teacher)
        for name, value in teacher.graph_encoder.state_dict().items():
            self.assertTrue(torch.equal(value, student.graph_encoder.state_dict()[name]))
        config = TGHardStudentTrainingConfig(epochs=2, frozen_graph_epochs=1)
        optimizer = build_tg_hard_student_optimizer(student, config)
        before = {name: value.clone() for name, value in student.graph_encoder.state_dict().items()}
        set_student_graph_encoder_trainable(student, False)
        run_tg_hard_student_epoch(
            student, [self.train_samples], torch.device("cpu"),
            TGHardStudentLossConfig(supervised_contrastive_weight=0.0),
            optimizer=optimizer, class_weights=torch.ones(2),
        )
        for name, value in before.items():
            self.assertTrue(torch.equal(value, student.graph_encoder.state_dict()[name]))
        set_student_graph_encoder_trainable(student, True)
        run_tg_hard_student_epoch(
            student, [self.train_samples], torch.device("cpu"),
            TGHardStudentLossConfig(supervised_contrastive_weight=0.0),
            optimizer=optimizer, class_weights=torch.ones(2),
        )
        self.assertTrue(any(
            not torch.equal(value, student.graph_encoder.state_dict()[name])
            for name, value in before.items()
        ))

    def test_full_trainer_saves_validation_threshold_and_resumes_contract(self):
        teacher = TGSoftTeacher(self.teacher_config)
        student = TGHardSGWClassifier(self.student_config)
        training = TGHardStudentTrainingConfig(
            epochs=2, frozen_graph_epochs=1, early_stopping_patience=2,
            max_train_batches=1, max_validation_batches=1,
        )
        with tempfile.TemporaryDirectory() as temporary:
            result = train_tg_hard_student(
                student, [self.train_samples], [self.validation_samples],
                [item.label for item in self.train_samples], torch.device("cpu"),
                TGHardStudentLossConfig(supervised_contrastive_weight=0.0), training,
                Path(temporary), self.scaler, "protocol", "teacher", "candidate", "theory",
                teacher_model=teacher,
            )
            try:
                checkpoint = torch.load(
                    str(result["last_checkpoint"]), map_location="cpu", weights_only=False
                )
            except TypeError:
                checkpoint = torch.load(str(result["last_checkpoint"]), map_location="cpu")
        self.assertEqual(result["epochs_completed"], 2)
        self.assertIn("classification_threshold", checkpoint)
        self.assertTrue(0.0 <= checkpoint["classification_threshold"] <= 1.0)
        self.assertTrue(checkpoint["teacher_encoder_initialized"])
        self.assertEqual(checkpoint["history"][0]["stage"], "frozen_graph_encoder")
        self.assertEqual(checkpoint["history"][1]["stage"], "fine_tune")


if __name__ == "__main__":
    unittest.main()
