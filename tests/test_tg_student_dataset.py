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
    TGHardStudentDataset,
    TGTeacherTarget,
    create_tg_hard_student_loader,
    save_tg_teacher_target,
)
from keysubgraph.features import (  # noqa: E402
    HardExportFeatureAdapter,
    TGTheoryFeatureStandardizer,
    save_hard_graph_cache,
)
from keysubgraph.theory import (  # noqa: E402
    SGWSequenceFeatures,
    TGSGWFeatureArtifact,
    save_tg_sgw_feature_artifact,
)


def _hard_cache(label=1, split="train"):
    def window(index, scale):
        return {
            "time_index": index,
            "time_start": float(index),
            "window_valid": True,
            "union_node_ids": [0, 1],
            "union_node_names": ["a", "b"],
            "union_community_labels": [0, 0],
            "union_edge_index": [[0, 1]],
            "union_original_edge_weights": [0.7 * scale],
            "subgraphs": [{
                "node_ids": [0, 1],
                "edge_index": [[0, 1]],
                "original_edge_weights": [0.7 * scale],
                "candidate_score": 0.8,
                "seed_node": 0,
            }],
        }
    return HardExportFeatureAdapter().build({
        "sample_key": "SITE/sample",
        "sample_id": "sample",
        "label": label,
        "split": split,
        "edge_presence_threshold": 0.0,
        "data_protocol_sha256": "protocol",
        "checkpoint_sha256": "teacher",
        "timepoints": [window(0, 1.0), window(1, 0.8)],
    })


def _theory(cache):
    sequence = SGWSequenceFeatures(
        h_core=torch.zeros(18),
        h_variation=torch.zeros(16),
        h_classification=torch.arange(34, dtype=torch.float32),
        transition_features=torch.zeros(1, 18),
        transition_mask=torch.ones(1, dtype=torch.bool),
        gw_solver_converged=(True,),
    )
    return TGSGWFeatureArtifact(
        cache.sample_key, cache.sample_id, cache.label, cache.split, sequence, True,
        cache.data_protocol_sha256, cache.teacher_checkpoint_sha256,
    )


class TGStudentDatasetTest(unittest.TestCase):
    def _write(self, root, hard=None, theory=None, teacher=None):
        hard = hard or _hard_cache()
        theory = theory or _theory(hard)
        teacher = teacher or TGTeacherTarget(
            hard.sample_key, hard.sample_id, hard.label, hard.split,
            torch.tensor([0.2, -0.1]), torch.arange(192, dtype=torch.float32),
            hard.data_protocol_sha256, hard.teacher_checkpoint_sha256,
        )
        for name in ("hard", "theory", "teacher"):
            (root / name).mkdir()
        save_hard_graph_cache(hard, root / "hard" / "sample.pt")
        save_tg_sgw_feature_artifact(theory, root / "theory" / "sample.pt")
        save_tg_teacher_target(teacher, root / "teacher" / "sample.pt")

    def test_strict_pairing_standardization_and_list_batch(self):
        scaler = TGTheoryFeatureStandardizer.fit(
            [torch.zeros(34), torch.arange(34, dtype=torch.float32)],
            data_protocol_sha256="protocol", teacher_checkpoint_sha256="teacher",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write(root)
            dataset = TGHardStudentDataset(
                root / "hard", root / "theory", root / "teacher", scaler, "train"
            )
            batch = next(iter(create_tg_hard_student_loader(dataset, 1, seed=7)))
        self.assertEqual(len(batch), 1)
        self.assertEqual(batch[0].sample_key, "SITE/sample")
        self.assertEqual(tuple(batch[0].standardized_theory_features.shape), (34,))
        self.assertTrue(torch.isfinite(batch[0].standardized_theory_features).all())

    def test_pairing_rejects_label_and_hash_mismatch(self):
        scaler = TGTheoryFeatureStandardizer.fit(
            [torch.zeros(34), torch.ones(34)],
            data_protocol_sha256="protocol", teacher_checkpoint_sha256="teacher",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hard = _hard_cache()
            target = TGTeacherTarget(
                hard.sample_key, hard.sample_id, 0, hard.split,
                torch.zeros(2), torch.zeros(192), "protocol", "teacher",
            )
            self._write(root, hard=hard, teacher=target)
            with self.assertRaisesRegex(ValueError, "label mismatch"):
                TGHardStudentDataset(
                    root / "hard", root / "theory", root / "teacher", scaler, "train"
                )

        wrong_scaler = dataclasses.replace(scaler, data_protocol_sha256="wrong")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write(root)
            with self.assertRaisesRegex(ValueError, "scaler protocol mismatch"):
                TGHardStudentDataset(
                    root / "hard", root / "theory", root / "teacher", wrong_scaler, "train"
                )


if __name__ == "__main__":
    unittest.main()
