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

from keysubgraph.features import (  # noqa: E402
    HardExportFeatureAdapter,
    TGTheoryFeatureStandardizer,
)
from keysubgraph.models import (  # noqa: E402
    TGHardClassifierConfig,
    TGHardSGWClassifier,
)
from keysubgraph.training import (  # noqa: E402
    load_tg_hard_student_checkpoint,
    save_tg_hard_student_checkpoint,
)


def _timepoint(index, weight_scale=1.0):
    return {
        "time_index": index,
        "time_start": float(index),
        "window_valid": True,
        "hard_union_available": True,
        "union_node_ids": [0, 1, 2],
        "union_node_names": ["a", "b", "c"],
        "union_community_labels": [0, 0, 1],
        "union_edge_index": [[0, 1], [1, 2]],
        "union_original_edge_weights": [0.8 * weight_scale, -0.5 * weight_scale],
        "subgraphs": [
            {
                "node_ids": [0, 1],
                "edge_index": [[0, 1]],
                "original_edge_weights": [0.8 * weight_scale],
                "candidate_score": 0.9,
                "seed_node": 0,
            },
            {
                "node_ids": [1, 2],
                "edge_index": [[1, 2]],
                "original_edge_weights": [-0.5 * weight_scale],
                "candidate_score": 0.7,
                "seed_node": 2,
            },
        ],
    }


def _cache():
    payload = {
        "sample_key": "SITE/sample",
        "sample_id": "sample",
        "label": 1,
        "split": "train",
        "edge_presence_threshold": 0.0,
        "data_protocol_sha256": "protocol",
        "checkpoint_sha256": "teacher",
        "timepoints": [_timepoint(0), _timepoint(1, 0.8)],
    }
    return HardExportFeatureAdapter().build(payload)


class TGHardClassifierTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.cache = _cache()
        self.config = TGHardClassifierConfig(
            signed_gnn_hidden_dim=8,
            signed_gnn_layers=1,
            classifier_hidden_dims=(16,),
            dropout=0.0,
        )

    def test_training_only_standardizer_and_floor(self):
        features = [torch.zeros(34), torch.arange(34, dtype=torch.float32)]
        with self.assertRaises(ValueError):
            TGTheoryFeatureStandardizer.fit(features, fit_split="validation")
        scaler = TGTheoryFeatureStandardizer.fit(
            features,
            fit_split="train",
            standard_deviation_floor=1.0e-4,
            data_protocol_sha256="protocol",
            teacher_checkpoint_sha256="teacher",
        )
        transformed = scaler.transform(torch.stack(features))
        self.assertLess(float(transformed.mean(dim=0).abs().max()), 1.0e-6)
        self.assertGreaterEqual(min(scaler.scale), 1.0e-4)

    def test_forward_is_226_dimensional_and_backpropagates(self):
        model = TGHardSGWClassifier(self.config)
        theory = torch.randn(1, 34)
        output = model((self.cache,), theory)
        self.assertEqual(tuple(output.logits.shape), (1, 2))
        self.assertEqual(tuple(output.neural_representation.shape), (1, 192))
        self.assertEqual(tuple(output.final_representation.shape), (1, 226))
        self.assertTrue(torch.equal(output.final_representation[:, 192:], theory))
        loss = torch.nn.functional.cross_entropy(output.logits, torch.tensor([1]))
        loss.backward()
        self.assertGreater(float(model.graph_encoder.layers[0].update[0].weight.grad.norm()), 0.0)
        self.assertGreater(float(model.classifier[-1].weight.grad.norm()), 0.0)

    def test_subgraph_order_and_masked_padding_do_not_change_prediction(self):
        model = TGHardSGWClassifier(self.config).eval()
        reversed_windows = tuple(
            dataclasses.replace(window, subgraphs=tuple(reversed(window.subgraphs)))
            for window in self.cache.windows
        )
        reordered = dataclasses.replace(self.cache, windows=reversed_windows)
        padded = dataclasses.replace(
            self.cache,
            windows=self.cache.windows + (None,),
            time_values=self.cache.time_values + (2.0,),
            time_mask=self.cache.time_mask + (False,),
        )
        theory = torch.randn(1, 34)
        with torch.no_grad():
            original = model((self.cache,), theory).logits
            permuted = model((reordered,), theory).logits
            with_padding = model((padded,), theory).logits
        self.assertTrue(torch.allclose(original, permuted, atol=1.0e-6))
        self.assertTrue(torch.allclose(original, with_padding, atol=1.0e-6))

    def test_checkpoint_round_trip_and_binding(self):
        scaler = TGTheoryFeatureStandardizer.fit(
            [torch.zeros(34), torch.ones(34)],
            data_protocol_sha256="protocol",
            teacher_checkpoint_sha256="teacher",
        )
        model = TGHardSGWClassifier(self.config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "student.pt"
            save_tg_hard_student_checkpoint(
                path,
                model,
                scaler,
                epoch=1,
                protocol_sha256="protocol",
                teacher_checkpoint_sha256="teacher",
                candidate_scaler_sha256="candidate-scaler",
                theory_scaler_sha256="theory-scaler",
                optimizer=optimizer,
                history=({"epoch": 1},),
            )
            restored = TGHardSGWClassifier(self.config)
            payload = load_tg_hard_student_checkpoint(
                path,
                restored,
                torch.device("cpu"),
                expected_protocol_sha256="protocol",
                expected_teacher_checkpoint_sha256="teacher",
            )
            with self.assertRaises(ValueError):
                load_tg_hard_student_checkpoint(
                    path,
                    TGHardSGWClassifier(self.config),
                    torch.device("cpu"),
                    expected_protocol_sha256="wrong",
                )
        self.assertEqual(payload["stage"], "hard_student")
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, restored.state_dict()[name]))


if __name__ == "__main__":
    unittest.main()
