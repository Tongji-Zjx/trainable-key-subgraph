from __future__ import absolute_import, division, print_function

import sys
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.models import (  # noqa: E402
    TGHardClassifierOutput,
    TGHardStudentLossConfig,
    compute_tg_hard_student_loss,
)


def _output(logits, representation):
    return TGHardClassifierOutput(
        logits=logits,
        final_representation=torch.zeros(logits.shape[0], 226),
        neural_representation=representation,
        projected_neural_representation=representation,
        theory_representation=torch.zeros(logits.shape[0], 34),
        encoded_windows=torch.zeros(logits.shape[0], 2, 96),
        time_mask=torch.ones(logits.shape[0], 2, dtype=torch.bool),
    )


class TGHardStudentLossTest(unittest.TestCase):
    def test_kd_matches_canonical_temperature_squared_formula(self):
        student_logits = torch.tensor([[0.2, -0.4], [0.1, 0.8]], requires_grad=True)
        teacher_logits = torch.tensor([[0.7, -0.1], [-0.2, 1.2]], requires_grad=True)
        representation = torch.randn(2, 192, requires_grad=True)
        config = TGHardStudentLossConfig(
            classification_weight=0.0,
            knowledge_distillation_weight=1.0,
            representation_distillation_weight=0.0,
            supervised_contrastive_weight=0.0,
            knowledge_distillation_temperature=2.0,
        )
        loss = compute_tg_hard_student_loss(
            _output(student_logits, representation),
            torch.tensor([0, 1]),
            teacher_logits,
            representation.detach(),
            config,
        )
        expected = 4.0 * F.kl_div(
            F.log_softmax(student_logits / 2.0, dim=-1),
            F.softmax(teacher_logits.detach() / 2.0, dim=-1),
            reduction="batchmean",
        )
        self.assertTrue(torch.allclose(loss.knowledge_distillation, expected))
        loss.total.backward()
        self.assertIsNotNone(student_logits.grad)
        self.assertIsNone(teacher_logits.grad)

    def test_all_terms_are_finite_and_representation_receives_gradient(self):
        logits = torch.randn(4, 2, requires_grad=True)
        student = torch.randn(4, 192, requires_grad=True)
        teacher = torch.randn(4, 192, requires_grad=True)
        loss = compute_tg_hard_student_loss(
            _output(logits, student),
            torch.tensor([0, 0, 1, 1]),
            torch.randn(4, 2, requires_grad=True),
            teacher,
        )
        self.assertTrue(torch.isfinite(loss.total))
        loss.total.backward()
        self.assertGreater(float(student.grad.norm()), 0.0)
        self.assertIsNone(teacher.grad)

    def test_supcon_is_safe_without_a_positive_pair(self):
        logits = torch.randn(2, 2, requires_grad=True)
        representation = torch.randn(2, 192, requires_grad=True)
        loss = compute_tg_hard_student_loss(
            _output(logits, representation),
            torch.tensor([0, 1]),
            torch.zeros(2, 2),
            torch.zeros(2, 192),
        )
        self.assertEqual(float(loss.supervised_contrastive), 0.0)


if __name__ == "__main__":
    unittest.main()
