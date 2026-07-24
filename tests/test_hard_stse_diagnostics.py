from __future__ import absolute_import, division, print_function

import unittest

import torch

from keysubgraph.data.graph_dataset import GraphSequenceBatch
from keysubgraph.hard_stse_diagnostics import (
    audit_hard_stse_output,
    representation_summary,
)
from keysubgraph.models.hard_stse_temporal_sgw import (
    HardSTSETemporalSGWClassifier,
)
from keysubgraph.models.hard_stse_types import HardSTSEConfig
from tests.test_full_graph_classifier import _sample


class HardSTSEDiagnosticsTest(unittest.TestCase):
    def test_hard_graph_audit_and_representation_statistics(self):
        torch.manual_seed(173)
        batch = GraphSequenceBatch(
            (_sample("audit-a", 0, 2), _sample("audit-b", 1, 3))
        )
        model = HardSTSETemporalSGWClassifier(
            HardSTSEConfig(
                variant="M2",
                selection_mode="learned",
                use_sgw=False,
                dropout=0.0,
            )
        )
        output = model(batch, epoch=30)
        audit = audit_hard_stse_output(batch, output)
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["sample_inventory"][1]["label"], 1)
        summary = representation_summary(output.neural_representation)
        self.assertEqual(summary["dimension"], 192)
        self.assertEqual(summary["sample_count"], 2)
        self.assertTrue(torch.isfinite(torch.tensor(
            summary["mean_feature_variance"]
        )))


if __name__ == "__main__":
    unittest.main()
