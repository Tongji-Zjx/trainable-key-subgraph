from __future__ import absolute_import, division, print_function

import tempfile
import unittest
from pathlib import Path

import torch

from keysubgraph.data.exact_stse_dataset import ExactSTSEBatch
from keysubgraph.models.dual_exact_sgw import (
    DualExactSGWBranch,
    DualSGWFeatureRecord,
    load_dual_sgw_feature_record,
    save_dual_sgw_feature_record,
)
from keysubgraph.models.dual_hard_sgw_selector import DualHardSGWSelector
from tests.test_exact_stse_model import _exact_sample


class DualExactSGWTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(317)
        self.batch = ExactSTSEBatch(
            (_exact_sample("exact-sgw", 1, 2),)
        )
        self.selected = DualHardSGWSelector()(
            self.batch, selection_mode="full"
        )

    def test_exact_branch_is_34_dimensional_non_temporal_and_detached(self):
        output = DualExactSGWBranch()(
            self.batch, self.selected.hard_windows
        )
        self.assertEqual(tuple(output.core.shape), (1, 18))
        self.assertEqual(tuple(output.variation.shape), (1, 16))
        self.assertEqual(tuple(output.representation.shape), (1, 34))
        self.assertFalse(output.representation.requires_grad)
        self.assertTrue(output.diagnostics["is_exact_gw"])
        self.assertTrue(output.diagnostics["exact_features_detached"])

    def test_feature_artifact_round_trip_is_immutable(self):
        output = DualExactSGWBranch()(
            self.batch, self.selected.hard_windows
        )
        record = DualSGWFeatureRecord(
            sample_key=self.batch[0].sample_key,
            label=1,
            split="train",
            selection_mode="full",
            selection_seed=42,
            core=output.core[0],
            variation=output.variation[0],
            representation=output.representation[0],
            transition_mask=output.transition_mask[0],
            protocol_sha256="protocol",
            selector_checkpoint_sha256="none",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "feature.pt"
            save_dual_sgw_feature_record(record, path)
            loaded = load_dual_sgw_feature_record(path)
            self.assertEqual(loaded.sample_key, record.sample_key)
            self.assertTrue(
                torch.equal(
                    loaded.representation, record.representation
                )
            )
            with self.assertRaises(FileExistsError):
                save_dual_sgw_feature_record(record, path)


if __name__ == "__main__":
    unittest.main()
