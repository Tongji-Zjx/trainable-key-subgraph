"""Tests for the validation-only SV V1 candidate gate."""

from __future__ import absolute_import, division, print_function

import importlib.util
import types
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "select_sv_v1_candidate.py"
SPEC = importlib.util.spec_from_file_location(
    "select_sv_v1_candidate", str(SCRIPT)
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _evaluation(final_auc, gin_auc):
    return {
        "metrics": {
            "balanced_accuracy": {
                "roc_auc": final_auc,
                "composite_auc": final_auc,
            }
        },
        "branch_metrics": {
            "gin": {"roc_auc": gin_auc},
            "static_spectral": {"roc_auc": 0.62},
        },
        "fusion_regret": {"roc_auc": 0.62 - final_auc},
    }


def _diagnostic(variant, rank, entropy=None, masked_auc=None):
    validation = {
        "representations": {
            "gin_representation": {
                "normalized_effective_rank": rank,
                "mean_pairwise_cosine": 0.90,
            },
            "gin_projection": {
                "normalized_effective_rank": 0.3,
                "mean_pairwise_cosine": 0.7,
            },
        }
    }
    if entropy is not None:
        validation["attention"] = {
            "normalized_entropy": {"median": entropy}
        }
        validation["attention_ablation_metrics"] = {
            "roc_auc": masked_auc
        }
    return {
        "variant": variant,
        "provenance": {"protocol_sha256": "same"},
        "checks": {
            "gin_representation_not_low_rank": True,
            "gin_projection_not_nearly_collinear": True,
        },
        "splits": {"validation": validation},
    }


def _args():
    return types.SimpleNamespace(
        maximum_fusion_regret=0.01,
        minimum_gin_normalized_rank=0.10,
        maximum_gin_projection_cosine=0.995,
        maximum_attention_entropy=0.99,
        minimum_attention_ablation_gain=0.0,
        minimum_v1b_gin_auc_gain=0.0,
    )


class SVV1CandidateSelectionTest(unittest.TestCase):
    def test_uniform_redundant_attention_falls_back_to_v1a(self):
        result = MODULE.select_candidate(
            _evaluation(0.63, 0.55),
            _diagnostic(
                "signed_gin_static_anchor_residual", 0.11
            ),
            _evaluation(0.63, 0.60),
            _diagnostic(
                "signed_gin_static_anchor_residual_attention",
                0.13,
                entropy=0.995,
                masked_auc=0.63,
            ),
            _args(),
        )
        self.assertEqual(result["selected_candidate"], "v1a")
        self.assertFalse(
            result["v1b_retention_checks"][
                "attention_not_nearly_uniform"
            ]
        )
        self.assertFalse(
            result["v1b_retention_checks"][
                "attention_ablation_has_positive_gain"
            ]
        )
        self.assertFalse(result["test_used"])

    def test_nonredundant_attention_can_select_v1b(self):
        result = MODULE.select_candidate(
            _evaluation(0.63, 0.55),
            _diagnostic(
                "signed_gin_static_anchor_residual", 0.11
            ),
            _evaluation(0.65, 0.61),
            _diagnostic(
                "signed_gin_static_anchor_residual_attention",
                0.14,
                entropy=0.90,
                masked_auc=0.64,
            ),
            _args(),
        )
        self.assertEqual(result["selected_candidate"], "v1b")
        self.assertTrue(
            all(result["v1b_retention_checks"].values())
        )


if __name__ == "__main__":
    unittest.main()
