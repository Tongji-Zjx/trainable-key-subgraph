from __future__ import absolute_import, division, print_function

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from keysubgraph.analysis.dual_classification_bottleneck import (
    FEATURE_BLOCKS,
    analyze_dual_classification_bottleneck,
    write_dual_classification_bottleneck_artifacts,
)


class DualClassificationBottleneckTest(unittest.TestCase):
    def _analysis(self):
        generator = np.random.RandomState(7)
        train_labels = np.asarray([0, 1] * 6, dtype=np.int64)
        validation_labels = np.asarray([0, 1] * 4, dtype=np.int64)
        train_proxy = generator.normal(size=(12, 34))
        train_exact = train_proxy + 0.1 * generator.normal(size=(12, 34))
        validation_proxy = generator.normal(size=(8, 34))
        validation_exact = validation_proxy + 0.1 * generator.normal(
            size=(8, 34)
        )
        train_base = np.linspace(0.2, 0.8, 12)
        validation_base = np.asarray(
            [0.2, 0.7, 0.4, 0.6, 0.3, 0.8, 0.5, 0.55]
        )
        train_paths = {
            "proxy_all": train_base,
            "exact_all": train_base[::-1],
        }
        validation_paths = {
            "proxy_all": validation_base,
            "exact_all": validation_base[::-1],
        }
        for index, (name, _, _) in enumerate(FEATURE_BLOCKS):
            train_paths["replace_{}".format(name)] = np.clip(
                train_base + 0.01 * index, 0.0, 1.0
            )
            validation_paths["replace_{}".format(name)] = np.clip(
                validation_base + 0.01 * index, 0.0, 1.0
            )
        train_layers = {
            "proxy_scaled_input": train_proxy,
            "proxy_hidden_activation": generator.normal(size=(12, 16)),
            "proxy_logits": generator.normal(size=(12, 2)),
        }
        validation_layers = {
            "proxy_scaled_input": validation_proxy,
            "proxy_hidden_activation": generator.normal(size=(8, 16)),
            "proxy_logits": generator.normal(size=(8, 2)),
        }
        permutation_aucs = {
            name: (0.50 + 0.01 * index, 0.52 + 0.01 * index)
            for index, (name, _, _) in enumerate(FEATURE_BLOCKS)
        }
        stability = [
            {
                "sample_key": "v0",
                "split": "validation",
                "label": 0,
                "time_index": 0,
                "node_score_margin": 0.02,
                "edge_score_margin": 0.01,
                "node_perturbation_jaccard": 0.9,
                "edge_perturbation_jaccard": 0.8,
                "temporal_node_jaccard": None,
                "temporal_edge_jaccard": None,
            }
        ]
        return analyze_dual_classification_bottleneck(
            train_labels=train_labels,
            validation_labels=validation_labels,
            train_proxy=train_proxy,
            train_exact=train_exact,
            validation_proxy=validation_proxy,
            validation_exact=validation_exact,
            path_probabilities={
                "train": train_paths,
                "validation": validation_paths,
            },
            permutation_aucs=permutation_aucs,
            layer_representations={
                "train": train_layers,
                "validation": validation_layers,
            },
            selector_stability_rows=stability,
        )

    def test_all_diagnostic_families_are_present(self):
        analysis = self._analysis()
        self.assertEqual(len(analysis["block_rows"]), 4)
        self.assertEqual(len(analysis["feature_rows"]), 34)
        self.assertEqual(len(analysis["layer_rows"]), 6)
        self.assertEqual(len(analysis["drift_rows"]), 3)
        self.assertEqual(len(analysis["selector_stability_rows"]), 1)
        self.assertIn(
            "effective_rank_retention", analysis["summary"]
        )
        self.assertIn(
            "node_perturbation_jaccard",
            analysis["summary"]["selector_stability"],
        )

    def test_artifacts_are_complete_and_immutable(self):
        analysis = self._analysis()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "diagnostic"
            paths = write_dual_classification_bottleneck_artifacts(
                output,
                analysis,
                {"read_only_frozen_models": True},
            )
            self.assertEqual(len(paths), 8)
            self.assertTrue(all(path.is_file() for path in paths.values()))
            payload = json.loads(
                paths["summary_json"].read_text(encoding="utf-8")
            )
            self.assertFalse(payload["provenance"].get("test_split_used", False))
            with self.assertRaises(FileExistsError):
                write_dual_classification_bottleneck_artifacts(
                    output, analysis, {}
                )

    def test_invalid_feature_shape_is_rejected(self):
        analysis = self._analysis()
        del analysis
        with self.assertRaisesRegex(ValueError, "finite"):
            analyze_dual_classification_bottleneck(
                train_labels=(0, 1),
                validation_labels=(0, 1),
                train_proxy=np.zeros((2, 33)),
                train_exact=np.zeros((2, 34)),
                validation_proxy=np.zeros((2, 34)),
                validation_exact=np.zeros((2, 34)),
                path_probabilities={
                    "train": {"proxy_all": (0.1, 0.9)},
                    "validation": {"proxy_all": (0.1, 0.9)},
                },
                permutation_aucs={
                    name: (0.5,) for name, _, _ in FEATURE_BLOCKS
                },
                layer_representations={
                    "train": {
                        "proxy_scaled_input": np.zeros((2, 34)),
                        "proxy_hidden_activation": np.zeros((2, 4)),
                        "proxy_logits": np.zeros((2, 2)),
                    },
                    "validation": {
                        "proxy_scaled_input": np.zeros((2, 34)),
                        "proxy_hidden_activation": np.zeros((2, 4)),
                        "proxy_logits": np.zeros((2, 2)),
                    },
                },
                selector_stability_rows=(),
            )


if __name__ == "__main__":
    unittest.main()
