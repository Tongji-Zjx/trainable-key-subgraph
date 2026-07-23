from __future__ import absolute_import, division, print_function

import unittest

import torch

from keysubgraph.data.graph_dataset import GraphSequenceBatch
from keysubgraph.full_graph_diagnostics import (
    FullGraphRepresentationMonitor,
    summarize_full_graph_inputs,
    validate_full_graph_batch_alignment,
)
from keysubgraph.models import FullGraphClassifierConfig, FullGraphSequenceClassifier
from tests.test_full_graph_classifier import _sample


class _Assignment(object):
    def __init__(self, sample_key, label):
        self.sample_key = sample_key
        self.label = label


class FullGraphDiagnosticsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(91)
        self.samples = (_sample("a", 0, 2), _sample("b", 1, 3))
        self.batch = GraphSequenceBatch(self.samples)
        self.assignments = {
            sample.sample_key: _Assignment(sample.sample_key, sample.label)
            for sample in self.samples
        }

    def test_alignment_binds_labels_lengths_and_order(self):
        model = FullGraphSequenceClassifier(
            FullGraphClassifierConfig(
                encoder_type="sgg_bigru_proto",
                signed_gnn_layers=1,
                classifier_hidden_dims=(8,),
                gated_gnn_dropout=0.0,
                classifier_dropout=0.0,
            )
        )
        output = model(self.batch)
        records = validate_full_graph_batch_alignment(
            self.batch, output, self.assignments
        )
        self.assertEqual([row["sample_key"] for row in records], ["a", "b"])
        self.assertEqual([row["label"] for row in records], [0, 1])
        self.assertEqual([row["num_timepoints"] for row in records], [2, 3])

    def test_signed_input_summary_accepts_masks_and_counts_both_signs(self):
        summary = summarize_full_graph_inputs(self.samples)
        self.assertEqual(summary["sample_count"], 2)
        self.assertEqual(summary["timepoint_count"], 5)
        self.assertEqual(summary["validation_failure_count"], 0)
        self.assertGreater(summary["positive_edge_count"], 0)
        self.assertGreater(summary["negative_edge_count"], 0)
        self.assertGreater(summary["edge_density"]["mean"], 0.0)
        self.assertEqual(summary["empty_edge_timepoints"], 0)

    def test_monitor_reports_each_stage_with_nonzero_variance(self):
        model = FullGraphSequenceClassifier(
            FullGraphClassifierConfig(
                encoder_type="sgg_bigru_proto",
                signed_gnn_layers=1,
                classifier_hidden_dims=(8,),
                gated_gnn_dropout=0.0,
                classifier_dropout=0.0,
            )
        )
        model.eval()
        monitor = FullGraphRepresentationMonitor(model)
        with torch.no_grad():
            output = model(self.batch)
        monitor.add_model_output(output)
        monitor.close()
        summary = monitor.summary()
        expected = (
            "node_projection_graph_mean",
            "gated_gnn_layer_1_graph_mean",
            "window_embedding",
            "temporal_sequence_representation",
            "prototype_attention",
            "prototype_fused_representation",
            "classifier_linear_1",
            "classifier_linear_2",
            "logits",
            "positive_probability",
        )
        for name in expected:
            self.assertIn(name, summary)
            self.assertGreater(summary[name]["row_count"], 0)
        self.assertGreater(
            summary["final_representation"]["mean_feature_variance"], 0.0
        )


if __name__ == "__main__":
    unittest.main()
