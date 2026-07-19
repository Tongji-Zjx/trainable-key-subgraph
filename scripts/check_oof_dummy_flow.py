"""CPU-only end-to-end smoke check for the frozen OOF confirmatory framework."""

from __future__ import absolute_import, print_function

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.crossfit.audit import (  # noqa: E402
    audit_fold_assignments, audit_perturbation_plan, audit_run_plan,
)
from keysubgraph.crossfit.oof_statistics import (  # noqa: E402
    aggregate_subjects, bootstrap_subject_mean, compute_model_contrasts,
    compute_perturbation_contrasts, dose_slope,
)
from keysubgraph.data.baseline_collate import BaselineBatch  # noqa: E402
from keysubgraph.models.baseline_classifier import (  # noqa: E402
    BaselineModelConfig, SignedSequenceBaseline,
)


def _batch(feature_offset=0.0):
    torch.manual_seed(901)
    features = torch.randn(3, 4, 12) + feature_offset
    adjacency = torch.zeros(3, 4, 4)
    adjacency[:, 0, 1] = adjacency[:, 1, 0] = 0.6
    adjacency[:, 1, 2] = adjacency[:, 2, 1] = -0.4
    node_mask = torch.ones(3, 4, dtype=torch.bool)
    return BaselineBatch(
        node_features=features, adjacency=adjacency,
        edge_mask=adjacency.abs() > 0.0, node_mask=node_mask,
        subgraph_to_window=torch.tensor([0, 1, 2]),
        window_to_sample=torch.tensor([0, 0, 1]),
        window_time_index=torch.tensor([0, 1, 0]),
        window_subgraph_count=torch.ones(3, dtype=torch.long),
        window_structural_features=torch.zeros(3, 11),
        window_structural_mask=torch.ones(3, 11, dtype=torch.bool),
        window_index=torch.tensor([[0, 1], [2, -1]]),
        time_mask=torch.tensor([[True, True], [True, False]]),
        labels=torch.tensor([0, 1]), sample_keys=("S/a", "S/b"),
        sample_ids=("a", "b"), subject_ids=("u1", "u2"), sites=("S", "S"),
    )


def _load(name):
    with (PROJECT_ROOT / "configs/crossfit" / name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    torch.manual_seed(902)
    probabilities = {}
    for variant, source, encoder_type in (
        ("A", "key", "signed"), ("B", "random", "signed"),
        ("C", "key", "node_only"), ("D", "random", "node_only"),
    ):
        model = SignedSequenceBaseline(BaselineModelConfig(
            encoder_type=encoder_type, node_hidden_dim=8, fusion_dim=12,
            gru_hidden_dim=10, classifier_hidden_dim=6,
            signed_gnn_layers=1, signed_gnn_dropout=0.0,
            classifier_dropout=0.0, history_mode="independent_bag",
        ))
        batch = _batch(0.0 if source == "key" else 0.15)
        output = model(batch)
        loss = F.cross_entropy(output.logits, batch.labels)
        loss.backward()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("dummy A-D loss is non-finite")
        probabilities[variant] = torch.softmax(output.logits.detach(), dim=-1)[:, 1]
    model_rows = []
    for sample_index, (sample_key, subject_id, label) in enumerate((
        ("S/a", "u1", 0), ("S/b", "u2", 1)
    )):
        for variant in ("A", "B", "C", "D"):
            model_rows.append({
                "outer_fold": 0, "model_seed": 42, "sample_key": sample_key,
                "subject_id": subject_id, "session_id": "1", "label": label,
                "variant": variant,
                "class_1_probability": float(probabilities[variant][sample_index]),
            })
    contrasts = compute_model_contrasts(model_rows)
    subjects = aggregate_subjects(contrasts, ("dsc", "seg", "tpa"))
    tpa = bootstrap_subject_mean(subjects, "tpa", repeats=100, seed=42)

    perturbation_rows = []
    for sample_key, subject_id, label in (("S/a", "u1", 0), ("S/b", "u2", 1)):
        perturbation_rows.append({
            "outer_fold": 0, "model_seed": 42, "sample_key": sample_key,
            "subject_id": subject_id, "session_id": "1", "label": label,
            "dose": 0.0, "mode": "none", "repeat_index": None,
            "class_1_probability": 0.45 if label == 0 else 0.55,
        })
        for dose in (0.25, 0.50):
            perturbation_rows.append({
                "outer_fold": 0, "model_seed": 42, "sample_key": sample_key,
                "subject_id": subject_id, "session_id": "1", "label": label,
                "dose": dose, "mode": "targeted", "repeat_index": None,
                "class_1_probability": 0.5,
            })
            for repeat in range(5):
                perturbation_rows.append({
                    "outer_fold": 0, "model_seed": 42, "sample_key": sample_key,
                    "subject_id": subject_id, "session_id": "1", "label": label,
                    "dose": dose, "mode": "random", "repeat_index": repeat,
                    "class_1_probability": 0.47 if label == 0 else 0.53,
                })
    perturbations = compute_perturbation_contrasts(perturbation_rows)
    dose_subjects = aggregate_subjects(
        perturbations, ("targeted_damage", "random_damage", "dose_contrast")
    )
    slopes = dose_slope(dose_subjects)
    audits = {
        "folds": audit_fold_assignments(_load("fold_assignments.json")),
        "runs": audit_run_plan(_load("oof_run_plan.json")),
        "perturbations": audit_perturbation_plan(_load("perturbation_inference_plan.json")),
    }
    result = {
        "status": "ok", "device": "cpu", "variants_checked": 4,
        "sample_contrast_count": len(contrasts),
        "subject_count": len(subjects), "dose_slope_subject_count": len(slopes),
        "dummy_tpa_mean": tpa["mean"], "audits": audits,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
