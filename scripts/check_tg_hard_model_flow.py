"""Run one hard-student forward/backward/checkpoint cycle on cached samples."""

from __future__ import absolute_import, print_function

import argparse
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.data_split import file_sha256  # noqa: E402
from keysubgraph.features import (  # noqa: E402
    TGTheoryFeatureStandardizer,
    load_hard_graph_cache,
)
from keysubgraph.models import TGHardSGWClassifier  # noqa: E402
from keysubgraph.models import (  # noqa: E402
    TGHardStudentLossConfig,
    compute_tg_hard_student_loss,
)
from keysubgraph.data.tg_student_dataset import load_tg_teacher_target  # noqa: E402
from keysubgraph.theory import load_tg_sgw_feature_artifact  # noqa: E402
from keysubgraph.training import (  # noqa: E402
    load_tg_hard_student_checkpoint,
    save_tg_hard_student_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hard-cache", type=Path, required=True)
    parser.add_argument("--theory-feature", type=Path, required=True)
    parser.add_argument("--theory-scaler", type=Path, required=True)
    parser.add_argument("--candidate-scaler", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-target", type=Path)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    cache = load_hard_graph_cache(args.hard_cache)
    theory_artifact = load_tg_sgw_feature_artifact(args.theory_feature)
    scaler = TGTheoryFeatureStandardizer.load(args.theory_scaler)
    if cache.sample_key != theory_artifact.sample_key:
        raise ValueError("hard cache and theory feature sample mismatch")
    if cache.data_protocol_sha256 != theory_artifact.data_protocol_sha256:
        raise ValueError("hard cache and theory feature protocol mismatch")
    if scaler.data_protocol_sha256 != cache.data_protocol_sha256:
        raise ValueError("theory scaler protocol mismatch")
    if scaler.teacher_checkpoint_sha256 != cache.teacher_checkpoint_sha256:
        raise ValueError("theory scaler teacher mismatch")
    device = torch.device(args.device)
    model = TGHardSGWClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    raw_theory = theory_artifact.features.h_classification.unsqueeze(0).to(device)
    standardized = scaler.transform(raw_theory)
    output = model((cache,), standardized)
    labels = torch.tensor([cache.label], dtype=torch.long, device=device)
    if args.teacher_target is not None:
        target = load_tg_teacher_target(args.teacher_target)
        if target.sample_key != cache.sample_key or target.label != cache.label:
            raise ValueError("teacher target does not match the hard sample")
        if target.data_protocol_sha256 != cache.data_protocol_sha256:
            raise ValueError("teacher target protocol mismatch")
        if target.teacher_checkpoint_sha256 != cache.teacher_checkpoint_sha256:
            raise ValueError("teacher target checkpoint mismatch")
        loss_parts = compute_tg_hard_student_loss(
            output,
            labels,
            target.logits.unsqueeze(0),
            target.representation.unsqueeze(0),
            TGHardStudentLossConfig(supervised_contrastive_weight=0.0),
        )
        loss = loss_parts.total
    else:
        loss_parts = None
        loss = torch.nn.functional.cross_entropy(output.logits, labels)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("hard student smoke loss is non-finite")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    save_tg_hard_student_checkpoint(
        args.output_checkpoint,
        model,
        scaler,
        epoch=1,
        protocol_sha256=cache.data_protocol_sha256,
        teacher_checkpoint_sha256=cache.teacher_checkpoint_sha256,
        candidate_scaler_sha256=file_sha256(args.candidate_scaler),
        theory_scaler_sha256=file_sha256(args.theory_scaler),
        optimizer=optimizer,
        history=({"epoch": 1, "loss": float(loss.detach().cpu())},),
    )
    restored = TGHardSGWClassifier().to(device)
    load_tg_hard_student_checkpoint(
        args.output_checkpoint,
        restored,
        device,
        expected_protocol_sha256=cache.data_protocol_sha256,
        expected_teacher_checkpoint_sha256=cache.teacher_checkpoint_sha256,
    )
    print(json.dumps({
        "sample_key": cache.sample_key,
        "valid_windows": cache.num_valid_windows,
        "valid_transitions": int(theory_artifact.features.transition_mask.sum()),
        "loss": float(loss.detach().cpu()),
        "classification_loss": (
            float(loss_parts.classification.detach().cpu()) if loss_parts is not None else None
        ),
        "knowledge_distillation_loss": (
            float(loss_parts.knowledge_distillation.detach().cpu()) if loss_parts is not None else None
        ),
        "representation_distillation_loss": (
            float(loss_parts.representation_distillation.detach().cpu()) if loss_parts is not None else None
        ),
        "gradient_norm": float(gradient_norm.detach().cpu()),
        "neural_dim": int(output.neural_representation.shape[1]),
        "theory_dim": int(output.theory_representation.shape[1]),
        "final_dim": int(output.final_representation.shape[1]),
        "checkpoint": str(args.output_checkpoint.resolve()),
        "status": "ok",
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
