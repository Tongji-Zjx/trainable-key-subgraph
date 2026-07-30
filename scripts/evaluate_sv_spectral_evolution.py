"""Evaluate frozen S+E with its validation threshold and optional time shuffle."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.data_split import file_sha256  # noqa: E402
from keysubgraph.data.sv_spectral_evolution import (  # noqa: E402
    SVSpectralEvolutionDataset,
    create_sv_spectral_evolution_loader,
)
from keysubgraph.models.sv_signed_gin import (  # noqa: E402
    SVSignedGINClassifier,
    SVSignedGINConfig,
)
from keysubgraph.models.sv_spectral_evolution import (  # noqa: E402
    SV_SPECTRAL_EVOLUTION_VARIANT,
    SVSpectralEvolutionClassifier,
    SVSpectralEvolutionConfig,
)
from keysubgraph.training.sv_signed_gin_trainer import (  # noqa: E402
    load_sv_signed_gin_checkpoint,
)
from keysubgraph.training.sv_spectral_evolution_trainer import (  # noqa: E402
    load_sv_spectral_evolution_checkpoint,
    run_sv_spectral_evolution_epoch,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--static-scaler", type=Path, required=True)
    parser.add_argument("--transition-scaler", type=Path, required=True)
    parser.add_argument("--anchor-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--threshold-strategy",
        choices=("balanced_accuracy", "accuracy"),
        default="balanced_accuracy",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--shuffle-time", action="store_true")
    parser.add_argument("--shuffle-seed", type=int, default=2026)
    parser.add_argument("--evaluation-variant")
    return parser.parse_args()


def _trusted_load(path, device):
    try:
        return torch.load(
            str(Path(path).resolve()),
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(str(Path(path).resolve()), map_location=device)


def _atomic_json(path, payload):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def main():
    args = parse_args()
    device = torch.device(args.device)
    anchor_raw = _trusted_load(args.anchor_checkpoint, device)
    anchor = SVSignedGINClassifier(
        SVSignedGINConfig(**anchor_raw["model_config"])
    ).to(device)
    load_sv_signed_gin_checkpoint(
        args.anchor_checkpoint, anchor, device
    )
    raw = _trusted_load(args.checkpoint, device)
    model = SVSpectralEvolutionClassifier(
        anchor,
        SVSpectralEvolutionConfig(**raw["model_config"]),
    ).to(device)
    checkpoint = load_sv_spectral_evolution_checkpoint(
        args.checkpoint, model, device
    )
    thresholds = checkpoint.get("validation_thresholds", {})
    if args.threshold_strategy not in thresholds:
        raise ValueError("checkpoint has no frozen validation threshold")
    expected = checkpoint["provenance"]
    checks = (
        expected["static_scaler_sha256"]
        == file_sha256(args.static_scaler),
        expected["transition_scaler_sha256"]
        == file_sha256(args.transition_scaler),
        expected["anchor_checkpoint_sha256"]
        == file_sha256(args.anchor_checkpoint),
    )
    if not all(checks):
        raise ValueError("spectral evolution evaluation provenance mismatch")
    dataset = SVSpectralEvolutionDataset(
        args.manifest,
        args.static_scaler,
        args.transition_scaler,
        shuffle_time=args.shuffle_time,
        shuffle_seed=args.shuffle_seed,
    )
    if (
        expected["protocol_sha256"]
        != dataset.manifest["protocol_sha256"]
        or expected["selector_checkpoint_sha256"]
        != dataset.manifest["selector_checkpoint_sha256"]
    ):
        raise ValueError("spectral evolution manifest provenance mismatch")
    loader = create_sv_spectral_evolution_loader(
        dataset,
        batch_size=args.batch_size,
        seed=int(checkpoint["training_config"]["seed"]),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    threshold = float(thresholds[args.threshold_strategy])
    metrics = run_sv_spectral_evolution_epoch(
        model,
        loader,
        device,
        checkpoint["class_weights"],
        threshold=threshold,
        include_predictions=True,
        auxiliary_loss_weight=float(
            checkpoint["training_config"]["auxiliary_loss_weight"]
        ),
    )
    variant = args.evaluation_variant or (
        SV_SPECTRAL_EVOLUTION_VARIANT
        + ("_time_shuffled" if args.shuffle_time else "")
    )
    result = {
        "artifact_type": "sv_spectral_evolution_evaluation",
        "split": dataset.split,
        "variant": variant,
        "threshold_strategy": args.threshold_strategy,
        "threshold": threshold,
        "threshold_fit_split": "validation",
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "manifest_sha256": file_sha256(args.manifest),
        "static_scaler_sha256": file_sha256(args.static_scaler),
        "transition_scaler_sha256": file_sha256(
            args.transition_scaler
        ),
        "anchor_checkpoint_sha256": file_sha256(
            args.anchor_checkpoint
        ),
        "time_order": "shuffled" if args.shuffle_time else "original",
        "shuffle_seed": args.shuffle_seed if args.shuffle_time else None,
        "metrics": {
            key: value
            for key, value in metrics.items()
            if key != "predictions"
        },
        "predictions": metrics["predictions"],
    }
    _atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "variant": variant,
                "split": dataset.split,
                "roc_auc": metrics["roc_auc"],
                "site_stratified_roc_auc": metrics[
                    "site_stratified_roc_auc"
                ],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1"],
                "residual_gate": metrics["residual_gate"],
                "time_order": result["time_order"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
