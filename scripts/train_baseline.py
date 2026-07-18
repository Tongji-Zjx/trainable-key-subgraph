"""Train the neutral signed hard-subgraph sequence baseline."""

from __future__ import absolute_import, print_function

import argparse
import json
import sys
import time
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.baseline_collate import create_baseline_loader  # noqa: E402
from keysubgraph.data.baseline_dataset import BaselineHardSubgraphDataset  # noqa: E402
from keysubgraph.models.baseline_classifier import (  # noqa: E402
    BaselineModelConfig,
    HISTORY_MODES,
    SignedSequenceBaseline,
    TEMPORAL_ORDERS,
)
from keysubgraph.features.structural_prior import (  # noqa: E402
    STATIC_WINDOW_STRUCTURAL_FEATURES,
    STRUCTURAL_GROUPS,
    TEMPORAL_STRUCTURAL_GROUPS,
    TEMPORAL_WINDOW_STRUCTURAL_FEATURES,
    fit_structural_transform,
    fit_temporal_structural_transform,
    structural_group_configuration,
    temporal_structural_group_configuration,
)
from keysubgraph.training.baseline_trainer import (  # noqa: E402
    BaselineTrainingConfig,
    set_baseline_seed,
    train_baseline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "baseline_training" / "key_full_seed42",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument(
        "--selection-metric",
        choices=("unweighted_log_loss", "roc_auc"),
        default="unweighted_log_loss",
    )
    parser.add_argument("--node-hidden", type=int, default=64)
    parser.add_argument("--signed-layers", type=int, default=2)
    parser.add_argument("--fusion-dim", type=int, default=128)
    parser.add_argument("--gru-hidden", type=int, default=128)
    parser.add_argument("--classifier-hidden", type=int, default=64)
    parser.add_argument("--signed-dropout", type=float, default=0.1)
    parser.add_argument("--classifier-dropout", type=float, default=0.2)
    parser.add_argument(
        "--history-mode", choices=HISTORY_MODES, default="full"
    )
    parser.add_argument("--history-keep-ratio", type=float, default=1.0)
    parser.add_argument(
        "--temporal-order", choices=TEMPORAL_ORDERS, default="ordered"
    )
    parser.add_argument("--permutation-seed", type=int, default=42)
    parser.add_argument(
        "--structural-group",
        choices=("neutral",) + STRUCTURAL_GROUPS,
        default="neutral",
        help="A-E enables the parameter-matched structural experiment interface.",
    )
    parser.add_argument("--prior-beta", type=float, default=1.0)
    parser.add_argument("--prior-permutation-seed", type=int, default=42)
    parser.add_argument(
        "--structural-transform",
        type=Path,
        help="Pre-fitted train-only A-E artifact; avoids refitting for every seed.",
    )
    parser.add_argument(
        "--temporal-structural-group",
        choices=TEMPORAL_STRUCTURAL_GROUPS,
        help="A/B/F/G/H static-plus-first-difference experiment interface v2.",
    )
    parser.add_argument("--structural-delta-permutation-seed", type=int, default=42)
    parser.add_argument(
        "--smoke", action="store_true", help="Run one batch per partition for one epoch."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_dataset = BaselineHardSubgraphDataset(PROJECT_ROOT, args.train_manifest)
    validation_dataset = BaselineHardSubgraphDataset(
        PROJECT_ROOT, args.validation_manifest
    )
    structural_transform = None
    structural_version = 0
    use_structural_features = False
    use_structural_deltas = False
    structural_delta_order = "ordered"
    prior_mode = "none"
    structural_group = args.structural_group
    if args.temporal_structural_group is not None:
        if args.structural_group != "neutral":
            raise ValueError("static A-E and temporal A/B/F/G/H groups are mutually exclusive")
        structural_version = 2
        structural_group = args.temporal_structural_group
        condition = temporal_structural_group_configuration(structural_group)
        use_structural_features = condition["use_structural_features"]
        use_structural_deltas = condition["use_structural_deltas"]
        structural_delta_order = condition["structural_delta_order"]
        if args.structural_transform is not None:
            with args.structural_transform.resolve().open("r", encoding="utf-8") as handle:
                structural_transform = json.load(handle)
            if structural_transform.get("structural_group") != structural_group:
                raise ValueError("pre-fitted temporal structural transform group differs")
            if int(structural_transform.get("structural_delta_permutation_seed")) != args.structural_delta_permutation_seed:
                raise ValueError("pre-fitted temporal permutation seed differs")
        else:
            structural_transform = fit_temporal_structural_transform(
                train_dataset,
                structural_group,
                permutation_seed=args.structural_delta_permutation_seed,
            )
    elif args.structural_group != "neutral":
        structural_version = 1
        use_structural_features, prior_mode = structural_group_configuration(
            args.structural_group
        )
        if args.structural_transform is not None:
            with args.structural_transform.resolve().open("r", encoding="utf-8") as handle:
                structural_transform = json.load(handle)
            if structural_transform.get("structural_group") != args.structural_group:
                raise ValueError("pre-fitted structural transform group differs")
            if float(structural_transform.get("beta")) != args.prior_beta:
                raise ValueError("pre-fitted structural transform beta differs")
            if int(structural_transform.get("permutation_seed")) != args.prior_permutation_seed:
                raise ValueError("pre-fitted structural permutation seed differs")
        else:
            structural_transform = fit_structural_transform(
                train_dataset,
                args.structural_group,
                beta=args.prior_beta,
                permutation_seed=args.prior_permutation_seed,
            )
    elif args.structural_transform is not None:
        raise ValueError("neutral baseline cannot use a structural transform")
    train_loader = create_baseline_loader(
        train_dataset,
        args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=args.device != "cpu",
    )
    validation_loader = create_baseline_loader(
        validation_dataset,
        args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=args.device != "cpu",
    )
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    set_baseline_seed(args.seed)
    model = SignedSequenceBaseline(
        BaselineModelConfig(
            node_hidden_dim=args.node_hidden,
            signed_gnn_layers=args.signed_layers,
            signed_gnn_dropout=args.signed_dropout,
            fusion_dim=args.fusion_dim,
            gru_hidden_dim=args.gru_hidden,
            classifier_hidden_dim=args.classifier_hidden,
            classifier_dropout=args.classifier_dropout,
            history_mode=args.history_mode,
            history_keep_ratio=args.history_keep_ratio,
            temporal_order=args.temporal_order,
            permutation_seed=args.permutation_seed,
            use_structural_features=use_structural_features,
            use_structural_deltas=use_structural_deltas,
            structural_interface_version=structural_version,
            structural_group=structural_group,
            structural_feature_dim=(
                len(TEMPORAL_WINDOW_STRUCTURAL_FEATURES)
                if structural_version == 2
                else len(STATIC_WINDOW_STRUCTURAL_FEATURES)
            ),
            prior_mode=prior_mode,
            prior_beta=0.0 if structural_version == 2 else args.prior_beta,
            prior_permutation_seed=(
                args.structural_delta_permutation_seed
                if structural_version == 2
                else args.prior_permutation_seed
            ),
            structural_delta_order=structural_delta_order,
            structural_delta_permutation_seed=args.structural_delta_permutation_seed,
        )
    )
    config = BaselineTrainingConfig(
        epochs=1 if args.smoke else args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip,
        seed=args.seed,
        early_stopping_patience=args.early_stopping_patience,
        selection_metric=args.selection_metric,
        max_train_batches=1 if args.smoke else None,
        max_validation_batches=1 if args.smoke else None,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    result = train_baseline(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        train_labels=[record.label for record in train_dataset.records],
        device=device,
        config=config,
        output_dir=args.output_dir,
        train_manifest_path=args.train_manifest,
        validation_manifest_path=args.validation_manifest,
        project_root=PROJECT_ROOT,
        structural_transform=structural_transform,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    printable = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in result.items()
    }
    printable.update(
        {
            "device": str(device),
            "smoke": bool(args.smoke),
            "history_mode": args.history_mode,
            "history_keep_ratio": args.history_keep_ratio,
            "temporal_order": args.temporal_order,
            "permutation_seed": args.permutation_seed,
            "structural_group": structural_group,
            "prior_mode": prior_mode,
            "use_structural_deltas": use_structural_deltas,
            "structural_delta_order": structural_delta_order,
            "structural_delta_permutation_seed": args.structural_delta_permutation_seed,
            "elapsed_seconds": time.perf_counter() - started,
            "cuda_peak_memory_mib": (
                torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
                if device.type == "cuda"
                else None
            ),
        }
    )
    print(json.dumps(printable, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
