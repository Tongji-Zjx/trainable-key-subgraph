"""Train SV classifiers, including fixed theory-geometry branch ablations."""

from __future__ import absolute_import, division, print_function

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
from keysubgraph.data.sv_signed_gin_dataset import (  # noqa: E402
    SVMultiBudgetDataset,
    SVSignedGINDataset,
    create_sv_signed_gin_loader,
)
from keysubgraph.data.sv_theory_geometry import (  # noqa: E402
    SVTheoryAugmentedDataset,
)
from keysubgraph.data.sv_spectral_diffusion import (  # noqa: E402
    SVSpectralDiffusionAugmentedDataset,
)
from keysubgraph.models.sv_signed_gin import (  # noqa: E402
    SV_DEFAULT_VARIANT,
    SV_SIGNED_GIN_MESSAGE_MODES,
    SV_SIGNED_GIN_POOLING_MODES,
    SV_SIGNED_GIN_VARIANTS,
    SVSignedGINClassifier,
    SVSignedGINConfig,
)
from keysubgraph.training.sv_signed_gin_trainer import (  # noqa: E402
    SVSignedGINTrainingConfig,
    train_sv_signed_gin_classifier,
)
from keysubgraph.training.trainer import (  # noqa: E402
    set_reproducible_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument("--theory-train-cache", type=Path)
    parser.add_argument("--theory-validation-cache", type=Path)
    parser.add_argument("--theory-scaler", type=Path)
    parser.add_argument("--spectral-train-manifest", type=Path)
    parser.add_argument("--spectral-validation-manifest", type=Path)
    parser.add_argument("--spectral-scaler", type=Path)
    parser.add_argument(
        "--multi-budget-train-manifests", nargs=3, type=Path
    )
    parser.add_argument(
        "--multi-budget-validation-manifests", nargs=3, type=Path
    )
    parser.add_argument("--multi-budget-scalers", nargs=3, type=Path)
    parser.add_argument(
        "--variant",
        choices=SV_SIGNED_GIN_VARIANTS,
        default=SV_DEFAULT_VARIANT,
        help=(
            "model variant; defaults to the formal SVG architecture "
            "(Static-spectral + Variation + SignedGIN)"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--static-anchor-epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=2
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument(
        "--early-stopping-patience", type=int, default=15
    )
    parser.add_argument(
        "--selection-metric",
        choices=("roc_auc", "composite_auc"),
        default="composite_auc",
    )
    parser.add_argument("--gin-hidden-dim", type=int, default=64)
    parser.add_argument("--gin-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument(
        "--message-mode",
        choices=SV_SIGNED_GIN_MESSAGE_MODES,
        default=None,
    )
    parser.add_argument(
        "--pooling",
        choices=SV_SIGNED_GIN_POOLING_MODES,
        default=None,
    )
    parser.add_argument(
        "--gin-residual",
        dest="gin_residual",
        action="store_true",
    )
    parser.add_argument(
        "--no-gin-residual",
        dest="gin_residual",
        action="store_false",
    )
    parser.add_argument(
        "--gin-jumping-knowledge",
        dest="gin_jumping_knowledge",
        action="store_true",
    )
    parser.add_argument(
        "--no-gin-jumping-knowledge",
        dest="gin_jumping_knowledge",
        action="store_false",
    )
    parser.add_argument(
        "--gin-compact-readout",
        dest="gin_compact_readout",
        action="store_true",
    )
    parser.add_argument(
        "--no-gin-compact-readout",
        dest="gin_compact_readout",
        action="store_false",
    )
    parser.add_argument(
        "--gin-batch-normalization",
        dest="gin_batch_normalization",
        action="store_true",
    )
    parser.add_argument(
        "--no-gin-batch-normalization",
        dest="gin_batch_normalization",
        action="store_false",
    )
    parser.add_argument(
        "--gin-residual-attention", action="store_true"
    )
    parser.add_argument(
        "--auxiliary-loss-weight", type=float, default=None
    )
    parser.add_argument(
        "--signed-delta-q-weight", type=float, default=None
    )
    parser.add_argument(
        "--training-recipe",
        choices=("current", "author_a1"),
        default="current",
    )
    parser.add_argument(
        "--site-class-balanced-sampler",
        action="store_true",
        help=(
            "balance (site,class) strata in training batches; site is "
            "never passed to the model"
        ),
    )
    parser.add_argument(
        "--residual-gate-penalty-weight", type=float, default=0.01
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overfit-samples", type=int)
    parser.add_argument("--disable-early-stopping", action="store_true")
    parser.set_defaults(
        gin_residual=None,
        gin_jumping_knowledge=None,
        gin_compact_readout=None,
        gin_batch_normalization=None,
    )
    return parser.parse_args()


def _resolve_architecture_defaults(args):
    """Apply the formal SVG profile without changing explicit ablations."""

    is_default_svg = args.variant in (
        SV_DEFAULT_VARIANT,
        "signed_gin_multibranch_spectral_direction",
        "signed_gin_multibranch_diffusion_geometry",
        "signed_gin_multibranch_theory_geometry",
        "svg_v2_b1_hks",
        "svg_v2_c1_diffusion",
        "svg_v2_c3_hks_diffusion",
        "svg_v2_g2_signed_delta_q",
        "svg_v2_g2_signed_delta_q_gin32",
        "svg_v2_c3_f1_residual",
        "svg_v2_c3_g2",
        "svg_v2_d1_community_pooling",
        "svg_v2_e1_multi_budget",
    )
    if args.message_mode is None:
        args.message_mode = (
            "signed_normalized" if is_default_svg else "signed_weighted"
        )
    if args.pooling is None:
        args.pooling = "mean_std" if is_default_svg else "attention"
    for name in (
        "gin_residual",
        "gin_jumping_knowledge",
        "gin_compact_readout",
        "gin_batch_normalization",
    ):
        if getattr(args, name) is None:
            setattr(args, name, bool(is_default_svg))
    if args.auxiliary_loss_weight is None:
        args.auxiliary_loss_weight = 0.25 if is_default_svg else 0.0
    if args.signed_delta_q_weight is None:
        args.signed_delta_q_weight = (
            0.05
            if args.variant in (
                "svg_v2_g2_signed_delta_q",
                "svg_v2_g2_signed_delta_q_gin32",
                "svg_v2_c3_g2",
            )
            else 0.0
        )
    args.label_smoothing = 0.0
    args.scheduler_mode = "plateau"
    args.minimum_epochs = 0
    args.balanced_batch_sampler = False
    args.use_class_weights = True
    if args.training_recipe == "author_a1":
        # Frozen five-day A1 profile; no hyperparameter grid is opened here.
        args.epochs = 60
        args.static_anchor_epochs = 60
        args.learning_rate = 1.0e-4
        args.weight_decay = 5.0e-5
        args.gradient_clip = 5.0
        args.early_stopping_patience = 10
        args.batch_size = 4
        args.gradient_accumulation_steps = 8
        args.label_smoothing = 0.10
        args.scheduler_mode = "cosine"
        args.minimum_epochs = 30
        args.balanced_batch_sampler = True
        args.use_class_weights = False
    if args.site_class_balanced_sampler:
        if args.balanced_batch_sampler:
            raise ValueError(
                "site/class and class-only balanced samplers conflict"
            )
        args.use_class_weights = False
    return args


def _balanced_limit(dataset, count):
    if count is None:
        return
    if count < 2:
        raise ValueError("SV overfit sample count must be at least two")
    by_class = {
        label: [
            index
            for index, value in enumerate(dataset.labels)
            if value == label
        ]
        for label in (0, 1)
    }
    left = count // 2
    right = count - left
    if len(by_class[0]) < left or len(by_class[1]) < right:
        raise ValueError("SV overfit subset cannot contain both classes")
    indices = by_class[0][:left] + by_class[1][:right]
    dataset.samples = [dataset.samples[index] for index in indices]
    dataset.sites = [dataset.sites[index] for index in indices]
    dataset.subject_ids = [
        dataset.subject_ids[index] for index in indices
    ]


def main():
    args = _resolve_architecture_defaults(parse_args())
    # The model is instantiated below, so the CLI seed must be applied here
    # rather than only inside the trainer after parameters already exist.
    set_reproducible_seed(args.seed)
    if args.smoke and args.overfit_samples is not None:
        raise ValueError("SV smoke and overfit modes are mutually exclusive")
    model_config = SVSignedGINConfig(
        variant=args.variant,
        gin_hidden_dim=args.gin_hidden_dim,
        gin_layers=args.gin_layers,
        dropout=args.dropout,
        message_mode=args.message_mode,
        pooling=args.pooling,
        gin_residual=args.gin_residual,
        gin_jumping_knowledge=args.gin_jumping_knowledge,
        gin_compact_readout=args.gin_compact_readout,
        gin_batch_normalization=args.gin_batch_normalization,
        gin_residual_attention=args.gin_residual_attention,
    )
    validation_manifest = (
        args.train_manifest
        if args.overfit_samples is not None
        else args.validation_manifest
    )
    theory_arguments = (
        args.theory_train_cache,
        args.theory_validation_cache,
        args.theory_scaler,
    )
    spectral_arguments = (
        args.spectral_train_manifest,
        args.spectral_validation_manifest,
        args.spectral_scaler,
    )
    multi_budget_arguments = (
        args.multi_budget_train_manifests,
        args.multi_budget_validation_manifests,
        args.multi_budget_scalers,
    )
    if model_config.uses_multi_budget:
        if any(value is None for value in multi_budget_arguments):
            raise ValueError(
                "E1 requires train/validation manifests and scalers for "
                "all three budgets"
            )
        if any(value is not None for value in theory_arguments):
            raise ValueError("E1 cannot use legacy theory sidecars")
        if any(value is not None for value in spectral_arguments):
            raise ValueError("E1 cannot use spectral sidecars")
        train = SVMultiBudgetDataset(
            args.multi_budget_train_manifests,
            args.multi_budget_scalers,
            include_windows=True,
        )
        validation = SVMultiBudgetDataset(
            (
                args.multi_budget_train_manifests
                if args.overfit_samples is not None
                else args.multi_budget_validation_manifests
            ),
            args.multi_budget_scalers,
            include_windows=True,
        )
    elif any(value is not None for value in multi_budget_arguments):
        raise ValueError(
            "multi-budget inputs were supplied to a single-budget model"
        )
    elif model_config.uses_theory_geometry:
        if model_config.uses_spectral_diffusion_sidecar:
            raise ValueError("legacy and SVG-v2 sidecars cannot be combined")
        if any(value is None for value in theory_arguments):
            raise ValueError(
                "theory-geometry variants require train/validation "
                "sidecars and a train-only theory scaler"
            )
        validation_theory_cache = (
            args.theory_train_cache
            if args.overfit_samples is not None
            else args.theory_validation_cache
        )
        train = SVTheoryAugmentedDataset(
            args.train_manifest,
            args.scaler,
            args.theory_train_cache,
            args.theory_scaler,
            include_windows=model_config.uses_gin,
        )
        validation = SVTheoryAugmentedDataset(
            validation_manifest,
            args.scaler,
            validation_theory_cache,
            args.theory_scaler,
            include_windows=model_config.uses_gin,
        )
    elif model_config.uses_spectral_diffusion_sidecar:
        if any(value is None for value in spectral_arguments):
            raise ValueError(
                "SVG-v2 spectral variants require train/validation "
                "spectral manifests and a train-only scaler"
            )
        if any(value is not None for value in theory_arguments):
            raise ValueError("legacy theory sidecars were supplied to SVG-v2")
        validation_spectral_manifest = (
            args.spectral_train_manifest
            if args.overfit_samples is not None
            else args.spectral_validation_manifest
        )
        train = SVSpectralDiffusionAugmentedDataset(
            args.train_manifest,
            args.scaler,
            args.spectral_train_manifest,
            args.spectral_scaler,
            include_windows=True,
        )
        validation = SVSpectralDiffusionAugmentedDataset(
            validation_manifest,
            args.scaler,
            validation_spectral_manifest,
            args.spectral_scaler,
            include_windows=True,
        )
    else:
        if any(value is not None for value in theory_arguments):
            raise ValueError(
                "theory sidecars were supplied to a non-theory variant"
            )
        if any(value is not None for value in spectral_arguments):
            raise ValueError(
                "spectral-diffusion sidecars were supplied to a base variant"
            )
        train = SVSignedGINDataset(
            args.train_manifest,
            args.scaler,
            include_windows=model_config.uses_gin,
        )
        validation = SVSignedGINDataset(
            validation_manifest,
            args.scaler,
            include_windows=model_config.uses_gin,
        )
    if train.split != "train":
        raise ValueError("SV training manifest must be train")
    if args.overfit_samples is None and validation.split != "validation":
        raise ValueError("SV validation manifest must be validation")
    if args.overfit_samples is not None:
        _balanced_limit(train, args.overfit_samples)
        _balanced_limit(validation, args.overfit_samples)
    elif args.smoke:
        _balanced_limit(train, min(4, len(train)))

    train_loader = create_sv_signed_gin_loader(
        train,
        batch_size=args.batch_size,
        seed=args.seed,
        shuffle=not (
            args.balanced_batch_sampler
            or args.site_class_balanced_sampler
        ),
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        balanced_batch_sampler=args.balanced_batch_sampler,
        site_class_balanced_sampler=(
            args.site_class_balanced_sampler
        ),
    )
    validation_loader = create_sv_signed_gin_loader(
        validation,
        batch_size=args.batch_size,
        seed=args.seed,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    model = SVSignedGINClassifier(model_config)
    epochs = 1 if args.smoke else args.epochs
    static_anchor_epochs = (
        1 if args.smoke else args.static_anchor_epochs
    )
    patience = (
        0
        if args.disable_early_stopping
        else args.early_stopping_patience
    )
    selection_metric = (
        "roc_auc"
        if args.smoke or args.overfit_samples is not None
        else args.selection_metric
    )
    training_config = SVSignedGINTrainingConfig(
        epochs=epochs,
        static_anchor_epochs=static_anchor_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip,
        gradient_accumulation_steps=(
            args.gradient_accumulation_steps
        ),
        early_stopping_patience=patience,
        selection_metric=selection_metric,
        auxiliary_loss_weight=args.auxiliary_loss_weight,
        residual_gate_penalty_weight=(
            args.residual_gate_penalty_weight
        ),
        signed_delta_q_weight=args.signed_delta_q_weight,
        label_smoothing=args.label_smoothing,
        scheduler_mode=args.scheduler_mode,
        minimum_epochs=(0 if args.smoke else args.minimum_epochs),
        use_class_weights=args.use_class_weights,
        seed=args.seed,
        max_train_batches=2 if args.smoke else None,
        max_validation_batches=2 if args.smoke else None,
    )
    manifest = train.manifest
    provenance = {
        "protocol_sha256": manifest["protocol_sha256"],
        "selector_checkpoint_sha256": manifest[
            "selector_checkpoint_sha256"
        ],
        "selection_mode": manifest["selection_mode"],
        "selection_seed": int(manifest["selection_seed"]),
        "train_manifest_sha256": file_sha256(args.train_manifest),
        "validation_manifest_sha256": file_sha256(
            validation_manifest
        ),
        "scaler_sha256": file_sha256(args.scaler),
    }
    if model_config.uses_theory_geometry:
        provenance.update(
            {
                "theory_train_cache_sha256": file_sha256(
                    args.theory_train_cache
                ),
                "theory_validation_cache_sha256": file_sha256(
                    validation_theory_cache
                ),
                "theory_scaler_sha256": file_sha256(
                    args.theory_scaler
                ),
            }
        )
    if model_config.uses_spectral_diffusion_sidecar:
        provenance.update(
            {
                "spectral_train_manifest_sha256": file_sha256(
                    args.spectral_train_manifest
                ),
                "spectral_validation_manifest_sha256": file_sha256(
                    validation_spectral_manifest
                ),
                "spectral_scaler_sha256": file_sha256(
                    args.spectral_scaler
                ),
            }
        )
    if model_config.uses_multi_budget:
        provenance.update(
            {
                "multi_budget_train_manifest_sha256": [
                    file_sha256(path)
                    for path in train.manifest_paths
                ],
                "multi_budget_validation_manifest_sha256": [
                    file_sha256(path)
                    for path in validation.manifest_paths
                ],
                "multi_budget_scaler_sha256": [
                    file_sha256(path) for path in train.scaler_paths
                ],
                "multi_budget_grid": train.manifest[
                    "multi_budget_grid"
                ],
            }
        )
    result = train_sv_signed_gin_classifier(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        train_labels=train.labels,
        device=torch.device(args.device),
        config=training_config,
        output_dir=args.output_dir,
        provenance=provenance,
    )
    result.update(
        {
            "device": args.device,
            "effective_batch_size": (
                args.batch_size * args.gradient_accumulation_steps
            ),
            "smoke": bool(args.smoke),
            "overfit_samples": args.overfit_samples,
            "auxiliary_loss_weight": args.auxiliary_loss_weight,
            "residual_gate_penalty_weight": (
                args.residual_gate_penalty_weight
            ),
            "signed_delta_q_weight": args.signed_delta_q_weight,
            "training_recipe": args.training_recipe,
            "balanced_batch_sampler": args.balanced_batch_sampler,
            "site_class_balanced_sampler": (
                args.site_class_balanced_sampler
            ),
            "label_smoothing": args.label_smoothing,
        }
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
