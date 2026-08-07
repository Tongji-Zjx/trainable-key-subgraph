"""Train the differentiable selector-proxy stage of Dual-STSE-HardSGW."""

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

from keysubgraph.data.data_protocol import (  # noqa: E402
    protocol_node_name_policy,
    protocol_partitions,
    validate_data_protocol,
)
from keysubgraph.data.data_split import file_sha256  # noqa: E402
from keysubgraph.data.exact_stse_dataset import (  # noqa: E402
    ExactSTSEDataset,
    create_exact_stse_loader,
)
from keysubgraph.models.dual_stse_hard_sgw import (  # noqa: E402
    DualSTSEHardSGWClassifier,
)
from keysubgraph.models.dual_stse_hard_sgw_types import (  # noqa: E402
    DUAL_SELECTOR_ARCHITECTURES,
    DualSTSEHardSGWConfig,
)
from keysubgraph.models.dual_stse_hard_sgw_loss import (  # noqa: E402
    DualSTSEHardSGWLossConfig,
)
from keysubgraph.training.dual_stse_hard_sgw_trainer import (  # noqa: E402
    DualTrainingConfig,
    train_dual_stage,
)
from keysubgraph.training.trainer import set_reproducible_seed  # noqa: E402


class _InMemoryExactSTSEDataset(object):
    """Materialize immutable samples once to avoid repeated torch.load calls."""

    def __init__(self, source):
        self.split = source.split
        self.assignments = source.assignments
        self.samples = tuple(source[index] for index in range(len(source)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT
        / "configs"
        / "data_protocol_exact_stse_no_coord_full.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        help=(
            "trusted selector checkpoint used to initialize all compatible "
            "parameters; newly introduced parameters retain their defaults"
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--cache-dataset-memory",
        action="store_true",
        help="load each immutable graph sequence once before training",
    )
    parser.add_argument(
        "--fast-runtime",
        action="store_true",
        help="skip synchronization-heavy selector logging only",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-validation-batches", type=int)
    parser.add_argument(
        "--selector-architecture",
        choices=DUAL_SELECTOR_ARCHITECTURES,
        default="legacy_mlp",
    )
    parser.add_argument("--critical-subgraph-count", type=int)
    parser.add_argument("--selector-graph-layers", type=int, default=2)
    parser.add_argument("--selector-spectral-dim", type=int, default=8)
    parser.add_argument("--object-overlap-minimum", type=float, default=0.05)
    parser.add_argument("--object-overlap-maximum", type=float, default=0.30)
    parser.add_argument("--object-node-ratio", type=float, default=0.10)
    parser.add_argument("--object-temporal-state", action="store_true")
    parser.add_argument(
        "--structural-temporal-memory",
        action="store_true",
        help=(
            "carry ROI-aligned soft node/edge memberships, use Sinkhorn slot "
            "alignment, enable latent object state, and use history-aware hardening"
        ),
    )
    parser.add_argument("--memory-diffusion", type=float, default=0.15)
    parser.add_argument(
        "--sinkhorn-temperature", type=float, default=0.10
    )
    parser.add_argument("--sinkhorn-iterations", type=int, default=8)
    parser.add_argument(
        "--history-continuity-bonus", type=float, default=0.25
    )
    parser.add_argument("--history-switch-margin", type=float, default=0.05)
    parser.add_argument(
        "--selector-objective",
        choices=("current", "full_soft", "full_soft_hard"),
        default="current",
        help=(
            "current reproduces the original STE hard objective; "
            "full_soft adds the explicit signed soft path; "
            "full_soft_hard also controls soft-to-hard quantization"
        ),
    )
    parser.add_argument("--soft-warmup-epochs", type=int, default=3)
    parser.add_argument(
        "--selector-soft-ce-weight", type=float, default=0.25
    )
    parser.add_argument(
        "--selector-hard-ce-weight", type=float, default=0.25
    )
    parser.add_argument(
        "--soft-hard-spectral-weight", type=float, default=0.05
    )
    parser.add_argument(
        "--soft-hard-gw-weight", type=float, default=0.02
    )
    parser.add_argument(
        "--soft-hard-kd-weight", type=float, default=0.05
    )
    parser.add_argument("--object-overlap-weight", type=float, default=0.10)
    parser.add_argument(
        "--object-reconstruction-weight", type=float, default=0.10
    )
    parser.add_argument("--object-coverage-weight", type=float, default=0.05)
    parser.add_argument("--object-temporal-weight", type=float, default=0.05)
    parser.add_argument(
        "--object-node-continuity-weight", type=float, default=0.10
    )
    parser.add_argument(
        "--object-edge-continuity-weight", type=float, default=0.05
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    for name in ("max_train_batches", "max_validation_batches"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError("--{} must be positive".format(name.replace("_", "-")))
    if args.cache_dataset_memory and args.num_workers != 0:
        raise ValueError(
            "in-memory selector data requires --num-workers 0"
        )
    set_reproducible_seed(args.seed)
    protocol = validate_data_protocol(args.protocol, PROJECT_ROOT)
    if tuple(protocol_partitions(protocol)) != (
        "train",
        "validation",
        "test",
    ):
        raise ValueError(
            "dual selector requires a frozen partitioned protocol"
        )
    paths = protocol["paths"]
    node_name_policy = protocol_node_name_policy(protocol)
    common = (
        PROJECT_ROOT / paths["dataset_root"],
        PROJECT_ROOT / paths["sample_index_csv"],
        PROJECT_ROOT / paths["splits_csv"],
    )
    train_dataset = ExactSTSEDataset(
        *common,
        "train",
        protocol["edge_presence_threshold"],
        require_coordinates=False,
        node_name_policy=node_name_policy,
    )
    validation_dataset = ExactSTSEDataset(
        *common,
        "validation",
        protocol["edge_presence_threshold"],
        require_coordinates=False,
        node_name_policy=node_name_policy,
    )
    if args.cache_dataset_memory:
        train_dataset = _InMemoryExactSTSEDataset(train_dataset)
        validation_dataset = _InMemoryExactSTSEDataset(
            validation_dataset
        )
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    train_loader = create_exact_stse_loader(
        train_dataset,
        args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = create_exact_stse_loader(
        validation_dataset,
        args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    critical_subgraph_count = (
        args.critical_subgraph_count
        if args.critical_subgraph_count is not None
        else 3
        if args.selector_architecture == "theory_multi_object"
        else 5
    )
    model_config = DualSTSEHardSGWConfig(
        selector_architecture=args.selector_architecture,
        critical_subgraph_count=critical_subgraph_count,
        selector_graph_layers=args.selector_graph_layers,
        selector_spectral_dim=args.selector_spectral_dim,
        selector_spectral_cache=True,
        selector_fast_runtime=args.fast_runtime,
        selector_object_overlap_minimum=args.object_overlap_minimum,
        selector_object_overlap_maximum=args.object_overlap_maximum,
        selector_object_temporal_state=(
            args.object_temporal_state
            or args.structural_temporal_memory
        ),
        selector_structural_temporal_memory=(
            args.structural_temporal_memory
        ),
        selector_memory_diffusion=args.memory_diffusion,
        selector_sinkhorn_temperature=args.sinkhorn_temperature,
        selector_sinkhorn_iterations=args.sinkhorn_iterations,
        critical_node_ratio_per_object=args.object_node_ratio,
        critical_history_continuity_bonus=(
            args.history_continuity_bonus
        ),
        critical_history_switch_margin=args.history_switch_margin,
    )
    model = DualSTSEHardSGWClassifier(model_config)
    initialization = None
    if args.initial_checkpoint is not None:
        checkpoint_path = args.initial_checkpoint.resolve()
        checkpoint = torch.load(
            str(checkpoint_path), map_location="cpu", weights_only=False
        )
        source = checkpoint.get("model_state_dict")
        if not isinstance(source, dict):
            raise ValueError("initial checkpoint has no model_state_dict")
        target = model.state_dict()
        compatible = {
            name: value
            for name, value in source.items()
            if name in target and tuple(value.shape) == tuple(target[name].shape)
        }
        incompatible = model.load_state_dict(compatible, strict=False)
        initialization = {
            "path": str(checkpoint_path),
            "sha256": file_sha256(checkpoint_path),
            "loaded_tensor_count": len(compatible),
            "missing_keys": tuple(incompatible.missing_keys),
            "unexpected_keys": tuple(incompatible.unexpected_keys),
        }
        print(
            "initialized {} compatible tensors from {}; missing={}".format(
                len(compatible),
                checkpoint_path,
                list(incompatible.missing_keys),
            )
        )
    result = train_dual_stage(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        train_labels=[item.label for item in train_dataset.assignments],
        device=device,
        training_config=DualTrainingConfig(
            stage="selector_proxy",
            epochs=1 if args.smoke else args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            gradient_clip_norm=args.gradient_clip,
            early_stopping_patience=args.early_stopping_patience,
            seed=args.seed,
            max_train_batches=(
                1 if args.smoke else args.max_train_batches
            ),
            max_validation_batches=(
                1 if args.smoke else args.max_validation_batches
            ),
        ),
        loss_config=DualSTSEHardSGWLossConfig(
            selector_objective=args.selector_objective,
            selector_soft_ce_weight=args.selector_soft_ce_weight,
            selector_hard_ce_weight=args.selector_hard_ce_weight,
            soft_hard_spectral_weight=(
                args.soft_hard_spectral_weight
            ),
            soft_hard_gw_weight=args.soft_hard_gw_weight,
            soft_hard_kd_weight=args.soft_hard_kd_weight,
            soft_warmup_epochs=args.soft_warmup_epochs,
            object_overlap_weight=args.object_overlap_weight,
            object_reconstruction_weight=(
                args.object_reconstruction_weight
            ),
            object_coverage_weight=args.object_coverage_weight,
            object_temporal_weight=args.object_temporal_weight,
            object_node_continuity_weight=(
                args.object_node_continuity_weight
            ),
            object_edge_continuity_weight=(
                args.object_edge_continuity_weight
            ),
        ),
        output_dir=args.output_dir,
        protocol_sha256=file_sha256(args.protocol),
        provenance={
            "stse_checkpoint_sha256": "not_used_in_selector_stage",
            "selector_checkpoint_sha256": "trained_by_this_stage",
            "sgw_scaler_sha256": "not_applicable",
            "selector_objective": args.selector_objective,
            "selector_architecture": args.selector_architecture,
            "critical_subgraph_count": critical_subgraph_count,
            "object_temporal_state": (
                args.object_temporal_state
                or args.structural_temporal_memory
            ),
            "structural_temporal_memory": (
                args.structural_temporal_memory
            ),
            "memory_diffusion": args.memory_diffusion,
            "sinkhorn_temperature": args.sinkhorn_temperature,
            "sinkhorn_iterations": args.sinkhorn_iterations,
            "history_continuity_bonus": args.history_continuity_bonus,
            "history_switch_margin": args.history_switch_margin,
            "initialization": initialization,
        },
    )
    print(
        json.dumps(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in result.items()
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
