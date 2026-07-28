"""Run all frozen SG2 classification-bottleneck diagnostics at once."""

from __future__ import absolute_import, division, print_function

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

from keysubgraph.analysis.sv_signed_gin_bottleneck import (  # noqa: E402
    analyze_sv_signed_gin_bottleneck,
    collect_sv_diagnostics,
    selection_control_probe,
    write_sv_signed_gin_bottleneck_artifacts,
)
from keysubgraph.data.data_split import file_sha256  # noqa: E402
from keysubgraph.data.sv_signed_gin_dataset import (  # noqa: E402
    SVSignedGINDataset,
)
from keysubgraph.models.sv_signed_gin import (  # noqa: E402
    SVSignedGINClassifier,
    SVSignedGINConfig,
)
from keysubgraph.training.sv_signed_gin_trainer import (  # noqa: E402
    load_sv_signed_gin_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument(
        "--validation-manifest", type=Path, required=True
    )
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument(
        "--max-validation-samples", type=int, default=0
    )
    for name in ("random", "full"):
        parser.add_argument(
            "--{}-train-manifest".format(name), type=Path
        )
        parser.add_argument(
            "--{}-validation-manifest".format(name), type=Path
        )
        parser.add_argument("--{}-scaler".format(name), type=Path)
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


def _device(name):
    if name == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    return torch.device(name)


def _parameter_fingerprint(model):
    total = 0.0
    squared = 0.0
    count = 0
    for parameter in model.parameters():
        values = parameter.detach().to(
            device="cpu", dtype=torch.float64
        )
        total += float(values.sum())
        squared += float(values.square().sum())
        count += int(values.numel())
    return (count, total, squared)


def _validate_primary_inputs(
    train_dataset,
    validation_dataset,
    scaler_path,
    checkpoint,
):
    if (
        train_dataset.split != "train"
        or validation_dataset.split != "validation"
    ):
        raise ValueError(
            "diagnostic requires train and validation manifests"
        )
    if set(train_dataset.sample_keys).intersection(
        validation_dataset.sample_keys
    ):
        raise ValueError("diagnostic train/validation samples overlap")
    keys = (
        "protocol_sha256",
        "selector_checkpoint_sha256",
        "selection_mode",
        "selection_seed",
    )
    for key in keys:
        if train_dataset.manifest[key] != (
            validation_dataset.manifest[key]
        ):
            raise ValueError(
                "diagnostic manifests disagree on {}".format(key)
            )
    provenance = checkpoint.get("provenance", {})
    expected = {
        "protocol_sha256": train_dataset.manifest[
            "protocol_sha256"
        ],
        "selector_checkpoint_sha256": train_dataset.manifest[
            "selector_checkpoint_sha256"
        ],
        "selection_mode": train_dataset.manifest["selection_mode"],
        "selection_seed": int(
            train_dataset.manifest["selection_seed"]
        ),
        "train_manifest_sha256": file_sha256(
            train_dataset.manifest_path
        ),
        "validation_manifest_sha256": file_sha256(
            validation_dataset.manifest_path
        ),
        "scaler_sha256": file_sha256(scaler_path),
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ValueError(
                "diagnostic checkpoint provenance mismatch: {}".format(
                    key
                )
            )
    if expected["selection_mode"] != "learned":
        raise ValueError("primary SG2 diagnosis requires learned selection")
    return expected


def _control_arguments(args, name):
    return (
        getattr(args, "{}_train_manifest".format(name)),
        getattr(args, "{}_validation_manifest".format(name)),
        getattr(args, "{}_scaler".format(name)),
    )


def _load_selection_controls(args, learned_train, learned_validation):
    controls = {
        "learned": {
            "train": learned_train,
            "validation": learned_validation,
        }
    }
    for name in ("random", "full"):
        values = _control_arguments(args, name)
        present = [value is not None for value in values]
        if any(present) and not all(present):
            raise ValueError(
                "{} selection control requires two manifests and scaler".format(
                    name
                )
            )
        if all(present):
            train = SVSignedGINDataset(values[0], values[2])
            validation = SVSignedGINDataset(values[1], values[2])
            if (
                train.split != "train"
                or validation.split != "validation"
                or train.manifest["selection_mode"] != name
                or validation.manifest["selection_mode"] != name
            ):
                raise ValueError(
                    "{} selection control provenance is invalid".format(
                        name
                    )
                )
            controls[name] = {
                "train": train,
                "validation": validation,
            }
    return controls


def _progress(split):
    last = [0]

    def report(completed, total):
        if (
            completed == total
            or completed == 1
            or completed - last[0] >= 25
        ):
            print(
                "{} diagnostic {}/{}".format(split, completed, total),
                flush=True,
            )
            last[0] = completed

    return report


def main():
    args = parse_args()
    if args.seed < 0:
        raise ValueError("diagnostic seed cannot be negative")
    if (
        args.max_train_samples < 0
        or args.max_validation_samples < 0
    ):
        raise ValueError("diagnostic sample limits cannot be negative")
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("bottleneck diagnostic output exists")
    device = _device(args.device)
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint_sha_before = file_sha256(checkpoint_path)
    raw = _trusted_load(checkpoint_path, torch.device("cpu"))
    model = SVSignedGINClassifier(
        SVSignedGINConfig(**raw["model_config"])
    ).to(device)
    checkpoint = load_sv_signed_gin_checkpoint(
        checkpoint_path, model, device
    )
    if model.config.variant != "signed_gin_static_variation":
        raise ValueError(
            "one-shot bottleneck diagnosis requires SG2 checkpoint"
        )
    if model.config.message_mode != "signed_weighted":
        raise ValueError(
            "signed cancellation diagnosis requires signed_weighted mode"
        )
    thresholds = checkpoint.get("validation_thresholds", {})
    if "balanced_accuracy" not in thresholds:
        raise ValueError(
            "SG2 checkpoint has no frozen validation BA threshold"
        )
    train = SVSignedGINDataset(
        args.train_manifest,
        args.scaler,
        max_samples=(
            args.max_train_samples
            if args.max_train_samples > 0
            else None
        ),
    )
    validation = SVSignedGINDataset(
        args.validation_manifest,
        args.scaler,
        max_samples=(
            args.max_validation_samples
            if args.max_validation_samples > 0
            else None
        ),
    )
    provenance = _validate_primary_inputs(
        train, validation, args.scaler, checkpoint
    )
    fingerprint_before = _parameter_fingerprint(model)
    started = time.perf_counter()
    print(
        "START read-only SG2 bottleneck diagnostic device={}".format(
            device
        ),
        flush=True,
    )
    train_collection = collect_sv_diagnostics(
        model, train, device, _progress("train")
    )
    validation_collection = collect_sv_diagnostics(
        model, validation, device, _progress("validation")
    )
    result = analyze_sv_signed_gin_bottleneck(
        train_collection,
        validation_collection,
        model,
        float(thresholds["balanced_accuracy"]),
        device,
        seed=args.seed,
    )
    controls = _load_selection_controls(args, train, validation)
    selection_rows = (
        selection_control_probe(controls, args.seed)
        if len(controls) > 1
        else []
    )
    fingerprint_after = _parameter_fingerprint(model)
    checkpoint_sha_after = file_sha256(checkpoint_path)
    if (
        fingerprint_after != fingerprint_before
        or checkpoint_sha_after != checkpoint_sha_before
    ):
        raise RuntimeError("read-only diagnostic changed frozen inputs")
    result["provenance"] = {
        **provenance,
        "checkpoint_sha256": checkpoint_sha_before,
        "train_sample_count": len(train),
        "validation_sample_count": len(validation),
    }
    result["immutability"] = {
        "parameter_fingerprint_unchanged": True,
        "checkpoint_sha256_before": checkpoint_sha_before,
        "checkpoint_sha256_after": checkpoint_sha_after,
        "checkpoint_unchanged": True,
    }
    result["elapsed_seconds"] = float(time.perf_counter() - started)
    paths = write_sv_signed_gin_bottleneck_artifacts(
        result, output_dir, selection_rows
    )
    all_row = next(
        row
        for row in result["channel_masking"]
        if row["condition"] == "all"
    )
    mask_gin = next(
        row
        for row in result["channel_masking"]
        if row["condition"] == "mask_gin"
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "test_used": False,
                "parameter_update_count": 0,
                "validation_all_auc": all_row["roc_auc"],
                "validation_mask_gin_auc": mask_gin["roc_auc"],
                "mask_gin_delta_auc": mask_gin[
                    "delta_auc_vs_all"
                ],
                "elapsed_seconds": result["elapsed_seconds"],
                "outputs": paths,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
