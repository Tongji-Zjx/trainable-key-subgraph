"""Cache frozen G2/static logits and delta-Q transition summaries."""

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
from keysubgraph.data.g2_safeq import (  # noqa: E402
    G2_SAFEQ_FEATURE_SCHEMA_VERSION,
    G2_SAFEQ_MANIFEST_SCHEMA_VERSION,
)
from keysubgraph.data.sv_signed_gin_dataset import (  # noqa: E402
    create_sv_signed_gin_loader,
)
from keysubgraph.data.sv_spectral_diffusion import (  # noqa: E402
    SVSpectralDiffusionAugmentedDataset,
)
from keysubgraph.models.g2_safeq import (  # noqa: E402
    aggregate_transition_hidden,
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
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument("--spectral-manifest", type=Path, required=True)
    parser.add_argument("--spectral-scaler", type=Path, required=True)
    parser.add_argument("--g2-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _trusted_load(path, device):
    try:
        return torch.load(
            str(Path(path).resolve()), map_location=device, weights_only=False
        )
    except TypeError:
        return torch.load(str(Path(path).resolve()), map_location=device)


def _atomic_json(path, payload):
    temporary = Path(path).with_suffix(Path(path).suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def _atomic_torch(path, payload):
    temporary = Path(path).with_suffix(Path(path).suffix + ".tmp")
    torch.save(payload, str(temporary))
    os.replace(str(temporary), str(path))


def main():
    args = parse_args()
    if int(args.batch_size) < 1 or int(args.num_workers) < 0:
        raise ValueError("invalid SafeQ precompute loader configuration")
    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "manifest.json"
    feature_path = output_dir / "features.pt"
    if (manifest_path.exists() or feature_path.exists()) and not args.overwrite:
        raise FileExistsError("SafeQ cache already exists")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    raw = _trusted_load(args.g2_checkpoint, device)
    model = SVSignedGINClassifier(
        SVSignedGINConfig(**raw["model_config"])
    ).to(device)
    checkpoint = load_sv_signed_gin_checkpoint(
        args.g2_checkpoint, model, device
    )
    if model.config.variant != "svg_v2_g2_signed_delta_q":
        raise ValueError("SafeQ requires the frozen G2-D16 checkpoint")
    if model.signed_delta_q_head is None:
        raise ValueError("SafeQ requires the signed delta-Q auxiliary head")
    dataset = SVSpectralDiffusionAugmentedDataset(
        args.manifest,
        args.scaler,
        args.spectral_manifest,
        args.spectral_scaler,
        include_windows=True,
    )
    if dataset.split != args.split:
        raise ValueError("SafeQ precompute split mismatch")
    expected = checkpoint["provenance"]
    checks = (
        expected["protocol_sha256"] == dataset.manifest["protocol_sha256"],
        expected["selector_checkpoint_sha256"]
        == dataset.manifest["selector_checkpoint_sha256"],
        expected["selection_mode"] == dataset.manifest["selection_mode"],
        int(expected["selection_seed"])
        == int(dataset.manifest["selection_seed"]),
        expected["scaler_sha256"] == file_sha256(args.scaler),
        expected["spectral_scaler_sha256"]
        == file_sha256(args.spectral_scaler),
    )
    if not all(checks):
        raise ValueError("SafeQ frozen G2 provenance mismatch")
    split_manifest_key = "spectral_{}_manifest_sha256".format(args.split)
    if split_manifest_key in expected and (
        expected[split_manifest_key] != file_sha256(args.spectral_manifest)
    ):
        raise ValueError("SafeQ spectral manifest provenance mismatch")
    loader = create_sv_signed_gin_loader(
        dataset,
        batch_size=args.batch_size,
        seed=int(checkpoint["training_config"]["seed"]),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    metadata = {
        key: (str(site), str(subject))
        for key, site, subject in zip(
            dataset.sample_keys, dataset.sites, dataset.subject_ids
        )
    }
    values = {}
    model.eval()
    with torch.no_grad():
        for cpu_batch in loader:
            batch = cpu_batch.to(device)
            result = model(batch)
            if result.branch_logits is None or "static_spectral" not in result.branch_logits:
                raise RuntimeError("SafeQ G2 output has no static-spectral branch")
            summaries, valid = aggregate_transition_hidden(
                result.signed_delta_q_hidden,
                result.signed_delta_q_sample_indices,
                len(cpu_batch),
                model.config.gin_hidden_dim,
            )
            base = result.logits[:, 1] - result.logits[:, 0]
            static_branch = result.branch_logits["static_spectral"]
            static = static_branch[:, 1] - static_branch[:, 0]
            for index, sample in enumerate(cpu_batch):
                site, subject = metadata[sample.sample_key]
                values[sample.sample_key] = {
                    "site": site,
                    "subject_id": subject,
                    "label": int(sample.label),
                    "base_logit": base[index].detach().cpu(),
                    "static_logit": static[index].detach().cpu(),
                    "transition_summary": summaries[index].detach().cpu(),
                    "has_valid_transition": valid[index].detach().cpu(),
                }
    keys = tuple(dataset.sample_keys)
    if set(keys) != set(values) or len(keys) != len(values):
        raise RuntimeError("SafeQ precompute did not cover the split exactly")
    payload = {
        "artifact_type": "g2_safeq_features",
        "schema_version": G2_SAFEQ_FEATURE_SCHEMA_VERSION,
        "split": args.split,
        "sample_keys": keys,
        "sites": tuple(values[key]["site"] for key in keys),
        "subject_ids": tuple(values[key]["subject_id"] for key in keys),
        "labels": torch.tensor(
            [values[key]["label"] for key in keys], dtype=torch.long
        ),
        "base_logits": torch.stack(
            [values[key]["base_logit"] for key in keys]
        ).to(torch.float32),
        "static_logits": torch.stack(
            [values[key]["static_logit"] for key in keys]
        ).to(torch.float32),
        "transition_summaries": torch.stack(
            [values[key]["transition_summary"] for key in keys]
        ).to(torch.float32),
        "has_valid_transition": torch.stack(
            [values[key]["has_valid_transition"] for key in keys]
        ).to(torch.bool),
    }
    _atomic_torch(feature_path, payload)
    manifest = {
        "artifact_type": "g2_safeq_manifest",
        "schema_version": G2_SAFEQ_MANIFEST_SCHEMA_VERSION,
        "split": args.split,
        "sample_count": len(keys),
        "transition_hidden_dim": int(model.config.gin_hidden_dim),
        "summary_dim": int(payload["transition_summaries"].shape[1]),
        "feature_file": feature_path.name,
        "feature_sha256": file_sha256(feature_path),
        "protocol_sha256": expected["protocol_sha256"],
        "selector_checkpoint_sha256": expected[
            "selector_checkpoint_sha256"
        ],
        "selection_mode": expected["selection_mode"],
        "selection_seed": int(expected["selection_seed"]),
        "g2_checkpoint_sha256": file_sha256(args.g2_checkpoint),
        "g2_manifest_sha256": file_sha256(args.manifest),
        "g2_scaler_sha256": file_sha256(args.scaler),
        "g2_spectral_manifest_sha256": file_sha256(
            args.spectral_manifest
        ),
        "g2_spectral_scaler_sha256": file_sha256(args.spectral_scaler),
        "g2_variant": model.config.variant,
        "frozen_g2": True,
        "train_only_scalers": True,
        "uses_site_input": False,
        "uses_coordinates": False,
    }
    _atomic_json(manifest_path, manifest)
    print(
        "SafeQ cache: {} samples={} valid_transitions={}/{}".format(
            manifest_path,
            len(keys),
            int(payload["has_valid_transition"].sum()),
            len(keys),
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
