"""Cache aligned frozen G2 and author short-term representations for F2."""

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

from keysubgraph.data.data_protocol import (  # noqa: E402
    protocol_node_name_policy,
    validate_data_protocol,
)
from keysubgraph.data.data_split import file_sha256  # noqa: E402
from keysubgraph.data.graph_dataset import GraphSequenceDataset  # noqa: E402
from keysubgraph.data.sv_signed_gin_dataset import (  # noqa: E402
    create_sv_signed_gin_loader,
)
from keysubgraph.data.sv_spectral_diffusion import (  # noqa: E402
    SVSpectralDiffusionAugmentedDataset,
)
from keysubgraph.data.svg_short_term_representation_f2 import (  # noqa: E402
    SVG_SHORT_TERM_REPRESENTATION_F2_FEATURE_SCHEMA_VERSION,
    SVG_SHORT_TERM_REPRESENTATION_F2_MANIFEST_SCHEMA_VERSION,
)
from keysubgraph.models.sv_signed_gin import (  # noqa: E402
    SVSignedGINClassifier,
    SVSignedGINConfig,
)
from keysubgraph.training.author_short_term_trainer import (  # noqa: E402
    create_author_short_term_evaluation_loader,
    model_from_author_short_term_checkpoint,
)
from keysubgraph.training.sv_signed_gin_trainer import (  # noqa: E402
    load_sv_signed_gin_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--short-term-checkpoint", type=Path, required=True)
    parser.add_argument("--g2-manifest", type=Path, required=True)
    parser.add_argument("--g2-scaler", type=Path, required=True)
    parser.add_argument("--g2-spectral-manifest", type=Path, required=True)
    parser.add_argument("--g2-spectral-scaler", type=Path, required=True)
    parser.add_argument("--g2-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--short-term-batch-size", type=int, default=16)
    parser.add_argument("--g2-batch-size", type=int, default=4)
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


def _collect_short_term(model, loader, device):
    model.eval()
    values = {}
    with torch.no_grad():
        for cpu_batch in loader:
            batch = cpu_batch.to(device)
            output = model(batch)
            logits = output.logits.detach().cpu()
            representations = output.final_representation.detach().cpu()
            for index, sample in enumerate(cpu_batch.samples):
                values[sample.sample_key] = {
                    "label": int(sample.label),
                    "site": str(sample.site),
                    "subject_id": str(sample.subject_id),
                    "logit": logits[index].clone(),
                    "representation": representations[index].clone(),
                }
    return values


def _collect_g2(model, loader, dataset, device):
    model.eval()
    metadata = {
        key: (str(site), str(subject))
        for key, site, subject in zip(
            dataset.sample_keys, dataset.sites, dataset.subject_ids
        )
    }
    values = {}
    with torch.no_grad():
        for cpu_batch in loader:
            batch = cpu_batch.to(device)
            output = model(batch)
            logits = output.logits.detach().cpu()
            representations = output.final_representation.detach().cpu()
            for index, sample in enumerate(cpu_batch):
                site, subject = metadata[sample.sample_key]
                values[sample.sample_key] = {
                    "label": int(sample.label),
                    "site": site,
                    "subject_id": subject,
                    "anchor_logit": (
                        logits[index, 1] - logits[index, 0]
                    ).clone(),
                    "representation": representations[index].clone(),
                }
    return values


def main():
    args = parse_args()
    if (
        args.short_term_batch_size < 1
        or args.g2_batch_size < 1
        or args.num_workers < 0
    ):
        raise ValueError("invalid representation F2 loader configuration")
    output = args.output_dir.resolve()
    manifest_path = output / "manifest.json"
    feature_path = output / "features.pt"
    if (manifest_path.exists() or feature_path.exists()) and not args.overwrite:
        raise FileExistsError("representation F2 cache already exists")
    output.mkdir(parents=True, exist_ok=True)
    protocol = validate_data_protocol(args.protocol, PROJECT_ROOT)
    protocol_hash = file_sha256(args.protocol)
    device = torch.device(args.device)

    short_model, short_checkpoint = model_from_author_short_term_checkpoint(
        args.short_term_checkpoint, device
    )
    if short_checkpoint.get("protocol_sha256") != protocol_hash:
        raise ValueError("short-term checkpoint protocol mismatch")
    paths = protocol["paths"]
    short_dataset = GraphSequenceDataset(
        PROJECT_ROOT / paths["dataset_root"],
        PROJECT_ROOT / paths["sample_index_csv"],
        PROJECT_ROOT / paths["splits_csv"],
        args.split,
        protocol["edge_presence_threshold"],
        node_name_policy=protocol_node_name_policy(protocol),
    )
    short_loader = create_author_short_term_evaluation_loader(
        short_dataset,
        args.short_term_batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    raw_g2 = _trusted_load(args.g2_checkpoint, device)
    g2_model = SVSignedGINClassifier(
        SVSignedGINConfig(**raw_g2["model_config"])
    ).to(device)
    g2_checkpoint = load_sv_signed_gin_checkpoint(
        args.g2_checkpoint, g2_model, device
    )
    if g2_model.config.variant != "svg_v2_g2_signed_delta_q":
        raise ValueError("representation F2 requires the frozen G2-D16 model")
    g2_dataset = SVSpectralDiffusionAugmentedDataset(
        args.g2_manifest,
        args.g2_scaler,
        args.g2_spectral_manifest,
        args.g2_spectral_scaler,
        include_windows=True,
    )
    if g2_dataset.split != args.split:
        raise ValueError("representation F2 G2 split mismatch")
    expected = g2_checkpoint["provenance"]
    checks = (
        expected["protocol_sha256"] == protocol_hash,
        expected["protocol_sha256"] == g2_dataset.manifest["protocol_sha256"],
        expected["scaler_sha256"] == file_sha256(args.g2_scaler),
        expected["spectral_scaler_sha256"]
        == file_sha256(args.g2_spectral_scaler),
    )
    if not all(checks):
        raise ValueError("representation F2 G2 provenance mismatch")
    g2_loader = create_sv_signed_gin_loader(
        g2_dataset,
        batch_size=args.g2_batch_size,
        seed=int(g2_checkpoint["training_config"]["seed"]),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    short_values = _collect_short_term(short_model, short_loader, device)
    g2_values = _collect_g2(g2_model, g2_loader, g2_dataset, device)
    if set(short_values) != set(g2_values):
        raise ValueError("representation F2 branch sample sets differ")
    keys = sorted(short_values)
    for key in keys:
        left, right = short_values[key], g2_values[key]
        if (
            left["label"] != right["label"]
            or left["site"] != right["site"]
            or left["subject_id"] != right["subject_id"]
        ):
            raise ValueError("representation F2 branch identity mismatch")
    payload = {
        "artifact_type": "svg_short_term_representation_f2_features",
        "schema_version": SVG_SHORT_TERM_REPRESENTATION_F2_FEATURE_SCHEMA_VERSION,
        "split": args.split,
        "sample_keys": tuple(keys),
        "sites": tuple(short_values[key]["site"] for key in keys),
        "subject_ids": tuple(short_values[key]["subject_id"] for key in keys),
        "labels": torch.tensor(
            [short_values[key]["label"] for key in keys], dtype=torch.long
        ),
        "g2_anchor_logits": torch.stack(
            [g2_values[key]["anchor_logit"] for key in keys], dim=0
        ).to(torch.float32),
        "g2_representations": torch.stack(
            [g2_values[key]["representation"] for key in keys], dim=0
        ).to(torch.float32),
        "short_term_representations": torch.stack(
            [short_values[key]["representation"] for key in keys], dim=0
        ).to(torch.float32),
    }
    _atomic_torch(feature_path, payload)
    manifest = {
        "artifact_type": "svg_short_term_representation_f2_manifest",
        "schema_version": SVG_SHORT_TERM_REPRESENTATION_F2_MANIFEST_SCHEMA_VERSION,
        "split": args.split,
        "sample_count": len(keys),
        "g2_representation_dim": int(payload["g2_representations"].shape[1]),
        "short_term_representation_dim": int(
            payload["short_term_representations"].shape[1]
        ),
        "feature_file": feature_path.name,
        "feature_sha256": file_sha256(feature_path),
        "protocol_sha256": protocol_hash,
        "short_term_checkpoint_sha256": file_sha256(args.short_term_checkpoint),
        "g2_checkpoint_sha256": file_sha256(args.g2_checkpoint),
        "g2_manifest_sha256": file_sha256(args.g2_manifest),
        "g2_scaler_sha256": file_sha256(args.g2_scaler),
        "g2_spectral_manifest_sha256": file_sha256(
            args.g2_spectral_manifest
        ),
        "g2_spectral_scaler_sha256": file_sha256(args.g2_spectral_scaler),
        "g2_variant": g2_model.config.variant,
        "frozen_encoders": True,
        "uses_site_input": False,
        "uses_coordinates": False,
    }
    _atomic_json(manifest_path, manifest)
    print(
        "representation F2 cache: {} samples={} g2_dim={} short_dim={}".format(
            manifest_path,
            len(keys),
            manifest["g2_representation_dim"],
            manifest["short_term_representation_dim"],
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

