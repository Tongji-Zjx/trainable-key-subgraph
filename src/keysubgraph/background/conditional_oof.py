"""Leakage-safe derived TGE caches for conditional development OOF.

The selector and trajectory cache are treated as frozen upstream artifacts.
Only the downstream TGE readout and background branch are refitted.  TGE
samples persist their split inside the serialized dataclass, so a valid inner
validation or OOF target cannot be created by editing JSON alone.  This module
therefore materializes split-relabeled *derived* artifacts while leaving every
source cache untouched.
"""

from __future__ import absolute_import, division, print_function

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import torch

from keysubgraph.tge.dataset import (
    file_sha256,
    read_tge_manifest,
    tge_subject_group_key,
    trusted_torch_load,
)
from keysubgraph.tge.types import TGESample


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def _stable_key(seed: int, stratum: str, group: str) -> str:
    value = "{}|{}|{}".format(int(seed), str(stratum), str(group))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stratified_inner_groups(
    records: Sequence[Mapping[str, object]],
    validation_fraction: float = 0.20,
    seed: int = 20260905,
) -> Tuple[set, set]:
    """Split subject groups using deterministic site-by-label strata.

    A stratum containing at least two subject groups contributes at least one
    group to both inner splits.  Singletons remain in training; this avoids
    inventing duplicated observations merely to preserve a rare stratum.
    """

    if not 0.0 < float(validation_fraction) < 0.5:
        raise ValueError("inner validation fraction must lie in (0,0.5)")
    grouped: Dict[str, list] = {}
    group_metadata: Dict[str, Tuple[str, int]] = {}
    for row in records:
        group = tge_subject_group_key(dict(row))
        site = str(row.get("site", ""))
        label = int(row.get("label", -1))
        if not site or label not in (0, 1):
            raise ValueError("conditional OOF records require site and binary label")
        previous = group_metadata.setdefault(group, (site, label))
        if previous != (site, label):
            raise ValueError("a subject group spans site or label strata")
        grouped.setdefault(group, []).append(row)
    if len(grouped) < 4:
        raise ValueError("conditional OOF requires at least four subject groups")

    strata: Dict[str, list] = {}
    for group, (site, label) in group_metadata.items():
        strata.setdefault("{}|{}".format(site, label), []).append(group)

    inner_validation = set()
    for stratum, groups in sorted(strata.items()):
        ordered = sorted(groups, key=lambda item: _stable_key(seed, stratum, item))
        if len(ordered) < 2:
            continue
        requested = int(round(len(ordered) * float(validation_fraction)))
        count = max(1, min(len(ordered) - 1, requested))
        inner_validation.update(ordered[:count])
    inner_train = set(grouped) - inner_validation
    if not inner_train or not inner_validation:
        raise ValueError("conditional OOF inner split is empty")

    def labels(groups: Iterable[str]) -> set:
        return {group_metadata[group][1] for group in groups}

    if labels(inner_train) != {0, 1} or labels(inner_validation) != {0, 1}:
        raise ValueError("conditional OOF inner splits must contain both classes")
    return inner_train, inner_validation


def _resolve_artifact(manifest_path: Path, record: Mapping[str, object]) -> Path:
    path = Path(str(record["artifact_path"]))
    if not path.is_absolute():
        path = (manifest_path.parent / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _derived_manifest(
    source: Mapping[str, object],
    source_manifest: Path,
    records: Sequence[Mapping[str, object]],
    split: str,
    derivation: Mapping[str, object],
) -> dict:
    payload = {
        key: value
        for key, value in source.items()
        if key not in ("records", "sample_count", "split", "_manifest_path")
    }
    payload.update(
        {
            "artifact_type": "tge_preprocessed_manifest",
            "split": str(split),
            "sample_count": len(records),
            "records": [dict(row) for row in records],
            "conditional_oof_derivation": dict(derivation),
            "source_manifest": str(source_manifest.resolve()),
            "source_manifest_sha256": file_sha256(source_manifest),
        }
    )
    return payload


def _materialize_relabelled_records(
    source_manifest: Path,
    records: Sequence[Mapping[str, object]],
    output_dir: Path,
    split: str,
    role: str,
) -> list:
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    output = []
    for row in records:
        source_path = _resolve_artifact(source_manifest, row)
        expected_hash = str(row.get("artifact_sha256", ""))
        if expected_hash and file_sha256(source_path) != expected_hash:
            raise ValueError("source TGE artifact hash mismatch")
        sample = trusted_torch_load(source_path)
        if not isinstance(sample, TGESample):
            raise ValueError("source TGE artifact has unexpected payload")
        if sample.sample_key != str(row["sample_key"]):
            raise ValueError("source TGE artifact identity mismatch")
        provenance = dict(sample.provenance)
        provenance["conditional_oof_split_derivation"] = {
            "role": str(role),
            "source_split": str(sample.split),
            "derived_split": str(split),
            "source_artifact": str(source_path),
            "source_artifact_sha256": file_sha256(source_path),
        }
        derived = replace(sample, split=str(split), provenance=provenance)
        filename = hashlib.sha256(
            (str(role) + "|" + str(sample.sample_key)).encode("utf-8")
        ).hexdigest()[:20] + ".pt"
        target = sample_dir / filename
        temporary = target.with_suffix(".pt.tmp")
        torch.save(derived, str(temporary))
        os.replace(str(temporary), str(target))
        updated = dict(row)
        updated["split"] = str(split)
        updated["artifact_path"] = str(target.relative_to(output_dir))
        updated["artifact_sha256"] = file_sha256(target)
        updated["conditional_oof_source_artifact_sha256"] = file_sha256(
            source_path
        )
        output.append(updated)
    return output


def build_conditional_oof_cache(
    train_manifest: Path,
    target_manifest: Path,
    output_dir: Path,
    validation_fraction: float = 0.20,
    seed: int = 20260905,
) -> dict:
    """Create one nested split without modifying either source manifest."""

    train_manifest = Path(train_manifest).resolve()
    target_manifest = Path(target_manifest).resolve()
    output_dir = Path(output_dir).resolve()
    source_train = read_tge_manifest(train_manifest, "train")
    source_target = read_tge_manifest(target_manifest, "validation")

    target_groups = {
        tge_subject_group_key(row) for row in source_target["records"]
    }
    source_groups = {
        tge_subject_group_key(row) for row in source_train["records"]
    }
    if target_groups.intersection(source_groups):
        raise ValueError("conditional OOF target overlaps source training groups")
    if source_train.get("protocol_sha256") != source_target.get("protocol_sha256"):
        raise ValueError("conditional OOF source protocol mismatch")
    if source_train.get("selector_checkpoint_sha256") != source_target.get(
        "selector_checkpoint_sha256"
    ):
        raise ValueError("conditional OOF source selector mismatch")

    inner_train_groups, inner_validation_groups = stratified_inner_groups(
        source_train["records"], validation_fraction=validation_fraction, seed=seed
    )
    inner_train_rows = []
    inner_validation_source_rows = []
    for row in source_train["records"]:
        group = tge_subject_group_key(row)
        if group in inner_train_groups:
            updated = dict(row)
            updated["artifact_path"] = str(_resolve_artifact(train_manifest, row))
            inner_train_rows.append(updated)
        elif group in inner_validation_groups:
            inner_validation_source_rows.append(row)
        else:
            raise RuntimeError("conditional OOF group allocation is incomplete")

    derivation = {
        "artifact_type": "mokse_conditional_oof_split_derivation_v1",
        "conditional_on_frozen_selector_and_trajectory_cache": True,
        "end_to_end_selector_oof": False,
        "validation_fraction": float(validation_fraction),
        "seed": int(seed),
        "target_source_split": "validation",
        "target_derived_split": "test",
    }
    inner_train_path = output_dir / "inner_train" / "manifest.json"
    inner_validation_dir = output_dir / "inner_validation"
    target_dir = output_dir / "oof_target"
    inner_validation_rows = _materialize_relabelled_records(
        train_manifest,
        inner_validation_source_rows,
        inner_validation_dir,
        "validation",
        "checkpoint_validation",
    )
    target_rows = _materialize_relabelled_records(
        target_manifest,
        source_target["records"],
        target_dir,
        "test",
        "development_oof_target",
    )
    _atomic_json(
        inner_train_path,
        _derived_manifest(
            source_train, train_manifest, inner_train_rows, "train", derivation
        ),
    )
    _atomic_json(
        inner_validation_dir / "manifest.json",
        _derived_manifest(
            source_train,
            train_manifest,
            inner_validation_rows,
            "validation",
            derivation,
        ),
    )
    _atomic_json(
        target_dir / "manifest.json",
        _derived_manifest(
            source_target, target_manifest, target_rows, "test", derivation
        ),
    )

    report = {
        "artifact_type": "mokse_conditional_oof_cache_v1",
        "conditional_on_frozen_selector_and_trajectory_cache": True,
        "end_to_end_selector_oof": False,
        "source_train_manifest": str(train_manifest),
        "source_target_manifest": str(target_manifest),
        "manifests": {
            "inner_train": str(inner_train_path),
            "inner_validation": str(inner_validation_dir / "manifest.json"),
            "oof_target": str(target_dir / "manifest.json"),
        },
        "sample_count": {
            "inner_train": len(inner_train_rows),
            "inner_validation": len(inner_validation_rows),
            "oof_target": len(target_rows),
        },
        "group_count": {
            "inner_train": len(inner_train_groups),
            "inner_validation": len(inner_validation_groups),
            "oof_target": len(target_groups),
        },
        "all_downstream_fit_and_target_groups_disjoint": True,
        "derivation": derivation,
    }
    _atomic_json(output_dir / "derivation.json", report)
    return report

