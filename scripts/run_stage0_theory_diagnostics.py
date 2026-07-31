"""Run one fold of the frozen full-vs-hard Stage-0 SGW diagnosis."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
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
from keysubgraph.data.exact_stse_dataset import (  # noqa: E402
    ExactSTSEDataset,
)
from keysubgraph.data.sv_signed_gin_artifact import (  # noqa: E402
    load_sv_signed_gin_record,
)
from keysubgraph.theory.class_margin_diagnostics import (  # noqa: E402
    apply_standardizer,
    class_margin_metrics,
    component_margin_metrics,
    fit_train_only_standardizer,
    stratified_paired_bootstrap,
)
from keysubgraph.theory.sgw_core_features import (  # noqa: E402
    SGWCoreConfig,
    compute_sgw_core_sequence,
    load_stage0_sample_artifact,
    save_stage0_sample_artifact,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--hard-train-manifest", type=Path, required=True)
    parser.add_argument("--hard-test-manifest", type=Path, required=True)
    parser.add_argument("--fold-assignments", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260801)
    parser.add_argument("--laplacian-eta", type=float, default=1.0e-3)
    parser.add_argument("--diffusion-time", type=float, default=1.0)
    parser.add_argument("--gw-entropic-reg", type=float, default=1.0e-2)
    parser.add_argument("--gw-max-iter", type=int, default=100)
    parser.add_argument("--gw-sinkhorn-iter", type=int, default=100)
    parser.add_argument("--gw-tolerance", type=float, default=1.0e-7)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _atomic_json(path, payload):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            _json_ready(payload),
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    os.replace(str(temporary), str(path))


def _atomic_csv(path, rows, fieldnames):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def _atomic_npz(path, **arrays):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(str(temporary), str(path))


def _json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _code_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            universal_newlines=True,
        ).strip()
    except Exception:
        return "unknown"


def _sample_filename(sample_key):
    return hashlib.sha256(str(sample_key).encode("utf-8")).hexdigest() + ".pt"


def _read_manifest(path, expected_split, protocol_sha256):
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if (
        payload.get("artifact_type")
        != "sv_hard_sgw_signed_gin_manifest"
        or payload.get("split") != expected_split
        or payload.get("protocol_sha256") != protocol_sha256
    ):
        raise ValueError("Stage-0 hard manifest provenance mismatch")
    rows = payload.get("records", [])
    if len(rows) != int(payload.get("sample_count", -1)):
        raise ValueError("Stage-0 hard manifest count mismatch")
    result = {}
    for row in rows:
        key = str(row["sample_key"])
        if key in result:
            raise ValueError("Stage-0 hard manifest has duplicate samples")
        feature_path = Path(row["feature_path"])
        if not feature_path.is_absolute():
            feature_path = path.parent / feature_path
        result[key] = (row, feature_path.resolve())
    return payload, result


def _expected_fold_keys(path, fold, role):
    with Path(path).resolve().open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("purpose") != "confirmatory_cross_fitted_fold_roles":
        raise ValueError("Stage-0 requires confirmatory fold assignments")
    rows = [
        row
        for row in payload.get("assignments", [])
        if int(row["outer_fold"]) == int(fold) and row["role"] == role
    ]
    keys = {str(row["sample_key"]) for row in rows}
    if len(keys) != len(rows) or not keys:
        raise ValueError("Stage-0 fold role is empty or duplicated")
    return keys


def _edge_count(adjacency, threshold):
    mask = adjacency.abs() > float(threshold)
    mask = torch.triu(mask, diagonal=1)
    return int(mask.sum().item())


def _retention(full_adjacencies, hard_adjacencies, threshold):
    node_values = []
    edge_values = []
    for full, hard in zip(full_adjacencies, hard_adjacencies):
        if hard is None:
            continue
        node_values.append(
            float(hard.shape[0]) / float(max(1, full.shape[0]))
        )
        full_edges = _edge_count(full, threshold)
        hard_edges = _edge_count(hard, threshold)
        edge_values.append(
            float(hard_edges) / float(max(1, full_edges))
        )
    if not node_values:
        return 0.0, 0.0
    return (
        sum(node_values) / float(len(node_values)),
        sum(edge_values) / float(len(edge_values)),
    )


def _side_payload(result):
    return {
        "core": result.core.detach().cpu(),
        "window_quantiles": result.window_quantiles.detach().cpu(),
        "window_mask": result.window_mask.detach().cpu(),
        "transition_features": result.transition_features.detach().cpu(),
        "transition_mask": result.transition_mask.detach().cpu(),
        "gw_solver_converged": tuple(result.gw_solver_converged),
        "valid_transition_count": result.valid_transition_count,
    }


def _process_split(
    protocol,
    protocol_path,
    hard_manifest_path,
    fold_assignments,
    fold,
    split,
    role,
    output_dir,
    device,
    config,
    max_samples,
    overwrite,
):
    protocol_sha256 = file_sha256(protocol_path)
    hard_manifest, hard_rows = _read_manifest(
        hard_manifest_path, split, protocol_sha256
    )
    expected_keys = _expected_fold_keys(fold_assignments, fold, role)
    if set(hard_rows) != expected_keys:
        raise ValueError(
            "Stage-0 hard manifest does not match frozen {} role".format(role)
        )
    paths = protocol["paths"]
    dataset = ExactSTSEDataset(
        PROJECT_ROOT / paths["dataset_root"],
        PROJECT_ROOT / paths["sample_index_csv"],
        PROJECT_ROOT / paths["splits_csv"],
        split,
        protocol["edge_presence_threshold"],
        require_coordinates=False,
        node_name_policy=protocol_node_name_policy(protocol),
    )
    if set(item.sample_key for item in dataset.assignments) != expected_keys:
        raise ValueError("Stage-0 dataset does not match frozen fold role")
    selected_count = len(dataset)
    if max_samples is not None:
        if max_samples < 1:
            raise ValueError("Stage-0 max samples must be positive")
        selected_count = min(selected_count, int(max_samples))
    cache_dir = Path(output_dir).resolve() / "cache" / split
    cache_dir.mkdir(parents=True, exist_ok=True)
    extractor = config.build_extractor()
    common_provenance = {
        "fold": int(fold),
        "protocol_sha256": protocol_sha256,
        "fold_assignments_sha256": file_sha256(fold_assignments),
        "hard_manifest_sha256": file_sha256(hard_manifest_path),
        "selector_checkpoint_sha256": hard_manifest[
            "selector_checkpoint_sha256"
        ],
        "selection_mode": hard_manifest["selection_mode"],
        "selection_seed": int(hard_manifest["selection_seed"]),
        "feature_schema_sha256": config.schema_sha256(),
        "edge_presence_threshold": float(
            protocol["edge_presence_threshold"]
        ),
        "code_commit": _code_commit(),
    }
    artifacts = []
    for index in range(selected_count):
        sample = dataset[index].graph
        row, hard_path = hard_rows[sample.sample_key]
        if file_sha256(hard_path) != row["feature_sha256"]:
            raise ValueError("Stage-0 hard record hash mismatch")
        artifact_path = cache_dir / _sample_filename(sample.sample_key)
        expected_provenance = dict(common_provenance)
        expected_provenance["hard_record_sha256"] = row["feature_sha256"]
        if artifact_path.is_file() and not overwrite:
            cached = load_stage0_sample_artifact(artifact_path)
            if (
                cached.get("sample_key") != sample.sample_key
                or cached.get("provenance") != expected_provenance
            ):
                raise ValueError("existing Stage-0 cache provenance mismatch")
            artifacts.append(cached)
            print(
                "reused {} {}/{} {}".format(
                    split, index + 1, selected_count, sample.sample_key
                ),
                flush=True,
            )
            continue
        hard_record = load_sv_signed_gin_record(hard_path)
        if hard_record.sample_key != sample.sample_key:
            raise ValueError("Stage-0 full/hard sample identity mismatch")
        if len(hard_record.windows) != sample.num_timepoints:
            raise ValueError("Stage-0 full/hard window count mismatch")
        times = [float(value) for value in sample.window_starts.tolist()]
        for time, window in zip(times, hard_record.windows):
            if window is not None and abs(float(window.time_start) - time) > 1.0e-6:
                raise ValueError("Stage-0 full/hard window time mismatch")
        full_adjacencies = tuple(value.to(device) for value in sample.adjacency)
        hard_adjacencies = tuple(
            window.adjacency.to(device) if window is not None else None
            for window in hard_record.windows
        )
        full_result = compute_sgw_core_sequence(
            full_adjacencies,
            times,
            sample.edge_presence_threshold,
            config=config,
            extractor=extractor,
        )
        hard_result = compute_sgw_core_sequence(
            hard_adjacencies,
            times,
            sample.edge_presence_threshold,
            config=config,
            extractor=extractor,
        )
        masks_agree = torch.equal(
            full_result.transition_mask.cpu(),
            hard_result.transition_mask.cpu(),
        )
        eligible = bool(masks_agree and hard_result.valid_transition_count > 0)
        node_ratio, edge_ratio = _retention(
            full_adjacencies,
            hard_adjacencies,
            sample.edge_presence_threshold,
        )
        paired_error = float(
            torch.linalg.vector_norm(
                full_result.core - hard_result.core
            ).detach().cpu()
        )
        payload = {
            "sample_key": sample.sample_key,
            "sample_id": sample.sample_id,
            "subject_id": sample.subject_id,
            "site": sample.site,
            "label": int(sample.label),
            "split": split,
            "fold": int(fold),
            "eligible": eligible,
            "transition_mask_agreement": bool(masks_agree),
            "node_retention_rate": node_ratio,
            "edge_retention_rate": edge_ratio,
            "paired_core_error": paired_error,
            "valid_transition_count": hard_result.valid_transition_count,
            "full": _side_payload(full_result),
            "hard": _side_payload(hard_result),
            "provenance": expected_provenance,
        }
        save_stage0_sample_artifact(
            artifact_path, payload, overwrite=overwrite
        )
        artifacts.append(payload)
        print(
            "processed {} {}/{} {} transitions={} error={:.6f}".format(
                split,
                index + 1,
                selected_count,
                sample.sample_key,
                hard_result.valid_transition_count,
                paired_error,
            ),
            flush=True,
        )
        del full_result, hard_result, full_adjacencies, hard_adjacencies
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    return artifacts, common_provenance


def _arrays(artifacts):
    eligible = [item for item in artifacts if item["eligible"]]
    if not eligible:
        raise ValueError("Stage-0 split has no eligible samples")
    return {
        "sample_keys": np.asarray(
            [item["sample_key"] for item in eligible], dtype=str
        ),
        "sites": np.asarray([item["site"] for item in eligible], dtype=str),
        "subject_ids": np.asarray(
            [item["subject_id"] for item in eligible], dtype=str
        ),
        "labels": np.asarray(
            [item["label"] for item in eligible], dtype=np.int64
        ),
        "full": np.stack(
            [item["full"]["core"].numpy() for item in eligible]
        ).astype(np.float64),
        "hard": np.stack(
            [item["hard"]["core"].numpy() for item in eligible]
        ).astype(np.float64),
        "valid_transitions": np.asarray(
            [item["hard"]["valid_transition_count"] for item in eligible],
            dtype=np.int64,
        ),
        "node_retention": np.asarray(
            [item["node_retention_rate"] for item in eligible],
            dtype=np.float64,
        ),
        "edge_retention": np.asarray(
            [item["edge_retention_rate"] for item in eligible],
            dtype=np.float64,
        ),
    }


def _error_groups(artifacts, field):
    groups = defaultdict(list)
    for item in artifacts:
        if item["eligible"]:
            groups[str(item[field])].append(float(item["paired_core_error"]))
    return {
        key: {
            "sample_count": len(values),
            "mean_paired_error": sum(values) / float(len(values)),
        }
        for key, values in sorted(groups.items())
    }


def _quantile_error_groups(artifacts, field):
    eligible = [item for item in artifacts if item["eligible"]]
    values = np.asarray([float(item[field]) for item in eligible])
    boundaries = np.quantile(values, (0.25, 0.50, 0.75))
    groups = defaultdict(list)
    for item, value in zip(eligible, values):
        index = int(np.searchsorted(boundaries, value, side="right"))
        groups[str(index)].append(float(item["paired_core_error"]))
    return {
        key: {
            "sample_count": len(current),
            "mean_paired_error": sum(current) / float(len(current)),
        }
        for key, current in sorted(groups.items())
    }


def _report_markdown(fold, raw, standardized, bootstrap, eligible_count, total):
    interval = bootstrap["intervals"]
    lines = [
        "# SVG Stage 0 理论条件诊断（Fold {}）".format(fold),
        "",
        "- outer-test样本：{}".format(total),
        "- 理论指标有效样本：{}".format(eligible_count),
        "- 主ground metric：未标准化18维欧氏距离",
        "- Bootstrap：{}次，seed={}".format(
            bootstrap["repeats"], bootstrap["seed"]
        ),
        "",
        "| 指标 | Raw | Raw 95% CI | Train-only标准化敏感性 |",
        "|---|---:|---:|---:|",
    ]
    labels = (
        ("delta_full", "完整图类别间隔"),
        ("eta_0_pair", "类别0配对半径"),
        ("eta_1_pair", "类别1配对半径"),
        ("eta_0_ot", "类别0 OT半径"),
        ("eta_1_ot", "类别1 OT半径"),
        ("lower_bound_pair", "配对理论下界"),
        ("lower_bound_ot", "OT理论下界"),
        ("delta_hard", "硬图类别间隔"),
    )
    for name, label in labels:
        current = interval[name]
        lines.append(
            "| {} | {:.6f} | [{:.6f}, {:.6f}] | {:.6f} |".format(
                label,
                raw[name],
                current["lower_95"],
                current["upper_95"],
                standardized[name],
            )
        )
    lines.extend(
        [
            "",
            "## 数值检查",
            "",
            "- OT半径不大于配对上界：{}".format(
                "通过" if raw["checks"]["eta_ot_not_above_pair"] else "失败"
            ),
            "- 硬图间隔不低于OT下界：{}".format(
                "通过"
                if raw["checks"]["hard_margin_not_below_ot_lower_bound"]
                else "失败"
            ),
            "- 配对下界95% CI下界为正：{}".format(
                "是" if interval["lower_bound_pair"]["lower_95"] > 0.0 else "否"
            ),
            "- OT下界95% CI下界为正：{}".format(
                "是" if interval["lower_bound_ot"]["lower_95"] > 0.0 else "否"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main():
    args = parse_args()
    if args.fold < 0 or args.bootstrap_repeats < 1:
        raise ValueError("Stage-0 fold/repeat configuration is invalid")
    protocol_path = Path(args.protocol).resolve()
    protocol = validate_data_protocol(protocol_path, PROJECT_ROOT)
    device = torch.device(args.device)
    config = SGWCoreConfig(
        laplacian_eta=args.laplacian_eta,
        diffusion_time=args.diffusion_time,
        gw_entropic_reg=args.gw_entropic_reg,
        gw_max_iter=args.gw_max_iter,
        gw_sinkhorn_iter=args.gw_sinkhorn_iter,
        gw_tolerance=args.gw_tolerance,
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_artifacts, train_provenance = _process_split(
        protocol,
        protocol_path,
        args.hard_train_manifest,
        args.fold_assignments,
        args.fold,
        "train",
        "inner_train",
        output_dir,
        device,
        config,
        args.max_train_samples,
        args.overwrite,
    )
    test_artifacts, test_provenance = _process_split(
        protocol,
        protocol_path,
        args.hard_test_manifest,
        args.fold_assignments,
        args.fold,
        "test",
        "outer_test",
        output_dir,
        device,
        config,
        args.max_test_samples,
        args.overwrite,
    )
    train = _arrays(train_artifacts)
    test = _arrays(test_artifacts)
    scaler = fit_train_only_standardizer(train["full"])
    standardized_full = apply_standardizer(test["full"], scaler)
    standardized_hard = apply_standardizer(test["hard"], scaler)
    raw_metrics = class_margin_metrics(
        test["full"], test["hard"], test["labels"]
    )
    standardized_metrics = class_margin_metrics(
        standardized_full, standardized_hard, test["labels"]
    )
    components = component_margin_metrics(
        test["full"], test["hard"], test["labels"]
    )
    print(
        "START exact bootstrap repeats={}".format(args.bootstrap_repeats),
        flush=True,
    )
    bootstrap = stratified_paired_bootstrap(
        test["full"],
        test["hard"],
        test["labels"],
        repeats=args.bootstrap_repeats,
        seed=args.bootstrap_seed,
    )
    print("FINISH exact bootstrap", flush=True)

    _atomic_npz(
        output_dir / "train_full_core_features.npz",
        sample_keys=train["sample_keys"],
        labels=train["labels"],
        core=train["full"],
        valid_transition_count=train["valid_transitions"],
    )
    _atomic_npz(
        output_dir / "train_hard_core_features.npz",
        sample_keys=train["sample_keys"],
        labels=train["labels"],
        core=train["hard"],
        valid_transition_count=train["valid_transitions"],
    )
    _atomic_npz(
        output_dir / "full_core_features.npz",
        sample_keys=test["sample_keys"],
        labels=test["labels"],
        sites=test["sites"],
        subject_ids=test["subject_ids"],
        core=test["full"],
        valid_transition_count=test["valid_transitions"],
        feature_valid_mask=np.ones_like(test["full"], dtype=np.bool_),
    )
    _atomic_npz(
        output_dir / "hard_core_features.npz",
        sample_keys=test["sample_keys"],
        labels=test["labels"],
        sites=test["sites"],
        subject_ids=test["subject_ids"],
        core=test["hard"],
        valid_transition_count=test["valid_transitions"],
        feature_valid_mask=np.ones_like(test["hard"], dtype=np.bool_),
    )
    scaler_payload = dict(scaler)
    _atomic_json(output_dir / "train_only_core_scaler.json", scaler_payload)
    distribution_payload = {
        "raw_primary": raw_metrics,
        "train_only_standardized_sensitivity": standardized_metrics,
        "component_metrics": components,
    }
    _atomic_json(
        output_dir / "class_distribution_metrics.json",
        distribution_payload,
    )
    _atomic_json(output_dir / "bootstrap_metrics.json", bootstrap)

    error_rows = []
    for item in test_artifacts:
        full = item["full"]["core"].numpy()
        hard = item["hard"]["core"].numpy()
        absolute = np.abs(full - hard)
        error_rows.append(
            {
                "fold": int(args.fold),
                "sample_key": item["sample_key"],
                "subject_id": item["subject_id"],
                "site": item["site"],
                "label": int(item["label"]),
                "eligible": int(bool(item["eligible"])),
                "valid_transition_count": int(
                    item["hard"]["valid_transition_count"]
                ),
                "node_retention_rate": item["node_retention_rate"],
                "edge_retention_rate": item["edge_retention_rate"],
                "paired_core_error": item["paired_core_error"],
                "spectral_direction_error": float(absolute[:16].mean()),
                "spectral_speed_error": float(absolute[16]),
                "gw_speed_error": float(absolute[17]),
            }
        )
    _atomic_csv(
        output_dir / "sample_level_errors.csv",
        error_rows,
        (
            "fold",
            "sample_key",
            "subject_id",
            "site",
            "label",
            "eligible",
            "valid_transition_count",
            "node_retention_rate",
            "edge_retention_rate",
            "paired_core_error",
            "spectral_direction_error",
            "spectral_speed_error",
            "gw_speed_error",
        ),
    )
    decomposition = {
        "per_dimension_mean_absolute_error": np.abs(
            test["full"] - test["hard"]
        ).mean(axis=0),
        "by_class": _error_groups(test_artifacts, "label"),
        "by_site": _error_groups(test_artifacts, "site"),
        "by_valid_transition_count_quartile": _quantile_error_groups(
            test_artifacts, "valid_transition_count"
        ),
        "by_node_retention_quartile": _quantile_error_groups(
            test_artifacts, "node_retention_rate"
        ),
        "by_edge_retention_quartile": _quantile_error_groups(
            test_artifacts, "edge_retention_rate"
        ),
    }
    _atomic_json(output_dir / "error_decomposition.json", decomposition)
    report = _report_markdown(
        args.fold,
        raw_metrics,
        standardized_metrics,
        bootstrap,
        len(test["sample_keys"]),
        len(test_artifacts),
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")

    generated = (
        "train_full_core_features.npz",
        "train_hard_core_features.npz",
        "full_core_features.npz",
        "hard_core_features.npz",
        "train_only_core_scaler.json",
        "sample_level_errors.csv",
        "class_distribution_metrics.json",
        "bootstrap_metrics.json",
        "error_decomposition.json",
        "report.md",
    )
    manifest = {
        "schema_version": 1,
        "artifact_type": "svg_stage0_theory_diagnostics",
        "fold": int(args.fold),
        "primary_definition": {
            "feature": "raw_exact_sgw_core_18d",
            "ground_metric": "euclidean",
            "standardized_result_role": "sensitivity_only",
        },
        "train_sample_count": len(train["sample_keys"]),
        "outer_test_sample_count": len(test_artifacts),
        "eligible_outer_test_sample_count": len(test["sample_keys"]),
        "train_provenance": train_provenance,
        "test_provenance": test_provenance,
        "bootstrap_repeats": int(args.bootstrap_repeats),
        "bootstrap_seed": int(args.bootstrap_seed),
        "files": {
            name: file_sha256(output_dir / name) for name in generated
        },
    }
    _atomic_json(output_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "manifest": str(output_dir / "manifest.json"),
                "raw_primary": raw_metrics,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
