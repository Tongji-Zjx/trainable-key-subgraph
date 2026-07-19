"""Command planning for one fail-closed cross-fitting fold dry run."""

from __future__ import absolute_import, print_function

from pathlib import Path


VARIANTS = (
    ("A", "key", "signed"), ("B", "random", "signed"),
    ("C", "key", "node_only"), ("D", "random", "node_only"),
)


def build_fold_commands(project_root, fold, seed=42, device="cuda", smoke=False):
    root = Path(project_root).resolve()
    python = "python"
    fold_root = Path("outputs/crossfit/fold_{}".format(fold))
    protocol = fold_root / "protocol/data_protocol.json"
    extractor_dir = fold_root / "extractor"
    extractor_checkpoint = extractor_dir / "best_checkpoint.pt"
    key_exports = fold_root / "key_exports"
    controls = fold_root / "controls"
    control_manifest = controls / "key_random_control_manifest.json"
    manifest_root = Path("outputs/crossfit/manifests")
    commands = []
    extractor = [
        python, "-u", "scripts/train_soft_extractor.py", "--protocol", str(protocol),
        "--output-dir", str(extractor_dir), "--device", device, "--seed", "42",
        "--selection-metric", "loss",
    ]
    if smoke:
        extractor.append("--smoke")
    commands.append(("extractor", extractor, extractor_checkpoint))
    for split in ("train", "validation", "test"):
        command = [
            python, "-u", "scripts/export_hard_subgraphs.py", "--protocol", str(protocol),
            "--checkpoint", str(extractor_checkpoint), "--split", split,
            "--device", device, "--output-dir", str(key_exports),
        ]
        commands.append((
            "key_export_{}".format(split), command,
            key_exports / "_completion/{}.json".format(split),
        ))
    commands.append(("key_random_controls", [
        python, "-u", "scripts/build_crossfit_key_random_exports.py",
        "--protocol", str(protocol), "--key-export-root", str(key_exports),
        "--output-root", str(controls), "--random-seed", "42",
    ], control_manifest))
    commands.append(("baseline_manifests", [
        python, "-u", "scripts/prepare_oof_manifests.py", "--fold", str(fold),
        "--protocol", str(protocol), "--control-root", str(controls),
        "--checkpoint", str(extractor_checkpoint), "--output-root", str(manifest_root),
    ], manifest_root / "fold_{}".format(fold) / "random_outer_test/baseline_manifest.json"))
    for variant, source, encoder in VARIANTS:
        run_dir = Path("outputs/crossfit/downstream/fold{}_{}_seed{}".format(fold, variant, seed))
        train_manifest = manifest_root / "fold_{}".format(fold) / "{}_inner_train/baseline_manifest.json".format(source)
        validation_manifest = manifest_root / "fold_{}".format(fold) / "{}_inner_validation/baseline_manifest.json".format(source)
        test_manifest = manifest_root / "fold_{}".format(fold) / "{}_outer_test/baseline_manifest.json".format(source)
        train = [
            python, "-u", "scripts/train_baseline.py",
            "--train-manifest", str(train_manifest),
            "--validation-manifest", str(validation_manifest),
            "--output-dir", str(run_dir), "--device", device,
            "--seed", str(seed), "--encoder-type", encoder,
            "--history-mode", "independent_bag",
            "--selection-metric", "unweighted_log_loss",
        ]
        if smoke:
            train.append("--smoke")
        commands.append(("train_{}".format(variant), train, run_dir / "best_checkpoint.pt"))
        commands.append(("evaluate_{}".format(variant), [
            python, "-u", "scripts/evaluate_baseline.py",
            "--manifest", str(test_manifest), "--checkpoint", str(run_dir / "best_checkpoint.pt"),
            "--output", str(run_dir / "outer_test_predictions.json"), "--device", device,
        ], run_dir / "outer_test_predictions.json"))
    return commands
