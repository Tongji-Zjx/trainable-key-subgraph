"""Run resumable cache, audit, critical training and optional Author-ST fusion."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run(name, command, done_path=None):
    if done_path is not None and Path(done_path).is_file():
        print("SKIP {}: {} exists".format(name, done_path), flush=True)
        return
    print("START {}".format(name), flush=True)
    subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)
    print("FINISH {}".format(name), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--selector-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--short-term-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--fusion-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gw-max-iter", type=int, default=100)
    parser.add_argument("--gw-sinkhorn-iter", type=int, default=100)
    parser.add_argument("--uot-iterations", type=int, default=100)
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()
    root = args.output_root.resolve()
    cache = root / "cache"
    scaler = root / "scaler.pt"
    critical = root / "critical"
    fusion = root / "fusion"
    python = sys.executable
    commands = []
    for split in ("train", "validation", "test"):
        manifest = cache / split / "manifest.json"
        commands.append((
            "cache_{}".format(split),
            [
                python, "-u", "scripts/precompute_multiview_critical.py",
                "--protocol", str(args.protocol), "--split", split,
                "--selector-checkpoint", str(args.selector_checkpoint),
                "--output-dir", str(cache / split), "--device", args.device,
                "--num-workers", str(args.num_workers), "--selection-seed", str(args.seed),
                "--gw-max-iter", str(args.gw_max_iter),
                "--gw-sinkhorn-iter", str(args.gw_sinkhorn_iter),
                "--object-uot-iterations", str(args.uot_iterations),
            ],
            manifest,
        ))
    commands.append((
        "fit_train_scaler",
        [python, "-u", "scripts/fit_multiview_critical_scaler.py",
         "--train-manifest", str(cache / "train" / "manifest.json"),
         "--output", str(scaler)],
        scaler,
    ))
    for split in ("train", "validation", "test"):
        commands.append((
            "audit_{}".format(split),
            [python, "-u", "scripts/audit_multiview_critical_cache.py",
             "--manifest", str(cache / split / "manifest.json"),
             "--output", str(root / "audit_{}.json".format(split))],
            root / "audit_{}.json".format(split),
        ))
    commands.append((
        "train_critical",
        [python, "-u", "scripts/train_multiview_critical.py",
         "--train-manifest", str(cache / "train" / "manifest.json"),
         "--validation-manifest", str(cache / "validation" / "manifest.json"),
         "--scaler", str(scaler), "--output-dir", str(critical),
         "--device", args.device, "--epochs", str(args.epochs),
         "--batch-size", str(args.batch_size), "--num-workers", str(args.num_workers),
         "--seed", str(args.seed)],
        critical / "best_checkpoint.pt",
    ))
    for split in ("validation", "test"):
        commands.append((
            "evaluate_critical_{}".format(split),
            [python, "-u", "scripts/evaluate_multiview_critical.py",
             "--manifest", str(cache / split / "manifest.json"),
             "--scaler", str(scaler), "--checkpoint", str(critical / "best_checkpoint.pt"),
             "--output", str(critical / "{}_evaluation.json".format(split)),
             "--device", args.device, "--batch-size", str(args.batch_size),
             "--num-workers", str(args.num_workers)],
            critical / "{}_evaluation.json".format(split),
        ))
    commands.append((
        "diagnose_critical_validation",
        [python, "-u", "scripts/diagnose_multiview_critical.py",
         "--manifest", str(cache / "validation" / "manifest.json"),
         "--scaler", str(scaler), "--checkpoint", str(critical / "best_checkpoint.pt"),
         "--output", str(critical / "validation_diagnostic.json"),
         "--device", args.device, "--num-workers", str(args.num_workers)],
        critical / "validation_diagnostic.json",
    ))
    if args.short_term_checkpoint is not None:
        for split in ("validation", "test"):
            commands.append((
                "evaluate_author_short_term_{}".format(split),
                [python, "-u", "scripts/evaluate_author_short_term.py",
                 "--protocol", str(args.protocol),
                 "--checkpoint", str(args.short_term_checkpoint),
                 "--split", split, "--output-dir", str(root / "short_term"),
                 "--device", args.device, "--batch-size", str(args.batch_size),
                 "--num-workers", str(args.num_workers)],
                root / "short_term" / "{}_evaluation.json".format(split),
            ))
        commands.append((
            "train_fusion",
            [python, "-u", "scripts/train_multiview_short_term_fusion.py",
             "--protocol", str(args.protocol),
             "--train-manifest", str(cache / "train" / "manifest.json"),
             "--validation-manifest", str(cache / "validation" / "manifest.json"),
             "--scaler", str(scaler),
             "--critical-checkpoint", str(critical / "best_checkpoint.pt"),
             "--short-term-checkpoint", str(args.short_term_checkpoint),
             "--output-dir", str(fusion), "--device", args.device,
             "--epochs", str(args.fusion_epochs), "--batch-size", str(args.batch_size),
             "--num-workers", str(args.num_workers), "--seed", str(args.seed)],
            fusion / "best_checkpoint.pt",
        ))
        for split in ("validation", "test"):
            commands.append((
                "evaluate_fusion_{}".format(split),
                [python, "-u", "scripts/evaluate_multiview_short_term_fusion.py",
                 "--protocol", str(args.protocol),
                 "--manifest", str(cache / split / "manifest.json"),
                 "--scaler", str(scaler),
                 "--critical-checkpoint", str(critical / "best_checkpoint.pt"),
                 "--short-term-checkpoint", str(args.short_term_checkpoint),
                 "--fusion-checkpoint", str(fusion / "best_checkpoint.pt"),
                 "--split", split,
                 "--output", str(fusion / "{}_evaluation.json".format(split)),
                 "--device", args.device, "--batch-size", str(args.batch_size),
                 "--num-workers", str(args.num_workers)],
                fusion / "{}_evaluation.json".format(split),
            ))
    if args.print_only:
        print(json.dumps(
            [{"name": name, "command": command, "done": str(done)} for name, command, done in commands],
            ensure_ascii=False, indent=2,
        ))
        return 0
    root.mkdir(parents=True, exist_ok=True)
    for name, command, done in commands:
        _run(name, command, done)
    print("all multi-view stages completed: {}".format(root), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
