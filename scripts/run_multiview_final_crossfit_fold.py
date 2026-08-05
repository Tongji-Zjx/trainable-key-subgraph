"""Run one frozen S-residual/UOT/G plus Author-ST fusion outer fold."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.data_split import file_sha256  # noqa: E402


ARCHITECTURE = {
    "static_mode": "residual",
    "v_mode": "uot",
    "use_g": True,
    "short_term": "author_no_coordinate_short_term",
    "fusion": "critical_representation_residual",
}


def _trusted_load(path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def _run(name, command, artifact, print_only=False):
    row = {"stage": name, "command": [str(x) for x in command], "done": str(artifact)}
    if print_only:
        return row
    if Path(artifact).is_file():
        print("SKIP {}: {} exists".format(name, artifact), flush=True)
        return row
    print("START {}".format(name), flush=True)
    subprocess.run([str(x) for x in command], cwd=str(PROJECT_ROOT), check=True)
    if not Path(artifact).is_file():
        raise RuntimeError("{} did not create {}".format(name, artifact))
    print("FINISH {}".format(name), flush=True)
    return row


def _json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--source-crossfit-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--fusion-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--fusion-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gw-max-iter", type=int, default=100)
    parser.add_argument("--gw-sinkhorn-iter", type=int, default=100)
    parser.add_argument("--uot-iterations", type=int, default=100)
    parser.add_argument("--print-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.fold not in (0, 1, 2):
        raise ValueError("final ADHD cross-fit expects fold 0, 1, or 2")
    source = args.source_crossfit_root.resolve()
    source_fold = source / "fold_{}".format(args.fold)
    protocol = source_fold / "protocol" / "data_protocol.json"
    selector = source_fold / "selector" / "best_checkpoint.pt"
    author = (
        source_fold
        / "author_short_term_no_coord"
        / "training_seed{}".format(args.seed)
        / "best_checkpoint.pt"
    )
    for path in (protocol, selector, author):
        if not path.is_file():
            raise FileNotFoundError(str(path))
    protocol_sha = file_sha256(protocol)
    author_payload = _trusted_load(author)
    if author_payload.get("protocol_sha256") != protocol_sha:
        raise ValueError("fold-local Author-ST checkpoint/protocol mismatch")
    if int(author_payload.get("training_config", {}).get("seed", -1)) != args.seed:
        raise ValueError("fold-local Author-ST checkpoint seed mismatch")
    if "balanced_accuracy" not in author_payload.get("validation_thresholds", {}):
        raise ValueError("fold-local Author-ST checkpoint has no frozen threshold")

    python = sys.executable
    fold_root = args.output_root.resolve() / "fold_{}".format(args.fold)
    cache = fold_root / "cache"
    scaler = fold_root / "scaler.pt"
    critical = fold_root / "critical_seed{}".format(args.seed)
    author_eval = fold_root / "author_evaluation_seed{}".format(args.seed)
    fusion = fold_root / "fusion_seed{}".format(args.seed)
    plan = []
    for split in ("train", "validation", "test"):
        plan.append(_run(
            "cache_{}".format(split),
            [python, "-u", "scripts/precompute_multiview_critical.py",
             "--protocol", protocol, "--split", split,
             "--selector-checkpoint", selector,
             "--output-dir", cache / split, "--device", args.device,
             "--num-workers", args.num_workers, "--selection-seed", args.seed,
             "--gw-max-iter", args.gw_max_iter,
             "--gw-sinkhorn-iter", args.gw_sinkhorn_iter,
             "--object-uot-iterations", args.uot_iterations],
            cache / split / "manifest.json", args.print_only,
        ))
    plan.append(_run(
        "fit_scaler",
        [python, "-u", "scripts/fit_multiview_critical_scaler.py",
         "--train-manifest", cache / "train" / "manifest.json",
         "--output", scaler],
        scaler, args.print_only,
    ))
    for split in ("train", "validation", "test"):
        plan.append(_run(
            "audit_{}".format(split),
            [python, "-u", "scripts/audit_multiview_critical_cache.py",
             "--manifest", cache / split / "manifest.json",
             "--output", fold_root / "audit_{}.json".format(split)],
            fold_root / "audit_{}.json".format(split), args.print_only,
        ))

    plan.append(_run(
        "train_critical",
        [python, "-u", "scripts/train_multiview_critical.py",
         "--train-manifest", cache / "train" / "manifest.json",
         "--validation-manifest", cache / "validation" / "manifest.json",
         "--scaler", scaler, "--output-dir", critical,
         "--device", args.device, "--epochs", args.epochs,
         "--batch-size", args.batch_size, "--num-workers", args.num_workers,
         "--seed", args.seed, "--static-mode", "residual",
         "--early-stopping-patience", 15],
        critical / "best_evaluation.json", args.print_only,
    ))
    for split in ("validation", "test"):
        plan.append(_run(
            "evaluate_critical_{}".format(split),
            [python, "-u", "scripts/evaluate_multiview_critical.py",
             "--manifest", cache / split / "manifest.json", "--scaler", scaler,
             "--checkpoint", critical / "best_checkpoint.pt",
             "--output", critical / "{}_evaluation.json".format(split),
             "--device", args.device, "--batch-size", args.batch_size,
             "--num-workers", args.num_workers],
            critical / "{}_evaluation.json".format(split), args.print_only,
        ))
        plan.append(_run(
            "evaluate_author_{}".format(split),
            [python, "-u", "scripts/evaluate_author_short_term.py",
             "--protocol", protocol, "--checkpoint", author, "--split", split,
             "--threshold-strategy", "balanced_accuracy",
             "--output-dir", author_eval, "--device", args.device,
             "--batch-size", 8, "--num-workers", args.num_workers],
            author_eval / "{}_evaluation.json".format(split), args.print_only,
        ))

    plan.append(_run(
        "train_fusion",
        [python, "-u", "scripts/train_multiview_short_term_fusion.py",
         "--protocol", protocol,
         "--train-manifest", cache / "train" / "manifest.json",
         "--validation-manifest", cache / "validation" / "manifest.json",
         "--scaler", scaler, "--critical-checkpoint", critical / "best_checkpoint.pt",
         "--short-term-checkpoint", author, "--output-dir", fusion,
         "--device", args.device, "--epochs", args.fusion_epochs,
         "--batch-size", args.fusion_batch_size, "--num-workers", args.num_workers,
         "--seed", args.seed, "--early-stopping-patience", 10],
        fusion / "best_evaluation.json", args.print_only,
    ))
    for split in ("validation", "test"):
        plan.append(_run(
            "evaluate_fusion_{}".format(split),
            [python, "-u", "scripts/evaluate_multiview_short_term_fusion.py",
             "--protocol", protocol, "--manifest", cache / split / "manifest.json",
             "--scaler", scaler, "--critical-checkpoint", critical / "best_checkpoint.pt",
             "--short-term-checkpoint", author,
             "--fusion-checkpoint", fusion / "best_checkpoint.pt", "--split", split,
             "--output", fusion / "{}_evaluation.json".format(split),
             "--device", args.device, "--batch-size", args.fusion_batch_size,
             "--num-workers", args.num_workers],
            fusion / "{}_evaluation.json".format(split), args.print_only,
        ))

    if args.print_only:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    spec = {
        "artifact_type": "multiview_final_crossfit_fold_v1",
        "outer_fold": args.fold,
        "seed": args.seed,
        "architecture": ARCHITECTURE,
        "protocol": str(protocol),
        "protocol_sha256": protocol_sha,
        "selector_checkpoint": str(selector),
        "selector_checkpoint_sha256": file_sha256(selector),
        "author_checkpoint": str(author),
        "author_checkpoint_sha256": file_sha256(author),
        "test_used_for_selection": False,
        "threshold_source": "fold_inner_validation",
    }
    _json(fold_root / "fold_complete.json", spec)
    print("FOLD {} COMPLETE".format(args.fold), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
