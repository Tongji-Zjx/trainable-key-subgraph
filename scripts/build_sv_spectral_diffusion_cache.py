"""Build resumable exact HKS/eigenbasis sidecars for an SVG hard cache."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.data_split import file_sha256  # noqa: E402
from keysubgraph.data.sv_signed_gin_artifact import (  # noqa: E402
    load_sv_signed_gin_record,
)
from keysubgraph.data.sv_spectral_diffusion import (  # noqa: E402
    build_sv_spectral_diffusion_record,
    load_sv_spectral_diffusion_record,
    save_sv_spectral_diffusion_record,
    sv_spectral_diffusion_filename,
    write_sv_spectral_diffusion_manifest,
)
from keysubgraph.features.sv_spectral_diffusion import (  # noqa: E402
    SVSpectralDiffusionExtractor,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--laplacian-eta", type=float, default=1.0e-3)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("max-samples must be positive")
    manifest_path = args.manifest.resolve()
    with manifest_path.open("r", encoding="utf-8") as handle:
        source_manifest = json.load(handle)
    if source_manifest.get("artifact_type") != "sv_hard_sgw_signed_gin_manifest":
        raise ValueError("spectral-diffusion source manifest is invalid")
    rows = source_manifest.get("records", [])
    if args.max_samples is not None:
        rows = rows[: int(args.max_samples)]
    source_manifest_sha256 = file_sha256(manifest_path)
    output_dir = args.output_dir.resolve()
    record_dir = output_dir / "records"
    extractor = SVSpectralDiffusionExtractor(
        laplacian_eta=args.laplacian_eta
    )
    pairs = []
    for index, row in enumerate(rows):
        source_path = Path(row["feature_path"])
        if not source_path.is_absolute():
            source_path = manifest_path.parent / source_path
        source_path = source_path.resolve()
        source_sha256 = file_sha256(source_path)
        if source_sha256 != row["feature_sha256"]:
            raise ValueError("source hard-cache artifact hash mismatch")
        output_path = record_dir / sv_spectral_diffusion_filename(
            row["sample_key"]
        )
        if output_path.exists() and not args.overwrite:
            record = load_sv_spectral_diffusion_record(output_path)
            if (
                record.sample_key != row["sample_key"]
                or record.source_feature_sha256 != source_sha256
                or record.source_manifest_sha256 != source_manifest_sha256
            ):
                raise ValueError("existing spectral-diffusion cache is stale")
        else:
            source = load_sv_signed_gin_record(source_path)
            if args.device != "cpu":
                from dataclasses import replace

                windows = tuple(
                    replace(
                        window,
                        node_features=window.node_features.to(args.device),
                        adjacency=window.adjacency.to(args.device),
                    )
                    if window is not None
                    else None
                    for window in source.windows
                )
                source = replace(source, windows=windows)
            record = build_sv_spectral_diffusion_record(
                source,
                source_sha256,
                source_manifest_sha256,
                extractor,
            )
            # Keep immutable artifacts portable and cheap to load.
            from dataclasses import replace

            cpu_windows = tuple(
                replace(
                    window,
                    eigenvalues=window.eigenvalues.detach().cpu(),
                    eigenvectors=window.eigenvectors.detach().cpu(),
                    hks=window.hks.detach().cpu(),
                    spectral_quantiles=window.spectral_quantiles.detach().cpu(),
                )
                if window is not None
                else None
                for window in record.windows
            )
            record = replace(
                record,
                windows=cpu_windows,
                window_mask=record.window_mask.cpu(),
                transition_mask=record.transition_mask.cpu(),
            )
            save_sv_spectral_diffusion_record(
                record, output_path, overwrite=args.overwrite
            )
        pairs.append((record, output_path))
        if (index + 1) % 10 == 0 or index + 1 == len(rows):
            print(
                "processed {}/{} {}".format(
                    index + 1, len(rows), record.sample_key
                ),
                flush=True,
            )
    manifest = write_sv_spectral_diffusion_manifest(
        pairs,
        output_dir / "manifest.json",
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest),
                "sample_count": len(pairs),
                "source_manifest_sha256": source_manifest_sha256,
                "laplacian_eta": extractor.laplacian_eta,
                "hks_time_scales": list(extractor.hks_time_scales),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
