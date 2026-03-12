from __future__ import annotations

import argparse
from pathlib import Path

from .api import estimate_pairs


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate gamma noise coefficients from original/denoised OCT folder pairs."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--pairs-csv",
        type=Path,
        help="CSV with required columns: original_folder, denoised_folder.",
    )
    group.add_argument(
        "--original-folder",
        type=Path,
        help="Original folder for single-pair mode.",
    )
    parser.add_argument(
        "--denoised-folder",
        type=Path,
        help="Denoised folder for single-pair mode (required when --original-folder is used).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/noise_model"),
        help="Root output directory for reports (default: outputs/noise_model).",
    )

    parser.add_argument("--transform-mode", choices=("db", "gamma"), default="db")
    parser.add_argument("--max-pixel", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=0.4)
    parser.add_argument("--amp-low", type=float, default=None)
    parser.add_argument("--amp-high", type=float, default=None)
    parser.add_argument("--db-low", type=float, default=-40.0)
    parser.add_argument("--db-high", type=float, default=0.0)
    parser.add_argument("--eps", type=float, default=1e-12)

    parser.add_argument(
        "--sample-frames",
        type=int,
        default=6,
        help="Number of equally spaced frames for triptych plots (default: 6).",
    )
    parser.add_argument(
        "--max-speckle-samples",
        type=int,
        default=2_000_000,
        help="Maximum number of speckle samples per fit (default: 2000000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for sampling when downsampling voxels.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if args.original_folder is not None and args.denoised_folder is None:
        raise SystemExit("--denoised-folder is required when --original-folder is provided.")
    if args.original_folder is None and args.denoised_folder is not None:
        raise SystemExit("--original-folder is required when --denoised-folder is provided.")

    if args.pairs_csv is not None:
        results = estimate_pairs(
            pairs_csv=args.pairs_csv,
            output_root=args.output_root,
            transform_mode=args.transform_mode,
            max_pixel=args.max_pixel,
            gamma=args.gamma,
            amp_low=args.amp_low,
            amp_high=args.amp_high,
            db_low=args.db_low,
            db_high=args.db_high,
            eps=args.eps,
            sample_frames=args.sample_frames,
            max_speckle_samples=args.max_speckle_samples,
            seed=args.seed,
        )
    else:
        results = estimate_pairs(
            pairs=[(args.original_folder, args.denoised_folder)],
            output_root=args.output_root,
            transform_mode=args.transform_mode,
            max_pixel=args.max_pixel,
            gamma=args.gamma,
            amp_low=args.amp_low,
            amp_high=args.amp_high,
            db_low=args.db_low,
            db_high=args.db_high,
            eps=args.eps,
            sample_frames=args.sample_frames,
            max_speckle_samples=args.max_speckle_samples,
            seed=args.seed,
        )

    for result in results:
        print(
            f"Estimated noise model for {result.original_folder} vs {result.denoised_folder} -> "
            f"{result.output_dir}"
        )


if __name__ == "__main__":
    main()
