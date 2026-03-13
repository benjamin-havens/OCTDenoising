from __future__ import annotations

import argparse
from pathlib import Path

from .api import degrade_folder_to_snr


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Degrade OCT folders with multiplicative gamma speckle to a target SNR."
    )
    parser.add_argument("--input-folder", type=Path, required=True)
    parser.add_argument("--target-snr", type=float, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--beta", type=float, required=True)

    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/speckle"))
    parser.add_argument("--domain", choices=("amplitude", "pixel"), default="amplitude")
    parser.add_argument("--snr-units", choices=("db", "linear"), default="db")
    parser.add_argument("--match-mode", choices=("exact", "analytic"), default="exact")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--transform-mode", choices=("db", "gamma"), default="db")
    parser.add_argument("--max-pixel", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=0.4)
    parser.add_argument("--amp-low", type=float, default=None)
    parser.add_argument("--amp-high", type=float, default=None)
    parser.add_argument("--db-low", type=float, default=-40.0)
    parser.add_argument("--db-high", type=float, default=0.0)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--input-floor", type=float, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    result = degrade_folder_to_snr(
        input_folder=args.input_folder,
        target_snr=args.target_snr,
        alpha=args.alpha,
        beta=args.beta,
        output_dir=args.output_dir,
        output_root=args.output_root,
        domain=args.domain,
        snr_units=args.snr_units,
        match_mode=args.match_mode,
        transform_mode=args.transform_mode,
        max_pixel=args.max_pixel,
        gamma=args.gamma,
        amp_low=args.amp_low,
        amp_high=args.amp_high,
        db_low=args.db_low,
        db_high=args.db_high,
        eps=args.eps,
        input_floor=args.input_floor,
        seed=args.seed,
    )
    print(
        f"Degraded {result.input_folder} -> {result.output_dir} "
        f"(target={result.summary['target_snr']['db']:.6g} dB, "
        f"achieved={result.summary['achieved_snr']['db']:.6g} dB)"
    )


if __name__ == "__main__":
    main()
