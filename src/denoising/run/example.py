from __future__ import annotations

import argparse
import ast
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile

from ..denoisers import BaseDenoiser, MMTVDenoiser, TemporalMeanDenoiser
from ..oct_processing import (
    TransformConfig,
    linear_amplitude_to_pixels,
    pixels_to_linear_amplitude,
)

_DENOISER_REGISTRY: dict[str, type[BaseDenoiser]] = {
    MMTVDenoiser.method_name: MMTVDenoiser,
    TemporalMeanDenoiser.method_name: TemporalMeanDenoiser,
}

_DEFAULT_DENOISER_KWARGS: dict[str, dict[str, object]] = {
    MMTVDenoiser.method_name: {"alpha": 2, "beta": 2},
    TemporalMeanDenoiser.method_name: {"radius": 1},
}


def _build_comparison_output_path(
    output_dir: Path,
    method_name: str,
    now: datetime | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    base_name = f"{timestamp}_{method_name}_comparison"
    candidate = output_dir / f"{base_name}.png"
    suffix = 1
    while candidate.exists():
        candidate = output_dir / f"{base_name}_{suffix:02d}.png"
        suffix += 1
    return candidate


def _parse_denoiser_arg(arg: str) -> tuple[str, object]:
    if "=" not in arg:
        raise ValueError(
            f"Invalid --denoiser-arg value: '{arg}'. Expected format key=value."
        )

    key, raw_value = arg.split("=", 1)
    key = key.strip()
    raw_value = raw_value.strip()
    if not key:
        raise ValueError("Denoiser argument key cannot be empty.")

    try:
        value = ast.literal_eval(raw_value)
    except (SyntaxError, ValueError):
        value = raw_value

    return key, value


def _build_denoiser(method: str, denoiser_args: list[str]) -> BaseDenoiser:
    denoiser_cls = _DENOISER_REGISTRY[method]
    kwargs = dict(_DEFAULT_DENOISER_KWARGS.get(method, {}))
    for arg in denoiser_args:
        key, value = _parse_denoiser_arg(arg)
        kwargs[key] = value
    return denoiser_cls(**kwargs)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a denoiser on one input OCT frame and save a side-by-side plot."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("inputs/example.tif"),
        help="Path to input TIFF image (default: inputs/example.tif).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for comparison PNG output (default: outputs).",
    )
    parser.add_argument(
        "--method",
        choices=sorted(_DENOISER_REGISTRY),
        default=MMTVDenoiser.method_name,
        help="Denoiser method to run.",
    )
    parser.add_argument(
        "--denoiser-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Constructor arg override for the selected denoiser. "
            "Repeat as needed, e.g. --denoiser-arg radius=2."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = _parse_args(argv)

    with tifffile.TiffFile(args.input) as img:
        y_pixels = img.asarray()
        if y_pixels.ndim > 2:
            y_pixels = y_pixels.mean(axis=2)

    if np.issubdtype(y_pixels.dtype, np.integer):
        max_pixel = int(np.iinfo(y_pixels.dtype).max)
    else:
        max_pixel = 255

    transform = TransformConfig(
        mode="gamma",
        max_pixel=max_pixel,
        gamma=0.4,
        amp_low=1e-3,
        amp_high=1.0,
    )

    y_linear = pixels_to_linear_amplitude(y_pixels, transform)
    denoiser = _build_denoiser(args.method, args.denoiser_arg)
    x_hat_linear = denoiser.denoise_sequence(y_linear[np.newaxis, ...])[0]

    out_dtype = np.uint16 if max_pixel > 255 else np.uint8
    x_hat_pixels = linear_amplitude_to_pixels(
        x_hat_linear, transform, out_dtype=out_dtype
    )

    fig, axes = plt.subplots(2, 1, figsize=(6, 6), constrained_layout=True)

    axes[0].imshow(y_pixels, cmap="gray")
    axes[0].set_title("Input (pixels)")
    axes[0].axis("off")

    axes[1].imshow(x_hat_pixels, cmap="gray")
    axes[1].set_title(f"Denoised ({denoiser.method_name})")
    axes[1].axis("off")

    output_path = _build_comparison_output_path(args.output_dir, denoiser.method_name)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
