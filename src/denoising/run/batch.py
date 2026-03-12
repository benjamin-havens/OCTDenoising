from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import numpy as np
import tifffile

from ..denoisers import BaseDenoiser
from ..oct_processing import (
    TransformConfig,
    linear_amplitude_to_pixels,
    pixels_to_linear_amplitude,
)

_FRAME_NUMBER_PATTERN = re.compile(r"(\d+)$")


def denoise_folders(
    folder_paths: Sequence[Path | str],
    denoiser: BaseDenoiser,
    *,
    output_root: Path | str = Path("outputs"),
    transform_config: TransformConfig | None = None,
) -> list[Path]:
    """
    Denoise each input folder and write 1:1 TIFF outputs preserving source filenames.
    """

    output_root = Path(output_root)
    output_paths: list[Path] = []
    for folder_path in folder_paths:
        folder = Path(folder_path)
        output_paths.append(
            _denoise_folder(
                folder,
                denoiser=denoiser,
                output_root=output_root,
                transform_config=transform_config,
            )
        )
    return output_paths


def _denoise_folder(
    folder: Path,
    *,
    denoiser: BaseDenoiser,
    output_root: Path,
    transform_config: TransformConfig | None,
) -> Path:
    frame_paths = _get_sorted_frame_paths(folder)
    frame_numbers = [_extract_trailing_frame_number(path) for path in frame_paths]
    _validate_contiguous_frame_numbers(frame_numbers, folder)

    frames_pixels = _load_folder_frames(frame_paths)
    source_dtype = frames_pixels.dtype
    config = transform_config or _default_transform_config(source_dtype)

    frames_linear = pixels_to_linear_amplitude(frames_pixels, config)
    denoised_linear = np.asarray(denoiser.denoise_sequence(frames_linear), dtype=np.float64)
    _validate_denoised_shape(denoised_linear, frames_linear.shape, folder, denoiser)

    out_dtype = source_dtype if np.issubdtype(source_dtype, np.integer) else None
    denoised_pixels = linear_amplitude_to_pixels(
        denoised_linear,
        config,
        out_dtype=out_dtype,
    )

    method_dir = output_root / denoiser.method_name
    output_dir = method_dir / f"denoised_{denoiser.method_name}_{folder.name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    for frame_path, frame_data in zip(frame_paths, denoised_pixels):
        tifffile.imwrite(output_dir / frame_path.name, frame_data)

    return output_dir


def _get_sorted_frame_paths(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise ValueError(f"Input folder does not exist or is not a directory: {folder}")

    frame_paths = sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    )
    if not frame_paths:
        raise ValueError(f"No TIFF files found in folder: {folder}")

    frame_numbers = [_extract_trailing_frame_number(path) for path in frame_paths]
    if len(set(frame_numbers)) != len(frame_numbers):
        raise ValueError(
            f"Duplicate frame numbers detected in folder {folder}. "
            "Frame number must be unique per TIFF file."
        )

    indexed_paths = sorted(zip(frame_numbers, frame_paths), key=lambda item: item[0])
    return [path for _, path in indexed_paths]


def _extract_trailing_frame_number(path: Path) -> int:
    match = _FRAME_NUMBER_PATTERN.search(path.stem)
    if match is None:
        raise ValueError(
            f"Filename must end with a frame number before extension: {path.name}"
        )
    return int(match.group(1))


def _validate_contiguous_frame_numbers(frame_numbers: list[int], folder: Path) -> None:
    for prev, curr in zip(frame_numbers, frame_numbers[1:]):
        if curr != prev + 1:
            raise ValueError(
                f"Frame numbers must be contiguous in folder {folder}. "
                f"Found gap between {prev} and {curr}."
            )


def _load_folder_frames(frame_paths: Sequence[Path]) -> np.ndarray:
    first_frame = _read_2d_tiff(frame_paths[0])
    expected_shape = first_frame.shape
    expected_dtype = first_frame.dtype

    frames = [first_frame]
    for frame_path in frame_paths[1:]:
        frame = _read_2d_tiff(frame_path)
        if frame.shape != expected_shape:
            raise ValueError(
                f"All frames must share the same shape. "
                f"Expected {expected_shape}, got {frame.shape} for {frame_path.name}."
            )
        if frame.dtype != expected_dtype:
            raise ValueError(
                f"All frames must share the same dtype. "
                f"Expected {expected_dtype}, got {frame.dtype} for {frame_path.name}."
            )
        frames.append(frame)

    return np.stack(frames, axis=0)


def _read_2d_tiff(path: Path) -> np.ndarray:
    with tifffile.TiffFile(path) as img:
        frame = img.asarray()
    if frame.ndim != 2:
        raise ValueError(
            f"Each TIFF must contain a single 2D frame. Got shape {frame.shape} for {path.name}."
        )
    return frame


def _validate_denoised_shape(
    denoised_linear: np.ndarray,
    expected_shape: tuple[int, int, int],
    folder: Path,
    denoiser: BaseDenoiser,
) -> None:
    if denoised_linear.ndim != 3:
        raise ValueError(
            f"{denoiser.method_name} must return a 3D array (n_frames, height, width). "
            f"Got shape {denoised_linear.shape} for folder {folder}."
        )
    if denoised_linear.shape != expected_shape:
        raise ValueError(
            f"{denoiser.method_name} must return shape {expected_shape}, "
            f"got {denoised_linear.shape} for folder {folder}."
        )


def _default_transform_config(dtype: np.dtype) -> TransformConfig:
    if np.issubdtype(dtype, np.integer) and np.iinfo(dtype).max > 255:
        max_pixel = 65535
    else:
        max_pixel = 255
    return TransformConfig(mode="db", max_pixel=max_pixel)
