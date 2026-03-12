from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import tifffile

_FRAME_NUMBER_PATTERN = re.compile(r"(\d+)$")


@dataclass(frozen=True)
class FolderStack:
    folder: Path
    frame_paths: list[Path]
    frame_numbers: list[int]
    frames: np.ndarray


def read_folder_stack(folder_path: Path | str) -> FolderStack:
    folder = Path(folder_path)
    frame_paths = _get_sorted_frame_paths(folder)
    frame_numbers = [_extract_trailing_frame_number(path) for path in frame_paths]
    _validate_contiguous_frame_numbers(frame_numbers, folder)
    frames = _load_folder_frames(frame_paths)
    return FolderStack(
        folder=folder,
        frame_paths=frame_paths,
        frame_numbers=frame_numbers,
        frames=frames,
    )


def validate_matching_stacks(original: FolderStack, denoised: FolderStack) -> None:
    if original.frame_numbers != denoised.frame_numbers:
        raise ValueError(
            "Original and denoised folders must contain matching frame numbers in the same "
            f"order. Got {original.folder} vs {denoised.folder}."
        )

    if original.frames.shape != denoised.frames.shape:
        raise ValueError(
            "Original and denoised stacks must have the same shape. "
            f"Got {original.frames.shape} vs {denoised.frames.shape}."
        )

    if original.frames.dtype != denoised.frames.dtype:
        raise ValueError(
            "Original and denoised stacks must have the same dtype. "
            f"Got {original.frames.dtype} vs {denoised.frames.dtype}."
        )


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
                f"All frames must share the same shape. Expected {expected_shape}, "
                f"got {frame.shape} for {frame_path.name}."
            )
        if frame.dtype != expected_dtype:
            raise ValueError(
                f"All frames must share the same dtype. Expected {expected_dtype}, "
                f"got {frame.dtype} for {frame_path.name}."
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
