from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TransformConfig:
    """Configuration for linear-amplitude <-> display-pixel transforms."""

    mode: str = "gamma"
    max_pixel: int = 255
    gamma: float = 0.4
    amp_low: float | None = None
    amp_high: float | None = None
    db_low: float = -40.0
    db_high: float = 0.0
    eps: float = 1e-12


def _validate_config(config: TransformConfig) -> None:
    if config.mode not in {"db", "gamma"}:
        raise ValueError(f"Unsupported mode {config.mode!r}; expected 'db' or 'gamma'.")
    if config.max_pixel <= 0:
        raise ValueError("max_pixel must be positive.")
    if config.eps <= 0:
        raise ValueError("eps must be positive.")

    if config.mode == "db":
        if config.db_high <= config.db_low:
            raise ValueError("db_high must be greater than db_low.")
    else:
        if config.gamma <= 0:
            raise ValueError("gamma must be positive for gamma mode.")
        if config.amp_low is None or config.amp_high is None:
            raise ValueError("amp_low and amp_high are required for gamma mode.")
        if config.amp_low < 0:
            raise ValueError("amp_low must be non-negative.")
        if config.amp_high <= config.amp_low:
            raise ValueError("amp_high must be greater than amp_low.")


def _clip_pixels(pixels: np.ndarray, config: TransformConfig) -> np.ndarray:
    return np.clip(pixels, 0.0, float(config.max_pixel))


def pixels_to_linear_amplitude(pixels, config: TransformConfig):
    """Convert display pixels to linear amplitudes."""
    _validate_config(config)
    p = _clip_pixels(np.asarray(pixels, dtype=np.float64), config)

    if config.mode == "db":
        db = config.db_low + (p / float(config.max_pixel)) * (config.db_high - config.db_low)
        return 10.0 ** (db / 20.0)

    t_low = config.amp_low**config.gamma
    t_high = config.amp_high**config.gamma
    t = (p / float(config.max_pixel)) * (t_high - t_low) + t_low
    return np.maximum(t, config.eps) ** (1.0 / config.gamma)


def linear_amplitude_to_pixels(amplitude, config: TransformConfig, out_dtype=None):
    """Convert linear amplitudes to display pixels."""
    _validate_config(config)
    a = np.asarray(amplitude, dtype=np.float64)

    if config.mode == "db":
        db = 20.0 * np.log10(np.maximum(a, config.eps))
        db = np.clip(db, config.db_low, config.db_high)
        pixels = ((db - config.db_low) / (config.db_high - config.db_low)) * float(
            config.max_pixel
        )
    else:
        t = np.maximum(a, 0.0) ** config.gamma
        t_low = config.amp_low**config.gamma
        t_high = config.amp_high**config.gamma
        pixels = ((t - t_low) / (t_high - t_low)) * float(config.max_pixel)
        pixels = _clip_pixels(pixels, config)

    if out_dtype is None:
        return pixels

    if np.issubdtype(np.dtype(out_dtype), np.integer):
        pixels = np.rint(pixels)
    return pixels.astype(out_dtype)
