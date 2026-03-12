from __future__ import annotations

import numpy as np

from .base import BaseDenoiser


class TemporalMeanDenoiser(BaseDenoiser):
    """Denoiser that averages each frame with neighboring frames in a fixed radius."""

    method_name = "temporal_mean"

    def __init__(self, radius: int):
        if not isinstance(radius, int) or radius <= 0:
            raise ValueError("radius must be a positive integer.")
        self.radius = radius

    def denoise_sequence(self, frames_linear: np.ndarray) -> np.ndarray:
        frames = self.validate_linear_sequence(frames_linear)
        n_frames = frames.shape[0]
        denoised = np.empty_like(frames, dtype=np.float64)

        for i in range(n_frames):
            start = max(0, i - self.radius)
            stop = min(n_frames, i + self.radius + 1)
            denoised[i] = frames[start:stop].mean(axis=0)

        return denoised
