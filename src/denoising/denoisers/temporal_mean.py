from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d

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
        size = 2 * self.radius + 1
        denoised = uniform_filter1d(
            frames,
            size=size,
            axis=0,
            mode="reflect",
        )
        return denoised
