from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseDenoiser(ABC):
    """Base interface for denoisers operating on linear-amplitude frame sequences."""

    method_name: str = "base"

    @abstractmethod
    def denoise_sequence(self, frames_linear: np.ndarray) -> np.ndarray:
        """Return denoised linear-amplitude frames with the same sequence shape."""

    @staticmethod
    def validate_linear_sequence(frames_linear) -> np.ndarray:
        frames = np.asarray(frames_linear, dtype=np.float64)
        if frames.ndim != 3:
            raise ValueError(
                f"frames_linear must be a 3D array (n_frames, height, width), got {frames.shape}."
            )
        if frames.shape[0] == 0:
            raise ValueError("frames_linear must contain at least one frame.")
        return frames
