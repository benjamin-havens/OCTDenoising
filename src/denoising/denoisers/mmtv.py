from __future__ import annotations

import numpy as np
import odl
import tqdm

from .base import BaseDenoiser


class MMTVDenoiser(BaseDenoiser):
    """MMTV denoiser operating on linear-amplitude OCT frame sequences."""

    method_name = "mmtv"

    def __init__(
        self,
        *,
        max_iterations: int = 1000,
        max_inner_iterations: int = 20,
        alpha: float = 1.1,
        beta: float = 0.9,
        tv_reg_strength: float = 1e-2,
        convergence_threshold: float = 1e-6,
        min_intensity: float = 1e-6,
        use_3d_tv: bool = False,
    ):
        if min_intensity <= 0:
            raise ValueError("min_intensity must be positive.")

        self.max_iterations = max_iterations
        self.max_inner_iterations = max_inner_iterations
        self.alpha = alpha
        self.beta = beta
        self.tv_reg_strength = tv_reg_strength
        self.convergence_threshold = convergence_threshold
        self.min_intensity = min_intensity
        self.use_3d_tv = use_3d_tv

        if self.use_3d_tv:
            self.method_name = "mmtv_3d_tv"

    def denoise_sequence(self, frames_linear: np.ndarray) -> np.ndarray:
        frames = self.validate_linear_sequence(frames_linear)
        if self.use_3d_tv:
            return self._denoise_nd(frames)

        denoised = np.empty_like(frames, dtype=np.float64)
        for idx, frame in enumerate(frames):
            denoised[idx] = self._denoise_nd(frame)
        return denoised

    def _denoise_nd(self, y_linear: np.ndarray) -> np.ndarray:
        y_linear = np.asarray(y_linear, dtype=np.float64)
        if y_linear.ndim not in (2, 3):
            raise ValueError(f"y_linear must be 2D or 3D, got shape {y_linear.shape}.")

        lambda_ = self.tv_reg_strength
        shape = list(y_linear.shape)
        ndim = y_linear.ndim

        space = odl.uniform_discr([0] * ndim, shape, shape)
        y = space.element(np.maximum(y_linear, self.min_intensity))
        eye, grad = odl.IdentityOperator(space), odl.Gradient(space)
        L = odl.BroadcastOperator(eye, grad)
        f = odl.solvers.IndicatorBox(space, self.min_intensity, np.inf)
        x_k = y.copy()

        op_norm = 1.1 * odl.power_method_opnorm(L, maxiter=4, xstart=y)
        tau = sigma = 1.0 / op_norm

        for _ in tqdm.trange(self.max_iterations):
            x_prev = x_k.copy()

            t = ((self.beta / self.alpha) * x_k * y**2) ** (1.0 / 3.0)
            g = odl.solvers.SeparableSum(
                odl.solvers.L2NormSquared(space).translated(t),
                lambda_ * odl.solvers.L1Norm(grad.range),
            )

            odl.solvers.pdhg(
                x_k,
                f,
                g,
                L,
                self.max_inner_iterations,
                tau=tau,
                sigma=sigma,
            )

            diff = x_k - x_prev
            denom = float(x_k.norm())
            rel_change = (float(diff.norm()) ** 2) / ((denom * denom) + 1e-30)
            if rel_change < self.convergence_threshold:
                break

        return x_k.asarray()
