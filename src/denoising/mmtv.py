# %% IMPORTS
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import tqdm
import odl

from .oct_processing import (
    TransformConfig,
    linear_amplitude_to_pixels,
    pixels_to_linear_amplitude,
)


# %% MMTV
def MMTV_denoise(
    y_linear,
    *,
    max_iterations=1000,
    max_inner_iterations=20,
    alpha=1.1,
    beta=0.9,
    tv_reg_strength=1e-2,
    convergence_threshold=1e-6,
    min_intensity=1e-6,
):
    """
    Denoise linear-amplitude OCT data with MMTV.

    :param y_linear: ndarray of linear amplitudes
    :param max_iterations: how many iterations of MMTV to run
    :param max_inner_iterations: how many iterations of PDHG to run per MM step
    :param alpha: gamma speckle shape parameter
    :param beta: gamma speckle scale parameter
    :param tv_reg_strength: lambda, TV regularization strength
    :param convergence_threshold: stop when ||x_k - x_{k-1}||^2 / ||x_k||^2 < threshold
    :param min_intensity: positivity floor used by the lower-bound box constraint
    """

    y_linear = np.asarray(y_linear, dtype=np.float64)
    if y_linear.ndim != 2:
        raise ValueError(f"y_linear must be 2D, got shape {y_linear.shape}.")
    if min_intensity <= 0:
        raise ValueError("min_intensity must be positive.")

    h, w = y_linear.shape
    lambda_ = tv_reg_strength

    # Set up the minimizations in ODL's format
    space = odl.uniform_discr([0, 0], [h, w], [h, w])
    y = space.element(np.maximum(y_linear, min_intensity))
    eye, grad = odl.IdentityOperator(space), odl.Gradient(space)
    L = odl.BroadcastOperator(eye, grad)  # Project image to itself plus gradient
    f = odl.solvers.IndicatorBox(space, 1e-6, 1)  # Amplitudes ought not be leq 0
    x_k = y.copy()

    # Pick tau and sigma for PDHG (Chambolle-Pock)
    op_norm = 1.1 * odl.power_method_opnorm(L, maxiter=4, xstart=y)
    tau = sigma = 1.0 / op_norm  # tau * sigma* ||L||^2 needs to be < 1 for convergence

    pbar = tqdm.trange(max_iterations)
    for k in pbar:
        x_prev = x_k.copy()

        # Set up the regularized least squares
        # t is the analytic solution to the majorant of the NLL
        t = ((beta / alpha) * x_k * y**2) ** (1 / 3)
        g_funcs = [
            odl.solvers.L2NormSquared(space).translated(t),
            lambda_ * odl.solvers.L1Norm(grad.range),
        ]
        g = odl.solvers.SeparableSum(*g_funcs)

        # Minimize 1/2 ||x_k - t||^2 + lambda ||grad x_k||_1 + f(x_k)
        # That is, be close to t but respect TV regularization and box constraints
        odl.solvers.pdhg(x_k, f, g, L, max_inner_iterations, tau=tau, sigma=sigma)

        # Relative change criterion
        #   ||x_k - x_{k-1}||^2 / ||x_k||^2 < eps?
        diff = x_k - x_prev
        denom = float(x_k.norm())
        denom2 = (denom * denom) + 1e-30
        rel_change = (float(diff.norm()) ** 2) / denom2

        pbar.set_postfix(rel_change=f"{rel_change:.3e}")
        if rel_change < convergence_threshold:
            break

    return x_k.asarray()


# %% MAIN
def _build_comparison_output_path(
    output_dir: Path, now: datetime | None = None
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    base_name = f"{timestamp}_mmtv_comparison"
    candidate = output_dir / f"{base_name}.png"
    suffix = 1
    while candidate.exists():
        candidate = output_dir / f"{base_name}_{suffix:02d}.png"
        suffix += 1
    return candidate


def main():
    import tifffile
    import matplotlib.pyplot as plt

    example_path = Path("inputs/example.tif")
    with tifffile.TiffFile(example_path) as img:
        y_pixels = img.asarray()
        if y_pixels.ndim > 2:
            y_pixels = y_pixels.mean(axis=2)

    if np.issubdtype(y_pixels.dtype, np.integer):
        max_pixel = int(np.iinfo(y_pixels.dtype).max)
    else:
        max_pixel = 255

    # Gamma-mode display transform with explicit calibrated amplitude cut levels.
    transform = TransformConfig(
        mode="gamma",
        max_pixel=max_pixel,
        gamma=0.4,
        amp_low=1e-3,
        amp_high=1.0,
    )

    y_linear = pixels_to_linear_amplitude(y_pixels, transform)
    x_hat_linear = MMTV_denoise(y_linear, alpha=2, beta=2)

    out_dtype = np.uint16 if max_pixel > 255 else np.uint8
    x_hat_pixels = linear_amplitude_to_pixels(
        x_hat_linear, transform, out_dtype=out_dtype
    )

    out_dir = Path("outputs")
    fig, axes = plt.subplots(2, 1, figsize=(6, 6), constrained_layout=True)

    axes[0].imshow(y_pixels, cmap="gray")
    axes[0].set_title("Input (pixels)")
    axes[0].axis("off")

    axes[1].imshow(x_hat_pixels, cmap="gray")
    axes[1].set_title("Denoised (pixels)")
    axes[1].axis("off")

    output_path = _build_comparison_output_path(out_dir)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()


# %%
