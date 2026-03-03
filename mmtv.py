# %% IMPORTS
import odl
import tqdm
import numpy as np


# %% MMTV functions
def pixels_to_intensity(x, low, high):
    db = (x / 255.0) * (high - low) - (high - low)
    return 10 ** (db / 20)


def intensity_to_pixels(x, low, high):
    db = np.clip(20 * np.log10(x), low, high)
    return (db + (high - low)) / (high - low) * 255.0


def MMTV_denoise(
    y,
    *,
    max_iterations=1000,
    max_inner_iterations=20,
    alpha=1.1,
    beta=0.9,
    tv_reg_strength=1e-2,
    convergence_threshold=1e-6,
    db_low=-40,
    db_high=0,
):
    """
    Denoise `y` assuming gamma distribution with parameters `alpha` and `beta` for the multiplicative speckle,
    and using TV regularization (L1-norm of image gradient) for its ability to preserve edges.

    :param y: ndarray of scaled log intensities (0-255)
    :param max_iterations: how many iterations of MMTV to run
    :param max_inner_iterations: how many iterations of PDHG to use for minimizing each regularized majorant
    :param alpha: parameter of gamma distribution for speckle noise
    :param beta: parameter of gamma distribution for speckle noise
    :param tv_reg_strength: lambda, the relative strength of TV regularization
    :param convergence_threshold: when the approximate solution changes relatively less than this (||x_k - x_{k-1}||^2 / ||x_k||^2), stop even if `max_iterations` is not yet reached.
    :param db_low: What dB a pixel value of 0 corresponds to.
    :param db_high: What dB a pixel value of 255 corresponds to.
    """
    h, w = y.shape
    lambda_ = tv_reg_strength

    # Set up the minimizations in ODL's format
    space = odl.uniform_discr([0, 0], [h, w], [h, w])
    y = space.element(pixels_to_intensity(y, db_low, db_high))
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

    return intensity_to_pixels(x_k.asarray(), db_low, db_high).astype(int)


# %% MAIN
def main():
    import tifffile
    from pathlib import Path
    import matplotlib.pyplot as plt

    example_path = Path("OCT_img176.tif")
    with tifffile.TiffFile(example_path) as img:
        y = img.asarray()
        if len(y.shape) > 2:
            y = y.mean(axis=2)  # (H, W, C) -> (H, W) (grayscale)
    x_hat = MMTV_denoise(y)

    # Save comparison plot
    out_dir = Path(".")
    fig, axes = plt.subplots(2, 1, figsize=(6, 6), constrained_layout=True)

    axes[0].imshow(y, cmap="gray")
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(x_hat, cmap="gray")
    axes[1].set_title("Denoised")
    axes[1].axis("off")

    plt.savefig(out_dir / "mmtv_comparison.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()


# %%
