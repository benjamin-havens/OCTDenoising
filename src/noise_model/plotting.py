from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from .stats import GammaFit, gamma_pdf_rate


def save_triptych_figure(
    *,
    original: np.ndarray,
    denoised: np.ndarray,
    speckle: np.ndarray,
    domain_name: str,
    frame_index: int,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    _imshow_gray(axes[0], np.asarray(original, dtype=np.float64), "Original")
    _imshow_gray(axes[1], np.asarray(denoised, dtype=np.float64), "Denoised")
    _imshow_speckle(axes[2], np.asarray(speckle, dtype=np.float64), "Speckle")
    fig.suptitle(f"{domain_name} domain - frame {frame_index}")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_distribution_overlay_figure(
    *,
    samples: np.ndarray,
    fits: dict[str, GammaFit],
    domain_name: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples = np.asarray(samples, dtype=np.float64)
    samples = samples[np.isfinite(samples) & (samples > 0.0)]
    if samples.size == 0:
        return

    lo = float(np.min(samples))
    hi = float(np.percentile(samples, 99.5))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.hist(samples, bins=120, density=True, alpha=0.35, color="#4c78a8", label="speckle")

    x = np.linspace(lo, hi, 400)
    for method, color in (("mom", "#f58518"), ("mle", "#54a24b")):
        fit = fits.get(method)
        if fit is None or not fit.valid:
            continue
        y = gamma_pdf_rate(x, fit.alpha, fit.beta)
        ax.plot(x, y, color=color, linewidth=2.0, label=f"{method.upper()} fit")

    ax.set_title(f"Speckle histogram + Gamma fits ({domain_name})")
    ax.set_xlabel("Speckle value")
    ax.set_ylabel("Density")
    ax.legend(loc="best")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_fit_traces_figure(
    *,
    frame_rows: list[dict[str, float | int | str]],
    domain_name: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not frame_rows:
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    _plot_method_trace(axes[0], frame_rows, "alpha", "Alpha")
    _plot_method_trace(axes[1], frame_rows, "beta", "Beta (rate)")
    axes[1].set_xlabel("Frame index")
    fig.suptitle(f"Per-frame gamma coefficients ({domain_name})")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_error_trends_figure(
    *,
    frame_rows: list[dict[str, float | int | str]],
    domain_name: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not frame_rows:
        return

    ordered_rows = sorted(frame_rows, key=lambda row: (int(row["frame_index"]), str(row["fit_method"])))
    frame_indices = sorted({int(row["frame_index"]) for row in ordered_rows})

    recon_by_frame = {}
    for frame_idx in frame_indices:
        first = next(row for row in ordered_rows if int(row["frame_index"]) == frame_idx)
        recon_by_frame[frame_idx] = (
            float(first["mae"]),
            float(first["rmse"]),
            float(first["relative_rmse"]),
        )

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)

    mae = [recon_by_frame[idx][0] for idx in frame_indices]
    rmse = [recon_by_frame[idx][1] for idx in frame_indices]
    rel_rmse = [recon_by_frame[idx][2] for idx in frame_indices]
    axes[0, 0].plot(frame_indices, mae, label="MAE", color="#4c78a8")
    axes[0, 0].plot(frame_indices, rmse, label="RMSE", color="#f58518")
    axes[0, 0].set_title("Reconstruction MAE/RMSE")
    axes[0, 0].set_xlabel("Frame index")
    axes[0, 0].legend(loc="best")

    axes[0, 1].plot(frame_indices, rel_rmse, label="Relative RMSE", color="#54a24b")
    axes[0, 1].set_title("Relative RMSE")
    axes[0, 1].set_xlabel("Frame index")
    axes[0, 1].legend(loc="best")

    _plot_method_metric(axes[1, 0], ordered_rows, "kl_divergence", "KL divergence")
    _plot_method_metric(axes[1, 1], ordered_rows, "ks_statistic", "KS statistic")
    axes[1, 0].set_xlabel("Frame index")
    axes[1, 1].set_xlabel("Frame index")
    fig.suptitle(f"Per-frame error trends ({domain_name})")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _imshow_gray(ax: plt.Axes, image: np.ndarray, title: str) -> None:
    vmin, vmax = _robust_limits(image)
    ax.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")


def _imshow_speckle(ax: plt.Axes, image: np.ndarray, title: str) -> None:
    values = image[np.isfinite(image)]
    if values.size > 0:
        hi = float(np.percentile(values, 99.0))
        display = np.clip(image, 0.0, hi if hi > 0 else np.max(values))
    else:
        display = image
    im = ax.imshow(display, cmap="viridis")
    ax.set_title(title)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)


def _robust_limits(image: np.ndarray) -> tuple[float | None, float | None]:
    values = image[np.isfinite(image)]
    if values.size == 0:
        return None, None
    low = float(np.percentile(values, 1.0))
    high = float(np.percentile(values, 99.0))
    if high <= low:
        return None, None
    return low, high


def _plot_method_trace(
    ax: plt.Axes,
    frame_rows: Iterable[dict[str, float | int | str]],
    field: str,
    ylabel: str,
) -> None:
    for method, color in (("mom", "#f58518"), ("mle", "#54a24b")):
        method_rows = sorted(
            (
                row
                for row in frame_rows
                if str(row["fit_method"]) == method and np.isfinite(float(row[field]))
            ),
            key=lambda row: int(row["frame_index"]),
        )
        if not method_rows:
            continue
        x = [int(row["frame_index"]) for row in method_rows]
        y = [float(row[field]) for row in method_rows]
        ax.plot(x, y, linewidth=1.8, color=color, label=method.upper())
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")


def _plot_method_metric(
    ax: plt.Axes,
    frame_rows: list[dict[str, float | int | str]],
    field: str,
    title: str,
) -> None:
    for method, color in (("mom", "#f58518"), ("mle", "#54a24b")):
        method_rows = sorted(
            (
                row
                for row in frame_rows
                if str(row["fit_method"]) == method and np.isfinite(float(row[field]))
            ),
            key=lambda row: int(row["frame_index"]),
        )
        if not method_rows:
            continue
        x = [int(row["frame_index"]) for row in method_rows]
        y = [float(row[field]) for row in method_rows]
        ax.plot(x, y, linewidth=1.8, color=color, label=method.upper())
    ax.set_title(title)
    ax.legend(loc="best")
