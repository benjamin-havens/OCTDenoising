from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as scipy_stats


@dataclass(frozen=True)
class GammaFit:
    method: str
    alpha: float
    beta: float
    n_samples: int

    @property
    def valid(self) -> bool:
        return bool(
            np.isfinite(self.alpha)
            and np.isfinite(self.beta)
            and self.alpha > 0
            and self.beta > 0
            and self.n_samples > 1
        )

    def to_dict(self) -> dict[str, float | int]:
        return {
            "method": self.method,
            "alpha": float(self.alpha),
            "beta": float(self.beta),
            "n_samples": int(self.n_samples),
            "valid": self.valid,
        }


def compute_speckle_ratio(
    original: np.ndarray,
    denoised: np.ndarray,
    *,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    raw_denoised = np.asarray(denoised, dtype=np.float64)
    denoised_safe = np.maximum(raw_denoised, float(eps))
    original = np.asarray(original, dtype=np.float64)
    speckle = original / denoised_safe
    valid_mask = (
        np.isfinite(speckle)
        & np.isfinite(original)
        & np.isfinite(raw_denoised)
        & (raw_denoised > float(eps))
        & (speckle > 0.0)
    )
    return speckle, valid_mask


def downsample_values(
    values: np.ndarray,
    *,
    max_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size == 0:
        return values
    if max_samples <= 0 or values.size <= max_samples:
        return values
    idx = rng.choice(values.size, size=max_samples, replace=False)
    return values[idx]


def fit_gamma_mom(samples: np.ndarray) -> GammaFit:
    samples = np.asarray(samples, dtype=np.float64)
    n = int(samples.size)
    if n < 2:
        return GammaFit(method="mom", alpha=np.nan, beta=np.nan, n_samples=n)

    mean = float(np.mean(samples))
    var = float(np.var(samples))
    if mean <= 0.0 or var <= 0.0 or not np.isfinite(mean) or not np.isfinite(var):
        return GammaFit(method="mom", alpha=np.nan, beta=np.nan, n_samples=n)

    alpha = (mean * mean) / var
    beta = mean / var
    return GammaFit(method="mom", alpha=float(alpha), beta=float(beta), n_samples=n)


def fit_gamma_mle(samples: np.ndarray) -> GammaFit:
    samples = np.asarray(samples, dtype=np.float64)
    n = int(samples.size)
    if n < 2:
        return GammaFit(method="mle", alpha=np.nan, beta=np.nan, n_samples=n)

    try:
        alpha, loc, scale = scipy_stats.gamma.fit(samples, floc=0.0)
    except Exception:
        return GammaFit(method="mle", alpha=np.nan, beta=np.nan, n_samples=n)

    if not np.isfinite(alpha) or not np.isfinite(scale) or alpha <= 0.0 or scale <= 0.0:
        return GammaFit(method="mle", alpha=np.nan, beta=np.nan, n_samples=n)
    if loc != 0.0:
        return GammaFit(method="mle", alpha=np.nan, beta=np.nan, n_samples=n)

    beta = 1.0 / scale
    return GammaFit(method="mle", alpha=float(alpha), beta=float(beta), n_samples=n)


def fit_gamma_methods(samples: np.ndarray) -> dict[str, GammaFit]:
    samples = np.asarray(samples, dtype=np.float64)
    return {
        "mom": fit_gamma_mom(samples),
        "mle": fit_gamma_mle(samples),
    }


def gamma_pdf_rate(x: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    scale = 1.0 / float(beta)
    return scipy_stats.gamma.pdf(x, a=float(alpha), loc=0.0, scale=scale)


def compute_reconstruction_metrics(
    original: np.ndarray,
    denoised: np.ndarray,
    *,
    eps: float,
) -> dict[str, float]:
    original = np.asarray(original, dtype=np.float64)
    denoised = np.asarray(denoised, dtype=np.float64)
    finite = np.isfinite(original) & np.isfinite(denoised)
    if not np.any(finite):
        return {"mae": np.nan, "rmse": np.nan, "relative_rmse": np.nan}

    diff = original[finite] - denoised[finite]
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    denom = float(np.sqrt(np.mean(original[finite] * original[finite])))
    relative_rmse = float(rmse / (denom + float(eps)))
    return {"mae": mae, "rmse": rmse, "relative_rmse": relative_rmse}


def compute_ks_metrics(samples: np.ndarray, fit: GammaFit) -> dict[str, float]:
    if not fit.valid:
        return {"ks_statistic": np.nan, "ks_pvalue": np.nan}
    dist = scipy_stats.gamma(a=fit.alpha, loc=0.0, scale=1.0 / fit.beta)
    try:
        stat, pvalue = scipy_stats.kstest(np.asarray(samples, dtype=np.float64), dist.cdf)
    except Exception:
        return {"ks_statistic": np.nan, "ks_pvalue": np.nan}
    return {"ks_statistic": float(stat), "ks_pvalue": float(pvalue)}


def compute_kl_divergence(
    samples: np.ndarray,
    fit: GammaFit,
    *,
    bins: int = 200,
    eps: float = 1e-12,
) -> float:
    if not fit.valid:
        return float("nan")

    values = np.asarray(samples, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size < 2:
        return float("nan")

    low = float(np.min(values))
    high = float(np.max(values))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return float("nan")

    hist, edges = np.histogram(values, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    p = hist * widths
    q = gamma_pdf_rate(centers, fit.alpha, fit.beta) * widths

    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    p = p / np.sum(p)
    q = q / np.sum(q)
    return float(np.sum(p * np.log(p / q)))


def compute_distribution_metrics(
    samples: np.ndarray,
    fit: GammaFit,
    *,
    bins: int = 200,
    eps: float = 1e-12,
) -> dict[str, float]:
    kl = compute_kl_divergence(samples, fit, bins=bins, eps=eps)
    ks = compute_ks_metrics(samples, fit)
    return {
        "kl_divergence": float(kl),
        **ks,
    }
