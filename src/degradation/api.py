from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import tifffile

from denoising.oct_processing import (
    TransformConfig,
    linear_amplitude_to_pixels,
    pixels_to_linear_amplitude,
)
from noise_model.io_utils import read_folder_stack


@dataclass(frozen=True)
class ArrayDegradationResult:
    degraded: np.ndarray
    lambda_value: float
    target_snr_linear: float
    target_snr_db: float
    achieved_snr_linear: float
    achieved_snr_db: float
    input_domain: str
    snr_units: str
    match_mode: str
    scaling_rule: str
    alpha: float
    beta: float
    eps: float
    input_floor: float


@dataclass(frozen=True)
class FolderDegradationResult:
    input_folder: Path
    output_dir: Path
    metrics_json: Path
    summary: dict[str, object]


def compute_snr(clean_intensity: np.ndarray, noisy_intensity: np.ndarray, units: str = "db") -> float:
    clean = np.asarray(clean_intensity, dtype=np.float64)
    noisy = np.asarray(noisy_intensity, dtype=np.float64)
    if clean.shape != noisy.shape:
        raise ValueError(
            f"clean_intensity and noisy_intensity must have the same shape; "
            f"got {clean.shape} vs {noisy.shape}."
        )
    if clean.size == 0:
        raise ValueError("clean_intensity and noisy_intensity must be non-empty.")

    units_norm = units.strip().lower()
    if units_norm not in {"linear", "db"}:
        raise ValueError(f"Unsupported units {units!r}; expected 'linear' or 'db'.")

    finite = np.isfinite(clean) & np.isfinite(noisy)
    if not np.any(finite):
        raise ValueError("No finite values available to compute SNR.")

    clean_f = clean[finite]
    noisy_f = noisy[finite]
    noise = noisy_f - clean_f

    signal_mean = float(np.mean(clean_f))
    noise_std = float(np.std(noise))
    if signal_mean < 0.0:
        raise ValueError("clean_intensity mean must be non-negative.")
    if noise_std == 0.0:
        snr_linear = float("inf")
    else:
        snr_linear = signal_mean / noise_std

    if units_norm == "linear":
        return float(snr_linear)
    if not np.isfinite(snr_linear):
        return float("inf")
    return float(10.0 * np.log10(max(snr_linear, np.finfo(np.float64).tiny)))


def degrade_array_to_snr(
    clean_array: np.ndarray,
    target_snr: float,
    alpha: float,
    beta: float,
    *,
    input_domain: str = "amplitude",
    snr_units: str = "db",
    match_mode: str = "exact",
    scaling_rule: str = "affine",
    eps: float = 1e-12,
    input_floor: float | None = None,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> ArrayDegradationResult:
    _validate_gamma(alpha=alpha, beta=beta)
    target_snr_linear = _normalize_target_snr(target_snr=target_snr, snr_units=snr_units)
    input_domain_norm = _normalize_choice(
        input_domain, valid={"amplitude", "intensity"}, name="input_domain"
    )
    match_mode_norm = _normalize_choice(match_mode, valid={"exact", "analytic"}, name="match_mode")
    scaling_rule_norm = _normalize_choice(scaling_rule, valid={"affine"}, name="scaling_rule")
    if eps <= 0.0 or not np.isfinite(eps):
        raise ValueError("eps must be finite and positive.")
    effective_input_floor = _normalize_input_floor(input_floor=input_floor, eps=eps)

    resolved_rng = _resolve_rng(seed=seed, rng=rng)
    clean = np.asarray(clean_array, dtype=np.float64)
    if clean.size == 0:
        raise ValueError("clean_array must be non-empty.")

    clean_intensity = _to_intensity(clean, input_domain=input_domain_norm)
    clean_intensity = np.maximum(clean_intensity, effective_input_floor)
    finite = np.isfinite(clean_intensity)
    if not np.any(finite):
        raise ValueError("clean_array does not contain finite values.")

    signal_mean = float(np.mean(clean_intensity[finite]))
    if signal_mean <= 0.0:
        raise ValueError("clean intensity mean must be strictly positive.")

    raw_noise = resolved_rng.gamma(shape=alpha, scale=1.0 / beta, size=clean_intensity.shape)
    if match_mode_norm == "analytic":
        lambda_value = _analytic_lambda(
            clean_intensity=clean_intensity,
            raw_noise=raw_noise,
            target_snr_linear=target_snr_linear,
        )
    else:
        lambda_value = _exact_lambda(
            clean_intensity=clean_intensity,
            raw_noise=raw_noise,
            target_snr_linear=target_snr_linear,
            eps=eps,
        )

    noisy_intensity = _apply_affine_speckle(
        clean_intensity=clean_intensity,
        raw_noise=raw_noise,
        lambda_value=lambda_value,
        eps=eps,
    )
    degraded = _from_intensity(noisy_intensity, output_domain=input_domain_norm)

    achieved_linear = compute_snr(clean_intensity, noisy_intensity, units="linear")
    achieved_db = compute_snr(clean_intensity, noisy_intensity, units="db")
    target_db = _linear_to_db(target_snr_linear)

    return ArrayDegradationResult(
        degraded=degraded,
        lambda_value=float(lambda_value),
        target_snr_linear=float(target_snr_linear),
        target_snr_db=float(target_db),
        achieved_snr_linear=float(achieved_linear),
        achieved_snr_db=float(achieved_db),
        input_domain=input_domain_norm,
        snr_units=snr_units.strip().lower(),
        match_mode=match_mode_norm,
        scaling_rule=scaling_rule_norm,
        alpha=float(alpha),
        beta=float(beta),
        eps=float(eps),
        input_floor=float(effective_input_floor),
    )


def degrade_folder_to_snr(
    input_folder: Path | str,
    target_snr: float,
    alpha: float,
    beta: float,
    *,
    output_dir: Path | str | None = None,
    output_root: Path | str = Path("outputs/speckle"),
    domain: str = "amplitude",
    snr_units: str = "db",
    match_mode: str = "exact",
    transform_config: TransformConfig | None = None,
    transform_mode: str = "db",
    max_pixel: int | None = None,
    gamma: float = 0.4,
    amp_low: float | None = None,
    amp_high: float | None = None,
    db_low: float = -40.0,
    db_high: float = 0.0,
    eps: float = 1e-12,
    input_floor: float | None = None,
    seed: int = 0,
) -> FolderDegradationResult:
    stack = read_folder_stack(input_folder)
    domain_norm = _normalize_choice(domain, valid={"amplitude", "pixel"}, name="domain")

    source_dtype = stack.frames.dtype
    clean_pixels = np.asarray(stack.frames, dtype=np.float64)
    if domain_norm == "amplitude":
        config = _build_transform_config(
            source_dtype=source_dtype,
            transform_config=transform_config,
            transform_mode=transform_mode,
            max_pixel=max_pixel,
            gamma=gamma,
            amp_low=amp_low,
            amp_high=amp_high,
            db_low=db_low,
            db_high=db_high,
            eps=eps,
        )
        clean_for_degrade = pixels_to_linear_amplitude(clean_pixels, config)
        result = degrade_array_to_snr(
            clean_for_degrade,
            target_snr=target_snr,
            alpha=alpha,
            beta=beta,
            input_domain="amplitude",
            snr_units=snr_units,
            match_mode=match_mode,
            scaling_rule="affine",
            eps=eps,
            input_floor=input_floor,
            seed=seed,
        )
        out_dtype = source_dtype
        degraded_pixels = linear_amplitude_to_pixels(result.degraded, config, out_dtype=out_dtype)
    else:
        config = None
        result = degrade_array_to_snr(
            clean_pixels,
            target_snr=target_snr,
            alpha=alpha,
            beta=beta,
            input_domain="intensity",
            snr_units=snr_units,
            match_mode=match_mode,
            scaling_rule="affine",
            eps=eps,
            input_floor=input_floor,
            seed=seed,
        )
        degraded_pixels = _cast_to_dtype(result.degraded, source_dtype)

    if output_dir is not None:
        resolved_output_dir = Path(output_dir)
    else:
        resolved_output_dir = Path(output_root) / f"degraded_speckle_{stack.folder.name}"
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    for frame_path, frame in zip(stack.frame_paths, degraded_pixels):
        tifffile.imwrite(resolved_output_dir / frame_path.name, frame)

    summary = {
        "input_folder": str(stack.folder),
        "output_dir": str(resolved_output_dir),
        "domain": domain_norm,
        "target_snr": {
            "value": float(target_snr),
            "units": snr_units.strip().lower(),
            "linear": float(result.target_snr_linear),
            "db": float(result.target_snr_db),
        },
        "achieved_snr": {
            "linear": float(result.achieved_snr_linear),
            "db": float(result.achieved_snr_db),
        },
        "alpha": float(alpha),
        "beta": float(beta),
        "lambda": float(result.lambda_value),
        "match_mode": result.match_mode,
        "scaling_rule": result.scaling_rule,
        "seed": int(seed),
        "shape": [int(x) for x in degraded_pixels.shape],
        "dtype": str(source_dtype),
        "eps": float(eps),
        "input_floor": float(result.input_floor),
        "transform_config": asdict(config) if config is not None else None,
    }

    metrics_json = resolved_output_dir / "degradation_metrics.json"
    _write_json(metrics_json, summary)
    return FolderDegradationResult(
        input_folder=stack.folder,
        output_dir=resolved_output_dir,
        metrics_json=metrics_json,
        summary=summary,
    )


def _validate_gamma(*, alpha: float, beta: float) -> None:
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be finite and positive.")
    if not np.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be finite and positive.")


def _normalize_target_snr(*, target_snr: float, snr_units: str) -> float:
    units = _normalize_choice(snr_units, valid={"db", "linear"}, name="snr_units")
    if not np.isfinite(target_snr):
        raise ValueError("target_snr must be finite.")
    if units == "linear":
        if target_snr <= 0.0:
            raise ValueError("target_snr must be positive when snr_units='linear'.")
        return float(target_snr)
    return float(10.0 ** (float(target_snr) / 10.0))


def _normalize_choice(value: str, *, valid: set[str], name: str) -> str:
    norm = value.strip().lower()
    if norm not in valid:
        expected = ", ".join(sorted(valid))
        raise ValueError(f"Unsupported {name} {value!r}; expected one of: {expected}.")
    return norm


def _normalize_input_floor(*, input_floor: float | None, eps: float) -> float:
    if input_floor is None:
        resolved = float(eps)
    else:
        resolved = float(input_floor)
    if not np.isfinite(resolved) or resolved <= 0.0:
        raise ValueError("input_floor must be finite and positive.")
    return resolved


def _resolve_rng(*, seed: int | None, rng: np.random.Generator | None) -> np.random.Generator:
    if seed is not None and rng is not None:
        raise ValueError("Provide exactly one of `seed` or `rng`.")
    if rng is not None:
        return rng
    return np.random.default_rng(seed)


def _to_intensity(values: np.ndarray, *, input_domain: str) -> np.ndarray:
    if input_domain == "amplitude":
        return np.square(values, dtype=np.float64)

    if np.any(values < 0.0):
        raise ValueError("Intensity inputs must be non-negative.")
    return np.asarray(values, dtype=np.float64)


def _from_intensity(noisy_intensity: np.ndarray, *, output_domain: str) -> np.ndarray:
    if output_domain == "amplitude":
        return np.sqrt(np.maximum(noisy_intensity, 0.0))
    return noisy_intensity


def _analytic_lambda(
    *,
    clean_intensity: np.ndarray,
    raw_noise: np.ndarray,
    target_snr_linear: float,
) -> float:
    base_noise = clean_intensity * (raw_noise - 1.0)
    finite = np.isfinite(base_noise) & np.isfinite(clean_intensity)
    if not np.any(finite):
        raise ValueError("No finite values available for analytic matching.")

    signal_mean = float(np.mean(clean_intensity[finite]))
    base_std = float(np.std(base_noise[finite]))
    if signal_mean <= 0.0:
        raise ValueError("clean intensity mean must be strictly positive.")
    target_noise_std = signal_mean / target_snr_linear

    if base_std == 0.0:
        if target_noise_std == 0.0:
            return 0.0
        raise ValueError("Cannot match target SNR because sampled noise has zero variance.")
    return max(0.0, target_noise_std / base_std)


def _exact_lambda(
    *,
    clean_intensity: np.ndarray,
    raw_noise: np.ndarray,
    target_snr_linear: float,
    eps: float,
) -> float:
    if target_snr_linear <= 0.0:
        raise ValueError("target_snr_linear must be positive.")

    low = 0.0
    high = 1.0
    high_snr = _achieved_linear_snr(
        clean_intensity=clean_intensity,
        raw_noise=raw_noise,
        lambda_value=high,
        eps=eps,
    )

    for _ in range(60):
        if not np.isfinite(high_snr) or high_snr > target_snr_linear:
            high *= 2.0
            high_snr = _achieved_linear_snr(
                clean_intensity=clean_intensity,
                raw_noise=raw_noise,
                lambda_value=high,
                eps=eps,
            )
            continue
        break
    else:
        raise ValueError("Unable to bracket a lambda value that reaches the requested SNR.")

    for _ in range(80):
        mid = 0.5 * (low + high)
        mid_snr = _achieved_linear_snr(
            clean_intensity=clean_intensity,
            raw_noise=raw_noise,
            lambda_value=mid,
            eps=eps,
        )
        if np.isfinite(mid_snr) and mid_snr > target_snr_linear:
            low = mid
        else:
            high = mid
        if np.isfinite(mid_snr):
            rel_err = abs(mid_snr - target_snr_linear) / target_snr_linear
            if rel_err < 1e-6:
                break

    return max(0.0, high)


def _achieved_linear_snr(
    *,
    clean_intensity: np.ndarray,
    raw_noise: np.ndarray,
    lambda_value: float,
    eps: float,
) -> float:
    noisy_intensity = _apply_affine_speckle(
        clean_intensity=clean_intensity,
        raw_noise=raw_noise,
        lambda_value=lambda_value,
        eps=eps,
    )
    return compute_snr(clean_intensity, noisy_intensity, units="linear")


def _apply_affine_speckle(
    *,
    clean_intensity: np.ndarray,
    raw_noise: np.ndarray,
    lambda_value: float,
    eps: float,
) -> np.ndarray:
    speckle = np.maximum(eps, 1.0 + float(lambda_value) * (raw_noise - 1.0))
    return clean_intensity * speckle


def _cast_to_dtype(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.rint(np.clip(arr, info.min, info.max)).astype(dtype)
    return arr.astype(dtype)


def _linear_to_db(snr_linear: float) -> float:
    if not np.isfinite(snr_linear):
        return float("inf")
    return float(10.0 * np.log10(max(snr_linear, np.finfo(np.float64).tiny)))


def _build_transform_config(
    *,
    source_dtype: np.dtype,
    transform_config: TransformConfig | None,
    transform_mode: str,
    max_pixel: int | None,
    gamma: float,
    amp_low: float | None,
    amp_high: float | None,
    db_low: float,
    db_high: float,
    eps: float,
) -> TransformConfig:
    if transform_config is not None:
        return transform_config

    if max_pixel is None:
        if np.issubdtype(source_dtype, np.integer) and np.iinfo(source_dtype).max > 255:
            inferred_max_pixel = 65535
        else:
            inferred_max_pixel = 255
    else:
        inferred_max_pixel = max_pixel

    return TransformConfig(
        mode=transform_mode,
        max_pixel=inferred_max_pixel,
        gamma=gamma,
        amp_low=amp_low,
        amp_high=amp_high,
        db_low=db_low,
        db_high=db_high,
        eps=eps,
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")


def _json_default(value: object) -> object:
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    raise TypeError(f"Type is not JSON serializable: {type(value)!r}")
