from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from denoising.oct_processing import (
    TransformConfig,
    pixels_to_linear_amplitude,
)

from .io_utils import read_folder_stack, validate_matching_stacks
from .plotting import (
    save_distribution_overlay_figure,
    save_error_trends_figure,
    save_fit_traces_figure,
    save_triptych_figure,
)
from .stats import (
    GammaFit,
    compute_distribution_metrics,
    compute_reconstruction_metrics,
    compute_speckle_ratio,
    downsample_values,
    fit_gamma_methods,
)


@dataclass(frozen=True)
class FolderPair:
    original_folder: Path
    denoised_folder: Path


@dataclass(frozen=True)
class PairEstimationResult:
    original_folder: Path
    denoised_folder: Path
    output_dir: Path
    metrics_json: Path
    frame_metrics_csv: Path
    volume_metrics_csv: Path
    summary: dict[str, object]


def estimate_pairs(
    pairs: Sequence[tuple[Path | str, Path | str] | FolderPair] | None = None,
    *,
    pairs_csv: Path | str | None = None,
    output_root: Path | str = Path("outputs/noise_model"),
    transform_config: TransformConfig | None = None,
    transform_mode: str = "db",
    max_pixel: int | None = None,
    gamma: float = 0.4,
    amp_low: float | None = None,
    amp_high: float | None = None,
    db_low: float = -40.0,
    db_high: float = 0.0,
    eps: float = 1e-12,
    sample_frames: int = 6,
    max_speckle_samples: int = 2_000_000,
    seed: int = 0,
) -> list[PairEstimationResult]:
    folder_pairs = _resolve_pairs(pairs, pairs_csv=pairs_csv)
    if sample_frames <= 0:
        raise ValueError("sample_frames must be positive.")

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    master_rng = np.random.default_rng(seed)
    results: list[PairEstimationResult] = []
    for pair in folder_pairs:
        pair_seed = int(master_rng.integers(0, 2**31 - 1))
        pair_rng = np.random.default_rng(pair_seed)
        result = _estimate_single_pair(
            pair=pair,
            output_root=output_root,
            transform_config=transform_config,
            transform_mode=transform_mode,
            max_pixel=max_pixel,
            gamma=gamma,
            amp_low=amp_low,
            amp_high=amp_high,
            db_low=db_low,
            db_high=db_high,
            eps=eps,
            sample_frames=sample_frames,
            max_speckle_samples=max_speckle_samples,
            rng=pair_rng,
        )
        results.append(result)
    return results


def load_pairs_csv(path: Path | str) -> list[FolderPair]:
    csv_path = Path(path)
    if not csv_path.is_file():
        raise ValueError(f"Pair CSV does not exist: {csv_path}")

    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("Pair CSV must contain a header row.")

        fieldnames = {name.strip() for name in reader.fieldnames}
        required = {"original_folder", "denoised_folder"}
        if not required.issubset(fieldnames):
            raise ValueError(
                "Pair CSV must include required columns: original_folder, denoised_folder."
            )

        pairs: list[FolderPair] = []
        for row_idx, row in enumerate(reader, start=2):
            normalized_row = {
                (key.strip() if key is not None else key): value
                for key, value in row.items()
            }
            original_text = (normalized_row.get("original_folder") or "").strip()
            denoised_text = (normalized_row.get("denoised_folder") or "").strip()
            if not original_text and not denoised_text:
                continue
            if not original_text or not denoised_text:
                raise ValueError(
                    f"CSV row {row_idx} must include both original_folder and denoised_folder."
                )
            pairs.append(
                FolderPair(
                    original_folder=Path(original_text),
                    denoised_folder=Path(denoised_text),
                )
            )

    if not pairs:
        raise ValueError(f"No folder pairs found in CSV: {csv_path}")
    return pairs


def _resolve_pairs(
    pairs: Sequence[tuple[Path | str, Path | str] | FolderPair] | None,
    *,
    pairs_csv: Path | str | None,
) -> list[FolderPair]:
    has_pairs = pairs is not None
    has_csv = pairs_csv is not None
    if has_pairs == has_csv:
        raise ValueError("Provide exactly one of `pairs` or `pairs_csv`.")

    if has_csv:
        return load_pairs_csv(pairs_csv)

    resolved: list[FolderPair] = []
    assert pairs is not None
    for item in pairs:
        if isinstance(item, FolderPair):
            resolved.append(item)
            continue

        if len(item) != 2:
            raise ValueError("Each pair must contain exactly two folder paths.")
        original, denoised = item
        resolved.append(
            FolderPair(original_folder=Path(original), denoised_folder=Path(denoised))
        )

    if not resolved:
        raise ValueError("At least one folder pair is required.")
    return resolved


def _estimate_single_pair(
    *,
    pair: FolderPair,
    output_root: Path,
    transform_config: TransformConfig | None,
    transform_mode: str,
    max_pixel: int | None,
    gamma: float,
    amp_low: float | None,
    amp_high: float | None,
    db_low: float,
    db_high: float,
    eps: float,
    sample_frames: int,
    max_speckle_samples: int,
    rng: np.random.Generator,
) -> PairEstimationResult:
    original_stack = read_folder_stack(pair.original_folder)
    denoised_stack = read_folder_stack(pair.denoised_folder)
    validate_matching_stacks(original_stack, denoised_stack)

    config = _build_transform_config(
        source_dtype=original_stack.frames.dtype,
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

    original_pixels = np.asarray(original_stack.frames, dtype=np.float64)
    denoised_pixels = np.asarray(denoised_stack.frames, dtype=np.float64)
    original_linear = pixels_to_linear_amplitude(original_pixels, config)
    denoised_linear = pixels_to_linear_amplitude(denoised_pixels, config)

    pair_output_dir = output_root / _pair_slug(pair)
    triptych_dir = pair_output_dir / "figures" / "triptych"
    distribution_dir = pair_output_dir / "figures" / "distribution"
    diagnostics_dir = pair_output_dir / "figures" / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    domain_payloads = {
        "linear": {
            "original": original_linear,
            "denoised": denoised_linear,
            "ratio_power": 2,
        },
        "pixel": {
            "original": original_pixels,
            "denoised": denoised_pixels,
            "ratio_power": 2,
        },
    }

    domain_metrics: dict[str, object] = {}
    frame_rows: list[dict[str, float | int | str]] = []
    volume_rows: list[dict[str, float | int | str]] = []

    for domain_name, payload in domain_payloads.items():
        original_arr = payload["original"]
        denoised_arr = payload["denoised"]
        ratio_power = int(payload["ratio_power"])
        analysis = _analyze_domain(
            domain_name=domain_name,
            original=original_arr,
            denoised=denoised_arr,
            ratio_power=ratio_power,
            max_speckle_samples=max_speckle_samples,
            eps=eps,
            rng=rng,
        )
        domain_metrics[domain_name] = analysis["metrics"]
        frame_rows.extend(analysis["frame_rows"])
        volume_rows.extend(analysis["volume_rows"])

        sampled_indices = _sample_frame_indices(original_arr.shape[0], sample_frames)
        for frame_idx in sampled_indices:
            output_path = triptych_dir / f"{domain_name}_frame_{frame_idx:04d}.png"
            save_triptych_figure(
                original=original_arr[frame_idx],
                denoised=denoised_arr[frame_idx],
                speckle=analysis["speckle"][frame_idx],
                domain_name=domain_name,
                frame_index=frame_idx,
                output_path=output_path,
            )

        save_distribution_overlay_figure(
            samples=analysis["volume_samples"],
            fits=analysis["volume_fits"],
            domain_name=domain_name,
            output_path=distribution_dir / f"{domain_name}_volume_histogram.png",
        )
        save_fit_traces_figure(
            frame_rows=analysis["frame_rows"],
            domain_name=domain_name,
            output_path=diagnostics_dir / f"{domain_name}_fit_traces.png",
        )
        save_error_trends_figure(
            frame_rows=analysis["frame_rows"],
            domain_name=domain_name,
            output_path=diagnostics_dir / f"{domain_name}_error_trends.png",
        )

    summary = {
        "original_folder": str(pair.original_folder),
        "denoised_folder": str(pair.denoised_folder),
        "frame_count": int(original_stack.frames.shape[0]),
        "frame_shape": [int(x) for x in original_stack.frames.shape[1:]],
        "dtype": str(original_stack.frames.dtype),
        "transform_config": asdict(config),
        "domains": domain_metrics,
    }

    metrics_json = pair_output_dir / "metrics.json"
    frame_metrics_csv = pair_output_dir / "frame_metrics.csv"
    volume_metrics_csv = pair_output_dir / "volume_metrics.csv"

    pair_output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(metrics_json, summary)
    _write_csv(frame_metrics_csv, frame_rows)
    _write_csv(volume_metrics_csv, volume_rows)

    return PairEstimationResult(
        original_folder=pair.original_folder,
        denoised_folder=pair.denoised_folder,
        output_dir=pair_output_dir,
        metrics_json=metrics_json,
        frame_metrics_csv=frame_metrics_csv,
        volume_metrics_csv=volume_metrics_csv,
        summary=summary,
    )


def _analyze_domain(
    *,
    domain_name: str,
    original: np.ndarray,
    denoised: np.ndarray,
    ratio_power: int,
    max_speckle_samples: int,
    eps: float,
    rng: np.random.Generator,
) -> dict[str, object]:
    speckle, valid_mask = compute_speckle_ratio(
        original,
        denoised,
        eps=eps,
        ratio_power=ratio_power,
    )

    volume_reconstruction = compute_reconstruction_metrics(original, denoised, eps=eps)
    volume_samples = downsample_values(
        speckle[valid_mask], max_samples=max_speckle_samples, rng=rng
    )
    volume_fits = fit_gamma_methods(volume_samples)
    volume_distribution = {
        method: compute_distribution_metrics(volume_samples, fit, eps=eps)
        for method, fit in volume_fits.items()
    }

    frame_reconstruction: list[dict[str, float | int]] = []
    frame_fits: list[dict[str, object]] = []
    frame_distribution: list[dict[str, object]] = []
    frame_rows: list[dict[str, float | int | str]] = []
    volume_rows: list[dict[str, float | int | str]] = []

    n_frames = int(original.shape[0])
    for frame_idx in range(n_frames):
        frame_recon = compute_reconstruction_metrics(
            original[frame_idx], denoised[frame_idx], eps=eps
        )
        frame_reconstruction.append({"frame_index": frame_idx, **frame_recon})

        frame_samples = downsample_values(
            speckle[frame_idx][valid_mask[frame_idx]],
            max_samples=max_speckle_samples,
            rng=rng,
        )
        fits = fit_gamma_methods(frame_samples)
        dist_metrics = {
            method: compute_distribution_metrics(frame_samples, fit, eps=eps)
            for method, fit in fits.items()
        }
        frame_fits.append(
            {
                "frame_index": frame_idx,
                "mom": fits["mom"].to_dict(),
                "mle": fits["mle"].to_dict(),
            }
        )
        frame_distribution.append(
            {
                "frame_index": frame_idx,
                "mom": dist_metrics["mom"],
                "mle": dist_metrics["mle"],
            }
        )

        for method in ("mom", "mle"):
            fit = fits[method]
            dist = dist_metrics[method]
            frame_rows.append(
                {
                    "domain": domain_name,
                    "frame_index": frame_idx,
                    "fit_method": method,
                    "n_samples": fit.n_samples,
                    "alpha": float(fit.alpha),
                    "beta": float(fit.beta),
                    "mae": float(frame_recon["mae"]),
                    "rmse": float(frame_recon["rmse"]),
                    "relative_rmse": float(frame_recon["relative_rmse"]),
                    "kl_divergence": float(dist["kl_divergence"]),
                    "ks_statistic": float(dist["ks_statistic"]),
                    "ks_pvalue": float(dist["ks_pvalue"]),
                }
            )

    for method in ("mom", "mle"):
        fit = volume_fits[method]
        dist = volume_distribution[method]
        volume_rows.append(
            {
                "domain": domain_name,
                "fit_method": method,
                "n_samples": fit.n_samples,
                "alpha": float(fit.alpha),
                "beta": float(fit.beta),
                "mae": float(volume_reconstruction["mae"]),
                "rmse": float(volume_reconstruction["rmse"]),
                "relative_rmse": float(volume_reconstruction["relative_rmse"]),
                "kl_divergence": float(dist["kl_divergence"]),
                "ks_statistic": float(dist["ks_statistic"]),
                "ks_pvalue": float(dist["ks_pvalue"]),
            }
        )

    return {
        "speckle": speckle,
        "volume_samples": volume_samples,
        "volume_fits": volume_fits,
        "frame_rows": frame_rows,
        "volume_rows": volume_rows,
        "metrics": {
            "speckle_ratio_power": int(ratio_power),
            "reconstruction": {
                "volume": volume_reconstruction,
                "per_frame": frame_reconstruction,
            },
            "fits": {
                "volume": {
                    "mom": volume_fits["mom"].to_dict(),
                    "mle": volume_fits["mle"].to_dict(),
                },
                "per_frame": frame_fits,
            },
            "distribution_metrics": {
                "volume": volume_distribution,
                "per_frame": frame_distribution,
            },
        },
    }


def _sample_frame_indices(frame_count: int, sample_frames: int) -> list[int]:
    if frame_count <= sample_frames:
        return list(range(frame_count))
    indices = np.linspace(0, frame_count - 1, num=sample_frames, dtype=int)
    return sorted({int(idx) for idx in indices})


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


def _pair_slug(pair: FolderPair) -> str:
    original = pair.original_folder
    denoised = pair.denoised_folder
    digest = hashlib.sha1(
        f"{original.resolve()}|{denoised.resolve()}".encode("utf-8")
    ).hexdigest()[:10]
    return f"{original.name}__{denoised.name}_{digest}"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")


def _write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "domain",
        "frame_index",
        "fit_method",
        "n_samples",
        "alpha",
        "beta",
        "mae",
        "rmse",
        "relative_rmse",
        "kl_divergence",
        "ks_statistic",
        "ks_pvalue",
    ]
    has_frame_index = any("frame_index" in row for row in rows)
    if not has_frame_index:
        fieldnames = [name for name in fieldnames if name != "frame_index"]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            normalized = {key: row.get(key, "") for key in fieldnames}
            writer.writerow(normalized)


def _json_default(value: object) -> object:
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, GammaFit):
        return value.to_dict()
    raise TypeError(f"Type is not JSON serializable: {type(value)!r}")
