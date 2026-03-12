from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path

from denoising.denoisers import MMTVDenoiser, TemporalMeanDenoiser
from denoising.run.batch import denoise_folders
from noise_model import estimate_pairs


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Top-level OCT pipeline: denoise folders from CSV and estimate noise model."
    )
    parser.add_argument("--folders-csv", type=Path, required=True)
    parser.add_argument(
        "--denoiser",
        choices=("temporal_mean", "mmtv"),
        required=True,
    )
    parser.add_argument(
        "--row-index",
        type=int,
        default=None,
        help="0-based data-row index. If omitted, all rows are processed.",
    )
    parser.add_argument(
        "--denoise-output-root",
        type=Path,
        default=Path("outputs"),
    )
    parser.add_argument(
        "--noise-output-root",
        type=Path,
        default=Path("outputs/noise_model"),
    )

    parser.add_argument("--temporal-radius", type=int, default=1)

    parser.add_argument("--mmtv-max-iterations", type=int, default=1000)
    parser.add_argument("--mmtv-max-inner-iterations", type=int, default=20)
    parser.add_argument("--mmtv-alpha", type=float, default=1.1)
    parser.add_argument("--mmtv-beta", type=float, default=0.9)
    parser.add_argument("--mmtv-tv-reg-strength", type=float, default=1e-2)
    parser.add_argument("--mmtv-convergence-threshold", type=float, default=1e-6)
    parser.add_argument("--mmtv-min-intensity", type=float, default=1e-6)
    parser.add_argument("--mmtv-use-3d-tv", action="store_true")

    parser.add_argument("--transform-mode", choices=("db", "gamma"), default="db")
    parser.add_argument("--max-pixel", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=0.4)
    parser.add_argument("--amp-low", type=float, default=None)
    parser.add_argument("--amp-high", type=float, default=None)
    parser.add_argument("--db-low", type=float, default=-40.0)
    parser.add_argument("--db-high", type=float, default=0.0)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--sample-frames", type=int, default=6)
    parser.add_argument("--max-speckle-samples", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def _canonical_header(name: str) -> str:
    return name.strip().lower()


def _find_column(fieldnames: list[str], *candidates: str) -> str | None:
    by_canonical = {_canonical_header(name): name for name in fieldnames}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in by_canonical:
            return by_canonical[key]
    return None


def _load_csv(path: Path) -> tuple[list[str], list[dict[str, str]], str, str]:
    if not path.is_file():
        raise ValueError(f"CSV does not exist: {path}")

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header row: {path}")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV has no data rows: {path}")

    original_col = _find_column(fieldnames, "FOLDER", "original_folder")
    denoised_col = _find_column(fieldnames, "DENOISED", "denoised_folder")
    if original_col is None:
        raise ValueError(
            "CSV must contain original-folder column: FOLDER or original_folder."
        )
    if denoised_col is None:
        raise ValueError(
            "CSV must contain denoised-folder column: DENOISED or denoised_folder."
        )

    return fieldnames, rows, original_col, denoised_col


def _write_csv_atomic(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        newline="",
        encoding="utf-8",
        dir=path.parent,
        prefix=f"{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
        writer = csv.DictWriter(tmp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    os.replace(tmp_path, path)


def _build_denoiser(args: argparse.Namespace):
    if args.denoiser == "temporal_mean":
        return TemporalMeanDenoiser(radius=args.temporal_radius)

    return MMTVDenoiser(
        max_iterations=args.mmtv_max_iterations,
        max_inner_iterations=args.mmtv_max_inner_iterations,
        alpha=args.mmtv_alpha,
        beta=args.mmtv_beta,
        tv_reg_strength=args.mmtv_tv_reg_strength,
        convergence_threshold=args.mmtv_convergence_threshold,
        min_intensity=args.mmtv_min_intensity,
        use_3d_tv=args.mmtv_use_3d_tv,
    )


def _resolve_row_indices(total_rows: int, row_index: int | None) -> list[int]:
    if row_index is None:
        return list(range(total_rows))
    if row_index < 0 or row_index >= total_rows:
        raise ValueError(
            f"--row-index out of range: {row_index}. Valid range is 0 to {total_rows - 1}."
        )
    return [row_index]


def _process_row(
    *,
    row_idx: int,
    row: dict[str, str],
    original_col: str,
    denoised_col: str,
    denoiser,
    args: argparse.Namespace,
) -> tuple[str, bool]:
    original_text = (row.get(original_col) or "").strip()
    if not original_text:
        raise ValueError(
            f"row {row_idx}: original folder column '{original_col}' is empty."
        )

    original_folder = Path(original_text)
    if not original_folder.is_dir():
        raise ValueError(
            f"row {row_idx}: original folder does not exist: {original_folder}"
        )

    denoised_text = (row.get(denoised_col) or "").strip()
    denoised_folder = Path(denoised_text) if denoised_text else None
    csv_updated = False

    if denoised_folder is None or not denoised_folder.is_dir():
        denoised_folder = denoise_folders(
            [original_folder],
            denoiser=denoiser,
            output_root=args.denoise_output_root,
        )[0]
        row[denoised_col] = str(denoised_folder)
        csv_updated = True

    results = estimate_pairs(
        pairs=[(original_folder, denoised_folder)],
        output_root=args.noise_output_root,
        transform_mode=args.transform_mode,
        max_pixel=args.max_pixel,
        gamma=args.gamma,
        amp_low=args.amp_low,
        amp_high=args.amp_high,
        db_low=args.db_low,
        db_high=args.db_high,
        eps=args.eps,
        sample_frames=args.sample_frames,
        max_speckle_samples=args.max_speckle_samples,
        seed=args.seed,
    )
    noise_output = results[0].output_dir
    message = (
        f"row {row_idx}: original={original_folder} denoised={denoised_folder} "
        f"noise_output={noise_output}"
    )
    return message, csv_updated


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    fieldnames, rows, original_col, denoised_col = _load_csv(args.folders_csv)
    indices = _resolve_row_indices(len(rows), args.row_index)
    denoiser = _build_denoiser(args)

    csv_changed = False
    failures: list[str] = []
    for row_idx in indices:
        row = rows[row_idx]
        try:
            message, changed = _process_row(
                row_idx=row_idx,
                row=row,
                original_col=original_col,
                denoised_col=denoised_col,
                denoiser=denoiser,
                args=args,
            )
            csv_changed = csv_changed or changed
            print(f"[ok] {message}")
        except Exception as exc:
            failure = f"row {row_idx}: {exc}"
            if args.row_index is not None:
                raise SystemExit(failure) from exc
            failures.append(failure)
            print(f"[error] {failure}")

    if csv_changed:
        _write_csv_atomic(args.folders_csv, fieldnames, rows)
        print(f"[info] updated CSV in place: {args.folders_csv}")

    if failures:
        print("\nFailed rows:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
