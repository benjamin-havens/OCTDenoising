# OCT Denoising

This repository contains OCT denoising utilities with a sequence-oriented denoiser framework, including MMTV in [`src/denoising/denoisers/mmtv.py`](./src/denoising/denoisers/mmtv.py), based on:

Varadarajan, D., Magnain, C., Fogarty, M., Boas, D. A., Fischl, B., & Wang, H. (2022). *A novel algorithm for multiplicative speckle noise reduction in ex vivo human brain OCT images*. NeuroImage, 257, 119304. https://doi.org/10.1016/j.neuroimage.2022.119304

## What This Repo Does

- Models speckle as multiplicative Gamma noise.
- Uses majorize-minimize updates with TV regularization.
- Solves regularized subproblems with PDHG via ODL.
- Provides a denoiser interface for running different methods over frame sequences.
- Provides run utilities for:
  - single-example comparison plots (`denoising.run.example`)
  - batch folder denoising (`denoising.run.batch`)
- Provides explicit OCT scaling transforms in [`src/denoising/oct_processing.py`](./src/denoising/oct_processing.py):
  - `mode="db"`: dB cut levels and linear dB scaling to/from pixel range.
  - `mode="gamma"`: power-law display compression with calibrated amplitude cut levels.

## Processing API

`denoising.oct_processing` exposes:

- `TransformConfig`
- `pixels_to_linear_amplitude(pixels, config)`
- `linear_amplitude_to_pixels(amplitude, config, out_dtype=None)`

### TransformConfig defaults

- `mode="gamma"`
- `max_pixel=255` (can also be `65535`)
- `gamma=0.4`
- `db_low=-40`, `db_high=0`
- `eps=1e-12`

Notes:

- In `gamma` mode, `amp_low` and `amp_high` are required and should be calibrated (noise floor and reflector ceiling).
- In `db` mode, `db_low` and `db_high` define which dB interval maps to `[0, max_pixel]`.

## MMTV API

`MMTVDenoiser` implements the sequence denoiser interface:

```python
from denoising.denoisers import MMTVDenoiser

denoiser = MMTVDenoiser(max_iterations=100)
denoised_sequence = denoiser.denoise_sequence(frames_linear)
```

- Input: 3D linear-amplitude array with shape `(n_frames, height, width)`.
- Output: denoised 3D linear-amplitude array with the same shape.
- Constraint: positivity floor only (`x >= min_intensity`), no upper bound.
- TV mode:
  - `use_3d_tv=False` (default): 2D TV per frame.
  - `use_3d_tv=True`: a single 3D TV solve over the full volume.

## Denoiser Interface

`denoising.denoisers` exposes:

- `BaseDenoiser`: base class for linear-amplitude sequences with:
  - `method_name`
  - `denoise_sequence(frames_linear)`
- `MMTVDenoiser(..., use_3d_tv=False)`: MMTV denoiser with optional 3D TV over a volume.
- `TemporalMeanDenoiser(radius=...)`: nearest-neighbor temporal averaging with window `2*radius+1`.

## Usage Examples

### 1) Pixel image -> linear amplitude -> denoise -> pixels

```python
import numpy as np
from denoising.oct_processing import (
    TransformConfig,
    pixels_to_linear_amplitude,
    linear_amplitude_to_pixels,
)
from denoising.denoisers import MMTVDenoiser

y_pixels = ...  # 8-bit or 16-bit OCT display image

config = TransformConfig(
    mode="gamma",
    max_pixel=255,      # or 65535
    gamma=0.4,
    amp_low=1e-3,       # calibrated
    amp_high=1.0,       # calibrated
)

y_linear = pixels_to_linear_amplitude(y_pixels, config)
denoiser = MMTVDenoiser()
x_hat_linear = denoiser.denoise_sequence(y_linear[np.newaxis, ...])[0]
x_hat_pixels = linear_amplitude_to_pixels(x_hat_linear, config, out_dtype=np.uint8)
```

### 2) Direct linear-amplitude workflow (ML target generation)

```python
from denoising.denoisers import MMTVDenoiser

volume_linear = ...  # shape: (n_frames, height, width)
denoiser = MMTVDenoiser(use_3d_tv=True)
denoised_volume = denoiser.denoise_sequence(volume_linear)
```

### 3) Batch denoise a list of folders (Python API)

Each input folder should contain TIFF files whose stem ends in a frame number (for example `scan_15.tif`). Frame numbers must be unique and contiguous, but do not need to start at 1.

```python
from pathlib import Path

from denoising.denoisers import TemporalMeanDenoiser
from denoising.oct_processing import TransformConfig
from denoising.run.batch import denoise_folders

folders = [Path("inputs/volume_a"), Path("inputs/volume_b")]
denoiser = TemporalMeanDenoiser(radius=2)
config = TransformConfig(
    mode="gamma",
    max_pixel=255,
    gamma=0.4,
    amp_low=1e-3,
    amp_high=1.0,
)

output_dirs = denoise_folders(
    folders,
    denoiser,
    output_root=Path("outputs"),
    transform_config=config,
)
```

Outputs are written under:

- `outputs/<method>/denoised_<method>_<input_folder_name>/`
- Output filenames match input filenames 1:1.

## Requirements

- Python 3.9+
- Packages:
  - `numpy`
  - `odl`
  - `scipy`
  - `tqdm`
  - `tifffile`
  - `matplotlib`
  - `pytest` (for tests)

Install with:

```bash
pip install numpy odl scipy tqdm tifffile matplotlib pytest
```

## Tests

Run:

```bash
PYTHONPATH=src pytest -q
```

## Run Demo

From the repository root:

```bash
PYTHONPATH=src python -m denoising.run.example
```

The comparison image is written to `outputs/` with a local-time prefix:
`YYYYMMDD_HHMMSS_<method>_comparison.png` (or `..._01.png`, `..._02.png` on collisions).

## Top-Level Pipeline CLI

`src/main.py` orchestrates:

1. CSV row selection
2. denoising each row's original folder when needed
3. noise-model estimation on the original/denoised pair
4. in-place CSV update of denoised folder paths

Example (strict denoiser names):

```bash
python src/main.py \
  --folders-csv inputs/phase_1_folders.csv \
  --denoiser temporal_mean \
  --row-index 2
```

Notes:

- `--row-index` is 0-based.
- Supported denoisers are `temporal_mean` and `mmtv`.
- If a row has an empty or missing denoised path, denoising is recomputed and the CSV is updated.

## Noise Model Estimation

`noise_model` estimates Gamma coefficients (`alpha`, `beta`, shape/rate form) from paired
original + denoised OCT folders.

Supported entry modes:

- Single pair: `--original-folder` + `--denoised-folder`
- Batch CSV: `--pairs-csv`

CSV schema (required columns):

- `original_folder`
- `denoised_folder`

### Noise model CLI

Single pair:

```bash
PYTHONPATH=src python -m noise_model.main \
  --original-folder /path/to/original_volume \
  --denoised-folder /path/to/denoised_volume \
  --output-root outputs/noise_model
```

Batch CSV:

```bash
PYTHONPATH=src python -m noise_model.main \
  --pairs-csv /path/to/pairs.csv \
  --output-root outputs/noise_model
```

Useful options:

- `--transform-mode {db,gamma}`
- `--max-pixel`, `--gamma`, `--amp-low`, `--amp-high`, `--db-low`, `--db-high`, `--eps`
- `--sample-frames` (default: `6`)
- `--max-speckle-samples` (default: `2000000`)
- `--seed` (default: `0`)

### Python API

```python
from noise_model import estimate_pairs

results = estimate_pairs(
    pairs=[("/path/to/original_volume", "/path/to/denoised_volume")],
    output_root="outputs/noise_model",
)
```

Each pair writes:

- `metrics.json`
- `frame_metrics.csv`
- `volume_metrics.csv`
- `figures/triptych/*.png`
- `figures/distribution/*.png`
- `figures/diagnostics/*.png`

Current tests cover:

- dB and gamma round-trip correctness.
- dB regression where `db_high != 0`.
- dtype/range behavior for 8-bit and 16-bit outputs.
- MMTV linear-domain behavior and positivity floor.
- sequence denoiser behavior (`MMTVDenoiser`, `TemporalMeanDenoiser`).
- batch folder denoising path/validation/output guarantees.
