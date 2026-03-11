# OCT Denoising (MMTV)

This repository contains a Python implementation of MMTV denoising for OCT images in [`src/denoising/mmtv.py`](./src/denoising/mmtv.py), based on:

Varadarajan, D., Magnain, C., Fogarty, M., Boas, D. A., Fischl, B., & Wang, H. (2022). *A novel algorithm for multiplicative speckle noise reduction in ex vivo human brain OCT images*. NeuroImage, 257, 119304. https://doi.org/10.1016/j.neuroimage.2022.119304

## What This Repo Does

- Models speckle as multiplicative Gamma noise.
- Uses majorize-minimize updates with TV regularization.
- Solves regularized subproblems with PDHG via ODL.
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

`MMTV_denoise` operates on **linear amplitude** directly:

```python
x_hat_linear = MMTV_denoise(y_linear)
```

- Input: 2D linear-amplitude array.
- Output: denoised 2D linear-amplitude array.
- Constraint: positivity floor only (`x >= min_intensity`), no upper bound.

## Usage Examples

### 1) Pixel image -> linear amplitude -> denoise -> pixels

```python
import numpy as np
from denoising.oct_processing import (
    TransformConfig,
    pixels_to_linear_amplitude,
    linear_amplitude_to_pixels,
)
from denoising.mmtv import MMTV_denoise

y_pixels = ...  # 8-bit or 16-bit OCT display image

config = TransformConfig(
    mode="gamma",
    max_pixel=255,      # or 65535
    gamma=0.4,
    amp_low=1e-3,       # calibrated
    amp_high=1.0,       # calibrated
)

y_linear = pixels_to_linear_amplitude(y_pixels, config)
x_hat_linear = MMTV_denoise(y_linear)
x_hat_pixels = linear_amplitude_to_pixels(x_hat_linear, config, out_dtype=np.uint8)
```

### 2) Direct linear-amplitude workflow (ML target generation)

```python
from denoising.mmtv import MMTV_denoise

y_linear = ...  # FFT amplitude domain data
x_hat_linear = MMTV_denoise(y_linear)
```

## Requirements

- Python 3.9+
- Packages:
  - `numpy`
  - `odl`
  - `tqdm`
  - `tifffile`
  - `matplotlib`
  - `pytest` (for tests)

Install with:

```bash
pip install numpy odl tqdm tifffile matplotlib pytest
```

## Tests

Run:

```bash
PYTHONPATH=src pytest -q
```

## Run Demo

From the repository root:

```bash
PYTHONPATH=src python -m denoising.mmtv
```

The comparison image is written to `outputs/` with a local-time prefix:
`YYYYMMDD_HHMMSS_mmtv_comparison.png` (or `..._01.png`, `..._02.png` on collisions).

Current tests cover:

- dB and gamma round-trip correctness.
- dB regression where `db_high != 0`.
- dtype/range behavior for 8-bit and 16-bit outputs.
- MMTV linear-domain behavior and positivity floor.
