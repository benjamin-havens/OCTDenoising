# OCT Denoising (MMTV)

This repository contains a Python implementation of MMTV denoising for OCT images in [`mmtv.py`](./mmtv.py), based on:

Varadarajan, D., Magnain, C., Fogarty, M., Boas, D. A., Fischl, B., & Wang, H. (2022). *A novel algorithm for multiplicative speckle noise reduction in ex vivo human brain OCT images*. NeuroImage, 257, 119304. https://doi.org/10.1016/j.neuroimage.2022.119304

## What `mmtv.py` does

- Assumes multiplicative speckle noise with a Gamma model.
- Uses majorize-minimize updates with TV regularization.
- Solves each regularized subproblem with PDHG via ODL.
- Reads an OCT `.tif` image, denoises it, and saves `mmtv_comparison.png`.

## Requirements

- Python 3.9+
- Packages:
  - `numpy`
  - `odl`
  - `tqdm`
  - `tifffile`
  - `matplotlib`

Install with:

```bash
pip install numpy odl tqdm tifffile matplotlib
```

## Demo usage

Place an input image named `example.tif` in the project root (or change `example_path` in `main()`), then run:

```bash
python mmtv.py
```

Output:

- `mmtv_comparison.png` (input vs denoised image)

## Status

- `mmtv.py`: active denoising implementation.
- `artifact_detection.py`: work in progress (WIP).
