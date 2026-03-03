import numpy as np
import pytest

import mmtv

pytestmark = pytest.mark.skipif(
    mmtv.odl is None, reason="odl is required for mmtv tests"
)


def test_mmtv_accepts_linear_amplitudes_above_one():
    y_linear = np.full((16, 16), 2.0, dtype=np.float64)

    denoised = mmtv.MMTV_denoise(
        y_linear,
        max_iterations=1,
        max_inner_iterations=3,
        tv_reg_strength=1e-2,
        min_intensity=1e-6,
    )

    assert denoised.shape == y_linear.shape
    assert np.isfinite(denoised).all()
    assert denoised.min() >= 1e-6
    assert denoised.max() > 1.0


def test_mmtv_rejects_non_positive_min_intensity():
    y_linear = np.ones((8, 8), dtype=np.float64)
    with pytest.raises(ValueError, match="min_intensity must be positive"):
        mmtv.MMTV_denoise(y_linear, min_intensity=0.0)
