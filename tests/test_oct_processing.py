import numpy as np
import pytest

from denoising.oct_processing import (
    TransformConfig,
    linear_amplitude_to_pixels,
    pixels_to_linear_amplitude,
)


@pytest.mark.parametrize(
    ("db_low", "db_high"),
    [
        (-40.0, 0.0),
        (-20.0, 10.0),
        (10.0, 50.0),
    ],
)
def test_db_round_trip_pixels(db_low, db_high):
    config = TransformConfig(mode="db", db_low=db_low, db_high=db_high, max_pixel=255)
    pixels = np.linspace(0.0, 255.0, 513)

    amplitude = pixels_to_linear_amplitude(pixels, config)
    restored = linear_amplitude_to_pixels(amplitude, config)

    assert np.allclose(restored, pixels, atol=1e-8)


def test_db_nonzero_high_regression():
    config = TransformConfig(mode="db", db_low=-20.0, db_high=10.0, max_pixel=255)

    low_amplitude = pixels_to_linear_amplitude(np.array([0.0]), config)[0]
    round_trip_low = linear_amplitude_to_pixels(np.array([low_amplitude]), config)[0]

    assert np.isclose(low_amplitude, 10.0 ** (config.db_low / 20.0))
    assert np.isclose(round_trip_low, 0.0, atol=1e-8)


@pytest.mark.parametrize("max_pixel", [255, 65535])
def test_gamma_round_trip_pixels(max_pixel):
    config = TransformConfig(
        mode="gamma",
        max_pixel=max_pixel,
        gamma=0.4,
        amp_low=1e-3,
        amp_high=2.0,
    )
    pixels = np.linspace(0.0, float(max_pixel), 1025)

    amplitude = pixels_to_linear_amplitude(pixels, config)
    restored = linear_amplitude_to_pixels(amplitude, config)

    assert np.allclose(restored, pixels, atol=1e-8)


def test_output_dtype_and_range():
    config = TransformConfig(mode="db", db_low=-40, db_high=0, max_pixel=255)
    amplitude = np.array([1e-4, 1e-2, 0.1, 1.0, 10.0], dtype=np.float64)

    pixels_u8 = linear_amplitude_to_pixels(amplitude, config, out_dtype=np.uint8)
    assert pixels_u8.dtype == np.uint8
    assert pixels_u8.min() >= 0
    assert pixels_u8.max() <= 255

    config16 = TransformConfig(mode="db", db_low=-40, db_high=0, max_pixel=65535)
    pixels_u16 = linear_amplitude_to_pixels(amplitude, config16, out_dtype=np.uint16)
    assert pixels_u16.dtype == np.uint16
    assert pixels_u16.min() >= 0
    assert pixels_u16.max() <= 65535


def test_gamma_requires_explicit_cut_levels():
    with pytest.raises(ValueError):
        linear_amplitude_to_pixels(np.array([0.1]), TransformConfig(mode="gamma"))
