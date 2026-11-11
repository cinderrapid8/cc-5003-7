"""Numerical tests for partitioned overlap-save convolution."""

import numpy as np
import pytest

from convo import dsp
from convo.synth import unit_impulse


def test_matches_direct_convolution_short():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(37)
    h = rng.standard_normal(13)
    expected = dsp.direct_convolve(x, h)
    for b in (4, 8, 16, 32, 64):
        got = dsp.overlap_save_convolve(x, h, b)
        assert got.shape == expected.shape
        np.testing.assert_allclose(got, expected, atol=1e-12, rtol=1e-12)


def test_block_size_longer_than_signal():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(5)
    h = rng.standard_normal(4)
    np.testing.assert_allclose(
        dsp.overlap_save_convolve(x, h, 1024), dsp.direct_convolve(x, h), atol=1e-12
    )


def test_output_length_is_full_linear_convolution():
    n, l = 1000, 300
    x = np.random.default_rng(2).standard_normal(n)
    h = np.random.default_rng(3).standard_normal(l)
    got = dsp.overlap_save_convolve(x, h, 128)
    assert got.size == n + l - 1
    np.testing.assert_allclose(got, dsp.direct_convolve(x, h), atol=1e-10)


def test_unit_impulse_reproduces_impulse_response():
    """Convolving a unit impulse must return the IR itself (shift 0)."""
    h = np.array([0.0, 0.4, -0.7, 1.0, 0.25, -0.1, 0.02])
    impulse = unit_impulse(seconds=1.0)
    out = dsp.overlap_save_convolve(impulse, h, 8)
    np.testing.assert_allclose(out[: h.size], h, atol=1e-12)
    assert np.all(np.abs(out[h.size:]) < 1e-12)
    # And against direct convolution as oracle.
    np.testing.assert_allclose(
        out, dsp.direct_convolve(impulse, h), atol=1e-12
    )


def test_impulse_spanning_block_boundary():
    """An IR whose nonzero taps sit across a block boundary must stay aligned."""
    b = 16
    h = np.zeros(40)
    h[14] = 1.0   # end of first partition
    h[16] = -0.5  # first sample of second partition
    h[31] = 0.25  # last sample of second partition
    x = np.zeros(50)
    x[0] = 1.0
    x[20] = 0.8
    out = dsp.overlap_save_convolve(x, h, b)
    np.testing.assert_allclose(out, dsp.direct_convolve(x, h), atol=1e-12)
    # Direct check of the impulse-response alignment.
    assert out[14] == pytest.approx(1.0, abs=1e-12)
    assert out[16] == pytest.approx(-0.5, abs=1e-12)
    assert out[31] == pytest.approx(0.25, abs=1e-12)


def test_multi_block_impulse_train():
    """Many blocks: every delayed copy must line up with the oracle."""
    rng = np.random.default_rng(4)
    x = np.zeros(500)
    x[::37] = rng.choice([-1, 1], size=x[::37].size)
    h = rng.standard_normal(77)
    for b in (16, 32, 64):
        np.testing.assert_allclose(
            dsp.overlap_save_convolve(x, h, b),
            dsp.direct_convolve(x, h),
            atol=1e-10,
        )


def test_empty_input_and_empty_ir():
    h = np.array([1.0, 0.5])
    assert dsp.overlap_save_convolve(np.zeros(0), h).size == 0
    x = np.array([1.0, 0.5])
    assert dsp.overlap_save_convolve(x, np.zeros(0)).size == 0
    assert dsp.direct_convolve(np.zeros(0), h).size == 0


def test_single_sample_ir():
    """Extremely short IR: gain applied sample-by-sample, length N."""
    x = np.linspace(-1, 1, 100)
    h = np.array([0.7])
    out = dsp.overlap_save_convolve(x, h, 16)
    np.testing.assert_allclose(out, x * 0.7, atol=1e-12)
    assert out.size == x.size


def test_two_sample_ir_first_block_zero_padding():
    """First block has no past samples: explicit zero padding must hold."""
    x = np.array([1.0, 2.0, 3.0])
    h = np.array([1.0, 0.5])
    out = dsp.overlap_save_convolve(x, h, 4)
    np.testing.assert_allclose(out, dsp.direct_convolve(x, h), atol=1e-12)
    assert out[0] == pytest.approx(1.0)


def test_invalid_block_size_rejected():
    with pytest.raises(ValueError):
        dsp.overlap_save_convolve(np.ones(8), np.ones(4), block_size=12)


def test_auto_block_size_runs_and_matches():
    rng = np.random.default_rng(5)
    x = rng.standard_normal(5000)
    h = rng.standard_normal(4097)  # forces the larger auto block size
    got = dsp.overlap_save_convolve(x, h)
    np.testing.assert_allclose(got, dsp.direct_convolve(x, h), atol=1e-9)
    assert got.size == 5000 + 4097 - 1
