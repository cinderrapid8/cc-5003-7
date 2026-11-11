"""Pipeline-level tests: tail policy, pre-delay, channel maps, cancellation."""

import numpy as np
import pytest

from convo import dsp, pipeline
from convo.pipeline import JobCancelled, RenderParams, TAIL_KEEP, TAIL_TRUNCATE


def _mono(n=400, seed=0):
    return np.random.default_rng(seed).standard_normal(n)[None, :]


def _stereo_ir(ir_len=128):
    rng = np.random.default_rng(9)
    h = rng.standard_normal((2, ir_len))
    h[1] *= 0.5  # channels deliberately different
    return h


# --------------------------------------------------------------------- tail
def test_keep_tail_output_includes_full_reverb_tail():
    sr = 1000
    x = _mono(n=200)
    h = np.ones((1, 50))
    res = pipeline.render(x, sr, h, sr, RenderParams(tail_policy=TAIL_KEEP))
    # pre=0: length = N + L - 1, i.e. 49 tail samples beyond input end.
    assert res.output_samples == 200 + 50 - 1
    assert res.tail_samples == 49
    assert res.cut_samples == 0
    # The tail must contain real reverb energy, not zeros.
    assert np.max(np.abs(res.audio[0, 200:])) > 0


def test_truncate_requires_explicit_policy_and_reports_cut():
    sr = 1000
    x = _mono(n=200)
    h = np.ones((1, 50))
    res = pipeline.render(x, sr, h, sr, RenderParams(tail_policy=TAIL_TRUNCATE))
    assert res.output_samples == 200
    assert res.tail_samples == 0
    assert res.cut_samples == 49
    np.testing.assert_allclose(res.audio.shape[1], 200)


def test_default_policy_is_keep_not_silent_truncation():
    assert RenderParams().tail_policy == TAIL_KEEP


def test_predelay_shifts_wet_only_and_extends_length():
    sr = 1000
    x = np.ones((1, 10))
    h = np.ones((1, 5))
    pre_ms = 20.0
    pre = int(round(pre_ms / 1000 * sr))  # 20 samples
    res = pipeline.render(
        x, sr, h, sr, RenderParams(wet_ratio=1.0, pre_delay_ms=pre_ms)
    )
    assert res.output_samples == pre + 10 + 5 - 1
    # Fully wet: first `pre` samples must be silence (dry is zero at wet=1).
    assert np.all(res.audio[0, :pre] == 0.0)
    # At pre the wet convolution starts at its own sample 0.
    assert res.audio[0, pre] == pytest.approx(1.0)


def test_dry_path_has_no_predelay():
    sr = 1000
    x = np.ones((1, 10))
    h = np.ones((1, 4))
    res = pipeline.render(
        x, sr, h, sr, RenderParams(wet_ratio=0.0, pre_delay_ms=15.0)
    )
    # Fully dry: signal starts immediately at t=0 regardless of pre-delay.
    assert res.audio[0, 0] == pytest.approx(1.0)
    assert np.allclose(res.audio[0, :10], 1.0)


def test_wet_dry_mix_gains():
    sr = 1000
    x = np.array([[1.0, 0.0, 0.0, 0.0]])
    h = np.array([[1.0]])  # identity IR
    res = pipeline.render(x, sr, h, sr, RenderParams(wet_ratio=0.3))
    assert res.audio[0, 0] == pytest.approx(0.3 + 0.7)


# ------------------------------------------------------------- channel maps
def test_stereo_ir_is_never_averaged_by_default():
    sr = 1000
    x = _mono(n=64)
    h = _stereo_ir()
    res = pipeline.render(
        x, sr, h, sr, RenderParams(channel_map="auto", wet_ratio=1.0)
    )
    assert res.output_channels == 2
    # Mono input through two distinct IR channels gives two distinct outputs.
    assert not np.allclose(res.audio[0], res.audio[1])
    expected_l = dsp.overlap_save_convolve(x[0], h[0])
    expected_r = dsp.overlap_save_convolve(x[0], h[1])
    np.testing.assert_allclose(res.audio[0], expected_l, atol=1e-10)
    np.testing.assert_allclose(res.audio[1], expected_r, atol=1e-10)


def test_parallel_stereo_mapping():
    sr = 1000
    rng = np.random.default_rng(3)
    x = rng.standard_normal((2, 80))
    h = rng.standard_normal((2, 32))
    res = pipeline.render(x, sr, h, sr, RenderParams(channel_map="parallel",
                                                     wet_ratio=1.0))
    np.testing.assert_allclose(
        res.audio[0], dsp.overlap_save_convolve(x[0], h[0]), atol=1e-10
    )
    np.testing.assert_allclose(
        res.audio[1], dsp.overlap_save_convolve(x[1], h[1]), atol=1e-10
    )


def test_cross_mapping_swaps_input_channels():
    sr = 1000
    rng = np.random.default_rng(4)
    x = rng.standard_normal((2, 80))
    h = rng.standard_normal((2, 32))
    res = pipeline.render(x, sr, h, sr, RenderParams(channel_map="cross",
                                                     wet_ratio=1.0))
    np.testing.assert_allclose(
        res.audio[0], dsp.overlap_save_convolve(x[1], h[0]), atol=1e-10
    )
    np.testing.assert_allclose(
        res.audio[1], dsp.overlap_save_convolve(x[0], h[1]), atol=1e-10
    )


def test_explicit_downmix_is_opt_in_only():
    sr = 1000
    rng = np.random.default_rng(5)
    x = rng.standard_normal((2, 80))
    h = rng.standard_normal((2, 32))
    # Default parallel must differ from the explicit input downmix.
    auto = pipeline.render(
        x, sr, h, sr, RenderParams(channel_map="auto", wet_ratio=1.0)
    )
    down = pipeline.render(
        x, sr, h, sr,
        RenderParams(channel_map="explicit_input_downmix", wet_ratio=1.0),
    )
    assert not np.allclose(auto.audio, down.audio)
    x_mono = x.mean(axis=0)
    np.testing.assert_allclose(
        down.audio[0], dsp.overlap_save_convolve(x_mono, h[0]), atol=1e-10
    )


def test_explicit_ir_downmix_averages_ir_only():
    sr = 1000
    rng = np.random.default_rng(11)
    x = rng.standard_normal((2, 120))
    h = rng.standard_normal((2, 40))
    res = pipeline.render(
        x, sr, h, sr,
        RenderParams(channel_map="explicit_ir_downmix", wet_ratio=1.0),
    )
    h_mono = h.mean(axis=0)
    np.testing.assert_allclose(
        res.audio[0], dsp.overlap_save_convolve(x[0], h_mono), atol=1e-10
    )
    np.testing.assert_allclose(
        res.audio[1], dsp.overlap_save_convolve(x[1], h_mono), atol=1e-10
    )


def test_resampled_stereo_ir_keeps_distinct_channels():
    from convo.wavio import resample_to
    rng = np.random.default_rng(12)
    h = rng.standard_normal((2, 100))
    up = resample_to(h, 24000, 48000)
    assert up.shape == (2, 200)
    assert not np.allclose(up[0], up[1])


def test_illegal_mapping_rejected():
    sr = 1000
    x = _mono()
    h = np.ones((1, 16))
    with pytest.raises(pipeline.PipelineError):
        pipeline.render(x, sr, h, sr, RenderParams(channel_map="cross"))


def test_stereo_input_mono_ir_keeps_two_outputs():
    sr = 1000
    x = np.random.default_rng(6).standard_normal((2, 64))
    h = np.ones((1, 20))
    res = pipeline.render(x, sr, h, sr, RenderParams(wet_ratio=1.0))
    assert res.output_channels == 2
    np.testing.assert_allclose(
        res.audio[0], dsp.overlap_save_convolve(x[0], h[0]), atol=1e-10
    )
    np.testing.assert_allclose(
        res.audio[1], dsp.overlap_save_convolve(x[1], h[0]), atol=1e-10
    )


# ------------------------------------------------------------- resampling
def test_resampling_mismatched_rates_runs_at_input_rate():
    sr_in = 48000
    x = np.ones((1, 4800))  # 100 ms
    h = np.ones((1, 100))
    res = pipeline.render(x, sr_in, h, 24000, RenderParams())
    assert res.sample_rate == 48000
    # IR of 100 samples @24k -> ~200 samples @48k.
    ir_resampled_len = res.ir_samples
    assert 190 <= ir_resampled_len <= 210
    assert res.output_samples == 4800 + ir_resampled_len - 1


# ------------------------------------------------------------- validation
def test_empty_input_and_empty_ir_rejected():
    with pytest.raises(pipeline.PipelineError):
        pipeline.render(np.zeros((1, 0)), 1000, np.ones((1, 4)), 1000,
                        RenderParams())
    with pytest.raises(pipeline.PipelineError):
        pipeline.render(np.ones((1, 4)), 1000, np.zeros((1, 0)), 1000,
                        RenderParams())


def test_bad_params_rejected():
    with pytest.raises(pipeline.PipelineError):
        RenderParams(wet_ratio=1.5).validate()
    with pytest.raises(pipeline.PipelineError):
        RenderParams(pre_delay_ms=-1).validate()
    with pytest.raises(pipeline.PipelineError):
        RenderParams(tail_policy="discard").validate()
    with pytest.raises(pipeline.PipelineError):
        RenderParams(block_size=100).validate()


# ------------------------------------------------------------- cancellation
def test_cancel_between_blocks_raises():
    sr = 1000
    x = np.random.default_rng(7).standard_normal((1, 5000))
    h = np.random.default_rng(8).standard_normal((1, 400))
    calls = {"n": 0}

    def cancel_after_first():
        calls["n"] += 1
        return calls["n"] > 1  # cancel after the first checkpoint

    with pytest.raises(JobCancelled):
        pipeline.render(
            x, sr, h, sr,
            RenderParams(block_size=128),
            cancel_check=cancel_after_first,
        )


def test_cancel_checkpoint_is_per_block_not_per_sample():
    # A tiny single-block render must complete even with many checks;
    # cancellation is cooperative and checked between blocks/pairs.
    sr = 1000
    x = np.ones((1, 16))
    h = np.ones((1, 8))
    res = pipeline.render(
        x, sr, h, sr, RenderParams(block_size=1024),
        cancel_check=lambda: False,
    )
    assert res.output_samples == 16 + 8 - 1


# --------------------------------------------------------- reproducibility
def test_same_inputs_and_params_reproduce_bit_identical_output():
    sr = 1000
    rng = np.random.default_rng(10)
    x = rng.standard_normal((2, 300))
    h = rng.standard_normal((2, 96))
    params = RenderParams(wet_ratio=0.6, pre_delay_ms=5.0,
                          tail_policy=TAIL_KEEP, channel_map="parallel",
                          block_size=64)
    r1 = pipeline.render(x, sr, h, sr, params)
    r2 = pipeline.render(x, sr, h, sr, params)
    np.testing.assert_array_equal(r1.audio, r2.audio)
    assert r1.summary() == r2.summary()


def test_summary_reports_tail_and_length():
    sr = 1000
    x = _mono(n=120)
    h = np.ones((1, 30))
    res = pipeline.render(x, sr, h, sr, RenderParams(block_size=32))
    s = res.summary()
    assert s["output_samples"] == 120 + 30 - 1
    assert s["tail_samples_kept"] == 29
    assert s["block_size"] == 32
    assert s["fft_size"] == 64
    assert s["output_channels"] == 1
