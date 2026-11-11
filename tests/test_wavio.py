"""WAV I/O round-trips and sample-rate conversion."""

import os
import tempfile

import numpy as np
import pytest

from convo import wavio


@pytest.fixture()
def tmpdir():
    with tempfile.TemporaryDirectory() as d:
        yield d


def test_mono_and_stereo_roundtrip(tmpdir):
    mono = np.linspace(-0.9, 0.9, 8000)[None, :]
    p = os.path.join(tmpdir, "mono.wav")
    wavio.write_wav(p, mono, 48000)
    back, sr = wavio.read_wav(p)
    assert sr == 48000 and back.shape[0] == 1
    np.testing.assert_allclose(back, mono, atol=5e-5)

    stereo = np.stack([np.sin(np.linspace(0, 20, 4000)),
                       np.cos(np.linspace(0, 20, 4000))])
    p2 = os.path.join(tmpdir, "stereo.wav")
    wavio.write_wav(p2, stereo, 44100)
    back2, sr2 = wavio.read_wav(p2)
    assert back2.shape[0] == 2 and sr2 == 44100
    np.testing.assert_allclose(back2, stereo, atol=5e-5)
    # L and R stay distinct: no implicit mono averaging on the I/O boundary.
    assert not np.allclose(back2[0], back2[1])


def test_8bit_and_24bit_read(tmpdir):
    for width, dtype, scale in ((1, np.uint8, 128), (3, None, None)):
        sr, n = 8000, 100
        import wave
        p = os.path.join(tmpdir, f"w{width}.wav")
        if width == 1:
            # Standard unsigned 8-bit PCM: bias 128, range 127.
            pcm = np.round(
                np.sin(np.linspace(0, 10, n)) * 0.9 * 127 + 128
            ).clip(0, 255).astype(np.uint8)
            with wave.open(p, "wb") as wf:
                wf.setnchannels(1); wf.setsampwidth(1)
                wf.setframerate(sr); wf.writeframes(pcm.tobytes())
        else:
            pcm = (np.sin(np.linspace(0, 10, n)) * 0.9 * 8388607).astype("<i4")
            raw = pcm.astype("<i4").tobytes()
            packed = b"".join(raw[i:i + 3] for i in range(0, len(raw), 4))
            with wave.open(p, "wb") as wf:
                wf.setnchannels(1); wf.setsampwidth(3)
                wf.setframerate(sr); wf.writeframes(packed)
        audio, got_sr = wavio.read_wav(p)
        assert got_sr == sr and audio.shape == (1, n)
        np.testing.assert_allclose(audio[0], np.sin(np.linspace(0, 10, n)) * 0.9,
                                   atol=2 / 128 if width == 1 else 2e-6)


def test_over_unity_is_scaled_not_hard_clipped(tmpdir):
    x = np.array([[0.5, 2.0, -3.0]])
    p = os.path.join(tmpdir, "hot.wav")
    wavio.write_wav(p, x, 16000)
    back, _ = wavio.read_wav(p)
    assert np.max(np.abs(back)) <= 1.0
    # Relative shape preserved (uniform scale).
    np.testing.assert_allclose(back[0] / back[0, 1], x[0] / x[0, 1], atol=1e-4)


def test_resample_doubles_and_halves_length():
    sr = 48000
    x = np.random.default_rng(0).standard_normal((2, 4800))
    up = wavio.resample_to(x, sr, 96000)
    assert up.shape[0] == 2 and 9500 <= up.shape[1] <= 9700
    down = wavio.resample_to(x, sr, 24000)
    assert 2350 <= down.shape[1] <= 2450
    # Identity resample returns the same data untouched.
    same = wavio.resample_to(x, sr, sr)
    np.testing.assert_array_equal(same, x)


def test_resample_preserves_low_frequency_content():
    sr0, sr1 = 24000, 48000
    t = np.arange(sr0) / sr0
    x = np.sin(2 * np.pi * 440 * t)[None, :]
    up = wavio.resample_to(x, sr0, sr1)[0]
    t2 = np.arange(up.size) / sr1
    expected = np.sin(2 * np.pi * 440 * t2)
    # Drop filter-edge samples before comparing amplitudes.
    np.testing.assert_allclose(up[200:-200], expected[200:-200], atol=0.05)


def test_corrupt_wav_raises_audio_error(tmpdir):
    p = os.path.join(tmpdir, "bad.wav")
    with open(p, "wb") as f:
        f.write(b"not a wav file at all")
    with pytest.raises(wavio.AudioError):
        wavio.read_wav(p)
