"""WAV reading/writing and sample-rate conversion.

All signals are represented internally as float64 in the range [-1, 1], shaped
``(channels, samples)`` with channels == 1 (mono) or 2 (stereo). Channel order
is always [left, right]; stereo impulse responses are never averaged down to
mono.
"""

from __future__ import annotations

import io
import wave

import numpy as np
from scipy import signal as sps


class AudioError(ValueError):
    pass


def read_wav(path: str) -> tuple[np.ndarray, int]:
    """Read a PCM WAV into a (channels, samples) float64 array."""
    try:
        with wave.open(path, "rb") as wf:
            nch = wf.getnchannels()
            sw = wf.getsampwidth()
            sr = wf.getframerate()
            nframes = wf.getnframes()
            raw = wf.readframes(nframes)
    except (wave.Error, EOFError, OSError) as exc:
        raise AudioError(f"cannot read WAV {path!r}: {exc}") from exc

    if nch not in (1, 2):
        raise AudioError(f"unsupported channel count {nch}; expected 1 or 2")
    if sr <= 0:
        raise AudioError("invalid sample rate in WAV header")

    if sw == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif sw == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    elif sw == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    elif sw == 3:
        triples = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int64)
        i32 = triples[:, 0] | (triples[:, 1] << 8) | (triples[:, 2] << 16)
        i32 = np.where(i32 >= 1 << 23, i32 - (1 << 24), i32)
        data = i32.astype(np.float64) / 8388608.0
    else:
        raise AudioError(f"unsupported sample width {sw * 8}-bit")

    if data.size == 0:
        return np.zeros((nch, 0), dtype=np.float64), sr
    data = data.reshape(-1, nch).T  # (channels, samples)
    return np.ascontiguousarray(data), sr


def write_wav(path: str, audio: np.ndarray, sample_rate: int) -> None:
    """Write a (channels, samples) float array as 16-bit PCM WAV."""
    a = np.asarray(audio, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, :]
    nch, n = a.shape
    if nch not in (1, 2):
        raise AudioError(f"unsupported output channel count {nch}")
    peak = float(np.max(np.abs(a))) if n else 0.0
    if peak > 1.0:
        # Soft headroom protection: scale down rather than hard-clip.
        a = a / (peak * 1.001)
    pcm = np.clip(a, -1.0, 1.0)
    pcm = (pcm * 32767.0).round().astype("<i2")
    interleaved = pcm.T.reshape(-1)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(nch)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(interleaved.tobytes())


def wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    write_wav_to_buffer(buf, audio, sample_rate)
    return buf.getvalue()


def write_wav_to_buffer(buf: io.IOBase, audio: np.ndarray, sample_rate: int) -> None:
    a = np.asarray(audio, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, :]
    nch, n = a.shape
    peak = float(np.max(np.abs(a))) if n else 0.0
    if peak > 1.0:
        a = a / (peak * 1.001)
    pcm = (np.clip(a, -1.0, 1.0) * 32767.0).round().astype("<i2")
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(nch)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.T.reshape(-1).tobytes())


def resample_to(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Polyphase resample (channels, samples) from src_sr to dst_sr.

    Uses zero-phase polyphase filtering so transients are not shifted. Each
    channel is filtered independently; L/R are never mixed.
    """
    if src_sr == dst_sr:
        return np.asarray(audio, dtype=np.float64)
    if src_sr <= 0 or dst_sr <= 0:
        raise AudioError("sample rates must be positive")
    from math import gcd

    g = gcd(int(src_sr), int(dst_sr))
    up = int(dst_sr) // g
    down = int(src_sr) // g
    return np.stack(
        [sps.resample_poly(ch, up, down) for ch in np.asarray(audio, dtype=np.float64)]
    )
