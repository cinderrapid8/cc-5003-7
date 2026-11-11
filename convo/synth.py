"""Synthetic dry test signals: footsteps, door close, metal clash.

These are *dry source* signals only. Every wet render is produced at request
time by convolving the source with the uploaded/generated room impulse
response — nothing here is a pre-rendered reverberant file.
"""

from __future__ import annotations

import numpy as np


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _env(n: int, attack: int, decay_tau: float) -> np.ndarray:
    e = np.ones(n, dtype=np.float64)
    if attack > 0:
        e[:attack] = np.linspace(0.0, 1.0, attack, endpoint=False)
    tail = np.exp(-(np.arange(n - attack) + 1) / decay_tau)
    e[attack:] = tail
    return e


def footsteps(sr: int = 48000, steps: int = 6, seed: int = 7) -> np.ndarray:
    """A series of heel-strike impulses (filtered noise burst + low thump)."""
    rng = _rng(seed)
    gap = int(0.48 * sr)
    hit = int(0.09 * sr)
    out = np.zeros(steps * gap + hit, dtype=np.float64)
    for k in range(steps):
        start = k * gap
        noise = rng.standard_normal(hit)
        # Broadband click weighted toward low frequencies.
        t = np.arange(hit) / sr
        thump = np.sin(2 * np.pi * (70 + 10 * (k % 2)) * t)
        burst = 0.6 * noise + 0.8 * thump
        burst *= _env(hit, max(1, sr // 4800), hit / 6.0)
        out[start : start + hit] += burst
    peak = np.max(np.abs(out))
    return out / peak * 0.9


def door_close(sr: int = 48000, seed: int = 23) -> np.ndarray:
    """One short percussive slam: sharp crack plus a low resonant body."""
    rng = _rng(seed)
    n = int(0.35 * sr)
    t = np.arange(n) / sr
    crack_n = int(0.004 * sr)
    crack = rng.standard_normal(n) * np.exp(-t / 0.0015)
    body = (
        0.9 * np.sin(2 * np.pi * 110.0 * t)
        + 0.5 * np.sin(2 * np.pi * 165.0 * t)
        + 0.3 * np.sin(2 * np.pi * 230.0 * t)
    ) * np.exp(-t / 0.05)
    sig = crack * 0.7 + body
    sig[:crack_n] += rng.standard_normal(crack_n) * 0.5
    return sig / np.max(np.abs(sig)) * 0.9


def metal_clash(sr: int = 48000, seed: int = 41) -> np.ndarray:
    """Inharmonic metallic partials with a noisy strike and long ring."""
    rng = _rng(seed)
    n = int(1.2 * sr)
    t = np.arange(n) / sr
    # Inharmonic ratios give a metallic, non-musical clang.
    partials = [(1840, 1.0, 0.18), (2765, 0.7, 0.25), (4153, 0.5, 0.33),
                (5531, 0.35, 0.42), (7240, 0.22, 0.55)]
    sig = np.zeros(n, dtype=np.float64)
    for f, amp, tau in partials:
        sig += amp * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)) * np.exp(-t / tau)
    strike_n = int(0.006 * sr)
    sig[:strike_n] += 0.8 * rng.standard_normal(strike_n)
    return sig / np.max(np.abs(sig)) * 0.9


def room_impulse_response(
    sr: int = 48000,
    rt60_s: float = 0.9,
    stereo: bool = True,
    seed: int = 101,
) -> np.ndarray:
    """Synthetic room IR: direct click, early reflections, exponential decay.

    Stereo returns two decorrelated channels (different reflection patterns);
    the two channels are kept distinct and never averaged.
    """
    rng = _rng(seed)
    n = int(rt60_s * sr * 1.2)
    t = np.arange(n) / sr
    decay = np.exp(-6.9078 * t / rt60_s)  # ~ -60 dB at rt60
    channels = 2 if stereo else 1
    out = np.zeros((channels, n), dtype=np.float64)
    for c in range(channels):
        noise = rng.standard_normal(n)
        sig = noise * decay
        # Direct sound and a handful of discrete early reflections.
        sig[0] += 1.0
        for k in range(6):
            pos = int(rng.uniform(0.005, 0.08) * sr)
            if pos < n:
                sig[pos] += rng.uniform(0.25, 0.6) * decay[pos]
        # Gentle low-pass character of room air.
        kernel = np.ones(3) / 3.0
        sig = np.convolve(sig, kernel, mode="same")
        out[c] = sig / np.max(np.abs(sig)) * 0.95
    return out if stereo else out[0]


def unit_impulse(sr: int = 48000, seconds: float = 0.02) -> np.ndarray:
    """A single sample of amplitude 1 — convolving it must reproduce the IR."""
    sig = np.zeros(max(1, int(seconds * sr)), dtype=np.float64)
    sig[0] = 1.0
    return sig


SOUND_BUILDERS = {
    "footsteps": footsteps,
    "door_close": door_close,
    "metal_clash": metal_clash,
}
