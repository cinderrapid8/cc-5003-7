"""Block-based frequency-domain convolution (uniformly partitioned overlap-save).

The input signal is never convolved sample-by-sample. For an impulse response of
length L and block size B, the IR is split into P = ceil(L / B) partitions of B
samples and every input block is transformed once with an FFT of size 2B.
Partition products are accumulated over an explicit block history, so the cost
per output block is O(P * B log B), independent of total signal length.

Indexing convention (overlap-save):
  * input block p packs [x[(p-1)B .. pB-1], x[pB .. (p+1)B-1]] into a 2B
    buffer — previous block in the left half, current block in the right half;
    out-of-range samples are zero. Block 0 therefore has an all-zero left half
    (explicit first-block zero padding), and blocks after the input end reuse
    the trailing samples with right-side zero padding;
  * IR partition q occupies the first B samples of its own 2B buffer;
  * output block p is sum over q of H_q * X_{p-q}; the inverse FFT yields 2B
    samples whose first B are circular wrap-around (discarded) and whose last B
    are the linear-convolution output;
  * total blocks = ceil(N/B) + P: the final IR partition still produces one
    full tail block after the last input block is ingested.

The full linear convolution of length N + L - 1 is always returned. Callers
decide whether to keep or cut the reverb tail — this module never truncates.
"""

from __future__ import annotations

import numpy as np


def _next_block_size(ir_len: int, requested: int | None) -> int:
    if requested is not None:
        b = int(requested)
        if b < 1 or (b & (b - 1)) != 0:
            raise ValueError("block_size must be a positive power of two")
        return b
    target = 2048
    if ir_len > 16384:
        target = 4096
    return target


def _input_spectrum(x: np.ndarray, p: int, b: int, fft_len: int) -> np.ndarray:
    """Build and transform input block p (previous block | current block)."""
    buf = np.zeros(fft_len, dtype=np.float64)
    prev = x[(p - 1) * b : p * b]
    cur = x[p * b : (p + 1) * b]
    buf[: prev.size] = prev       # empty for p == 0: first-block zero padding
    buf[b : b + cur.size] = cur   # empty after the signal end: tail zero pad
    return np.fft.rfft(buf)


def overlap_save_convolve(
    signal: np.ndarray,
    impulse: np.ndarray,
    block_size: int | None = None,
) -> np.ndarray:
    """Full linear convolution via partitioned overlap-save.

    Parameters
    ----------
    signal:
        1-D float array, the dry input.
    impulse:
        1-D float array, the impulse response.
    block_size:
        FFT block size B (power of two). The FFT length is 2B. Chosen
        automatically when omitted.

    Returns
    -------
    np.ndarray
        Full linear convolution, length ``len(signal) + len(impulse) - 1``.
        An empty input or empty IR yields an empty array.
    """
    x = np.asarray(signal, dtype=np.float64)
    h = np.asarray(impulse, dtype=np.float64)
    if x.ndim != 1 or h.ndim != 1:
        raise ValueError("overlap_save_convolve expects 1-D arrays")
    if x.size == 0 or h.size == 0:
        return np.zeros(0, dtype=np.float64)

    n = x.size
    ir_len = h.size
    b = _next_block_size(ir_len, block_size)
    fft_len = 2 * b
    half = fft_len // 2 + 1

    # IR partitions of exactly B samples (last one zero-padded at its end),
    # each in the first half of a 2B FFT buffer, transformed exactly once.
    partitions = int(np.ceil(ir_len / b))
    h_spectra = np.empty((partitions, half), dtype=np.complex128)
    for q in range(partitions):
        buf = np.zeros(fft_len, dtype=np.float64)
        part = h[q * b : (q + 1) * b]
        buf[: part.size] = part
        h_spectra[q] = np.fft.rfft(buf)

    input_blocks = int(np.ceil(n / b))
    # One output block per input block plus P tail blocks.
    total_blocks = input_blocks + partitions
    out = np.zeros(total_blocks * b, dtype=np.float64)

    # Explicit block history: X[p] is read by blocks p .. p+P-1, so keeping
    # one spectrum per processed block avoids any circular-slot aliasing.
    x_history: list[np.ndarray] = []
    for p in range(total_blocks):
        # Out-of-range slices are empty, so tail blocks are built from the
        # same rule as input blocks with zero padding on both sides.
        x_history.append(_input_spectrum(x, p, b, fft_len))

        acc = np.zeros(half, dtype=np.complex128)
        for q in range(partitions):
            if p - q >= 0:  # input block p-q has actually been ingested
                acc += h_spectra[q] * x_history[p - q]

        time_domain = np.fft.irfft(acc, n=fft_len)
        # Overlap-save: discard the first B (circular wrap), keep the last B.
        out[p * b : (p + 1) * b] = time_domain[b:]

    full_len = n + ir_len - 1
    return out[:full_len].astype(np.float64, copy=False)


def direct_convolve(signal: np.ndarray, impulse: np.ndarray) -> np.ndarray:
    """Reference O(N*L) direct convolution, used as the numerical oracle."""
    x = np.asarray(signal, dtype=np.float64).ravel()
    h = np.asarray(impulse, dtype=np.float64).ravel()
    if x.size == 0 or h.size == 0:
        return np.zeros(0, dtype=np.float64)
    return np.convolve(x, h, mode="full")
