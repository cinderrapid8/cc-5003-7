"""Rendering pipeline: resample, map channels, pre-delay, block convolution, mix.

Tail policy (explicit, never silently dropped):
  * ``keep`` (default): the render keeps the full reverb tail, output length is
    pre_delay + N + L - 1 samples. Input ending != reverb ending.
  * ``truncate``: everything after pre_delay + N samples is cut. This must be
    requested explicitly by the caller.

Channel mapping is explicit. A stereo impulse response is *never* averaged to
mono unless the caller selects a mapping whose documented behavior is exactly
that downmix.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Callable

import numpy as np

from . import dsp
from .wavio import resample_to

TAIL_KEEP = "keep"
TAIL_TRUNCATE = "truncate"

# input_channels x ir_channels -> allowed mapping names
SUPPORTED_MAPS = {
    (1, 1): ("mono",),
    (1, 2): ("mono_input_stereo_ir",),
    (2, 1): ("stereo_input_mono_ir", "explicit_input_downmix"),
    (2, 2): (
        "parallel",  # L=xL*hL, R=xR*hR (default)
        "cross",  # L=xR*hL, R=xL*hR
        "explicit_input_downmix",  # avg(xL,xR) * hL / hR -> stereo
        "explicit_ir_downmix",  # xL/xR * avg(hL,hR) -> stereo
    ),
}


class PipelineError(ValueError):
    pass


class JobCancelled(RuntimeError):
    pass


@dataclass
class RenderParams:
    wet_ratio: float = 0.5  # 0 = fully dry, 1 = fully wet
    pre_delay_ms: float = 0.0
    tail_policy: str = TAIL_KEEP
    channel_map: str = "auto"
    block_size: int | None = None

    def validate(self) -> None:
        if not 0.0 <= self.wet_ratio <= 1.0:
            raise PipelineError("wet_ratio must be within [0, 1]")
        if self.pre_delay_ms < 0.0:
            raise PipelineError("pre_delay_ms must be >= 0")
        if self.tail_policy not in (TAIL_KEEP, TAIL_TRUNCATE):
            raise PipelineError(f"unknown tail_policy {self.tail_policy!r}")
        if self.block_size is not None:
            b = self.block_size
            if b < 1 or (b & (b - 1)) != 0:
                raise PipelineError("block_size must be a power of two")


@dataclass
class RenderResult:
    audio: np.ndarray
    sample_rate: int
    output_samples: int
    output_channels: int
    duration_s: float
    tail_samples: int  # reverb-tail samples beyond the (delayed) input end
    cut_samples: int  # samples removed by explicit truncation
    block_size: int
    fft_size: int
    input_samples: int
    ir_samples: int
    peak_per_channel: list[float] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "sample_rate": self.sample_rate,
            "output_channels": self.output_channels,
            "output_samples": self.output_samples,
            "duration_s": round(self.duration_s, 4),
            "tail_samples_kept": self.tail_samples,
            "tail_samples_cut": self.cut_samples,
            "block_size": self.block_size,
            "fft_size": self.fft_size,
            "input_samples": self.input_samples,
            "ir_samples": self.ir_samples,
            "peak_per_channel": [round(v, 6) for v in self.peak_per_channel],
        }


def _check(cancel: Callable[[], bool] | None) -> None:
    if cancel is not None and cancel():
        raise JobCancelled("render cancelled between convolution blocks")


def _resolve_map(in_ch: int, ir_ch: int, requested: str) -> str:
    allowed = SUPPORTED_MAPS[(in_ch, ir_ch)]
    if requested == "auto":
        defaults = {
            (1, 1): "mono",
            (1, 2): "mono_input_stereo_ir",
            (2, 1): "stereo_input_mono_ir",
            (2, 2): "parallel",
        }
        return defaults[(in_ch, ir_ch)]
    if requested not in allowed:
        raise PipelineError(
            f"channel_map {requested!r} is not valid for {in_ch}ch input + "
            f"{ir_ch}ch IR; allowed: {', '.join(allowed)}"
        )
    return requested


def _build_channel_pairs(
    x: np.ndarray, h: np.ndarray, mapping: str
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return the (input channel, IR channel) pairs to convolve."""
    in_ch, ir_ch = x.shape[0], h.shape[0]
    if mapping == "mono":
        return [(x[0], h[0])]
    if mapping == "mono_input_stereo_ir":
        # Mono source projected through a true stereo IR — no averaging.
        return [(x[0], h[0]), (x[0], h[1])]
    if mapping == "stereo_input_mono_ir":
        return [(x[0], h[0]), (x[1], h[0])]
    if mapping == "parallel":
        return [(x[0], h[0]), (x[1], h[1])]
    if mapping == "cross":
        return [(x[1], h[0]), (x[0], h[1])]
    if mapping == "explicit_input_downmix":
        x_mono = x.mean(axis=0)  # explicit, user-requested downmix
        if ir_ch == 1:
            return [(x_mono, h[0])]
        return [(x_mono, h[0]), (x_mono, h[1])]
    if mapping == "explicit_ir_downmix":
        h_mono = h.mean(axis=0)  # explicit, user-requested downmix
        return [(x[0], h_mono), (x[1], h_mono)]
    raise PipelineError(f"unsupported channel_map {mapping!r}")


def render(
    input_audio: np.ndarray,
    input_sr: int,
    ir_audio: np.ndarray,
    ir_sr: int,
    params: RenderParams,
    cancel_check: Callable[[], bool] | None = None,
) -> RenderResult:
    """Run the full render pipeline.

    ``input_audio`` / ``ir_audio`` are (channels, samples) float arrays.
    Raises :class:`JobCancelled` if ``cancel_check`` returns True between blocks.
    """
    params.validate()
    x = np.asarray(input_audio, dtype=np.float64)
    h = np.asarray(ir_audio, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    if h.ndim == 1:
        h = h[None, :]
    if x.shape[0] not in (1, 2) or h.shape[0] not in (1, 2):
        raise PipelineError("only mono and stereo signals are supported")
    if x.shape[1] == 0:
        raise PipelineError("input signal is empty")
    if h.shape[1] == 0:
        raise PipelineError("impulse response is empty")

    # 1. Sample-rate conversion first; the IR is resampled to the input rate so
    #    both share one clock for convolution and delay arithmetic.
    if ir_sr != input_sr:
        h = resample_to(h, ir_sr, input_sr)
    sr = input_sr

    # 2. Explicit channel mapping (no implicit stereo->mono averaging).
    mapping = _resolve_map(x.shape[0], h.shape[0], params.channel_map)
    pairs = _build_channel_pairs(x, h, mapping)

    pre = int(round(params.pre_delay_ms / 1000.0 * sr))
    n_in = x.shape[1]
    ir_len = h.shape[1]

    # 3. Per-pair partitioned overlap-save convolution.
    wet_chunks: list[np.ndarray] = []
    dry_chunks: list[np.ndarray] = []
    used_block = params.block_size or dsp._next_block_size(ir_len, None)
    for x_ch, h_ch in pairs:
        _check(cancel_check)
        wet = dsp.overlap_save_convolve(x_ch, h_ch, params.block_size)
        wet_chunks.append(wet)
        # Dry counterpart is the exact source channel fed into this pair.
        dry_chunks.append(x_ch)
    wet_full = np.stack(wet_chunks)  # (out_ch, N + L - 1)
    dry_full = np.stack(dry_chunks)
    out_ch = wet_full.shape[0]

    conv_len = n_in + ir_len - 1
    keep_len = pre + conv_len  # full tail retained
    if params.tail_policy == TAIL_KEEP:
        final_len = keep_len
        cut = 0
    else:
        # Explicit user choice: cut the tail that extends past the input end.
        final_len = pre + n_in
        cut = max(0, keep_len - final_len)

    wet_gain = params.wet_ratio
    dry_gain = 1.0 - params.wet_ratio
    output = np.zeros((out_ch, final_len), dtype=np.float64)
    # Wet path: pre-delay shifts the whole reverberant signal.
    wet_end = min(pre + conv_len, final_len)
    output[:, pre:wet_end] += wet_gain * wet_full[:, : wet_end - pre]
    # Dry path stays at time zero; its silence tail is implicit in output.
    dry_end = min(n_in, final_len)
    output[:, :dry_end] += dry_gain * dry_full[:, :dry_end]

    _check(cancel_check)

    tail_kept = max(0, final_len - pre - n_in)
    return RenderResult(
        audio=output,
        sample_rate=sr,
        output_samples=final_len,
        output_channels=out_ch,
        duration_s=final_len / sr,
        tail_samples=tail_kept,
        cut_samples=cut,
        block_size=used_block,
        fft_size=2 * used_block,
        input_samples=n_in,
        ir_samples=ir_len,
        peak_per_channel=[float(np.max(np.abs(output[c]))) for c in range(out_ch)],
    )


def params_snapshot(params: RenderParams, input_meta: dict, ir_meta: dict) -> dict:
    """Serializable snapshot persisted with every job for reproducibility."""
    return {
        "params": asdict(params),
        "resolved_tail_policy": params.tail_policy,
        "input": input_meta,
        "impulse_response": ir_meta,
    }
