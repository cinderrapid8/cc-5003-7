"""Background render executor with cooperative cancellation."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

from . import pipeline, synth, wavio
from .pipeline import JobCancelled, RenderParams
from .storage import JobRecord, JobStore

SOUND_SR = 48000


class JobManager:
    def __init__(self, store: JobStore, max_workers: int = 2):
        self.store = store
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="render"
        )
        self._futures: dict[str, object] = {}
        self._fut_lock = threading.Lock()

    # ------------------------------------------------------------- submission
    def submit(self, job_id: str) -> None:
        rec = self.store.get(job_id)
        if rec.state not in ("queued", "interrupted"):
            return
        self.store.update(job_id, state="queued", error="", cancel_requested=False)
        fut = self._pool.submit(self._run, job_id)
        with self._fut_lock:
            self._futures[job_id] = fut

    def _run(self, job_id: str) -> None:
        rec = self.store.get(job_id)
        try:
            self.store.update(job_id, state="running")
            input_audio, input_sr = wavio.read_wav(rec.input_path)
            ir_audio, ir_sr = wavio.read_wav(rec.ir_path)
            params = RenderParams(**rec.params["params"])

            def cancel_check() -> bool:
                return self.store.is_cancel_requested(job_id)

            result = pipeline.render(
                input_audio, input_sr, ir_audio, ir_sr, params, cancel_check
            )
            wavio.write_wav(rec.output_path, result.audio, result.sample_rate)
            self.store.update(
                job_id,
                state="completed",
                output_path=rec.output_path,
                summary=result.summary(),
            )
        except JobCancelled:
            self.store.update(job_id, state="cancelled", error="cancelled by user")
        except Exception as exc:  # rendered result must not crash the service
            self.store.update(job_id, state="failed", error=f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------ cancellation
    def cancel(self, job_id: str) -> JobRecord:
        return self.store.request_cancel(job_id)

    # ------------------------------------------------------------- housekeeping
    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


def materialize_sound(kind: str, out_path: str) -> dict:
    """Generate and persist one of the built-in dry test sounds."""
    builder = synth.SOUND_BUILDERS.get(kind)
    if builder is None:
        raise ValueError(f"unknown test sound {kind!r}")
    audio = builder(SOUND_SR)
    wavio.write_wav(out_path, audio, SOUND_SR)
    return {
        "name": kind,
        "path": out_path,
        "sample_rate": SOUND_SR,
        "channels": 1,
        "samples": int(audio.shape[0]),
        "duration_s": round(audio.shape[0] / SOUND_SR, 4),
    }


def materialize_ir(out_path: str, rt60_s: float, stereo: bool, sr: int) -> dict:
    audio = synth.room_impulse_response(sr=sr, rt60_s=rt60_s, stereo=stereo)
    wavio.write_wav(out_path, audio, sr)
    if audio.ndim == 1:
        audio = audio[None, :]
    return {
        "name": f"synthetic_room_rt60_{rt60_s}s",
        "path": out_path,
        "sample_rate": sr,
        "channels": audio.shape[0],
        "samples": audio.shape[1],
        "duration_s": round(audio.shape[1] / sr, 4),
        "rt60_s": rt60_s,
        "stereo": stereo,
    }
