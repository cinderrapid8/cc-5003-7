"""HTTP API and static web UI for the foley convolution workbench."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from . import jobs as jobs_mod
from . import pipeline, synth, wavio
from .pipeline import RenderParams, SUPPORTED_MAPS, TAIL_KEEP, TAIL_TRUNCATE
from .storage import JobRecord, JobStore

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    app.state.manager.shutdown()


def create_app(data_root: str = "data") -> FastAPI:
    app = FastAPI(title="Foley Convolution Workbench", version="1.0.0")
    store = JobStore(data_root)
    app.state.store = store
    app.state.manager = jobs_mod.JobManager(store)

    # ------------------------------------------------------------- static UI
    @app.get("/")
    def index():
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))

    # ------------------------------------------------------------- metadata
    @app.get("/api/sounds")
    def list_sounds():
        return {
            "sounds": [
                {"kind": k, "builder": k} for k in synth.SOUND_BUILDERS
            ],
            "sample_rate": jobs_mod.SOUND_SR,
        }

    @app.get("/api/channel-maps")
    def channel_maps():
        return {
            f"{in_ch}ch_input_{ir_ch}ch_ir": list(maps)
            for (in_ch, ir_ch), maps in SUPPORTED_MAPS.items()
        }

    @app.get("/api/jobs")
    def list_jobs():
        return {"jobs": [r.to_dict() for r in store.list()]}

    # ------------------------------------------------------------- job create
    @app.post("/api/jobs")
    async def create_job(
        sound: Optional[str] = Form(default=None),
        input_file: Optional[UploadFile] = File(default=None),
        ir_rt60: float = Form(default=0.9),
        ir_stereo: bool = Form(default=True),
        ir_sr: int = Form(default=jobs_mod.SOUND_SR),
        ir_file: Optional[UploadFile] = File(default=None),
        wet_ratio: float = Form(0.5),
        pre_delay_ms: float = Form(0.0),
        tail_policy: str = Form(TAIL_KEEP),
        channel_map: str = Form("auto"),
        block_size: Optional[int] = Form(default=None),
        input_name: Optional[str] = Form(default=None),
        ir_name: Optional[str] = Form(default=None),
    ):
        job_id = store.new_id()
        input_path = store.input_wav_path(job_id)
        ir_path = store.ir_wav_path(job_id)
        output_path = store.output_wav_path(job_id)

        # ---- dry input: built-in test sound or uploaded WAV
        if sound is not None:
            if sound not in synth.SOUND_BUILDERS:
                raise HTTPException(400, f"unknown sound {sound!r}")
            input_meta = jobs_mod.materialize_sound(sound, input_path)
        elif input_file is not None:
            _save_upload(input_file, input_path)
            input_meta = _probe(input_path, input_name or input_file.filename)
        else:
            raise HTTPException(400, "provide either 'sound' or 'input_file'")

        # ---- impulse response: synthetic room or uploaded WAV
        if ir_file is not None:
            _save_upload(ir_file, ir_path)
            ir_meta = _probe(ir_path, ir_name or ir_file.filename)
        else:
            if not 0.05 <= ir_rt60 <= 8.0:
                raise HTTPException(400, "ir_rt60 must be within [0.05, 8.0] s")
            if ir_sr not in (8000, 11025, 12000, 16000, 22050, 24000,
                             32000, 44100, 48000, 96000):
                raise HTTPException(400, "unsupported ir_sr")
            ir_meta = jobs_mod.materialize_ir(ir_path, ir_rt60, ir_stereo, ir_sr)

        # ---- parameter validation happens before anything is queued
        params = RenderParams(
            wet_ratio=wet_ratio,
            pre_delay_ms=pre_delay_ms,
            tail_policy=tail_policy,
            channel_map=channel_map,
            block_size=block_size,
        )
        try:
            params.validate()
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

        # Confirm the mapping is legal for these channel counts (fail fast).
        try:
            pipeline._resolve_map(input_meta["channels"], ir_meta["channels"], channel_map)
        except pipeline.PipelineError as exc:
            raise HTTPException(400, str(exc)) from exc

        snapshot = pipeline.params_snapshot(params, input_meta, ir_meta)
        rec = JobRecord(
            job_id=job_id,
            state="queued",
            input_name=input_meta["name"],
            ir_name=ir_meta["name"],
            input_path=input_path,
            ir_path=ir_path,
            output_path=output_path,
            params=snapshot,
        )
        store.create(rec)
        app.state.manager.submit(job_id)
        return rec.to_dict()

    # ------------------------------------------------------------- job control
    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        return _safe_get(store, job_id).to_dict()

    @app.get("/api/jobs/{job_id}/snapshot")
    def get_snapshot(job_id: str):
        return _safe_get(store, job_id).params

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        rec = app.state.manager.cancel(job_id)
        if rec.state not in ("queued", "running"):
            raise HTTPException(409, f"job is {rec.state}, cannot cancel")
        return rec.to_dict()

    @app.post("/api/jobs/{job_id}/rerun")
    def rerun_job(job_id: str):
        rec = _safe_get(store, job_id)
        if not os.path.exists(rec.input_path) or not os.path.exists(rec.ir_path):
            raise HTTPException(409, "saved input/IR missing; cannot reproduce render")
        if rec.state in ("queued", "running"):
            raise HTTPException(409, "job is still active")
        app.state.manager.submit(job_id)
        return store.get(job_id).to_dict()

    # ------------------------------------------------------------- downloads
    @app.get("/api/jobs/{job_id}/output")
    def download_output(job_id: str):
        rec = _safe_get(store, job_id)
        if rec.state != "completed" or not os.path.exists(rec.output_path):
            raise HTTPException(404, f"no completed output (state={rec.state})")
        return FileResponse(
            rec.output_path,
            media_type="audio/wav",
            filename=f"render_{job_id}.wav",
        )

    @app.get("/api/jobs/{job_id}/input")
    def download_input(job_id: str):
        rec = _safe_get(store, job_id)
        return FileResponse(rec.input_path, media_type="audio/wav",
                            filename=f"input_{job_id}.wav")

    @app.get("/api/jobs/{job_id}/ir")
    def download_ir(job_id: str):
        rec = _safe_get(store, job_id)
        return FileResponse(rec.ir_path, media_type="audio/wav",
                            filename=f"ir_{job_id}.wav")

    @app.exception_handler(wavio.AudioError)
    async def audio_error_handler(request, exc):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    return app


def _save_upload(upload: UploadFile, dest: str) -> None:
    if not (upload.filename or "").lower().endswith(".wav"):
        raise HTTPException(400, f"{upload.filename!r} is not a .wav file")
    with open(dest, "wb") as f:
        f.write(upload.file.read())
    # Fail fast on unreadable uploads.
    wavio.read_wav(dest)


def _probe(path: str, name: str) -> dict:
    audio, sr = wavio.read_wav(path)
    return {
        "name": name,
        "path": path,
        "sample_rate": int(sr),
        "channels": int(audio.shape[0]),
        "samples": int(audio.shape[1]),
        "duration_s": round(audio.shape[1] / sr, 4),
    }


def _safe_get(store: JobStore, job_id: str) -> JobRecord:
    try:
        return store.get(job_id)
    except Exception as exc:
        raise HTTPException(404, str(exc)) from exc


app = create_app()
