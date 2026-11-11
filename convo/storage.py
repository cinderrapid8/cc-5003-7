"""On-disk job store and in-process task registry with cancellation.

Layout under the data root::

    data/
      inputs/<job_id>.wav            # saved dry input signal
      irs/<job_id>.wav               # saved impulse response
      outputs/<job_id>.wav           # rendered, downloadable track
      jobs/<job_id>.json             # parameter + state snapshot (atomic)

Reloading the project rebuilds the registry from ``jobs/*.json``; a job left
``running`` by a crashed process is marked ``interrupted`` and can be re-rendered
from its saved inputs, IR and parameter snapshot to reproduce the same space.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

STATUSES = ("queued", "running", "completed", "failed", "cancelled", "interrupted")


class StorageError(RuntimeError):
    pass


@dataclass
class JobRecord:
    job_id: str
    state: str = "queued"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    input_name: str = ""
    ir_name: str = ""
    input_path: str = ""
    ir_path: str = ""
    output_path: str = ""
    params: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)
    error: str = ""
    cancel_requested: bool = False
    seq: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "state": self.state,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "input_name": self.input_name,
            "ir_name": self.ir_name,
            "input_path": self.input_path,
            "ir_path": self.ir_path,
            "output_path": self.output_path,
            "params": self.params,
            "summary": self.summary,
            "error": self.error,
            "cancel_requested": self.cancel_requested,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "JobRecord":
        known = {k: d[k] for k in d if k in cls.__dataclass_fields__}
        return cls(**known)


class JobStore:
    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        self.jobs_dir = os.path.join(self.root, "jobs")
        self.inputs_dir = os.path.join(self.root, "inputs")
        self.irs_dir = os.path.join(self.root, "irs")
        self.outputs_dir = os.path.join(self.root, "outputs")
        for d in (self.jobs_dir, self.inputs_dir, self.irs_dir, self.outputs_dir):
            os.makedirs(d, exist_ok=True)
        self._records: dict[str, JobRecord] = {}
        self._lock = threading.RLock()
        self._seq = 0
        self._reload()

    # ------------------------------------------------------------------ paths
    def new_id(self) -> str:
        return uuid.uuid4().hex

    def input_wav_path(self, job_id: str) -> str:
        return os.path.join(self.inputs_dir, f"{job_id}.wav")

    def ir_wav_path(self, job_id: str) -> str:
        return os.path.join(self.irs_dir, f"{job_id}.wav")

    def output_wav_path(self, job_id: str) -> str:
        return os.path.join(self.outputs_dir, f"{job_id}.wav")

    def _snapshot_path(self, job_id: str) -> str:
        return os.path.join(self.jobs_dir, f"{job_id}.json")

    # ------------------------------------------------------------- persistence
    def _reload(self) -> None:
        for name in sorted(os.listdir(self.jobs_dir)):
            if not name.endswith(".json"):
                continue
            with open(os.path.join(self.jobs_dir, name), "r", encoding="utf-8") as f:
                rec = JobRecord.from_dict(json.load(f))
            if rec.state in ("queued", "running"):
                rec.state = "interrupted"
                rec.error = "process restarted before the render finished"
                self._write_snapshot(rec)
            self._records[rec.job_id] = rec
        self._seq = max((r.seq for r in self._records.values()), default=0)

    def _write_snapshot(self, rec: JobRecord) -> None:
        rec.updated_at = time.time()
        tmp = self._snapshot_path(rec.job_id) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(rec.to_dict(), f, indent=2, ensure_ascii=False)
        os.replace(tmp, self._snapshot_path(rec.job_id))

    # ---------------------------------------------------------------- registry
    def create(self, rec: JobRecord) -> None:
        with self._lock:
            self._seq += 1
            rec.seq = self._seq
            self._records[rec.job_id] = rec
            self._write_snapshot(rec)

    def get(self, job_id: str) -> JobRecord:
        with self._lock:
            rec = self._records.get(job_id)
            if rec is None:
                raise StorageError(f"unknown job {job_id}")
            return rec

    def list(self) -> list[JobRecord]:
        with self._lock:
            return sorted(self._records.values(), key=lambda r: r.seq,
                          reverse=True)

    def update(self, job_id: str, **fields: Any) -> JobRecord:
        with self._lock:
            rec = self.get(job_id)
            for k, v in fields.items():
                if k not in JobRecord.__dataclass_fields__:
                    raise StorageError(f"unknown record field {k!r}")
                setattr(rec, k, v)
            self._write_snapshot(rec)
            return rec

    def request_cancel(self, job_id: str) -> JobRecord:
        with self._lock:
            rec = self.get(job_id)
            if rec.state in ("queued", "running"):
                rec.cancel_requested = True
                self._write_snapshot(rec)
            return rec

    def is_cancel_requested(self, job_id: str) -> bool:
        with self._lock:
            return self.get(job_id).cancel_requested
