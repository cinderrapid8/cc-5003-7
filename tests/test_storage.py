"""Job store persistence, restart recovery and cancellation flags."""

import json
import os
import tempfile

import pytest

from convo.storage import JobRecord, JobStore, StorageError


@pytest.fixture()
def store():
    with tempfile.TemporaryDirectory() as d:
        yield JobStore(os.path.join(d, "data"))


def _rec(store, **kw):
    rid = store.new_id()
    rec = JobRecord(job_id=rid, **kw)
    store.create(rec)
    return rec


def test_create_writes_snapshot_files(store):
    rec = _rec(store, state="queued", input_name="footsteps",
               params={"params": {"wet_ratio": 0.5}})
    path = os.path.join(store.jobs_dir, f"{rec.job_id}.json")
    assert os.path.exists(path)
    with open(path, encoding="utf-8") as f:
        disk = json.load(f)
    assert disk["input_name"] == "footsteps"
    assert disk["params"]["params"]["wet_ratio"] == 0.5
    assert disk["state"] == "queued"


def test_update_is_atomic_and_persisted(store):
    rec = _rec(store)
    store.update(rec.job_id, state="running")
    store.update(rec.job_id, state="completed", summary={"output_samples": 99})
    again = JobStore(store.root)  # reload from disk
    got = again.get(rec.job_id)
    assert got.state == "completed"
    assert got.summary == {"output_samples": 99}


def test_running_job_marked_interrupted_on_restart(store):
    rec = _rec(store, state="running")
    rid = rec.job_id
    # Simulate a process restart by rebuilding the store from disk.
    reopened = JobStore(store.root)
    got = reopened.get(rid)
    assert got.state == "interrupted"
    assert "restarted" in got.error


def test_queued_job_also_recovered_on_restart(store):
    rec = _rec(store, state="queued")
    reopened = JobStore(store.root)
    assert reopened.get(rec.job_id).state == "interrupted"


def test_cancel_flag_persists_and_is_observable(store):
    rec = _rec(store, state="running")
    store.request_cancel(rec.job_id)
    assert store.is_cancel_requested(rec.job_id) is True
    reopened = JobStore(store.root)
    assert reopened.is_cancel_requested(rec.job_id) is True
    # A finished job cannot be cancelled anymore.
    store.update(rec.job_id, state="completed")
    before = store.get(rec.job_id).cancel_requested
    store.request_cancel(rec.job_id)
    assert store.get(rec.job_id).cancel_requested == before


def test_unknown_job_raises(store):
    with pytest.raises(StorageError):
        store.get("does-not-exist")


def test_listing_newest_first(store):
    r1 = _rec(store)
    r2 = _rec(store)
    ids = [r.job_id for r in store.list()]
    assert ids[0] == r2.job_id and ids[1] == r1.job_id
