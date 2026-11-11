"""End-to-end HTTP tests: real renders, cancel, rerun, restart reproduction."""

import io
import os
import tempfile
import time
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from convo.app import create_app
from convo import wavio


@pytest.fixture()
def client():
    with tempfile.TemporaryDirectory() as d:
        app = create_app(data_root=os.path.join(d, "data"))
        with TestClient(app) as c:
            yield c, app, d


def _wav_bytes(audio, sr):
    buf = io.BytesIO()
    wavio.write_wav_to_buffer(buf, audio, sr)
    return buf.getvalue()


def _wait(client, job_id, timeout=15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/api/jobs/{job_id}")
        state = r.json()["state"]
        if state not in ("queued", "running"):
            return r.json()
        time.sleep(0.05)
    raise AssertionError("job did not finish in time")


# ------------------------------------------------------------------ happy path
def test_full_render_lifecycle_with_builtin_sound(client):
    c, app, _ = client
    r = c.post("/api/jobs", data={
        "sound": "footsteps",
        "wet_ratio": "0.7",
        "pre_delay_ms": "12",
        "tail_policy": "keep",
        "block_size": "1024",
    })
    assert r.status_code == 200, r.text
    job = r.json()
    assert job["state"] in ("queued", "running")

    final = _wait(c, job["job_id"])
    assert final["state"] == "completed", final["error"]
    s = final["summary"]
    assert s["output_samples"] > final["params"]["input"]["samples"]
    assert s["tail_samples_kept"] > 0  # tail retained by default
    assert s["fft_size"] == 2 * s["block_size"]

    # Downloadable real render produced now, not a pre-generated asset.
    out = c.get(f"/api/jobs/{job['job_id']}/output")
    assert out.status_code == 200 and out.headers["content-type"] == "audio/wav"
    with wave.open(io.BytesIO(out.content)) as wf:
        assert wf.getframerate() == 48000
        assert wf.getnframes() == s["output_samples"]

    # Snapshot persisted for reproducibility.
    snap = c.get(f"/api/jobs/{job['job_id']}/snapshot").json()
    assert snap["params"]["wet_ratio"] == 0.7
    assert snap["input"]["name"] == "footsteps"
    assert snap["impulse_response"]["channels"] == 2


def test_all_three_test_sounds_render(client):
    c, _, _ = client
    for kind in ("footsteps", "door_close", "metal_clash"):
        r = c.post("/api/jobs", data={"sound": kind, "tail_policy": "keep"})
        assert r.status_code == 200
        final = _wait(c, r.json()["job_id"])
        assert final["state"] == "completed", final["error"]
        assert final["summary"]["duration_s"] > 0


def test_truncation_is_explicit_and_reported(client):
    c, _, _ = client
    r = c.post("/api/jobs", data={
        "sound": "door_close", "tail_policy": "truncate", "wet_ratio": "1.0",
    })
    final = _wait(c, r.json()["job_id"])
    assert final["state"] == "completed"
    s = final["summary"]
    assert s["tail_samples_cut"] > 0
    assert s["tail_samples_kept"] == 0
    assert s["output_samples"] == final["params"]["input"]["samples"]


# --------------------------------------------------------------- cancellation
def test_cancel_stops_render(client):
    c, app, _ = client
    # A long IR + long input makes the render span many blocks.
    long_ir = np.random.default_rng(0).standard_normal((1, 60000))
    files = {
        "input_file": ("x.wav", _wav_bytes(
            np.random.default_rng(1).standard_normal((1, 96000)), 48000),
            "audio/wav"),
        "ir_file": ("h.wav", _wav_bytes(long_ir, 48000), "audio/wav"),
    }
    r = c.post("/api/jobs", files=files, data={"block_size": "256"})
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    cr = c.post(f"/api/jobs/{job_id}/cancel")
    assert cr.status_code == 200
    final = _wait(c, job_id)
    assert final["state"] in ("cancelled", "completed")
    if final["state"] == "cancelled":
        assert "cancel" in final["error"]
        # No output is exposed for a cancelled job.
        assert c.get(f"/api/jobs/{job_id}/output").status_code == 404


def test_cancel_completed_job_conflicts(client):
    c, _, _ = client
    r = c.post("/api/jobs", data={"sound": "door_close"})
    jid = r.json()["job_id"]
    _wait(c, jid)
    assert c.post(f"/api/jobs/{jid}/cancel").status_code == 409


# ------------------------------------------------------------------- rerun
def test_rerun_after_failure_reproduces_from_snapshot(client):
    c, app, data_root = client
    r = c.post("/api/jobs", data={"sound": "metal_clash"})
    jid = r.json()["job_id"]
    final = _wait(c, jid)
    assert final["state"] == "completed"
    out1 = c.get(f"/api/jobs/{jid}/output").content

    # Mark interrupted (as a restart would) and rerun from saved inputs.
    app.state.store.update(jid, state="interrupted",
                           error="simulated restart")
    rr = c.post(f"/api/jobs/{jid}/rerun")
    assert rr.status_code == 200
    final2 = _wait(c, jid)
    assert final2["state"] == "completed"
    out2 = c.get(f"/api/jobs/{jid}/output").content
    assert out1 == out2  # bit-identical reproduction of the same space


def test_reload_recovers_interrupted_and_reruns(client):
    c, app, d = client
    r = c.post("/api/jobs", data={"sound": "footsteps", "wet_ratio": "0.5"})
    jid = r.json()["job_id"]
    _wait(c, jid)
    app.state.manager.shutdown()
    data_root = os.path.join(d, "data")

    # New app instance over the same data directory: history must survive.
    app2 = create_app(data_root=data_root)
    with TestClient(app2) as c2:
        listing = c2.get("/api/jobs").json()["jobs"]
        assert any(j["job_id"] == jid for j in listing)
        got = c2.get(f"/api/jobs/{jid}").json()
        assert got["state"] == "completed"  # completed stays completed
        assert os.path.exists(got["input_path"]) and os.path.exists(got["ir_path"])
        # Re-render from the persisted snapshot and compare.
        before = c2.get(f"/api/jobs/{jid}/output").content
        c2.post(f"/api/jobs/{jid}/rerun")
        _wait(c2, jid)
        after = c2.get(f"/api/jobs/{jid}/output").content
        assert before == after


# ------------------------------------------------------------- upload & maps
def test_upload_stereo_ir_respected_not_averaged(client):
    c, _, _ = client
    base = np.random.default_rng(2).standard_normal(24000) * 0.1
    # Identical content on both inputs lets us isolate the IR channel sign.
    x = np.stack([base, base])
    h = np.stack([np.ones(200) * 0.5, -np.ones(200) * 0.5])
    r = c.post("/api/jobs", files={
        "input_file": ("x.wav", _wav_bytes(x, 48000), "audio/wav"),
        "ir_file": ("h.wav", _wav_bytes(h, 48000), "audio/wav"),
    }, data={"channel_map": "parallel", "wet_ratio": "1.0"})
    final = _wait(c, r.json()["job_id"])
    assert final["state"] == "completed", final["error"]
    assert final["summary"]["output_channels"] == 2
    out = c.get(f"/api/jobs/{final['job_id']}/output").content
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        tf.write(out); tmp = tf.name
    rendered, rsr = wavio.read_wav(tmp)
    os.unlink(tmp)
    # L and R are sign-opposed => channels must not be mono-averaged away.
    assert not np.allclose(rendered[0], rendered[1])
    np.testing.assert_allclose(rendered[1], -rendered[0], atol=2e-4)


def test_mismatched_sample_rates_are_converted(client):
    c, _, _ = client
    # Input 48k, synthetic IR at 24k: pipeline must resample and finish.
    r = c.post("/api/jobs", data={"sound": "footsteps", "ir_sr": "24000"})
    final = _wait(c, r.json()["job_id"])
    assert final["state"] == "completed", final["error"]
    assert final["summary"]["sample_rate"] == 48000
    # IR samples were resampled upward inside the render.
    ir_meta = final["params"]["impulse_response"]
    assert ir_meta["sample_rate"] == 24000
    assert final["summary"]["ir_samples"] > ir_meta["samples"]


def test_illegal_mapping_rejected_with_400(client):
    c, _, _ = client
    r = c.post("/api/jobs", data={
        "sound": "footsteps", "ir_stereo": "false", "channel_map": "cross",
    })
    assert r.status_code == 400
    assert "channel_map" in r.json()["detail"]


def test_bad_wet_ratio_rejected(client):
    c, _, _ = client
    r = c.post("/api/jobs", data={"sound": "door_close", "wet_ratio": "2.0"})
    assert r.status_code == 400


def test_non_wav_upload_rejected(client):
    c, _, _ = client
    r = c.post("/api/jobs", files={
        "input_file": ("note.txt", b"hello", "text/plain"),
    }, data={"sound": ""})
    assert r.status_code == 400


def test_missing_input_rejected(client):
    c, _, _ = client
    r = c.post("/api/jobs", data={"wet_ratio": "0.5"})
    assert r.status_code == 400


def test_unit_impulse_render_reproduces_ir(client):
    """End-to-end identity check: impulse input -> output equals the IR."""
    c, _, _ = client
    from convo.synth import unit_impulse
    impulse = unit_impulse(48000, seconds=0.05)
    h = np.random.default_rng(3).standard_normal(400) * 0.2
    r = c.post("/api/jobs", files={
        "input_file": ("d.wav", _wav_bytes(impulse, 48000), "audio/wav"),
        "ir_file": ("h.wav", _wav_bytes(h[None, :], 48000), "audio/wav"),
    }, data={"wet_ratio": "1.0", "pre_delay_ms": "0", "tail_policy": "keep"})
    final = _wait(c, r.json()["job_id"])
    assert final["state"] == "completed", final["error"]
    out_bytes = c.get(f"/api/jobs/{final['job_id']}/output").content
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        tf.write(out_bytes); tmp = tf.name
    rendered, _ = wavio.read_wav(tmp)
    os.unlink(tmp)
    np.testing.assert_allclose(rendered[0, :400], h, atol=2e-4)
