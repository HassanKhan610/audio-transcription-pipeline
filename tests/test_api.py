import time

import pytest
from fastapi.testclient import TestClient

from app.api import Service, create_app
from app.pipeline import Pipeline
from tests.conftest import FakeTranscriber


@pytest.fixture
def client():
    service = Service(pipeline_factory=lambda: Pipeline(FakeTranscriber()))
    with TestClient(create_app(service)) as c:
        yield c


def _upload(client, path, **form):
    with open(path, "rb") as f:
        return client.post("/v1/transcriptions", files={"file": (path.name, f)}, data=form)


def _wait_done(client, job_id):
    for _ in range(100):
        job = client.get(f"/v1/transcriptions/{job_id}").json()
        if job["status"] in ("completed", "failed"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_async_job_flow(client, samples):
    r = _upload(client, samples / "long.wav")
    assert r.status_code == 202 and r.json()["status"] in ("queued", "processing")
    job = _wait_done(client, r.json()["id"])
    assert job["status"] == "completed"
    assert job["result"]["segments"][0].keys() >= {"start", "end", "text"}


def test_short_clip_can_wait_inline(client, samples):
    r = _upload(client, samples / "short.mp3", wait="true")
    assert r.status_code == 200 and r.json()["status"] == "completed"


def test_invalid_file_rejected_before_queueing(client, samples):
    r = _upload(client, samples / "not_audio.mp3")
    assert r.status_code == 400


def test_same_file_twice_reuses_the_job(client, samples):
    first = _upload(client, samples / "short.mp3", wait="true").json()
    second = _upload(client, samples / "short.mp3")
    assert second.json()["id"] == first["id"] and second.status_code == 200


def test_subtitle_export(client, samples):
    job = _upload(client, samples / "short.mp3", wait="true").json()
    srt = client.get(f"/v1/transcriptions/{job['id']}?format=srt")
    assert srt.status_code == 200 and "-->" in srt.text


def test_transient_failure_is_retried(samples):
    service = Service(pipeline_factory=lambda: Pipeline(FakeTranscriber(fail_times=1)))
    with TestClient(create_app(service)) as c:
        job = _upload(c, samples / "short.mp3", wait="true").json()
    assert job["status"] == "completed" and job["attempts"] == 2


def test_unknown_job_is_404(client):
    assert client.get("/v1/transcriptions/nope").status_code == 404
