"""HTTP API.

Transcription is slow, so the API is job-based: POST returns 202 with a job id
straight away and the work happens in the background. Short clips can ask for
`wait=true` to get the transcript in the same response.

In this demo the "queue" is an in-process thread pool with one worker. The
README describes the production version (Redis queue + separate workers).
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.concurrency import run_in_threadpool

from app import audio
from app.config import settings
from app.formats import to_srt, to_vtt
from app.pipeline import Options, Pipeline
from app.store import JobStore
from app.transcriber import SUPPORTED_LANGUAGES, get_transcriber

log = logging.getLogger(__name__)
MAX_ATTEMPTS = 3


class Service:
    def __init__(self, pipeline_factory=lambda: Pipeline(get_transcriber())):
        self.pipeline_factory = pipeline_factory
        self._pipeline: Pipeline | None = None
        self.store = JobStore(settings.data_dir / "jobs.db")
        self.upload_dir = settings.data_dir / "uploads"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        # One worker: the model already uses all CPU cores, running two jobs at
        # once on the same machine would only make both slower.
        self.executor = ThreadPoolExecutor(max_workers=1)

    @property
    def pipeline(self) -> Pipeline:
        if self._pipeline is None:  # model loads on first use, not at import
            self._pipeline = self.pipeline_factory()
        return self._pipeline

    def process(self, job_id: str, options: Options) -> None:
        job = self.store.get(job_id)
        src = self.upload_dir / job["sha256"]
        for attempt in range(job["attempts"] + 1, MAX_ATTEMPTS + 1):
            self.store.update(job_id, status="processing", attempts=attempt, error=None)
            try:
                result = self.pipeline.run(
                    src, options,
                    progress=lambda done, total: self.store.update(job_id, progress=f"{done}/{total}"),
                )
                self.store.update(job_id, status="completed", result=result)
                return
            except (audio.InvalidAudioError, ValueError) as e:
                # Permanent: bad file or bad input. Retrying the same thing won't help.
                self.store.update(job_id, status="failed", error=str(e))
                return
            except Exception as e:  # transient: crash, OOM, disk hiccup
                log.exception("job %s attempt %s failed", job_id, attempt)
                self.store.update(job_id, error=f"attempt {attempt}: {e}")
        # Finished chunks are checkpointed, so each retry above resumed where the
        # previous one stopped. After MAX_ATTEMPTS the job is parked as failed.
        self.store.update(job_id, status="failed")

    def resume_unfinished(self) -> None:
        """On startup, re-queue jobs that were running when the server stopped."""
        for job in self.store.unfinished():
            opts = Options(**json.loads(job["options_json"]))
            log.info("resuming job %s", job["id"])
            self.executor.submit(self.process, job["id"], opts)


def create_app(service: Service | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = service or Service()
        app.state.service.resume_unfinished()
        yield
        app.state.service.executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="Transcription Pipeline", version="1.0.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/v1/transcriptions", status_code=202)
    async def create_transcription(
        file: UploadFile = File(...),
        language: str | None = Form(None, description="ISO code, e.g. 'en'. Auto-detected if empty."),
        word_timestamps: bool = Form(False),
        split_channels: bool = Form(False, description="Stereo calls: transcribe each channel as its own speaker."),
        wait: bool = Form(False, description=f"Return the transcript inline (only for clips up to {settings.single_pass_max_s:.0f}s)."),
    ):
        svc: Service = app.state.service
        language = (language or "").strip().lower() or None
        if language and language not in SUPPORTED_LANGUAGES:
            raise HTTPException(400, f"unsupported language '{language}', use an ISO code like 'en' or leave empty")
        options = Options(language=language, word_timestamps=word_timestamps, split_channels=split_channels)

        tmp = svc.upload_dir / f"incoming-{id(file)}"
        size = await run_in_threadpool(_save_limited, file, tmp, settings.max_upload_mb * 1024 * 1024)
        if size is None:
            tmp.unlink(missing_ok=True)
            raise HTTPException(413, f"file is larger than {settings.max_upload_mb} MB")

        # Reject bad files before they take a slot in the queue.
        try:
            info = await run_in_threadpool(audio.probe, tmp)
        except audio.InvalidAudioError as e:
            tmp.unlink(missing_ok=True)
            raise HTTPException(400, str(e))
        if info.duration > settings.max_duration_s:
            tmp.unlink(missing_ok=True)
            raise HTTPException(400, f"audio longer than {settings.max_duration_s:.0f}s")

        sha = await run_in_threadpool(audio.sha256_of, tmp)
        existing = svc.store.find_reusable(sha, options.key())
        if existing:
            tmp.unlink(missing_ok=True)
            return _job_response(existing, status_code=200 if existing["status"] == "completed" else 202)

        tmp.replace(svc.upload_dir / sha)
        job = svc.store.create(sha, options.key(), json.dumps(asdict(options)), file.filename or "upload")

        if wait and info.duration <= settings.single_pass_max_s:
            await run_in_threadpool(svc.process, job["id"], options)
            return _job_response(svc.store.get(job["id"]), status_code=200)

        svc.executor.submit(svc.process, job["id"], options)
        return _job_response(svc.store.get(job["id"]), status_code=202)

    @app.get("/v1/transcriptions/{job_id}")
    def get_transcription(job_id: str, format: str = Query("json", pattern="^(json|srt|vtt)$")):
        job = app.state.service.store.get(job_id)
        if not job:
            raise HTTPException(404, "job not found")
        if format != "json":
            if job["status"] != "completed":
                raise HTTPException(409, f"job is {job['status']}")
            body = to_srt(job["result"]) if format == "srt" else to_vtt(job["result"])
            return PlainTextResponse(body, media_type="text/vtt" if format == "vtt" else "application/x-subrip")
        return _job_response(job)

    return app


def _save_limited(upload: UploadFile, dst: Path, limit: int) -> int | None:
    """Stream the upload to disk in blocks; stop early if it goes over the limit."""
    written = 0
    with open(dst, "wb") as out:
        while block := upload.file.read(1 << 20):
            written += len(block)
            if written > limit:
                return None
            out.write(block)
    return written


def _job_response(job: dict, status_code: int = 200) -> JSONResponse:
    body = {k: job[k] for k in ("id", "status", "attempts", "progress", "error", "filename", "created_at", "updated_at")}
    if job["status"] == "completed":
        body["result"] = job["result"]
    return JSONResponse(body, status_code=status_code)


app = create_app()
