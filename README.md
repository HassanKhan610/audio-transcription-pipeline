# Transcription Pipeline

A small service that takes an audio file (WAV, MP3, M4A, OGG, even video), transcribes it with an open-source Whisper model, and returns the text with timestamps per segment. It works as a CLI or as an HTTP API.

```
upload ─▶ ffprobe check ─▶ ffmpeg → 16 kHz mono WAV ─▶ split at silences (long files)
       ─▶ faster-whisper per chunk (checkpointed) ─▶ merge on word timestamps ─▶ JSON / SRT / VTT
```

## Quick start

Needs Python 3.11+ and ffmpeg (`brew install ffmpeg` / `apt install ffmpeg`).

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# CLI
python transcribe.py samples/short.mp3
python transcribe.py samples/call_stereo.m4a --split-channels --format srt

# API (docs at http://localhost:8000/docs)
uvicorn app.api:app --reload
curl -F file=@samples/short.mp3 -F wait=true localhost:8000/v1/transcriptions

# Tests (use a fake model, run in ~5s)
pytest
```

Or with Docker: `docker build -t transcriber . && docker run -p 8000:8000 -v $PWD/data:/data transcriber`

The first real run downloads the Whisper model (~500 MB for `small`). Pick a different size with `WHISPER_MODEL=tiny|base|small|medium|large-v3`.

## Example output

```json
{
  "duration": 6.08,
  "language": "en",
  "language_probability": 0.99,
  "model": "small",
  "text": "Hello, thanks for calling, I'd like to book a viewing for the two bedroom apartment on Saturday morning.",
  "segments": [
    {"id": 0, "start": 0.0,  "end": 4.98, "text": "Hello, thanks for calling, I'd like to book a viewing for the two bedroom apartment on", "avg_logprob": -0.237, "no_speech_prob": 0.005},
    {"id": 1, "start": 4.98, "end": 6.06, "text": "Saturday morning.", "avg_logprob": -0.237, "no_speech_prob": 0.005}
  ]
}
```

## Project layout

| File | What it does |
|---|---|
| `app/audio.py` | ffprobe inspection, ffmpeg normalization, silence detection, cutting |
| `app/chunking.py` | choosing cut points, planning overlapping chunks, merging results (pure functions) |
| `app/transcriber.py` | faster-whisper wrapper, model loaded once per process |
| `app/pipeline.py` | ties it together, with per-chunk checkpoints |
| `app/api.py` | FastAPI job API, retries, dedupe |
| `app/store.py` | job table (SQLite here, Postgres in production) |
| `app/formats.py` | SRT / WebVTT export |
| `transcribe.py` | CLI |
| `samples/` | test audio: short MP3, 100s WAV with pauses, stereo "call", and a fake `.mp3` |

---

## Design decisions

### Model: faster-whisper, running locally
- Open-source, no per-minute API cost, and audio never leaves the machine (voice is personal data).
- The CTranslate2 build is several times faster than the reference Whisper with the same accuracy, and `int8` makes it usable on a plain CPU.
- The model loads **once per process** and is reused. Loading it per request would cost seconds and hundreds of MB each time.

Two settings mattered more than the model size:
- **`vad_filter=True`:** Whisper is known to invent text on silence or noise ("Thank you for watching!"). The built-in Silero VAD removes non-speech before decoding.
- **`condition_on_previous_text=False`:** by default each window is conditioned on the previous text, and on long audio a single mistake can turn into the same line repeated over and over. Turning it off trades a bit of cross-sentence consistency for robustness.

### Different audio formats
- **Don't trust the extension.** Every upload goes through `ffprobe` first. A text file renamed to `.mp3`, or a video with no audio track, is rejected with a clear `400` before it takes a place in the queue (see `samples/not_audio.mp3`).
- **Normalize everything to one format.** ffmpeg converts any input to 16 kHz, mono, 16-bit PCM WAV, which is what Whisper uses internally. The rest of the pipeline only deals with one format, and supporting a new container or codec needs no code change.
- **Stereo call recordings** (`split_channels=true`): phone systems often record the caller and the agent on separate channels. Instead of downmixing, each channel is transcribed separately and tagged `channel_0` / `channel_1`. That gives "who said what" for free, with no diarization model.

### Long audio files
Three problems: memory, request timeouts, and words cut in half at chunk edges.

1. **Split at silences, not every N seconds.** `ffmpeg silencedetect` finds pauses. Around every 30s the cut snaps to the nearest pause within ±5s, and falls back to a hard cut if there isn't one (music, non-stop talking).
2. **Overlap + word-level merge.** Each chunk gets 1s of extra audio on both sides so no word is heard only in half. Every word then belongs to exactly one chunk, decided by its midpoint. This is done **per word, not per segment**: in testing, Whisper attached the last word of one chunk ("journey.") to the first segment of the next, and segment-level dedupe let it through twice. Word timestamps are always enabled internally for chunked files for this reason (`test_word_glued_across_boundary_is_not_duplicated`).
3. **Lock the language after the first chunk.** Auto-detection runs once; later chunks reuse it, so a quiet or noisy chunk can't be decoded as a different language.
4. **Checkpoint every chunk.** Each chunk's result is written to disk (atomically, via rename) as soon as it's done. If the worker dies at minute 50 of a 60-minute file, the retry resumes from chunk 51 instead of starting over. The normalized WAV is kept too, so ffmpeg doesn't run again.
5. **Background jobs.** Long files never hold an HTTP request open (see API below).

### Output for downstream use
- Each segment keeps `avg_logprob` and `no_speech_prob`. Downstream systems (search, an LLM, human QA) can flag low-confidence parts instead of trusting everything the same.
- Optional word-level timestamps (`word_timestamps=true`) for subtitles or click-to-seek.
- JSON by default, plus `?format=srt` / `?format=vtt`.

---

## API

| Method | Path | |
|---|---|---|
| `POST` | `/v1/transcriptions` | multipart `file` + optional `language`, `word_timestamps`, `split_channels`, `wait`. Returns **202** with a job id. |
| `GET` | `/v1/transcriptions/{id}` | status (`queued` → `processing` → `completed` / `failed`), progress like `"2/4"` chunks, and the result |
| `GET` | `/v1/transcriptions/{id}?format=srt\|vtt` | subtitles |
| `GET` | `/health` | liveness |

- **Async by default.** Transcription takes seconds to minutes, so POST returns right away and the client polls. `wait=true` returns the transcript inline, only for clips ≤ 45s.
- **Dedupe.** Files are identified by SHA-256. Uploading the same file with the same options returns the existing job instead of transcribing again.
- **Upload limits.** The upload is streamed to disk in 1 MB blocks and stopped as soon as it passes `MAX_UPLOAD_MB`, so a huge file can't fill memory. Duration is capped by `MAX_DURATION_S`.

### Failures and retries (implemented)
- **Permanent errors** (corrupt file, no audio stream, too long): fail immediately with the reason. Retrying can't fix the file.
- **Transient errors** (crash, out of memory, disk error): retried up to 3 times. Thanks to the checkpoints, each retry continues from the last finished chunk.
- **Server restart:** on startup, jobs left `queued` or `processing` are re-queued automatically.

---

## Production design (not implemented, by design of the task)

What I'd change to run this for real traffic:

**Concurrent uploads**
- Clients upload straight to object storage with a **pre-signed S3 URL**. Large files never pass through the API servers; the API only creates a job.
- Jobs go on a **Redis queue** (Celery or ARQ). A pool of **GPU/CPU workers** pulls from it, one job per worker at a time, with the model loaded at worker start. The in-process single-thread executor here plays that role.
- Workers **autoscale on queue depth**, not API traffic. A burst of 500 uploads makes the queue longer; nothing falls over.
- **Per-tenant rate limits and concurrency caps**, plus a separate queue for short clips so they aren't stuck behind a 3-hour recording.
- For very long files, chunks can be fanned out to several workers in parallel, since each chunk is already independent and checkpointed.

**Storage**
- **Audio → S3**, keyed by `tenant/{id}/audio/{sha256}`, encrypted with KMS, with lifecycle rules for retention (e.g. move to cheaper storage after 30 days, delete after the agreed period).
- **Jobs and transcripts → PostgreSQL**:
  - a `jobs` table (same columns as `app/store.py`)
  - a `segments` table `(job_id, idx, start, end, text, speaker, avg_logprob)`: one row per segment rather than one big blob, so you can search inside transcripts and jump to a timestamp
  - full-text search via `tsvector`; `pgvector` embeddings if transcripts feed RAG / LLM features
  - row-level security per tenant
- Store the **model name and version** with every transcript, so after a model upgrade we know which transcripts to re-run.

**Retries and recovery**
- Everything implemented above, plus:
  - **worker heartbeats:** a job stuck in `processing` with no heartbeat for a few minutes is re-queued (the worker died)
  - **exponential backoff** between attempts
  - a **dead-letter queue** for jobs that fail every attempt, with an alert and a manual "re-run" action after a fix
  - for out-of-memory failures, a retry can drop to a smaller model or shorter chunks

**API**
- API keys per tenant.
- An `Idempotency-Key` header so a client retrying a POST doesn't create two jobs.
- **Signed webhooks** (HMAC) when a job finishes, so clients don't have to poll.
- Accept `audio_url` as an alternative to upload.
- Versioned under `/v1`, with the OpenAPI docs FastAPI already generates.

**Observability:** structured logs per job and stage (probe / normalize / transcribe / merge), metrics for queue depth, real-time factor (processing time ÷ audio length) and failure rate by error type.

## Known limitations
- The single in-process worker means one transcription at a time per server (intentional: Whisper already uses all CPU cores).
- The `small` model trades some accuracy for speed. In the sample call it heard "tap" as "tab". `medium` or `large-v3` on a GPU fixes most of these.
- No speaker diarization for mono recordings (only channel-based speakers).

## Development note
Built with AI-assisted development (used for scaffolding and test writing). Design decisions, testing on real audio, and final review are my own.
