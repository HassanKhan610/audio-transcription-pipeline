"""The transcription pipeline: probe -> normalize -> chunk -> transcribe -> merge.

Each chunk's result is written to disk as soon as it finishes. If the process
dies halfway through a long file, running the same job again skips the chunks
that are already done instead of starting from zero.
"""

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from app import audio
from app.chunking import choose_cut_points, merge_chunk_segments, plan_chunks
from app.config import settings
from app.transcriber import Transcriber

log = logging.getLogger(__name__)

ProgressFn = Callable[[int, int], None]


@dataclass(frozen=True)
class Options:
    language: str | None = None
    word_timestamps: bool = False
    split_channels: bool = False  # transcribe left/right channels separately (call recordings)

    def key(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha1(raw.encode()).hexdigest()[:8]


def _write_json_atomic(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(path)  # rename is atomic, so a crash never leaves half a checkpoint


class Pipeline:
    def __init__(self, transcriber: Transcriber, work_root: Path | None = None):
        self.transcriber = transcriber
        self.work_root = work_root or settings.data_dir / "work"

    def run(self, src: Path, options: Options = Options(), progress: ProgressFn | None = None) -> dict:
        info = audio.probe(src)
        if info.duration > settings.max_duration_s:
            raise audio.InvalidAudioError(
                f"audio is {info.duration:.0f}s, limit is {settings.max_duration_s:.0f}s"
            )

        file_hash = audio.sha256_of(src)
        work = self.work_root / f"{file_hash[:16]}-{options.key()}"
        work.mkdir(parents=True, exist_ok=True)

        if options.split_channels and info.channels == 2:
            tracks = [(0, "channel_0"), (1, "channel_1")]
        else:
            tracks = [(None, None)]

        segments: list[dict] = []
        language, language_prob = options.language, None
        for channel, speaker in tracks:
            result = self._transcribe_track(src, work, info.duration, channel, options, progress)
            language = language or result["language"]
            language_prob = language_prob or result["language_probability"]
            for seg in result["segments"]:
                if speaker:
                    seg["speaker"] = speaker
                segments.append(seg)

        segments.sort(key=lambda s: s["start"])
        if not options.word_timestamps:
            for seg in segments:
                seg.pop("words", None)
        for i, seg in enumerate(segments):
            seg["id"] = i

        return {
            "sha256": file_hash,
            "duration": round(info.duration, 3),
            "source": {"codec": info.codec, "container": info.container,
                       "channels": info.channels, "sample_rate": info.sample_rate},
            "language": language,
            "language_probability": language_prob,
            "model": getattr(self.transcriber, "model_size", "unknown"),
            "text": " ".join(s["text"] for s in segments).strip(),
            "segments": segments,
        }

    def _transcribe_track(self, src, work, duration, channel, options, progress) -> dict:
        tag = "mono" if channel is None else f"ch{channel}"
        normalized = work / f"{tag}.wav"
        if not normalized.exists():  # kept between retries, so ffmpeg only runs once
            audio.normalize(src, normalized, channel=channel)

        if duration <= settings.single_pass_max_s:
            cuts = []
        else:
            silences = audio.detect_silences(normalized)
            cuts = choose_cut_points(duration, silences, settings.chunk_target_s, settings.chunk_search_window_s)
        chunks = plan_chunks(duration, cuts, settings.chunk_overlap_s)

        language = options.language
        language_prob = None
        merged: list[dict] = []
        for chunk in chunks:
            checkpoint = work / f"{tag}_chunk_{chunk.index:04d}.json"
            if checkpoint.exists():
                result = json.loads(checkpoint.read_text())
                log.info("chunk %s/%s loaded from checkpoint", chunk.index + 1, len(chunks))
            else:
                if len(chunks) == 1:
                    chunk_path = normalized
                else:
                    chunk_path = audio.cut(normalized, work / f"{tag}_chunk_{chunk.index:04d}.wav",
                                           chunk.start, chunk.end)
                # Word timestamps are always on for chunked files: the boundary
                # de-duplication needs them. They're stripped later if not asked for.
                want_words = options.word_timestamps or len(chunks) > 1
                result = self.transcriber.transcribe(chunk_path, language, want_words)
                _write_json_atomic(checkpoint, result)
                if chunk_path != normalized:
                    chunk_path.unlink(missing_ok=True)

            # Detect the language on the first chunk, then lock it in. Otherwise a
            # quiet or noisy chunk can get detected as a different language.
            if language is None:
                language, language_prob = result["language"], result["language_probability"]
            language_prob = language_prob or result.get("language_probability")

            merged += merge_chunk_segments(chunk, result["segments"], is_last=chunk.index == len(chunks) - 1)
            if progress:
                progress(chunk.index + 1, len(chunks))

        return {"language": language, "language_probability": language_prob, "segments": merged}
