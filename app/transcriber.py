"""Thin wrapper around faster-whisper.

The model is loaded once per process and reused; loading it per request would
cost several seconds and a lot of memory each time.
"""

from functools import lru_cache
from pathlib import Path
from typing import Protocol

from app.config import settings


class Transcriber(Protocol):
    def transcribe(self, path: Path, language: str | None, word_timestamps: bool) -> dict: ...


class WhisperTranscriber:
    def __init__(self, model_size: str, device: str, compute_type: str):
        from faster_whisper import WhisperModel

        self.model_size = model_size
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, path: Path, language: str | None = None, word_timestamps: bool = False) -> dict:
        segments, info = self.model.transcribe(
            str(path),
            language=language,
            # Whisper tends to invent text ("Thank you for watching") on silence;
            # the built-in Silero VAD removes silent parts before decoding.
            vad_filter=True,
            # Stops the model from getting stuck repeating a line on long audio.
            condition_on_previous_text=False,
            word_timestamps=word_timestamps,
            beam_size=5,
        )
        out = []
        for s in segments:  # generator: decoding happens while we iterate
            seg = {
                "start": round(s.start, 3),
                "end": round(s.end, 3),
                "text": s.text.strip(),
                "avg_logprob": round(s.avg_logprob, 3),
                "no_speech_prob": round(s.no_speech_prob, 3),
            }
            if word_timestamps and s.words:
                seg["words"] = [
                    {"start": round(w.start, 3), "end": round(w.end, 3), "word": w.word.strip(),
                     "probability": round(w.probability, 3)}
                    for w in s.words
                ]
            out.append(seg)
        return {
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
            "segments": out,
        }


@lru_cache(maxsize=1)
def get_transcriber() -> Transcriber:
    return WhisperTranscriber(settings.model_size, settings.device, settings.compute_type)
