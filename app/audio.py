"""Audio inspection and normalization, built on ffprobe/ffmpeg.

Every input is converted to 16 kHz mono 16-bit PCM WAV before it reaches the
model, so the rest of the pipeline only ever deals with one format.
"""

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

TARGET_SAMPLE_RATE = 16_000


class InvalidAudioError(Exception):
    """File is unreadable, has no audio stream, or is otherwise unusable.

    This is a permanent failure: retrying the same file will not help.
    """


@dataclass
class AudioInfo:
    duration: float
    channels: int
    sample_rate: int
    codec: str
    container: str


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def probe(path: Path) -> AudioInfo:
    """Inspect the real contents of the file instead of trusting its extension."""
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise InvalidAudioError(f"ffprobe could not read the file: {result.stderr.strip()[:200]}")

    data = json.loads(result.stdout or "{}")
    audio_streams = [s for s in data.get("streams", []) if s.get("codec_type") == "audio"]
    if not audio_streams:
        raise InvalidAudioError("file has no audio stream")

    stream = audio_streams[0]
    fmt = data.get("format", {})
    duration = float(stream.get("duration") or fmt.get("duration") or 0)
    if duration <= 0:
        raise InvalidAudioError("audio stream is empty")

    return AudioInfo(
        duration=duration,
        channels=int(stream.get("channels", 1)),
        sample_rate=int(stream.get("sample_rate", 0)),
        codec=stream.get("codec_name", "unknown"),
        container=fmt.get("format_name", "unknown"),
    )


def normalize(src: Path, dst: Path, channel: int | None = None) -> Path:
    """Convert any input to 16 kHz mono WAV.

    If `channel` is given, only that channel is kept (used for stereo call
    recordings where each side of the conversation is on its own channel).
    Otherwise all channels are downmixed to mono.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(src), "-vn"]
    if channel is not None:
        cmd += ["-af", f"pan=mono|c0=c{channel}"]
    else:
        cmd += ["-ac", "1"]
    cmd += ["-ar", str(TARGET_SAMPLE_RATE), "-c:a", "pcm_s16le", str(dst)]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise InvalidAudioError(f"ffmpeg conversion failed: {result.stderr.strip()[:200]}")
    return dst


_SILENCE_START = re.compile(r"silence_start: (-?[\d.]+)")
_SILENCE_END = re.compile(r"silence_end: (-?[\d.]+)")


def detect_silences(path: Path, noise_db: int = -35, min_silence_s: float = 0.4) -> list[tuple[float, float]]:
    """Return (start, end) of silent stretches, used to pick safe cut points."""
    cmd = [
        "ffmpeg", "-nostdin", "-v", "info", "-i", str(path),
        "-af", f"silencedetect=noise={noise_db}dB:d={min_silence_s}",
        "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    starts = [max(0.0, float(x)) for x in _SILENCE_START.findall(result.stderr)]
    ends = [float(x) for x in _SILENCE_END.findall(result.stderr)]
    return list(zip(starts, ends))


def cut(src: Path, dst: Path, start: float, end: float) -> Path:
    """Extract [start, end) from an already-normalized WAV."""
    cmd = [
        "ffmpeg", "-nostdin", "-y", "-v", "error",
        "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(src),
        "-c", "copy", str(dst),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return dst
