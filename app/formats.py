"""Export a transcript as SRT or WebVTT subtitles."""


def _ts(seconds: float, sep: str) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _label(seg: dict) -> str:
    return f"[{seg['speaker']}] {seg['text']}" if seg.get("speaker") else seg["text"]


def to_srt(result: dict) -> str:
    blocks = []
    for i, seg in enumerate(result["segments"], start=1):
        blocks.append(f"{i}\n{_ts(seg['start'], ',')} --> {_ts(seg['end'], ',')}\n{_label(seg)}\n")
    return "\n".join(blocks)


def to_vtt(result: dict) -> str:
    lines = ["WEBVTT", ""]
    for seg in result["segments"]:
        lines += [f"{_ts(seg['start'], '.')} --> {_ts(seg['end'], '.')}", _label(seg), ""]
    return "\n".join(lines)
