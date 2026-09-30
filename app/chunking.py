"""Splitting long audio into chunks and stitching the results back together.

Pure functions only (no ffmpeg, no model), so the tricky parts are easy to test.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    index: int
    start: float  # where ffmpeg cuts, including overlap
    end: float
    keep_from: float  # segments are kept only if their midpoint is in [keep_from, keep_to)
    keep_to: float


def choose_cut_points(
    duration: float,
    silences: list[tuple[float, float]],
    target: float,
    window: float,
) -> list[float]:
    """Pick cut points roughly every `target` seconds, snapped to silence.

    For each ideal cut we look for a silence within +/- `window` seconds and cut
    in the middle of it, so we don't split a word. If there is no silence nearby
    (music, constant talking) we fall back to the hard cut; the overlap between
    chunks covers that case.
    """
    cuts: list[float] = []
    last = 0.0
    while duration - last > target + window:
        ideal = last + target
        nearby = [
            (s + e) / 2
            for s, e in silences
            if abs((s + e) / 2 - ideal) <= window and (s + e) / 2 > last + 1.0
        ]
        cut = min(nearby, key=lambda m: abs(m - ideal)) if nearby else ideal
        cuts.append(round(cut, 3))
        last = cut
    return cuts


def plan_chunks(duration: float, cuts: list[float], overlap: float) -> list[Chunk]:
    """Turn cut points into chunks that overlap slightly at each boundary.

    Each boundary gets `overlap` seconds of extra audio on both sides so a word
    sitting on the cut is heard in full by at least one chunk. Ownership of the
    overlap is split at the boundary itself, so each segment is kept exactly once.
    """
    bounds = [0.0, *cuts, duration]
    chunks = []
    for i in range(len(bounds) - 1):
        keep_from, keep_to = bounds[i], bounds[i + 1]
        chunks.append(
            Chunk(
                index=i,
                start=max(0.0, keep_from - overlap),
                end=min(duration, keep_to + overlap),
                keep_from=keep_from,
                keep_to=keep_to,
            )
        )
    return chunks


def merge_chunk_segments(chunk: Chunk, segments: list[dict], is_last: bool) -> list[dict]:
    """Shift a chunk's segments to the full-file timeline and drop whatever
    belongs to the neighbouring chunk.

    When word timestamps are available the decision is made per word: Whisper
    often glues the tail of the previous sentence onto the start of a segment
    ("journey. Paragraph 5, ..."), so dropping whole segments would either lose
    or duplicate words at the boundary. Without words we fall back to the
    segment midpoint.
    """

    def owned(start: float, end: float) -> bool:
        mid = (start + end) / 2
        return chunk.keep_from <= mid and (mid < chunk.keep_to or is_last)

    offset = chunk.start
    kept = []
    for seg in segments:
        if seg.get("words"):
            words = [
                {**w, "start": round(w["start"] + offset, 3), "end": round(w["end"] + offset, 3)}
                for w in seg["words"]
            ]
            words = [w for w in words if owned(w["start"], w["end"])]
            if not words:
                continue
            kept.append({
                **seg,
                "start": words[0]["start"],
                "end": words[-1]["end"],
                "text": " ".join(w["word"] for w in words),
                "words": words,
            })
        else:
            start, end = round(seg["start"] + offset, 3), round(seg["end"] + offset, 3)
            if owned(start, end):
                kept.append({**seg, "start": start, "end": end})
    return kept
