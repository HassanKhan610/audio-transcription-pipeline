from app.chunking import Chunk, choose_cut_points, merge_chunk_segments, plan_chunks


def test_short_audio_is_not_cut():
    assert choose_cut_points(duration=20, silences=[], target=30, window=5) == []


def test_cut_snaps_to_nearest_silence():
    # ideal cut at 30s, silence centred at 32s -> cut there, not mid-word at 30s
    cuts = choose_cut_points(duration=70, silences=[(31.5, 32.5)], target=30, window=5)
    assert cuts[0] == 32.0


def test_falls_back_to_hard_cut_without_silence():
    cuts = choose_cut_points(duration=100, silences=[], target=30, window=5)
    assert cuts == [30, 60, 90]


def test_silence_outside_window_is_ignored():
    cuts = choose_cut_points(duration=70, silences=[(10, 11)], target=30, window=5)
    assert cuts[0] == 30


def test_chunks_overlap_and_cover_whole_file():
    chunks = plan_chunks(duration=90, cuts=[30, 60], overlap=1.0)
    assert [(c.start, c.end) for c in chunks] == [(0, 31), (29, 61), (59, 90)]
    assert chunks[0].keep_from == 0 and chunks[-1].keep_to == 90


def test_merge_shifts_timestamps_to_file_timeline():
    chunk = Chunk(index=1, start=29, end=61, keep_from=30, keep_to=60)
    out = merge_chunk_segments(chunk, [{"start": 2.0, "end": 5.0, "text": "hi"}], is_last=False)
    assert out == [{"start": 31.0, "end": 34.0, "text": "hi"}]


def test_segment_in_overlap_is_kept_by_exactly_one_chunk():
    # The same words (file time 29.2-30.4, midpoint 29.8) show up in both chunks.
    first = Chunk(index=0, start=0, end=31, keep_from=0, keep_to=30)
    second = Chunk(index=1, start=29, end=61, keep_from=30, keep_to=60)
    seg_in_first = {"start": 29.2, "end": 30.4, "text": "boundary"}
    seg_in_second = {"start": 0.2, "end": 1.4, "text": "boundary"}

    kept = merge_chunk_segments(first, [seg_in_first], is_last=False) + \
        merge_chunk_segments(second, [seg_in_second], is_last=False)
    assert [s["text"] for s in kept] == ["boundary"]


def test_word_timestamps_are_shifted_too():
    chunk = Chunk(index=1, start=29, end=61, keep_from=30, keep_to=60)
    seg = {"start": 2.0, "end": 3.0, "text": "hi", "words": [{"start": 2.0, "end": 2.5, "word": "hi"}]}
    out = merge_chunk_segments(chunk, [seg], is_last=False)
    assert out[0]["words"][0]["start"] == 31.0


def test_word_glued_across_boundary_is_not_duplicated():
    # Real case from the sample file: chunk 1 starts 1s before the cut and
    # Whisper attaches the last word of chunk 0 ("journey.") to its first segment.
    first = Chunk(index=0, start=0, end=33, keep_from=0, keep_to=32)
    second = Chunk(index=1, start=31, end=63, keep_from=32, keep_to=62)
    seg_first = {"start": 28.0, "end": 31.6, "text": "the whole journey.", "words": [
        {"start": 28.0, "end": 29.0, "word": "the"}, {"start": 29.0, "end": 30.5, "word": "whole"},
        {"start": 30.5, "end": 31.6, "word": "journey."}]}
    seg_second = {"start": 0.0, "end": 3.0, "text": "journey. Paragraph 5", "words": [
        {"start": 0.0, "end": 0.6, "word": "journey."}, {"start": 1.2, "end": 2.0, "word": "Paragraph"},
        {"start": 2.0, "end": 3.0, "word": "5"}]}

    merged = merge_chunk_segments(first, [seg_first], is_last=False) + \
        merge_chunk_segments(second, [seg_second], is_last=True)
    text = " ".join(s["text"] for s in merged)
    assert text == "the whole journey. Paragraph 5"
    assert merged[1]["start"] == 32.2  # segment start moves to its first kept word
