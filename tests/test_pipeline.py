import pytest

from app.pipeline import Options, Pipeline
from tests.conftest import FakeTranscriber


def test_long_file_is_chunked_and_timestamps_are_ordered(samples, tmp_path):
    fake = FakeTranscriber()
    result = Pipeline(fake, work_root=tmp_path).run(samples / "long.wav")
    assert fake.calls >= 3  # ~100s of audio, ~30s chunks
    starts = [s["start"] for s in result["segments"]]
    assert starts == sorted(starts) and starts[-1] > 60


def test_rerun_resumes_from_checkpoints(samples, tmp_path):
    crashing = FakeTranscriber(fail_times=0)
    pipeline = Pipeline(crashing, work_root=tmp_path)
    first = pipeline.run(samples / "long.wav")
    done_calls = crashing.calls

    # Same file again: every chunk is already checkpointed, model is not called.
    pipeline.run(samples / "long.wav")
    assert crashing.calls == done_calls
    assert first["segments"]


def test_crash_mid_file_resumes_from_last_finished_chunk(samples, tmp_path):
    class CrashOnThird(FakeTranscriber):
        def transcribe(self, *a, **kw):
            if self.calls == 2 and not getattr(self, "crashed", False):
                self.crashed = True
                raise RuntimeError("worker died")
            return super().transcribe(*a, **kw)

    t = CrashOnThird()
    pipeline = Pipeline(t, work_root=tmp_path)
    with pytest.raises(RuntimeError):
        pipeline.run(samples / "long.wav")
    assert t.calls == 2  # two chunks finished and were saved

    pipeline.run(samples / "long.wav")
    # Only the remaining chunks were transcribed on the retry.
    total_chunks = len(list(tmp_path.glob("*/mono_chunk_*.json")))
    assert t.calls == total_chunks


def test_stereo_call_is_split_into_speakers(samples, tmp_path):
    result = Pipeline(FakeTranscriber(), work_root=tmp_path).run(
        samples / "call_stereo.m4a", Options(split_channels=True))
    assert {s["speaker"] for s in result["segments"]} == {"channel_0", "channel_1"}
