from pathlib import Path

import pytest

SAMPLES = Path(__file__).parent.parent / "samples"


class FakeTranscriber:
    """Stands in for Whisper: returns one segment per call, so tests run in
    milliseconds and don't need to download a model."""

    model_size = "fake"

    def __init__(self, fail_times: int = 0):
        self.calls = 0
        self.fail_times = fail_times

    def transcribe(self, path, language=None, word_timestamps=False):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("simulated worker crash")
        return {"language": language or "en", "language_probability": 0.99,
                "segments": [{"start": 0.5, "end": 1.5, "text": f"call {self.calls}",
                              "avg_logprob": -0.2, "no_speech_prob": 0.01}]}


@pytest.fixture
def samples():
    return SAMPLES


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path):
    """Each test gets its own job DB and upload folder."""
    from app.config import settings

    original = settings.data_dir
    object.__setattr__(settings, "data_dir", tmp_path / "data")  # settings is frozen
    yield settings.data_dir
    object.__setattr__(settings, "data_dir", original)
