import pytest

from app import audio


def test_probe_reads_real_contents(samples):
    info = audio.probe(samples / "short.mp3")
    assert info.codec == "mp3" and info.channels == 1 and 5 < info.duration < 8


def test_probe_rejects_fake_file_despite_mp3_extension(samples):
    with pytest.raises(audio.InvalidAudioError):
        audio.probe(samples / "not_audio.mp3")


@pytest.mark.parametrize("name", ["short.mp3", "long.wav", "call_stereo.m4a"])
def test_every_format_normalizes_to_16k_mono_wav(samples, tmp_path, name):
    out = audio.normalize(samples / name, tmp_path / "out.wav")
    info = audio.probe(out)
    assert (info.codec, info.channels, info.sample_rate) == ("pcm_s16le", 1, 16000)


def test_stereo_channel_can_be_extracted(samples, tmp_path):
    out = audio.normalize(samples / "call_stereo.m4a", tmp_path / "right.wav", channel=1)
    assert audio.probe(out).channels == 1


def test_detects_pauses_in_long_file(samples, tmp_path):
    wav = audio.normalize(samples / "long.wav", tmp_path / "long16k.wav")
    assert len(audio.detect_silences(wav)) >= 5
