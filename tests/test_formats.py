from app.formats import to_srt, to_vtt

RESULT = {"segments": [
    {"start": 0.0, "end": 2.5, "text": "Hello there."},
    {"start": 3661.25, "end": 3662.0, "text": "Late line.", "speaker": "channel_1"},
]}


def test_srt():
    srt = to_srt(RESULT)
    assert "1\n00:00:00,000 --> 00:00:02,500\nHello there." in srt
    assert "2\n01:01:01,250 --> 01:01:02,000\n[channel_1] Late line." in srt


def test_vtt():
    vtt = to_vtt(RESULT)
    assert vtt.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:02.500" in vtt
