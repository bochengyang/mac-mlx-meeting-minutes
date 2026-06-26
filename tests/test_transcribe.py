"""Tests for transcription (fake audio loader + fake whisper, no model/ffmpeg)."""

from pathlib import Path

import numpy as np
import pytest

from meeting_minutes.transcribe import (
    make_text_filter,
    transcribe_session,
    transcribe_wav,
)
from meeting_minutes.transcript import Segment


def test_transcribe_wav_maps_segments_with_speaker():
    """Load audio (ffmpeg-free), pass the waveform to whisper, map to Segment."""
    fake_audio = np.zeros(4, dtype=np.float32)
    fake_result = {
        "segments": [
            {"start": 0.0, "end": 1.2, "text": " hello"},
            {"start": 1.5, "end": 2.0, "text": "morning "},
        ]
    }
    captured = {}

    def fake_load_audio(path):
        captured["path"] = path
        return fake_audio

    def fake_transcribe(audio, **kwargs):
        captured["audio"] = audio
        captured["kwargs"] = kwargs
        return fake_result

    segs = transcribe_wav(
        "them.wav",
        "Them",
        model="mlx-community/whisper-large-v3-turbo",
        transcribe_fn=fake_transcribe,
        load_audio=fake_load_audio,
    )

    assert segs == [
        Segment(start=0.0, end=1.2, speaker="Them", text="hello"),
        Segment(start=1.5, end=2.0, speaker="Them", text="morning"),
    ]
    assert captured["path"] == "them.wav"
    assert captured["audio"] is fake_audio
    assert captured["kwargs"]["path_or_hf_repo"] == "mlx-community/whisper-large-v3-turbo"


def test_transcribe_session_writes_merged_minutes(tmp_path):
    """Transcribe both wavs with proper speakers and write merged minutes.md."""
    calls = []

    def fake_transcribe(path, speaker, *, model, text_filter=None, detect_speech=None):
        calls.append((Path(path).name, speaker, model))
        return {
            "Me": [Segment(5.0, "Me", "morning")],
            "Them": [Segment(2.0, "Them", "hello")],
        }[speaker]

    out = transcribe_session(
        tmp_path,
        title="Weekly sync",
        date="2026-06-25",
        model="M",
        transcribe=fake_transcribe,
    )

    assert out == tmp_path / "minutes.md"
    assert out.read_text(encoding="utf-8") == (
        "# Weekly sync\n\n"
        "> Date: 2026-06-25\n\n"
        "- **[00:00:02] Them**: hello\n"
        "- **[00:00:05] Me**: morning\n"
    )
    assert ("me.wav", "Me", "M") in calls
    assert ("them.wav", "Them", "M") in calls


def test_transcribe_wav_applies_text_filter():
    """A text_filter is applied to each segment's stripped text."""
    fake_result = {
        "segments": [
            {"start": 0.0, "end": 1.0, "text": " 软件 "},
            {"start": 1.0, "end": 2.0, "text": "鼠标"},
        ]
    }

    def fake_transcribe(audio, **kwargs):
        return fake_result

    def fake_load_audio(path):
        return None

    def fake_filter(text):
        return text.replace("软件", "軟體").replace("鼠标", "滑鼠")

    segs = transcribe_wav(
        "a.wav",
        "Me",
        transcribe_fn=fake_transcribe,
        load_audio=fake_load_audio,
        text_filter=fake_filter,
    )

    assert segs == [
        Segment(start=0.0, end=1.0, speaker="Me", text="軟體"),
        Segment(start=1.0, end=2.0, speaker="Me", text="滑鼠"),
    ]


def test_transcribe_wav_uses_anti_hallucination_options():
    """transcribe_wav passes anti-hallucination decoding options to whisper."""
    captured = {}

    def fake_transcribe(audio, **kwargs):
        captured.update(kwargs)
        return {"segments": []}

    def fake_load_audio(path):
        return None

    transcribe_wav(
        "a.wav",
        "Me",
        transcribe_fn=fake_transcribe,
        load_audio=fake_load_audio,
    )

    assert captured["condition_on_previous_text"] is False
    assert captured["word_timestamps"] is True
    assert captured["hallucination_silence_threshold"] == 2.0
    assert captured["temperature"] == 0.0  # no temperature fallback -> reproducible


def test_transcribe_wav_chunks_by_vad_and_offsets_timestamps():
    """With detect_speech, only speech chunks are transcribed and timestamps are offset."""
    audio = np.arange(100, dtype=np.float32)  # 10s at sample_rate=10
    captured_starts = []

    def fake_load_audio(path):
        return audio

    def fake_detect_speech(path):
        return [(1.0, 2.0), (5.0, 6.0)]  # far apart -> two chunks (0.8-2.2), (4.8-6.2)

    def fake_transcribe(chunk_audio, **kwargs):
        first = float(chunk_audio[0])
        captured_starts.append(first)
        return {"segments": [{"start": 0.1, "end": 0.5, "text": f"chunk@{first:.0f}"}]}

    segs = transcribe_wav(
        "them.wav",
        "Them",
        transcribe_fn=fake_transcribe,
        load_audio=fake_load_audio,
        detect_speech=fake_detect_speech,
        sample_rate=10,
    )

    # chunk 1 audio starts at sample round(0.8*10)=8; chunk 2 at round(4.8*10)=48
    assert captured_starts == [8.0, 48.0]
    # segment timestamps offset by each chunk's start (0.8 and 4.8)
    assert [(s.speaker, s.text) for s in segs] == [
        ("Them", "chunk@8"),
        ("Them", "chunk@48"),
    ]
    assert segs[0].start == pytest.approx(0.9)
    assert segs[0].end == pytest.approx(1.3)
    assert segs[1].start == pytest.approx(4.9)
    assert segs[1].end == pytest.approx(5.3)


def test_make_text_filter_converts_to_taiwan_traditional():
    """make_text_filter('s2twp') converts Simplified to Taiwan Traditional; 'none' disables."""
    convert = make_text_filter("s2twp")
    assert convert("这个软件") == "這個軟體"

    assert make_text_filter("none") is None


def test_transcribe_session_labels_remote_speakers_when_num_speakers_given(tmp_path):
    """With num_speakers, them.wav is diarized into Speaker labels; me stays Me."""
    captured = {}

    def fake_transcribe(path, speaker, *, model, text_filter=None, detect_speech=None):
        if Path(path).name == "me.wav":
            return [Segment(start=10.0, end=12.0, speaker="Me", text="hi")]
        return [
            Segment(start=0.0, end=3.0, speaker="Them", text="hello"),
            Segment(start=4.0, end=6.0, speaker="Them", text="morning"),
        ]

    def fake_diarize(path, num_speakers, *, device="auto"):
        captured["args"] = (Path(path).name, num_speakers)
        return [(0.0, 3.5, "SPEAKER_00"), (3.5, 6.5, "SPEAKER_01")]

    out = transcribe_session(
        tmp_path,
        title="Meeting",
        date="2026-06-25",
        num_speakers=2,
        transcribe=fake_transcribe,
        diarize=fake_diarize,
    )

    assert captured["args"] == ("them.wav", 2)
    assert out.read_text(encoding="utf-8") == (
        "# Meeting\n\n"
        "> Date: 2026-06-25\n\n"
        "- **[00:00:00] Speaker 1**: hello\n"
        "- **[00:00:04] Speaker 2**: morning\n"
        "- **[00:00:10] Me**: hi\n"
    )


def test_transcribe_session_forwards_device_to_diarize(tmp_path):
    """The chosen device is passed through to diarization."""
    captured = {}

    def fake_transcribe(path, speaker, *, model, text_filter=None, detect_speech=None):
        if Path(path).name == "me.wav":
            return []
        return [Segment(start=0.0, end=3.0, speaker="Them", text="hello")]

    def fake_diarize(path, num_speakers, *, device="auto"):
        captured["device"] = device
        return [(0.0, 3.0, "SPEAKER_00")]

    transcribe_session(
        tmp_path,
        title="Meeting",
        date="2026-06-25",
        num_speakers=2,
        device="mps",
        transcribe=fake_transcribe,
        diarize=fake_diarize,
    )

    assert captured["device"] == "mps"


def test_transcribe_session_skips_diarization_for_single_speaker(tmp_path):
    """num_speakers=1 (one other party) needs no diarization; keep plain Them."""
    diarize_calls = []

    def fake_transcribe(path, speaker, *, model, text_filter=None, detect_speech=None):
        if Path(path).name == "me.wav":
            return []
        return [Segment(start=0.0, end=2.0, speaker="Them", text="hello")]

    def fake_diarize(path, num_speakers, *, device="auto"):
        diarize_calls.append(num_speakers)
        return []

    out = transcribe_session(
        tmp_path,
        title="Meeting",
        date="2026-06-25",
        num_speakers=1,
        transcribe=fake_transcribe,
        diarize=fake_diarize,
    )

    assert diarize_calls == []  # not called for a single speaker
    content = out.read_text(encoding="utf-8")
    assert "- **[00:00:00] Them**: hello" in content
    assert "Speaker 1" not in content


def test_transcribe_session_forwards_detect_speech_when_vad(tmp_path):
    """With vad=True the speech detector is forwarded so transcribe chunks by VAD."""
    received = {}

    def fake_transcribe(path, speaker, *, model, text_filter=None, detect_speech=None):
        received[Path(path).name] = detect_speech
        return [Segment(start=0.0, end=1.0, speaker=speaker, text="hi")]

    def fake_detect_speech(path):
        return [(0.0, 1.0)]

    transcribe_session(
        tmp_path,
        title="Meeting",
        date="2026-06-25",
        vad=True,
        transcribe=fake_transcribe,
        detect_speech=fake_detect_speech,
    )

    assert received["me.wav"] is fake_detect_speech
    assert received["them.wav"] is fake_detect_speech


def test_transcribe_session_does_not_chunk_when_vad_disabled(tmp_path):
    """With vad=False no detector is forwarded, so transcribe runs whole-file."""
    received = {}

    def fake_transcribe(path, speaker, *, model, text_filter=None, detect_speech=None):
        received[Path(path).name] = detect_speech
        return []

    transcribe_session(
        tmp_path,
        title="Meeting",
        date="2026-06-25",
        vad=False,
        transcribe=fake_transcribe,
    )

    assert received["me.wav"] is None
    assert received["them.wav"] is None
