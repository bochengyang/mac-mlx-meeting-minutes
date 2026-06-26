"""Tests for diarization speaker labeling (pure overlap logic, no model)."""

from meeting_minutes.diarize import label_remote_speakers, select_device
from meeting_minutes.transcript import Segment


def test_select_device_auto_prefers_mps_when_available():
    """'auto' picks the Apple GPU (MPS) when present, else falls back to CPU."""
    assert select_device("auto", mps_available=True) == "mps"
    assert select_device("auto", mps_available=False) == "cpu"


def test_select_device_explicit_choice_is_respected():
    """An explicit device is returned as-is regardless of MPS availability."""
    assert select_device("cpu", mps_available=True) == "cpu"
    assert select_device("mps", mps_available=False) == "mps"


def test_label_remote_speakers_by_overlap():
    """Relabel each segment with the speaker whose turn overlaps it most.

    Distinct diarization speakers are numbered by first appearance:
    SPEAKER_00 -> Speaker 1, SPEAKER_01 -> Speaker 2, ...
    """
    segments = [
        Segment(start=0.0, end=3.0, speaker="Them", text="hello"),
        Segment(start=4.0, end=6.0, speaker="Them", text="morning"),
        Segment(start=7.0, end=9.0, speaker="Them", text="bye"),
    ]
    turns = [
        (0.0, 3.5, "SPEAKER_00"),
        (3.5, 6.5, "SPEAKER_01"),
        (6.5, 10.0, "SPEAKER_00"),
    ]

    result = label_remote_speakers(segments, turns, prefix="Speaker")

    assert [s.speaker for s in result] == ["Speaker 1", "Speaker 2", "Speaker 1"]
    assert [s.text for s in result] == ["hello", "morning", "bye"]
    assert [s.start for s in result] == [0.0, 4.0, 7.0]
