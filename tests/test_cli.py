"""Tests for CLI helpers."""

from meeting_minutes.__main__ import parse_speaker_count


def test_parse_speaker_count():
    """A positive integer enables diarization; anything else disables it."""
    assert parse_speaker_count("3") == 3
    assert parse_speaker_count("  5 ") == 5
    assert parse_speaker_count("") is None
    assert parse_speaker_count("abc") is None
    assert parse_speaker_count("0") is None
    assert parse_speaker_count("-2") is None
