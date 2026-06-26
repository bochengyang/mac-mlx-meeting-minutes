"""Tests for transcript merge logic (pure functions, no audio/permissions)."""

from meeting_minutes.transcript import (
    Segment,
    merge_transcript,
    render_meeting_minutes,
)


def test_merges_two_speakers_in_chronological_order():
    """Segments from two sources should merge into Markdown by timeline."""
    me = [Segment(start=5.0, speaker="Me", text="morning")]
    them = [Segment(start=2.0, speaker="Them", text="hello")]

    md = merge_transcript(me + them)

    assert md == (
        "- **[00:00:02] Them**: hello\n"
        "- **[00:00:05] Me**: morning"
    )


def test_coalesces_consecutive_same_speaker_segments():
    """Consecutive segments from the same speaker merge into one line."""
    segments = [
        Segment(start=2.0, speaker="Me", text="hello"),
        Segment(start=4.0, speaker="Me", text="morning"),
        Segment(start=6.0, speaker="Them", text="got it"),
    ]

    md = merge_transcript(segments)

    assert md == (
        "- **[00:00:02] Me**: hello morning\n"
        "- **[00:00:06] Them**: got it"
    )


def test_renders_full_document_with_header():
    """render_meeting_minutes should output complete Markdown with title and date."""
    segments = [Segment(start=2.0, speaker="Them", text="hello")]

    doc = render_meeting_minutes(segments, title="Weekly sync", date="2026-06-25")

    assert doc == (
        "# Weekly sync\n\n"
        "> Date: 2026-06-25\n\n"
        "- **[00:00:02] Them**: hello\n"
    )
