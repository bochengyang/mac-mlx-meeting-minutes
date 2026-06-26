"""Tests for VAD speech-region merging (pure logic, no model)."""

from meeting_minutes.vad import merge_speech_regions


def test_merge_speech_regions_pads_and_merges_close_regions():
    """Regions are padded, then those within max_gap are merged into one chunk."""
    regions = [(1.0, 2.0), (2.5, 3.0)]

    chunks = merge_speech_regions(regions, pad=0.2, max_gap=0.8, max_duration=28.0)

    assert chunks == [(0.8, 3.2)]


def test_merge_speech_regions_splits_distant_regions():
    """Regions separated by more than max_gap stay as separate chunks."""
    regions = [(1.0, 2.0), (5.0, 6.0)]

    chunks = merge_speech_regions(regions, pad=0.2, max_gap=0.8, max_duration=28.0)

    assert chunks == [(0.8, 2.2), (4.8, 6.2)]


def test_merge_speech_regions_clamps_pad_at_zero():
    """Padding never produces a negative chunk start."""
    chunks = merge_speech_regions([(0.1, 1.0)], pad=0.2, max_gap=0.8, max_duration=28.0)

    assert chunks == [(0.0, 1.2)]


def test_merge_speech_regions_respects_max_duration():
    """Merging stops before a chunk would exceed max_duration."""
    regions = [(0.0, 10.0), (10.5, 20.0), (20.5, 30.0)]

    chunks = merge_speech_regions(regions, pad=0.0, max_gap=1.0, max_duration=15.0)

    assert chunks == [(0.0, 10.0), (10.5, 20.0), (20.5, 30.0)]


def test_merge_speech_regions_empty_returns_empty():
    """No speech regions yields no chunks."""
    assert merge_speech_regions([], pad=0.2, max_gap=0.8, max_duration=28.0) == []
