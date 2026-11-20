"""Tests for CLI helpers and argument parsing."""

from meeting_minutes.__main__ import (
    build_parser,
    normalize_optional_arg,
    parse_diarization_choice,
    parse_speaker_count,
)
from meeting_minutes.transcribe import (
    DEFAULT_ALIGNER_MODEL,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
)


def test_parse_speaker_count():
    """A positive integer enables diarization; anything else disables it."""
    assert parse_speaker_count("3") == 3
    assert parse_speaker_count("  5 ") == 5
    assert parse_speaker_count("") is None
    assert parse_speaker_count("abc") is None
    assert parse_speaker_count("0") is None
    assert parse_speaker_count("-2") is None


def test_build_parser_supports_common_transcription_options_for_record_and_transcribe():
    """Both record and transcribe share model, aligner, language, hotwords, offline, and title."""
    parser = build_parser()

    rec_default = parser.parse_args(["record", "--transcribe"])
    assert rec_default.model == DEFAULT_MODEL
    assert rec_default.aligner_model == DEFAULT_ALIGNER_MODEL
    assert rec_default.language == DEFAULT_LANGUAGE
    assert rec_default.hotwords is None
    assert rec_default.no_hotwords is False
    assert rec_default.offline is False
    assert rec_default.title == "Meeting Minutes"

    rec_custom = parser.parse_args(
        [
            "record",
            "--transcribe",
            "--model",
            "M",
            "--language",
            "auto",
            "--hotwords",
            "h.txt",
            "--offline",
            "--title",
            "Standup",
        ]
    )
    assert rec_custom.model == "M"
    assert rec_custom.language == "auto"
    assert rec_custom.hotwords == "h.txt"
    assert rec_custom.offline is True
    assert rec_custom.title == "Standup"

    tr_custom = parser.parse_args(
        [
            "transcribe",
            "recordings/demo",
            "--no-hotwords",
            "--aligner-model",
            "none",
            "--offline",
        ]
    )
    assert tr_custom.session_dir == "recordings/demo"
    assert tr_custom.no_hotwords is True
    assert tr_custom.aligner_model == "none"
    assert tr_custom.offline is True


def test_normalize_optional_arg_handles_none_and_auto():
    """Optional string flags 'none', 'auto', and blanks normalize to None."""
    assert normalize_optional_arg("none") is None
    assert normalize_optional_arg("auto") is None
    assert normalize_optional_arg("  ") is None
    assert normalize_optional_arg(None) is None
    assert normalize_optional_arg("Chinese") == "Chinese"


def test_parse_diarization_choice():
    """Enter/auto enables auto-count diarization; 2+ sets exact count; 1/no/invalid skips."""
    assert parse_diarization_choice("") == (True, None)
    assert parse_diarization_choice("   ") == (True, None)
    assert parse_diarization_choice("auto") == (True, None)
    assert parse_diarization_choice(" Auto ") == (True, None)

    assert parse_diarization_choice("2") == (True, 2)
    assert parse_diarization_choice("  5 ") == (True, 5)

    for skip_val in ("1", "0", "no", "N", "none", "skip", "abc", "-2"):
        assert parse_diarization_choice(skip_val) == (False, None)


def test_build_parser_supports_diarize_flag_on_transcribe():
    """transcribe exposes --diarize (default False) alongside --num-speakers."""
    parser = build_parser()

    default_args = parser.parse_args(["transcribe", "recordings/demo"])
    assert default_args.diarize is False
    assert default_args.num_speakers is None

    auto_args = parser.parse_args(["transcribe", "recordings/demo", "--diarize"])
    assert auto_args.diarize is True
    assert auto_args.num_speakers is None
