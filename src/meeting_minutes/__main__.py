"""Command line interface for meeting_minutes."""

from __future__ import annotations

import argparse
from pathlib import Path

from meeting_minutes.recorder import record_session
from meeting_minutes.transcribe import (
    DEFAULT_ALIGNER_MODEL,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    make_text_filter,
    resolve_hotwords,
    transcribe_session,
)


def parse_speaker_count(raw) -> int | None:
    """Parse a positive speaker count, or return None when omitted or invalid."""
    value = raw.strip()
    if value.isdigit():
        count = int(value)
        if count > 0:
            return count
    return None


def parse_diarization_choice(raw: str) -> tuple[bool, int | None]:
    """Parse diarization prompt input into (enable_diarize, num_speakers)."""
    value = raw.strip().lower()
    if not value or value == "auto":
        return True, None
    if value.isdigit():
        count = int(value)
        if count >= 2:
            return True, count
    return False, None


def normalize_optional_arg(raw: str | None) -> str | None:
    """Normalize optional string arguments ('none', 'auto', or blank) to None."""
    if raw is None:
        return None
    value = raw.strip()
    if not value or value.lower() in {"none", "auto"}:
        return None
    return value


def derive_date_from_session_dir(session_dir) -> str:
    """Derive a minutes date from a session directory name."""
    name = Path(session_dir).name
    prefix = name[:8]
    if len(prefix) == 8 and prefix.isdigit():
        return f"{prefix[:4]}-{prefix[4:6]}-{prefix[6:8]}"
    return name


def add_common_options(parser) -> None:
    """Add options shared by the record and transcribe subcommands."""
    parser.add_argument("--title", default="Meeting Minutes", help="Minutes title")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="ASR model repository or local path",
    )
    parser.add_argument(
        "--aligner-model",
        default=DEFAULT_ALIGNER_MODEL,
        help="ForcedAligner model repository or 'none' to disable alignment",
    )
    parser.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        help="Transcription language or 'auto' for automatic detection",
    )
    parser.add_argument(
        "--hotwords",
        default=None,
        help="Path to hotwords file (defaults to hotwords.txt if present)",
    )
    parser.add_argument(
        "--no-hotwords",
        action="store_true",
        help="Disable hotwords even if hotwords.txt exists",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Require models to already exist in local Hugging Face cache",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable VAD speech filtering (kept on by default to drop silence/hallucinations)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps"],
        default="auto",
        help="Diarization compute device. auto = Apple GPU (MPS) when available, else CPU",
    )
    parser.add_argument(
        "--zh-variant",
        default="s2twp",
        help="OpenCC config for Chinese conversion, e.g. s2twp = Simplified to Taiwan Traditional, none = no conversion",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(prog="meeting_minutes")
    subparsers = parser.add_subparsers(dest="command", required=True)

    record_parser = subparsers.add_parser("record", help="Record a meeting session")
    record_parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory for the recording session",
    )
    record_parser.add_argument(
        "--binary",
        default="bin/audiotee",
        help="Path to the audiotee binary",
    )
    record_parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Sample rate for microphone and system audio",
    )
    record_parser.add_argument(
        "--transcribe",
        action="store_true",
        help="Transcribe the session after recording",
    )
    add_common_options(record_parser)

    transcribe_parser = subparsers.add_parser(
        "transcribe", help="Transcribe a recorded meeting session"
    )
    transcribe_parser.add_argument("session_dir", help="Recording session directory")
    transcribe_parser.add_argument(
        "--date",
        default=None,
        help="Minutes date. Defaults to a value derived from the session directory.",
    )
    transcribe_parser.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Number of other-party speakers for diarization (omit to skip speaker labeling)",
    )
    transcribe_parser.add_argument(
        "--diarize",
        action="store_true",
        help="Auto-detect and label other-party speakers",
    )
    add_common_options(transcribe_parser)

    return parser


def _transcribe_from_args(
    session_dir: Path,
    args: argparse.Namespace,
    *,
    date: str,
    num_speakers: int | None,
    enable_diarize: bool = False,
) -> Path:
    hotwords_path = normalize_optional_arg(args.hotwords)
    disable_hotwords = args.no_hotwords or (args.hotwords is not None and hotwords_path is None)
    return transcribe_session(
        session_dir,
        title=args.title,
        date=date,
        model=args.model,
        aligner_model=normalize_optional_arg(args.aligner_model),
        language=normalize_optional_arg(args.language),
        hotwords=resolve_hotwords(hotwords_path, no_hotwords=disable_hotwords),
        offline=args.offline,
        text_filter=make_text_filter(args.zh_variant),
        num_speakers=num_speakers,
        enable_diarize=enable_diarize,
        vad=not args.no_vad,
        device=args.device,
    )


def main() -> None:
    """Run the meeting_minutes command line interface."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "record":
        session_dir = record_session(
            args.out_dir,
            binary=args.binary,
            sample_rate=args.sample_rate,
        )
        print(f"Recording directory: {session_dir}")
        print(f"Microphone WAV: {session_dir / 'me.wav'}")
        print(f"System audio WAV: {session_dir / 'them.wav'}")
        if args.transcribe:
            enable_diarize, num_speakers = parse_diarization_choice(
                input(
                    "Other-party speakers (Enter/auto = auto-detect, 2+ = exact count, 1/no = skip): "
                )
            )
            minutes_path = _transcribe_from_args(
                session_dir,
                args,
                date=derive_date_from_session_dir(session_dir),
                num_speakers=num_speakers,
                enable_diarize=enable_diarize,
            )
            print(minutes_path)
    elif args.command == "transcribe":
        session_dir = Path(args.session_dir)
        minutes_path = _transcribe_from_args(
            session_dir,
            args,
            date=args.date or derive_date_from_session_dir(session_dir),
            num_speakers=args.num_speakers,
            enable_diarize=args.diarize,
        )
        print(minutes_path)


if __name__ == "__main__":
    main()
