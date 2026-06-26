"""Command line interface for meeting_minutes."""

from __future__ import annotations

import argparse
from pathlib import Path

from meeting_minutes.recorder import record_session
from meeting_minutes.transcribe import make_text_filter, transcribe_session


def parse_speaker_count(raw) -> int | None:
    """Parse a positive speaker count, or return None when omitted or invalid."""
    value = raw.strip()
    if value.isdigit():
        count = int(value)
        if count > 0:
            return count
    return None


def derive_date_from_session_dir(session_dir) -> str:
    """Derive a minutes date from a session directory name."""
    name = Path(session_dir).name
    prefix = name[:8]
    if len(prefix) == 8 and prefix.isdigit():
        return f"{prefix[:4]}-{prefix[4:6]}-{prefix[6:8]}"
    return name


def add_common_options(parser) -> None:
    """Add options shared by the record and transcribe subcommands."""
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


def main() -> None:
    """Run the meeting_minutes command line interface."""
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
    transcribe_parser.add_argument("--title", default="Meeting Minutes", help="Minutes title")
    transcribe_parser.add_argument(
        "--model",
        default="mlx-community/whisper-large-v3-turbo",
        help="Whisper model repository",
    )
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
    add_common_options(transcribe_parser)

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
            num_speakers = parse_speaker_count(
                input(
                    "Other-party speaker count (enter a number to label speakers, or press Enter to skip): "
                )
            )
            text_filter = make_text_filter(args.zh_variant)
            minutes_path = transcribe_session(
                session_dir,
                date=derive_date_from_session_dir(session_dir),
                text_filter=text_filter,
                num_speakers=num_speakers,
                vad=not args.no_vad,
                device=args.device,
            )
            print(minutes_path)
    elif args.command == "transcribe":
        session_dir = Path(args.session_dir)
        text_filter = make_text_filter(args.zh_variant)
        minutes_path = transcribe_session(
            session_dir,
            title=args.title,
            date=args.date or derive_date_from_session_dir(session_dir),
            model=args.model,
            text_filter=text_filter,
            num_speakers=args.num_speakers,
            vad=not args.no_vad,
            device=args.device,
        )
        print(minutes_path)


if __name__ == "__main__":
    main()
