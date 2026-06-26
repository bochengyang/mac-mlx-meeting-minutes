"""Data structures and merge logic for transcript segments.

After the two audio sources (my microphone / the other party's system audio)
are transcribed separately, timestamped segments are merged into one Markdown
meeting minutes document.
"""

from dataclasses import dataclass


@dataclass
class Segment:
    """A transcript text segment.

    start: Seconds relative to the start of recording
    speaker: Speaker label (for example, "Me" / "Them")
    text: Segment text
    end: End time in seconds relative to the start of recording
    """

    start: float
    speaker: str
    text: str
    end: float = 0.0


def _format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def merge_transcript(segments: list[Segment]) -> str:
    """Sort all segments by start time and output Markdown bullets.

    Consecutive segments from the same speaker are merged into one line, using
    the timestamp from the first segment in the group.
    """
    ordered = sorted(segments, key=lambda seg: seg.start)

    groups: list[tuple[float, str, list[str]]] = []
    for seg in ordered:
        if groups and groups[-1][1] == seg.speaker:
            groups[-1][2].append(seg.text)
        else:
            groups.append((seg.start, seg.speaker, [seg.text]))

    lines = [
        f"- **[{_format_timestamp(start)}] {speaker}**: {' '.join(texts)}"
        for start, speaker, texts in groups
    ]
    return "\n".join(lines)


def render_meeting_minutes(
    segments: list[Segment], *, title: str, date: str
) -> str:
    """Output complete Markdown meeting minutes with title and date headers."""
    body = merge_transcript(segments)
    return f"# {title}\n\n> Date: {date}\n\n{body}\n"
