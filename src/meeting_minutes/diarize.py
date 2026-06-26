"""Speaker diarization and segment speaker-labeling helpers."""

from dataclasses import replace

from meeting_minutes.transcript import Segment


def select_device(preference: str = "auto", *, mps_available: bool | None = None) -> str:
    """Resolve a torch device name for diarization.

    "auto" uses the Apple GPU (MPS) when available — roughly 17x faster than CPU
    on diarization — and falls back to CPU otherwise. "cpu"/"mps" force that
    device regardless of availability.
    """
    if preference in ("cpu", "mps"):
        return preference

    if mps_available is None:
        import torch

        mps_available = torch.backends.mps.is_available()

    return "mps" if mps_available else "cpu"


def diarize_wav(
    path, num_speakers=None, *, pipeline=None, device: str = "auto"
) -> list[tuple[float, float, str]]:
    """Run speaker diarization on a mono 16-bit WAV file."""
    import os
    import warnings

    import torch
    warnings.filterwarnings("ignore", message=".*torchcodec.*")
    from pyannote.audio import Pipeline

    from meeting_minutes.audio import load_wav_float32_with_rate

    if pipeline is None:
        resolved = select_device(device)
        if resolved == "mps":
            # Some pyannote ops lack MPS kernels; fall back to CPU per-op instead of crashing.
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1")
        pipeline.to(torch.device(resolved))

    audio, sample_rate = load_wav_float32_with_rate(path)
    waveform = torch.from_numpy(audio).unsqueeze(0)

    kwargs = {"num_speakers": num_speakers} if num_speakers else {}
    result = pipeline({"waveform": waveform, "sample_rate": sample_rate}, **kwargs)
    diar = getattr(result, "speaker_diarization", None)
    if diar is None:
        diar = getattr(result, "diarization", None)
    if diar is None:
        diar = result

    return [
        (turn.start, turn.end, speaker)
        for turn, _, speaker in diar.itertracks(yield_label=True)
    ]


def label_remote_speakers(
    segments: list[Segment],
    turns: list[tuple[float, float, str]],
    *,
    prefix: str = "Speaker",
) -> list[Segment]:
    """Relabel segments with diarization speakers chosen by maximum overlap."""
    speaker_labels: dict[str, str] = {}
    labeled_segments: list[Segment] = []

    for segment in segments:
        overlaps: dict[str, float] = {}
        for turn_start, turn_end, speaker_id in turns:
            overlap = max(
                0.0,
                min(segment.end, turn_end) - max(segment.start, turn_start),
            )
            if overlap > 0.0:
                overlaps[speaker_id] = overlaps.get(speaker_id, 0.0) + overlap

        if overlaps:
            speaker_id = max(overlaps, key=overlaps.__getitem__)
            if speaker_id not in speaker_labels:
                speaker_labels[speaker_id] = f"{prefix} {len(speaker_labels) + 1}"
            speaker = speaker_labels[speaker_id]
        else:
            speaker = segment.speaker

        labeled_segments.append(replace(segment, speaker=speaker))

    return labeled_segments
