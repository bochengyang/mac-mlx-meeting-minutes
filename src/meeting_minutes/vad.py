"""Voice activity detection helpers."""


def detect_speech(path, *, model=None, threshold: float = 0.5) -> list[tuple[float, float]]:
    """Detect speech regions in a WAV file with Silero VAD."""
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    from meeting_minutes.audio import load_wav_float32_with_rate

    if model is None:
        model = load_silero_vad()

    audio, sample_rate = load_wav_float32_with_rate(path)
    timestamps = get_speech_timestamps(
        torch.from_numpy(audio),
        model,
        sampling_rate=sample_rate,
        return_seconds=True,
        threshold=threshold,
    )
    return [(t["start"], t["end"]) for t in timestamps]


def merge_speech_regions(
    regions: list[tuple[float, float]],
    *,
    pad: float = 0.2,
    max_gap: float = 0.8,
    max_duration: float = 28.0,
) -> list[tuple[float, float]]:
    """Merge nearby speech regions into padded chunks for transcription.

    Each region is expanded by ``pad`` seconds on both sides (clamped at 0).
    Consecutive regions whose gap is at most ``max_gap`` are merged, but a chunk
    is never grown past ``max_duration`` seconds. The result drives VAD-first
    chunked transcription, keeping Whisper away from long non-speech stretches
    where it otherwise loops into hallucinations.
    """
    padded = sorted(
        (max(0.0, start - pad), end + pad) for start, end in regions
    )

    chunks: list[tuple[float, float]] = []
    for start, end in padded:
        if chunks and start - chunks[-1][1] <= max_gap and end - chunks[-1][0] <= max_duration:
            chunks[-1] = (chunks[-1][0], max(chunks[-1][1], end))
        else:
            chunks.append((start, end))

    return chunks
