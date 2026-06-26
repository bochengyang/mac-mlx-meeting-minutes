"""Transcribe audio files into transcript segments."""

from pathlib import Path

from meeting_minutes.transcript import Segment, render_meeting_minutes


def transcribe_wav(
    path,
    speaker,
    *,
    model="mlx-community/whisper-large-v3-turbo",
    transcribe_fn=None,
    load_audio=None,
    text_filter=None,
    detect_speech=None,
    sample_rate=16000,
) -> list[Segment]:
    """Transcribe a WAV file and attach a speaker label to each segment.

    When ``detect_speech`` is given, the audio is split into speech chunks
    (VAD-first) and each chunk is transcribed on its own. This keeps Whisper
    away from long non-speech stretches, where it otherwise loops into
    repetitive hallucinations that swallow the surrounding real speech.
    Without ``detect_speech`` the whole file is transcribed in one pass.
    """
    if load_audio is None:
        from meeting_minutes.audio import load_wav_float32

        load_audio = load_wav_float32

    if transcribe_fn is None:
        import mlx_whisper

        transcribe_fn = mlx_whisper.transcribe

    audio = load_audio(path)

    options = dict(
        path_or_hf_repo=model,
        condition_on_previous_text=False,  # reduce Whisper hallucinations on silence
        word_timestamps=True,
        hallucination_silence_threshold=2.0,
        temperature=0.0,  # disable temperature fallback so output is reproducible
    )

    def to_segments(result, offset):
        out = []
        for seg in result["segments"]:
            text = seg["text"].strip()
            if text_filter is not None:
                text = text_filter(text)
            out.append(
                Segment(
                    start=offset + seg["start"],
                    end=offset + seg["end"],
                    speaker=speaker,
                    text=text,
                )
            )
        return out

    if detect_speech is None:
        return to_segments(transcribe_fn(audio, **options), 0.0)

    from meeting_minutes.vad import merge_speech_regions

    segments: list[Segment] = []
    for start, end in merge_speech_regions(detect_speech(path)):
        chunk = audio[round(start * sample_rate):round(end * sample_rate)]
        segments.extend(to_segments(transcribe_fn(chunk, **options), start))
    return segments


def make_text_filter(zh_variant):
    """Create a text filter for the requested Chinese variant."""
    if zh_variant in (None, "", "none"):
        return None

    import opencc

    return opencc.OpenCC(zh_variant).convert


def transcribe_session(
    session_dir,
    *,
    title="Meeting Minutes",
    date,
    model="mlx-community/whisper-large-v3-turbo",
    transcribe=None,
    text_filter=None,
    num_speakers=None,
    diarize=None,
    vad: bool = False,
    detect_speech=None,
    device: str = "auto",
) -> Path:
    """Transcribe a recorded session and write merged meeting minutes."""
    if transcribe is None:
        transcribe = transcribe_wav

    session_path = Path(session_dir)

    speech_detector = None
    if vad:
        if detect_speech is None:
            from meeting_minutes.vad import detect_speech

        speech_detector = detect_speech

    me = transcribe(
        session_path / "me.wav", "Me", model=model, text_filter=text_filter, detect_speech=speech_detector
    )
    them = transcribe(
        session_path / "them.wav", "Them", model=model, text_filter=text_filter, detect_speech=speech_detector
    )

    if num_speakers and num_speakers >= 2:
        if diarize is None:
            from meeting_minutes.diarize import diarize_wav

            diarize = diarize_wav

        from meeting_minutes.diarize import label_remote_speakers

        print(f"Separating {num_speakers} speakers...")
        turns = diarize(session_path / "them.wav", num_speakers, device=device)
        them = label_remote_speakers(them, turns)

    document = render_meeting_minutes(me + them, title=title, date=date)

    out = session_path / "minutes.md"
    out.write_text(document, encoding="utf-8")
    return out
