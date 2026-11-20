"""Transcribe audio files into transcript segments."""

from __future__ import annotations

import os
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from meeting_minutes.audio import pcm_to_wav
from meeting_minutes.transcript import Segment, render_meeting_minutes

DEFAULT_MODEL = "mlx-community/Qwen3-ASR-1.7B-bf16"
DEFAULT_ALIGNER_MODEL = "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"
DEFAULT_LANGUAGE = "Chinese"
DEFAULT_DIARIZE_MODEL = "pyannote/speaker-diarization-community-1"
DEFAULT_HOTWORDS_PATH = Path("hotwords.txt")

_SPLIT_PUNCTUATION = set("，。！？；：、,.!?;:…")
_CLOSING_PUNCTUATION = set("」』）)]}”’\"'")

_MODEL_CACHE: dict[str, Any] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def load_hotwords(path: Path | str) -> list[str]:
    """Load deduplicated hotwords from a UTF-8 file, ignoring blanks and # comments."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    terms: list[str] = []
    seen: set[str] = set()
    for raw in lines:
        term = raw.strip()
        if not term or term.startswith("#") or term in seen:
            continue
        seen.add(term)
        terms.append(term)
    return terms


def resolve_hotwords(
    path: Path | str | None = None,
    *,
    no_hotwords: bool = False,
    default_path: Path | str = DEFAULT_HOTWORDS_PATH,
) -> list[str]:
    """Resolve hotwords from an explicit path, optional default file, or disable flag."""
    if no_hotwords:
        return []
    if path is not None:
        return load_hotwords(path)
    default = Path(default_path)
    if default.is_file():
        return load_hotwords(default)
    return []


def _resolve_hf_cache_dir(cache_dir: Path | str | None = None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir).expanduser()
    if env_cache := os.environ.get("HF_HUB_CACHE"):
        return Path(env_cache).expanduser()
    if env_home := os.environ.get("HF_HOME"):
        return Path(env_home).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _is_model_cached(model_id: str, hub_cache: Path) -> bool:
    local = Path(model_id).expanduser()
    if local.is_dir():
        return True
    snapshots = hub_cache / f"models--{model_id.replace('/', '--')}" / "snapshots"
    if not snapshots.is_dir():
        return False
    return any(p.is_dir() and any(p.iterdir()) for p in snapshots.iterdir())


_OFFLINE_ENV_KEYS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")


def _enable_offline_env() -> None:
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


@contextmanager
def _scoped_offline_env():
    """Restore Hugging Face offline env vars to their prior state on exit."""
    previous = {key: os.environ.get(key) for key in _OFFLINE_ENV_KEYS}
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def configure_offline_mode(
    models: Iterable[str],
    offline: bool = False,
    cache_dir: Path | str | None = None,
) -> bool:
    """Lock Hugging Face Hub offline when all required models exist in cache.

    Always disables implicit Hub token usage. Returns True if offline env vars
    were enabled, False when initial download is allowed, and raises
    RuntimeError when offline=True but required models are missing from cache.
    """
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    hub_cache = _resolve_hf_cache_dir(cache_dir)
    required = [m for m in models if m]
    missing = [m for m in required if not _is_model_cached(m, hub_cache)]

    if not missing:
        _enable_offline_env()
        return True

    if offline:
        raise RuntimeError(
            f"Offline mode enabled, but required model(s) missing from cache ({hub_cache}): "
            + ", ".join(missing)
        )
    return False


def _extract_item_fields(item: Any) -> tuple[float, float, str]:
    if isinstance(item, dict):
        start = item.get("start", item.get("start_time", 0.0))
        end = item.get("end", item.get("end_time", start))
        text = item.get("text", "")
    else:
        start = getattr(item, "start", getattr(item, "start_time", 0.0))
        end = getattr(item, "end", getattr(item, "end_time", start))
        text = getattr(item, "text", "")
    start_f = max(0.0, float(start))
    end_f = max(start_f, float(end))
    return start_f, end_f, str(text).strip()


def _punctuation_split_offset(
    between: str,
    chunk_text: str,
    prev_char_end: int,
    char_start: int,
) -> int | None:
    """Return slice offset within ``between`` after clause-ending punctuation, or None."""
    if (
        len(between) == 1
        and between in ".:"
        and prev_char_end > 0
        and chunk_text[prev_char_end - 1].isdigit()
        and char_start < len(chunk_text)
        and chunk_text[char_start].isdigit()
    ):
        return None

    split_idx = -1
    for idx, ch in enumerate(between):
        if ch in _SPLIT_PUNCTUATION:
            split_idx = idx
    if split_idx == -1:
        return None

    cut = split_idx + 1
    while cut < len(between) and (
        between[cut] in _SPLIT_PUNCTUATION or between[cut] in _CLOSING_PUNCTUATION
    ):
        cut += 1
    return cut


def align_text_to_segments(
    items: Iterable[Any],
    chunk_text: str,
    chunk_duration: float = 0.0,
    *,
    pause_threshold: float = 0.6,
) -> list[dict[str, Any]]:
    """Group forced-alignment items into clauses split on punctuation or long pauses."""
    clean_chunk = chunk_text.strip()
    if not clean_chunk:
        return []

    normalized: list[tuple[float, float, str]] = []
    for raw in items:
        start, end, tok = _extract_item_fields(raw)
        if tok:
            normalized.append((start, end, tok))

    if not normalized:
        return [{"start": 0.0, "end": max(0.0, float(chunk_duration)), "text": clean_chunk}]

    aligned: list[tuple[float, float, int, int]] = []
    cursor = 0
    lower_text = chunk_text.lower()
    unmatched_start: float | None = None

    for start, end, tok in normalized:
        idx = chunk_text.find(tok, cursor)
        if idx == -1:
            idx = lower_text.find(tok.lower(), cursor)
        if idx != -1:
            eff_start = (
                min(start, unmatched_start)
                if not aligned and unmatched_start is not None
                else start
            )
            char_end = idx + len(tok)
            aligned.append((eff_start, end, idx, char_end))
            cursor = char_end
        elif aligned:
            prev_s, prev_e, prev_cs, prev_ce = aligned[-1]
            aligned[-1] = (prev_s, max(prev_e, end), prev_cs, prev_ce)
        elif unmatched_start is None:
            unmatched_start = start

    if not aligned:
        return [
            {
                "start": normalized[0][0],
                "end": max(normalized[0][0], normalized[-1][1]),
                "text": clean_chunk,
            }
        ]

    clauses: list[dict[str, Any]] = []
    clause_char_start = 0
    clause_start, clause_end, _, prev_char_end = aligned[0]
    prev_time_end = clause_end

    for start, end, char_start, char_end in aligned[1:]:
        between = chunk_text[prev_char_end:char_start]
        punct_cut = _punctuation_split_offset(between, chunk_text, prev_char_end, char_start)

        if punct_cut is not None:
            clause_char_end = prev_char_end + punct_cut
        elif start - prev_time_end > pause_threshold:
            clause_char_end = prev_char_end
        else:
            clause_char_end = None

        if clause_char_end is not None:
            text = chunk_text[clause_char_start:clause_char_end].strip()
            if text:
                clauses.append({"start": clause_start, "end": clause_end, "text": text})
            clause_char_start = clause_char_end
            clause_start = start
            clause_end = end
        else:
            clause_end = max(clause_end, end)

        prev_char_end = char_end
        prev_time_end = end

    tail_text = chunk_text[clause_char_start:].strip()
    if tail_text:
        clauses.append({"start": clause_start, "end": clause_end, "text": tail_text})

    return clauses


def _get_stt_model(model_id: str) -> Any:
    with _MODEL_CACHE_LOCK:
        if model_id not in _MODEL_CACHE:
            from mlx_audio.stt import load

            _MODEL_CACHE[model_id] = load(model_id)
        return _MODEL_CACHE[model_id]


def _float32_to_pcm16(audio: np.ndarray) -> bytes:
    clipped = np.clip(audio, -1.0, 32767.0 / 32768.0)
    return (clipped * 32768.0).round().astype("<i2").tobytes()


def _infer_aligner_language(text: str, language: str | None) -> str:
    if language:
        if language.strip().lower() == "cantonese":
            return "Chinese"
        return language
    cjk_count = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    ascii_letters = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    if cjk_count < 3 and ascii_letters > cjk_count:
        return "English"
    return "Chinese"


def _default_transcribe(
    audio: np.ndarray,
    *,
    model: str,
    language: str | None,
    hotwords: list[str] | None,
    sample_rate: int = 16000,
) -> Any:
    asr_model = _get_stt_model(model)
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / "chunk.wav"
        pcm_to_wav(_float32_to_pcm16(audio), wav_path, sample_rate=sample_rate)
        return asr_model.generate(
            str(wav_path),
            language=language,
            hotwords=hotwords or None,
        )


def _default_align(
    audio: np.ndarray,
    text: str,
    *,
    model: str,
    language: str | None,
    sample_rate: int = 16000,
) -> Any:
    aligner = _get_stt_model(model)
    aligner_lang = _infer_aligner_language(text, language)
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / "chunk.wav"
        pcm_to_wav(_float32_to_pcm16(audio), wav_path, sample_rate=sample_rate)
        return aligner.generate(
            audio=str(wav_path),
            text=text,
            language=aligner_lang,
        )


def _extract_asr_text(result: Any) -> str:
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, dict):
        return str(result.get("text", "")).strip()
    return str(getattr(result, "text", "")).strip()


def _extract_align_items(result: Any) -> list[Any]:
    if result is None:
        return []
    if hasattr(result, "items") and not isinstance(result, dict):
        return list(result.items)
    return list(result)


def transcribe_wav(
    path: Path | str,
    speaker: str,
    *,
    model: str = DEFAULT_MODEL,
    aligner_model: str | None = DEFAULT_ALIGNER_MODEL,
    language: str | None = DEFAULT_LANGUAGE,
    hotwords: list[str] | None = None,
    offline: bool = False,
    transcribe_fn=None,
    align_fn=None,
    load_audio=None,
    text_filter=None,
    detect_speech=None,
    sample_rate: int = 16000,
) -> list[Segment]:
    """Transcribe a WAV file and attach a speaker label to each segment."""
    models_needed: list[str] = []
    if transcribe_fn is None:
        models_needed.append(model)
    if align_fn is None and aligner_model:
        models_needed.append(aligner_model)

    with _scoped_offline_env():
        if models_needed:
            configure_offline_mode(models_needed, offline=offline)

        if load_audio is None:
            from meeting_minutes.audio import load_wav_float32

            load_audio = load_wav_float32

        if transcribe_fn is None:
            transcribe_fn = lambda chunk, *, model, language, hotwords: _default_transcribe(
                chunk,
                model=model,
                language=language,
                hotwords=hotwords,
                sample_rate=sample_rate,
            )

        if align_fn is None:
            align_fn = lambda chunk, text, *, model, language: _default_align(
                chunk,
                text,
                model=model,
                language=language,
                sample_rate=sample_rate,
            )

        audio = load_audio(path)
        if len(audio) == 0:
            return []

        if detect_speech is None:
            chunks = [(0.0, audio)]
        else:
            from meeting_minutes.vad import merge_speech_regions

            chunks = [
                (start, audio[round(start * sample_rate) : round(end * sample_rate)])
                for start, end in merge_speech_regions(detect_speech(path))
            ]

        segments: list[Segment] = []
        for offset, chunk in chunks:
            if len(chunk) == 0:
                continue
            chunk_text = _extract_asr_text(
                transcribe_fn(chunk, model=model, language=language, hotwords=hotwords)
            )
            if not chunk_text:
                continue

            chunk_duration = len(chunk) / sample_rate
            if aligner_model is not None:
                try:
                    items = _extract_align_items(
                        align_fn(chunk, chunk_text, model=aligner_model, language=language)
                    )
                    clauses = align_text_to_segments(
                        items, chunk_text, chunk_duration=chunk_duration
                    )
                except Exception:
                    clauses = [{"start": 0.0, "end": chunk_duration, "text": chunk_text}]
            else:
                clauses = [{"start": 0.0, "end": chunk_duration, "text": chunk_text}]

            for clause in clauses:
                text = clause["text"].strip()
                if text_filter is not None:
                    text = text_filter(text).strip()
                if not text:
                    continue
                segments.append(
                    Segment(
                        start=offset + clause["start"],
                        end=offset + clause["end"],
                        speaker=speaker,
                        text=text,
                    )
                )

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
    title: str = "Meeting Minutes",
    date: str,
    model: str = DEFAULT_MODEL,
    aligner_model: str | None = DEFAULT_ALIGNER_MODEL,
    language: str | None = DEFAULT_LANGUAGE,
    hotwords: list[str] | None = None,
    offline: bool = False,
    transcribe=None,
    text_filter=None,
    num_speakers: int | None = None,
    enable_diarize: bool = False,
    diarize=None,
    vad: bool = False,
    detect_speech=None,
    device: str = "auto",
) -> Path:
    """Transcribe a recorded session and write merged meeting minutes."""
    session_path = Path(session_dir)
    me_path = session_path / "me.wav"
    them_path = session_path / "them.wav"
    has_me = me_path.exists()
    has_them = them_path.exists()

    if not has_me and not has_them:
        raise FileNotFoundError(
            f"No audio tracks found in {session_path} (expected me.wav and/or them.wav)"
        )

    run_diarize = has_them and (
        (num_speakers is not None and num_speakers >= 2)
        or (enable_diarize and num_speakers is None)
    )

    models_to_check: list[str] = []
    if transcribe is None:
        models_to_check.append(model)
        if aligner_model:
            models_to_check.append(aligner_model)
    if run_diarize and diarize is None:
        models_to_check.append(DEFAULT_DIARIZE_MODEL)

    with _scoped_offline_env():
        if offline and models_to_check:
            configure_offline_mode(models_to_check, offline=True)

        if transcribe is None:
            transcribe = transcribe_wav

        speech_detector = None
        if vad:
            if detect_speech is None:
                from meeting_minutes.vad import detect_speech

            speech_detector = detect_speech

        transcribe_kwargs = dict(
            model=model,
            aligner_model=aligner_model,
            language=language,
            hotwords=hotwords,
            offline=offline,
            text_filter=text_filter,
            detect_speech=speech_detector,
        )
        with _scoped_offline_env():
            me = transcribe(me_path, "Me", **transcribe_kwargs) if has_me else []
            them = transcribe(them_path, "Them", **transcribe_kwargs) if has_them else []

        if run_diarize:
            with _scoped_offline_env():
                if diarize is None:
                    configure_offline_mode([DEFAULT_DIARIZE_MODEL], offline=offline)
                    from meeting_minutes.diarize import diarize_wav

                    diarize = diarize_wav
                elif offline:
                    _enable_offline_env()
                else:
                    configure_offline_mode([DEFAULT_DIARIZE_MODEL], offline=False)

                from meeting_minutes.diarize import label_remote_speakers

                if num_speakers is None:
                    print("Separating speakers (auto)...")
                else:
                    print(f"Separating {num_speakers} speakers...")
                turns = diarize(them_path, num_speakers, device=device)
                them = label_remote_speakers(them, turns)

    document = render_meeting_minutes(me + them, title=title, date=date)

    out = session_path / "minutes.md"
    out.write_text(document, encoding="utf-8")
    return out
