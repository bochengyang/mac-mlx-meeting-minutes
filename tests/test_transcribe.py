"""Tests for Qwen3-ASR transcription, hotwords, offline privacy lock, and single-track handling."""

import os
from pathlib import Path

import numpy as np
import pytest

from meeting_minutes.transcribe import (
    DEFAULT_ALIGNER_MODEL,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    align_text_to_segments,
    configure_offline_mode,
    load_hotwords,
    make_text_filter,
    resolve_hotwords,
    transcribe_session,
    transcribe_wav,
)
from meeting_minutes.transcript import Segment


def test_default_model_constants():
    """Default engine uses Qwen3-ASR-1.7B, Qwen3-ForcedAligner, and Chinese."""
    assert DEFAULT_MODEL == "mlx-community/Qwen3-ASR-1.7B-bf16"
    assert DEFAULT_ALIGNER_MODEL == "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"
    assert DEFAULT_LANGUAGE == "Chinese"


def test_load_hotwords_parses_comments_blanks_preserves_hash_terms_and_deduplicates(tmp_path):
    """Comments starting with # and blanks are skipped, C# is kept, duplicates removed."""
    hotwords_file = tmp_path / "hotwords.txt"
    hotwords_file.write_text(
        "# team acronyms\n\n  PR  \n  C#  \nTDX\n# duplicate below\nPR\nfeedback\n",
        encoding="utf-8",
    )

    assert load_hotwords(hotwords_file) == ["PR", "C#", "TDX", "feedback"]


def test_resolve_hotwords_handles_default_missing_explicit_missing_and_disabled(tmp_path):
    """Default file auto-loads if present, missing explicit file raises, disabled returns empty."""
    default_file = tmp_path / "hotwords.txt"
    assert resolve_hotwords(None, default_path=default_file) == []

    default_file.write_text("PR\nTDX\n", encoding="utf-8")
    assert resolve_hotwords(None, default_path=default_file) == ["PR", "TDX"]
    assert resolve_hotwords(None, no_hotwords=True, default_path=default_file) == []

    missing_explicit = tmp_path / "custom.txt"
    with pytest.raises(FileNotFoundError):
        resolve_hotwords(missing_explicit, default_path=default_file)


def test_configure_offline_mode_enables_offline_when_models_cached(tmp_path, monkeypatch):
    """When all required models exist in cache, offline env vars are locked on automatically."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.delenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", raising=False)

    snap = tmp_path / "models--mlx-community--Qwen3-ASR-1.7B-bf16" / "snapshots" / "rev1"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}", encoding="utf-8")

    is_offline = configure_offline_mode(
        ["mlx-community/Qwen3-ASR-1.7B-bf16"],
        offline=False,
        cache_dir=tmp_path,
    )

    assert is_offline is True
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"


def test_configure_offline_mode_raises_when_forced_offline_and_cache_missing(tmp_path, monkeypatch):
    """Forced offline mode fails loudly instead of hitting the network when cache is absent."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    with pytest.raises(RuntimeError, match="Offline"):
        configure_offline_mode(
            ["mlx-community/Qwen3-ASR-1.7B-bf16"],
            offline=True,
            cache_dir=tmp_path,
        )


def test_configure_offline_mode_allows_first_download_when_not_cached_and_not_forced(
    tmp_path, monkeypatch
):
    """When uncached and offline=False, allows initial download while disabling implicit token."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.delenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", raising=False)

    is_offline = configure_offline_mode(
        ["mlx-community/Qwen3-ASR-1.7B-bf16"],
        offline=False,
        cache_dir=tmp_path,
    )

    assert is_offline is False
    assert os.environ.get("HF_HUB_OFFLINE") != "1"
    assert os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"


def test_align_text_to_segments_splits_on_punctuation_and_long_pauses():
    """ForcedAligner items are grouped into clauses split at punctuation or >0.6s gaps."""
    items = [
        {"start": 0.1, "end": 0.4, "text": "大家好"},
        {"start": 0.45, "end": 0.8, "text": "今天"},
        {"start": 0.82, "end": 1.1, "text": "開會"},
        {"start": 1.9, "end": 2.3, "text": "看PR"},
    ]
    chunk_text = "大家好，今天開會 看PR"

    clauses = align_text_to_segments(items, chunk_text, chunk_duration=3.0)

    assert len(clauses) == 3
    assert clauses[0] == {"start": pytest.approx(0.1), "end": pytest.approx(0.4), "text": "大家好，"}
    assert clauses[1] == {"start": pytest.approx(0.45), "end": pytest.approx(1.1), "text": "今天開會"}
    assert clauses[2] == {"start": pytest.approx(1.9), "end": pytest.approx(2.3), "text": "看PR"}


def test_align_text_to_segments_handles_empty_items_and_inverted_timestamps():
    """Empty aligner items fall back to chunk bounds; inverted end < start is clamped."""
    assert align_text_to_segments([], "hello", chunk_duration=2.5) == [
        {"start": 0.0, "end": 2.5, "text": "hello"}
    ]

    clauses = align_text_to_segments(
        [{"start": 1.5, "end": 1.2, "text": "hi"}],
        "hi",
        chunk_duration=2.0,
    )
    assert clauses == [{"start": pytest.approx(1.5), "end": pytest.approx(1.5), "text": "hi"}]


def test_transcribe_wav_defaults_to_qwen3_chinese_and_forwards_hotwords():
    """transcribe_wav forwards Qwen3 defaults, language='Chinese', hotwords, and maps speaker."""
    fake_audio = np.zeros(16000, dtype=np.float32)
    captured = {}

    def fake_load_audio(path):
        captured["path"] = path
        return fake_audio

    def fake_transcribe(audio, *, model, language, hotwords):
        captured["transcribe"] = {
            "audio": audio,
            "model": model,
            "language": language,
            "hotwords": hotwords,
        }
        return "大家好，看PR"

    def fake_align(audio, text, *, model, language):
        captured["align"] = {
            "audio": audio,
            "text": text,
            "model": model,
            "language": language,
        }
        return [
            {"start": 0.0, "end": 0.5, "text": "大家好"},
            {"start": 0.55, "end": 0.9, "text": "看PR"},
        ]

    segs = transcribe_wav(
        "them.wav",
        "Them",
        hotwords=["PR", "TDX"],
        transcribe_fn=fake_transcribe,
        align_fn=fake_align,
        load_audio=fake_load_audio,
    )

    assert captured["path"] == "them.wav"
    assert captured["transcribe"]["model"] == DEFAULT_MODEL
    assert captured["transcribe"]["language"] == DEFAULT_LANGUAGE
    assert captured["transcribe"]["hotwords"] == ["PR", "TDX"]
    assert captured["align"]["model"] == DEFAULT_ALIGNER_MODEL
    assert captured["align"]["language"] == DEFAULT_LANGUAGE
    assert segs == [
        Segment(start=0.0, end=0.5, speaker="Them", text="大家好，"),
        Segment(start=0.55, end=0.9, speaker="Them", text="看PR"),
    ]


def test_transcribe_wav_applies_text_filter():
    """text_filter is applied to stripped segment text and empty segments are dropped."""
    fake_audio = np.zeros(16000, dtype=np.float32)

    def fake_transcribe(audio, *, model, language, hotwords):
        return " 软件，  ，鼠标 "

    def fake_align(audio, text, *, model, language):
        return [
            {"start": 0.0, "end": 0.4, "text": "软件"},
            {"start": 0.5, "end": 0.6, "text": ""},
            {"start": 0.7, "end": 1.0, "text": "鼠标"},
        ]

    def fake_filter(text):
        return text.replace("软件", "軟體").replace("鼠标", "滑鼠").strip("， ")

    segs = transcribe_wav(
        "a.wav",
        "Me",
        transcribe_fn=fake_transcribe,
        align_fn=fake_align,
        load_audio=lambda p: fake_audio,
        text_filter=fake_filter,
    )

    assert segs == [
        Segment(start=0.0, end=0.4, speaker="Me", text="軟體"),
        Segment(start=0.7, end=1.0, speaker="Me", text="滑鼠"),
    ]


def test_transcribe_wav_chunks_by_vad_aligns_clauses_and_offsets_timestamps():
    """With detect_speech, only VAD chunks are transcribed, aligned, and offset by chunk start."""
    audio = np.arange(100, dtype=np.float32)  # 10s at sample_rate=10
    captured_starts = []

    def fake_detect_speech(path):
        return [(1.0, 2.0), (5.0, 6.0)]  # padded to (0.8, 2.2) and (4.8, 6.2)

    def fake_transcribe(chunk_audio, *, model, language, hotwords):
        first = float(chunk_audio[0])
        captured_starts.append(first)
        return f"chunk@{first:.0f}"

    def fake_align(chunk_audio, text, *, model, language):
        return [{"start": 0.1, "end": 0.5, "text": text}]

    segs = transcribe_wav(
        "them.wav",
        "Them",
        transcribe_fn=fake_transcribe,
        align_fn=fake_align,
        load_audio=lambda p: audio,
        detect_speech=fake_detect_speech,
        sample_rate=10,
    )

    assert captured_starts == [8.0, 48.0]
    assert [(s.speaker, s.text) for s in segs] == [
        ("Them", "chunk@8"),
        ("Them", "chunk@48"),
    ]
    assert segs[0].start == pytest.approx(0.9)
    assert segs[0].end == pytest.approx(1.3)
    assert segs[1].start == pytest.approx(4.9)
    assert segs[1].end == pytest.approx(5.3)


def test_transcribe_wav_falls_back_to_chunk_bounds_when_aligner_raises():
    """If ForcedAligner raises, transcription degrades gracefully to chunk-level timestamps."""
    audio = np.arange(100, dtype=np.float32)  # 10s at sample_rate=10

    def fake_transcribe(chunk_audio, *, model, language, hotwords):
        return "完整句子"

    def broken_align(chunk_audio, text, *, model, language):
        raise RuntimeError("aligner failure")

    segs = transcribe_wav(
        "them.wav",
        "Them",
        transcribe_fn=fake_transcribe,
        align_fn=broken_align,
        load_audio=lambda p: audio,
        detect_speech=lambda p: [(1.0, 2.0)],  # padded to (0.8, 2.2)
        sample_rate=10,
    )

    assert len(segs) == 1
    assert segs[0].text == "完整句子"
    assert segs[0].start == pytest.approx(0.8)
    assert segs[0].end == pytest.approx(2.2)


def test_make_text_filter_converts_to_taiwan_traditional():
    """make_text_filter('s2twp') converts Simplified to Taiwan Traditional; 'none' disables."""
    convert = make_text_filter("s2twp")
    assert convert("这个软件") == "這個軟體"
    assert make_text_filter("none") is None


def test_transcribe_session_writes_merged_minutes(tmp_path):
    """Transcribe both wavs with proper speakers and write merged minutes.md."""
    (tmp_path / "me.wav").write_bytes(b"stub")
    (tmp_path / "them.wav").write_bytes(b"stub")
    calls = []

    def fake_transcribe(path, speaker, **kwargs):
        calls.append((Path(path).name, speaker, kwargs["model"]))
        return {
            "Me": [Segment(5.0, "Me", "morning")],
            "Them": [Segment(2.0, "Them", "hello")],
        }[speaker]

    out = transcribe_session(
        tmp_path,
        title="Weekly sync",
        date="2026-06-25",
        model="M",
        transcribe=fake_transcribe,
    )

    assert out == tmp_path / "minutes.md"
    assert out.read_text(encoding="utf-8") == (
        "# Weekly sync\n\n"
        "> Date: 2026-06-25\n\n"
        "- **[00:00:02] Them**: hello\n"
        "- **[00:00:05] Me**: morning\n"
    )
    assert ("me.wav", "Me", "M") in calls
    assert ("them.wav", "Them", "M") in calls


def test_transcribe_session_labels_remote_speakers_when_num_speakers_given(tmp_path):
    """With num_speakers, them.wav is diarized into Speaker labels; me stays Me."""
    (tmp_path / "me.wav").write_bytes(b"stub")
    (tmp_path / "them.wav").write_bytes(b"stub")
    captured = {}

    def fake_transcribe(path, speaker, **kwargs):
        if Path(path).name == "me.wav":
            return [Segment(start=10.0, end=12.0, speaker="Me", text="hi")]
        return [
            Segment(start=0.0, end=3.0, speaker="Them", text="hello"),
            Segment(start=4.0, end=6.0, speaker="Them", text="morning"),
        ]

    def fake_diarize(path, num_speakers, *, device="auto"):
        captured["args"] = (Path(path).name, num_speakers)
        return [(0.0, 3.5, "SPEAKER_00"), (3.5, 6.5, "SPEAKER_01")]

    out = transcribe_session(
        tmp_path,
        title="Meeting",
        date="2026-06-25",
        num_speakers=2,
        transcribe=fake_transcribe,
        diarize=fake_diarize,
    )

    assert captured["args"] == ("them.wav", 2)
    assert out.read_text(encoding="utf-8") == (
        "# Meeting\n\n"
        "> Date: 2026-06-25\n\n"
        "- **[00:00:00] Speaker 1**: hello\n"
        "- **[00:00:04] Speaker 2**: morning\n"
        "- **[00:00:10] Me**: hi\n"
    )


def test_transcribe_session_forwards_device_to_diarize(tmp_path):
    """The chosen device is passed through to diarization."""
    (tmp_path / "me.wav").write_bytes(b"stub")
    (tmp_path / "them.wav").write_bytes(b"stub")
    captured = {}

    def fake_transcribe(path, speaker, **kwargs):
        if Path(path).name == "me.wav":
            return []
        return [Segment(start=0.0, end=3.0, speaker="Them", text="hello")]

    def fake_diarize(path, num_speakers, *, device="auto"):
        captured["device"] = device
        return [(0.0, 3.0, "SPEAKER_00")]

    transcribe_session(
        tmp_path,
        title="Meeting",
        date="2026-06-25",
        num_speakers=2,
        device="mps",
        transcribe=fake_transcribe,
        diarize=fake_diarize,
    )

    assert captured["device"] == "mps"


def test_transcribe_session_skips_diarization_for_single_speaker(tmp_path):
    """num_speakers=1 (one other party) needs no diarization; keep plain Them."""
    (tmp_path / "me.wav").write_bytes(b"stub")
    (tmp_path / "them.wav").write_bytes(b"stub")
    diarize_calls = []

    def fake_transcribe(path, speaker, **kwargs):
        if Path(path).name == "me.wav":
            return []
        return [Segment(start=0.0, end=2.0, speaker="Them", text="hello")]

    def fake_diarize(path, num_speakers, *, device="auto"):
        diarize_calls.append(num_speakers)
        return []

    out = transcribe_session(
        tmp_path,
        title="Meeting",
        date="2026-06-25",
        num_speakers=1,
        transcribe=fake_transcribe,
        diarize=fake_diarize,
    )

    assert diarize_calls == []
    content = out.read_text(encoding="utf-8")
    assert "- **[00:00:00] Them**: hello" in content
    assert "Speaker 1" not in content


def test_transcribe_session_forwards_detect_speech_when_vad(tmp_path):
    """With vad=True the speech detector is forwarded so transcribe chunks by VAD."""
    (tmp_path / "me.wav").write_bytes(b"stub")
    (tmp_path / "them.wav").write_bytes(b"stub")
    received = {}

    def fake_transcribe(path, speaker, **kwargs):
        received[Path(path).name] = kwargs["detect_speech"]
        return [Segment(start=0.0, end=1.0, speaker=speaker, text="hi")]

    def fake_detect_speech(path):
        return [(0.0, 1.0)]

    transcribe_session(
        tmp_path,
        title="Meeting",
        date="2026-06-25",
        vad=True,
        transcribe=fake_transcribe,
        detect_speech=fake_detect_speech,
    )

    assert received["me.wav"] is fake_detect_speech
    assert received["them.wav"] is fake_detect_speech


def test_transcribe_session_does_not_chunk_when_vad_disabled(tmp_path):
    """With vad=False no detector is forwarded, so transcribe runs whole-file."""
    (tmp_path / "me.wav").write_bytes(b"stub")
    (tmp_path / "them.wav").write_bytes(b"stub")
    received = {}

    def fake_transcribe(path, speaker, **kwargs):
        received[Path(path).name] = kwargs["detect_speech"]
        return []

    transcribe_session(
        tmp_path,
        title="Meeting",
        date="2026-06-25",
        vad=False,
        transcribe=fake_transcribe,
    )

    assert received["me.wav"] is None
    assert received["them.wav"] is None


def test_transcribe_session_skips_missing_single_track_and_raises_when_both_missing(tmp_path):
    """Single-track sessions succeed cleanly; missing both wavs raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        transcribe_session(tmp_path, title="Empty", date="2026-11-20", transcribe=lambda *a, **k: [])

    # Only them.wav present -> skips me.wav without failing
    them_dir = tmp_path / "them_only"
    them_dir.mkdir()
    (them_dir / "them.wav").write_bytes(b"stub")
    seen_them = []
    out_them = transcribe_session(
        them_dir,
        title="Them Only",
        date="2026-11-20",
        transcribe=lambda p, spk, **k: (
            seen_them.append(Path(p).name) or [Segment(0.0, spk, "ok", 1.0)]
        ),
    )
    assert seen_them == ["them.wav"]
    assert "- **[00:00:00] Them**: ok" in out_them.read_text(encoding="utf-8")

    # Only me.wav present -> skips them.wav and skips diarization even when num_speakers=2
    me_dir = tmp_path / "me_only"
    me_dir.mkdir()
    (me_dir / "me.wav").write_bytes(b"stub")
    diarize_calls = []
    out_me = transcribe_session(
        me_dir,
        title="Me Only",
        date="2026-11-20",
        num_speakers=2,
        transcribe=lambda p, spk, **k: [Segment(0.0, spk, "solo", 1.0)],
        diarize=lambda *a, **k: diarize_calls.append(a) or [],
    )
    assert diarize_calls == []
    assert "- **[00:00:00] Me**: solo" in out_me.read_text(encoding="utf-8")


def test_transcribe_session_forwards_language_hotwords_and_offline(tmp_path):
    """transcribe_session forwards language, hotwords, aligner_model, and offline to transcribe."""
    (tmp_path / "them.wav").write_bytes(b"stub")
    captured = {}

    def fake_transcribe(path, speaker, **kwargs):
        captured.update(kwargs)
        return [Segment(0.0, speaker, "ok", 1.0)]

    transcribe_session(
        tmp_path,
        title="T",
        date="2026-11-20",
        model="custom-asr",
        aligner_model="custom-aligner",
        language="Chinese",
        hotwords=["PR", "TDX"],
        offline=True,
        transcribe=fake_transcribe,
    )

    assert captured["model"] == "custom-asr"
    assert captured["aligner_model"] == "custom-aligner"
    assert captured["language"] == "Chinese"
    assert captured["hotwords"] == ["PR", "TDX"]
    assert captured["offline"] is True


def test_transcribe_session_does_not_block_uncached_diarize_when_asr_cached(tmp_path, monkeypatch):
    """When ASR/aligner are cached but pyannote is not, ASR must not leave HF_HUB_OFFLINE=1 for diarize."""
    cache_dir = tmp_path / "hf_cache"
    for repo in (DEFAULT_MODEL, DEFAULT_ALIGNER_MODEL):
        snap = cache_dir / f"models--{repo.replace('/', '--')}" / "snapshots" / "rev1"
        snap.mkdir(parents=True)
        (snap / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setenv("HF_HUB_CACHE", str(cache_dir))
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    (session_dir / "them.wav").write_bytes(b"stub")

    observed = {}

    def fake_transcribe(path, speaker, **kwargs):
        configure_offline_mode(
            [kwargs["model"], kwargs["aligner_model"]],
            offline=kwargs["offline"],
        )
        observed["during_asr_offline"] = os.environ.get("HF_HUB_OFFLINE")
        return [Segment(0.0, speaker, "hello", 1.0)]

    def fake_diarize(path, num_speakers, *, device="auto"):
        observed["during_diarize_offline"] = os.environ.get("HF_HUB_OFFLINE")
        return [(0.0, 1.0, "SPEAKER_00")]

    transcribe_session(
        session_dir,
        title="Meeting",
        date="2026-11-20",
        num_speakers=2,
        offline=False,
        transcribe=fake_transcribe,
        diarize=fake_diarize,
    )

    assert observed["during_asr_offline"] == "1"
    assert observed["during_diarize_offline"] != "1"


def test_transcribe_session_auto_diarizes_when_enabled_without_num_speakers(tmp_path):
    """enable_diarize=True with num_speakers=None triggers auto-count diarization."""
    (tmp_path / "me.wav").write_bytes(b"stub")
    (tmp_path / "them.wav").write_bytes(b"stub")
    captured = {}

    def fake_transcribe(path, speaker, **kwargs):
        if Path(path).name == "me.wav":
            return [Segment(start=10.0, end=12.0, speaker="Me", text="hi")]
        return [
            Segment(start=0.0, end=3.0, speaker="Them", text="hello"),
            Segment(start=4.0, end=6.0, speaker="Them", text="morning"),
        ]

    def fake_diarize(path, num_speakers, *, device="auto"):
        captured["args"] = (Path(path).name, num_speakers)
        return [(0.0, 3.5, "SPEAKER_00"), (3.5, 6.5, "SPEAKER_01")]

    out = transcribe_session(
        tmp_path,
        title="Meeting",
        date="2026-11-20",
        enable_diarize=True,
        num_speakers=None,
        transcribe=fake_transcribe,
        diarize=fake_diarize,
    )

    assert captured["args"] == ("them.wav", None)
    assert out.read_text(encoding="utf-8") == (
        "# Meeting\n\n"
        "> Date: 2026-11-20\n\n"
        "- **[00:00:00] Speaker 1**: hello\n"
        "- **[00:00:04] Speaker 2**: morning\n"
        "- **[00:00:10] Me**: hi\n"
    )


def test_transcribe_session_enable_diarize_failure_paths(tmp_path, monkeypatch):
    """Verify edge/failure paths for enable_diarize: missing them.wav, num_speakers=1, uncached offline."""
    # Case 1: missing them.wav skips diarization cleanly
    me_only = tmp_path / "me_only"
    me_only.mkdir()
    (me_only / "me.wav").write_bytes(b"stub")
    diarize_calls = []
    out_me = transcribe_session(
        me_only,
        title="Solo",
        date="2026-11-20",
        enable_diarize=True,
        transcribe=lambda p, spk, **k: [Segment(0.0, spk, "solo", 1.0)],
        diarize=lambda *a, **k: diarize_calls.append(a) or [],
    )
    assert diarize_calls == []
    assert "- **[00:00:00] Me**: solo" in out_me.read_text(encoding="utf-8")

    # Case 2: explicit num_speakers=1 overrides enable_diarize=True and stays Them
    both_dir = tmp_path / "both"
    both_dir.mkdir()
    (both_dir / "me.wav").write_bytes(b"stub")
    (both_dir / "them.wav").write_bytes(b"stub")
    out_both = transcribe_session(
        both_dir,
        title="OneOnOne",
        date="2026-11-20",
        enable_diarize=True,
        num_speakers=1,
        transcribe=lambda p, spk, **k: [Segment(0.0, spk, "hey", 1.0)] if spk == "Them" else [],
        diarize=lambda *a, **k: diarize_calls.append(a) or [],
    )
    assert diarize_calls == []
    assert "- **[00:00:00] Them**: hey" in out_both.read_text(encoding="utf-8")

    # Case 3: offline=True + enable_diarize=True (diarize=None) fails loudly if pyannote uncached
    empty_cache = tmp_path / "empty_cache"
    empty_cache.mkdir()
    monkeypatch.setenv("HF_HUB_CACHE", str(empty_cache))
    with pytest.raises(RuntimeError, match="Offline"):
        transcribe_session(
            both_dir,
            title="OfflineFail",
            date="2026-11-20",
            enable_diarize=True,
            offline=True,
            transcribe=lambda p, spk, **k: [Segment(0.0, spk, "hey", 1.0)],
        )
