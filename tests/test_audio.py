"""Tests for audio utilities."""

import io
import struct
import wave

import numpy as np

from meeting_minutes.audio import (
    build_audiotee_command,
    load_wav_float32,
    pcm_to_wav,
    stream_pcm_to_wav,
)


def test_pcm_to_wav_writes_valid_wav(tmp_path):
    """Write raw PCM (s16le/16kHz/mono) as a WAV that wave can read back."""
    pcm = b"\x01\x00\x02\x00\x03\x00\x04\x00"  # 4 16-bit samples
    out = tmp_path / "out.wav"

    pcm_to_wav(pcm, out, sample_rate=16000, channels=1, sample_width=2)

    with wave.open(str(out), "rb") as w:
        assert w.getframerate() == 16000
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getnframes() == 4
        assert w.readframes(4) == pcm


def test_build_audiotee_command_uses_sample_rate():
    """Base invocation passes the binary path and target sample rate."""
    cmd = build_audiotee_command("bin/audiotee", sample_rate=16000)

    assert cmd == ["bin/audiotee", "--sample-rate", "16000"]


def test_build_audiotee_command_includes_process_ids():
    """include_pids appends --include-processes with each PID as a string."""
    cmd = build_audiotee_command(
        "bin/audiotee", sample_rate=16000, include_pids=[1234, 5678]
    )

    assert cmd == [
        "bin/audiotee",
        "--sample-rate",
        "16000",
        "--include-processes",
        "1234",
        "5678",
    ]


def test_build_audiotee_command_excludes_process_ids():
    """exclude_pids appends --exclude-processes with each PID as a string."""
    cmd = build_audiotee_command(
        "bin/audiotee", sample_rate=16000, exclude_pids=[1234]
    )

    assert cmd == [
        "bin/audiotee",
        "--sample-rate",
        "16000",
        "--exclude-processes",
        "1234",
    ]


def test_stream_pcm_to_wav_consumes_reader_into_wav(tmp_path):
    """Read PCM from a binary stream until EOF and write a valid WAV."""
    pcm = b"\x01\x00\x02\x00\x03\x00\x04\x00\x05\x00\x06\x00"  # 6 16-bit samples
    reader = io.BytesIO(pcm)
    out = tmp_path / "stream.wav"

    stream_pcm_to_wav(
        reader,
        out,
        sample_rate=16000,
        channels=1,
        sample_width=2,
        chunk_size=4,  # small chunk to exercise partial reads
    )

    with wave.open(str(out), "rb") as w:
        assert w.getframerate() == 16000
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getnframes() == 6
        assert w.readframes(6) == pcm


def test_load_wav_float32_normalizes_int16(tmp_path):
    """Load a 16-bit mono WAV into a float32 array normalized to [-1, 1)."""
    pcm = struct.pack("<4h", 0, 16384, -16384, 32767)  # known int16 samples
    path = tmp_path / "a.wav"
    pcm_to_wav(pcm, path, sample_rate=16000, channels=1, sample_width=2)

    audio = load_wav_float32(path)

    assert audio.dtype == np.float32
    expected = np.array([0, 16384, -16384, 32767], dtype=np.float32) / 32768.0
    np.testing.assert_allclose(audio, expected, atol=1e-6)
