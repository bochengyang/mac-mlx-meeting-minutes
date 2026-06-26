"""Audio capture and file utilities.

audiotee streams raw PCM (default s16le / 16kHz / mono) to stdout; this module
wraps those PCM bytes into standard WAV files for later transcription.
"""

import wave
from pathlib import Path

import numpy as np


def build_audiotee_command(
    binary: str | Path,
    *,
    sample_rate: int = 16000,
    include_pids: list[int] | None = None,
    exclude_pids: list[int] | None = None,
) -> list[str]:
    """Build the audiotee command line."""
    command = [str(binary), "--sample-rate", str(sample_rate)]

    if include_pids:
        command.extend(["--include-processes", *[str(pid) for pid in include_pids]])

    if exclude_pids:
        command.extend(["--exclude-processes", *[str(pid) for pid in exclude_pids]])

    return command


def pcm_to_wav(
    pcm: bytes,
    path: Path | str,
    *,
    sample_rate: int = 16000,
    channels: int = 1,
    sample_width: int = 2,
) -> None:
    """Write raw PCM bytes to a WAV file.

    sample_width is measured in bytes (16-bit = 2).
    """
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sample_width)
        w.setframerate(sample_rate)
        w.writeframes(pcm)


def load_wav_float32_with_rate(path: Path | str) -> tuple[np.ndarray, int]:
    """Load a 16-bit mono WAV file into a normalized float32 waveform and its sample rate."""
    with wave.open(str(path), "rb") as w:
        sample_rate = w.getframerate()
        pcm = w.readframes(w.getnframes())

    audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    return audio, sample_rate


def load_wav_float32(path: Path | str) -> np.ndarray:
    """Load a 16-bit mono WAV file into a normalized float32 waveform."""
    return load_wav_float32_with_rate(path)[0]


def stream_pcm_to_wav(
    reader,
    path: Path | str,
    *,
    sample_rate: int = 16000,
    channels: int = 1,
    sample_width: int = 2,
    chunk_size: int = 4096,
) -> None:
    """Read raw PCM bytes from a stream and write them to a WAV file.

    sample_width is measured in bytes (16-bit = 2).
    """
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sample_width)
        w.setframerate(sample_rate)

        while chunk := reader.read(chunk_size):
            w.writeframes(chunk)
