"""Recording orchestration for microphone and system audio."""

from __future__ import annotations

import queue
import subprocess
import threading
import wave
from datetime import datetime
from pathlib import Path

from meeting_minutes.audio import (
    build_audiotee_command,
    stream_pcm_to_wav,
)


def _stop_process(proc, grace: float) -> None:
    """Terminate a process, escalating to kill if it ignores the signal.

    audiotee may stay alive after SIGTERM (e.g. while a meeting app holds the
    audio tap); without the kill escalation it would keep writing to the WAV
    after recording stopped and leak as an orphan process.
    """
    try:
        proc.terminate()
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except Exception:
            pass
    except Exception:
        pass


def record_system_audio(
    out_path: Path | str,
    stop: threading.Event,
    *,
    binary: str | Path,
    sample_rate: int = 16000,
    popen=subprocess.Popen,
    terminate_grace: float = 2.0,
) -> None:
    """Record system audio from audiotee stdout into a WAV file."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cmd = build_audiotee_command(binary, sample_rate=sample_rate)
    proc = popen(cmd, stdout=subprocess.PIPE)

    def terminate_when_stopped() -> None:
        stop.wait()
        _stop_process(proc, terminate_grace)

    watcher = threading.Thread(target=terminate_when_stopped, daemon=True)
    watcher.start()

    if proc.stdout is None:
        stop.set()
        raise RuntimeError("audiotee stdout was not captured")

    try:
        stream_pcm_to_wav(proc.stdout, path, sample_rate=sample_rate)
    finally:
        stop.set()
        _stop_process(proc, terminate_grace)


def _drain_queue(chunks: queue.SimpleQueue[bytes], first: bytes = b"") -> bytes:
    """Drain all currently queued PCM byte chunks into a single buffer."""
    parts = [first] if first else []
    while True:
        try:
            item = chunks.get_nowait()
        except queue.Empty:
            break
        if item:
            parts.append(item)
    return b"".join(parts)


def record_microphone(
    out_path: Path | str,
    stop: threading.Event,
    *,
    sample_rate: int = 16000,
    channels: int = 1,
    input_stream=None,
) -> None:
    """Stream the default microphone as 16-bit PCM directly to a WAV file."""
    if input_stream is None:
        import sounddevice as sd

        input_stream = sd.InputStream

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    chunks: queue.SimpleQueue[bytes] = queue.SimpleQueue()

    def callback(indata, frames, time, status) -> None:
        del frames, time, status
        data = indata.tobytes() if hasattr(indata, "tobytes") else bytes(indata)
        if data:
            chunks.put(data)

    with open(path, "wb") as raw_file, wave.open(raw_file, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)

        try:
            with input_stream(
                samplerate=sample_rate,
                channels=channels,
                dtype="int16",
                callback=callback,
            ):
                while not stop.is_set():
                    try:
                        first = chunks.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    data = _drain_queue(chunks, first)
                    if data:
                        wav_file.writeframes(data)
                        raw_file.flush()
        finally:
            data = _drain_queue(chunks)
            if data:
                wav_file.writeframes(data)
                raw_file.flush()


def record_session(
    out_dir: Path | str | None = None,
    *,
    binary: str | Path = "bin/audiotee",
    sample_rate: int = 16000,
    wait_for_stop=None,
    mic_recorder=None,
    system_recorder=None,
) -> Path:
    """Record microphone and system audio until Enter or Ctrl+C."""
    if out_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        session_dir = Path.cwd() / "recordings" / timestamp
    else:
        session_dir = Path(out_dir)

    session_dir.mkdir(parents=True, exist_ok=True)
    me_path = session_dir / "me.wav"
    them_path = session_dir / "them.wav"

    if wait_for_stop is None:
        wait_for_stop = input
    if mic_recorder is None:
        mic_recorder = record_microphone
    if system_recorder is None:
        system_recorder = record_system_audio

    stop = threading.Event()
    threads = [
        threading.Thread(
            target=mic_recorder,
            args=(me_path, stop),
            kwargs={"sample_rate": sample_rate},
            daemon=True,
        ),
        threading.Thread(
            target=system_recorder,
            args=(them_path, stop),
            kwargs={"binary": binary, "sample_rate": sample_rate},
            daemon=True,
        ),
    ]

    for thread in threads:
        thread.start()

    print("Recording... press Enter to stop")
    try:
        wait_for_stop()
    except KeyboardInterrupt:
        print("\nStopping recording...")
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=5)

    return session_dir
