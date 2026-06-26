"""Recording orchestration for microphone and system audio."""

from __future__ import annotations

import subprocess
import threading
from datetime import datetime
from pathlib import Path

from meeting_minutes.audio import (
    build_audiotee_command,
    pcm_to_wav,
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


def record_microphone(
    out_path: Path | str,
    stop: threading.Event,
    *,
    sample_rate: int = 16000,
    channels: int = 1,
) -> None:
    """Record the default microphone as 16-bit PCM and write it as WAV."""
    import sounddevice as sd

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pcm = bytearray()
    lock = threading.Lock()

    def callback(indata, frames, time, status) -> None:
        del frames, time, status
        data = indata.tobytes() if hasattr(indata, "tobytes") else bytes(indata)
        with lock:
            pcm.extend(data)

    with sd.InputStream(
        samplerate=sample_rate,
        channels=channels,
        dtype="int16",
        callback=callback,
    ):
        while not stop.wait(0.1):
            pass

    with lock:
        data = bytes(pcm)

    pcm_to_wav(
        data,
        path,
        sample_rate=sample_rate,
        channels=channels,
        sample_width=2,
    )


def record_session(
    out_dir: Path | str | None = None,
    *,
    binary: str | Path = "bin/audiotee",
    sample_rate: int = 16000,
) -> Path:
    """Record microphone and system audio until Enter is pressed."""
    if out_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        session_dir = Path.cwd() / "recordings" / timestamp
    else:
        session_dir = Path(out_dir)

    session_dir.mkdir(parents=True, exist_ok=True)
    me_path = session_dir / "me.wav"
    them_path = session_dir / "them.wav"

    stop = threading.Event()
    threads = [
        threading.Thread(
            target=record_microphone,
            args=(me_path, stop),
            kwargs={"sample_rate": sample_rate},
            daemon=True,
        ),
        threading.Thread(
            target=record_system_audio,
            args=(them_path, stop),
            kwargs={"binary": binary, "sample_rate": sample_rate},
            daemon=True,
        ),
    ]

    for thread in threads:
        thread.start()

    print("Recording... press Enter to stop")
    input()

    stop.set()
    for thread in threads:
        thread.join(timeout=5)

    return session_dir
