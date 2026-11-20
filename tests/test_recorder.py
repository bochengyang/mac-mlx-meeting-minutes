"""Tests for the recording orchestrator (fake subprocess, no real devices)."""

import io
import subprocess
import threading
import time
import wave

import numpy as np
import pytest

from meeting_minutes.audio import pcm_to_wav
from meeting_minutes.recorder import (
    record_microphone,
    record_session,
    record_system_audio,
)


def test_record_system_audio_pipes_stdout_to_wav(tmp_path):
    """record_system_audio spawns audiotee and writes its stdout to a WAV."""
    pcm = b"\x10\x00\x20\x00\x30\x00\x40\x00"  # 4 16-bit samples

    class FakeProc:
        def __init__(self):
            self.stdout = io.BytesIO(pcm)

        def terminate(self):
            pass

    captured = {}

    def fake_popen(cmd, stdout=None):
        captured["cmd"] = cmd
        return FakeProc()

    out = tmp_path / "them.wav"
    stop = threading.Event()

    record_system_audio(
        out, stop, binary="bin/audiotee", sample_rate=16000, popen=fake_popen
    )

    assert captured["cmd"] == ["bin/audiotee", "--sample-rate", "16000"]
    with wave.open(str(out), "rb") as w:
        assert w.getnframes() == 4
        assert w.readframes(4) == pcm


def test_record_system_audio_force_kills_audiotee_that_ignores_terminate(tmp_path):
    """If audiotee ignores terminate(), the recorder escalates to kill() so it cannot leak."""
    pcm = b"\x10\x00\x20\x00"
    released = threading.Event()  # set once the process is force-killed -> stdout EOFs
    events = []

    class FakeStdout:
        def __init__(self):
            self._sent = False

        def read(self, n):
            if not self._sent:
                self._sent = True
                return pcm
            released.wait(timeout=5)  # block like a live audiotee until killed
            return b""

    class FakeProc:
        def __init__(self):
            self.stdout = FakeStdout()
            self._killed = False

        def terminate(self):
            events.append("terminate")  # SIGTERM ignored: does not release

        def wait(self, timeout=None):
            if self._killed:
                return 0
            raise subprocess.TimeoutExpired(cmd="audiotee", timeout=timeout)

        def kill(self):
            events.append("kill")
            self._killed = True
            released.set()

    def fake_popen(cmd, stdout=None):
        return FakeProc()

    out = tmp_path / "them.wav"
    stop = threading.Event()

    def run():
        record_system_audio(
            out,
            stop,
            binary="bin/audiotee",
            sample_rate=16000,
            popen=fake_popen,
            terminate_grace=0.05,
        )

    worker = threading.Thread(target=run)
    worker.start()
    stop.set()
    worker.join(timeout=5)

    assert not worker.is_alive()  # recorder returned instead of leaking the process
    assert "kill" in events  # escalated from terminate to kill


def test_record_microphone_streams_chunks_to_wav_before_stop(tmp_path):
    """record_microphone streams incoming PCM to disk continuously before stop is set."""
    chunk1 = np.arange(8192, dtype=np.int16)
    chunk2 = np.arange(4096, dtype=np.int16) + 100
    chunk1_sent = threading.Event()
    checked_on_disk = threading.Event()
    chunk2_sent = threading.Event()

    def fake_input_stream(*, samplerate, channels, dtype, callback):
        assert samplerate == 16000
        assert channels == 1
        assert dtype == "int16"

        class FakeStream:
            def __enter__(self):
                callback(chunk1, len(chunk1), None, None)
                chunk1_sent.set()

                def send_later():
                    if checked_on_disk.wait(timeout=2.0):
                        callback(chunk2, len(chunk2), None, None)
                        chunk2_sent.set()

                threading.Thread(target=send_later, daemon=True).start()
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        return FakeStream()

    out = tmp_path / "me.wav"
    stop = threading.Event()
    worker = threading.Thread(
        target=record_microphone,
        args=(out, stop),
        kwargs={"sample_rate": 16000, "input_stream": fake_input_stream},
        daemon=True,
    )
    worker.start()

    assert chunk1_sent.wait(timeout=2.0)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if out.exists() and out.stat().st_size > 44:
            break
        time.sleep(0.01)

    assert out.exists() and out.stat().st_size > 44
    checked_on_disk.set()

    assert chunk2_sent.wait(timeout=2.0)
    stop.set()
    worker.join(timeout=2.0)
    assert not worker.is_alive()

    with wave.open(str(out), "rb") as w:
        assert w.getnframes() == len(chunk1) + len(chunk2)
        assert w.readframes(w.getnframes()) == chunk1.tobytes() + chunk2.tobytes()


def test_record_microphone_preserves_written_audio_when_stream_crashes(tmp_path):
    """If the input stream fails after writing chunks, the partial WAV header is still valid."""
    chunk1 = np.ones(8192, dtype=np.int16) * 7
    chunk1_sent = threading.Event()

    def fake_input_stream(*, samplerate, channels, dtype, callback):
        class CrashingStream:
            def __enter__(self):
                callback(chunk1, len(chunk1), None, None)
                chunk1_sent.set()
                return self

            def __exit__(self, exc_type, exc, tb):
                raise RuntimeError("device disconnected")

        return CrashingStream()

    out = tmp_path / "me.wav"
    stop = threading.Event()
    errors: list[Exception] = []

    def run():
        try:
            record_microphone(out, stop, sample_rate=16000, input_stream=fake_input_stream)
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()

    assert chunk1_sent.wait(timeout=2.0)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if out.exists() and out.stat().st_size > 44:
            break
        time.sleep(0.01)

    stop.set()
    worker.join(timeout=2.0)
    assert not worker.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], RuntimeError)

    with wave.open(str(out), "rb") as w:
        assert w.getnframes() == len(chunk1)
        assert w.readframes(len(chunk1)) == chunk1.tobytes()


def test_record_session_saves_both_tracks_on_keyboard_interrupt(tmp_path):
    """Pressing Ctrl+C during record_session cleanly stops and finalizes both WAV tracks."""
    mic_started = threading.Event()
    sys_started = threading.Event()
    pcm = b"\x10\x00\x20\x00"

    def fake_mic(path, stop, *, sample_rate=16000):
        mic_started.set()
        stop.wait(timeout=5.0)
        pcm_to_wav(pcm, path, sample_rate=sample_rate)

    def fake_sys(path, stop, *, binary="bin/audiotee", sample_rate=16000):
        sys_started.set()
        stop.wait(timeout=5.0)
        pcm_to_wav(pcm, path, sample_rate=sample_rate)

    def fake_wait_for_stop():
        assert mic_started.wait(timeout=2.0)
        assert sys_started.wait(timeout=2.0)
        raise KeyboardInterrupt

    session_dir = record_session(
        tmp_path,
        wait_for_stop=fake_wait_for_stop,
        mic_recorder=fake_mic,
        system_recorder=fake_sys,
    )

    assert session_dir == tmp_path
    for filename in ("me.wav", "them.wav"):
        wav_path = tmp_path / filename
        assert wav_path.exists()
        with wave.open(str(wav_path), "rb") as w:
            assert w.getnframes() == 2
            assert w.readframes(2) == pcm
