"""Tests for the recording orchestrator (fake subprocess, no real devices)."""

import io
import subprocess
import threading
import wave

from meeting_minutes.recorder import record_system_audio


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
