"""Quitting the app stops the ComfyUI that belongs to it. python3 -m pytest tests/test_session_stop.py."""
import os
import signal
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("STUDIO_HOME", str(Path(tempfile.mkdtemp()) / "TensorFoldStudio"))
os.environ.setdefault("STUDIO_GPU_LOCK", str(Path(tempfile.mkdtemp()) / "studio-gpu.lock.d"))
sys.path.insert(0, str(ROOT / "app"))
import server  # noqa: E402
import telemetry  # noqa: E402


def test_importing_the_app_does_not_stop_comfy_on_its_own():
    assert server.stop_session_comfy not in server.app.router.on_shutdown


def test_quitting_stops_this_installs_comfy(monkeypatch, capsys):
    calls = []

    def run(cmd, **kw):
        calls.append((cmd, kw))
        return type("Done", (), {"stdout": "stopped\n"})()

    monkeypatch.setattr(server.sys, "platform", "linux")
    monkeypatch.setattr(server.subprocess, "run", run)
    server.stop_session_comfy()
    assert calls[0][0] == ["bash", str(server.PACK / "spark/comfy.sh"), "stop"]
    assert calls[0][1]["capture_output"] is True
    assert calls[0][1]["timeout"] == 60
    assert "ComfyUI stopped" in capsys.readouterr().out


def test_quitting_when_comfy_is_already_down_says_nothing(monkeypatch, capsys):
    monkeypatch.setattr(server.sys, "platform", "linux")
    monkeypatch.setattr(server.subprocess, "run", lambda cmd, **_kw: type("Done", (), {"stdout": "not running\n"})())
    server.stop_session_comfy()
    assert capsys.readouterr().out == ""


def test_a_mac_quits_its_own_comfy(monkeypatch):
    calls = []
    monkeypatch.setattr(server.sys, "platform", "darwin")
    monkeypatch.setattr(server.subprocess, "run", lambda cmd, **_kw: calls.append(cmd) or type("Done", (), {"stdout": "not running\n"})())
    server.stop_session_comfy()
    assert calls == [["bash", str(server.PACK / "mac/comfy.sh"), "stop"]]


def test_ctrl_c_ends_the_event_streams_before_the_server_closes():
    seen = {}
    telemetry.STOPPING.clear()
    server.begin_shutdown(lambda sig, _frame: seen.setdefault("sig", sig), signal.SIGINT, None)
    assert telemetry.STOPPING.is_set() and seen["sig"] == signal.SIGINT
    telemetry.STOPPING.clear()


def test_ctrl_c_is_not_raised_again_after_the_hooks():
    import os
    seen = []
    with server.catch_exit(lambda sig, _frame: seen.append(sig)):
        os.kill(os.getpid(), signal.SIGINT)
    assert seen == [signal.SIGINT]


def test_only_the_running_app_registers_the_stop():
    text = (ROOT / "app/server.py").read_text()
    before, main = text.split('if __name__ == "__main__":', 1)
    assert before.count("stop_session_comfy") == 1
    assert "on_shutdown.append(stop_session_comfy)" in main
    assert "with catch_exit(self.handle_exit)" in main
    launcher = (ROOT / "app/launcher.py").read_text()
    assert "signum == signal.SIGINT" in launcher
