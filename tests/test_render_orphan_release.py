"""If the progress reader dies mid-render the detached process group must still be reaped
(app/server.py run): it owns the GPU until it goes. python3 -m pytest tests/test_render_orphan_release.py"""
import os
import signal
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("STUDIO_HOME", str(Path(tempfile.mkdtemp()) / "TensorFoldStudio"))
sys.path.insert(0, str(ROOT / "app"))
import server  # noqa: E402


class Boom:
    """The STEP reader after the first line: dies like a torn write or a decode blow-up would."""

    def __init__(self):
        self.n = 0

    def search(self, line):
        self.n += 1
        if self.n > 1:
            raise RuntimeError("the reader died")
        return None


def test_a_dead_reader_still_releases_the_render():
    real_step, real_popen = server.STEP, server.subprocess.Popen
    spawned = []

    def spy(cmd, **kw):
        child = real_popen(cmd, **kw)
        spawned.append(child)
        return child

    server.STEP = Boom()
    server.subprocess.Popen = spy
    try:
        job = {"id": "j-orphan", "log": "", "status": "running", "progress": {}}
        try:
            server.run(job, [sys.executable, "-c",
                             "print('hi', flush=True); print('there', flush=True); import time; time.sleep(30)"], {}, "test")
            raise AssertionError("the reader death should surface")
        except RuntimeError:
            pass
        assert server.running.get("process") is None and server.running.get("id") is None
        child = spawned[0]
        deadline = time.time() + 10
        while child.poll() is None and time.time() < deadline:
            time.sleep(0.1)
        assert child.poll() is not None, "the render outlived its reader — the GPU stays owned"
    finally:
        server.STEP, server.subprocess.Popen = real_step, real_popen
        for child in spawned:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
