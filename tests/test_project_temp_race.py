"""Two saves of one project in flight must not fight over a shared temp (app/server.py save_project).
python3 -m pytest tests/test_project_temp_race.py"""
import json
import os
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("STUDIO_HOME", str(Path(tempfile.mkdtemp()) / "TensorFoldStudio"))
sys.path.insert(0, str(ROOT / "app"))
import server  # noqa: E402


def test_two_saves_in_flight_do_not_tear_the_project_file():
    file = server.project_file("p-race")
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps({"id": "p-race", "name": "start"}))
    real = json.dumps
    gate = threading.Barrier(2, timeout=5)

    def slow_dumps(data, **kw):
        text = real(data, **kw)
        gate.wait()  # both writers reach the swap-in together
        return text

    server.json.dumps = slow_dumps
    said = []

    def save(who):
        try:
            server.save_project("p-race", {"id": "p-race", "name": who})
        except OSError as error:
            said.append(str(error))

    try:
        savers = [threading.Thread(target=save, args=(who,)) for who in ("left", "right")]
        for saver in savers:
            saver.start()
        for saver in savers:
            saver.join(5)
    finally:
        server.json.dumps = real
    assert said == [], f"a second saver fought over the temp: {said}"
    saved = json.loads(file.read_text())
    assert saved["name"] in ("left", "right")
    assert not list(file.parent.glob("project.json*.new")), "a temp was left behind"
