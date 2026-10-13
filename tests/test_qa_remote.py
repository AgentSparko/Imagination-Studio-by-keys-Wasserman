"""The QA gate where its tools are missing, and on a clip fetched from another machine's app:
python3 tests/test_qa_remote.py. No GPU, no network beyond this machine: the "other app" is a second copy of this
one, served on a local port; the gate's own tools are stood in for."""
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("STUDIO_HOME", str(Path(tempfile.mkdtemp()) / "TensorFoldStudio"))
sys.path.insert(0, str(ROOT / "app"))
import qa  # noqa: E402
import remote  # noqa: E402
import server  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(server.app)
PRESET = "sharp" if "sharp" in server.PRESETS else next(iter(server.PRESETS))


def other_app():
    """This app again on a local port, as another machine's would be reached; (address, stop)."""

    import socket

    import uvicorn

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    served = uvicorn.Server(uvicorn.Config(server.app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=served.run, daemon=True).start()
    for _ in range(100):
        if served.started:
            break
        time.sleep(0.05)
    return f"http://127.0.0.1:{port}", lambda: setattr(served, "should_exit", True)


def wait(job_id: str) -> dict:
    for _ in range(200):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "failed", "cancelled"):
            return job
        time.sleep(0.05)
    raise AssertionError("the job did not end")


def test_what_the_gate_needs_is_said():
    had = qa.tool
    try:
        qa.tool = lambda name: None  # a worker Spark as installed: no ffmpeg, no whisper
        can = qa.ready()
        assert can == {"sheet": False, "transcript": False, "runs": False, "missing": [can["missing"][0]]} and "ffmpeg" in can["missing"][0]
        assert client.get("/api/state").json()["qa"]["runs"] is False
        qa.tool = lambda name: "/usr/bin/" + name if name in ("ffmpeg", "ffprobe") else None
        can = qa.ready()
        assert can["runs"] and can["sheet"]
    finally:
        qa.tool = had


def test_a_gate_that_cannot_run_is_in_the_jobs_result():
    clip = server.HOME / "renders" / "made.mp4"
    clip.write_bytes(b"clip")
    job = {"id": "j1", "kind": "video", "status": "running", "params": {"prompt": "a walk", "preset": PRESET, "seed": 42}}
    had = qa.tool
    try:
        qa.tool = lambda name: None
        assert server.gated(job, clip) is True  # the clip is kept and the job goes on
    finally:
        qa.tool = had
    said = job["qa"]
    assert said["passed"] is None and said["skipped"] is True and said["takes"] == []
    assert "did not run" in said["note"] and "ffmpeg" in said["note"] and "/api/qa" in said["note"] and "renders/made.mp4" in said["note"]
    assert server.public({**job, "log": "", "started": None})["qa"] == said  # it reaches the page and the API


def test_the_head_runs_the_gate_on_a_workers_clip():
    """The worker made the clip and could not check it; the head fetches it by the worker's address and checks it."""

    theirs = server.HOME / "renders" / "worker-take.mp4"
    theirs.write_bytes(b"\0\0\0\x18ftypmp42" + b"x" * 5000)
    done = {"id": "w1", "kind": "video", "status": "done", "title": "their take", "outputs": ["renders/worker-take.mp4"], "error": None, "log": "",
            "created": 0, "started": 1.0, "finished": 2.0, "progress": None, "estimate": None,
            "params": {"prompt": "A man says <d>[English] Good morning.</d>", "seed": 43, "preset": PRESET}}
    server.jobs["w1"] = done
    seen = {}

    def check(clip, prompt, picture, brain=None):  # the gate itself is qa.py's, tested where its tools are
        seen.update(clip=Path(clip), prompt=prompt, size=Path(clip).stat().st_size)
        Path(picture).parent.mkdir(parents=True, exist_ok=True)
        Path(picture).write_bytes(b"jpg")
        return {"audio": {"result": "pass", "heard": "good morning", "note": "2 of the line's 2 words heard"}, "identity": {"result": "skipped", "note": "look yourself"},
                "sharpness": [], "times": [1.0, 2.5, 4.0], "passed": True}

    base, stop = other_app()
    had_check, had_ready = qa.check, qa.ready
    qa.check, qa.ready = check, lambda: {"sheet": True, "transcript": True, "runs": True, "missing": []}
    try:
        # by the worker's job: the clip, the prompt and the seed are read from it
        queued = client.post("/api/qa", json={"from": base, "job": "w1"})
        assert queued.status_code == 200 and queued.json()["kind"] == "qa", queued.text
        job = wait(queued.json()["id"])
        assert job["status"] == "done", job
        assert job["qa"]["passed"] is True and job["qa"]["from"] == base and job["qa"]["takes"][0]["seed"] == 43
        assert seen["prompt"] == done["params"]["prompt"] and seen["size"] == theirs.stat().st_size
        fetched = server.HOME / job["outputs"][0]
        assert fetched.is_file() and fetched.parent.name == "fetched" and fetched.read_bytes() == theirs.read_bytes()
        assert (server.HOME / job["qa"]["takes"][0]["sheet"]).is_file()
        # by path, with the prompt given; and a clip that is already here
        again = wait(client.post("/api/qa", json={"from": base.replace("http://", ""), "path": "renders/worker-take.mp4", "prompt": "p2", "seed": 7}).json()["id"])
        assert again["status"] == "done" and seen["prompt"] == "p2" and again["qa"]["takes"][0]["seed"] == 7
        here = wait(client.post("/api/qa", json={"path": "renders/worker-take.mp4", "prompt": "p3"}).json()["id"])
        assert here["status"] == "done" and seen["clip"] == theirs and here["qa"]["from"] is None
        # what cannot be done is said
        missing = wait(client.post("/api/qa", json={"from": base, "path": "renders/none.mp4"}).json()["id"])
        assert missing["status"] == "failed" and "has no file renders/none.mp4" in missing["error"]
        gone = wait(client.post("/api/qa", json={"from": base, "job": "w404"}).json()["id"])
        assert gone["status"] == "failed" and "has no job w404" in gone["error"]
        assert client.post("/api/qa", json={"from": "ftp://x", "path": "renders/a.mp4"}).status_code == 400
        assert client.post("/api/qa", json={"prompt": "p"}).status_code == 400
        qa.ready = lambda: {"sheet": False, "transcript": False, "runs": False, "missing": ["ffmpeg (sudo apt install ffmpeg)"]}
        refused = client.post("/api/qa", json={"path": "renders/worker-take.mp4"})
        assert refused.status_code == 400 and "ffmpeg" in refused.json()["error"]
    finally:
        qa.check, qa.ready = had_check, had_ready
        stop()
        server.jobs.pop("w1", None)


def test_another_apps_address_and_paths():
    assert remote.address("spark-5:7870") == "http://spark-5:7870" and remote.address(" http://10.0.0.5:7870/ ") == "http://10.0.0.5:7870"
    for bad in ("", "file:///etc/passwd", "http://host/files/x", "http://user:pw@host:1", "http://host:1?x=1"):
        try:
            remote.address(bad)
            raise AssertionError(f"taken as an address: {bad!r}")
        except ValueError:
            pass
    for bad in ("", "../x.mp4", "renders/../../x.mp4", "renders/x.sh", "projects/p.json"):
        try:
            remote.fetch("http://127.0.0.1:1", bad, Path(tempfile.mkdtemp()) / "x.mp4")
            raise AssertionError(f"fetched: {bad!r}")
        except ValueError:
            pass


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
