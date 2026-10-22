"""Beats and the rough cut (app/beats.py, app/roughcut.py): python3 -m pytest tests/test_roughcut.py.
The beats are found in click tracks written here at a known tempo; the cut's arithmetic is checked on a made-up beat
grid; the route runs end to end with the queue stood in for (each job is done at once), so nothing renders."""
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import wave
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("STUDIO_HOME", str(Path(tempfile.mkdtemp()) / "ImaginationStudio"))
sys.path.insert(0, str(ROOT / "app"))
import server  # noqa: E402
import beats  # noqa: E402
import roughcut  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(server.app)
HOME = server.HOME
needs_ffmpeg = pytest.mark.skipif(not beats.ffmpeg() or not shutil.which("ffprobe", path=os.environ.get("PATH", "") + ":/opt/homebrew/bin:/usr/local/bin"),
                                  reason="ffmpeg and ffprobe are needed")


def click_track(path: Path, bpm: float, seconds: float = 24.0, rate: int = 44100, start: float = 0.3, accent_phase: int = 0) -> None:
    """A 16-bit WAV of clicks at ``bpm``: a short high tick on every beat, a low thump with it on every fourth beat
    from ``accent_phase`` (the bar's first beat), and a little noise under it all."""

    count, period = int(seconds * rate), 60.0 / bpm
    samples = [0.0] * count
    tick = int(0.03 * rate)
    beat, at = 0, start
    while at < seconds - 0.05:
        first = int(at * rate)
        for i in range(min(tick, count - first)):
            t = i / rate
            value = 0.4 * math.sin(2 * math.pi * 1500 * t) * math.exp(-t * 120)
            if beat % 4 == accent_phase:
                value += 0.5 * math.sin(2 * math.pi * 60 * t) * math.exp(-t * 40)
            samples[first + i] += value
        beat += 1
        at += period
    seed = 12345
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        frames = bytearray()
        for value in samples:
            seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
            noise = (seed / 0x7FFFFFFF - 0.5) * 0.02
            frames += struct.pack("<h", max(-32767, min(32767, int((value + noise) * 32767))))
        out.writeframes(bytes(frames))


# ---------- beats ----------

@needs_ffmpeg
@pytest.mark.parametrize("bpm", [90, 120, 128, 150])
def test_the_tempo_of_a_click_track_is_found_within_two_percent(tmp_path, bpm):
    pytest.importorskip("numpy")
    song = tmp_path / f"click{bpm}.wav"
    click_track(song, bpm)
    found = beats.analyse(song)
    assert abs(found["bpm"] - bpm) / bpm < 0.02, found["bpm"]
    gaps = [b - a for a, b in zip(found["beats"], found["beats"][1:])]
    assert all(abs(g - 60 / bpm) < 0.03 for g in gaps)
    assert abs(found["beats"][0] - 0.3) < 0.03 and found["confidence"] > 0.5
    assert abs(found["duration"] - 24.0) < 0.05 and found["bars"] == len(found["downbeats"]) >= 8


@needs_ffmpeg
def test_the_bar_starts_on_the_accented_beat(tmp_path):
    pytest.importorskip("numpy")
    song = tmp_path / "accent.wav"
    click_track(song, 120, accent_phase=2)  # the thump on the third click: the bar starts at 0.3 + 2 beats
    found = beats.analyse(song)
    assert abs(found["first"] - 1.3) < 0.03
    assert all(abs((b - a) - 2.0) < 0.03 for a, b in zip(found["downbeats"], found["downbeats"][1:]))


@needs_ffmpeg
def test_the_route_measures_a_song_once_and_keeps_it_beside_the_file(monkeypatch):
    pytest.importorskip("numpy")
    (HOME / "music").mkdir(parents=True, exist_ok=True)
    song = HOME / "music" / "beat-route.wav"
    click_track(song, 100)
    (HOME / "music" / "beat-route.take.json").write_text(json.dumps({"id": "beat-route", "path": "music/beat-route.wav", "title": "Clicks"}))
    first = client.post("/api/music/beats", json={"path": "music/beat-route.wav"}).json()
    assert abs(first["bpm"] - 100) < 2 and first["path"] == "music/beat-route.wav"
    assert (HOME / "music" / "beat-route.beats.json").is_file()
    monkeypatch.setattr(beats, "analyse", lambda path: pytest.fail("measured again"))
    again = client.post("/api/music/beats", json={"path": "beat-route"}).json()  # a take's id names its file
    assert again["beats"] == first["beats"]
    assert client.post("/api/music/beats", json={"path": "../outside.wav"}).status_code == 400
    assert client.post("/api/music/beats", json={"path": "music/none.wav"}).status_code == 404
    (HOME / "music" / "notes.txt").write_text("x")
    assert client.post("/api/music/beats", json={"path": "music/notes.txt"}).status_code == 400


# ---------- the cut's arithmetic ----------

GRID = {"bpm": 120.0, "beats": [i * 0.5 for i in range(60)], "downbeats": [i * 2.0 for i in range(15)], "duration": 30.0}


def spans(laid):
    return [(round(c["at"], 4), round(c["out"] - c["in"], 4)) for c in laid["timeline"]]


def test_each_clip_fills_a_bar_trimmed_around_its_middle():
    laid = roughcut.plan([{"path": "a.mp4", "seconds": 8}, {"path": "b.mp4", "seconds": 6}], GRID, "bar")
    assert spans(laid) == [(0.0, 2.0), (2.0, 2.0)]
    assert laid["timeline"][0]["in"] == 3.0 and laid["timeline"][0]["out"] == 5.0 and laid["timeline"][0]["duration"] == 8
    assert laid["timeline"][1]["in"] == 2.0 and laid["slot"] == 2.0
    start = roughcut.plan([{"path": "a.mp4", "seconds": 8}], GRID, "bar", align="start")
    assert start["timeline"][0]["in"] == 0.0 and start["timeline"][0]["out"] == 2.0


def test_beats_and_two_bars_and_a_pickup_before_the_first_downbeat():
    clips = [{"path": f"{n}.mp4", "seconds": 8} for n in "abc"]
    assert spans(roughcut.plan(clips, GRID, "beat")) == [(0.0, 0.5), (0.5, 0.5), (1.0, 0.5)]
    assert spans(roughcut.plan(clips, GRID, "two-bars")) == [(0.0, 4.0), (4.0, 4.0), (8.0, 4.0)]
    late = {**GRID, "beats": [0.3 + i * 0.5 for i in range(59)], "downbeats": [0.3 + i * 2.0 for i in range(14)]}
    # the first cut is the first downbeat at least half a bar in: 2.3, not 0.3
    assert spans(roughcut.plan(clips, late, "bar"))[:2] == [(0.0, 2.2917), (2.2917, 2.0)]


def test_a_short_clip_keeps_its_length_and_the_next_cut_is_on_the_grid_again():
    clips = [{"path": "a.mp4", "seconds": 8}, {"path": "b.mp4", "seconds": 1.25}, {"path": "c.mp4", "seconds": 8}]
    laid = roughcut.plan(clips, GRID, "bar")
    assert spans(laid) == [(0.0, 2.0), (2.0, 1.25), (3.25, 2.75)]
    assert [c["short"] for c in laid["timeline"]] == [False, True, False]
    assert laid["timeline"][1]["in"] == 0.0 and laid["timeline"][2]["at"] + 2.75 == pytest.approx(6.0)  # the downbeat at 4 is too near


def test_natural_cuts_on_the_last_beat_a_clip_reaches_and_the_song_end_drops_the_rest():
    laid = roughcut.plan([{"path": "a.mp4", "seconds": 3.7}, {"path": "b.mp4", "seconds": 2.2}], GRID, "natural")
    assert spans(laid) == [(0.0, 3.5), (3.5, 2.0)]
    short_song = {**GRID, "duration": 5.0}
    laid = roughcut.plan([{"path": f"{n}.mp4", "seconds": 8} for n in "abcd"], short_song, "bar")
    assert spans(laid) == [(0.0, 2.0), (2.0, 2.0), (4.0, 1.0)] and laid["dropped"] == ["d.mp4"] and laid["seconds"] == 5.0


def test_every_boundary_is_a_whole_frame():
    odd = {"bpm": 117.0, "beats": [0.11 + i * 60 / 117 for i in range(100)], "downbeats": [0.11 + i * 240 / 117 for i in range(25)], "duration": 50.0}
    laid = roughcut.plan([{"path": f"{n}.mp4", "seconds": 7.3} for n in range(12)], odd, "bar")
    for c in laid["timeline"]:
        for value in (c["at"], c["in"], c["out"] - c["in"]):
            assert abs(value * 24 - round(value * 24)) < 1e-3
        cut = c["at"] + c["out"] - c["in"]
        assert min(abs(cut - d) for d in odd["downbeats"]) < 1 / 24 or c is laid["timeline"][-1]


# ---------- the route, with the queue stood in for ----------

def make_clip(path: Path, seconds: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([beats.ffmpeg(), "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size=64x48:rate=24:duration={seconds}",
                    "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", str(path)], check=True, timeout=60)


@pytest.fixture
def stand_in(monkeypatch):
    """The queue: every job is recorded and done at once (an export's file is its first clip, a foley's a copy)."""

    pytest.importorskip("numpy")
    if not beats.ffmpeg():
        pytest.skip("ffmpeg is needed")
    sent = []

    def queue(body):
        job_id = f"rc-{len(sent)}-{time.time_ns()}"
        sent.append(body)
        params = body["params"]
        if body["kind"] == "foley":
            out = HOME / "sound" / "foley" / f"{job_id}.mp4"
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(HOME / params["video"], out)
        else:
            out = HOME / "exports" / f"{job_id}.mp4"
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(HOME / params["clips"][0]["path"], out)
        job = {"id": job_id, "kind": body["kind"], "status": "done", "title": body.get("title"), "params": params,
               "outputs": [str(out.relative_to(HOME))], "progress": None, "error": None, "log": "", "created": time.time()}
        with server.lock:
            server.jobs[job_id] = job
        return dict(job)
    monkeypatch.setattr(roughcut, "queue", queue)
    monkeypatch.setattr(roughcut, "POLL_S", 0.02)
    (HOME / "music").mkdir(parents=True, exist_ok=True)
    click_track(HOME / "music" / "rc-song.wav", 120, seconds=20, start=0.0)
    (HOME / "music" / "rc-song.take.json").write_text(json.dumps({"id": "rc-song", "path": "music/rc-song.wav", "title": "Click song", "created": 1}))
    for name, seconds in (("one", 5), ("two", 1.5), ("three", 4)):
        if not (HOME / "renders" / f"rc-{name}.mp4").is_file():
            make_clip(HOME / "renders" / f"rc-{name}.mp4", seconds)
    return sent


def finished(project, seconds=30):
    ends = time.time() + seconds
    while time.time() < ends:
        state = client.get(f"/api/roughcut/{project}").json()
        if state["stage"] in ("done", "failed"):
            return state
        time.sleep(0.05)
    raise AssertionError(state)


def write_film(film_id: str) -> None:
    folder = HOME / "slate" / film_id
    folder.mkdir(parents=True, exist_ok=True)
    take = lambda path, rating: {"id": path, "rating": rating, "mediaPath": f"renders/rc-{path}.mp4"}
    film = {"id": film_id, "name": "Night Bus", "scenes": [
        {"id": "s1", "name": "Stop", "shots": [
            {"id": "a", "takes": [take("three", "no-good"), take("one", "good")]},  # the good one, not the newer no-good
            {"id": "b", "takes": [take("three", "good"), take("two", "circled")]}]},  # circled beats good
        {"id": "s2", "name": "Ride", "shots": [
            {"id": "c", "takes": [{"id": "x", "rating": "good"}, take("three", None)]},  # newest with a file, unrated
            {"id": "d", "takes": [take("one", "no-good")]}]}]}  # no usable take: left out
    (folder / "project.json").write_text(json.dumps(film))


def test_a_film_is_cut_on_the_bar_exported_with_its_song_and_written_for_resolve(stand_in):
    write_film("rc-film")
    made = client.post("/api/roughcut", json={"film": "rc-film", "song": "music/rc-song.wav", "cut": "bar", "width": 1280, "height": 720}).json()
    assert made["title"] == "Rough cut of Night Bus" and abs(made["beats"]["bpm"] - 120) < 2 and made["foley"] == []
    assert [c["path"] for c in made["timeline"]] == ["renders/rc-one.mp4", "renders/rc-two.mp4", "renders/rc-three.mp4"]
    assert [c["short"] for c in made["timeline"]] == [False, True, False]  # the 1.5 s clip keeps its length
    state = finished(made["project"])
    assert state["stage"] == "done", state
    export = stand_in[-1]
    assert export["kind"] == "export" and export["params"]["width"] == 1280
    assert export["params"]["soundtrack"] == {"take": "rc-song", "music_db": -10.0}  # a Music take: the export lays it
    assert export["params"]["clips"][0]["out"] - export["params"]["clips"][0]["in"] == pytest.approx(2.0, abs=0.05)
    project = client.get(f"/api/projects/{made['project']}").json()
    assert project["name"] == "Rough cut of Night Bus" and len(project["timeline"]) == 3 and project["soundtrack"]["take"] == "rc-song"
    files = state["files"]
    xml = (HOME / files["fcpxml"]).read_text()
    assert "<fcpxml" in xml and 'id="rs"' in xml and "rc-song" in xml and "rc-one" in xml
    assert "rc-two.mp4" in (HOME / files["edl"]).read_text()
    with zipfile.ZipFile(HOME / files["zip"]) as bundle:
        names = bundle.namelist()
    assert any(n.endswith(".fcpxml") for n in names) and any(n.endswith("media/rc-song.wav") for n in names)
    assert any(n.endswith("media/rc-one.mp4") for n in names)
    assert state["video"].startswith("exports/")


def test_a_trailer_from_library_clips_with_foley_uses_the_foley_clips(stand_in):
    made = client.post("/api/roughcut", json={"clips": ["renders/rc-one.mp4", "renders/rc-three.mp4"], "song": "rc-song", "cut": "beat",
                                              "foley": True, "order": "shuffle", "seed": 3}).json()
    assert made["title"] == "Trailer" and len(made["foley"]) == 2 and made["job"] is None
    state = finished(made["project"])
    assert state["stage"] == "done", state
    foley = [b for b in stand_in if b["kind"] == "foley"]
    assert {b["params"]["video"] for b in foley} == {"renders/rc-one.mp4", "renders/rc-three.mp4"} and all(b["params"]["keep_sound"] for b in foley)
    export = stand_in[-1]
    assert export["kind"] == "export" and all(c["path"].startswith("sound/foley/") for c in export["params"]["clips"])
    assert all(c["out"] - c["in"] == pytest.approx(0.5, abs=0.05) for c in export["params"]["clips"])


def test_a_voice_take_or_sound_file_is_laid_under_the_finished_export(stand_in):
    (HOME / "voice").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(HOME / "music" / "rc-song.wav", HOME / "voice" / "rc-line.wav")
    made = client.post("/api/roughcut", json={"clips": ["renders/rc-one.mp4"], "song": "voice/rc-line.wav", "cut": "two-bars"}).json()
    state = finished(made["project"])
    assert state["stage"] == "done", state
    assert "soundtrack" not in stand_in[-1]["params"]  # not a Music take: laid under the export here, after it
    assert "rc-line" in (HOME / state["files"]["fcpxml"]).read_text()


def test_the_route_refuses_what_it_cannot_cut(stand_in):
    say = lambda body: client.post("/api/roughcut", json=body)
    assert say({"song": "rc-song"}).status_code == 400
    assert say({"clips": ["renders/rc-one.mp4"], "song": "rc-song", "cut": "every-frame"}).status_code == 400
    assert say({"clips": ["renders/rc-one.mp4"], "song": "rc-song", "order": "random"}).status_code == 400
    assert say({"clips": ["music/rc-song.wav"], "song": "rc-song"}).status_code == 400
    assert say({"clips": ["renders/rc-one.mp4"], "song": "music/none.wav"}).status_code == 404
    assert say({"film": "no-such-film", "song": "rc-song"}).status_code == 404
    assert say({"film": "x", "clips": ["renders/rc-one.mp4"], "song": "rc-song"}).status_code == 400
    (HOME / "slate" / "rc-empty").mkdir(parents=True, exist_ok=True)
    (HOME / "slate" / "rc-empty" / "project.json").write_text(json.dumps({"id": "rc-empty", "scenes": [{"shots": [{"takes": []}]}]}))
    assert "no shot" in say({"film": "rc-empty", "song": "rc-song"}).json()["error"]
    assert client.get("/api/roughcut/p-none").status_code == 404


# ---------- the agents' tools ----------

def test_the_agents_tools_cut_to_a_song_and_follow_it(monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("studio_mcp_rough", ROOT / "mcp/studio-mcp.py")
    mcp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mcp)
    sent = []

    def call(method, path, body=None, raw=None, kind="application/json"):
        sent.append((method, path, body))
        if path == "/api/roughcut":
            return {"project": "p1", "job": "j1", "foley": [], "beats": {"bpm": 120}, "title": "T", "dropped": [], "timeline": [{}, {}]}
        if path.startswith("/api/roughcut/"):
            return {"project": "p1", "stage": "done", "files": {"zip": "exports/t.zip"}, "jobs": {"j1": {}}}
        return {"bpm": 120.0}
    monkeypatch.setattr(mcp, "call", call)
    answer = lambda name, **args: mcp.answer({"method": "tools/call", "params": {"name": name, "arguments": args}})
    listed = {t["name"]: t for t in mcp.answer({"method": "tools/list"})["tools"]}
    assert listed["rough_cut"]["inputSchema"]["required"] == ["song"] and listed["music_beats"]["inputSchema"]["required"] == ["path"]
    assert json.loads(answer("music_beats", path="music/a.wav")["content"][0]["text"])["bpm"] == 120.0
    made = json.loads(answer("rough_cut", film="f1", song="music/a.wav", cut="bar", foley=False)["content"][0]["text"])
    assert sent[-1] == ("POST", "/api/roughcut", {"film": "f1", "song": "music/a.wav", "cut": "bar", "foley": False}) and made["cuts"] == 2
    assert answer("rough_cut", song="x")["isError"] and answer("rough_cut", film="f1", clips=["a.mp4"], song="x")["isError"]
    status = json.loads(answer("rough_cut_status", project="p1")["content"][0]["text"])
    assert status["stage"] == "done" and status["files"]["zip"] == "exports/t.zip"
    import agents
    assert all(name in agents.STUDIO_TOOLS for name in ("music_beats", "rough_cut", "rough_cut_status")) and "rough_cut" in agents.MAKERS
