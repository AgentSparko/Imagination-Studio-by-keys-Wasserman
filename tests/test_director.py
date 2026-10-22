"""The Director (app/director.py): a logline to a written, cast, rendered, checked and cut film. A scripted brain, a
queue whose jobs finish at once and a stand-in QA gate: no GPU, no network, no model."""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("STUDIO_HOME", str(Path(tempfile.mkdtemp()) / "ImaginationStudio"))
sys.path.insert(0, str(ROOT / "app"))
from fastapi.testclient import TestClient  # noqa: E402

import director  # noqa: E402
import server  # noqa: E402  (attaches Slate and the Director)
import slate  # noqa: E402

from test_slate_page import HARNESS, need_node  # noqa: E402

HOME = server.HOME
client = TestClient(server.app)

BIBLE = {"world": "A small coastal town in late autumn, grey light, wet stone.",
         "characters": [{"name": "Ada", "age": "34", "gender": "female", "hair": "short dark hair", "clothing": "a yellow raincoat",
                         "voice": {"ageGender": "a woman in her thirties", "timbre": "warm"}},
                        {"name": "Bo", "age": "60", "gender": "male", "hair": "grey beard", "clothing": "a fisherman's sweater"}],
         "locations": [{"name": "The Harbour", "interiorExterior": "exterior", "description": "A stone harbour with small boats", "timeOfDay": "dawn"}]}
SCENES = {"scenes": [{"name": "The Letter", "synopsis": "Ada finds a letter in a boat."}, {"name": "The Answer", "synopsis": "Bo reads it."}]}


def shots(bad: bool = False, later: bool = False) -> dict:
    """Two shots of a scene; the second scene's are Bo's."""

    who = "Bo" if later else "Ada"
    cast = [who, "Nobody"] if bad else [who]
    act = "reads the letter by the boats" if later else "kneels and lifts a letter from a boat"
    return {"shots": [{"name": "Wide", "intent": "Where we are", "prompt": f"# Subject\n{who} walks along the harbour wall.\n# Camera\nA slow track.",
                       "size": "WS", "movement": "track", "cast": cast, "location": "The Harbour", "lines": [], "soundscape": "Gulls and waves."},
                      {"name": "Close", "intent": "She sees it", "prompt": f"# Subject\n{who} {act}.",
                       "size": "CU", "cast": [who], "location": "The Harbour", "lines": [{"speaker": who, "text": "Who left this here?"}]}]}


class Brain:
    """Answers by what is asked; the first shots answer of a run can be one that does not hold up."""

    def __init__(self, bad_first_shots: bool = False):
        self.asked, self.bad = [], bad_first_shots

    def __call__(self, backend, system, prompt, tier, endpoint="", model=""):
        self.asked.append(prompt)
        if "Write the characters" in prompt or "Write the locations" in prompt:
            return "Here you go:\n```json\n" + json.dumps(BIBLE) + "\n```"
        if "Break this film into" in prompt:
            return json.dumps(SCENES)
        if "Write exactly" in prompt:
            bad, self.bad = self.bad, False
            return json.dumps(shots(bad, later="CURRENT SCENE: The Answer" in prompt))
        raise AssertionError("an unexpected question: " + prompt[:200])


class Queue:
    """The Studio's queue stood in for: a job is done when queued (its files written), unless ``hold`` says otherwise."""

    def __init__(self, fail_qa="", hold=False):
        self.jobs, self.bodies, self.fail_qa, self.hold, self.checked, self.made = {}, [], fail_qa, hold, [], []

    def submit(self, body):
        kind, params = body["kind"], body.get("params") or {}
        job_id = f"j{len(self.jobs) + 1}"
        if kind == "video":
            outputs = [params["output"]]
        elif kind == "scout":
            outputs = [f"scouts/{job_id}/scout_1.png", f"scouts/{job_id}/sheet.jpg"]
        elif kind == "turnaround":
            outputs = [f"turns/{job_id}/turn_sheet.png", f"turns/{job_id}/turn_view1.png", f"turns/{job_id}/other.txt"]
        elif kind == "export":
            outputs = [f"exports/{job_id}.mp4"]
        else:
            raise AssertionError(kind)
        for path in outputs:
            (HOME / path).parent.mkdir(parents=True, exist_ok=True)
            (HOME / path).write_bytes(b"x")
        job = {"id": job_id, "kind": kind, "status": "queued" if self.hold else "done", "params": params, "outputs": outputs, "title": body.get("title")}
        self.jobs[job_id] = job
        self.bodies.append(body)
        self.made.append(job)
        return job

    def lookup(self, job_id):
        return self.jobs.get(job_id)

    def cancel(self, job_id):
        if job_id in self.jobs:
            self.jobs[job_id]["status"] = "cancelled"

    def qa(self, body):
        """A QA job: the first clip whose prompt has the words of fail_qa fails."""

        self.checked.append(body["path"])
        job_id = f"q{len(self.jobs) + 1}"
        failed = bool(self.fail_qa) and self.fail_qa in body.get("prompt", "")
        if failed:
            self.fail_qa = ""
        self.jobs[job_id] = {"id": job_id, "kind": "qa", "status": "done", "qa": {"passed": not failed, "takes": [{"passed": not failed, "reason": "the face changes" if failed else ""}]}}
        return self.jobs[job_id]

    def videos(self):
        return [b for b in self.bodies if b["kind"] == "video"]


@pytest.fixture
def studio(monkeypatch):
    monkeypatch.setattr(director, "TICK", 0.01)
    monkeypatch.setattr(director, "ref2va", lambda: False)
    made = {}
    monkeypatch.setattr(director, "editor_new", lambda body: made.setdefault("edit", {"id": "pcut-1", "name": body["name"], "timeline": []}))
    monkeypatch.setattr(director, "editor_save", lambda pid, data: made.update(saved=json.loads(json.dumps(data))))
    director.runs.clear()

    def use(queue: Queue, brain: Brain):
        monkeypatch.setattr(director, "submit", queue.submit)
        monkeypatch.setattr(director, "job_of", queue.lookup)
        monkeypatch.setattr(director, "cancel", queue.cancel)
        monkeypatch.setattr(director, "qa_start", queue.qa)
        monkeypatch.setattr(slate, "ask", brain)
        return made
    return use


def new_film(logline="A harbour keeper finds a letter that was never sent.") -> str:
    film = client.post("/api/slate/projects", json={"name": "The Harbour"}).json()
    if logline:
        film["logline"] = logline
        client.put(f"/api/slate/projects/{film['id']}", json=film)
    return film["id"]


def until(project: str, states=("done", "failed", "stopped"), seconds=20) -> dict:
    end = time.time() + seconds
    while time.time() < end:
        run = client.get(f"/api/slate/direct/{project}").json()
        if run.get("state") in states and not run.get("running"):
            return run
        time.sleep(0.02)
    raise AssertionError(f"the run did not reach {states}: {run}")


def test_a_logline_becomes_a_checked_cut_film(studio):
    queue, brain = Queue(fail_qa="lifts a letter"), Brain(bad_first_shots=True)
    made = studio(queue, brain)
    project = new_film()
    started = client.post("/api/slate/direct", json={"project": project, "scenes": 2, "shotsPerScene": 2, "seconds": 5, "retakes": 1, "cut": True})
    assert started.status_code == 200, started.text
    run = until(project)
    assert run["state"] == "done", run
    film = slate.get_project(project)

    # writing: the bible, the scenes and their shots, through the First AD's actions
    assert [c["name"] for c in film["characters"]] == ["Ada", "Bo"] and film["locations"][0]["name"] == "The Harbour"
    assert film["world"].startswith("A small coastal town") and film["voices"][0]["characterId"] == film["characters"][0]["id"]
    assert [s["name"] for s in film["scenes"]] == ["The Letter", "The Answer"] and all(len(s["shots"]) == 2 for s in film["scenes"])
    first, second = film["scenes"][0]["shots"]
    assert first["tf"]["start"] == "text" and second["tf"]["start"] == "previous"  # a cut, then one continuous move
    assert first["tf"]["cast"] == [film["characters"][0]["id"]] and first["tf"]["location"] == film["locations"][0]["id"]
    assert second["tf"]["lines"][0]["text"] == "Who left this here?" and second["tf"]["lines"][0]["characterId"] == film["characters"][0]["id"]
    assert first["spec"]["durationSec"] == 5 and first["spec"]["size"] == "WS"
    # the shots answer that cast someone not in the bible was asked for again, once, with the reason
    again = [p for p in brain.asked if "YOUR LAST ANSWER COULD NOT BE USED" in p]
    assert len(again) == 1 and "Nobody" in again[0]

    # casting: a reference still and its turnaround for each character, a plate for the place, all pinned
    kinds = [b["kind"] for b in queue.bodies]
    assert kinds.count("scout") == 3 and kinds.count("turnaround") == 2
    ada = film["characters"][0]
    assert ada["images"][0].endswith("scout_1.png") and any(p.endswith("turn_sheet.png") for p in ada["images"])
    assert not any(p.endswith(("sheet.jpg", "other.txt")) for p in ada["images"])
    assert len(film["locations"][0]["images"]) == 1
    turn = next(b for b in queue.bodies if b["kind"] == "turnaround")
    assert turn["params"]["image"].endswith("scout_1.png")

    # rendering: every shot once, and the take that failed its check once more at the next seed
    videos = queue.videos()
    assert len(videos) == 5 and all(v["params"]["slate"]["project"] == project for v in videos)
    of = lambda shot: [v for v in videos if v["params"]["slate"]["shot"] == shot["id"]]
    order = [s for sc in film["scenes"] for s in sc["shots"]]
    assert [len(of(s)) for s in order] == [1, 2, 1, 1]
    assert of(second)[1]["params"]["seed"] == of(second)[0]["params"]["seed"] + 1
    # the shot that opens on the shot before was queued once that one had its take, from its clip, and so was its retake
    assert {v["params"]["first_frame_from"] for v in of(second)} == {of(first)[0]["params"]["output"]}
    takes = second["takes"]
    assert len(takes) == 2 and [t["rating"] for t in takes] == ["good", "no-good"]  # newest first: the retake is the good one
    assert takes[0]["seed"] == takes[1]["seed"] + 1 and takes[0]["origin"].startswith("The Director: retake")
    assert all([t["rating"] for t in s["takes"]] == ["good"] for s in order if s["id"] != second["id"])
    assert run["counts"]["retakes"] == 1 and run["counts"]["settled"] == 4 and run["counts"]["shots"] == 4

    # cutting: the best take of each shot, in order, on an Editor timeline and exported
    timeline = made["saved"]["timeline"]
    assert [c["path"] for c in timeline] == [of(s)[-1]["params"]["output"] for s in order]
    export = next(b for b in queue.bodies if b["kind"] == "export")
    assert export["params"]["clips"] == timeline
    files = run["cut"]["files"]
    assert files["mp4"].startswith("exports/") and files["fcpxml"] == "/api/export/timeline/pcut-1.fcpxml" and files["zip"].endswith(".zip")
    assert any("Done:" in line["text"] for line in run["log"])


def test_a_restart_continues_and_a_stop_stops(studio):
    queue, brain = Queue(hold=True), Brain()
    studio(queue, brain)
    project = new_film()
    assert client.post("/api/slate/direct", json={"project": project, "scenes": 1, "shotsPerScene": 2, "seconds": 5, "cut": False}).status_code == 200
    end = time.time() + 10
    while time.time() < end and not any(b["kind"] == "scout" for b in queue.bodies):
        time.sleep(0.02)
    # the pictures are made (held in the queue until now)
    for job in list(queue.jobs.values()):
        job["status"] = "done"
    while time.time() < end and len(queue.videos()) < 1:
        for job in queue.jobs.values():
            if job["kind"] != "video":
                job["status"] = "done"
        time.sleep(0.02)
    assert client.post("/api/slate/direct", json={"project": project}).status_code == 409  # one run a film
    assert queue.videos()  # the first shot is queued; the next opens on the shot before and waits for it
    # stopped on request: the jobs still waiting are taken out of the queue
    assert client.post(f"/api/slate/direct/{project}/stop", json={}).json()["stopped"] is True
    run = until(project)
    assert run["state"] == "stopped" and [j["status"] for j in queue.made if j["kind"] == "video"] == ["cancelled"]
    asked = len(brain.asked)

    # a restart: the queue's memory is gone, the file says the run was rendering, and the first clip got made
    status = director.status_file(project)
    saved = json.loads(status.read_text())
    saved["state"] = "rendering"
    status.write_text(json.dumps(saved))
    first = queue.videos()[0]["params"]["output"]
    assert (HOME / first).is_file()
    director.runs.clear()
    fresh = Queue()
    studio(fresh, brain)
    director.resume_all()
    run = until(project)
    assert run["state"] == "done", run
    assert len(brain.asked) == asked  # nothing written twice
    assert not any(b["kind"] in ("scout", "turnaround") for b in fresh.bodies)  # nor cast twice
    film = slate.get_project(project)
    shots_ = film["scenes"][0]["shots"]
    # the clip made before the restart is kept as the first shot's take; only the other is rendered, from it
    assert len(queue.videos()) == 1 and shots_[0]["takes"][0]["mediaPath"] == first
    assert [v["params"]["slate"]["shot"] for v in fresh.videos()] == [shots_[1]["id"]] and fresh.videos()[0]["params"]["first_frame_from"] == first
    assert all(s["takes"][0]["rating"] == "good" for s in shots_)
    assert any("Picked up after a restart" in line["text"] for line in run["log"])


def test_the_director_asks_for_a_logline_and_keeps_rendered_shots(studio):
    queue, brain = Queue(), Brain()
    studio(queue, brain)
    bare = new_film(logline="")
    assert client.post("/api/slate/direct", json={"project": bare}).status_code == 400
    assert client.post("/api/slate/direct", json={"project": "nope"}).status_code == 404
    assert client.get(f"/api/slate/direct/{bare}").json()["state"] == "none"
    # a film the First AD already laid out: its scene and shot stay, its rendered take is kept as it is
    project = new_film()
    film = slate.get_project(project)
    for action in ({"type": "add_character", "name": "Ada"}, {"type": "add_location", "name": "The Harbour", "description": "stone"},
                   {"type": "create_scene", "name": "One"}, {"type": "create_shot", "scene": "One", "name": "Made", "prompt": "# Subject\nWaves."},
                   {"type": "create_shot", "scene": "One", "name": "New", "prompt": "# Subject\nGulls.", "opensOn": "previous"}):
        assert "✓" in director.apply(film, action)
    film["characters"][0]["images"] = ["uploads/ada.png"]
    film["locations"][0]["images"] = ["uploads/harbour.png"]
    (HOME / "uploads").mkdir(exist_ok=True)
    (HOME / "takes-made.mp4").write_bytes(b"x")
    film["scenes"][0]["shots"][0]["takes"] = [{"id": "take-old", "mediaPath": "takes-made.mp4", "rating": "circled"}]
    slate.write(film)
    assert client.post("/api/slate/direct", json={"project": project, "cut": False}).status_code == 200
    run = until(project)
    assert run["state"] == "done", run
    assert brain.asked == [] and queue.bodies and all(b["kind"] == "video" for b in queue.bodies)
    assert len(queue.videos()) == 1 and queue.videos()[0]["params"]["first_frame_from"] == "takes-made.mp4"
    film = slate.get_project(project)
    assert film["scenes"][0]["shots"][0]["takes"] == [{"id": "take-old", "mediaPath": "takes-made.mp4", "rating": "circled"}]


def test_apply_mirrors_the_first_ads_actions():
    film = {"id": "p-x", "defaults": {"durationSec": 8}, "scenes": [], "characters": [], "locations": []}
    assert "✓ Added character" in director.apply(film, {"type": "add_character", "name": "Ada", "hair": "short"})
    assert "Changed character" in director.apply(film, {"type": "update_character", "character": "ada", "clothing": "a raincoat"})
    assert film["characters"][0]["hair"] == "short" and film["characters"][0]["clothing"] == "a raincoat" and len(film["characters"]) == 1
    assert film["characters"][0]["keyLightSide"] == "Key light from left"
    assert "✗ Scene" in director.apply(film, {"type": "create_shot", "scene": "nowhere"})
    director.apply(film, {"type": "create_scene", "name": "A"})
    assert "already exists" in director.apply(film, {"type": "create_scene", "name": "a"})
    director.apply(film, {"type": "create_shot", "scene": "A", "cast": ["Ada"], "lines": [{"speaker": "Ada", "text": " Hello. "}], "opensOn": "previous", "bars": 2})
    shot = film["scenes"][0]["shots"][0]
    assert shot["name"] == "Shot 01" and shot["tf"]["start"] == "previous" and shot["tf"]["bars"] == 2
    assert shot["tf"]["lines"] == [{"characterId": film["characters"][0]["id"], "voiceId": None, "text": "Hello.", "language": "English", "sung": False}]
    assert "Updated" in director.apply(film, {"type": "update_shot", "shot": "Shot 01", "prompt": "# Subject\nNew."})
    assert "✗ Unknown action" in director.apply(film, {"type": "dance"})


def test_the_panel_in_the_first_ad_tab():
    need_node()
    import subprocess

    code = r"""
    SL.p = film(); SL.p.logline = "A keeper finds a letter."; SL.view = "ad";
    answers["/api/music/takes"] = [{ id: "song-1", title: "Low Tide" }];
    DR.songs = [{ id: "song-1", title: "Low Tide" }];
    DR.st[SL.p.id] = { state: "none" };
    out.idle = text(slADView());
    out.button = !!(function find(n) { return n && typeof n === "object" && (n.tag === "button" && text(n).includes("Direct it") || (n.kids || []).flat(9).some(find)); })(slADView());
    globalThis.jobs = [{ id: "j7", status: "running" }];
    DR.st[SL.p.id] = { state: "rendering", running: true, done: ["writing", "casting"], asked: { cut: true }, counts: { shots: 9, rendered: 7, settled: 6, retakes: 2, cast: 3, castDone: 3 },
      current: { text: "Rendering: One · Close", job: "j7" }, log: [{ at: "2026-01-01T10:00:00.000Z", text: "The Director starts" }] };
    out.running = text(directorPanel(SL.p));
    DR.st[SL.p.id] = { state: "done", running: false, done: ["writing", "casting", "rendering", "cutting"], asked: { cut: true }, counts: { shots: 9, rendered: 9, settled: 9, retakes: 2 },
      cut: { files: { mp4: "exports/x.mp4", fcpxml: "/api/export/timeline/p1.fcpxml", edl: "/api/export/timeline/p1.edl", zip: "/api/export/timeline/p1.zip" } }, log: [] };
    out.done = text(directorPanel(SL.p));
    calls.length = 0; DR.st[SL.p.id] = { state: "none" }; await drStart(SL.p);
    out.posted = calls.find(c => c[0] === "/api/slate/direct");
    """
    with tempfile.TemporaryDirectory() as scratch:
        load = 'vm.runInThisContext(fs.readFileSync(process.argv[2], "utf8"));\n'
        assert HARNESS.count(load) == 1
        (Path(scratch) / "harness.js").write_text(HARNESS.replace(load, load + 'vm.runInThisContext(fs.readFileSync(process.argv[4], "utf8"));\n'))
        (Path(scratch) / "test.js").write_text(code)
        done = subprocess.run(["node", str(Path(scratch) / "harness.js"), str(ROOT / "app/static/slate.js"), str(Path(scratch) / "test.js"),
                               str(ROOT / "app/static/director.js")], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr[-2000:]
    out = json.loads(done.stdout.strip().splitlines()[-1])
    assert "Direct the whole film" in out["idle"] and "A keeper finds a letter." in out["idle"] and out["button"]
    assert "Cut it when done" in out["idle"] and "One retake" in out["idle"] and "Low Tide" in out["idle"]
    assert "7 of 9 shots rendered" in out["running"] and "2 retakes" in out["running"] and "job j7" in out["running"]
    assert "Stop" in out["running"] and "Rendering: One · Close" in out["running"] and "The log · 1 lines" in out["running"]
    assert "Direct it" not in out["running"]  # the form is put away while it runs
    assert "Watch the cut" in out["done"] and "FCPXML" in out["done"] and "Review takes" in out["done"]
    posted = out["posted"][1]
    assert posted["project"] == "p-test" and posted["logline"] == "A keeper finds a letter." and posted["scenes"] == 3 and posted["cut"] is True and posted["retakes"] == 1
