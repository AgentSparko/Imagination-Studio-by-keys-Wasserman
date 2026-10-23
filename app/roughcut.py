"""A rough cut on the beat: a film's best takes (or a Library selection) cut to a song, exported, and handed to Resolve.

``POST /api/roughcut`` takes the clips (a Slate film's best take of each shot in scene and shot order, or the paths
given), measures the song's beats (beats.py), and lays the clips end to end so that every cut falls on the song's
grid: each clip fills one slot of a beat, a bar or two bars (trimmed around its middle, or from its start), or with
``natural`` runs about its own length and is cut on the last beat it reaches. A clip shorter than its slot keeps its
whole length and the next clip is cut on the grid again. The cut becomes an Editor project, its export job is queued
with the song under it, and when the export is done the project's FCPXML, EDL and a zip with its media for DaVinci
Resolve are written to exports/. With ``foley`` each clip of 15 s or less first gets its sound made for its picture
(a foley job each), and the cut uses those clips.

``GET /api/roughcut/{project}`` says where it is: foley, export, song, files, done or failed, with the job ids.
"""

from __future__ import annotations

import json
import random
import re
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

from fastapi import APIRouter, HTTPException

import beats
import slate
import suite

FPS = 24
CUTS = ("beat", "bar", "two-bars", "natural")
ORDERS = ("given", "shuffle", "energy")
VIDEO = (".mp4", ".mov")
FOLEY_LONGEST, FOLEY_SHORTEST = 15.0, 1.0
MOST_CLIPS = 200
POLL_S = 1.0
FINISHED = ("done", "failed", "cancelled")

router = APIRouter()
HOME: Path = Path.home() / "ImaginationStudio"
queue = jobs = lock = inside = relative = media_seconds = public = None
SPARK = sys.platform.startswith("linux")
FOLLOWING: dict[str, threading.Thread] = {}


# ---------- the cut: where each clip starts and ends on the song's grid ----------

def grid(found: dict, cut: str) -> list[float]:
    """The song times a cut may fall on: every beat, every bar's first beat, or every other one."""

    if cut in ("beat", "natural"):
        return list(found.get("beats") or [])
    downbeats = list(found.get("downbeats") or [])
    return downbeats[::2] if cut == "two-bars" else downbeats


def plan(clips: list[dict], found: dict, cut: str = "natural", align: str = "centre", fps: int = FPS) -> dict:
    """The timeline of a rough cut: ``clips`` are [{path, seconds}] in order, ``found`` a song's beats (beats.py).

    Song time and timeline time are the same (the song starts with the first clip), and every boundary is a whole
    frame of the timeline, so no rounding adds up along the cut. Returns {"timeline": [{path, in, out, duration, at,
    short}], "dropped": [paths after the song ended], "seconds", "slot"}."""

    if cut not in CUTS:
        raise ValueError(f"cut is one of {', '.join(CUTS)}")
    points = grid(found, cut)
    beat_list = list(found.get("beats") or [])
    beat = (60.0 / float(found["bpm"])) if found.get("bpm") else (beat_list[1] - beat_list[0] if len(beat_list) > 1 else 0.5)
    gaps = [b - a for a, b in zip(points, points[1:])]
    slot = sorted(gaps)[len(gaps) // 2] if gaps else beat * {"beat": 1, "natural": 1, "bar": 4, "two-bars": 8}[cut]
    song_end = float(found.get("duration") or (points[-1] if points else 0))
    timeline, dropped, at_f = [], [], 0
    for clip in clips:
        length = float(clip.get("seconds") or 0) or 5.0
        at = at_f / fps
        if at >= song_end - 0.25:
            dropped.append(clip["path"])
            continue
        if cut == "natural":
            reach = min(at + length, song_end)
            on = [p for p in beat_list if at + 0.5 * beat <= p <= reach + 1e-6]
            end = on[-1] if on else reach
        else:
            end = next((p for p in points if p >= at + 0.5 * slot), song_end)
        end_f = round(min(end, song_end) * fps)
        span_f, own_f = end_f - at_f, int(length * fps + 1e-6)
        short = own_f < span_f
        if short:
            span_f = own_f  # shorter than its slot: the whole clip, and the next cut lands on the grid again
        if span_f <= 0:
            dropped.append(clip["path"])
            continue
        spare_f = max(0, own_f - span_f)
        in_f = spare_f // 2 if align == "centre" else 0
        timeline.append({"path": clip["path"], "in": round(in_f / fps, 6), "out": round((in_f + span_f) / fps, 6),
                         "duration": round(length, 4), "at": round(at_f / fps, 6), "short": short})
        at_f += span_f
    return {"timeline": timeline, "dropped": dropped, "seconds": round(at_f / fps, 6), "slot": round(slot, 4)}


def motion_energy(path: Path) -> float:
    """How much a clip moves: the mean change between small grey frames, six a second (0 when it cannot be read)."""

    import numpy as np

    tool = beats.ffmpeg()
    if not tool:
        return 0.0
    try:
        done = subprocess.run([tool, "-v", "error", "-i", str(path), "-vf", "fps=6,scale=32:18,format=gray", "-f", "rawvideo", "pipe:1"],
                              capture_output=True, timeout=60)
    except subprocess.TimeoutExpired:
        return 0.0
    frames = np.frombuffer(done.stdout, np.uint8)
    frames = frames[:len(frames) // 576 * 576].reshape(-1, 576).astype(np.float32)
    return float(np.abs(np.diff(frames, axis=0)).mean()) if len(frames) > 1 else 0.0


def ordered(clips: list[dict], order: str, seed: int) -> list[dict]:
    if order == "shuffle":
        out = list(clips)
        random.Random(seed).shuffle(out)
        return out
    if order == "energy":  # calm to busy: the cut builds with the song
        return sorted(clips, key=lambda clip: motion_energy(inside(clip["path"])))
    return list(clips)


# ---------- what is cut, and to what ----------

def film_clips(film_id: str) -> tuple[str, list[str]]:
    """(the film's name, its best take of each shot in scene and shot order: circled, else good, else the newest not
    marked no-good)."""

    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", film_id):
        raise HTTPException(400, "bad film id")
    file = HOME / "slate" / film_id / "project.json"
    if not file.is_file():
        raise HTTPException(404, "no such film")
    try:
        film = json.loads(file.read_text())
    except (OSError, ValueError):
        raise HTTPException(400, "the film's file could not be read")
    paths = []
    for scene in film.get("scenes") or []:
        for shot in scene.get("shots") or []:
            takes = shot.get("takes") or []
            circled = next((t["mediaPath"] for t in takes if t.get("mediaPath") and t.get("rating") == "circled"), None)
            good = next((t["mediaPath"] for t in takes if t.get("mediaPath") and t.get("rating") == "good"), None)
            best = circled or good or slate.best_take(shot)
            if best and (HOME / best).is_file():
                paths.append(best)
    if not paths:
        raise HTTPException(400, "no shot of this film has a finished take yet: render the shots first")
    return str(film.get("name") or film_id), paths


def song_of(said: str) -> dict:
    """The song: a Music take's id or path (laid by the export itself), or a voice take or any sound file of the
    studio folder (laid under the export when it is done). {path, take, title}."""

    target = beats.resolve(said)
    path = relative(target)
    take = None
    for file in (HOME / "music").glob("*.take.json"):
        try:
            meta = json.loads(file.read_text())
        except (OSError, ValueError):
            continue
        if meta.get("path") == path and meta.get("id"):
            take = meta
            break
    title = (take or {}).get("title") or target.stem
    return {"path": path, "take": take["id"] if take else None, "title": str(title)[:80]}


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")[:48] or "rough-cut"


# ---------- its state, kept beside its project ----------

def state_file(project_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", project_id):
        raise HTTPException(400, "bad project id")
    return HOME / "projects" / project_id / "roughcut.json"


def save_state(state: dict) -> None:
    file = state_file(state["project"])
    fresh = file.with_suffix(".json.new")
    fresh.write_text(json.dumps(state, indent=1))
    fresh.replace(file)


def load_project(project_id: str) -> dict:
    return json.loads((HOME / "projects" / project_id / "project.json").read_text())


def new_project(name: str) -> dict:
    """An Editor project as POST /api/projects makes it, under an id no other project has (two in one second differ)."""

    stamp, n = time.strftime("p%m%d-%H%M%S"), 1
    project_id = stamp
    while (HOME / "projects" / project_id).exists():
        n += 1
        project_id = f"{stamp}-{n}"
    (HOME / "projects" / project_id).mkdir(parents=True)
    return {"id": project_id, "name": name, "preset": "small" if SPARK else "draft", "quality": "fh8" if SPARK else "standard",
            "segments": [], "timeline": []}


def save_project(project: dict) -> None:
    file = HOME / "projects" / project["id"] / "project.json"
    fresh = file.with_suffix(".json.new")
    fresh.write_text(json.dumps(project, indent=1))
    fresh.replace(file)


def job_of(job_id: str) -> dict | None:
    with lock:
        job = jobs.get(job_id)
        return dict(job) if job else None


def wait(ids: list[str]) -> dict[str, dict]:
    """These jobs once each is done, failed or cancelled (a job that left the queue counts as failed)."""

    out = {}
    while len(out) < len(ids):
        for job_id in ids:
            if job_id in out:
                continue
            job = job_of(job_id)
            if job is None:
                out[job_id] = {"id": job_id, "status": "failed", "error": "the job left the queue", "outputs": []}
            elif job.get("status") in FINISHED:
                out[job_id] = job
        if len(out) < len(ids):
            time.sleep(POLL_S)
    return out


def export_body(project: dict, state: dict) -> dict:
    params = {"clips": [{k: one[k] for k in ("path", "in", "out", "duration")} for one in project["timeline"]],
              "width": state["width"], "height": state["height"]}
    if state["song"]["take"]:
        params["soundtrack"] = {"take": state["song"]["take"], "music_db": state["music_db"]}
    return {"kind": "export", "title": "Rough cut: " + state["title"], "params": params}


def editor_files(project: dict, state: dict) -> dict:
    """The cut for an editor, in exports/: <name>.fcpxml and .edl, and <name>-resolve.zip (the FCPXML with every clip
    and the song in media/ beside it, which Resolve imports with File > Import > Timeline)."""

    clips, rate, size = [], None, None
    for one in project["timeline"]:
        source = inside(one["path"])
        seen = suite.probe(source)
        rate = rate or seen.get("rate")
        size = size or ((seen["width"], seen["height"]) if seen.get("width") else None)
        clips.append({"path": source, "in": one["in"], "out": one["out"], "duration": one.get("duration"),
                      "file_seconds": seen.get("seconds"), "audio": seen["audio"] if seen.get("video") else True})
    rate, size = rate or (FPS, 1), size or (state["width"], state["height"])
    song_path = inside(state["song"]["path"])
    song = {"path": song_path, "seconds": suite.probe(song_path).get("seconds") or state["beats"].get("duration")}
    exports = HOME / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    base = f"{slug(state['title'])}-{project['id']}"
    (exports / f"{base}.fcpxml").write_text(suite.fcpxml(state["title"], clips, rate, size, song))
    (exports / f"{base}.edl").write_text(suite.edl(state["title"], clips, rate))
    names, used = {}, set()
    for path in [one["path"] for one in clips] + [song_path]:
        key = str(path)
        if key in names:
            continue
        stem, suffix, n = Path(path).stem, Path(path).suffix, 1
        name = stem + suffix
        while name in used:
            n += 1
            name = f"{stem}-{n}{suffix}"
        used.add(name)
        names[key] = name
    target = exports / f"{base}-resolve.zip"
    readme = (f"{state['title']}\n\nUnzip this folder, then in DaVinci Resolve choose File > Import > Timeline and pick {base}.fcpxml.\n"
              "The clips and the song are in media/: if Resolve shows them offline, relink them to that folder.\n")
    fresh = target.with_suffix(".zip.new")
    with zipfile.ZipFile(fresh, "w", zipfile.ZIP_STORED) as bundle:
        bundle.writestr(f"{base}/{base}.fcpxml", suite.fcpxml(state["title"], clips, rate, size, song, lambda p: "media/" + names[str(p)]))
        bundle.writestr(f"{base}/README.txt", readme)
        for key, name in names.items():
            bundle.write(key, f"{base}/media/{name}")
    fresh.replace(target)
    return {"fcpxml": f"exports/{base}.fcpxml", "edl": f"exports/{base}.edl", "zip": f"exports/{base}-resolve.zip"}


def follow(project_id: str) -> None:
    """The rough cut to its end, in a thread: its foley, its export, the song under it, the editor's files."""

    state = json.loads(state_file(project_id).read_text())

    def stage(name: str, **more) -> None:
        state.update(stage=name, **more)
        save_state(state)
    try:
        project = load_project(project_id)
        if state.get("foley"):
            stage("foley")
            done = wait(list(state["foley"].values()))
            made, notes = {}, []
            for path, job_id in state["foley"].items():
                job = done[job_id]
                if job.get("status") == "done" and job.get("outputs"):
                    made[path] = job["outputs"][0]
                else:
                    notes.append(f"{Path(path).name} keeps its own sound: its foley {job.get('status')}")
            for one in project["timeline"]:
                one["path"] = made.get(one["path"], one["path"])
            save_project(project)
            state["notes"] = state.get("notes", []) + notes
            job = queue(export_body(project, state))
            stage("export", job=job["id"])
        stage("export")
        finished = wait([state["job"]])[state["job"]]
        if finished.get("status") != "done" or not finished.get("outputs"):
            raise RuntimeError("the export " + str(finished.get("status")) + (": " + str(finished.get("error"))[-300:] if finished.get("error") else ""))
        video = finished["outputs"][0]
        if not state["song"]["take"]:  # a voice take or a sound file: laid under the finished export here
            stage("song")
            import music
            target = inside(video)
            mixed = target.with_suffix(".song.mp4")
            music.lay(target, inside(state["song"]["path"]), mixed, float(state["music_db"]))
            mixed.replace(target)
        stage("files", video=video)
        files = editor_files(project, state)
        stage("done", video=video, files=files, finished=time.time())
    except Exception as error:  # noqa: BLE001 - said on the panel, not lost in a thread
        stage("failed", error=str(error)[-400:] or type(error).__name__)
    finally:
        FOLLOWING.pop(project_id, None)


def start(project_id: str) -> None:
    thread = threading.Thread(target=follow, args=(project_id,), daemon=True, name=f"roughcut-{project_id}")
    FOLLOWING[project_id] = thread
    thread.start()


# ---------- the API ----------

def number(body: dict, key: str, default, low, high, kind=float):
    value = body.get(key)
    if value in (None, ""):
        return default
    try:
        value = kind(value)
    except (TypeError, ValueError):
        raise HTTPException(400, f"{key} is a number")
    if not low <= value <= high:
        raise HTTPException(400, f"{key} is between {low} and {high}")
    return value


@router.post("/api/roughcut")
def make(body: dict) -> dict:
    """{clips?: [paths] | film: id, song: path or take id, cut, foley, order, align?, title, width, height, music_db?,
    seed?}: the rough cut planned, its project made and its first jobs queued. Returns {project, job, foley, beats,
    timeline, dropped}."""

    body = body or {}
    cut = str(body.get("cut") or "natural")  # each shot near its own length, cut on a beat
    order = str(body.get("order") or "given")
    align = "start" if body.get("align") == "start" else "centre"
    if cut not in CUTS:
        raise HTTPException(400, "cut is beat, bar, two-bars or natural")
    if order not in ORDERS:
        raise HTTPException(400, "order is given, shuffle or energy")
    width = number(body, "width", None, 256, 4096, int)
    height = number(body, "height", None, 256, 4096, int)
    import music
    music_db = number(body, "music_db", music.MUSIC_DB, -30, 0)
    seed = number(body, "seed", random.randrange(1 << 30), 0, 1 << 31, int)
    if body.get("film") and body.get("clips"):
        raise HTTPException(400, "give a film or clips, not both")
    if body.get("film"):
        name, paths = film_clips(str(body["film"]))
    else:
        paths = body.get("clips")
        if not isinstance(paths, list) or not paths or not all(isinstance(p, str) for p in paths):
            raise HTTPException(400, "give a film, or clips: a list of clip paths in the studio folder")
        name = None
    if len(paths) > MOST_CLIPS:
        raise HTTPException(400, f"a rough cut takes at most {MOST_CLIPS} clips")
    clips = []
    for path in paths:
        target = inside(path)
        if not target.is_file() or target.suffix.lower() not in VIDEO:
            raise HTTPException(400, f"not a clip in the studio folder: {path}")
        seconds = media_seconds(str(target))[0]
        clips.append({"path": relative(target), "seconds": seconds})
    if width is None or height is None:  # the first clip's own size: a cut is never enlarged unless asked
        seen = suite.probe(inside(clips[0]["path"]))
        own = (int(seen["width"]), int(seen["height"])) if seen.get("width") and seen.get("height") else (1920, 1080)
        if width is None and height is None:
            width, height = own
        elif width is None:
            width = max(2, round(height * own[0] / own[1] / 2) * 2)
        else:
            height = max(2, round(width * own[1] / own[0] / 2) * 2)
    title = str(body.get("title") or "").strip()[:80] or (f"Rough cut of {name}" if name else "Trailer")
    song = song_of(str(body.get("song") or ""))
    try:
        found = beats.cached(inside(song["path"]))
    except ImportError:
        raise HTTPException(503, "finding the beats needs numpy, which is not installed in the app's Python")
    except (RuntimeError, subprocess.TimeoutExpired) as error:
        raise HTTPException(400, str(error)[-300:])
    if len(found.get("beats") or []) < 4:
        raise HTTPException(400, "no steady beat was found in that song: pick one with a clear pulse")
    laid = plan(ordered(clips, order, seed), found, cut, align)
    if not laid["timeline"]:
        raise HTTPException(400, "nothing fits: the song is too short for these clips")
    foley = {}
    if body.get("foley"):
        for one in laid["timeline"]:
            if one["path"] in foley or not FOLEY_SHORTEST <= one["duration"] <= FOLEY_LONGEST:
                continue
            job = queue({"kind": "foley", "title": "Foley: " + Path(one["path"]).name[:40],
                         "params": {"video": one["path"], "keep_sound": True}})
            foley[one["path"]] = job["id"]
    project = new_project(title)
    project["timeline"] = [{k: one[k] for k in ("path", "in", "out", "duration")} for one in laid["timeline"]]
    if song["take"]:
        project["soundtrack"] = {"take": song["take"], "music_db": music_db}
    project["roughcut"] = {"song": song["path"], "cut": cut, "order": order, "foley": bool(foley), "bpm": found["bpm"]}
    save_project(project)
    summary = {k: found.get(k) for k in ("bpm", "bars", "confidence", "first", "duration")}
    state = {"project": project["id"], "title": title, "stage": "foley" if foley else "export", "song": song, "cut": cut, "order": order,
             "seed": seed, "music_db": music_db, "width": width, "height": height, "foley": foley, "job": None,
             "beats": summary, "cuts": len(laid["timeline"]), "dropped": laid["dropped"], "notes": [], "created": time.time()}
    if not foley:
        state["job"] = queue(export_body(project, state))["id"]
    save_state(state)
    start(project["id"])
    return {"project": project["id"], "job": state["job"], "foley": list(foley.values()), "beats": summary,
            "timeline": laid["timeline"], "dropped": laid["dropped"], "title": title}


@router.get("/api/roughcut/{project_id}")
def status(project_id: str) -> dict:
    file = state_file(project_id)
    if not file.is_file():
        raise HTTPException(404, "no rough cut in that project")
    try:
        state = json.loads(file.read_text())
    except (OSError, ValueError):
        raise HTTPException(404, "the rough cut's file could not be read")
    ids = list((state.get("foley") or {}).values()) + ([state["job"]] if state.get("job") else [])
    found = {job_id: job_of(job_id) for job_id in ids}
    state["jobs"] = {job_id: public(job) if public else job for job_id, job in found.items() if job}
    if state.get("stage") not in ("done", "failed") and project_id not in FOLLOWING:
        state.update(stage="failed", error="the app was restarted before the rough cut was finished: make it again")
    return state


def attach(home: Path, queue_job, job_table: dict, job_lock, within, short, measure, shape=None) -> APIRouter:
    """``queue_job`` (server.queue_job), the job table and its lock, the path helpers, server.media_seconds and
    server.public."""

    global HOME, queue, jobs, lock, inside, relative, media_seconds, public
    HOME, queue, jobs, lock, inside, relative, media_seconds, public = home, queue_job, job_table, job_lock, within, short, measure, shape
    return router
