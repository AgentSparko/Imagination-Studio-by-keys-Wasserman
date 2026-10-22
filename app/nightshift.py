"""The night shift: a batch of clips rendered overnight, and dailies to wake up to.

A shift is a plan kept in <studio folder>/nightshift/<id>.json: Slate shots ({"kind": "shot", "film", "shot"}) and
plain jobs ({"kind": "job", "body": <a POST /api/jobs body>}; one job each, so not Restyle, Extend or Fix, whose
later steps follow the first). It starts at a time ("23:00", or a date and time), when
the queue has been idle for two minutes ("idle"), or at once ("now"). The scheduler thread queues its items one after
another, never more than two waiting ahead of the queue, and records each job's clip and checks as it finishes. A
plan is read again from its file on every tick, so a restart loses nothing: a running shift carries on from the next
item that has no finished clip.

When every item is done, the morning's dailies are made in <studio folder>/dailies/: a reel (each clip after a short
title slate naming it, its seed and its checks; 1280x720, 24 fps, sound kept), the Circle Take dailies package
(suite.dailies_package), notes in Markdown (what to look at first), and a push notification to the phones that
asked for them (push.py)."""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException

TICK = 5  # seconds between the scheduler's looks at the queue
IDLE_AFTER = 120  # "idle": the queue has had nothing queued or running for this long
AHEAD = 2  # at most this many of a shift's jobs wait in the queue at once
SLATE_SECONDS = 1.5  # the title slate before each clip of the reel
REEL_SIZE = (1280, 720)
REEL_FPS = 24
STARTS = ("idle", "now")
CLIPS = (".mp4", ".mov", ".m4v", ".webm")
FINISHED = ("done", "failed", "cancelled")
KEPT = ("prompt", "seed", "preset", "quality", "frames", "seconds", "width", "height", "loras", "steps", "output", "slate")
FONTS = ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf",
         "/usr/share/fonts/TTF/DejaVuSans.ttf", "/System/Library/Fonts/Helvetica.ttc",
         "/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf")

HOME: Path = Path(".")
Q: dict = {}  # the queue's own: jobs, order, lock, queue_job, cancel, presets, probe, package, path, notify
IDLE = {"since": None}  # when the queue last became empty, by this process's clock
guard = threading.RLock()  # one writer of the plan files at a time: the scheduler and the routes
MAKING: set = set()  # the shifts whose morning this process is making (outside the guard: it takes minutes)
_thread: threading.Thread | None = None


# ---------- plans on disk ----------

def folder() -> Path:
    return HOME / "nightshift"


def plan_file(plan_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", str(plan_id)):
        raise HTTPException(400, "bad night shift id")
    return folder() / f"{plan_id}.json"


def read(plan_id: str) -> dict:
    file = plan_file(plan_id)
    if not file.is_file():
        raise HTTPException(404, "no such night shift")
    return json.loads(file.read_text())


def write(plan: dict) -> dict:
    file = plan_file(plan["id"])
    file.parent.mkdir(parents=True, exist_ok=True)
    plan["updated"] = time.time()
    scratch = file.with_suffix(".tmp")
    scratch.write_text(json.dumps(plan, indent=1))
    scratch.replace(file)
    return plan


def plans() -> list[dict]:
    out = []
    for file in folder().glob("*.json"):
        try:
            out.append(json.loads(file.read_text()))
        except (OSError, ValueError):
            continue
    return sorted(out, key=lambda p: p.get("created") or 0, reverse=True)


def check_start(start) -> str:
    start = str(start or "idle").strip()
    if start in STARTS or re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", start):
        return start
    try:
        dt.datetime.fromisoformat(start.replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(400, 'start is "idle", "now", a time like "23:00", or a date and time like "2026-10-22T23:00"')
    return start


def check_items(items) -> list[dict]:
    if not isinstance(items, list) or not items:
        raise HTTPException(400, "a night shift needs items: Slate shots or jobs")
    out = []
    for item in items:
        if not isinstance(item, dict):
            raise HTTPException(400, "each item is an object")
        if item.get("kind") == "shot":
            if not item.get("film") or not item.get("shot"):
                raise HTTPException(400, "a shot item names its film and its shot")
            out.append({"kind": "shot", "film": str(item["film"]), "shot": str(item["shot"])})
        elif item.get("kind") == "job":
            if not isinstance(item.get("body"), dict) or not item["body"].get("kind"):
                raise HTTPException(400, "a job item carries the body of a job: {kind, title, params}")
            import revise
            if item["body"]["kind"] in revise.KINDS:  # a chain of jobs: the shift would follow its first step only
                raise HTTPException(400, f"a night shift takes single jobs: {item['body']['kind']} (Restyle, Extend, Fix) is queued from the Library")
            out.append({"kind": "job", "body": item["body"]})
        else:
            raise HTTPException(400, 'an item is {"kind": "shot", "film", "shot"} or {"kind": "job", "body"}')
    return out


def new_plan(body: dict) -> dict:
    items = list(body.get("items") or [])
    film = body.get("film")
    name = str(body.get("name") or "").strip()
    if film:
        project = film_project(str(film))
        wanted = set(map(str, body.get("shots") or []))
        found = film_shots(project, wanted)
        if wanted and len(found) < len(wanted):
            known = {s.get("id") for _, s in all_shots(project)} | {(s.get("name") or "").strip().lower() for _, s in all_shots(project)}
            missing = [w for w in wanted if w not in known and w.lower() not in known]
            if missing:
                raise HTTPException(404, "no such shot: " + ", ".join(missing[:5]))
        if not found and not items:
            raise HTTPException(400, "every shot of this film with a prompt already has a good take")
        items += [{"kind": "shot", "film": project["id"], "shot": shot["id"]} for _, shot in found]
        name = name or (project.get("name") or "Film") + ", the rest"
    plan = {"id": time.strftime("%m%d-%H%M%S-") + uuid.uuid4().hex[:6], "name": name or "Night shift",
            "start": check_start(body.get("start")), "items": check_items(items), "state": "planned",
            "progress": {"done": 0, "total": len(items), "current": None}, "results": [], "reel": None, "zip": None,
            "notes": None, "started": None, "finished": None, "created": time.time()}
    with guard:
        return write(plan)


# ---------- Slate films: what is left to render ----------

def film_project(ref: str) -> dict:
    import slate
    try:
        return slate.get_project(ref)
    except HTTPException:
        pass
    hits = [p for p in slate.list_projects() if (p.get("name") or "").strip().lower() == ref.strip().lower()]
    if len(hits) != 1:
        raise HTTPException(404 if not hits else 400, "no such film" if not hits else f"{len(hits)} films are called {ref}: give its id")
    return slate.get_project(hits[0]["id"])


def all_shots(project: dict) -> list[tuple[dict, dict]]:
    return [(scene, shot) for scene in project.get("scenes") or [] for shot in scene.get("shots") or []]


def has_prompt(shot: dict) -> bool:
    tf = shot.get("tf") or {}
    return bool((shot.get("prompt") or "").strip() or (tf.get("h3Auto") is False and (tf.get("h3") or "").strip()))


def good_take(shot: dict) -> bool:
    """A take whose clip is in the studio folder and that is marked good or circled (a take still rendering has no
    rating yet)."""
    return any(t.get("mediaPath") and t.get("rating") in ("good", "circled") and (HOME / t["mediaPath"]).is_file() for t in shot.get("takes") or [])


def film_shots(project: dict, wanted: set | None = None) -> list[tuple[dict, dict]]:
    """The shots to render: those named (by id or name), else every shot with a prompt and no good take."""
    pairs = all_shots(project)
    if wanted:
        lower = {w.lower() for w in wanted}
        return [(sc, sh) for sc, sh in pairs if sh.get("id") in wanted or (sh.get("name") or "").strip().lower() in lower]
    return [(sc, sh) for sc, sh in pairs if has_prompt(sh) and not good_take(sh)]


def render_shot(film: str, shot_id: str, chain: str | None) -> tuple[str, str]:
    """A Slate shot into the queue, as the page's own Render does (slate.js slRender), and its take logged in the
    shot. Returns the job's id and the clip it writes."""

    import slate
    project = slate.get_project(film)
    scene, shot, _ = slate.find_shot(project, shot_id)
    compiled = slate.compile_shot(project, shot_id, chain)
    tf, look = shot.get("tf") or {}, project.get("tf") or {}
    plan = dict(compiled["plan"])
    output = f"slate/{project['id']}/takes/{shot_id}-{int(time.time() * 1000):x}.mp4"
    prompt = tf["h3"].strip() if tf.get("h3Auto") is False and (tf.get("h3") or "").strip() else compiled["prompt"]
    params = {**plan, "prompt": prompt, "output": output}
    if not plan.get("references") and (plan.get("loras") or look.get("loras")):
        params["loras"] = plan.get("loras") or look.get("loras")
    defaults = project.get("defaults") or {}
    params["brain"] = {"backend": defaults.get("brain"), "endpoint": defaults.get("localEndpoint"), "model": defaults.get("localModel")}
    params["slate"] = {"project": project["id"], "scene": scene.get("id"), "shot": shot_id}
    job = Q["queue_job"]({"kind": "video", "title": f"{scene.get('name')} · {shot.get('name')}", "params": params})
    project = slate.get_project(film)  # logged into the film as it is after the job was queued
    _, shot, _ = slate.find_shot(project, shot_id)
    asked = job.get("params") or params
    shot.setdefault("takes", []).insert(0, {
        "id": f"take-{int(time.time() * 1000):x}-{uuid.uuid4().hex[:5]}", "loggedAt": slate.now(), "model": "minimax-h3",
        "prompt": prompt, "rating": "", "notes": "", "origin": "Night shift", "mediaPath": output, "jobId": job["id"],
        "seed": plan.get("seed"), "settings": {k: asked[k] for k in ("preset", "quality", "frames", "seconds", "width", "height", "loras") if asked.get(k) is not None}})
    slate.write(project)
    return job["id"], output


def note_take(result: dict, item: dict | None = None) -> None:
    """A finished shot's checks written beside its take in the film, as an imported take keeps them, and its rating
    from them: no-good when it failed or its checks did, else good. ``item``: the plan's item, for a result that
    has no job record left (the studio restarted)."""
    slated = (result.get("params") or {}).get("slate") or \
        ({"project": item["film"], "shot": item["shot"]} if item and item.get("kind") == "shot" else {})
    if not slated.get("project"):
        return
    import slate
    try:
        project = slate.get_project(slated["project"])
        _, shot, _ = slate.find_shot(project, slated["shot"])
    except HTTPException:
        return
    for take in shot.get("takes") or []:
        if take.get("jobId") == result.get("job"):
            for key in ("qa", "shot"):
                if result.get(key):
                    take[key] = result[key]
            if result.get("status") != "done":
                take["notes"] = (take.get("notes") or "") + ("" if not take.get("notes") else " · ") + "Night shift: " + (result.get("error") or result.get("status") or "")
            if not take.get("rating"):  # a rating the user gave meanwhile stays
                take["rating"] = "no-good" if verdict(result)[0] is False else "good"
            slate.write(project)
            return


# ---------- the scheduler ----------

def busy() -> bool:
    with Q["lock"]:
        return any(job.get("status") in ("queued", "running") for job in Q["jobs"].values())


def time_due(start: str, created: float, now: float) -> bool:
    if start == "now":
        return True
    if re.fullmatch(r"\d\d:\d\d", start):
        hour, minute = map(int, start.split(":"))
        made = dt.datetime.fromtimestamp(created)
        at = made.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if at < made:  # the first such time after the plan was made
            at += dt.timedelta(days=1)
        return now >= at.timestamp()
    when = dt.datetime.fromisoformat(start.replace("Z", "+00:00"))
    return now >= (when.timestamp() if when.tzinfo else time.mktime(when.timetuple()))


def due(plan: dict, now: float) -> bool:
    if plan.get("start") == "idle":
        return IDLE["since"] is not None and now - IDLE["since"] >= IDLE_AFTER
    return time_due(plan.get("start") or "now", plan.get("created") or now, now)


def snapshot(job: dict) -> dict:
    params = job.get("params") or {}
    return {"status": job.get("status"), "title": job.get("title"), "outputs": list(job.get("outputs") or []),
            "qa": job.get("qa"), "shot": job.get("shot"), "error": job.get("error"), "engine": job.get("engine"),
            "passes": job.get("passes"), "kind": job.get("kind"), "created": job.get("created"), "spark": job.get("spark"),
            "finished": job.get("finished") or time.time(), "seed": params.get("seed"),
            "params": {k: params[k] for k in KEPT if k in params}}


def queue_item(plan: dict, index: int) -> dict | None:
    """The plan's item at ``index`` into the queue: its result, or None when the queue is full for now."""

    item = plan["items"][index]
    try:
        if item["kind"] == "shot":
            earlier = next((r for r in plan["results"] if r["item"] == index - 1), None)
            chain = earlier.get("output") if earlier and plan["items"][index - 1].get("film") == item["film"] else None
            job_id, output = render_shot(item["film"], item["shot"], chain)
        else:
            job = Q["queue_job"](json.loads(json.dumps(item["body"])))
            job_id, output = job["id"], (item["body"].get("params") or {}).get("output")
    except HTTPException as error:
        if error.status_code in (429, 503):  # the queue is full, or the studio is updating: tried again next tick
            return None
        return {"item": index, "job": None, "status": "failed", "error": str(error.detail), "outputs": [], "finished": time.time()}
    except Exception as error:  # noqa: BLE001 - one bad item never stops the shift
        return {"item": index, "job": None, "status": "failed", "error": str(error)[-300:], "outputs": [], "finished": time.time()}
    return {"item": index, "job": job_id, "status": "queued", "output": output, "queued": time.time()}


def follow(plan: dict) -> bool:
    """The shift's jobs looked at, its next items queued; True when something changed."""

    changed = False
    with Q["lock"]:
        known = {r["job"]: dict(Q["jobs"][r["job"]]) for r in plan["results"] if r.get("job") in Q["jobs"]}
    kept = []
    for result in plan["results"]:
        if result["status"] in FINISHED:
            kept.append(result)
            continue
        job = known.get(result.get("job"))
        if job is None:
            # not in the queue: the studio restarted, or the job was cleared away. A clip it wrote is kept;
            # otherwise the item is queued again
            output = result.get("output")
            if output and (HOME / output).is_file():
                result.update(status="done", outputs=[output], finished=time.time())
                note_take(result, plan["items"][result["item"]])
                kept.append(result)
            changed = True
            continue
        if job.get("status") in FINISHED:
            result.update(snapshot(job))
            if not result["outputs"] and result.get("output") and (HOME / result["output"]).is_file():
                result["outputs"] = [result["output"]]
            note_take(result)
            changed = True
        elif job.get("status") != result["status"]:
            result["status"] = job.get("status")
            changed = True
        kept.append(result)
    plan["results"] = sorted(kept, key=lambda r: r["item"])
    if plan["state"] != "running":
        return changed
    handled = {r["item"] for r in plan["results"]}
    waiting = sum(1 for r in plan["results"] if r["status"] == "queued")
    for index in range(len(plan["items"])):
        if waiting >= AHEAD:
            break
        if index in handled:
            continue
        result = queue_item(plan, index)
        if result is None:
            break
        plan["results"].append(result)
        plan["results"].sort(key=lambda r: r["item"])
        handled.add(index)
        waiting += result["status"] == "queued"
        changed = True
    done = [r for r in plan["results"] if r["status"] in FINISHED]
    current = next((r["job"] for r in plan["results"] if r["status"] == "running"), None) or \
        next((r["job"] for r in plan["results"] if r["status"] == "queued"), None)
    progress = {"done": len(done), "total": len(plan["items"]), "current": current,
                "clips": sum(1 for r in done if r["status"] == "done" and clip_of(r))}
    if progress != plan.get("progress"):
        plan["progress"], changed = progress, True
    if len(done) == len(plan["items"]):
        plan["state"] = "morning"  # the scheduler makes the morning outside the guard (make_morning)
        changed = True
    return changed


def start_plan(plan: dict) -> None:
    plan.update(state="running", started=time.time())
    follow(plan)


def tick(now: float | None = None) -> None:
    """One look: the idle clock, every running shift followed, a planned one started when its time has come."""

    now = time.time() if now is None else now
    if busy():
        IDLE["since"] = None
    elif IDLE["since"] is None:
        IDLE["since"] = now
    with guard:
        everything = plans()
        running = [p for p in everything if p.get("state") == "running"]
        for plan in running:
            if follow(plan):
                write(plan)
        # a shift whose items are all finished (now, or before a restart): its morning, made below
        mornings = [p for p in everything if p.get("state") == "morning" and p["id"] not in MAKING]
        MAKING.update(p["id"] for p in mornings)
        if not any(p.get("state") == "running" for p in running):
            for plan in sorted((p for p in everything if p.get("state") == "planned"), key=lambda p: p.get("created") or 0):
                if due(plan, now):
                    start_plan(plan)
                    write(plan)
                    break
    for plan in mornings:
        make_morning(plan)


def make_morning(plan: dict) -> None:
    """The morning made outside the guard (a reel takes minutes, and the routes wait on the guard), then written."""

    try:
        morning(plan)
        with guard:
            if plan_file(plan["id"]).is_file():  # not forgotten meanwhile
                write(plan)
    finally:
        MAKING.discard(plan["id"])


def loop() -> None:
    while True:
        try:
            tick()
        except Exception as error:  # noqa: BLE001 - the scheduler never dies of one bad plan
            print(f"[nightshift] {type(error).__name__}: {error}", flush=True)
        time.sleep(TICK)


def start() -> threading.Thread:
    """The scheduler thread, once (server.py at start)."""
    global _thread
    if _thread is None or not _thread.is_alive():
        IDLE["since"] = None  # the idle clock begins with this process
        _thread = threading.Thread(target=loop, daemon=True, name="nightshift")
        _thread.start()
    return _thread


# ---------- the morning: reel, dailies package, notes, a notification ----------

def clip_of(result: dict) -> str | None:
    return next((p for p in result.get("outputs") or [] if Path(p).suffix.lower() in CLIPS and (HOME / p).is_file()), None)


def verdict(result: dict) -> tuple[bool | None, str]:
    """Whether a clip passed its checks, and why not: (True, ""), (False, reason) or (None, "not checked")."""

    if result.get("status") != "done":
        return False, (result.get("error") or result.get("status") or "not rendered").strip()
    if not clip_of(result):
        return False, "no clip was written"
    reasons, ran = [], False
    qa = result.get("qa") if isinstance(result.get("qa"), dict) else None
    if qa and qa.get("passed") is not None:
        ran = True
        if qa["passed"] is False:
            last = (qa.get("takes") or [{}])[-1]
            said = [f"{name}: {(last.get(name) or {}).get('note') or 'failed'}" for name in ("audio", "identity")
                    if (last.get(name) or {}).get("result") == "fail"]
            reasons.append("QA gate failed" + (" (" + "; ".join(said) + ")" if said else ""))
    shot = result.get("shot") if isinstance(result.get("shot"), dict) else None
    if shot and shot.get("passed") is not None:
        ran = True
        if shot["passed"] is False:
            last = (shot.get("takes") or [{}])[-1]
            said = [f"{edge} frame not held" for edge in ("first", "last") if last.get(edge) and not last[edge].get("held")]
            if last.get("cuts"):
                said.append(f"{len(last['cuts'])} cut{'s' if len(last['cuts']) > 1 else ''} inside the shot")
            reasons.append("shot check failed" + (" (" + ", ".join(said) + ")" if said else ""))
    if reasons:
        return False, "; ".join(reasons)
    return (True, "") if ran else (None, "not checked")


def label(plan: dict, result: dict) -> str:
    return str(result.get("title") or (plan["items"][result["item"]].get("body") or {}).get("title") or f"Item {result['item'] + 1}")


def tools() -> dict:
    path = Q.get("path", lambda: os.environ.get("PATH", ""))()
    return {name: shutil.which(name, path=path) for name in ("ffmpeg", "ffprobe", "fc-match")}


def font() -> str | None:
    """A font file on this machine for the title slates: fontconfig's sans, DejaVu Sans, Helvetica; None if none."""
    found = tools()
    if found["fc-match"]:
        for name in ("DejaVu Sans", "Helvetica", "sans-serif"):
            try:
                file = subprocess.run([found["fc-match"], "-f", "%{file}", name], capture_output=True, text=True, timeout=10).stdout.strip()
            except (OSError, subprocess.SubprocessError):
                break
            if file and Path(file).is_file():
                return file
    return next((f for f in FONTS if Path(f).is_file()), None)


def ffmpeg_has(ffmpeg: str, what: str, name: str) -> bool:
    try:
        listed = subprocess.run([ffmpeg, "-hide_banner", f"-{what}"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return re.search(rf"\s{re.escape(name)}\s", listed) is not None


def has_sound(ffprobe: str | None, clip: Path) -> bool:
    if not ffprobe:
        return False
    try:
        out = subprocess.run([ffprobe, "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", str(clip)],
                             capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return bool(out.strip())


def build_reel(clips: list[dict], target: Path) -> Path:
    """``clips``: [{path, lines}] in order. Each clip after a title slate of its lines, letterboxed to 1280x720 at 24 fps,
    its sound kept (silence where it has none), joined into ``target``."""

    found = tools()
    if not found["ffmpeg"]:
        raise RuntimeError("ffmpeg is not installed: the reel needs it")
    ffmpeg, (width, height) = found["ffmpeg"], REEL_SIZE
    video = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"] if ffmpeg_has(ffmpeg, "encoders", "libx264") else ["-c:v", "mpeg4", "-q:v", "3"]
    common = [*video, "-pix_fmt", "yuv420p", "-r", str(REEL_FPS), "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]
    typeface = font() if ffmpeg_has(ffmpeg, "filters", "drawtext") else None
    silence = ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
    fit = (f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
           f"setsar=1,fps={REEL_FPS}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="reel-") as scratch:
        parts = []
        if typeface:
            local = Path(scratch) / ("font" + Path(typeface).suffix.lower())
            try:
                local.symlink_to(typeface)
            except OSError:
                shutil.copyfile(typeface, local)
            typeface = local.name
        for number, clip in enumerate(clips):
            title, words = Path(scratch) / f"{number:03d}-slate.mp4", Path(scratch) / f"{number:03d}.txt"
            words.write_text("\n".join(clip["lines"]))
            draw = []
            if typeface:  # by plain names in the scratch folder (the commands run there): nothing in them to escape
                draw = ["-vf", f"drawtext=fontfile={typeface}:textfile={words.name}:expansion=none:fontcolor=white:fontsize=38:"
                               "line_spacing=14:x=(w-text_w)/2:y=(h-text_h)/2"]
            run([ffmpeg, "-y", "-v", "error", "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:r={REEL_FPS}:d={SLATE_SECONDS}",
                 *silence, "-t", str(SLATE_SECONDS), *draw, "-map", "0:v", "-map", "1:a", *common, "-shortest", str(title)], scratch)
            body = Path(scratch) / f"{number:03d}-clip.mp4"
            source = Path(clip["path"])
            sound = has_sound(found["ffprobe"], source)
            run([ffmpeg, "-y", "-v", "error", "-i", str(source), *([] if sound else silence), "-vf", fit,
                 "-map", "0:v:0", "-map", "0:a:0" if sound else "1:a", *common, "-shortest", str(body)])
            parts += [title, body]
        listing = Path(scratch) / "list.txt"
        listing.write_text("".join(f"file '{p}'\n" for p in parts))
        run([ffmpeg, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", "-movflags", "+faststart", str(target)])
    return target


def run(command: list[str], where: str | None = None) -> None:
    done = subprocess.run(command, capture_output=True, text=True, timeout=1800, cwd=where)
    if done.returncode != 0:
        raise RuntimeError("ffmpeg: " + (done.stderr.strip()[-400:] or f"exit {done.returncode}"))


def notes_text(plan: dict, rows: list[tuple]) -> str:
    """Markdown: each clip's verdict, why, and where it is; the ones to look at first."""

    looks = [r for r in rows if r[1] is False]
    rest = [r for r in rows if r[1] is not False]
    lines = [f"# {plan['name']}", "",
             f"Night shift {plan['id']}: {sum(1 for r in rows if r[3])} of {len(plan['items'])} clips rendered, "
             f"{len(looks)} to look at.", ""]
    started = plan.get("started")
    if started:
        lines += [f"Started {time.strftime('%Y-%m-%d %H:%M', time.localtime(started))}, finished "
                  f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(plan.get('finished') or time.time()))}.", ""]
    for heading, group in (("To look at", looks), ("Passed or not checked", rest)):
        if not group:
            continue
        lines += [f"## {heading}", ""]
        for result, ok, why, clip in group:
            word = "PASS" if ok else "FAIL" if ok is False else "NOT CHECKED"
            seed = f", seed {result.get('seed')}" if result.get("seed") is not None else ""
            lines.append(f"- **{label(plan, result)}**{seed}: {word}" + (f". {why}" if why and ok is not True else ""))
            lines.append(f"  - {clip or 'no clip'}")
        lines.append("")
    if plan.get("reel"):
        lines += [f"Reel: {plan['reel']}"]
    if plan.get("zip"):
        lines += [f"Dailies for Circle Take: {plan['zip']}"]
    return "\n".join(lines).rstrip() + "\n"


def morning(plan: dict) -> None:
    """Every item finished: the reel, the dailies package, the notes and the notification. Each part that cannot be
    made is said in the plan's ``problems``; the shift is done all the same."""

    plan.update(state="done", finished=time.time(), problems=[])
    stem = f"{time.strftime('%Y-%m-%d')}-{plan['id']}"
    out = HOME / "dailies"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for result in plan["results"]:
        ok, why = verdict(result)
        rows.append((result, ok, why, clip_of(result)))
    made = [(result, ok, clip) for result, ok, _, clip in rows if clip]
    if made:
        try:
            clips = []
            for result, ok, clip in made:
                seed = result.get("seed")
                clips.append({"path": str(HOME / clip), "lines": [label(plan, result)[:60], ("seed " + str(seed)) if seed is not None else "seed not given",
                                                                    "QA pass" if ok else "QA fail" if ok is False else "not checked"]})
            reel = build_reel(clips, out / f"{stem}.mp4")
            plan["reel"] = str(reel.relative_to(HOME))
        except Exception as error:  # noqa: BLE001
            plan["problems"].append(f"the reel was not made: {str(error)[-300:]}")
        try:
            takes = [({"id": result.get("job") or f"item{result['item']}", **result}, [HOME / clip]) for result, _, clip in made]
            answer = Q["package"](takes, Q.get("presets") or {}, Q["probe"], HOME)
            package = Path(answer.path)
            shutil.move(str(package), out / f"{stem}.zip")
            plan["zip"] = f"dailies/{stem}.zip"
        except Exception as error:  # noqa: BLE001
            plan["problems"].append(f"the dailies package was not made: {str(getattr(error, 'detail', error))[-300:]}")
    notes = out / f"{stem}.md"
    notes.write_text(notes_text(plan, rows))
    plan["notes"] = str(notes.relative_to(HOME))
    looks = sum(1 for _, ok, _, _ in rows if ok is False)
    title = "Night shift done"
    body = f"{len(made)} of {len(plan['items'])} clips, " + (f"{looks} to look at" if looks else "nothing to look at")
    plan["message"] = f"{title}: {body}"
    try:
        sent = Q.get("notify") and Q["notify"](title, body, "/#Home")
        plan["notified"] = sent if isinstance(sent, dict) else None
    except Exception as error:  # noqa: BLE001
        plan["problems"].append(f"the notification was not sent: {str(error)[-200:]}")


# ---------- the routes ----------

def public(plan: dict) -> dict:
    return {**plan, "items": plan.get("items"), "count": len(plan.get("items") or [])}


def attach(home: Path, jobs: dict, order: list, lock, queue_job, cancel, presets: dict, probe, package, path=None, notify=None) -> APIRouter:
    """``jobs``, ``order``, ``lock``, ``queue_job`` and ``cancel`` are the queue's (server.py); ``probe`` and ``package``
    make the Circle Take dailies (suite.py); ``path`` is the PATH the jobs' tools are found on; ``notify(title, body, url)``
    sends the morning's notification (push.py)."""

    global HOME
    HOME = Path(home)
    folder().mkdir(parents=True, exist_ok=True)
    Q.update(jobs=jobs, order=order, lock=lock, queue_job=queue_job, cancel=cancel, presets=presets, probe=probe,
             package=package, path=path or (lambda: os.environ.get("PATH", "")), notify=notify)
    router = APIRouter()

    @router.get("/api/nightshift")
    def listing() -> list[dict]:
        return [public(p) for p in plans()]  # each file is replaced whole: read without waiting for a morning being made

    @router.post("/api/nightshift")
    def plan_new(body: dict) -> dict:
        made = new_plan(body)
        if body.get("go") or made["start"] == "now":
            with guard:
                plan = read(made["id"])
                if not any(p.get("state") == "running" for p in plans()):
                    start_plan(plan)
                    write(plan)
                made = plan
        return public(made)

    @router.get("/api/nightshift/{plan_id}")
    def one(plan_id: str) -> dict:
        return public(read(plan_id))

    @router.delete("/api/nightshift/{plan_id}")
    def remove(plan_id: str) -> dict:
        with guard:
            plan = read(plan_id)
            if plan.get("state") == "running":
                raise HTTPException(409, "this night shift is running: cancel it first")
            plan_file(plan_id).unlink(missing_ok=True)
        return {"ok": True}

    @router.post("/api/nightshift/{plan_id}/start")
    def start_now(plan_id: str) -> dict:
        with guard:
            plan = read(plan_id)
            if plan.get("state") != "planned":
                raise HTTPException(409, f"this night shift is {plan.get('state')}")
            if any(p.get("state") == "running" for p in plans()):
                raise HTTPException(409, "another night shift is running: one at a time")
            start_plan(plan)
            return public(write(plan))

    @router.post("/api/nightshift/{plan_id}/cancel")
    def stop(plan_id: str) -> dict:
        with guard:
            plan = read(plan_id)
            if plan.get("state") not in ("planned", "running"):
                raise HTTPException(409, f"this night shift is {plan.get('state')}")
            for result in plan["results"]:
                if result.get("job") and result["status"] in ("queued", "running"):
                    try:
                        Q["cancel"](result["job"])
                    except HTTPException:
                        pass
                    result["status"] = "cancelled"
            plan.update(state="cancelled", finished=time.time())
            return public(write(plan))

    return router
