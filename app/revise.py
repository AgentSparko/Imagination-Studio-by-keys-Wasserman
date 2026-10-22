"""Work on a clip you already have: restyle it, extend it, or fix one region of it. Three Library actions.

    Restyle   the clip as a reference video (MiniMax H3 Ref2VA): its action, camera and timing are kept and it is drawn
              again in the look the prompt describes. The clip's own sound may be kept.
    Extend    a new clip that opens on the clip's last frame, then (by default) the two joined into one file:
              renders/<job>-extended.mp4. Over 15 seconds the new part is a long take, a chain of clips.
    Fix       two steps. The clip's first frame is redrawn by the picture model from an instruction (an `edit` job, the
              whole frame from the words, no mask), then the clip is made again from that frame as a reference picture
              and the clip as a reference video: the motion is the clip's, the look the corrected frame's.

Each is queued through the studio's own jobs (``video``, ``long`` and ``edit``), so the queue, the Sparks, the QA gate
and the Library treat the results like any other. A step that follows another (Extend's join, Fix's second clip) is
started by ``then``, which server.perform calls when the job whose ``params.then`` names a chain here has finished.
The chains live in memory, like the jobs.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException

KINDS = ("restyle", "extend", "fix")
TITLES = {"restyle": "Restyle", "extend": "Extend", "fix": "Fix"}
# presets the model renders at their own size (no 2x decoder, no crop of a larger picture): a clip of that size keeps it
NATIVE = ("small", "native", "p720", "sharp")
REFERENCE_SECONDS = (2.0, 15.0)   # what Ref2VA takes of a reference video (server.REFERENCE_CLIP_SECONDS)
EXTEND_SECONDS = (5, 30)
RESTYLE = "<Video 1> shows the action and camera. Re-render it as: {look}"
FIX = ("<Picture 1> is the corrected first frame of <Video 1>. Keep <Video 1>'s action, camera and timing exactly; "
       "render it with <Picture 1>'s look: {instruction}")
CONTINUE = "continue the shot"
IMAGE_MODELS = ("qwen", "qwen-full", "flux3k-fast", "flux3k-full")
TOOLS = ("/opt/homebrew/bin", "/usr/local/bin")


def tool(name: str) -> str | None:
    return shutil.which(name) or shutil.which(name, path=":".join(TOOLS))


def probe(path: Path) -> dict:
    """{width, height, duration, sound} of a clip, by ffprobe; a 400 with the reason when it cannot be read."""

    found = tool("ffprobe")
    if found is None:
        raise HTTPException(400, "working on a clip needs ffprobe, which comes with ffmpeg and is not installed on this machine")
    try:
        done = subprocess.run([found, "-v", "error", "-show_entries", "stream=codec_type,width,height:format=duration",
                               "-of", "json", str(path)], capture_output=True, text=True, timeout=30)
        data = json.loads(done.stdout or "{}")
        streams = data.get("streams") or []
        video = next(one for one in streams if one.get("codec_type") == "video")
        return {"width": int(video["width"]), "height": int(video["height"]),
                "duration": float((data.get("format") or {}).get("duration") or 0),
                "sound": any(one.get("codec_type") == "audio" for one in streams)}
    except (subprocess.TimeoutExpired, ValueError, StopIteration, KeyError, TypeError):
        raise HTTPException(400, f"{path.name} could not be read as a video")


def size_of(width: int, height: int, presets: dict, sides: tuple, most: int) -> dict:
    """The size a new clip is made at to match ``width`` x ``height``: the preset of exactly that size the model renders
    natively, else a size of its own (multiples of 32, within the sides and pixels a custom size may have) of about the
    same shape."""

    for name in NATIVE:
        if name in presets and tuple(presets[name][1]) == (width, height):
            return {"preset": name}
    low, high = sides
    scale = min(1.0, (most / max(width * height, 1)) ** 0.5)
    side = lambda n: max(low, min(high, int(round(n * scale / 32)) * 32))  # noqa: E731
    w, h = side(width), side(height)
    while w * h > most:  # rounding up may go over the pixels: the longer side gives way
        if w >= h:
            w -= 32
        else:
            h -= 32
    return {"preset": "custom", "width": w, "height": h}


def attach(studio):
    """``studio``: the server module (or a stand-in with its names): HOME, inside, relative, queue_job, build, jobs,
    frame_at, environment, PRESETS, CUSTOM_SIDES, CUSTOM_PIXELS, LONGEST, FPS, LEAD, VIDEO_SUFFIXES, hub, pace."""

    router = APIRouter()
    chains: dict[str, dict] = {}
    guard = threading.Lock()

    def clip_of(params: dict) -> Path:
        given = params.get("video")
        if not given:
            raise HTTPException(400, "choose a clip from the Library first")
        path = studio.inside(str(given))
        if not path.is_file() or path.suffix.lower() not in studio.VIDEO_SUFFIXES:
            raise HTTPException(400, f"{given} is not a clip in the studio folder")
        return path

    def number(params: dict, key: str, default, kind=float):
        try:
            return kind(params.get(key) if params.get(key) not in (None, "") else default)
        except (TypeError, ValueError):
            raise HTTPException(400, f"{key} is a number")

    def size(info: dict) -> dict:
        return size_of(info["width"], info["height"], studio.PRESETS, studio.CUSTOM_SIDES, studio.CUSTOM_PIXELS)

    def cut(clip: Path, seconds: float) -> Path:
        """The first ``seconds`` of a clip, as a file of its own in uploads/ (a reference video is at most 15 s)."""

        target = studio.HOME / "uploads" / f"{clip.stem[:40]}_first{seconds:g}s_{uuid.uuid4().hex[:6]}.mp4"
        target.parent.mkdir(parents=True, exist_ok=True)
        done = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(clip), "-t", f"{seconds:.3f}", "-c:v", "libx264",
                               "-crf", "16", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k",
                               str(target)], capture_output=True, text=True, env=studio.environment({}), timeout=600)
        if done.returncode or not target.is_file():
            raise HTTPException(400, f"the clip could not be cut to {seconds:g} s: {done.stderr.strip()[-200:]}")
        return target

    def reference(clip: Path, info: dict, asked) -> tuple[Path, float, list[str]]:
        """(the reference video, the new clip's seconds, notes): the clip itself, or its first part when it is longer
        than the model takes or than the length asked for."""

        low, high = REFERENCE_SECONDS
        length = info["duration"]
        if length and length < low - 0.05:
            raise HTTPException(400, f"the clip is {length:.1f} s: a clip worked on this way is at least {low:g} seconds long")
        if asked not in (None, ""):
            try:
                asked = float(asked)
            except (TypeError, ValueError):
                raise HTTPException(400, "seconds is a number")
            if not low <= asked <= high:
                raise HTTPException(400, f"seconds is between {low:g} and {high:g}")
            if length and asked > length + 0.05:
                raise HTTPException(400, f"the clip is only {length:.1f} s long")
        seconds = min(float(asked) if asked not in (None, "") else (length or high), high)
        notes = []
        if length and length > seconds + 0.05:
            if length - seconds >= 0.25:  # a 15 s clip of the model's is 15.08 s: cut without a word
                notes.append(f"The clip is {length:.1f} s long; " + (f"a new clip follows at most {high:g} s of it, so its first {high:g} s are used."
                                                                       if asked in (None, "") else f"its first {seconds:g} s are used."))
            return cut(clip, seconds), seconds, notes
        return clip, seconds, notes

    def paced(clip: Path) -> str:
        """Impacts or Flow from the clip's own movement (pace.py), or '' where it cannot be measured."""

        try:
            return studio.pace.choose("auto", studio.pace.measure(clip))
        except Exception:  # noqa: BLE001 - a pace is a help, never a reason to refuse
            return ""

    def own_prompt(clip: Path) -> str:
        """The prompt the clip was made from: its job's while the studio has it, else the takes list's, else ''."""

        path = studio.relative(clip)
        for job in list(studio.jobs.values()):
            if job.get("kind") in ("video", "studio", "long") and path in (job.get("outputs") or []):
                prompt = str((job.get("params") or {}).get("prompt") or "")
                if prompt.startswith(studio.LEAD):
                    prompt = prompt[len(studio.LEAD):]
                return prompt.strip()
        try:
            return str((studio.hub.takes().get(path) or {}).get("prompt") or "").strip()
        except Exception:  # noqa: BLE001
            return ""

    def plan(kind: str, params: dict) -> dict:
        """{steps: [{kind, title, params}], notes, join?} for one action; 400 with the reason when it cannot run."""

        clip = clip_of(params)
        name = clip.name
        seed = number(params, "seed", 0, int)
        loras = params.get("loras") or None
        if kind == "restyle":
            look = str(params.get("prompt") or "").strip()
            if not look:
                raise HTTPException(400, "say what the clip should look like now")
            info = probe(clip)
            ref, seconds, notes = reference(clip, info, params.get("seconds"))
            keep = bool(params.get("keep_sound", True))
            if keep and not info["sound"]:
                keep = False
                notes.append("The clip has no sound of its own: the new one gets sound made for it.")
            job = {"prompt": RESTYLE.format(look=look), "reference_videos": [studio.relative(ref)],
                   "reference_video_sound": keep, "seconds": seconds, "seed": seed, **size(info)}
            if loras:
                job["loras"] = loras
            if params.get("ref_turbo"):
                job["ref_turbo"] = True
            pace = paced(ref)
            if pace:
                job["pace"] = pace
            return {"steps": [{"kind": "video", "title": f"Restyle: {name}", "params": job}], "notes": notes}
        if kind == "extend":
            seconds = number(params, "seconds", 5)
            low, high = EXTEND_SECONDS
            if not low <= seconds <= high:
                raise HTTPException(400, f"an extension is {low} to {high} seconds")
            info = probe(clip)
            prompt = str(params.get("prompt") or "").strip() or own_prompt(clip) or CONTINUE
            job = {"first_frame_from": studio.relative(clip), "seed": seed, **size(info)}
            if loras:
                job["loras"] = loras
            if seconds <= studio.LONGEST:
                step = {"kind": "video", "params": {**job, "seconds": seconds,
                                                    "prompt": prompt if prompt.startswith("For the target video") else f"{studio.LEAD}\n\n{prompt}"}}
            else:  # longer than one clip: a long take, each part opening on the last frame of the one before
                step = {"kind": "long", "params": {**job, "total_seconds": int(round(seconds)), "prompt": prompt}}
            step["title"] = f"Extend: {name} +{seconds:g} s"
            join = params.get("join", True) is not False
            return {"steps": [step], "notes": [], "join": studio.relative(clip) if join else None}
        if kind == "fix":
            instruction = str(params.get("instruction") or "").strip()
            if not instruction:
                raise HTTPException(400, "say what to change, e.g. change the sky to a stormy sunset")
            model = str(params.get("image_model") or "qwen")
            if model not in IMAGE_MODELS:
                raise HTTPException(400, f"image_model is one of {', '.join(IMAGE_MODELS)}")
            info = probe(clip)
            ref, seconds, notes = reference(clip, info, None)
            keep = bool(params.get("keep_sound", True))
            if keep and not info["sound"]:
                keep = False
                notes.append("The clip has no sound of its own: the new one gets sound made for it.")
            frame = studio.HOME / "uploads" / f"{clip.stem[:40]}_first_{uuid.uuid4().hex[:6]}.png"
            frame.parent.mkdir(parents=True, exist_ok=True)
            try:
                studio.frame_at(clip, frame, 0)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
                raise HTTPException(400, "the clip's first frame could not be taken (is ffmpeg installed?)")
            edit = {"image": studio.relative(frame), "prompt": instruction, "image_model": model, "seed": seed, "size": "source"}
            if loras and model == "qwen":
                edit["loras"] = loras
            video = {"prompt": FIX.format(instruction=instruction), "references": [], "reference_videos": [studio.relative(ref)],
                     "reference_video_sound": keep, "seconds": seconds, "seed": seed, **size(info)}
            # the second step is checked now, with the frame as it is standing in for the corrected one
            studio.build("video", {**video, "references": [studio.relative(frame)]}, "check")
            return {"steps": [{"kind": "edit", "title": f"Fix: {name} (1/2: the first frame)", "params": edit},
                              {"kind": "video", "title": f"Fix: {name} (2/2: the clip)", "params": video}], "notes": notes}
        raise HTTPException(400, f"unknown action {kind}: one of {', '.join(KINDS)}")

    def shown(chain: dict) -> dict:
        steps = [{"id": one, "status": (studio.jobs.get(one) or {}).get("status", "gone")} for one in chain["steps"]]
        waiting = len(chain["pending"])
        if chain["error"]:
            status = "failed"
        elif any(s["status"] in ("failed", "cancelled", "gone") for s in steps):
            status = next(s["status"] for s in steps if s["status"] in ("failed", "cancelled", "gone"))
        elif not waiting and steps and steps[-1]["status"] == "done" and (not chain["join"] or chain["outputs"]):
            status = "done"
        else:
            status = "running"
        return {"id": chain["id"], "kind": chain["kind"], "title": chain["title"], "steps": steps, "of": len(steps) + waiting,
                "notes": chain["notes"], "error": chain["error"], "outputs": chain["outputs"], "status": status}

    def queue(body: dict) -> dict:
        """Queue an action's first step; the rest follow from ``then``. The first job, with ``revise`` (the chain)."""

        kind = body.get("kind")
        if kind not in KINDS:
            raise HTTPException(400, f"unknown action {kind}: one of {', '.join(KINDS)}")
        if body.get("params") is not None and not isinstance(body.get("params"), dict):
            raise HTTPException(400, "params is an object of the action's settings")
        made = plan(kind, dict(body.get("params") or {}))
        first, rest = made["steps"][0], made["steps"][1:]
        chain = {"id": uuid.uuid4().hex[:12], "kind": kind, "title": first["title"].split(" (")[0], "steps": [], "pending": rest,
                 "join": made.get("join"), "notes": made["notes"], "error": None, "outputs": [], "created": time.time()}
        if rest or chain["join"]:
            first["params"]["then"] = chain["id"]
        with guard:
            chains[chain["id"]] = chain
        try:
            job = studio.queue_job({"kind": first["kind"], "title": body.get("title") or first["title"], "params": first["params"],
                                    **({"sparks": body["sparks"]} if body.get("sparks") is not None else {})})
        except Exception:
            with guard:
                chains.pop(chain["id"], None)
            raise
        chain["steps"].append(job["id"])
        return {**job, "revise": shown(chain)}

    def join(source: Path, new: Path, target: Path) -> None:
        """The clip and its extension as one file: the extension's first frame repeats the clip's last, so it is
        dropped; the extension is brought to the clip's size and both to 24 frames a second and 48 kHz sound."""

        one, two = probe(source), probe(new)
        fps, width, height = studio.FPS, one["width"], one["height"]
        inputs = ["-i", str(source), "-i", str(new)]
        filters = [f"[0:v]fps={fps},scale={width}:{height},setsar=1,format=yuv420p[v0]",
                   f"[1:v]trim=start_frame=1,setpts=PTS-STARTPTS,fps={fps},scale={width}:{height},setsar=1,format=yuv420p[v1]"]
        sound = one["sound"] or two["sound"]
        if sound:
            for index, info in enumerate((one, two)):
                start = 1 / fps if index else 0
                if info["sound"]:
                    filters.append(f"[{index}:a]atrim=start={start},asetpts=PTS-STARTPTS,aresample=48000,"
                                   f"aformat=channel_layouts=stereo[a{index}]")
                else:  # silence for the part without sound, as long as its picture
                    length = max(info["duration"] - start, 0.05)
                    filters.append(f"anullsrc=r=48000:cl=stereo,atrim=duration={length:.3f}[a{index}]")
            filters.append("[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]")
            maps = ["-map", "[v]", "-map", "[a]", "-c:a", "aac", "-b:a", "256k"]
        else:
            filters.append("[v0][v1]concat=n=2:v=1:a=0[v]")
            maps = ["-map", "[v]"]
        target.parent.mkdir(parents=True, exist_ok=True)
        done = subprocess.run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", ";".join(filters), *maps,
                               "-c:v", "libx264", "-crf", "17", "-preset", "medium", "-pix_fmt", "yuv420p",
                               "-movflags", "+faststart", str(target)], capture_output=True, text=True,
                              env=studio.environment({}), timeout=1800)
        if done.returncode or not target.is_file():
            raise RuntimeError(f"ffmpeg could not join the clips: {done.stderr.strip()[-300:]}")

    def then(job: dict) -> None:
        """What follows a finished step of a chain: Extend's join, or Fix's second clip. Never fails the job itself:
        what went wrong is said on the chain and in the job's log."""

        chain = chains.get(str(job.get("params", {}).get("then") or ""))
        if chain is None or not chain["steps"] or chain["steps"][-1] != job["id"]:
            return
        try:
            if chain["join"]:
                new = studio.inside(job["outputs"][0])
                target = studio.HOME / "renders" / f"{job['id']}-extended.mp4"
                job["progress"] = {"stage": "joining with the clip it continues", "at": 0, "of": 1}
                join(studio.inside(chain["join"]), new, target)
                job["progress"] = None
                job["outputs"] = [studio.relative(target)] + [p for p in job["outputs"] if p != studio.relative(target)]
                chain["outputs"] = list(job["outputs"])
                job["log"] = (job.get("log") or "") + f"[studio] joined with {chain['join']}: {studio.relative(target)}\n"
            elif chain["pending"]:
                step = chain["pending"].pop(0)
                params = dict(step["params"])
                params["references"] = [job["outputs"][0]]  # the corrected first frame
                if chain["pending"]:
                    params["then"] = chain["id"]
                made = studio.queue_job({"kind": step["kind"], "title": step["title"], "params": params})
                chain["steps"].append(made["id"])
                job["log"] = (job.get("log") or "") + f"[studio] next step queued: {made['id']}\n"
        except Exception as error:  # noqa: BLE001
            job["progress"] = None
            chain["error"] = str(getattr(error, "detail", error))
            job["log"] = (job.get("log") or "") + f"[studio] the next step could not follow: {chain['error']}\n"

    @router.get("/api/revise")
    def listing() -> dict:
        return {"kinds": [{"kind": k, "title": TITLES[k]} for k in KINDS], "chains": [shown(c) for c in list(chains.values())]}

    @router.get("/api/revise/{chain_id}")
    def one(chain_id: str) -> dict:
        chain = chains.get(chain_id)
        if chain is None:
            raise HTTPException(404, "no such action")
        return shown(chain)

    @router.post("/api/revise")
    def submit(body: dict) -> dict:
        try:
            return queue(body)
        except (TypeError, ValueError, AttributeError, KeyError):
            raise HTTPException(400, "a setting was sent as the wrong type") from None

    return router, queue, then
