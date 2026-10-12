#!/usr/bin/env python3
"""TensorFold Studio web app: a create wizard, a production board and a clip editor over the pack's scripts.

    bash scripts/app.sh            # http://127.0.0.1:7870
    HOST=0.0.0.0 bash scripts/app.sh   # reachable from other machines on the network (no login: trusted LANs only)

One job runs at a time, in the order queued; every job is one of this pack's scripts with environment variables.
Projects, uploads and renders live under $STUDIO_HOME (default ~/TensorFoldStudio).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

HERE = Path(__file__).resolve().parent
PACK = HERE.parent
HOME = Path(os.environ.get("STUDIO_HOME", Path.home() / "TensorFoldStudio"))
for name in ("projects", "uploads", "renders", "scouts", "exports"):
    (HOME / name).mkdir(parents=True, exist_ok=True)

FPS = 24
DURATIONS = {5: 124, 8: 192, 10: 243, 15: 362}  # seconds -> frames (17n + 5)
# longer videos are chains of clips: each one opens on the last frame of the one before, and they are joined at the end
LONG = {30: "30 s", 60: "1 min", 120: "2 min", 300: "5 min", 600: "10 min", 900: "15 min", 1800: "30 min"}
LONGEST = max(DURATIONS)
# name -> (label, final size, environment). Generation happens at half size where X2 is set.
PRESETS = {
    "draft": ("Draft 1344x768 (generated at 672x384, 2x decoder)", (1344, 768), {"X2": "1"}),
    "2k": ("2K 2048x1152 (generated at 1024x576, 2x decoder)", (2048, 1152), {"TWOK": "1"}),
    "qhd": ("2K 2560x1440 (generated at 1280x736, 2x decoder)", (2560, 1440), {"QHD": "1"}),
    "small": ("Small 864x480 (native)", (864, 480), {"WIDTH": "864", "HEIGHT": "480"}),
    "native": ("Native 1344x768 (no 2x decoder)", (1344, 768), {"WIDTH": "1344", "HEIGHT": "768"}),
    "p720": ("720p 1280x720 (native)", (1280, 720), {"WIDTH": "1280", "HEIGHT": "736", "CROP": "1280x720"}),
    "p960": ("960p 1728x960 (generated at 480p, 2x decoder)", (1728, 960), {"X2": "1", "WIDTH": "1728", "HEIGHT": "960"}),
    "sharp": ("Wasserman Sharp Spec 1376x768 (native, one pass, no upscale)", (1376, 768), {"WIDTH": "1376", "HEIGHT": "768"}),
}
# Sam Wasserman's locked recipe for a sharp clip: this size with FastH3 at 8 passes (sparse attention 0.8), seed 42 and
# a retake at the next seed, 8 or 10 seconds, one generation pass with no upscale and no crop.
SHARP = {"preset": "sharp", "quality": "fh8", "seed": 42, "seconds": [8, 10]}
QA_TAKES = 2  # his QA gate (qa.py) follows each such clip: one that fails is rendered again at the next seed
QUALITIES = {
    "standard": ("Standard: Turbo 5 passes + base-model sound", {}),
    "high": ("High: 20 steps, no adapter, fast recipe (about 1.5x slower)", {"QUALITY": "high"}),
    "full": ("Full: plain 20 steps, no adapter (about 3x slower)", {"QUALITY": "full"}),
    "fast": ("Fast: Turbo 3 passes, adapter sound", {"POINTS": "4", "REVOICE": "0"}),
    "fh8": ("FastH3: 8 passes, sparse attention", {"ENGINE": "fasth3", "STEPS": "8"}),
    "fh4": ("FastH3: 4 passes, fastest and softer", {"ENGINE": "fasth3", "STEPS": "4"}),
    "fh20": ("FastH3: 20 passes, for prompts with several actions", {"ENGINE": "fasth3", "STEPS": "20"}),
}
SCOUT_SIZES = {"draft": (1344, 768), "2k": (1024, 576), "qhd": (1280, 736), "small": (864, 480),
               "native": (1344, 768), "p720": (1280, 736), "p960": (864, 480), "sharp": (1376, 768)}
# measured on a Mac Studio M5 Ultra, 8 second clips, standard quality, image step included (README)
MEASURED_8S = {"draft": 108, "2k": 299, "qhd": 635, "small": 174, "native": 709}
# FastH3 on the native engine (TensorFold 1.0's Zig + Metal runtime), 5 second text-to-video clips, same machine.
ZIG_ENGINE = Path(os.environ.get("PREFIX", Path.home() / ".local/opt/tensorfold-studio")) / "zig-engine"
# Measured through this app: 480p at 4, 8 and 20 passes (43, 66, 129 s) and 720p at 8 passes (199 to 217 s). The
# other cells are those plus the measured pass time (5.3 s at 480p, 20 s at 720p) and about 10 s for the 2x decoder.
ZIG_FASTH3_5S = {"fh4": {"small": 43, "p960": 53, "p720": 125, "qhd": 137, "draft": 31, "2k": 60, "native": 104},
                 "fh8": {"small": 66, "p960": 76, "p720": 205, "qhd": 217, "draft": 44, "2k": 93, "native": 175},
                 "fh20": {"small": 129, "p960": 139, "p720": 445, "qhd": 457, "draft": 83, "2k": 192, "native": 388}}
# Draft, 2K and Native at 8 passes are measured through this app too (44, 93 and 175 s; a pass is 3.2, 8.2 and 17.8 s,
# from which the 4 and 20 pass cells follow).
# How the time grows with length: a 10 second 720p clip took 388 s against about 205 s for 5 seconds, a 15 second 480p
# clip 231 s against 66 s (a pass is 22 s at 44.6K rows against 5.3 s at 15.4K).
ZIG_LENGTH_POWER = {"small": 1.15, "p960": 1.15, "draft": 1.15, "2k": 1.15, "p720": 0.95, "qhd": 0.95, "native": 0.95}
# A clip that starts from an image carries the image's rows and its vision tokens through every pass.
ZIG_FROM_IMAGE = 1.13  # measured through this app: 72 s from an image against 64 s from text, 480p, 8 passes
# FastH3 on the MLX engine (machines without the native engine):
# 5 second text-to-video clips, measured on the same machine: quality -> preset -> seconds
FASTH3_5S = {"fh4": {"small": 51, "p960": 61, "p720": 112, "qhd": 123},
             "fh8": {"small": 78, "p960": 88, "p720": 192, "qhd": 203},
             "fh20": {"small": 174, "p960": 184, "p720": 434, "qhd": 445}}
# Reference to video (MiniMax H3 Ref2VA, oneshot-setup.sh --ref2va): plain 20 passes with the pictures' rows carried
# through each, or LightX2V's 4-step Ref2VA adapter and the sound made again without it. Measured through
# scripts/video.sh on an M5 Ultra with one portrait reference, 5 seconds: preset -> seconds.
REF2VA_5S = {"small": 223}
REF2VA_TURBO_5S = {"small": 88}
# the same on a DGX Spark (oneshot-setup-spark.sh --ref2va), through ComfyUI's own blocks
SPARK_REF2VA = Path(os.environ.get("SPARK_MODELS", Path.home() / "spark-studio-models")) / "MiniMax-H3-Comfy"
SPARK_REF2VA_5S = {"small": 321}
SPARK_REF2VA_TURBO_5S = {"small": 107}
REF2VA_ADAPTER = "lightx2v_ref2v_4step_v0.1_ourlayout.safetensors"
REFERENCE_LIMIT = 9
# a size typed in by hand: multiples of 32, and no more pixels than the largest size the model is run at here
CUSTOM_SIDES, CUSTOM_PIXELS, CUSTOM_STEPS = (256, 1536), 1344 * 768, (1, 50)
LEAD = "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."

# On a DGX Spark (Linux, CUDA) the scripts hand over to spark/: FastH3 and Qwen-Image-2.1 through ComfyUI. There is no
# 2x decoder and no MiniMax H3 Turbo there, so only the native sizes and the FastH3 qualities are offered.
SPARK = sys.platform.startswith("linux")
if SPARK:
    PRESETS = {k: v for k, v in PRESETS.items() if "X2" not in v[2] and k not in ("2k", "qhd")}
    QUALITIES = {k: v for k, v in QUALITIES.items() if k.startswith("fh")}
# FastH3 on a DGX Spark (GB10), 5 second text-to-video clips with the models already loaded, measured through the
# scripts on TensorFold's CUDA engine with the 8-bit video decoder at 8 passes: 864x480 in 73 s (a pass 7.1 s, the
# decode 14 s), 1280x736 in 183 s (18 s, 28 s) and 1344x768 in about 195 s (20.5 s, 29 s). The 4 and 20 pass cells
# follow from the pass time.
SPARK_ENGINE_LIB = Path(os.environ.get("PREFIX", Path.home() / ".local/opt/tensorfold-studio")) / "tf-h3" / "libtf_h3.so"
SPARK_FASTH3_5S = {"fh4": {"small": 45, "p720": 111, "native": 113},
                   "fh8": {"small": 73, "p720": 183, "native": 195, "sharp": 202},  # sharp: Sam Wasserman's 330 to 380 s at 8 s
                   "fh20": {"small": 158, "p720": 399, "native": 441}}
# On ComfyUI's own blocks in bf16 (no engine built): 864x480 at 8 passes 118 s (a pass 10.5 s) and 1344x768 289 s
# (a pass 29.4 s with its sparse attention); 720p is scaled from 1344x768 by its rows.
SPARK_COMFY_5S = {"fh4": {"small": 76, "p720": 155, "native": 172},
                  "fh8": {"small": 118, "p720": 260, "native": 289},
                  "fh20": {"small": 244, "p720": 575, "native": 642}}


def machine() -> tuple[str, str]:
    """(platform, label) for the header: which build this is and what it runs on."""

    if SPARK:
        try:
            gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                                 text=True).stdout.strip().splitlines()[0]
        except (OSError, IndexError):
            gpu = "CUDA"
        return "spark", f"DGX SPARK · {gpu.upper()} · H3-CUDA"
    # no chip name: the same recipe runs from an M1 with 64 GB up
    return "mac", "MAC · APPLE SILICON MX · H3-METAL"


app = FastAPI(title="Wasserman Slate AI Studio, powered by TensorFold")
jobs: dict[str, dict] = {}
order: list[str] = []
lock = threading.Lock()
wake = threading.Event()
running: dict = {"process": None, "id": None}


def inside(path: str | Path) -> Path:
    """A path under the studio folder, or 400."""

    resolved = (HOME / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    if HOME.resolve() not in (resolved, *resolved.parents):
        raise HTTPException(400, "path is outside the studio folder")
    return resolved


def relative(path: Path) -> str:
    return str(Path(path).resolve().relative_to(HOME.resolve()))


def compose_references(fields: dict, count: int) -> str:
    """The Ref2VA model's own prompt form: who each picture is, what is kept of it, then the shot."""

    count = max(1, min(int(count), REFERENCE_LIMIT))
    names = [(text or "").strip() for text in (fields.get("subjects") or [])][:count]
    names += [""] * (count - len(names))
    subjects = [f"<Subject {i}> is {name or 'the main subject'} shown in <Picture {i}>." for i, name in enumerate(names, start=1)]
    kept = [f"<Subject {i}> (appears in [Shot 1]): fully_preserved - the identity, face, hair, clothing and colours "
            f"shown in <Picture {i}> are kept." for i in range(1, count + 1)]
    scene = (fields.get("scene") or "").strip()
    shot = " ".join(part.strip() for part in (scene, fields.get("camera", ""), fields.get("action", "")) if part and part.strip())
    spoken = []
    for index, line in enumerate(fields.get("dialogue") or [], start=1):
        text = (line.get("text") or "").strip()
        if not text:
            continue
        voice = (line.get("voice") or "a natural voice").strip()
        verb = "sings" if line.get("sung") else "speaks"
        language = (line.get("language") or "English").strip()
        spoken.append(f"<Subject {int(line.get('subject') or 1)}> (S{index}), in {voice}, {verb}: <d>[{language}] {text}</d>")
    if spoken:
        # a clip longer than its lines: without this the model may keep the mouth moving and invent words
        shot += (" " + " ".join(spoken) + " Mouth movements follow the words. After the last word the speaker stops"
                 " talking, closes the mouth and stays silent until the end of the clip.")
    else:
        shot += " Nobody speaks."
    sound = (fields.get("soundscape") or "Natural ambient sound for the scene.").strip()
    sound += " No speech after the spoken lines." if spoken else " No voices, no speech."
    music = (fields.get("music") or "N/A").strip()
    if music.upper() != "N/A":
        sound += f" Music: {music}."
    who = ", ".join(f"<Subject {i}>" for i in range(1, count + 1))
    return ("subject_definitions:\n" + "\n".join(subjects)
            + f"\n\nsummary:\n[subject reference] A new video with {who}, keeping how each looks in its picture. "
            + (scene or "A new scene.")
            + "\n\nretention_analysis:\n" + "\n".join(kept)
            + f"\n\ndetailed_description:\n[Shot 1] {shot.strip()} {sound}")


def compose(fields: dict, first_frame: bool) -> str:
    """MiniMax H3's three-field prompt from the wizard's parts."""

    shot = " ".join(part.strip() for part in (fields.get("scene", ""), fields.get("camera", ""),
                                               fields.get("action", "")) if part and part.strip())
    spoken = []
    for index, line in enumerate(fields.get("dialogue") or [], start=1):
        text = (line.get("text") or "").strip()
        if not text:
            continue
        voice = (line.get("voice") or "a natural voice").strip()
        verb = "sings" if line.get("sung") else "says"
        language = (line.get("language") or "English").strip()
        spoken.append(f"{(line.get('speaker') or '').strip() or 'The speaker'}, in {voice} (S{index}), {verb}: <d>[{language}] {text}</d>")
    if spoken:
        shot += (" " + " ".join(spoken) + " Mouth movements follow the words. After the last word the speaker stops"
                 " talking, closes the mouth and stays silent until the end of the clip.")
    else:
        shot += " Nobody speaks."
    sound = (fields.get("soundscape") or "Natural ambient sound for the scene.").strip()
    if "no speech" not in sound.lower():
        sound += " No speech after the spoken lines." if spoken else " No voices, no speech."
    music = (fields.get("music") or "N/A").strip()
    body = (f"integrated_multimodal_description: [Shot 1] {shot.strip()}\n\n"
            f"overall_soundscape: {sound}\n\nnon_diegetic_music: {music}")
    return f"{LEAD}\n\n{body}" if first_frame else body


def plan(total: int, piece: int = LONGEST) -> list[int]:
    """Clip lengths, in seconds, that add up to at least ``total``: whole pieces, then the shortest clip that covers the rest."""

    if piece not in DURATIONS:
        raise HTTPException(400, f"a segment is one of {sorted(DURATIONS)} seconds")
    pieces = [piece] * (total // piece)
    rest = total - piece * len(pieces)
    if rest > 0:
        pieces.append(min(d for d in DURATIONS if d >= rest))
    return pieces


def zig_ready() -> bool:
    """Whether FastH3 will run on the native engine here (scripts/video.sh decides the same way)."""

    if not (ZIG_ENGINE / "tf-h3-dit").is_file() or os.environ.get("FASTH3_ENGINE") == "mlx":
        return False
    try:
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout
    except OSError:
        return False
    return re.search(r"Apple M([5-9]|[1-9][0-9])", chip) is not None


try:
    M5 = re.search(r"Apple M([5-9]|[1-9][0-9])", subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                                               capture_output=True, text=True).stdout) is not None
except OSError:
    M5 = False


def turbo_ready() -> bool:
    """Whether MiniMax H3's own transformer and its Turbo adapter are installed (oneshot-setup.sh --turbo).

    The default install leaves them out: FastH3 needs the MiniMax H3 text encoder and decoders, not its transformer.
    """

    model = Path(os.environ.get("H3_MODEL_DIR", Path.home() / "h3-models/MiniMax-H3"))
    return any((model / "FL2VA" / "transformer").glob("*.safetensors")) and any((ZIG_ENGINE.parent / "adapters").glob("lightx2v*"))


def text_encoders() -> dict[str, str]:
    """Other MiniMax H3 text encoder files found on this machine: file name -> what TEXT_ENCODER is set to for it.
    A file put in the text_encoders folder beside the model (on a Spark, ComfyUI's own folder) is offered by name."""

    if SPARK:
        folder = SPARK_REF2VA.parent / "FastH3-Comfy/text_encoders"
        own = {"qwen3vl_32b_minimax_h3_int8_convrot.safetensors"}
        return {f.name: f.name for f in sorted(folder.glob("*minimax_h3*.safetensors")) if f.name not in own}
    folder = Path(os.environ.get("H3_MODEL_DIR", Path.home() / "h3-models/MiniMax-H3")).parent / "text_encoders"
    return {f.name: str(f) for f in sorted(folder.glob("*.safetensors"))}



def own_text_encoder(asked: str) -> str | None:
    """A text encoder file of the user's own choosing: what TEXT_ENCODER is set to for it, or None when there is no
    such file. On a Mac a path; on a Spark the name of a file in ComfyUI's text_encoders folder (a path's last part)."""
    if not asked.endswith(".safetensors"):
        return None
    if SPARK:
        name = Path(asked).name
        return name if (SPARK_REF2VA.parent / "FastH3-Comfy/text_encoders" / name).is_file() else None
    path = Path(asked).expanduser()
    return str(path) if path.is_file() else None

def ref2va_ready() -> bool:
    """Whether reference to video can run here: MiniMax H3's Ref2VA transformer (oneshot-setup.sh --ref2va) beside
    the text encoder and decoders every install has. A Mac feature: the passes run on the MLX engine."""

    if SPARK:
        return any(SPARK_REF2VA.glob("diffusion_models/minimax_h3_ref2va_pruned_*.safetensors"))
    model = Path(os.environ.get("H3_MODEL_DIR", Path.home() / "h3-models/MiniMax-H3"))
    return any((model / "Ref2VA" / "transformer").glob("*.safetensors"))


def custom(params: dict) -> tuple[dict, bool]:
    """(environment, whether anything was set) for a size or a number of passes chosen by hand."""

    env = {}
    if params.get("preset") == "custom":
        try:
            width, height = int(params["width"]), int(params["height"])
        except (KeyError, TypeError, ValueError):
            raise HTTPException(400, "a custom size needs a width and a height")
        low, high = CUSTOM_SIDES
        if width % 32 or height % 32 or not (low <= width <= high and low <= height <= high):
            raise HTTPException(400, f"width and height are multiples of 32 between {low} and {high}")
        if width * height > CUSTOM_PIXELS:
            raise HTTPException(400, f"at most {CUSTOM_PIXELS} pixels (1344x768), got {width}x{height}")
        env.update(WIDTH=str(width), HEIGHT=str(height))
    if params.get("steps") not in (None, ""):
        try:
            steps = int(params["steps"])
        except (TypeError, ValueError):
            raise HTTPException(400, "passes is a whole number")
        if not CUSTOM_STEPS[0] <= steps <= CUSTOM_STEPS[1]:
            raise HTTPException(400, f"passes is between {CUSTOM_STEPS[0]} and {CUSTOM_STEPS[1]}")
        env["STEPS"] = str(steps)
    if params.get("text_encoder") not in (None, "", "regular"):
        asked = str(params["text_encoder"]).strip()
        found = text_encoders().get(asked) or own_text_encoder(asked)
        if not found:
            raise HTTPException(400, f"no text encoder file at {asked} on this machine (a .safetensors file"
                                     + (" in ComfyUI's text_encoders folder)" if SPARK else ")"))
        env["TEXT_ENCODER"] = found
    return env, bool(env)


def ref2va_turbo_ready() -> bool:
    if SPARK:
        return ((SPARK_REF2VA / "diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors").is_file()
                and (SPARK_REF2VA / "loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors").is_file())
    return ref2va_ready() and (ZIG_ENGINE.parent / "adapters" / REF2VA_ADAPTER).is_file()


def references(params: dict) -> list[str]:
    """The job's reference pictures as paths under the studio folder, or [] when it has none."""

    found = [str(inside(path)) for path in (params.get("references") or []) if path]
    if not found:
        return []
    if len(found) > REFERENCE_LIMIT:
        raise HTTPException(400, f"at most {REFERENCE_LIMIT} reference pictures")
    if params.get("image") or params.get("first_frame_from"):
        raise HTTPException(400, "reference pictures do not combine with a first frame")
    if not ref2va_ready():
        raise HTTPException(400, "MiniMax H3 Ref2VA is not installed here (oneshot-setup.sh --ref2va adds it)")
    if params.get("ref_turbo") and not ref2va_turbo_ready():
        raise HTTPException(400, "the Ref2VA Turbo adapter is not installed here (oneshot-setup.sh --ref2va adds it)")
    missing = [path for path in found if not Path(path).is_file()]
    if missing:
        raise HTTPException(400, f"no reference picture at {relative(Path(missing[0]))}")
    return found


def estimate(preset: str, quality: str, seconds: int, text_only: bool = False, referenced: bool = False,
             ref_turbo: bool = False) -> int | None:
    if referenced:
        # the same MLX engine as the 20-step qualities, whose time grows with the length to the power 1.6
        if SPARK:
            base = (SPARK_REF2VA_TURBO_5S if ref_turbo else SPARK_REF2VA_5S).get(preset)
        else:
            base = (REF2VA_TURBO_5S if ref_turbo else REF2VA_5S).get(preset)
        return None if base is None or seconds > LONGEST else int(base * (seconds / 5.0) ** 1.6)
    if seconds > LONGEST:
        # every part after the first opens on a frame
        parts = [estimate(preset, quality, piece) for piece in plan(seconds)]
        return None if None in parts else int(sum(parts) + 2 * len(parts))
    if SPARK:
        base = (SPARK_FASTH3_5S if SPARK_ENGINE_LIB.exists() else SPARK_COMFY_5S).get(quality, {}).get(preset)
        # the pass time grows a little faster than the length: attention over all rows is part of every pass
        return None if base is None else int(base * (seconds / 5.0) ** 1.2 * (1.0 if text_only else 1.5))
    if zig_ready() and preset in ZIG_FASTH3_5S.get(quality, {}):
        return int(ZIG_FASTH3_5S[quality][preset] * (seconds / 5.0) ** ZIG_LENGTH_POWER[preset] * (1.0 if text_only else ZIG_FROM_IMAGE))
    if quality.startswith("fh") and not M5:
        return None  # FastH3 on an earlier chip runs with 8-bit weights on the MLX engine; no time is measured there
    if quality in FASTH3_5S and preset in FASTH3_5S[quality]:
        return int(FASTH3_5S[quality][preset] * (seconds / 5.0) ** 1.3)
    if quality.startswith("fh"):
        return None
    base = MEASURED_8S.get(preset)
    if base is None:
        return None
    scale = {"standard": 1.0, "high": 1.45, "full": 3.0, "fast": 0.58}[quality] * (seconds / 8.0) ** 1.6
    return int(base * scale)


def environment(extra: dict) -> dict:
    env = dict(os.environ)
    env.update({key: str(value) for key, value in extra.items() if value not in (None, "")})
    env["PATH"] = "/opt/homebrew/bin:" + env.get("PATH", "")
    return env


def build(kind: str, params: dict, job_id: str) -> tuple[list[str], dict, list[str]]:
    """(command, environment, output paths) for one job."""

    scripts = PACK / "scripts"
    seed = str(int(params.get("seed", 0)))
    if kind == "scout":
        folder = HOME / "scouts" / job_id
        prompt = params["prompt"]
        count = max(1, min(int(params.get("count", 4)), 12))
        # a scout is made at the size the video model will start from (the draft keeps a full 1344x768 still)
        width, height = SCOUT_SIZES.get(params.get("preset", "2k"), (1024, 576))
        if params.get("width") and params.get("height"):  # an image for its own sake, at the size asked for
            width, height = int(params["width"]), int(params["height"])
            if width % 16 or height % 16 or not (256 <= width <= 2048 and 256 <= height <= 2048):
                raise HTTPException(400, "image sides are multiples of 16 between 256 and 2048")
        env = {"SEED": seed, "WIDTH": width, "HEIGHT": height}
        first = int(seed)
        outputs = [str(folder / f"scout_s{first + i}.png") for i in range(count)] + [str(folder / "sheet.jpg")]
        return ["bash", str(scripts / "scout.sh"), prompt, str(count), str(folder)], env, outputs
    if kind == "export":
        return ["python3", "-c", "pass"], {}, []
    preset = params.get("preset", "small" if SPARK else "draft")
    quality = params.get("quality", "standard")
    seconds = int(params.get("seconds", 5))
    chosen, _ = custom(params)
    if seconds not in DURATIONS or (preset not in PRESETS and preset != "custom") or quality not in QUALITIES:
        raise HTTPException(400, "unknown duration, preset or quality")
    target = Path(params["output"]) if params.get("output") else HOME / "renders" / f"{job_id}.mp4"
    target = inside(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    env = {"SEED": seed, "FRAMES": DURATIONS[seconds], **(PRESETS[preset][2] if preset in PRESETS else {}), **QUALITIES[quality][1]}
    fast_h3 = SPARK or env.get("ENGINE") == "fasth3"
    prompt = params["prompt"]
    image = params.get("image")
    styles = lora_resolve(params.get("loras"))  # the LoRA hub (loras.py): files to merge, and their trigger words
    if styles:
        if kind == "video" and references(params):
            raise HTTPException(400, "a LoRA from the hub is for MiniMax H3 and FastH3 clips, not for reference to video")
        env["LORA"] = "\n".join(f"{one['path']}:{one['strength']:g}" for one in styles) + "\n"
        prompt = lora_lead(prompt, styles)
    if kind == "video":  # text to video, or from an image the user already has
        pictures = references(params)
        if pictures:
            # Ref2VA takes neither the Turbo adapter nor FastH3: the quality setting does not apply
            env = {k: v for k, v in env.items() if k not in QUALITIES[quality][1]}
            env["REF_IMAGES"] = "\n".join(pictures) + "\n"  # one per line, so a path may hold spaces
            if params.get("ref_turbo"):
                env["REF_TURBO"] = "1"
        env.update(passes(chosen, SPARK or (fast_h3 and not pictures)))  # Ref2VA runs on MiniMax H3
        if image:
            return (["bash", str(scripts / "studio.sh"), "", prompt, str(target)],
                    {**env, "IMAGE_FILE": str(inside(image)), "RAW_PROMPT": "1"}, [str(target)])
        size = PRESETS[preset][1] if preset in PRESETS else (int(env["WIDTH"]), int(env["HEIGHT"]))
        plain = {k: v for k, v in env.items() if k not in ("TWOK", "QHD")}
        if "TWOK" in env:
            plain.update(X2="1", WIDTH="2048", HEIGHT="1152")
        if "QHD" in env:
            plain.update(X2="1", WIDTH="2560", HEIGHT="1472", CROP="2560x1440")
        if "X2" in env and "WIDTH" not in plain:
            plain.update(WIDTH=str(size[0]), HEIGHT=str(size[1]))
        return ["bash", str(scripts / "video.sh"), prompt, str(target)], plain, [str(target)]
    if kind == "studio":  # text to image to video in one go
        env.update(passes(chosen, fast_h3))
        return (["bash", str(scripts / "studio.sh"), params["image_prompt"], prompt, str(target)],
                {**env, "RAW_PROMPT": "1"}, [str(target), str(target.with_suffix(".png"))])
    raise HTTPException(400, f"unknown job kind {kind}")


def passes(chosen: dict, steps: bool) -> dict:
    """A hand-set number of passes in the form the engine takes: STEPS for FastH3 and on a Spark, sigma points (one
    more than the passes) for MiniMax H3 on the Mac."""

    out = dict(chosen)
    if "STEPS" in out and not steps:
        out["POINTS"] = str(int(out.pop("STEPS")) + 1)
    return out


def last_frame(video: Path, image: Path) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-sseof", "-0.2", "-i", str(video), "-vsync", "0", "-update", "1",
                    "-q:v", "1", str(image)], check=True, env=environment({}))


def export(job: dict) -> None:
    """Cut each timeline clip to its in and out points and join them, with a short audio fade at each join."""

    params = job["params"]
    clips = params["clips"]
    if not clips:
        raise RuntimeError("the timeline is empty")
    width, height = int(params.get("width", 1920)), int(params.get("height", 1080))
    target = HOME / "exports" / f"{job['id']}.mp4"
    inputs, filters, labels = [], [], []
    for index, clip in enumerate(clips):
        source = inside(clip["path"])
        start = max(0.0, float(clip.get("in", 0.0)))
        end = float(clip.get("out", 0.0))
        inputs += ["-i", str(source)]
        span = f"trim=start={start}:end={end},setpts=PTS-STARTPTS," if end > start else ""
        aspan = f"atrim=start={start}:end={end},asetpts=PTS-STARTPTS," if end > start else ""
        length = (end - start) if end > start else float(clip.get("duration", 5.0))
        fade = min(0.08, length / 4)
        filters.append(f"[{index}:v]{span}scale={width}:{height}:force_original_aspect_ratio=decrease,"
                       f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={FPS}[v{index}]")
        filters.append(f"[{index}:a]{aspan}afade=t=in:d={fade},afade=t=out:st={max(length - fade, 0)}:d={fade},"
                       f"aresample=48000[a{index}]")
        labels.append(f"[v{index}][a{index}]")
    filters.append(f"{''.join(labels)}concat=n={len(clips)}:v=1:a=1[v][a]")
    command = ["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", ";".join(filters), "-map", "[v]", "-map",
               "[a]", "-c:v", "libx264", "-crf", "17", "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac",
               "-b:a", "256k", str(target)]
    done = subprocess.run(command, capture_output=True, text=True, env=environment({}))
    job["log"] += done.stderr
    if done.returncode:
        raise RuntimeError("ffmpeg failed")
    job["outputs"] = [relative(target)]


STEP = re.compile(r"\[tensorfold\] (step|voice) (\d+)/(\d+)")
RAN_ON = re.compile(r"\[tensorfold\] engine: (\w+)")


def run(job: dict, command: list[str], env: dict, label: str = "") -> bool:
    """Run one script for ``job``, following its progress; False when the job was cancelled meanwhile."""

    process = subprocess.Popen(command, cwd=PACK, env=environment(env), stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, start_new_session=True)
    running.update(process=process, id=job["id"])
    for line in process.stdout:
        job["log"] = (job["log"] + line)[-20000:]
        engine = RAN_ON.search(line)
        if engine:
            job["engine"] = engine.group(1)
        found = STEP.search(line)
        if found:
            kind, at, total = found.group(1), int(found.group(2)), int(found.group(3))
            job["progress"] = {"stage": label + ("picture" if kind == "step" else "sound"), "at": at, "of": total}
    code = process.wait()
    running.update(process=None, id=None)
    if job["status"] == "cancelled":
        return False
    if code:
        raise RuntimeError(f"the script exited with status {code}")
    return True


def gated(job: dict, clip: Path) -> bool:
    """Sam Wasserman's QA gate on a clip made to his Sharp Spec; False when the job was cancelled meanwhile."""

    params, takes = job["params"], []
    for take in range(1, QA_TAKES + 1):
        if take > 1:
            shutil.copy(clip, HOME / "qa" / f"{job['id']}-take{take - 1}.mp4")  # the failed take is kept
            params["seed"] = int(params.get("seed", 0)) + 1
            command, env, _ = build("video", params, job["id"])
            if not run(job, command, env, f"take {take} · "):
                return False
        job["progress"] = {"stage": "QA gate, take", "at": take, "of": QA_TAKES}
        picture = HOME / "qa" / f"{job['id']}-take{take}.jpg"
        try:
            report = qa.check(clip, params["prompt"], picture, params.get("brain"))
        except Exception as error:  # noqa: BLE001 - the clip is made; a gate that cannot run does not lose it
            job["qa"] = {"passed": None, "takes": takes, "error": str(error)[-300:]}
            break
        takes.append({**report, "take": take, "seed": int(params.get("seed", 0)), "sheet": relative(picture)})
        job["qa"] = {"passed": report["passed"], "takes": takes}
        if report["passed"] or job["status"] == "cancelled":
            break
    return job["status"] != "cancelled"


def long_take(job: dict) -> bool:
    """A video longer than one clip: render it piece by piece, each opening on the last frame of the one before, and join."""

    params = job["params"]
    total = int(params["total_seconds"])
    pieces = plan(total, int(params.get("segment_seconds", LONGEST)))
    beats = [line.strip() for line in params.get("beats") or [] if line.strip()]
    folder = HOME / "renders" / job["id"]
    folder.mkdir(parents=True, exist_ok=True)
    image, clips = params.get("image"), []
    for index, seconds in enumerate(pieces):
        prompt = params["prompt"]
        if beats:  # one line per piece says what happens in it; the last line holds for the rest
            beat = f"In this part of the video: {beats[min(index, len(beats) - 1)].rstrip('.')}."
            prompt = prompt.replace("[Shot 1] ", f"[Shot 1] {beat} ", 1) if "[Shot 1] " in prompt else f"{beat} {prompt}"
        if index:
            still = folder / f"seg{index:03d}_first.jpg"
            last_frame(clips[-1], still)
            image = relative(still)
        if image and not prompt.startswith("For the target video"):
            prompt = f"{LEAD}\n\n{prompt}"
        target = folder / f"seg{index + 1:03d}.mp4"
        piece = {**params, "prompt": prompt, "image": image, "seconds": seconds, "output": relative(target),
                 "seed": int(params.get("seed", 0)) + index}
        command, env, _ = build("video", piece, job["id"])
        if not run(job, command, env, f"part {index + 1}/{len(pieces)} · "):
            return False
        if not target.is_file():
            raise RuntimeError(f"part {index + 1} wrote no clip")
        clips.append(target)
        job["outputs"] = [relative(c) for c in clips]  # finished parts can be watched while the rest renders
    # join: every part after the first repeats the frame it started from, so that frame is dropped
    job["progress"] = {"stage": "joining", "at": len(clips), "of": len(clips)}
    listing = folder / "parts.txt"
    trimmed = []
    for index, clip in enumerate(clips):
        part = folder / f"join{index + 1:03d}.ts"
        cut = ["-vf", "trim=start_frame=1,setpts=PTS-STARTPTS", "-af", f"atrim=start={1 / FPS},asetpts=PTS-STARTPTS"] if index else []
        done = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(clip), *cut, "-c:v", "libx264", "-crf", "17",
                               "-preset", "medium", "-pix_fmt", "yuv420p", "-r", str(FPS), "-c:a", "aac", "-b:a",
                               "256k", "-ar", "48000", "-f", "mpegts", str(part)], capture_output=True, text=True,
                              env=environment({}))
        if done.returncode:
            job["log"] += done.stderr
            raise RuntimeError("ffmpeg failed while preparing a part")
        trimmed.append(part)
    listing.write_text("".join(f"file '{part}'\n" for part in trimmed))
    final = HOME / "renders" / f"{job['id']}.mp4"
    done = subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-t",
                           str(total), "-c", "copy", "-movflags", "+faststart", str(final)], capture_output=True,
                          text=True, env=environment({}))
    for part in trimmed:
        part.unlink(missing_ok=True)
    if done.returncode:
        job["log"] += done.stderr
        raise RuntimeError("ffmpeg failed while joining the parts")
    job["outputs"] = [relative(final)] + [relative(c) for c in clips]
    return True


def work() -> None:
    while True:
        wake.wait()
        with lock:
            job = next((jobs[i] for i in order if jobs[i]["status"] == "queued"), None)
            if job is None:
                wake.clear()
                continue
            job["status"], job["started"] = "running", time.time()
        try:
            params = job["params"]
            if params.get("first_frame_from"):  # continuity: open on the last frame of an earlier clip
                source = inside(params["first_frame_from"])
                if not source.is_file():
                    raise RuntimeError("the clip this one continues from has not been rendered")
                still = HOME / "uploads" / f"{job['id']}_first.jpg"
                last_frame(source, still)
                params["image"] = relative(still)
            if job["kind"] == "export":
                export(job)
            elif job["kind"] == "long":
                if not long_take(job):
                    continue
            else:
                command, env, outputs = build(job["kind"], params, job["id"])
                if not run(job, command, env):
                    continue
                job["outputs"] = [relative(Path(p)) for p in outputs if Path(p).is_file()]
                if not job["outputs"]:
                    raise RuntimeError("the script finished without writing its output")
                if job["kind"] == "video" and params.get("preset") == "sharp" and params.get("qa", True):
                    if not gated(job, Path(outputs[0])):
                        continue
            job["status"] = "done"
        except Exception as error:  # noqa: BLE001 - a failed job must not stop the queue
            if job["status"] != "cancelled":
                job["status"], job["error"] = "failed", str(error)
        finally:
            job["finished"] = time.time()


threading.Thread(target=work, daemon=True).start()


def public(job: dict) -> dict:
    out = {k: job.get(k) for k in ("id", "kind", "status", "title", "outputs", "progress", "error", "created",
                                    "started", "finished", "estimate", "engine", "qa")}
    # by this machine's clock: the page may be open on another one
    out["elapsed"] = (job.get("finished") or time.time()) - job["started"] if job.get("started") else None
    out["log"] = job["log"][-3000:]
    out["params"] = {k: v for k, v in job["params"].items() if k != "clips"}
    return out


@app.get("/api/state")
def state() -> dict:
    return {"presets": {k: {"label": v[0], "size": v[1]} for k, v in PRESETS.items()},
            "qualities": {k: v[0] for k, v in QUALITIES.items()}, "durations": DURATIONS, "long": LONG,
            "home": str(HOME), "pack": str(PACK), "version": (PACK / "VERSION").read_text().strip(),
            "zig": zig_ready(), "turbo": turbo_ready() and not SPARK, "x2": not SPARK,
            "ref2va": ref2va_ready(), "ref2va_turbo": ref2va_turbo_ready(), "reference_limit": REFERENCE_LIMIT,
            "text_encoders": sorted(text_encoders()), "sharp": SHARP if "sharp" in PRESETS and "fh8" in QUALITIES else None, "custom": {"sides": CUSTOM_SIDES, "pixels": CUSTOM_PIXELS, "steps": CUSTOM_STEPS},
            "platform": machine()[0], "platform_label": machine()[1],
            "engine": ("CUDA · TensorFold int8" if SPARK_ENGINE_LIB.exists() else "CUDA · ComfyUI") if SPARK else ("native Zig engine" if zig_ready() else "MLX engine")}


@app.post("/api/compose")
def compose_prompt(body: dict) -> dict:
    if body.get("references"):
        return {"prompt": compose_references(body.get("fields", {}), int(body["references"]))}
    return {"prompt": compose(body.get("fields", {}), bool(body.get("first_frame")))}


@app.post("/api/estimate")
def estimate_time(body: dict) -> dict:
    if body.get("preset") == "custom" or body.get("steps") not in (None, "") or body.get("loras"):
        return {"seconds": None}  # a LoRA changes which engine renders the clip: no time is measured for it
    return {"seconds": estimate(body.get("preset", "draft"), body.get("quality", "standard"),
                                int(body.get("seconds", 5)), bool(body.get("text_only")), bool(body.get("referenced")),
                                bool(body.get("ref_turbo")))}


@app.post("/api/jobs")
def submit(body: dict) -> dict:
    kind = body.get("kind")
    params = dict(body.get("params") or {})
    if kind not in ("scout", "video", "studio", "export", "long"):
        raise HTTPException(400, "unknown job kind")
    if kind == "long":
        total = int(params.get("total_seconds", 0))
        if not LONGEST < total <= max(LONG):
            raise HTTPException(400, f"a long video is over {LONGEST} seconds and at most {max(LONG) // 60} minutes")
        pieces = plan(total, int(params.get("segment_seconds", LONGEST)))
        build("video", {**params, "seconds": pieces[0], "output": None}, "check")
    if kind != "export" and not (params.get("prompt") or "").strip():
        raise HTTPException(400, "a prompt is needed")
    if kind != "video" and params.get("references"):
        raise HTTPException(400, "reference pictures are for single clips")
    referenced = kind == "video" and bool(references(params))
    if kind in ("video", "studio", "long") and not referenced and not str(params.get("quality", "standard")).startswith("fh") and not turbo_ready():
        raise HTTPException(400, "MiniMax H3 Turbo is not installed here (oneshot-setup.sh --turbo adds it): use FastH3, quality fh4, fh8 or fh20")
    job_id = time.strftime("%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    if kind not in ("export", "long"):
        build(kind, params, job_id)  # reject bad settings now, not when the job's turn comes
    job = {"id": job_id, "kind": kind, "status": "queued", "title": body.get("title") or kind, "params": params,
           "outputs": [], "progress": None, "error": None, "log": "", "created": time.time(), "started": None,
           "finished": None, "estimate": None if params.get("preset") == "custom" or params.get("steps") not in (None, "") or params.get("loras") else estimate(params.get("preset", ""), params.get("quality", "standard"),
                                                  int(params.get("total_seconds") or params.get("seconds", 5)),
                                                  kind == "video" and not params.get("image")
                                                  and not params.get("first_frame_from"), referenced,
                                                  bool(params.get("ref_turbo")))
           if kind in ("video", "studio", "long") else None}
    with lock:
        jobs[job_id] = job
        order.append(job_id)
    wake.set()
    return public(job)


@app.get("/api/jobs")
def list_jobs() -> list[dict]:
    return [public(jobs[i]) for i in reversed(order)]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    if job_id not in jobs:
        raise HTTPException(404, "no such job")
    return public(jobs[job_id])


@app.post("/api/jobs/{job_id}/cancel")
def cancel(job_id: str) -> dict:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    if job["status"] in ("queued", "running"):
        job["status"] = "cancelled"
        process = running["process"] if running["id"] == job_id else None
        if process is not None:
            os.killpg(process.pid, signal.SIGTERM)
    return public(job)


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "image.png").suffix.lower()
    if suffix not in (".png", ".jpg", ".jpeg", ".webp", ".mp4", ".mov", ".safetensors"):  # .safetensors: a LoRA for the hub
        raise HTTPException(400, "images (png, jpg, webp) and videos (mp4, mov) only")
    target = HOME / "uploads" / f"{uuid.uuid4().hex[:10]}{suffix}"
    with open(target, "wb") as handle:
        shutil.copyfileobj(file.file, handle)
    return {"path": relative(target)}


@app.get("/api/library")
def library() -> list[dict]:
    """Every clip and still under the studio folder, newest first."""

    found = []
    for folder, kinds in (("renders", (".mp4",)), ("exports", (".mp4",)), ("scouts", (".png",)),
                          ("uploads", (".png", ".jpg", ".jpeg", ".webp", ".mp4", ".mov")), ("projects", (".mp4",)), ("slate", (".mp4",))):
        for path in (HOME / folder).rglob("*"):
            if path.suffix.lower() in kinds and path.is_file():
                found.append({"path": relative(path), "kind": "video" if path.suffix.lower() in (".mp4", ".mov")
                              else "image", "folder": folder, "modified": path.stat().st_mtime,
                              "size": path.stat().st_size})
    return sorted(found, key=lambda item: -item["modified"])[:400]


@app.delete("/api/library")
def discard(path: str) -> dict:
    """Take a clip or still out of the library: it is moved to the studio's trash folder, not erased."""

    target = inside(path)
    if not target.is_file() or relative(target).split("/")[0] not in ("renders", "exports", "scouts", "uploads", "projects", "slate") \
            or target.suffix.lower() not in (".mp4", ".mov", ".png", ".jpg", ".jpeg", ".webp"):
        raise HTTPException(400, "not a library file")
    with lock:
        if any(job["status"] in ("queued", "running") and relative(target) in json.dumps(job["params"]) for job in jobs.values()):
            raise HTTPException(400, "a queued or running job uses this file")
    kept = HOME / "trash" / (time.strftime("%Y%m%d-%H%M%S-") + relative(target).replace("/", "__"))
    kept.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(target), str(kept))
    return {"trash": relative(kept)}


@app.get("/api/probe")
def probe(path: str) -> dict:
    target = inside(path)
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,nb_frames:format=duration", "-of", "json", str(target)],
                         capture_output=True, text=True, env=environment({}))
    data = json.loads(out.stdout or "{}")
    stream = (data.get("streams") or [{}])[0]
    return {"width": stream.get("width"), "height": stream.get("height"), "frames": stream.get("nb_frames"),
            "duration": float((data.get("format") or {}).get("duration") or 0)}


def project_file(project_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", project_id):
        raise HTTPException(400, "bad project id")
    return HOME / "projects" / project_id / "project.json"


@app.get("/api/projects")
def projects() -> list[dict]:
    out = []
    for file in sorted((HOME / "projects").glob("*/project.json"), key=lambda f: -f.stat().st_mtime):
        data = json.loads(file.read_text())
        out.append({"id": data["id"], "name": data.get("name", data["id"]), "segments": len(data.get("segments", [])),
                    "modified": file.stat().st_mtime})
    return out


@app.post("/api/projects")
def new_project(body: dict) -> dict:
    project_id = time.strftime("p%m%d-%H%M%S")
    data = {"id": project_id, "name": body.get("name") or "Untitled", "preset": "small" if SPARK else "draft", "quality": "fh8" if SPARK else "standard",
            "segments": [], "timeline": []}
    file = project_file(project_id)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(data, indent=1))
    return data


@app.get("/api/projects/{project_id}")
def get_project(project_id: str) -> dict:
    file = project_file(project_id)
    if not file.is_file():
        raise HTTPException(404, "no such project")
    return json.loads(file.read_text())


@app.put("/api/projects/{project_id}")
def save_project(project_id: str, body: dict) -> dict:
    file = project_file(project_id)
    if not file.is_file():
        raise HTTPException(404, "no such project")
    body["id"] = project_id
    file.write_text(json.dumps(body, indent=1))
    return body


@app.get("/api/wait")
def wait(ms: int = 1000):
    """Holds a response open; a page asked for with ?shot=1 loads an image from here so that a headless
    screenshot is taken after the app has drawn."""

    time.sleep(min(max(ms, 0), 10000) / 1000)
    return JSONResponse({"waited": ms})


@app.get("/files/{path:path}")
def files(path: str):
    target = inside(path)
    if not target.is_file():
        raise HTTPException(404, "no such file")
    return FileResponse(target)


@app.exception_handler(HTTPException)
async def errors(_, error: HTTPException):
    return JSONResponse({"error": error.detail}, status_code=error.status_code)


import loras  # noqa: E402  (Keys and Wasserman LoRA Hub, beside this file)
lora_router, lora_resolve, lora_lead = loras.attach(HOME, PACK, SPARK)
app.include_router(lora_router)
import qa  # noqa: E402  (Sam Wasserman's QA gate, beside this file)
import slate  # noqa: E402  (Wasserman Slate AI Studio, beside this file)

app.include_router(slate.attach(HOME, DURATIONS, compose, compose_references, REFERENCE_LIMIT,
                                lambda: {"ref2va": ref2va_ready(), "ref2va_turbo": ref2va_turbo_ready()}))
import motion  # noqa: E402  (Motion Previs: a Python port of Sam Wasserman's Motion Previs Studio analysis, beside this file)
app.include_router(motion.attach(HOME, PACK, SPARK))
app.mount("/", StaticFiles(directory=HERE / "static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "7870")),
                log_level="warning")
