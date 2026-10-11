"""Keys and Wasserman LoRA Hub: a library of style LoRAs for MiniMax H3 that a clip can be rendered with.

A recipe says where a LoRA's file comes from and how it is used: its trigger word, the phrases that pick one of its
looks, the strength it was made for. The presets that ship with the pack are in lora-presets.json beside this file;
recipes a user adds are kept in ~/TensorFoldStudio/loras/hub.json. No weights ship with the pack: a LoRA's file is
downloaded from its author's own page when the user installs it, under that file's own licence.

A job carries `loras: [{"id", "strength", "style"}]`. On a Mac the file is rewritten once into the layout the MLX
engine merges at load (scripts/convert-lora.py); on a DGX Spark ComfyUI merges the file as it is.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from fastapi import APIRouter, HTTPException

PRESETS = Path(__file__).with_name("lora-presets.json")
FIELDS = ("name", "author", "url", "page", "sha256", "trigger", "styles", "strength", "range", "base", "licence", "notes")
LIMIT = 4  # LoRAs on one clip


def attach(home: Path, pack: Path, spark: bool):
    folder = home / "loras"
    folder.mkdir(parents=True, exist_ok=True)
    book = folder / "hub.json"
    router = APIRouter()
    busy: dict[str, dict] = {}  # id -> {"stage", "done", "of"} while a file is being fetched
    lock = threading.Lock()

    def added() -> list[dict]:
        try:
            return json.loads(book.read_text())
        except (OSError, ValueError):
            return []

    def recipes() -> list[dict]:
        built = [{**r, "builtin": True} for r in json.loads(PRESETS.read_text())]
        taken = {r["id"] for r in built}
        return built + [{**r, "builtin": False} for r in added() if r.get("id") not in taken]

    def find(lora_id: str) -> dict:
        found = next((r for r in recipes() if r["id"] == lora_id), None)
        if found is None:
            raise HTTPException(404, f"no LoRA {lora_id} in the hub")
        return found

    def source(lora_id: str) -> Path:
        return folder / f"{lora_id}.safetensors"

    def ready(lora_id: str) -> Path:
        """The file a render reads: the author's on a Spark, the rewritten one on a Mac."""

        return source(lora_id) if spark else folder / f"{lora_id}.mlx.safetensors"

    def public(recipe: dict) -> dict:
        state = busy.get(recipe["id"])
        return {**recipe, "installed": ready(recipe["id"]).is_file() and not state, "installing": state}

    def check(path: Path) -> int:
        """The number of adapted modules, after making sure the file is a low-rank LoRA for MiniMax H3's transformer."""

        with open(path, "rb") as handle:
            head = handle.read(8)
            size = struct.unpack("<Q", head)[0] if len(head) == 8 else 0
            if not 0 < size < 50_000_000:
                raise ValueError("not a safetensors file")
            names = [k for k in json.loads(handle.read(size)) if k != "__metadata__"]
        low = [k for k in names if k.endswith((".lora_A.weight", ".lora_down.weight"))]
        if not low or not any(re.search(r"(^|\.)blocks\.\d+\.(attn|mlp)\.", k) for k in low):
            raise ValueError("not a LoRA for MiniMax H3's transformer (no lora_A tensors on blocks.N.attn / blocks.N.mlp)")
        return len(low)

    def prepare(lora_id: str) -> None:
        if spark or ready(lora_id).is_file():
            return
        model = os.environ.get("H3_MODEL_DIR", str(Path.home() / "h3-models/MiniMax-H3"))
        part = folder / f"{lora_id}.part.safetensors"  # the writer adds .safetensors to any other name
        done = subprocess.run([sys.executable, str(pack / "scripts/convert-lora.py"), str(source(lora_id)), str(part), model],
                              capture_output=True, text=True)
        if done.returncode:
            raise RuntimeError((done.stderr or done.stdout).strip()[-300:])
        part.replace(ready(lora_id))

    def fetch(recipe: dict) -> None:
        lora_id, state = recipe["id"], busy[recipe["id"]]
        part = folder / f"{lora_id}.part"
        try:
            if not source(lora_id).is_file():
                request = urllib.request.Request(recipe["url"], headers={"User-Agent": "TensorFoldStudio"})
                digest = hashlib.sha256()
                with urllib.request.urlopen(request, timeout=60) as reply, open(part, "wb") as out:
                    state["of"] = int(reply.headers.get("Content-Length") or 0)
                    while chunk := reply.read(1 << 20):
                        out.write(chunk)
                        digest.update(chunk)
                        state["done"] += len(chunk)
                if recipe.get("sha256") and digest.hexdigest() != recipe["sha256"]:
                    raise RuntimeError("the downloaded file does not match the recipe's sha256")
                check(part)
                part.replace(source(lora_id))
            state["stage"] = "preparing"
            prepare(lora_id)
            busy.pop(lora_id, None)
        except Exception as error:  # noqa: BLE001 - shown on the hub's button
            part.unlink(missing_ok=True)
            busy[lora_id] = {"stage": "failed", "error": str(error)[-300:], "done": 0, "of": 0}

    @router.get("/api/loras")
    def listing() -> list[dict]:
        return [public(r) for r in recipes()]

    @router.post("/api/loras")
    def add(body: dict) -> dict:
        name = (body.get("name") or "").strip()
        url, upload = (body.get("url") or "").strip(), (body.get("file") or "").strip()
        if not name or not (url or upload):
            raise HTTPException(400, "a LoRA needs a name and either a download address or an uploaded file")
        if url and not url.startswith(("https://", "http://")):
            raise HTTPException(400, "the download address starts with https://")
        lora_id = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:48] or "lora"
        with lock:
            taken = {r["id"] for r in recipes()}
            if lora_id in taken:
                lora_id += "-" + time.strftime("%H%M%S")
            styles = body.get("styles") if isinstance(body.get("styles"), dict) else {}
            try:
                strength = float(body.get("strength") or 1.0)
            except (TypeError, ValueError):
                raise HTTPException(400, "strength is a number")
            recipe = {"id": lora_id, "name": name, "author": (body.get("author") or "").strip(), "url": url,
                      "page": (body.get("page") or "").strip(), "trigger": (body.get("trigger") or "").strip(),
                      "styles": {str(k).strip(): str(v).strip() for k, v in styles.items() if str(k).strip() and str(v).strip()},
                      "strength": strength, "base": "MiniMax H3", "licence": (body.get("licence") or "").strip(),
                      "notes": (body.get("notes") or "").strip()}
            if upload:
                given = (home / upload).resolve()
                if home.resolve() not in given.parents or not given.is_file():
                    raise HTTPException(400, "the uploaded file is not in the studio's folder")
                try:
                    check(given)
                    shutil.move(str(given), source(lora_id))
                    prepare(lora_id)
                except (ValueError, RuntimeError) as error:
                    source(lora_id).unlink(missing_ok=True)
                    raise HTTPException(400, f"that file cannot be used: {error}")
            book.write_text(json.dumps(added() + [recipe], indent=1))
        return public({**recipe, "builtin": False})

    @router.post("/api/loras/{lora_id}/install")
    def install(lora_id: str) -> dict:
        recipe = find(lora_id)
        with lock:
            state = busy.get(lora_id)
            if not ready(lora_id).is_file() and (state is None or state["stage"] == "failed"):
                if not recipe.get("url") and not source(lora_id).is_file():
                    raise HTTPException(400, "this recipe has no download address")
                busy[lora_id] = {"stage": "downloading", "done": 0, "of": 0}
                threading.Thread(target=fetch, args=(recipe,), daemon=True).start()
        return public(recipe)

    @router.delete("/api/loras/{lora_id}")
    def remove(lora_id: str) -> dict:
        """A user's recipe leaves the hub; a preset only loses its downloaded file. Files go to the trash folder."""

        recipe = find(lora_id)
        with lock:
            trash = home / "trash"
            trash.mkdir(exist_ok=True)
            for path in (source(lora_id), ready(lora_id)):
                if path.is_file():
                    shutil.move(str(path), trash / (time.strftime("%Y%m%d-%H%M%S-") + path.name))
            busy.pop(lora_id, None)
            if not recipe["builtin"]:
                book.write_text(json.dumps([r for r in added() if r.get("id") != lora_id], indent=1))
        return {"removed": lora_id, "recipe": not recipe["builtin"]}

    def resolve(picked) -> list[dict]:
        """A job's `loras` as [{"path", "strength", "lead"}]; 400 when one is unknown or not installed."""

        out = []
        for item in picked or []:
            recipe = find(str(item.get("id")))
            if not ready(recipe["id"]).is_file():
                raise HTTPException(400, f"the LoRA {recipe['name']} is not installed: install it in the LoRA hub first")
            try:
                strength = float(item.get("strength") if item.get("strength") not in (None, "") else recipe.get("strength", 1.0))
            except (TypeError, ValueError):
                raise HTTPException(400, "a LoRA's strength is a number")
            if not 0 < strength <= 2:
                raise HTTPException(400, "a LoRA's strength is above 0 and at most 2")
            phrase = (recipe.get("styles") or {}).get(str(item.get("style") or ""), "")
            lead = ", ".join(part for part in (recipe.get("trigger", ""), phrase) if part)
            out.append({"path": str(ready(recipe["id"])), "strength": strength, "trigger": recipe.get("trigger", ""), "lead": lead})
        if len(out) > LIMIT:
            raise HTTPException(400, f"at most {LIMIT} LoRAs on one clip")
        return out

    def lead(prompt: str, picked: list[dict]) -> str:
        """The prompt with each LoRA's trigger and look phrase at the start of the shot, unless it is there already."""

        words = " ".join(p["lead"] if p["lead"].endswith(".") else p["lead"] + "." for p in picked
                         if p["lead"] and (not p["trigger"] or p["trigger"] not in prompt))
        if not words:
            return prompt
        return prompt.replace("[Shot 1] ", f"[Shot 1] {words} ", 1) if "[Shot 1] " in prompt else f"{words} {prompt}"

    return router, resolve, lead
