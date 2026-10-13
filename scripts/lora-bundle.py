"""The essential LoRA bundle: every preset of the Keys and Wasserman LoRA Hub in one folder, organised by the hub's
groups, so the Studio installs and uses them with no network afterwards. oneshot-essential-bundle.sh runs this.

    python3 scripts/lora-bundle.py [--dest DIR] [--categories camera,style] [--only ID,ID] [--source DIR_OR_URL]
                                   [--dry-run] [--verify] [--json] [--retries N]

Each file comes from its author's own page (or from --source / LORA_BUNDLE, another copy of a bundle, when given), is
checked against the preset's sha256 and kept with the author's model card and licence text beside it:

    DEST/01-camera-viewpoint/camera-motion/{camera_motion_h3_lora_v1_3000_pruned.safetensors, README.md, LICENSE}
    DEST/bundle.json     the menu: groups in order, and per LoRA its groups, description, needs, trigger, strength
                         range, licence, file, size, sha256, date fetched and state

No prompts. A second run fetches nothing. A dead or moved author link is reported, not fatal. Exit 0: every LoRA asked
for is present (one its author gates and has not approved for you is listed, not counted); 1: some are unavailable;
2: usage, or not enough disk. --verify checks what is there and fetches nothing. --json prints one JSON summary.
Each file stays under its own licence; the page named for it is its author's, for credit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

APP = Path(__file__).resolve().parent.parent / "app"
CARDS = ("README.md", "LICENSE", "LICENSE.md", "NOTICE", "WEIGHTS_LICENSE.md")
KEEP = ("id", "name", "author", "page", "url", "what", "needs", "trigger", "styles", "strength", "range", "base", "licence",
        "licence_tag", "notes", "status", "why", "comma", "sha256")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 22):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, target: Path, token: str, retries: int) -> None:
    """Download with resume: a .part file is continued with a Range request, and a failed try is repeated."""

    part = target.with_name(target.name + ".part")
    for attempt in range(retries + 1):
        headers = {"User-Agent": "TensorFoldStudio"}
        if token and "huggingface.co/" in url:
            headers["Authorization"] = "Bearer " + token
        have = part.stat().st_size if part.exists() else 0
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as reply:
                with open(part, "ab" if have and reply.status == 206 else "wb") as out:
                    while chunk := reply.read(1 << 20):
                        out.write(chunk)
            part.replace(target)
            return
        except urllib.error.HTTPError as error:
            if error.code == 416 and have:  # the part was already whole
                part.replace(target)
                return
            if error.code in (401, 403, 404, 410) or attempt == retries:
                raise
        except (urllib.error.URLError, OSError):
            if attempt == retries:
                raise
        time.sleep(2 * (attempt + 1))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="lora-bundle.py", description="Build or check the essential LoRA bundle.")
    parser.add_argument("--dest", default=os.path.join(os.environ.get("STUDIO_HOME", str(Path.home() / "TensorFoldStudio")), "lora-bundle"))
    parser.add_argument("--categories", default="", help="group ids, comma separated (default: all)")
    parser.add_argument("--only", default="", help="LoRA ids, comma separated")
    parser.add_argument("--source", default=os.environ.get("LORA_BUNDLE", ""), help="another bundle to copy from: a folder or a base address")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--retries", type=int, default=3)
    try:
        args = parser.parse_args(argv)
    except SystemExit:
        return 2
    groups = json.loads((APP / "lora-groups.json").read_text())
    folder_of = {g["id"]: g["folder"] for g in groups}
    presets = json.loads((APP / "lora-presets.json").read_text())
    wanted, only = set(filter(None, args.categories.split(","))), set(filter(None, args.only.split(",")))
    if wanted - set(folder_of) or only - {r["id"] for r in presets}:
        print("unknown: " + ", ".join(sorted((wanted - set(folder_of)) | (only - {r['id'] for r in presets})))
              + "\ngroups: " + ", ".join(folder_of), file=sys.stderr)
        return 2
    chosen = [r for r in presets if (not wanted or wanted & set(r["category"])) and (not only or r["id"] in only)]
    dest = Path(args.dest).expanduser()
    say = (lambda *a: None) if args.json else (lambda *a: print(*a, flush=True))
    token = os.environ.get("HF_TOKEN", "")
    source = args.source.rstrip("/")
    if source and not source.startswith(("http://", "https://")) and Path(source).expanduser().resolve() == dest.resolve():
        source = ""

    def place(recipe: dict) -> Path:
        return dest / folder_of[recipe["category"][0]] / recipe["id"] / recipe["url"].rsplit("/", 1)[-1]

    def good(recipe: dict, path: Path) -> bool:
        return path.is_file() and (not recipe.get("sha256") or sha(path) == recipe["sha256"])

    if not args.verify and not args.dry_run:
        dest.mkdir(parents=True, exist_ok=True)
        for recipe in chosen:  # a bundle laid out flat by an earlier version moves into its group's folder
            flat = dest / recipe["id"]
            if flat.is_dir() and not place(recipe).parent.exists():
                place(recipe).parent.parent.mkdir(parents=True, exist_ok=True)
                flat.rename(place(recipe).parent)
    missing = [r for r in chosen if not place(r).is_file()]
    need = sum(r.get("size") or 0 for r in missing)
    total = sum(r.get("size") or 0 for r in chosen)
    say(f"Essential LoRA bundle: {len(chosen)} LoRAs, {total / 1e9:.1f} GB in all; {len(missing)} to fetch ({need / 1e9:.1f} GB) into {dest}")
    if not args.verify and not args.dry_run:
        free = shutil.disk_usage(dest).free
        if need * 1.05 + (1 << 30) > free:
            print(f"not enough disk at {dest}: {need / 1e9:.1f} GB to fetch, {free / 1e9:.1f} GB free", file=sys.stderr)
            return 2
    states, started = {}, time.time()
    for recipe in chosen:
        target, name = place(recipe), recipe["url"].rsplit("/", 1)[-1]
        if target.is_file():
            states[recipe["id"]] = "already there" if good(recipe, target) else "checksum mismatch"
        elif args.verify or args.dry_run:
            states[recipe["id"]] = "to fetch" if args.dry_run else "missing"
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            tries = []
            if source.startswith(("http://", "https://")):
                tries.append((f"{source}/{folder_of[recipe['category'][0]]}/{recipe['id']}/{name}", ""))
            elif source:
                for local in (Path(source).expanduser() / folder_of[recipe["category"][0]] / recipe["id"] / name, Path(source).expanduser() / recipe["id"] / name):
                    if local.is_file():
                        shutil.copyfile(local, target)
                        for card in CARDS:
                            if (local.parent / card).is_file():
                                shutil.copyfile(local.parent / card, target.parent / card)
                        break
            if not target.is_file() and not (source and os.environ.get("LORA_BUNDLE_ONLY") == "1"):
                tries.append((recipe["url"], token))
            why = "unavailable at source"
            for url, auth in tries:
                if target.is_file():
                    break
                try:
                    fetch(url, target, auth, args.retries)
                except urllib.error.HTTPError as error:
                    why = "gated by its author" if error.code in (401, 403) else f"unavailable at source (HTTP {error.code})"
                except (urllib.error.URLError, OSError) as error:
                    why = f"unavailable at source ({getattr(error, 'reason', error)})"
            if not target.is_file():
                states[recipe["id"]] = why
            elif not good(recipe, target):
                target.rename(target.with_name(target.name + ".bad"))
                states[recipe["id"]] = "checksum mismatch"
            else:
                states[recipe["id"]] = "ok"
                if "huggingface.co/" in recipe["url"]:  # the author's card and licence, kept beside the file
                    base = recipe["url"].split("/resolve/")[0] + "/resolve/main/"
                    for card in CARDS:
                        if not (target.parent / card).is_file():
                            try:
                                fetch(base + card, target.parent / card, token, 0)
                            except (urllib.error.URLError, OSError):
                                pass
        if states[recipe["id"]] in ("missing", "unavailable at source") and (recipe.get("status") == "gated" or "gated" in (recipe.get("licence_tag") or "").lower()):
            states[recipe["id"]] = "gated by its author"  # not in this bundle because its author approves each download
        say(f"  {states[recipe['id']]:22} {recipe['name']} ({(recipe.get('size') or 0) / 1e6:.0f} MB)")
    present = lambda r: states.get(r["id"]) in ("ok", "already there")
    entries = []
    for order, recipe in enumerate(presets):
        if recipe not in chosen and not place(recipe).is_file():
            continue
        here = place(recipe).is_file()
        entries.append({**{k: recipe[k] for k in KEEP if k in recipe}, "category": recipe["category"], "order": order,
                        "folder": f"{folder_of[recipe['category'][0]]}/{recipe['id']}", "file": recipe["url"].rsplit("/", 1)[-1] if here else None,
                        "size": place(recipe).stat().st_size if here else recipe.get("size"),
                        "fetched": time.strftime("%Y-%m-%d", time.gmtime(place(recipe).stat().st_mtime)) if here else None,
                        "state": states.get(recipe["id"], "already there" if here else "missing")})
    if not args.verify and not args.dry_run:
        (dest / "bundle.json").write_text(json.dumps({"version": 1, "groups": groups, "loras": entries}, indent=1, ensure_ascii=False) + "\n")
    menu = [{"id": g["id"], "name": g["name"], "folder": g["folder"],
             "loras": [{"id": r["id"], "name": r["name"], "author": r["author"], "state": states[r["id"]],
                        "studio": {"ready": "works in the Studio", "gated": "its author approves each download"}.get(r["status"], "needs a Studio feature")}
                       for r in chosen if g["id"] in r["category"]]} for g in groups]
    menu = [g for g in menu if g["loras"]]
    bad = [r["id"] for r in chosen if not present(r) and states[r["id"]] != "gated by its author" and states[r["id"]] != "to fetch"]
    gated = [r["id"] for r in chosen if states[r["id"]] == "gated by its author"]
    summary = {"dest": str(dest), "loras": len(chosen), "present": sum(present(r) for r in chosen), "fetched": sum(states[r["id"]] == "ok" for r in chosen),
               "bytes": sum(place(r).stat().st_size for r in chosen if place(r).is_file()), "seconds": round(time.time() - started, 1),
               "unavailable": bad, "gated": gated, "states": states, "menu": menu}
    if args.json:
        print(json.dumps(summary, indent=1, ensure_ascii=False))
    else:
        say("\nThe hub's menu:")
        for g in menu:
            say(f"  {g['name']}")
            for item in g["loras"]:
                say(f"    {item['name']:26} by {item['author']:22} {item['state']}" + ("" if item["studio"] == "works in the Studio" else f" · {item['studio']}"))
        say(f"\n{summary['present']} of {summary['loras']} present, {summary['fetched']} fetched now, {summary['bytes'] / 1e9:.1f} GB, {summary['seconds']} s"
            + (f"; gated by their authors: {', '.join(gated)}" if gated else "") + (f"; UNAVAILABLE: {', '.join(bad)}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
