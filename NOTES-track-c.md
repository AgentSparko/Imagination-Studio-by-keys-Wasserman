# Track C notes: the Library's three actions (Turnaround sheet, Upscale, Remove a person)

Branch `v2.6.3-workflows`, one commit on `origin/main`. Nothing here is merged, tagged or released.

## Where things are

| Piece | File | Notes |
|---|---|---|
| Job kinds `turnaround`, `upscale`, `mask`, `remove` | `app/actions.py` (`attach`, `build`, `missing`) | wired into `app/server.py` in four places: `build()`, `submit()` (kinds, no prompt needed, title, `source`), `public()`, the router |
| Page | `app/static/actions.js` (`ACT`, `actButtons`, `actPanel`, `actHubNote`, `actHubButton`) | hooks in `index.html`: the script tag, `viewLibrary` (panel + tile buttons), the hub card (status `action`), `LORA_NEEDS.action`, the quick chips filter, three tutorial steps |
| Recipes | `app/lora-presets.json`: `status: "action"`, `action`, `platforms`, `mac` (why not on a Mac) | `app/loras.py` shows an action LoRA as unsupported where its platform is missing, and `resolve()` refuses it on a clip |
| Spark runner | `spark/spark_workflows.py` (`turnaround`, `upscale`, `mask`, `remove`; `plate`, `mend`, `cut`, `follow`, `server`) | talks to the authors' nodes over ComfyUI's API |
| Spark install | `spark/setup-workflows.sh`, called by `oneshot-setup-spark.sh` wherever Ref2VA is installed (`--no-workflows` skips) | node packs at pinned commits, SAM 3.1, the latent upscaler, two folders in `extra_model_paths.yaml` |
| Mac Turnaround | `h3_generate.py --sheet` (`write_sheet`), `scripts/turnaround.sh` (both machines) | the pack's own `h3_generate.py` is used, not the copy in `$PREFIX` |
| Tests | `tests/test_actions.py` (7), page tests in `~/wasserman-project/v2.6.3-workflows/ui_actions.py`, `ui_mac.py` | |

## State

- Turnaround: done on both machines, through the app on both.
- Upscale: done on the Spark, through the app. Not on a Mac (no ComfyUI, no latent upscaler in MLX).
- Remove a person: done on the Spark from the command line; see the README table. Not on a Mac.

## Known limits and next steps

0. Remove a person: the very last frame of the generated clip came out as a different scene in every run with the
   lengthened clip (frame index `frames - 1`, the turning point where the reversed tail begins). `mend()` now replaces
   such a frame from its neighbour; the cause in the windows is not found. To look: keep the untrimmed `silent.mp4`
   in `remove()` and view frames `frames - 2 .. frames + TAIL - 1`.
1. Remove a person on a hard clip (fast hands, the person filling a third of the frame) still leaves a faint green
   tint on a few frames after `mend()`. A second pass of the remover over the result (the leftover green is already
   the mask) would likely clean it; it doubles the time. Not built.
2. Upscale and Remove restart ComfyUI between each other (`PATCHED` in `spark_workflows.py`). A cleaner fix needs a
   change in one of the two node packs; neither is ours.
3. Not tested: 2048 turnaround views, clips over 5 s, windows 39 and 56, two people in a clip, a fresh install running
   `spark/setup-workflows.sh` through `oneshot-setup-spark.sh` end to end (the script itself ran on `.4`).
4. The Mac's deployed copy needs the new `h3_generate.py` and `scripts/turnaround.sh` (the pack's files; no engine
   change) and the hub to convert the Turnaround LoRA (it does so at start from the bundle).
5. Mac ports of Upscale and Remove would be new engine work: a latent upscaler network and guide layout in MLX; SAM,
   an aligned video guide and a history relay.
