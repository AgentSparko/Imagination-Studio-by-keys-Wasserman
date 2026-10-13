#!/usr/bin/env python3
"""Edit a picture: Qwen-Image-2.1 with the picture as its reference, on mflux's Qwen-Image-2.1 pipeline (MLX).

    qwen_image_edit.py MODEL_DIR --image photo.png --prompt "make it night" -o out.png
    qwen_image_edit.py MODEL_DIR --image photo.png --lora angles.safetensors:0.9 --set '[["<mva> back view, eye-level shot", "back.png"]]'

Qwen-Image-2.1 makes and edits pictures with one set of weights: the picture goes to the text encoder's eyes and into
the sequence as latents. A checkpoint that carries its own schedule (Qwen-Image-2.1-Turbo: sample_sigmas in
model_index.json) runs on it; any other runs --steps evenly spaced steps (40 for the base model, which is what image
LoRAs are trained on). mflux reads a LoRA in its author's layout, so nothing is converted.

Credit: the Qwen team for Qwen-Image-2.1 and Qwen-Image-2.1-Turbo, Filip Strand for mflux.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def lora_specs(specs: list[str]) -> tuple[list[str], list[float]]:
    """--lora values, PATH or PATH:STRENGTH, as (paths, strengths)."""

    paths, strengths = [], []
    for spec in specs:
        path, _, strength = spec.rpartition(":")
        try:
            value = float(strength)
        except ValueError:
            path, value = spec, 1.0
        if not path:
            path, value = spec, 1.0
        if not Path(path).is_file():
            raise SystemExit(f"no LoRA at {path}")
        paths.append(path)
        strengths.append(value)
    return paths, strengths


def own_schedule(nodes: tuple[float, ...]) -> None:
    """Make mflux's default schedule the checkpoint's own raw noise levels: shifted for the size like any other,
    without the terminal stretch (what scripts/image.sh does with the same levels on the TensorFold transformer)."""

    import mlx.core as mx
    from mflux.models.common.config import config as mflux_config
    from mflux.models.common.schedulers.linear_scheduler import LinearScheduler

    class Nodes(LinearScheduler):
        def _get_sigmas(self):
            if self.config.num_inference_steps != len(nodes):
                raise ValueError(f"this checkpoint runs on its own {len(nodes)} steps")
            return LinearScheduler.shift_sigmas(mx.array(list(nodes), dtype=mx.float32), self.config, use_terminal=False)

    mflux_config.LinearScheduler = Nodes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("model_dir")
    parser.add_argument("--image", required=True, help="the picture to edit")
    parser.add_argument("--prompt", default="", help="what to change (one picture, written to -o)")
    parser.add_argument("-o", "--output", default="")
    parser.add_argument("--set", default="", help='several edits of the one picture from one load: JSON [[prompt, output], ...], or @FILE holding it')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=40, help="for a checkpoint without its own schedule")
    parser.add_argument("--resolution", type=int, default=1024, help="the picture is brought to about this many pixels squared, keeping its shape")
    parser.add_argument("--width", type=int, default=0, help="with --height: another size for the result")
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--lora", action="append", default=[], help="an image LoRA, PATH or PATH:STRENGTH (repeat to stack)")
    parser.add_argument("--quantize", type=int, default=0, choices=(0, 4, 6, 8), help="bits for the weights; 0 keeps bfloat16 (8 for a 64 GB Mac)")
    args = parser.parse_args()
    if not Path(args.image).is_file():
        parser.error(f"no picture at {args.image}")
    if bool(args.width) != bool(args.height) or args.width % 32 or args.height % 32:
        parser.error("--width and --height go together, in multiples of 32")
    edits = json.loads(Path(args.set[1:]).read_text() if args.set.startswith("@") else args.set) if args.set else [[args.prompt, args.output]]
    if not edits or any(not str(p).strip() or not o for p, o in edits):
        parser.error("give --prompt and -o, or --set with a prompt and an output for each edit")
    paths, strengths = lora_specs(args.lora)
    index = json.loads((Path(args.model_dir) / "model_index.json").read_text())
    nodes = tuple(float(v) for v in index.get("sample_sigmas") or ())
    steps = len(nodes) or args.steps

    import mlx.core as mx
    from mflux.models.common.config import ModelConfig
    from mflux.models.qwen21.variants.edit.qwen_image_21_edit import QwenImage21Edit

    if nodes:
        own_schedule(nodes)
    started = time.perf_counter()
    model = QwenImage21Edit(quantize=args.quantize or None, model_path=args.model_dir, model_config=ModelConfig.qwen_image_21(),
                            lora_paths=paths or None, lora_scales=strengths or None)
    load_s = time.perf_counter() - started
    for prompt, output in edits:
        mark = time.perf_counter()
        made = model.generate_image(seed=args.seed, prompt=prompt, num_inference_steps=steps, image_paths=[args.image],
                                    width=args.width or None, height=args.height or None, output_resolution=args.resolution)
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        picture = made.image.convert("RGB") if target.suffix.lower() in (".jpg", ".jpeg") else made.image
        picture.save(target)
        print("[tensorfold] " + json.dumps({"output": str(target), "engine": "mlx", "edit": prompt[:80], "seed": args.seed, "steps": steps,
                                            "turbo": bool(nodes), "loras": [Path(p).name for p in paths], "size": list(made.image.size),
                                            "load_s": round(load_s, 1), "total_s": round(time.perf_counter() - mark, 1),
                                            "peak_gib": round(mx.get_peak_memory() / 2**30, 1)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
