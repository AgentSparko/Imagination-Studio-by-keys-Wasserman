"""Edit a picture on a DGX Spark (CUDA): Qwen-Image-2.1 with the picture as its reference, through ComfyUI.

    spark_edit.py --image photo.png --prompt "make it night" -o out.png
    spark_edit.py --image photo.png --lora angles.safetensors:0.9 --set '[["<mva> back view, eye-level shot", "back.png"], ...]'

Qwen-Image-2.1 makes and edits pictures with one set of weights: the picture goes to the text encoder's eyes and into
the sequence as latents (ComfyUI's TextEncodeQwenImage21). The Turbo checkpoint runs its own 8 steps; --base is the
40-step model, which is what image LoRAs are trained on. A LoRA is merged at run time on the bf16 file.

Credit: the Qwen team for Qwen-Image-2.1 and Qwen-Image-2.1-Turbo, ComfyUI for the runtime.
"""
import time
import subprocess
import argparse
import json
import sys
import uuid
from pathlib import Path

import spark_generate as sg


def workflow(args, picture: str, prompt: str, seed: int, loras: list[tuple[str, float]]) -> dict:
    w = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": sg.QWEN_BASE if args.base else sg.QWEN, "weight_dtype": "default"}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": sg.QWEN_TEXT, "type": "qwen_image", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": sg.QWEN_VAE}},
        "20": {"class_type": "LoadImage", "inputs": {"image": picture}},
        # the picture is seen by the text encoder and spliced into the sequence as latents; the node's own latent is
        # an empty one on the picture's size, the size an edit is made at
        "5": {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": ["3", 0], "prompt": prompt, "negative_prompt": args.negative,
                                                                "vae": ["4", 0], "resolution": args.resolution, "images.image_1": ["20", 0]}},
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["4", 0]}},
        "16": {"class_type": "SaveImage", "inputs": {"images": ["13", 0], "filename_prefix": f"studio/edit_{uuid.uuid4().hex[:8]}"}},
    }
    model = ["1", 0]
    for index, (name, strength) in enumerate(loras):
        w[str(80 + index)] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": model, "lora_name": name, "strength_model": strength}}
        model = [str(80 + index), 0]
    latent = ["5", 2]
    if args.width and args.height:
        w["6"] = {"class_type": "EmptyLatentImage", "inputs": {"width": args.width, "height": args.height, "batch_size": 1}}
        latent = ["6", 0]
    if args.base:
        w["12"] = {"class_type": "KSampler", "inputs": {"model": model, "seed": seed, "steps": args.steps, "cfg": args.cfg, "sampler_name": "euler",
                                                         "scheduler": "simple", "positive": ["5", 0], "negative": ["5", 1], "latent_image": latent, "denoise": 1.0}}
    else:
        w["8"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}}
        w["9"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
        w["10"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": sg.TURBO_SIGMAS}}
        w["11"] = {"class_type": "CFGGuider", "inputs": {"model": model, "positive": ["5", 0], "negative": ["5", 1], "cfg": 1.0}}
        w["12"] = {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["8", 0], "guider": ["11", 0], "sampler": ["9", 0], "sigmas": ["10", 0], "latent_image": latent}}
    return w


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--image", required=True, help="the picture to edit")
    parser.add_argument("--prompt", default="", help="what to change (one picture, written to -o)")
    parser.add_argument("-o", "--output", default="")
    parser.add_argument("--set", default="", help='several edits of the one picture from one load: JSON [[prompt, output], ...], or @FILE holding it')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base", action="store_true", help="the 40-step base model (when it has been downloaded) instead of Turbo's 8 steps")
    parser.add_argument("--steps", type=int, default=40, help="base model only")
    parser.add_argument("--cfg", type=float, default=1.0, help="base model only; 1 runs the prompt alone")
    parser.add_argument("--negative", default="")
    parser.add_argument("--resolution", type=int, default=1024, help="the picture is brought to about this many pixels squared, keeping its shape")
    parser.add_argument("--width", type=int, default=0, help="with --height: another size for the result (the edit shifts when it differs from the picture's)")
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--lora", action="append", default=[], help="an image LoRA, PATH or PATH:STRENGTH (repeat to stack)")
    args = parser.parse_args()
    if not Path(args.image).is_file():
        sys.exit(f"no picture at {args.image}")
    if bool(args.width) != bool(args.height) or args.width % 16 or args.height % 16:
        sys.exit("--width and --height go together, in multiples of 16")
    if args.resolution % 32 or not 256 <= args.resolution <= 2048:
        sys.exit("--resolution is a multiple of 32 between 256 and 2048")
    edits = json.loads(Path(args.set[1:]).read_text() if args.set.startswith("@") else args.set) if args.set else [[args.prompt, args.output]]
    if not edits or any(not str(p).strip() or not o for p, o in edits):
        sys.exit("give --prompt and -o, or --set with a prompt and an output for each edit")
    # After a Library action (Ref2VA in bfloat16, SAM, the latent upscaler) ComfyUI's own unloading leaves too little
    # for Qwen-Image beside it (measured: 12.7 GB free during an edit): that edit starts from a server with nothing loaded.
    last = sg.STATE.read_text().strip() if sg.STATE.exists() else ""
    if sg.up() and last.startswith("workflow"):
        print("[tensorfold] restarting ComfyUI: a picture edit after a Library action starts from a server with nothing loaded", flush=True)
        subprocess.run(["bash", str(sg.HERE / "comfy.sh"), "stop"], check=False)
        for _ in range(60):
            if not sg.up():
                break
            time.sleep(1)
        time.sleep(5)
    sg.ensure_server("image")
    if args.base and not sg.installed("unet", sg.QWEN_BASE):
        sys.exit("the base model is not installed: oneshot-setup-spark.sh --image-edit fetches Qwen-Image-2.1 (14 GB; Qwen Research License, non-commercial)")
    loras = sg.hub_loras(args.lora)
    for name, _ in loras:
        if not sg.installed("lora", name):
            sys.exit(f"ComfyUI does not list the LoRA {name}")
    picture = sg.upload(args.image)
    try:
        for prompt, output in edits:
            item, seconds = sg.render(workflow(args, picture, prompt, args.seed, loras), "12")
            sg.fetch(item, Path(output))
            print("[tensorfold] " + json.dumps({"output": str(output), "engine": "cuda", "edit": prompt[:80], "seed": args.seed, "turbo": not args.base,
                                                "loras": [name for name, _ in loras], "total_s": round(seconds, 1)}), flush=True)
    finally:
        for name in sg.UPLOADED:
            (sg.PREFIX / "ComfyUI" / "input" / name).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
