"""TensorFold Studio on a DGX Spark (CUDA): FastH3 video with sound and Qwen-Image-2.1 pictures through ComfyUI.

    spark_generate.py video --prompt "..." -o clip.mp4 [--first-frame photo.jpg] [--steps 8] [--frames 124]
    spark_generate.py video --prompt "Use <Picture 1> as the person. ..." --ref face.png -o clip.mp4
    spark_generate.py image --prompt "..." -o picture.png [--seeds 1,2,3] [--base --steps 40]

It talks to a ComfyUI server that it starts on first use (spark/comfy.sh) and leaves running with the models
loaded, so the next render skips the load. `bash spark/comfy.sh stop` frees the memory.

Credit: FastVideo (Hao AI Lab) for FastH3 and its sparse attention, MiniMax for H3, the Qwen team for
Qwen-Image-2.1-Turbo, ComfyUI for the runtime.
"""
import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREFIX = Path(os.environ.get("PREFIX", Path.home() / ".local/opt/tensorfold-studio"))
URL = os.environ.get("SPARK_COMFY_URL", "http://127.0.0.1:8190")
STATE = PREFIX / "comfy.kind"
ENGINE_LIB = PREFIX / "tf-h3" / "libtf_h3.so"
MODELS = Path(os.environ.get("SPARK_MODELS", Path.home() / "spark-studio-models"))

FASTH3 = {"bf16": "fastvideo_fasth3_8step_v2_pruned_bf16.safetensors",
          "int8": "fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors"}
# MiniMax H3 Ref2VA: reference pictures hold a face or a subject through the clip. The adapter (LightX2V's Ref2VA Turbo,
# 4 passes) is merged at run time, which only works on the bf16 file: on the 8-bit one the weights are rotated and an
# adapter added to them silently spoils the picture. Without the adapter the 8-bit file runs the full 20 steps.
REF2VA = {"bf16": "minimax_h3_ref2va_pruned_bf16.safetensors",
          "int8": "minimax_h3_ref2va_pruned_int8_convrot.safetensors"}
REF2VA_TURBO = "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
H3_TEXT = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
# TEXT_ENCODER=uncensored: linjian257's community build of the same encoder (oneshot-setup-spark.sh --uncensored;
# personal, non-commercial use under its own licence). The two never stay loaded together.
H3_TEXT_UNCENSORED = "qwen3vl_32b_minimax_h3_int8_convrot_uncensored-by-linjian257.safetensors"
TEXT_TAG = ""
# ComfyUI's 8-bit video decoder: half the decode time of the fp16 one on a GB10 (14 s against 26 s for 124 frames of
# 864x480), the same picture to 42 to 44 dB. SPARK_VAE=minimax_h3_video_vae_fp16.safetensors selects the other.
H3_VAE = os.environ.get("SPARK_VAE", "minimax_h3_video_vae_int8_convrot.safetensors")
H3_AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
QWEN = "qwen_image_2.1_turbo_bf16.safetensors"   # Qwen-Image-2.1-Turbo, the 8-step checkpoint
QWEN_BASE = "qwen_image_2.1_bf16.safetensors"    # the 40-step base model, only when it has been downloaded
QWEN_TEXT = "qwen3vl_8b_int8_convrot.safetensors"
QWEN_VAE = "qwen_image_2.1_vae_bf16.safetensors"
# the schedule saved with the Turbo checkpoint (model_index.json, sample_sigmas), then zero
TURBO_SIGMAS = "1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0.0"


def call(path, data=None):
    request = urllib.request.Request(URL + path, data=json.dumps(data).encode() if data is not None else None,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as reply:
        body = reply.read()
    return json.loads(body) if body else {}


def up() -> bool:
    try:
        call("/system_stats")
        return True
    except OSError:
        return False


def ensure_server(kind: str) -> None:
    """Start ComfyUI if it is not running; unload the other family's models when the job kind changes."""

    if not up():
        print("[tensorfold] starting ComfyUI (first render after a start loads the models)", flush=True)
        subprocess.run(["bash", str(HERE / "comfy.sh"), "start"], check=True)
        for _ in range(180):
            if up():
                break
            time.sleep(2)
        else:
            sys.exit(f"ComfyUI did not answer at {URL}: see {PREFIX}/comfy.log")
    kind += TEXT_TAG
    last = STATE.read_text().strip() if STATE.exists() else ""
    if last and last != kind:
        # the picture and video models together would not leave room for the render itself in 128 GB
        call("/free", {"unload_models": True, "free_memory": True})
    STATE.write_text(kind)


def upload(image: str) -> str:
    """Put an image in ComfyUI's input folder; returns the name LoadImage wants."""

    path = Path(image)
    name = f"studio_{uuid.uuid4().hex[:10]}{path.suffix.lower() or '.png'}"
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
            f"Content-Type: application/octet-stream\r\n\r\n").encode() + path.read_bytes() + (
            f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--{boundary}--\r\n").encode()
    request = urllib.request.Request(URL + "/upload/image", data=body,
                                     headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(request, timeout=120) as reply:
        return json.loads(reply.read())["name"]


async def run(workflow: dict, sampler: str) -> str:
    """Queue a workflow and print the sampler's progress as `[tensorfold] step i/n` until it ends; returns its id.

    The progress socket is opened before the job is queued, so a job that ends at once (ComfyUI had it cached) is
    not missed.
    """

    import aiohttp

    client = uuid.uuid4().hex
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(URL.replace("http", "ws", 1) + f"/ws?clientId={client}", max_msg_size=0) as ws:
            prompt_id = call("/prompt", {"prompt": workflow, "client_id": client})["prompt_id"]
            stages, running, since = [], None, time.time()
            async for message in ws:
                if message.type != aiohttp.WSMsgType.TEXT:
                    continue
                event = json.loads(message.data)
                data = event.get("data", {})
                if data.get("prompt_id") not in (None, prompt_id):
                    continue
                if event["type"] == "progress" and str(data.get("node")) == sampler:
                    print(f"[tensorfold] step {data['value']}/{data['max']}", flush=True)
                elif event["type"] == "executing":
                    # SPARK_TIMING=1: how long each node of the workflow ran
                    if running is not None:
                        stages.append((workflow[running]["class_type"], time.time() - since))
                    running, since = (str(data["node"]) if data.get("node") is not None else None), time.time()
                    if running is None:
                        break
                elif event["type"] in ("execution_error", "execution_interrupted", "execution_success"):
                    break
    if os.environ.get("SPARK_TIMING") == "1":
        print("[tensorfold] stages: " + ", ".join(f"{name} {took:.1f} s" for name, took in stages if took >= 0.05), flush=True)
    return prompt_id


def render(workflow: dict, sampler: str) -> tuple[dict, float]:
    """Run a workflow; returns (the saved file's record, seconds)."""

    started = time.time()
    try:
        prompt_id = asyncio.run(run(workflow, sampler))
    except KeyboardInterrupt:
        call("/interrupt", {})
        raise
    while True:
        history = call(f"/history/{prompt_id}")
        if prompt_id in history and history[prompt_id].get("status", {}).get("completed") is not None:
            break
        time.sleep(1)
    record = history[prompt_id]
    if record["status"].get("status_str") != "success":
        errors = [m[1] for m in record["status"].get("messages", []) if m[0] == "execution_error"]
        detail = errors[0].get("exception_message", "").strip() if errors else json.dumps(record["status"])[:600]
        sys.exit(f"ComfyUI failed: {detail} (log: {PREFIX}/comfy.log)")
    for output in record["outputs"].values():
        for items in output.values():
            for item in items if isinstance(items, list) else []:
                if isinstance(item, dict) and item.get("filename"):
                    return item, time.time() - started
    sys.exit("ComfyUI finished without writing a file")


def fetch(item: dict, target: Path) -> None:
    query = urllib.parse.urlencode({"filename": item["filename"], "subfolder": item.get("subfolder", ""),
                                    "type": item.get("type", "output")})
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL + "/view?" + query, timeout=600) as reply:
        target.write_bytes(reply.read())


def installed(kind: str, name: str) -> bool:
    """Whether ComfyUI lists a model file (it reads its folders again on every call)."""

    loader, field = {"unet": ("UNETLoader", "unet_name"), "lora": ("LoraLoaderModelOnly", "lora_name")}[kind]
    return name in call(f"/object_info/{loader}")[loader]["input"]["required"][field][0]


def reference(args) -> None:
    """A clip from reference pictures (MiniMax H3 Ref2VA): `<Picture 1>` in the prompt is the first --ref."""

    if len(args.ref) > 9:
        sys.exit("at most 9 reference pictures")
    if args.first_frame:
        sys.exit("--ref and --first-frame do not combine: the reference model has no first-frame input")
    ensure_server("ref2va")
    mode = args.ref_mode
    name = REF2VA["bf16"] if mode == "turbo" or not installed("unet", REF2VA["int8"]) else REF2VA["int8"]
    missing = [f for kind, f in (("unet", name),) + ((("lora", REF2VA_TURBO),) if mode == "turbo" else ()) if not installed(kind, f)]
    if missing:
        sys.exit(f"not installed: {', '.join(missing)} (bash oneshot-setup-spark.sh --ref2va)")
    # the bf16 and the 8-bit transformer never stay loaded together
    ensure_server("ref2va-" + name)
    steps = args.steps if args.steps is not None else (4 if mode == "turbo" else 20)
    prompt = prompt_of(args)
    if "<Picture" not in prompt:
        prompt = "Use " + " and ".join(f"<Picture {i + 1}>" for i in range(len(args.ref))) + " as the subject. " + prompt
    model = ["1", 0]
    w = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": name, "weight_dtype": "default"}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": H3_TEXT, "type": "minimax", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": H3_VAE}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": H3_AUDIO_VAE}},
        "6": {"class_type": "MiniMaxH3ReferenceToVideo",
              "inputs": {"clip": ["3", 0], "vae": ["4", 0], "audio_vae": ["5", 0], "prompt": prompt, "width": args.width,
                         "height": args.height, "length": args.frames, "ref_image_size": args.ref_size}},
        "8": {"class_type": "RandomNoise", "inputs": {"noise_seed": args.seed}},
        # the adapter was trained on plain Euler steps; the 20-step model on the multistep sampler ComfyUI ships it with
        "9": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler" if mode == "turbo" else "res_multistep"}},
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["4", 0]}},
        "14": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["12", 0], "vae": ["5", 0]}},
        "15": {"class_type": "CreateVideo", "inputs": {"images": ["13", 0], "audio": ["14", 0], "fps": 24}},
        "16": {"class_type": "SaveVideo", "inputs": {"video": ["15", 0], "filename_prefix": f"studio/ref_{uuid.uuid4().hex[:8]}",
                                                     "format": "auto", "codec": "auto"}},
    }
    for i, picture in enumerate(args.ref):
        w[str(30 + i)] = {"class_type": "LoadImage", "inputs": {"image": upload(picture)}}
        w["6"]["inputs"][f"ref_images.ref_image_{i}"] = [str(30 + i), 0]
    if mode == "turbo":
        w["2"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": model, "lora_name": REF2VA_TURBO, "strength_model": 1.0}}
        model = ["2", 0]
    w["10"] = {"class_type": "BasicScheduler", "inputs": {"model": model, "scheduler": "simple", "steps": steps, "denoise": 1.0}}
    w["11"] = {"class_type": "BasicGuider", "inputs": {"model": model, "conditioning": ["6", 0]}}
    w["12"] = {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["8", 0], "guider": ["11", 0], "sampler": ["9", 0], "sigmas": ["10", 0],
                          "latent_image": ["6", 1]}}
    kind = "Ref2VA with the Turbo adapter" if mode == "turbo" else "Ref2VA"
    print(f"[tensorfold] engine: cuda (ComfyUI, MiniMax H3 {kind}, {name.split('pruned_')[1].split('.')[0]}), "
          f"{len(args.ref)} reference picture(s), generating {args.width}x{args.height}, {args.frames} frames", flush=True)
    item, seconds = render(w, "12")
    target = Path(args.output)
    fetch(item, target)
    print("[tensorfold] " + json.dumps({"output": str(target), "engine": "cuda", "model": "ref2va", "mode": mode,
                                        "forwards": steps, "total_s": round(seconds, 1), "frames": args.frames,
                                        "size": [args.width, args.height], "references": len(args.ref)}), flush=True)


def prompt_of(args) -> str:
    text = Path(args.prompt_file).read_text() if args.prompt_file else args.prompt
    if not text or not text.strip():
        sys.exit("give --prompt or --prompt-file")
    return text.strip()


def video(args) -> None:
    for name, value, step in (("width", args.width, 32), ("height", args.height, 32)):
        if value % step:
            sys.exit(f"--{name} must be a multiple of {step}, got {value}")
    if (args.frames - 5) % 17:
        sys.exit(f"--frames must be 17n + 5 (124, 192, 243, 362), got {args.frames}")
    if args.ref:
        return reference(args)
    if args.steps is None:
        args.steps = 8
    loras = []
    for spec in args.lora:
        # a style LoRA is merged by ComfyUI at run time, which only works on the bf16 file and on ComfyUI's own
        # blocks: the 8-bit file's weights are rotated, and the TensorFold engine reads its own copy of the weights
        path, _, strength = spec.rpartition(":") if spec.rsplit(":", 1)[-1].replace(".", "", 1).isdigit() else (spec, "", "1.0")
        source = Path(path).resolve()
        if not source.is_file():
            sys.exit(f"no LoRA at {path}")
        link = MODELS / "loras" / f"hub_{source.name}"
        if not link.exists() or link.resolve() != source:
            link.parent.mkdir(parents=True, exist_ok=True)
            link.unlink(missing_ok=True)
            link.symlink_to(source)
        loras.append((link.name, float(strength)))
    if loras:
        if args.engine == "tensorfold" or args.weights != "bf16":
            print("[tensorfold] a LoRA renders on ComfyUI's own blocks with the bf16 weights", flush=True)
        args.engine, args.weights = "comfy", "bf16"
    sparsity = args.sparsity
    if sparsity is None and args.engine == "tensorfold":
        sparsity = 0.8    # what FastH3 was trained with; the engine's sparse kernel is the faster one at every size
    if sparsity is None:
        # sparse attention pays off once the video rows dominate. Measured on a GB10, a pass at 864x480 takes
        # 10.5 s dense and 11.1 s sparse; at 1344x768, 33.4 s dense and 29.4 s sparse
        sparsity = 0.8 if args.width * args.height >= 700_000 else 0.0
    if args.engine == "tensorfold" and not ENGINE_LIB.exists():
        sys.exit(f"no TensorFold engine at {ENGINE_LIB}: run spark/build-engine.sh, or use --engine comfy")
    last = STATE.read_text().strip() if STATE.exists() else ""
    if up() and (loras or "lora" in last):
        # On a Spark the GPU and the system share one pool. With the bf16 weights (42 GB) a text encoder (26 GB) that
        # comes back for a new prompt makes ComfyUI move the model aside and load it again, and the two copies do not
        # fit: measured, free memory fell from 50 GB to 8. A server with nothing loaded keeps about 30 GB free.
        print("[tensorfold] restarting ComfyUI: a LoRA render starts from a server with nothing loaded", flush=True)
        subprocess.run(["bash", str(HERE / "comfy.sh"), "stop"], check=False)
        for _ in range(60):
            if not up():
                break
            time.sleep(1)
        time.sleep(3)
    ensure_server("video+lora" if loras else "video")
    model = ["2", 0]
    w = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": FASTH3[args.weights], "weight_dtype": "default"}},
        # FastH3's schedule: video shift 10, audio shift 3 (fastvideo_inference.json)
        "2": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["1", 0], "shift_video": 10.0, "shift_audio": 3.0}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": H3_TEXT, "type": "minimax", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": H3_VAE}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": H3_AUDIO_VAE}},
        "6": {"class_type": "MiniMaxH3ImageToVideo",
              "inputs": {"clip": ["3", 0], "vae": ["4", 0], "prompt": prompt_of(args), "width": args.width,
                         "height": args.height, "length": args.frames}},
        "7": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["6", 0]}},
        "8": {"class_type": "RandomNoise", "inputs": {"noise_seed": args.seed}},
        "9": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "12": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["8", 0], "guider": ["11", 0], "sampler": ["9", 0], "sigmas": ["10", 0],
                          "latent_image": ["6", 1]}},
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["4", 0]}},
        "14": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["12", 0], "vae": ["5", 0]}},
        "15": {"class_type": "CreateVideo", "inputs": {"images": ["13", 0], "audio": ["14", 0], "fps": 24}},
        "16": {"class_type": "SaveVideo", "inputs": {"video": ["15", 0], "filename_prefix": f"studio/clip_{uuid.uuid4().hex[:8]}",
                                                     "format": "auto", "codec": "auto"}},
    }
    if args.first_frame:
        w["20"] = {"class_type": "LoadImage", "inputs": {"image": upload(args.first_frame)}}
        w["6"]["inputs"]["first_frame"] = ["20", 0]
    for index, (name, strength) in enumerate(loras):
        if not installed("lora", name):
            sys.exit(f"ComfyUI does not list the LoRA {name}")
        node = str(30 + index)
        w[node] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": w["2"]["inputs"]["model"], "lora_name": name, "strength_model": strength}}
        w["2"]["inputs"]["model"] = [node, 0]
    if args.engine == "tensorfold":
        # the 50 blocks of every pass on TensorFold's CUDA family, int8 on the tensor cores, with FastH3's own sparse
        # attention; ComfyUI keeps the text encoder, the sampler and the decoders
        w["22"] = {"class_type": "TensorFoldH3Blocks",
                   "inputs": {"model": model, "unet_name": FASTH3["bf16"], "sparsity": sparsity}}
        model = ["22", 0]
    elif sparsity > 0:
        # ComfyUI's compiled block-sparse kernel with FastVideo's selection: 3D video cubes, the top share kept per
        # query cube, FastH3's own gate weights for the coarse branch; text and audio rows stay exact
        w["21"] = {"class_type": "BlockSparseAttention",
                   "inputs": {"model": model, "selection": "vsa", "selection.keep_percent": round(100 * (1 - sparsity), 1),
                              "start_percent": 0.0, "end_percent": 1.0, "dense_blocks": "", "min_tokens": 0,
                              "extra_tokens": 0, "sink_conditioning": "exact_kv_and_rows", "verbose": False}}
        model = ["21", 0]
    w["10"] = {"class_type": "BasicScheduler", "inputs": {"model": model, "scheduler": "simple", "steps": args.steps,
                                                          "denoise": 1.0}}
    w["11"] = {"class_type": "CFGGuider", "inputs": {"model": model, "positive": ["6", 0], "negative": ["7", 0],
                                                     "cfg": 1.0}}
    attention = f"sparse attention {sparsity}" if sparsity > 0 else "dense attention"
    runs = "TensorFold int8 blocks in ComfyUI" if args.engine == "tensorfold" else f"ComfyUI, FastH3 {args.weights}"
    print(f"[tensorfold] engine: cuda ({runs}, {attention}), generating "
          f"{args.width}x{args.height}, {args.frames} frames", flush=True)
    item, seconds = render(w, "12")
    target = Path(args.output)
    fetch(item, target)
    if args.crop:
        width, height = args.crop.lower().split("x")
        cropped = target.with_suffix(".crop.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(target), "-vf", f"crop={width}:{height}", "-c:v",
                        "libx264", "-crf", "17", "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "copy",
                        str(cropped)], check=True)
        cropped.replace(target)
    print("[tensorfold] " + json.dumps({"output": str(target), "engine": "cuda", "forwards": args.steps,
                                        "total_s": round(seconds, 1), "frames": args.frames,
                                        "size": [args.width, args.height], "sparsity": sparsity}), flush=True)


def image(args) -> None:
    for name, value in (("width", args.width), ("height", args.height)):
        if value % 16:
            sys.exit(f"--{name} must be a multiple of 16, got {value}")
    ensure_server("image")
    text = prompt_of(args)
    target = Path(args.output)
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    for seed in seeds:
        w = {
            "1": {"class_type": "UNETLoader", "inputs": {"unet_name": QWEN_BASE if args.base else QWEN,
                                                         "weight_dtype": "default"}},
            "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": QWEN_TEXT, "type": "qwen_image", "device": "default"}},
            "4": {"class_type": "VAELoader", "inputs": {"vae_name": QWEN_VAE}},
            "5": {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": ["3", 0], "prompt": text,
                                                                    "negative_prompt": args.negative, "resolution": 1024}},
            "6": {"class_type": "EmptyLatentImage", "inputs": {"width": args.width, "height": args.height, "batch_size": 1}},
            "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["4", 0]}},
            "16": {"class_type": "SaveImage", "inputs": {"images": ["13", 0], "filename_prefix": f"studio/image_{uuid.uuid4().hex[:8]}"}},
        }
        if args.base:
            w["12"] = {"class_type": "KSampler",
                       "inputs": {"model": ["1", 0], "seed": seed, "steps": args.steps, "cfg": args.cfg,
                                  "sampler_name": "euler", "scheduler": "simple", "positive": ["5", 0],
                                  "negative": ["5", 1], "latent_image": ["6", 0], "denoise": 1.0}}
        else:
            w["8"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}}
            w["9"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
            w["10"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": TURBO_SIGMAS}}
            w["11"] = {"class_type": "CFGGuider", "inputs": {"model": ["1", 0], "positive": ["5", 0],
                                                             "negative": ["5", 1], "cfg": 1.0}}
            w["12"] = {"class_type": "SamplerCustomAdvanced",
                       "inputs": {"noise": ["8", 0], "guider": ["11", 0], "sampler": ["9", 0], "sigmas": ["10", 0],
                                  "latent_image": ["6", 0]}}
        item, seconds = render(w, "12")
        out = target.with_name(f"{target.stem}_s{seed}{target.suffix}") if args.seeds else target
        fetch(item, out)
        print("[tensorfold] " + json.dumps({"output": str(out), "engine": "cuda", "width": args.width,
                                            "height": args.height, "seed": seed, "turbo": not args.base,
                                            "total_s": round(seconds, 1)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    kinds = parser.add_subparsers(dest="kind", required=True)
    v = kinds.add_parser("video")
    v.add_argument("--prompt", default=""); v.add_argument("--prompt-file", default="")
    v.add_argument("-o", "--output", required=True)
    v.add_argument("--width", type=int, default=864); v.add_argument("--height", type=int, default=480)
    v.add_argument("--frames", type=int, default=124); v.add_argument("--seed", type=int, default=0)
    v.add_argument("--steps", type=int, default=None, help="passes: 8 is what FastH3 was trained for (the default); "
                   "with --ref, 4 with the Turbo adapter and 20 without")
    v.add_argument("--ref", action="append", default=[], help="a reference picture (repeat for up to 9): the clip "
                   "comes from MiniMax H3 Ref2VA and `<Picture 1>` in the prompt is the first one")
    v.add_argument("--ref-mode", choices=("turbo", "full"), default="full",
                   help="full: 20 steps; turbo: the bf16 model with LightX2V's 4-pass adapter")
    v.add_argument("--ref-size", choices=("match", "max"), default="match",
                   help="match: references scaled down to the clip's pixel area (what the adapter was trained with); "
                        "max: up to 2048 pixels on the short edge, several times slower")
    v.add_argument("--first-frame", default="", help="an image the clip starts from")
    v.add_argument("--text-encoder", choices=("regular", "uncensored"), default=os.environ.get("TEXT_ENCODER") or "regular",
                   help="uncensored: a community build of MiniMax H3's text encoder (oneshot-setup-spark.sh --uncensored)")
    v.add_argument("--crop", default="", help="WxH centre crop of the finished clip")
    v.add_argument("--weights", choices=sorted(FASTH3), default=os.environ.get("SPARK_WEIGHTS", "bf16"))
    v.add_argument("--engine", choices=("tensorfold", "comfy"),
                   default=os.environ.get("SPARK_ENGINE") or ("tensorfold" if ENGINE_LIB.exists() else "comfy"),
                   help="what runs the transformer blocks: TensorFold's CUDA family (the default once "
                        "spark/build-engine.sh has built it) or ComfyUI's own")
    v.add_argument("--lora", action="append", default=[], help="a style LoRA, PATH or PATH:STRENGTH (repeat to stack)")
    v.add_argument("--sparsity", type=float, default=None, help="sparse attention (0 is dense); default by size")
    v.set_defaults(run=video)
    i = kinds.add_parser("image")
    i.add_argument("--prompt", default=""); i.add_argument("--prompt-file", default="")
    i.add_argument("-o", "--output", required=True)
    i.add_argument("--width", type=int, default=1344); i.add_argument("--height", type=int, default=768)
    i.add_argument("--seed", type=int, default=0); i.add_argument("--seeds", default="", help="several pictures: 1,2,3")
    i.add_argument("--base", action="store_true", help="the base model without the turbo adapter")
    i.add_argument("--steps", type=int, default=40, help="base model only")
    i.add_argument("--cfg", type=float, default=4.0, help="base model only")
    i.add_argument("--negative", default="")
    i.set_defaults(run=image)
    args = parser.parse_args()
    if getattr(args, "text_encoder", "regular") == "uncensored":
        global H3_TEXT, TEXT_TAG
        H3_TEXT, TEXT_TAG = H3_TEXT_UNCENSORED, "+uncensored"
        ensure_server("video")
        if H3_TEXT not in call("/object_info/CLIPLoader")["CLIPLoader"]["input"]["required"]["clip_name"][0]:
            sys.exit(f"not installed: {H3_TEXT} (bash oneshot-setup-spark.sh --uncensored)")
    args.run(args)


if __name__ == "__main__":
    main()
