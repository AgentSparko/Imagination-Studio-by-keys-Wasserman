"""Render one MiniMax H3 clip with TensorFold's H3 family (copy of tools/h3_generate_dev.py at the pinned commit).

    PYTHONPATH=src:<minimax-h3-mlx> <minimax-h3-mlx>/.venv/bin/python tools/h3_generate_dev.py \
        ~/h3-models/MiniMax-H3 --prompt-file prompt.txt -o out.mp4

The text encoder, the audio decoder and the MP4 writer come from minimax-h3-mlx until the family has its own;
the transformer, sampler, adapters and video decoder are `tensorfold.families.h3`. `--parity` also runs
the reference transformer on the first step's inputs and reports the difference.

`--ref-image` (repeatable) is reference to video, on the Ref2VA transformer: each picture is named in the prompt as
`<Picture 1>`, `<Picture 2>`, ... and fixes how a person or a thing looks without being a frame of the clip.
`model_dir` is then the folder holding `Ref2VA/` (or the Ref2VA folder itself). `--ref-video` (up to 3, each 2 to
15 s) and `--ref-audio` (up to 3, each 2 to 15 s, only beside a picture or a video) are the same model's other two
kinds of reference: `<Video 1>`, ... carry a subject and its motion, `<Audio 1>`, ... a voice or music. The request
is always pictures, then videos, then audio clips. They need a TensorFold whose H3 family packs references
(`packing.reference_layout`); pictures alone run on older ones.

`--text-encoder` takes the text encoder's weights from one ComfyUI int8 file (`int8_tensorwise`, with or without
ConvRot) instead of the model folder, for community builds of the encoder that are published only in that form.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
import subprocess
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from tensorfold.families.h3 import config as h3
from tensorfold.families.h3.lora import merge, settle
from tensorfold.families.h3.packing import unpack_audio, unpatchify
from tensorfold.families.h3.sampler import denoise
from tensorfold.families.h3.schedule import parse_subset
from tensorfold.families.h3.weights import int8_attention, int8_mlp, load_dit


class _NoEval:
    """`mlx.core` with `eval` switched off, so weights that were loaded lazily stay on disk until they are used."""

    def __getattr__(self, name):
        return getattr(mx, name)

    @staticmethod
    def eval(*_):
        return None


def _streamed(encoder):
    """Make the text encoder run one layer at a time, each layer's weights read for its turn and dropped after it.

    The encoder is 50 layers of a 32B model: held whole it is 48 GiB, the largest thing in a FastH3 render. Run this
    way it needs one layer at a time (under 2 GiB) and returns the same rows, so a clip fits a 64 GB Mac.
    """

    model = encoder.language

    def hidden_states(input_ids, position_ids, inputs_embeds=None, visual_pos_masks=None, deepstack_visual_embeds=None):
        from mlx_vlm.models.base import create_attention_mask

        h = model.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        mask = create_attention_mask(h, None)
        position_embeddings = None
        if position_ids is not None and not model.layers[0].self_attn.rotary_emb.fused_apply:
            position_embeddings = model.layers[0].self_attn.rotary_emb(h, position_ids)
        for index in range(len(model.layers)):
            h = model.layers[index](h, mask, None, position_ids, position_embeddings)
            if deepstack_visual_embeds is not None and index < len(deepstack_visual_embeds):
                h = model._deepstack_process(h, visual_pos_masks, deepstack_visual_embeds[index])
            mx.eval(h)
            model.layers[index] = None  # the encoder is used once
            mx.clear_cache()
        return h

    encoder._hidden_states = hidden_states


def _hadamard(size: int):
    """ConvRot's rotation: the regular Hadamard matrix of a power-of-4 size, scaled to be orthogonal (comfy_kitchen's
    `_build_hadamard`). It is symmetric, so it is also its own inverse."""

    h4 = np.array([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], np.float32)
    h = h4
    while h.shape[0] < size:
        h = np.kron(h, h4)
    if h.shape[0] != size:
        raise ValueError(f"a ConvRot group is a power of 4, got {size}")
    return mx.array(h / np.sqrt(np.float32(size)))


def comfy_int8_weights(path, dtype=mx.bfloat16):
    """The tensors of a ComfyUI `int8_tensorwise` checkpoint by name, in ``dtype``, and how many were int8 and rotated.

    A quantized layer is three tensors: `weight` (int8), `weight_scale` (float32, one per output row) and
    `comfy_quant` (its settings, JSON as bytes). The weight is ``int8 * scale``; with `convrot` the columns were
    rotated in groups before rounding (``W_rot = W @ H^T`` per group of `convrot_groupsize` columns, H orthogonal), so
    the same product turns them back. Nothing is read here: the arrays are lazy, each computed from the file when it
    is first used, so a caller that drops a layer after using it never holds the whole model.
    """

    tensors = mx.load(str(path))
    weights, used, rotated = {}, set(), 0
    for name, tensor in tensors.items():
        stem, _, leaf = name.rpartition(".")
        if leaf in ("comfy_quant", "weight_scale"):
            continue
        settings, scale = f"{stem}.comfy_quant", f"{stem}.weight_scale"
        if leaf != "weight" or settings not in tensors:
            if tensor.dtype not in (mx.bfloat16, mx.float16, mx.float32):
                raise ValueError(f"{name} is {tensor.dtype} without quantization settings")
            weights[name] = tensor.astype(dtype)
            continue
        conf = json.loads(bytes(np.array(tensors[settings])))
        if conf.get("format") != "int8_tensorwise" or tensor.dtype != mx.int8 or tensor.ndim != 2:
            raise ValueError(f"{name}: only int8_tensorwise matrices are read, got {conf} as {tensor.dtype}")
        if scale not in tensors or tensors[scale].shape != (tensor.shape[0], 1):
            raise ValueError(f"{name}: no scale per output row")
        used.update((settings, scale))
        weight = tensor.astype(mx.float32) * tensors[scale].astype(mx.float32)
        if conf.get("convrot", False):
            group = int(conf.get("convrot_groupsize", 256))
            rows, columns = weight.shape
            if columns % group:
                raise ValueError(f"{name}: {columns} columns are not whole groups of {group}")
            weight = (weight.reshape(rows, columns // group, group) @ _hadamard(group).T).reshape(rows, columns)
            rotated += 1
        weights[name] = weight.astype(dtype)
    stray = sorted(name for name in tensors if name.endswith((".comfy_quant", ".weight_scale")) and name not in used)
    if stray:
        raise ValueError(f"{path}: {len(stray)} quantization tensors without an int8 weight, e.g. {stray[:4]}")
    return weights, len(used) // 2, rotated


def _from_comfy_int8(base, path):
    """``base`` (minimax-h3-mlx's text encoder) reading its weights from one ComfyUI int8 file instead of the model
    folder: `model.*` is the language model's 50 layers and its embedding, `visual.*` the vision tower, under the
    names the MLX modules already use. The folder still gives the config and the tokenizer.

    Every weight the modules have must be in the file and every tensor of the file must have a place; the one
    exception is the final norm, which a file cut at the conditioning layer does not carry and H3 never applies.
    """

    class Encoder(base):
        def _load_weights(self, model_dir, dtype, verbose):
            from mlx.utils import tree_flatten, tree_unflatten

            weights, quantized, rotated = comfy_int8_weights(path, dtype)
            parts = {"model.": self.language, "visual.": self.vision}
            unknown = sorted(name for name in weights if not name.startswith(tuple(parts)))
            for prefix, module in parts.items():
                mine = {name[len(prefix):]: tensor for name, tensor in weights.items() if name.startswith(prefix)}
                if module is None:  # a request without pictures does not build the vision tower
                    continue
                expected = {name: tensor.shape for name, tensor in tree_flatten(module.parameters())}
                if prefix == "model.":
                    expected.pop("norm.weight", None)
                sanitize = getattr(module, "sanitize", None)  # the patch convolution's layout, as in the base loader
                if sanitize is not None:
                    mine = sanitize(mine)
                missing = sorted(expected.keys() - mine.keys())
                unknown += sorted(prefix + name for name in mine.keys() - expected.keys())
                wrong = sorted(name for name in expected.keys() & mine.keys() if mine[name].shape != expected[name])
                if missing or wrong:
                    raise KeyError(f"{path}: {len(missing)} weights of {prefix}* missing, e.g. {missing[:4]}; "
                                   f"{len(wrong)} of another shape, e.g. {wrong[:4]}")
                module.update(tree_unflatten(list(mine.items())))
            if unknown:
                raise KeyError(f"{path}: {len(unknown)} tensors the text encoder has no place for, e.g. {unknown[:4]}")
            self.skipped_tensors = 0
            print(f"[tensorfold] text encoder: {Path(path).name}, {quantized} int8 weights ({rotated} rotated) "
                  f"decoded to {dtype}", flush=True)

    return Encoder


VISION_STATS = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Qwen3-VL's own: pixels to -1..1 (used when the file is absent)
MLX_VISION_STATS = ([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])


def vision_stats(root: Path, name: str = "preprocessor_config.json"):
    """The mean and deviation the text encoder's vision tower was trained with, per colour channel.

    They are read from the checkpoint's own `processor/` folder (`preprocessor_config.json` for pictures,
    `video_preprocessor_config.json` for video frames): 0.5 and 0.5 for MiniMax H3, which is Qwen3-VL's
    normalisation to -1..1. A folder without that file gets the same 0.5 and 0.5. minimax-h3-mlx builds its
    image processor with the values of the earlier Qwen-VL models instead (``MLX_VISION_STATS``), which shows the
    tower a picture with nearly twice the contrast; releases up to 2.5 used those.
    """

    path = root / "processor" / name
    if not path.is_file():
        return VISION_STATS
    with open(path) as handle:
        conf = json.load(handle)
    mean, std = conf.get("image_mean"), conf.get("image_std")
    if not (isinstance(mean, list) and isinstance(std, list) and len(mean) == len(std) == 3):
        return VISION_STATS
    return [float(v) for v in mean], [float(v) for v in std]


def _text_encoder(root: Path, vision: bool, weights=None, checkpoint_stats: bool = True):
    """minimax-h3-mlx's text encoder, run a layer at a time, with or without its vision tower.

    Pictures are normalised with the checkpoint's own mean and deviation (:func:`vision_stats`);
    ``checkpoint_stats`` false keeps the values minimax-h3-mlx's image processor is built with, for comparisons
    with clips made before this was corrected.
    """

    import minimax_h3_mlx.text_encoder as module
    from minimax_h3_mlx.text_encoder import MiniMaxH3TextEncoder

    if weights is not None:
        MiniMaxH3TextEncoder = _from_comfy_int8(MiniMaxH3TextEncoder, weights)
    original = module.mx
    module.mx = _NoEval()  # the loader ends by reading every weight into memory; leave them for _streamed
    try:
        encoder = MiniMaxH3TextEncoder(root / "text_encoder", dtype=mx.bfloat16, load_vision=vision)
    finally:
        module.mx = original
    if encoder.vision is not None:
        mx.eval(encoder.vision.parameters())
    _streamed(encoder)
    if vision:
        # the checkpoint's processor folder asks for PyTorch; the image processor built from the vision config
        # needs only numpy, and the image arrives already on the render canvas
        from minimax_h3_mlx.text_encoder import _FallbackProcessor

        processor = encoder._build_image_processor()
        processor.image_mean, processor.image_std = vision_stats(root) if checkpoint_stats else MLX_VISION_STATS
        encoder._processor = _FallbackProcessor(processor)
    return encoder


def encode_text(root: Path, prompt: str, images=None, weights=None, checkpoint_stats: bool = True):
    """Text rows and their tags. ``images`` (one picture or a list) go ahead of the prompt, each as
    ``<Picture i>: `` and its vision block, which is how the prompt refers to them. ``weights`` is a ComfyUI int8
    file to take the encoder's weights from, in place of the ones in the model folder."""

    if images is not None and not isinstance(images, (list, tuple)):
        images = [images]
    images = list(images) if images else None
    encoder = _text_encoder(root, images is not None, weights, checkpoint_stats)
    embeds, tags = encoder.encode(prompt, images)
    mx.eval(embeds)
    del encoder
    gc.collect()
    mx.clear_cache()
    return embeds, tags


def encode_first_frame(root: Path, image, width: int, height: int, patch):
    """Conditioning rows for a first frame, from minimax-h3-mlx's video VAE encoder (not ported)."""

    from minimax_h3_mlx.load import load_video_vae
    from minimax_h3_mlx.pipeline import encode_keyframe_rows

    vae = load_video_vae(root / "video_vae")
    rows = encode_keyframe_rows(vae, [image], height, width, patch)
    mx.eval(rows)
    del vae
    gc.collect()
    mx.clear_cache()
    return rows


REFERENCE_MULTIPLE = 32  # a reference picture's sides, as the canvas's: whole 2x2 patches of 16x latents
REFERENCE_LIMIT = 9  # pictures the Ref2VA model takes in one request
_POSITION_SCALE = 32  # the spatial axes of the rotary positions, as in tensorfold.families.h3.packing


def reference_size(width: int, height: int, canvas_width: int, canvas_height: int) -> tuple[int, int]:
    """The size a reference picture is encoded at: scaled down, never up, to the clip's pixel area with its own
    aspect ratio kept, each side rounded to a multiple of 32 (ComfyUI's "match", h3.c's default)."""

    scale = min(1.0, math.sqrt((canvas_width * canvas_height) / (width * height)))
    return (max(REFERENCE_MULTIPLE, round(width * scale / REFERENCE_MULTIPLE) * REFERENCE_MULTIPLE),
            max(REFERENCE_MULTIPLE, round(height * scale / REFERENCE_MULTIPLE) * REFERENCE_MULTIPLE))


def prepare_references(paths, canvas_width: int, canvas_height: int):
    from PIL import Image

    images = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        size = reference_size(*image.size, canvas_width, canvas_height)
        images.append(image if image.size == size else image.resize(size, Image.Resampling.LANCZOS))
    return images


def encode_references(root: Path, images, patch):
    """Conditioning rows of the reference pictures, in order, and each one's latent (height, width).

    A picture is one latent frame of the video VAE at the picture's own size: the middle of the encoder's
    distribution (not a draw from it, unlike a first frame), normalised and cut into the same 2x2 patches as the
    clip's own latents.
    """

    from minimax_h3_mlx.load import load_video_vae
    from minimax_h3_mlx.pipeline import PIXEL_MEAN, PIXEL_STD

    from tensorfold.families.h3.packing import patchify

    vae = load_video_vae(root / "video_vae")
    cfg = vae.config
    latents_mean = mx.array(np.array(cfg.latents_mean, np.float32)).reshape(1, -1, 1, 1, 1)
    latents_std = mx.array(np.array(cfg.latents_std, np.float32)).reshape(1, -1, 1, 1, 1)
    pixel_mean = np.array(PIXEL_MEAN, np.float32).reshape(1, 3, 1, 1, 1)
    pixel_std = np.array(PIXEL_STD, np.float32).reshape(1, 3, 1, 1, 1)
    rows, shapes = [], []
    for image in images:
        pixels = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)[None, :, None]
        pixels = (pixels / 255.0 - pixel_mean) / pixel_std
        moments = vae._encode_clip(mx.array(pixels).transpose(0, 2, 3, 4, 1))  # (1, 1, h, w, 2 C), channels last
        mean = moments[..., :cfg.latent_channels].transpose(0, 4, 1, 2, 3).astype(mx.float32)
        if mean.shape[2] != 1:
            raise ValueError(f"a reference picture should encode to one latent frame, got {mean.shape[2]}")
        normalized = (mean - latents_mean) / latents_std
        rows.append(patchify(normalized, patch))
        shapes.append((int(mean.shape[3]), int(mean.shape[4])))
    rows = mx.concatenate(rows)
    mx.eval(rows)
    del vae
    gc.collect()
    mx.clear_cache()
    return rows, tuple(shapes)


def reference_layout(shapes):
    """A stand-in for the family's ``layout`` that packs reference pictures: [text | references | audio | video].

    Reference ``i`` is one latent frame at time ``text + i`` on a height/width grid normalised by its own area, and
    the clip's audio and video start at ``text + len(shapes)`` instead of ``text``. Its rows are video rows that are
    held (noised once, pinned at the keyframe timestep, never stepped), which is what the sampler already does
    with the rows the layout counts as ``condition_video_rows``. The clip's own rows come from the family's layout,
    called with one placeholder text row per reference so that their clock starts where it must.
    """

    from tensorfold.families.h3 import packing

    def grid(size: int, patch: int, sqrt_area: float) -> np.ndarray:
        ratio = size / sqrt_area
        left = (1.0 - ratio) / 2.0
        return np.linspace(left, left + ratio, size // patch, endpoint=False) * _POSITION_SCALE

    def build(text_tags, latent_frames, latent_height, latent_width, audio_latents, patch, keyframes=()):
        if keyframes:
            raise ValueError("reference pictures cannot be combined with a first or last frame")
        _, ph, pw = patch
        text_tags = np.asarray(text_tags, dtype=np.int64)
        text, count = int(text_tags.shape[0]), len(shapes)
        base = packing.layout(np.concatenate([text_tags, np.full(count, packing.TAG_TEXT, dtype=np.int64)]),
                              latent_frames, latent_height, latent_width, audio_latents, patch)
        positions = np.array(base.position_ids)
        blocks = []
        for index, (height, width) in enumerate(shapes):
            sqrt_area = np.sqrt(height * width)
            hh, ww = np.meshgrid(grid(height, ph, sqrt_area), grid(width, pw, sqrt_area), indexing="ij")
            block = np.empty((hh.size, 3), dtype=np.float64)
            block[:, 0] = float(text + index)
            block[:, 1], block[:, 2] = hh.reshape(-1), ww.reshape(-1)
            blocks.append(block.astype(np.float32))
        held = sum(block.shape[0] for block in blocks)
        positions = np.concatenate([positions[:text], *blocks, positions[text + count:]])
        rows = int(positions.shape[0])
        audio_start = text + held
        video_start = audio_start + audio_latents * packing.AUDIO_CHANNELS
        video_rows = np.concatenate([np.arange(text, audio_start), np.arange(video_start, rows)])
        audio_rows = np.arange(audio_start, video_start)
        tags = np.empty(rows, dtype=np.int64)
        tags[:text] = text_tags
        tags[audio_rows] = packing.TAG_AUDIO
        tags[video_rows] = packing.TAG_VIDEO
        return packing.Layout(rows, mx.array(positions), mx.array(tags.astype(np.int32)),
                              mx.array(video_rows.astype(np.int32)), mx.array(audio_rows.astype(np.int32)),
                              mx.array(np.arange(text, dtype=np.int32)), held)

    return build


VIDEO_LIMIT = 3  # reference videos the Ref2VA model takes in one request
AUDIO_LIMIT = 3  # reference audio clips
TOTAL_LIMIT = 12  # references of all kinds
CLIP_SECONDS = (2.0, 15.0)  # the length of one reference video or audio clip
TOTAL_SECONDS = 15.0  # the summed length of the reference videos, and of the audio clips
CLIP_TOLERANCE = 0.05  # seconds a container's duration may miss a limit by
ASPECT_LIMIT = 4.0  # a reference video is between 1:4 and 4:1
MODEL_SHORT_EDGE, MODEL_MAX_PIXELS = 768, 768 * 1344  # the model's own canvas for a reference video
QWEN_SAMPLE_STRIDE = 12  # the text encoder sees a reference video at 2 fps: every 12th of its 24 fps frames
QWEN_FRAMES_PER_BLOCK = 2  # and reads two sampled frames as one vision block
FRAMES_PER_CHUNK, LATENTS_PER_CHUNK = 17, 5


def _tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise SystemExit(f"{name} is not on the PATH: reference videos and audio are read with it (brew install ffmpeg)")
    return path


def probe_media(path) -> dict:
    """What a media file holds: ``video`` (its displayed width and height) or None, ``audio`` (its channel count) or
    None, and ``seconds``."""

    done = subprocess.run([_tool("ffprobe"), "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
                          capture_output=True, text=True)
    if done.returncode:
        raise SystemExit(f"cannot read {path}: {done.stderr.strip() or 'not a media file'}")
    data = json.loads(done.stdout)
    found = {"video": None, "audio": None, "seconds": 0.0}
    for stream in data.get("streams", []):
        kind = stream.get("codec_type")
        if kind == "video" and found["video"] is None and not stream.get("disposition", {}).get("attached_pic"):
            width, height = int(stream["width"]), int(stream["height"])
            num, _, den = str(stream.get("sample_aspect_ratio", "1:1")).partition(":")
            if num.isdigit() and den.isdigit() and int(num) > 0 and int(den) > 0:
                width = round(width * int(num) / int(den))
            turn = [side.get("rotation") for side in stream.get("side_data_list", []) if "rotation" in side]
            turn = float(turn[0]) if turn else float(stream.get("tags", {}).get("rotate", 0) or 0)
            if round(turn / 90.0) % 2:  # ffmpeg shows a rotated clip upright, so its sides swap
                width, height = height, width
            found["video"] = (width, height)
            found["seconds"] = max(found["seconds"], float(stream.get("duration") or 0.0))
        elif kind == "audio" and found["audio"] is None:
            found["audio"] = int(stream.get("channels") or 0)
            found["audio_seconds"] = float(stream.get("duration") or 0.0)
    found["seconds"] = found["seconds"] or float(data.get("format", {}).get("duration") or 0.0)
    found["audio_seconds"] = found.get("audio_seconds") or found["seconds"]
    return found


def check_clip_lengths(kind: str, lengths) -> None:
    """The model card's limits for reference videos or audio clips: each 2 to 15 s, 15 s in all."""

    low, high = CLIP_SECONDS
    for path, seconds in lengths:
        if not low - CLIP_TOLERANCE <= seconds <= high + CLIP_TOLERANCE:
            raise SystemExit(f"a reference {kind} is {low:g} to {high:g} seconds long, and {path} is {seconds:.1f} s")
    total = sum(seconds for _, seconds in lengths)
    if total > TOTAL_SECONDS + CLIP_TOLERANCE:
        raise SystemExit(f"the reference {kind}s are at most {TOTAL_SECONDS:g} seconds in all, got {total:.1f} s")


def reference_video_size(width: int, height: int, canvas_width: int, canvas_height: int,
                         mode: str = "match") -> tuple[int, int]:
    """The size a reference video is encoded at, each side a multiple of 32, its own aspect ratio kept.

    ``full`` is the model's own canvas for that aspect ratio (768 on the short side, at most 768x1344 pixels), never
    enlarged from a smaller clip (as h3.c and ComfyUI do). ``match`` also brings it down to the clip's pixel area,
    as reference pictures are: every latent of a reference rides through every pass, so a 5 s reference at the full
    canvas more than triples the rows of an 864x480 clip.
    """

    ratio = width / height
    if not 1.0 / ASPECT_LIMIT <= ratio <= ASPECT_LIMIT:
        raise SystemExit(f"a reference video is between 1:4 and 4:1, got {width}x{height}")
    wide, tall = (MODEL_SHORT_EDGE * ratio, float(MODEL_SHORT_EDGE)) if ratio >= 1.0 else \
        (float(MODEL_SHORT_EDGE), MODEL_SHORT_EDGE / ratio)
    if wide * tall > MODEL_MAX_PIXELS:
        scale = math.sqrt(MODEL_MAX_PIXELS / (wide * tall))
        wide, tall = wide * scale, tall * scale
    if width * height < wide * tall:
        wide, tall = float(width), float(height)
    if mode == "match" and wide * tall > canvas_width * canvas_height:
        scale = math.sqrt(canvas_width * canvas_height / (wide * tall))
        wide, tall = wide * scale, tall * scale
    return (max(REFERENCE_MULTIPLE, round(wide / REFERENCE_MULTIPLE) * REFERENCE_MULTIPLE),
            max(REFERENCE_MULTIPLE, round(tall / REFERENCE_MULTIPLE) * REFERENCE_MULTIPLE))


def read_video_frames(path, width: int, height: int, limit: int) -> np.ndarray:
    """Up to ``limit`` frames of a video as (frames, height, width, 3) uint8, at 24 fps and the given size: the
    constant-rate resample and LANCZOS scale the model's own pipeline decodes a reference with."""

    done = subprocess.run([_tool("ffmpeg"), "-v", "error", "-i", str(path), "-map", "0:v:0", "-an", "-vf",
                           f"fps={h3.FPS},scale={width}:{height}:flags=lanczos,setsar=1", "-frames:v", str(limit),
                           "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"], capture_output=True)
    size = width * height * 3
    if done.returncode or len(done.stdout) < size:
        raise SystemExit(f"cannot read frames from {path}: {done.stderr.decode(errors='replace').strip()}")
    return np.frombuffer(done.stdout[: len(done.stdout) // size * size], dtype=np.uint8).reshape(-1, height, width, 3)


def read_waveform(path, rate: int, seconds: float, channels: int) -> np.ndarray:
    """The first ``seconds`` of a file's sound as (2, samples) float32 at ``rate``; mono is repeated on both sides."""

    mix = ["-ac", "2"] if channels > 2 else []
    done = subprocess.run([_tool("ffmpeg"), "-v", "error", "-i", str(path), "-map", "0:a:0", "-vn", *mix, "-ar",
                           str(rate), "-t", f"{seconds:.6f}", "-f", "f32le", "pipe:1"], capture_output=True)
    count = min(channels, 2)
    if done.returncode or len(done.stdout) < 4 * count:
        raise SystemExit(f"cannot read sound from {path}: {done.stderr.decode(errors='replace').strip()}")
    wave = np.frombuffer(done.stdout[: len(done.stdout) // (4 * count) * 4 * count], dtype=np.float32)
    wave = wave.reshape(-1, count).T[:, : round(seconds * rate)]
    return np.ascontiguousarray(np.repeat(wave, 2, axis=0) if count == 1 else wave)


def encoded_frames(frames: int) -> int:
    """Frames of a reference video the video VAE encodes: the count cut down to ``17 n + 5`` (22 at least)."""

    return max(1, (frames - LATENTS_PER_CHUNK) // FRAMES_PER_CHUNK) * FRAMES_PER_CHUNK + LATENTS_PER_CHUNK


def sample_video(frames: np.ndarray):
    """The frames of a reference video the text encoder sees and the time written before each vision block.

    Every 12th frame (2 fps); the encoder reads them two at a time, the last one twice when their number is odd,
    and a block is labelled with the mean of its two times, to one decimal. Also returns how many were sampled
    before the last was repeated.
    """

    picked = frames[::QWEN_SAMPLE_STRIDE]
    sampled = int(picked.shape[0])
    times = [index / (h3.FPS / QWEN_SAMPLE_STRIDE) for index in range(sampled)]
    if sampled % QWEN_FRAMES_PER_BLOCK:
        picked, times = np.concatenate([picked, picked[-1:]]), times + times[-1:]
    stamps = [(times[i] + times[i + 1]) / 2 for i in range(0, len(times), QWEN_FRAMES_PER_BLOCK)]
    return picked, stamps, sampled


def video_patches(frames: np.ndarray, mean, std, patch: int, merge: int, budget: int | None = None,
                  sampled: int | None = None):
    """Sampled frames (an even number, sides multiples of patch * merge) as the vision tower's patch rows and their
    (blocks, grid height, grid width), in the Qwen3-VL video processor's layout.

    ``budget`` is that processor's limit on frames * height * width; frames over it are scaled down as it does
    (bicubic, by the ``sampled`` count before any repeat), which only a long reference at the model's full canvas
    reaches.
    """

    from PIL import Image

    count, height, width = frames.shape[:3]
    factor = patch * merge
    sampled = sampled or count
    # the processor's own test: the sampled count rounded to whole blocks (half to even), times the frame's area
    if budget is not None and round(sampled / QWEN_FRAMES_PER_BLOCK) * QWEN_FRAMES_PER_BLOCK * height * width > budget:
        beta = math.sqrt(sampled * height * width / budget)
        height, width = (max(factor, math.floor(side / beta / factor) * factor) for side in (height, width))
        frames = np.stack([np.asarray(Image.fromarray(frame).resize((width, height), Image.Resampling.BICUBIC))
                           for frame in frames])
    pixels = (frames.astype(np.float32) / 255.0 - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    blocks, grid_h, grid_w = count // QWEN_FRAMES_PER_BLOCK, height // patch, width // patch
    pixels = pixels.transpose(0, 3, 1, 2).reshape(blocks, QWEN_FRAMES_PER_BLOCK, 3, grid_h // merge, merge, patch,
                                                  grid_w // merge, merge, patch)
    rows = pixels.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(blocks * grid_h * grid_w, -1)
    return np.ascontiguousarray(rows), (blocks, grid_h, grid_w)


def reference_request(encoder, prompt: str, items, video_stats, budget: int | None = None):
    """The token sequence of a Ref2VA request with videos or audio, after MiniMax's ``build_ref2va_presentation``.

    Each reference puts a label ahead of the prompt, in order and numbered per kind: ``<Picture i>: `` and a vision
    block; ``<Audio j>: `` alone (sound never reaches the text encoder); ``<Video k>: `` and one vision block per two
    sampled frames, each after ``<1.2 seconds>``. A video used with its sound has ``<Audio j>: `` before its
    ``<Video k>: ``. Returns the ids, their tags (a vision block is tagged video), the patch rows of every vision
    block in order, the grid of each block, and the picture and video grids the positions are built from.
    """

    from tensorfold.families.h3.packing import TAG_TEXT, TAG_VIDEO

    tokenizer, processor = encoder.tokenizer, encoder._processor.image_processor
    patch, merge = encoder.vision_config.patch_size, encoder.merge_size
    start, end = encoder.vision_start_token_id, encoder.vision_end_token_id
    image_pad, video_pad = encoder.image_token_id, encoder.model_config.video_token_id
    ids, tags, rows, grids, image_grids, video_grids = [], [], [], [], [], []
    counts = {"image": 0, "video": 0, "audio": 0}

    def label(value: str):
        tokens = tokenizer(value, add_special_tokens=False)["input_ids"]
        ids.extend(tokens)
        tags.extend([TAG_TEXT] * len(tokens))

    def block(pad: int, tokens: int):
        ids.extend([start] + [pad] * tokens + [end])
        tags.extend([TAG_VIDEO] * (tokens + 2))

    for item in items:
        if item.get("sound") is not None:
            counts["audio"] += 1
            label(f"<Audio {counts['audio']}>: ")
        if item["kind"] == "image":
            counts["image"] += 1
            label(f"<Picture {counts['image']}>: ")
            vision = processor(images=[item["image"]], return_tensors="np")
            grid = [int(v) for v in np.asarray(vision["image_grid_thw"])[0]]
            block(image_pad, grid[0] * grid[1] * grid[2] // merge**2)
            rows.append(np.asarray(vision["pixel_values"], np.float32))
            grids.append(grid)
            image_grids.append(grid)
        elif item["kind"] == "video":
            counts["video"] += 1
            label(f"<Video {counts['video']}>: ")
            picked, stamps, sampled = sample_video(item["frames"])
            patches, (blocks, grid_h, grid_w) = video_patches(picked, *video_stats, patch, merge, budget, sampled)
            for stamp in stamps:
                label(f"<{stamp:.1f} seconds>")
                block(video_pad, grid_h * grid_w // merge**2)
            rows.append(patches)
            grids.extend([[1, grid_h, grid_w]] * blocks)
            video_grids.append([blocks, grid_h, grid_w])
    label(prompt)
    return ids, np.array(tags, dtype=np.int64), np.concatenate(rows), grids, image_grids, video_grids


def encode_reference_text(root: Path, prompt: str, items, weights=None, checkpoint_stats: bool = True):
    """Text rows and tags of a Ref2VA request whose references include a video or an audio clip."""

    from minimax_h3_mlx.text_encoder import _masked_scatter
    from mlx_vlm.models.qwen3_vl.language import LanguageModel

    encoder = _text_encoder(root, True, weights, checkpoint_stats)
    with open(root / "processor" / "video_preprocessor_config.json") as handle:
        budget = json.load(handle).get("size", {}).get("longest_edge")
    ids, tags, rows, grids, image_grids, video_grids = reference_request(
        encoder, prompt, items, vision_stats(root, "video_preprocessor_config.json"), budget)
    input_ids = mx.array(np.array([ids], dtype=np.int32))
    hidden, deepstack = encoder.vision(mx.array(rows).astype(encoder.dtype), mx.array(np.array(grids, np.int32)),
                                       output_hidden_states=True)
    embeds = encoder.language.embed_tokens(input_ids)
    mask = (input_ids == encoder.image_token_id) | (input_ids == encoder.model_config.video_token_id)
    expanded = mx.broadcast_to(mask[..., None], embeds.shape)
    features = hidden.astype(embeds.dtype)
    if int(expanded.sum().item()) != features.size:
        raise ValueError(f"the request reserves {int(mask.sum().item())} vision rows and the vision tower made "
                         f"{features.shape[0]}")
    embeds = _masked_scatter(embeds, expanded, features)
    position_ids, _ = LanguageModel.get_rope_index(
        encoder, input_ids, image_grid_thw=mx.array(np.array(image_grids, np.int32)) if image_grids else None,
        video_grid_thw=mx.array(np.array(video_grids, np.int32)) if video_grids else None, attention_mask=None)
    states = encoder._hidden_states(input_ids, position_ids, inputs_embeds=embeds, visual_pos_masks=mask,
                                    deepstack_visual_embeds=deepstack)
    mx.eval(states)
    del encoder
    gc.collect()
    mx.clear_cache()
    return states, tags


def prepare_reference_media(images, videos, audios, width: int, height: int, frames: int, rate: int,
                            video_sound: bool = False, video_size: str = "match"):
    """Read every reference of a request, in the order the model is given them: pictures, videos, audio clips.

    Returns one dict per reference: ``kind``, ``path``, ``name`` (what the prompt calls it) and ``image`` (PIL),
    ``frames`` (uint8, at most the clip's own frame count) or ``sound`` ((2, samples) float32 at ``rate``, cut to
    the clip's length). A video carries ``sound`` too when ``video_sound`` asks for its own soundtrack as a
    reference; that sound is then numbered among the ``<Audio j>`` ahead of the audio clips.
    """

    items = [{"kind": "image", "path": path, "image": image, "name": f"<Picture {index + 1}>"}
             for index, (path, image) in enumerate(zip(images, prepare_references(images, width, height), strict=True))]
    seconds = frames / h3.FPS
    probes = [(path, probe_media(path)) for path in videos]
    for path, probe in probes:
        if probe["video"] is None:
            raise SystemExit(f"{path} has no picture: a reference video is a video file")
    check_clip_lengths("video", [(path, probe["seconds"]) for path, probe in probes])
    sounds = [(path, probe_media(path)) for path in audios]
    for path, probe in sounds:
        if not probe["audio"]:
            raise SystemExit(f"{path} has no sound: a reference audio clip is an audio file (or a video with sound)")
    check_clip_lengths("audio clip", [(path, probe["audio_seconds"]) for path, probe in sounds])
    spoken = 0
    for index, (path, probe) in enumerate(probes):
        size = reference_video_size(*probe["video"], width, height, video_size)
        item = {"kind": "video", "path": path, "frames": read_video_frames(path, *size, frames),
                "name": f"<Video {index + 1}>", "sound": None}
        if video_sound and probe["audio"]:
            spoken += 1
            item["sound"] = read_waveform(path, rate, min(seconds, item["frames"].shape[0] / h3.FPS), probe["audio"])
            item["sound_name"] = f"<Audio {spoken}>"
        items.append(item)
    for index, (path, probe) in enumerate(sounds):
        items.append({"kind": "audio", "path": path, "name": f"<Audio {spoken + index + 1}>",
                      "sound": read_waveform(path, rate, seconds, probe["audio"])})
    return items


def encode_reference_rows(root: Path, items, patch):
    """Conditioning rows of a request's references and their geometry, in order.

    Returns the video rows (pictures and video frames), the audio rows (soundtracks and audio clips, channel-major)
    or None, and one ``packing.Reference`` per item. A video is cut to ``17 n + 5`` frames and encoded in chunks of
    17, the last one padded with its final frame and the three latents past the end dropped, which leaves
    ``5 n + 2`` latent frames. As for a picture, the middle of the encoder's distribution is used. Sound is the
    audio VAE's posterior mean per stereo side, normalised like the clip's own audio latents and left clean.
    """

    from minimax_h3_mlx.load import load_audio_vae, load_video_vae
    from minimax_h3_mlx.pipeline import PIXEL_MEAN, PIXEL_STD

    from tensorfold.families.h3.packing import Reference, patchify

    video_rows, audio_rows, geometry = [], [], []
    visual = [item for item in items if item["kind"] != "audio"]
    shapes = {}
    if visual:
        vae = load_video_vae(root / "video_vae")
        cfg = vae.config
        latents_mean = mx.array(np.array(cfg.latents_mean, np.float32)).reshape(1, -1, 1, 1, 1)
        latents_std = mx.array(np.array(cfg.latents_std, np.float32)).reshape(1, -1, 1, 1, 1)
        pixel_mean, pixel_std = np.array(PIXEL_MEAN, np.float32), np.array(PIXEL_STD, np.float32)
        for item in visual:
            if item["kind"] == "image":
                frames = np.asarray(item["image"], dtype=np.uint8)[None]
                chunks = [frames]
            else:
                frames = item["frames"][: encoded_frames(item["frames"].shape[0])]
                short = -frames.shape[0] % cfg.clip_length
                frames = np.concatenate([frames, np.repeat(frames[-1:], short, axis=0)]) if short else frames
                chunks = [frames[at:at + cfg.clip_length] for at in range(0, frames.shape[0], cfg.clip_length)]
            means = []
            for chunk in chunks:  # channels last: (1, frames, height, width, 3)
                pixels = (chunk.astype(np.float32) / 255.0 - pixel_mean) / pixel_std
                moments = vae._encode_clip(mx.array(pixels[None]))
                means.append(moments[..., :cfg.latent_channels].astype(mx.float32))
                mx.eval(means[-1])
            mean = mx.concatenate(means, axis=1)
            if item["kind"] == "video" and cfg.token_drop > 0:
                mean = mean[:, :-cfg.token_drop]
            mean = mean.transpose(0, 4, 1, 2, 3)
            if item["kind"] == "image" and mean.shape[2] != 1:
                raise ValueError(f"a reference picture should encode to one latent frame, got {mean.shape[2]}")
            video_rows.append(patchify((mean - latents_mean) / latents_std, patch))
            mx.eval(video_rows[-1])
            shapes[id(item)] = tuple(int(v) for v in mean.shape[2:])
        del vae
        gc.collect()
        mx.clear_cache()
    lengths = {}
    if any(item.get("sound") is not None for item in items):
        vae = load_audio_vae(root / "audio_vae")
        cfg = vae.config
        latents_mean = mx.array(np.array(cfg.latents_mean, np.float32)).reshape(1, 1, -1)
        latents_std = mx.array(np.array(cfg.latents_std, np.float32)).reshape(1, 1, -1)
        for item in items:
            if item.get("sound") is None:
                continue
            mean, _ = vae.encode(mx.array(item["sound"])[:, None, :])  # each stereo side is a batch item
            mean = mean.astype(mx.float32).transpose(0, 2, 1)  # (2, latents, channels)
            audio_rows.append(((mean - latents_mean) / latents_std).reshape(-1, mean.shape[-1]))
            mx.eval(audio_rows[-1])
            lengths[id(item)] = int(mean.shape[1])
        del vae
        gc.collect()
        mx.clear_cache()
    for item in items:
        if item["kind"] == "audio":
            geometry.append(Reference("audio", audio_latents=lengths[id(item)]))
        else:
            count, tall, wide = shapes[id(item)]
            geometry.append(Reference(item["kind"], tall, wide, count, lengths.get(id(item), 0)))
    return (mx.concatenate(video_rows) if video_rows else None,
            mx.concatenate(audio_rows) if audio_rows else None, tuple(geometry))


def _lazy_fasth3():
    """Let the native engine's export read only the FastH3 weights it uses.

    The engine's h3_case.py imports this file as its tool and calls `fasth3.load_fasth3`, which ends by reading the
    whole 65 GB checkpoint into memory (about 50 GiB at its peak). The export needs the small projections and the
    modulation tables; the block weights are read by the native program itself. With the final `eval` skipped the
    weights stay on disk until something uses them, and the export peaks at a few GiB with the same result.
    """

    from tensorfold.families.h3 import fasth3

    load = fasth3.load_fasth3
    if getattr(load, "lazy", False):
        return

    def load_fasth3(*args, **kwargs):
        evaluate = mx.eval
        mx.eval = lambda *_: None
        try:
            return load(*args, **kwargs)
        finally:
            mx.eval = evaluate

    load_fasth3.lazy = True
    fasth3.load_fasth3 = load_fasth3


if __name__ == "h3_generate_dev":  # imported by the native engine's export, not run as the MLX renderer
    _lazy_fasth3()
    mx.set_cache_limit(2 * 2**30)  # freed buffers are otherwise kept for reuse, tens of GiB of them


AUDIO_PEAK = 0.89  # -1 dBFS: where the loudest sample of a clip is brought to when the decoder overshoots


def unclipped_audio(audio_vae, latents, hard_clip: bool = False):
    """The audio decoder's waveform without its hard clip at +-1, scaled down as a whole when it overshoots.

    minimax-h3-mlx's vocoder ends in `mx.clip(x, -1, 1)`. Few-step renders drive it past full scale in places and
    the clip flattens those peaks, which is audible as harshness. Here the clip is lifted for the call and a clip
    whose peak exceeds `AUDIO_PEAK` is turned down by one gain, so no sample is flattened.
    """

    import minimax_h3_mlx.audio_vae as module

    class _Passthrough:
        def __getattr__(self, name):
            return getattr(mx, name)

        @staticmethod
        def clip(x, *_):
            return x

    original = module.mx
    module.mx = _Passthrough()
    try:
        wave = np.array(audio_vae.decode(latents))[:, 0, :].astype(np.float32)
    finally:
        module.mx = original
    peak = float(np.abs(wave).max())
    over = int((np.abs(wave) > 1.0).sum())
    if hard_clip:  # the reference decoder's behaviour, kept for comparisons
        print(f"[tensorfold] audio peak {peak:.2f}, {over} samples hard-clipped at full scale", flush=True)
        return np.clip(wave, -1.0, 1.0)
    gain = min(1.0, AUDIO_PEAK / peak) if peak > 0 else 1.0
    print(f"[tensorfold] audio peak {peak:.2f} before any clip, {over} samples past full scale, gain "
          f"{20 * np.log10(gain):.1f} dB", flush=True)
    return wave * np.float32(gain)


def decode(model_dir, root: Path, latents, config, int8: bool = True, upscale_decoder=None, hard_clip: bool = False):
    """Frames from TensorFold's video decoder; the audio decoder is still minimax-h3-mlx's."""

    from minimax_h3_mlx.load import load_audio_vae

    from tensorfold.families.h3.vae_video import load_video_decoder

    parts, mark = {}, time.perf_counter()

    def lap(name):
        nonlocal mark
        parts[name] = round(time.perf_counter() - mark, 2)
        mark = time.perf_counter()

    video_decoder = load_video_decoder(model_dir, int8=int8, upscale_decoder=upscale_decoder)
    audio_vae = load_audio_vae(root / "audio_vae")
    lap("load_vaes")
    video = unpatchify(latents.video_rows, latents.latent_frames, latents.latent_height, latents.latent_width,
                       video_decoder.config.latent_channels, config.patch_size)
    frames = video_decoder.frames(video)
    lap("video_decode")
    acfg = audio_vae.config
    audio = unpack_audio(latents.audio_rows, latents.audio_latents)
    amean = mx.array(np.array(acfg.latents_mean, np.float32)).reshape(1, -1, 1)
    astd = mx.array(np.array(acfg.latents_std, np.float32)).reshape(1, -1, 1)
    wave = unclipped_audio(audio_vae, (audio * astd + amean).astype(mx.float32), hard_clip)
    lap("audio_decode")
    print(f"[tensorfold] decode_parts {parts}", flush=True)
    return frames, wave, acfg.sampling_rate


def parity(root: Path, dit, args, text, tags):
    """Largest and relative difference between this transformer and the reference on one real forward."""

    from minimax_h3_mlx.load import load_dit as load_reference

    from tensorfold.families.h3.packing import layout, timestep_plan
    from tensorfold.families.h3.sampler import start_noise
    from tensorfold.families.h3.schedule import AUDIO_SHIFT, VIDEO_SHIFT, Schedule

    config = dit.config
    frames, lat_h, lat_w = h3.latent_frames(args.frames), args.height // 16, args.width // 16
    packed = layout(tags, frames, lat_h, lat_w, h3.audio_latents(args.frames), config.patch_size)
    video, audio = start_noise(config, frames, lat_h, lat_w, h3.audio_latents(args.frames), args.seed)
    table, plan = timestep_plan(packed, Schedule(VIDEO_SHIFT, args.points).timesteps,
                                Schedule(AUDIO_SHIFT, args.points).timesteps)
    # the reference rounds the latent rows to bfloat16 on the way in; do the same so the comparison is like for like
    call = (video[None].astype(mx.bfloat16), audio[None].astype(mx.bfloat16), text.astype(mx.bfloat16), table,
            plan[0], packed.tags,
            packed.position_ids, packed.video_rows, packed.audio_rows, packed.text_rows)
    ours = dit(*call)
    mx.eval(*ours)
    reference = load_reference(root / "transformer")
    theirs = reference(*call)
    mx.eval(*theirs)
    for name, a, b in zip(("video", "audio"), ours, theirs, strict=True):
        a, b = a.astype(mx.float32), b.astype(mx.float32)
        diff = float(mx.max(mx.abs(a - b)).item())
        rel = float((mx.linalg.norm(a - b) / mx.linalg.norm(b)).item())
        print(f"[tensorfold] parity {name}: max abs diff {diff:.3e}, relative {rel:.3e}, "
              f"reference rms {float(mx.sqrt(mx.mean(b * b)).item()):.3f}", flush=True)
    del reference
    gc.collect()
    mx.clear_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("model_dir")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--width", type=int, default=864)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--points", type=int, default=21, help="sigma points; one fewer forwards")
    parser.add_argument("--subset", default=None, help="N:i0,i1,... keeps those points of an N-point grid")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lora", action="append", default=[], help="PATH or PATH:STRENGTH, merged at load")
    parser.add_argument("--int8-mlp", action="store_true", help="run the block MLPs through the int8 kernel")
    parser.add_argument("--int8-qkv", action="store_true", help="int8 QKV projection")
    parser.add_argument("--int8-out", action="store_true", help="int8 attention-output projection")
    parser.add_argument("--unfused-qkv", action="store_true", help="int8 QKV without the fused norm and rotation")
    parser.add_argument("--keep-adaln", action="store_true", help="keep the AdaLN projection weights loaded")
    parser.add_argument("--audio-shift", type=float, default=None, help="sigma shift of the audio schedule (released: 3)")
    parser.add_argument("--upscale-vae", help="safetensors of a packed-head (2x) video decoder; frames come out larger")
    parser.add_argument("--crop", help="WxH: centre-crop the decoded frames before the MP4 is written")
    parser.add_argument("--fasth3", help="a FastH3 checkpoint folder: its transformer, schedule and routed attention")
    parser.add_argument("--fasth3-q8", action="store_true",
                        help="with --fasth3: the 8-bit weights written by fasth3_q8.py, for chips without tensor units")
    parser.add_argument("--vsa-impl", default="tensor", choices=("tensor", "reference", "simd"),
                        help="routed attention: our int8 tile kernel, or FastVideo's reference or SIMD-group forms")
    parser.add_argument("--fasth3-steps", type=int, default=0, metavar="N",
                        help="with --fasth3: N forwards instead of the trained count; rungs are spread evenly from "
                             "the checkpoint's first rung, which keeps the trained rungs when N divides their number")
    parser.add_argument("--dense", action="store_true", help="with --fasth3: dense attention instead of routed")
    parser.add_argument("--step-cache", type=float, default=0.0,
                        help="reuse the last velocity while the summed relative move stays under this (0.05)")
    parser.add_argument("--attention-every", type=int, default=0,
                        help="compute attention every N-th middle step and reuse it otherwise (2)")
    parser.add_argument("--fast-gates", help="A,B,C: opening and closing steps that always compute attention, and "
                                             "steps before the velocity cache may skip")
    parser.add_argument("--save-frames", help="also write the decoded frames to this .npy file, for comparisons")
    parser.add_argument("--revoice", type=int, default=0, metavar="STEPS",
                        help="after the run, denoise the audio again in STEPS steps with the model without adapters")
    parser.add_argument("--revoice-exact", action="store_true", help="re-voice with whole-sequence forwards")
    parser.add_argument("--audio-hard-clip", action="store_true",
                        help="clip the decoded audio at full scale, as the reference decoder does (for comparisons)")
    parser.add_argument("--float-vae", action="store_true", help="video decoder in float32, without int8 kernels")
    parser.add_argument("--first-frame", default=None, help="image the clip starts from (image to video)")
    parser.add_argument("--ref-image", action="append", default=[], metavar="PATH",
                        help="a reference picture (repeatable, up to 9): <Picture 1>, <Picture 2>, ... in the "
                             "prompt. Needs the Ref2VA transformer; not with --first-frame or --fasth3")
    parser.add_argument("--ref-video", action="append", default=[], metavar="PATH",
                        help="a reference video (repeatable, up to 3, each 2 to 15 s, 15 s in all): <Video 1>, ... in "
                             "the prompt, after the pictures. Read at 24 fps up to the clip's own length")
    parser.add_argument("--ref-audio", action="append", default=[], metavar="PATH",
                        help="a reference audio clip (repeatable, up to 3, each 2 to 15 s, 15 s in all): <Audio 1>, "
                             "... in the prompt. Only beside a reference picture or video")
    parser.add_argument("--ref-video-sound", action="store_true",
                        help="also use each reference video's own soundtrack as a reference; those sounds are then "
                             "<Audio 1>, ... ahead of the --ref-audio clips")
    parser.add_argument("--ref-video-size", default="match", choices=("match", "full"),
                        help="match: a reference video is brought down to the clip's pixel area (default); full: "
                             "the model's own 768-pixel canvas, several times slower")
    parser.add_argument("--vision-stats", default="checkpoint", choices=("checkpoint", "mlx"),
                        help="how a first frame or a reference picture is normalised for the text encoder's vision "
                             "tower: the checkpoint's own processor values (default) or minimax-h3-mlx's fixed "
                             "ones, which releases up to 2.5 used (for comparisons)")
    parser.add_argument("--text-encoder", default=None, metavar="PATH",
                        help="a ComfyUI int8 (int8_tensorwise, ConvRot) safetensors of the Qwen3-VL text encoder, "
                             "decoded to bfloat16 as it loads; the tokenizer stays the model folder's")
    parser.add_argument("--parity", action="store_true")
    args = parser.parse_args()
    if args.text_encoder and not (args.text_encoder.endswith(".safetensors") and Path(args.text_encoder).is_file()):
        raise SystemExit(f"--text-encoder takes a .safetensors file, and {args.text_encoder} is not one")

    references = None
    wanted = bool(args.ref_image or args.ref_video or args.ref_audio)
    if wanted:
        if args.first_frame or args.fasth3:
            raise SystemExit("references cannot be combined with --first-frame or --fasth3")
        if len(args.ref_image) > REFERENCE_LIMIT:
            raise SystemExit(f"at most {REFERENCE_LIMIT} reference pictures, got {len(args.ref_image)}")
        if len(args.ref_video) > VIDEO_LIMIT:
            raise SystemExit(f"at most {VIDEO_LIMIT} reference videos, got {len(args.ref_video)}")
        if len(args.ref_audio) > AUDIO_LIMIT:
            raise SystemExit(f"at most {AUDIO_LIMIT} reference audio clips, got {len(args.ref_audio)}")
        total = len(args.ref_image) + len(args.ref_video) + len(args.ref_audio)
        if total > TOTAL_LIMIT:
            raise SystemExit(f"at most {TOTAL_LIMIT} references in all, got {total}")
        if args.ref_audio and not (args.ref_image or args.ref_video):
            raise SystemExit("a reference audio clip needs a reference picture or video beside it: the model does "
                             "not take sound alone")
        if (Path(args.model_dir) / "Ref2VA" / "transformer").is_dir():
            args.model_dir = str(Path(args.model_dir) / "Ref2VA")
    root = h3.pipeline_root(args.model_dir)
    if root is None:
        raise SystemExit(f"{args.model_dir} is not a MiniMax H3 pipeline folder")
    media = None
    if wanted:
        with open(root / "model_index.json") as handle:
            partition = json.load(handle).get("_minimax_h3", {}).get("partition")
        if partition != "ref2va":
            raise SystemExit(f"references need the Ref2VA transformer; {root} is {partition or 'not marked'} "
                             "(oneshot-setup.sh --ref2va fetches it)")
        missing = [name for name in ("text_encoder", "tokenizer", "video_vae", "audio_vae") if not (root / name).exists()]
        if missing:
            raise SystemExit(f"{root} lacks {', '.join(missing)}: link them from FL2VA (oneshot-setup.sh --ref2va)")
        if args.ref_video or args.ref_audio:
            from tensorfold.families.h3 import packing

            if not hasattr(packing, "reference_layout"):
                raise SystemExit("reference videos and audio need a TensorFold whose H3 family packs references "
                                 "(packing.reference_layout); this one takes reference pictures only")
            if not (root / "processor" / "video_preprocessor_config.json").is_file():
                raise SystemExit(f"{root} lacks processor/video_preprocessor_config.json: link processor from FL2VA "
                                 "(oneshot-setup.sh --ref2va)")
            with open(root / "audio_vae" / "metadata.json") as handle:
                rate = int(json.load(handle)["metadata"]["kwargs"]["sample_rate"])
            media = prepare_reference_media(args.ref_image, args.ref_video, args.ref_audio, args.width, args.height,
                                            args.frames, rate, args.ref_video_sound, args.ref_video_size)
            for item in media:
                if item["kind"] == "image":
                    what = "{}x{}".format(*item["image"].size)
                elif item["kind"] == "video":
                    count, tall, wide = item["frames"].shape[:3]
                    what = f"{wide}x{tall}, {count} frames ({count / h3.FPS:.1f} s)"
                    if item["sound"] is not None:
                        what += f", its sound is {item['sound_name']} ({item['sound'].shape[1] / rate:.1f} s)"
                else:
                    what = f"{item['sound'].shape[1] / rate:.1f} s"
                print(f"[tensorfold] reference {item['name']}: {item['path']} ({what})", flush=True)
        else:
            references = prepare_references(args.ref_image, args.width, args.height)
    prompt = Path(args.prompt_file).read_text() if args.prompt_file else args.prompt
    started = time.perf_counter()
    image = None
    if args.first_frame:
        from minimax_h3_mlx.packing import prepare_keyframe_image
        from PIL import Image

        image = prepare_keyframe_image(Image.open(args.first_frame).convert("RGB"), args.height, args.width,
                                       stretch=True)
    stats = args.vision_stats == "checkpoint"
    condition = audio_condition = geometry = None
    patch = h3.DiTConfig.from_checkpoint(args.model_dir).patch_size
    if media is not None:
        text, tags = encode_reference_text(root, prompt, media, args.text_encoder, stats)
        condition, audio_condition, geometry = encode_reference_rows(root, media, patch)
        print(f"[tensorfold] references: {len(geometry)} in all, "
              f"{0 if condition is None else condition.shape[0]} video rows, "
              f"{0 if audio_condition is None else audio_condition.shape[0]} audio rows held", flush=True)
    else:
        text, tags = encode_text(root, prompt, references if references is not None else image, args.text_encoder,
                                 stats)
    if references is not None:
        from tensorfold.families.h3 import sampler

        condition, shapes = encode_references(root, references, patch)
        sampler.layout = reference_layout(shapes)  # the sampler packs the references ahead of the clip
        print(f"[tensorfold] references: {len(shapes)} pictures at "
              f"{', '.join(f'{w * 16}x{h * 16}' for h, w in shapes)}, {condition.shape[0]} rows", flush=True)
    if image is not None:
        condition = encode_first_frame(root, image, args.width, args.height, patch)
    text_seconds = time.perf_counter() - started
    print(f"[tensorfold] text: {text.shape[1]} rows in {text_seconds:.1f}s", flush=True)

    started = time.perf_counter()
    fast = None
    if args.fasth3:
        from tensorfold.families.h3 import fasth3

        if args.fasth3_q8:
            import fasth3_q8

            dit, gates, fast = fasth3_q8.load(args.fasth3)
            # the blocks' weights are MLX quantized layers now; the int8 tensor-unit kernels do not apply to them
            args.int8_mlp = args.int8_qkv = args.int8_out = False
            mx.set_cache_limit(8 * 2**30)  # room for reuse without the default tens of GiB; 4 GiB cost a third in speed
            print("[tensorfold] FastH3 with 8-bit weights (MLX quantized layers)", flush=True)
        else:
            dit, gates, fast = fasth3.load_fasth3(args.fasth3)
        print(f"[tensorfold] FastH3: {fast.forwards} forwards, video shift {fast.video_shift}, sparsity "
              f"{fast.sparsity}, tile {fast.tile}, task {fast.task}", flush=True)
    else:
        dit = load_dit(args.model_dir)
    for spec in args.lora:
        path, _, strength = spec.partition(":")
        print(f"[tensorfold] {merge(dit, path, float(strength or 1.0))}", flush=True)
    if args.int8_mlp:
        print(f"[tensorfold] int8 MLP in {int8_mlp(dit)} blocks", flush=True)
    if args.int8_qkv or args.int8_out:
        changed = int8_attention(dit, qkv=args.int8_qkv, out=args.int8_out, fused=not args.unfused_qkv)
        print(f"[tensorfold] int8 attention projections: {changed}", flush=True)
    rounded = settle(dit)
    if rounded and args.lora:
        print(f"[tensorfold] {rounded} adapted projections rounded to bfloat16 (not on an int8 kernel)", flush=True)
    load_seconds = time.perf_counter() - started
    if args.parity:
        parity(root, dit, args, text, tags)

    points, subset = args.points, None
    if args.subset:
        points, subset = parse_subset(args.subset)
    schedule = {}
    if fast is not None:
        nodes = fast.nodes
        if args.fasth3_steps and args.fasth3_steps != len(nodes):
            nodes = tuple(nodes[0] * (1 - i / args.fasth3_steps) for i in range(args.fasth3_steps))
            print(f"[tensorfold] FastH3 resampled to {len(nodes)} forwards (trained for {fast.forwards})", flush=True)
        schedule = {"nodes": nodes, "video_shift": fast.video_shift}
        if args.audio_shift is None:
            args.audio_shift = fast.audio_shift
        if fast.sparsity > 0 and not args.dense:
            schedule["prepare"] = fasth3.route(dit, gates, fast.sparsity, fast.tile, args.vsa_impl)
    # reference videos and audio go through the family's own reference layout and its held audio rows
    held = {} if geometry is None else {"references": geometry, "audio_condition": audio_condition}
    held_audio = {} if audio_condition is None else {"audio_condition": audio_condition}
    started = time.perf_counter()
    latents = denoise(dit, text, tags, args.width, args.height, args.frames, points, args.seed, subset,
                      release=not args.keep_adaln, condition=condition,
                      keyframes=("first",) if image is not None else (),
                      on_step=lambda i, n, s: print(f"[tensorfold] step {i}/{n} {s:.2f}s", flush=True),
                      audio_shift=args.audio_shift, step_cache=args.step_cache,
                      attention_every=args.attention_every,
                      gates=tuple(int(v) for v in args.fast_gates.split(",")) if args.fast_gates else None,
                      **held, **schedule)
    denoise_seconds = time.perf_counter() - started
    revoice_seconds = 0.0
    if args.revoice:
        from tensorfold.families.h3.sampler import revoice

        started = time.perf_counter()
        del dit
        gc.collect()
        mx.clear_cache()
        dit = load_dit(args.model_dir)  # no adapter: the audio comes from the released weights
        if args.int8_mlp:
            int8_mlp(dit)
        if args.int8_qkv or args.int8_out:
            int8_attention(dit, qkv=args.int8_qkv, out=args.int8_out, fused=not args.unfused_qkv)
        latents.audio_rows = revoice(dit, text, latents, args.revoice + 1, args.seed, condition,
                                     exact=args.revoice_exact, release=not args.keep_adaln, **held_audio,
                                     on_step=lambda i, n, s: print(f"[tensorfold] voice {i}/{n} {s:.2f}s", flush=True))
        revoice_seconds = time.perf_counter() - started
        print(f"[tensorfold] re-voiced in {revoice_seconds:.1f}s", flush=True)
    del dit
    gc.collect()
    mx.clear_cache()

    started = time.perf_counter()
    frames, wave, rate = decode(args.model_dir, root, latents, h3.DiTConfig.from_checkpoint(args.model_dir),
                                int8=not args.float_vae, upscale_decoder=args.upscale_vae,
                                hard_clip=args.audio_hard_clip)
    if args.save_frames:
        np.save(args.save_frames, frames[::8])
    if args.crop:
        crop_w, crop_h = (int(v) for v in args.crop.lower().split("x"))
        full_h, full_w = frames.shape[1:3]
        if crop_w > full_w or crop_h > full_h or (full_w - crop_w) % 2 or (full_h - crop_h) % 2:
            raise SystemExit(f"cannot centre-crop {full_w}x{full_h} frames to {crop_w}x{crop_h}")
        top, left = (full_h - crop_h) // 2, (full_w - crop_w) // 2
        frames = np.ascontiguousarray(frames[:, top : top + crop_h, left : left + crop_w])
    from minimax_h3_mlx.media import save_mp4

    mux_started = time.perf_counter()
    save_mp4(args.output, frames, h3.FPS, audio=wave, sample_rate=rate)
    print(f"[tensorfold] mux {time.perf_counter() - mux_started:.2f}s", flush=True)
    decode_seconds = time.perf_counter() - started
    report = {"output": args.output, "rows": latents.packed.rows, "forwards": len(latents.step_seconds),
              "attention_reused": latents.reused_steps, "skipped": latents.skipped_steps,
              "text_s": round(text_seconds, 1), "load_s": round(load_seconds, 1),
              "denoise_s": round(denoise_seconds, 1), "revoice_s": round(revoice_seconds, 1), "decode_s": round(decode_seconds, 1),
              "per_forward_s": round(float(np.median(latents.step_seconds)), 2),
              "peak_gib": round(mx.get_peak_memory() / 2**30, 1)}
    print("[tensorfold] " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
