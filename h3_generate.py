"""Render one MiniMax H3 clip with TensorFold's H3 family (copy of tools/h3_generate_dev.py at the pinned commit).

    PYTHONPATH=src:<minimax-h3-mlx> <minimax-h3-mlx>/.venv/bin/python tools/h3_generate_dev.py \
        ~/h3-models/MiniMax-H3 --prompt-file prompt.txt -o out.mp4

The text encoder, the audio decoder and the MP4 writer come from minimax-h3-mlx until the family has its own;
the transformer, sampler, adapters and video decoder are `tensorfold.families.h3`. `--parity` also runs
the reference transformer on the first step's inputs and reports the difference.

`--ref-image` (repeatable) is reference to video, on the Ref2VA transformer: each picture is named in the prompt as
`<Picture 1>`, `<Picture 2>`, ... and fixes how a person or a thing looks without being a frame of the clip.
`model_dir` is then the folder holding `Ref2VA/` (or the Ref2VA folder itself).

`--text-encoder` takes the text encoder's weights from one ComfyUI int8 file (`int8_tensorwise`, with or without
ConvRot) instead of the model folder, for community builds of the encoder that are published only in that form.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
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


def encode_text(root: Path, prompt: str, images=None, weights=None, checkpoint_stats: bool = True):
    """Text rows and their tags. ``images`` (one picture or a list) go ahead of the prompt, each as
    ``<Picture i>: `` and its vision block, which is how the prompt refers to them. ``weights`` is a ComfyUI int8
    file to take the encoder's weights from, in place of the ones in the model folder. Pictures are normalised
    with the checkpoint's own mean and deviation (:func:`vision_stats`); ``checkpoint_stats`` false keeps the
    values minimax-h3-mlx's image processor is built with, for comparisons with clips made before this was corrected."""

    if images is not None and not isinstance(images, (list, tuple)):
        images = [images]
    images = list(images) if images else None
    import minimax_h3_mlx.text_encoder as module
    from minimax_h3_mlx.text_encoder import MiniMaxH3TextEncoder

    if weights is not None:
        MiniMaxH3TextEncoder = _from_comfy_int8(MiniMaxH3TextEncoder, weights)
    original = module.mx
    module.mx = _NoEval()  # the loader ends by reading every weight into memory; leave them for _streamed
    try:
        encoder = MiniMaxH3TextEncoder(root / "text_encoder", dtype=mx.bfloat16, load_vision=images is not None)
    finally:
        module.mx = original
    if encoder.vision is not None:
        mx.eval(encoder.vision.parameters())
    _streamed(encoder)
    if images is not None:
        # the checkpoint's processor folder asks for PyTorch; the image processor built from the vision config
        # needs only numpy, and the image arrives already on the render canvas. Its mean and deviation are the
        # checkpoint's own, not the ones minimax-h3-mlx builds it with
        from minimax_h3_mlx.text_encoder import _FallbackProcessor

        processor = encoder._build_image_processor()
        processor.image_mean, processor.image_std = vision_stats(root) if checkpoint_stats else MLX_VISION_STATS
        encoder._processor = _FallbackProcessor(processor)
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
    if args.ref_image:
        if args.first_frame or args.fasth3:
            raise SystemExit("--ref-image cannot be combined with --first-frame or --fasth3")
        if len(args.ref_image) > REFERENCE_LIMIT:
            raise SystemExit(f"at most {REFERENCE_LIMIT} reference pictures, got {len(args.ref_image)}")
        if (Path(args.model_dir) / "Ref2VA" / "transformer").is_dir():
            args.model_dir = str(Path(args.model_dir) / "Ref2VA")
    root = h3.pipeline_root(args.model_dir)
    if root is None:
        raise SystemExit(f"{args.model_dir} is not a MiniMax H3 pipeline folder")
    if args.ref_image:
        with open(root / "model_index.json") as handle:
            partition = json.load(handle).get("_minimax_h3", {}).get("partition")
        if partition != "ref2va":
            raise SystemExit(f"--ref-image needs the Ref2VA transformer; {root} is {partition or 'not marked'} "
                             "(oneshot-setup.sh --ref2va fetches it)")
        missing = [name for name in ("text_encoder", "tokenizer", "video_vae", "audio_vae") if not (root / name).exists()]
        if missing:
            raise SystemExit(f"{root} lacks {', '.join(missing)}: link them from FL2VA (oneshot-setup.sh --ref2va)")
        references = prepare_references(args.ref_image, args.width, args.height)
    prompt = Path(args.prompt_file).read_text() if args.prompt_file else args.prompt
    started = time.perf_counter()
    image = None
    if args.first_frame:
        from minimax_h3_mlx.packing import prepare_keyframe_image
        from PIL import Image

        image = prepare_keyframe_image(Image.open(args.first_frame).convert("RGB"), args.height, args.width,
                                       stretch=True)
    text, tags = encode_text(root, prompt, references if references is not None else image, args.text_encoder,
                             args.vision_stats == "checkpoint")
    condition = None
    if references is not None:
        from tensorfold.families.h3 import sampler

        condition, shapes = encode_references(root, references, h3.DiTConfig.from_checkpoint(args.model_dir).patch_size)
        sampler.layout = reference_layout(shapes)  # the sampler packs the references ahead of the clip
        print(f"[tensorfold] references: {len(shapes)} pictures at "
              f"{', '.join(f'{w * 16}x{h * 16}' for h, w in shapes)}, {condition.shape[0]} rows", flush=True)
    if image is not None:
        condition = encode_first_frame(root, image, args.width, args.height,
                                       h3.DiTConfig.from_checkpoint(args.model_dir).patch_size)
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
    started = time.perf_counter()
    latents = denoise(dit, text, tags, args.width, args.height, args.frames, points, args.seed, subset,
                      release=not args.keep_adaln, condition=condition,
                      keyframes=("first",) if image is not None else (),
                      on_step=lambda i, n, s: print(f"[tensorfold] step {i}/{n} {s:.2f}s", flush=True),
                      audio_shift=args.audio_shift, step_cache=args.step_cache,
                      attention_every=args.attention_every,
                      gates=tuple(int(v) for v in args.fast_gates.split(",")) if args.fast_gates else None, **schedule)
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
                                     exact=args.revoice_exact, release=not args.keep_adaln,
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
