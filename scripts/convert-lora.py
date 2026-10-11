"""Rewrite a MiniMax H3 LoRA in ComfyUI's layout into the layout h3_generate.py --lora reads (the Mac engines).

    $PREFIX/venv/bin/python scripts/convert-lora.py 80s_horror_wk.safetensors out.safetensors $H3_MODEL_DIR

The source has `diffusion_model.<module>.lora_A/lora_B.weight` (or kohya's `lora_down`/`lora_up` with `.alpha`)
under the checkpoint's own module names and row order. The result drops the prefix, multiplies alpha / rank into
lora_B and stores the fused QKV rows as [q_all; k_all; v_all], which the loader interleaves back per head. Nothing is
merged here: --lora PATH:STRENGTH does that at load.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import mlx.core as mx

PAIRS = ((".lora_A.weight", ".lora_B.weight"), (".lora_down.weight", ".lora_up.weight"))


def main(source: str, target: str, model_dir: str) -> None:
    with open(source, "rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        metadata = json.loads(handle.read(size)).get("__metadata__") or {}
    config = json.loads(next(Path(model_dir).glob("FL2VA/transformer/config.json")).read_text())
    heads, dim = int(config["num_attention_heads"]), int(config["attention_head_dim"])
    weights = mx.load(source)
    out, ranks, scales = {}, set(), set()
    for key in sorted(weights):
        pair = next((p for p in PAIRS if key.endswith(p[0])), None)
        if pair is None:
            if key.endswith((PAIRS[0][1], PAIRS[1][1], ".alpha")):
                continue
            raise SystemExit(f"{source}: unsupported tensor {key} (a low-rank LoRA with lora_A/lora_B is read)")
        stem = key[: -len(pair[0])]
        module = stem[len("diffusion_model."):] if stem.startswith("diffusion_model.") else stem
        a, b = weights[key], weights[stem + pair[1]].astype(mx.float32)
        rank = a.shape[0]
        if a.ndim != 2 or b.ndim != 2:
            raise SystemExit(f"{source}: {key} is not a linear layer's update")
        alpha = float(weights[stem + ".alpha"].item()) if stem + ".alpha" in weights else float(metadata.get("ss_network_alpha") or rank)
        if module.endswith("qkv_proj"):
            if b.shape[0] != 3 * heads * dim:
                raise SystemExit(f"{source}: {module} has {b.shape[0]} rows, the model's QKV has {3 * heads * dim}")
            b = b.reshape(heads, 3, dim, rank).transpose(1, 0, 2, 3).reshape(3 * heads * dim, rank)
        out[module + ".lora_A.weight"] = a.astype(mx.bfloat16)
        out[module + ".lora_B.weight"] = (b * (alpha / rank)).astype(mx.bfloat16)
        ranks.add(rank)
        scales.add(alpha / rank)
    if not out:
        raise SystemExit(f"{source}: no lora_A/lora_B tensors found")
    mx.save_safetensors(target, out, metadata={
        "converted_from": Path(source).name, "source_namespace": "ComfyUI checkpoint names",
        "source_rank": ",".join(map(str, sorted(ranks))), "peft_scale_folded_into_B": "1",
        "qkv_layout": "contiguous [q_all;k_all;v_all]; loader interleaves exactly once", "fc1_layout": "[gate;value]",
        **{k: str(v) for k, v in metadata.items() if k.startswith("modelspec.") or k == "notice"}})
    print(f"wrote {target}: {len(out) // 2} modules, rank {sorted(ranks)}, scale {sorted(scales)} in lora_B")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
