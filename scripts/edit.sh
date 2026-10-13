#!/bin/bash
# Edit a picture with Qwen-Image-2.1: the picture goes in as the model's reference, the prompt says what changes.
#   bash scripts/edit.sh photo.png "make it night" out.png
#   SET_FILE=edits.json bash scripts/edit.sh photo.png        # several edits from one load: [[prompt, output], ...]
#   LORA=/path/angles.safetensors:0.9 bash scripts/edit.sh photo.png "<mva> back view, eye-level shot" back.png
# Environment: PREFIX, QWEN_MODEL_DIR (Qwen-Image-2.1-Turbo, its own 8 steps), BASE=1 (the 40-step base model at
# QWEN_BASE_DIR: oneshot-setup.sh --image-edit; what image LoRAs are trained on), STEPS (base model only), SEED,
# RESOLUTION (the picture is brought to about this many pixels squared, default 1024), LORA (one PATH or PATH:STRENGTH
# a line), QUANTIZE (bits for the weights on a Mac: 8 on a 64 GB Mac; unset keeps bfloat16, 43 GiB at its peak).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# On a DGX Spark (Linux, CUDA) the same interface is served by spark/edit.sh.
[ "$(uname -s)" != Linux ] || exec bash "$HERE/../spark/edit.sh" "$@"
PREFIX="${PREFIX:-$HOME/.local/opt/tensorfold-studio}"
QWEN_MODEL_DIR="${QWEN_MODEL_DIR:-$HOME/qwen-models/Qwen-Image-2.1-Turbo}"
QWEN_BASE_DIR="${QWEN_BASE_DIR:-$(dirname "$QWEN_MODEL_DIR")/Qwen-Image-2.1}"
IMAGE="${1:-}"; PROMPT="${2:-}"; OUT="${3:-}"
[ -f "$IMAGE" ] || { echo "give the picture to edit: bash scripts/edit.sh photo.png \"what changes\" out.png" >&2; exit 1; }
MODEL="$QWEN_MODEL_DIR"
if [ "${BASE:-0}" = 1 ]; then
  MODEL="$QWEN_BASE_DIR"
  [ -d "$MODEL/transformer" ] || { echo "BASE=1 needs the base model: bash oneshot-setup.sh --image-edit (Qwen-Image-2.1, 13 GB more; Qwen Research License, non-commercial)" >&2; exit 1; }
fi
[ -f "$MODEL/processor/preprocessor_config.json" ] || [ -d "$MODEL/processor" ] || { echo "$MODEL has no processor folder: editing needs the whole checkpoint (text encoder with its eyes, processor, both halves of the image coder)" >&2; exit 1; }
export PATH="/opt/homebrew/bin:$PATH"
ARGS=("$MODEL" --image "$IMAGE" --seed "${SEED:-0}" --resolution "${RESOLUTION:-1024}" --steps "${STEPS:-40}")
if [ -n "${SET_FILE:-}" ]; then ARGS+=(--set "@$SET_FILE"); else ARGS+=(--prompt "$PROMPT" -o "$OUT"); fi
[ -z "${QUANTIZE:-}" ] || ARGS+=(--quantize "$QUANTIZE")
if [ -n "${LORA:-}" ]; then
  while IFS= read -r ONE; do [ -z "$ONE" ] || ARGS+=(--lora "$ONE"); done <<<"$LORA"
fi
exec "$PREFIX/venv/bin/python" "$HERE/../qwen_image_edit.py" "${ARGS[@]}"
