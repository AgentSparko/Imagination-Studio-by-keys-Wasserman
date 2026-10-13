#!/bin/bash
# Edit a picture with Qwen-Image-2.1 on a DGX Spark (CUDA). scripts/edit.sh hands over to this on Linux; same interface:
#   bash scripts/edit.sh photo.png "make it night" out.png
#   SET_FILE=edits.json bash scripts/edit.sh photo.png        # several edits from one load: [[prompt, output], ...]
# Environment: PREFIX, BASE=1 (the 40-step base model: oneshot-setup-spark.sh --image-edit), STEPS (base model only),
# SEED, RESOLUTION, LORA (one PATH or PATH:STRENGTH a line; merged at run time on the bfloat16 file).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${PREFIX:-$HOME/.local/opt/tensorfold-studio}"
[ -x "$PREFIX/venv/bin/python" ] || { echo "run oneshot-setup-spark.sh first: no environment at $PREFIX" >&2; exit 1; }
IMAGE="${1:-}"; PROMPT="${2:-}"; OUT="${3:-}"
[ -f "$IMAGE" ] || { echo "give the picture to edit: bash scripts/edit.sh photo.png \"what changes\" out.png" >&2; exit 1; }
ARGS=(--image "$IMAGE" --seed "${SEED:-0}" --resolution "${RESOLUTION:-1024}")
[ "${BASE:-0}" != 1 ] || ARGS+=(--base --steps "${STEPS:-40}")
if [ -n "${SET_FILE:-}" ]; then ARGS+=(--set "@$SET_FILE"); else ARGS+=(--prompt "$PROMPT" -o "$OUT"); fi
if [ -n "${LORA:-}" ]; then
  while IFS= read -r ONE; do [ -z "$ONE" ] || ARGS+=(--lora "$ONE"); done <<<"$LORA"
fi
PREFIX="$PREFIX" exec "$PREFIX/venv/bin/python" "$HERE/spark_edit.py" "${ARGS[@]}"
