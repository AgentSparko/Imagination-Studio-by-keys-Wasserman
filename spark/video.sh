#!/bin/bash
# One clip with FastH3 on a DGX Spark (CUDA). scripts/video.sh hands over to this on Linux; same interface:
#   bash scripts/video.sh "a prompt" out.mp4
#   FIRST_FRAME=photo.jpg bash scripts/video.sh "what happens next" out.mp4
#   WIDTH=1344 HEIGHT=768 STEPS=8 FRAMES=124 SEED=3 bash scripts/video.sh "a prompt" out.mp4
#   REF_IMAGES="face.png" bash scripts/video.sh "Use <Picture 1> as the person. She walks into a cafe." out.mp4
# REF_IMAGES (one to nine pictures, separated by spaces; one per line when a path has spaces) renders with MiniMax H3
# Ref2VA (`oneshot-setup-spark.sh --ref2va`), which holds a face or a subject through the clip: 20 steps by default,
# REF_TURBO=1 for LightX2V's 4-pass Ref2VA adapter (107 s against 321 s for 5 s at 864x480). REF_SIZE=match|max.
# Environment: PREFIX, WIDTH, HEIGHT (multiples of 32, default 864x480), FRAMES (17n+5), SEED, STEPS (4, 8 or 20
# passes), FIRST_FRAME, PROMPT_FILE, CROP=WxH, SPARK_WEIGHTS (bf16, the default, or int8: no faster on a Spark, 22 GB
# less memory), SPARK_SPARSITY (0 is dense attention; unset picks by size), SPARK_ENGINE (tensorfold: the transformer's
# blocks on TensorFold's CUDA engine in int8, the default once it is built; comfy: ComfyUI's own blocks in bf16).
# TEXT_ENCODER=uncensored renders with linjian257's community build of the text encoder (`oneshot-setup-spark.sh
# --uncensored`; personal, non-commercial use under its own licence); unset or regular is MiniMax's own.
# Not on the Spark: MiniMax H3 Turbo (ENGINE=h3, QUALITY) and the 2x decoder (X2, TWOK, QHD).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${PREFIX:-$HOME/.local/opt/tensorfold-studio}"
[ "${ENGINE:-fasth3}" = fasth3 ] || { echo "the Spark build renders with FastH3 only (ENGINE=fasth3)" >&2; exit 2; }
[ "${X2:-0}" != 1 ] || { echo "the 2x decoder is not part of the Spark build: use WIDTH and HEIGHT up to 1344x768" >&2; exit 2; }
[ -x "$PREFIX/venv/bin/python" ] || { echo "run oneshot-setup-spark.sh first: no environment at $PREFIX" >&2; exit 1; }
PROMPT="${1:-}"; OUT="${2:-outputs/h3.mp4}"
ARGS=(video -o "$OUT" --width "${WIDTH:-864}" --height "${HEIGHT:-480}" --frames "${FRAMES:-124}"
      --seed "${SEED:-0}")
REFS=()
if [ -n "${REF_IMAGES:-}" ]; then
  case "$REF_IMAGES" in *$'\n'*) while IFS= read -r r; do [ -z "$r" ] || REFS+=("$r"); done <<<"$REF_IMAGES";;
    *) read -r -a REFS <<<"$REF_IMAGES";; esac
fi
for r in ${REFS[@]+"${REFS[@]}"}; do [ -f "$r" ] || { echo "no reference picture at $r" >&2; exit 2; }; ARGS+=(--ref "$r"); done
if [ ${#REFS[@]} -gt 0 ]; then
  [ -z "${STEPS:-}" ] || ARGS+=(--steps "$STEPS")
  if [ "${REF_TURBO:-0}" = 1 ]; then ARGS+=(--ref-mode turbo); else ARGS+=(--ref-mode full); fi
  [ -z "${REF_SIZE:-}" ] || ARGS+=(--ref-size "$REF_SIZE")
  KIND="MiniMax H3 Ref2VA, ${#REFS[@]} reference picture(s)"
else
  ARGS+=(--steps "${STEPS:-8}"); KIND="FastH3 ${STEPS:-8} passes"
fi
[ -z "${FIRST_FRAME:-}" ] || ARGS+=(--first-frame "$FIRST_FRAME")
[ -z "${TEXT_ENCODER:-}" ] || ARGS+=(--text-encoder "$TEXT_ENCODER")
[ -z "${CROP:-}" ] || ARGS+=(--crop "$CROP")
[ -z "${SPARK_SPARSITY:-}" ] || ARGS+=(--sparsity "$SPARK_SPARSITY")
[ -z "${SPARK_ENGINE:-}" ] || ARGS+=(--engine "$SPARK_ENGINE")
# style LoRAs from the hub, one PATH:STRENGTH per line (FastH3 clips; ComfyUI merges them into the bf16 weights)
if [ -n "${LORA:-}" ]; then
  [ ${#REFS[@]} = 0 ] || { echo "a LoRA from the hub is for FastH3 clips, not for reference to video" >&2; exit 2; }
  while IFS= read -r ONE; do [ -z "$ONE" ] || ARGS+=(--lora "$ONE"); done <<<"$LORA"
fi
if [ -n "${PROMPT_FILE:-}" ]; then ARGS+=(--prompt-file "$PROMPT_FILE"); else ARGS+=(--prompt "$PROMPT"); fi
START=$(date +%s)
PREFIX="$PREFIX" "$PREFIX/venv/bin/python" "$HERE/spark_generate.py" "${ARGS[@]}"
echo "[studio] $KIND, ${WIDTH:-864}x${HEIGHT:-480}, total $(( $(date +%s) - START )) s -> $OUT"
