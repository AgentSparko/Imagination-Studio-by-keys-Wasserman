#!/usr/bin/env bash
# Train a LoRA in the Studio (Looks -> Train a LoRA, app/train.py): the trainer and the two files it needs that the
# Studio's renders do not. oneshot-setup-spark.sh --train runs this.
#
#   bash spark/setup-train.sh            # install (about 6 GB of weights and an 8 GB environment)
#   bash spark/setup-train.sh --verify   # check only: no download, no install
#
# The trainer: kohya-ss's musubi-tuner (Apache License 2.0) at f8a1b03 (2026-09-27, version 0.3.6), cloned into
#   $PREFIX/train/musubi-tuner, with its pinned requirements in its own environment $PREFIX/train/venv (Python 3.12,
#   PyTorch for CUDA 13 as the Studio's own environment has it, bitsandbytes for the 8-bit AdamW optimizer; where
#   bitsandbytes does not run on this GPU the plain AdamW is used, which holds a few GB more).
# The files, into $SPARK_MODELS/Train-H3:
#   vae/minimax_h3_video_vae_fp16.safetensors (5.2 GB), Comfy-Org/MiniMax-H3 at e5eb578a: MiniMax H3's video encoder in
#     fp16. musubi-tuner reads this one; the 8-bit file the renders use does not load in it. MiniMax H3 Community
#     License, as the rest of MiniMax H3.
#   adapters/ostris_dd_adapter_v2.safetensors (310 MB), ostris/minimax_h3_training_adapter at a7203c31, file
#     minimax_h3_training_adapter_v2.safetensors: ostris's de-distillation adapter for MiniMax H3's FL2VA model, merged
#     under the LoRA while it trains and not used when it renders. Its page states no licence of its own; it is made
#     from MiniMax H3, whose Community License applies to it.
# It also needs what --h3-base installs (MiniMax H3's FL2VA transformer, bf16, 40 GB), fetched here when it is missing,
# and the H3 text encoder and audio decoder every Spark install has.
# Environment: PREFIX, SPARK_MODELS.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${PREFIX:-$HOME/.local/opt/tensorfold-studio}"
MODELS="${SPARK_MODELS:-$HOME/spark-studio-models}"
TRAIN="$PREFIX/train"
MUSUBI_COMMIT=f8a1b03794a49239a3539015075f5123d6c07d66
COMFY_REPO=Comfy-Org/MiniMax-H3; COMFY_REVISION=e5eb578a89295337b8ff433a035929ce0279e0b6
VAE_FILE=vae/minimax_h3_video_vae_fp16.safetensors
VAE_SHA256=7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522; VAE_BYTES=5207808496
ADAPTER_REPO=ostris/minimax_h3_training_adapter; ADAPTER_REVISION=a7203c31a0016c73d1e8f8df7cc119a880015621
ADAPTER_FILE=minimax_h3_training_adapter_v2.safetensors; ADAPTER_AS=adapters/ostris_dd_adapter_v2.safetensors
ADAPTER_SHA256=5646c07bada64c87478d2b89b7f564bf723b2c8330d608a16c9dcbaad8d3cb58; ADAPTER_BYTES=310168544
DIT=MiniMax-H3-Comfy/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors
VERIFY=0
for arg in "$@"; do
  case "$arg" in
    --verify) VERIFY=1;;
    *) echo "unknown option $arg" >&2; exit 2;;
  esac
done
ok()  { echo "  ✓ $*"; }
step(){ echo; echo "==> $*"; }
die() { echo "  ✗ $*" >&2; exit 1; }
size_of() { stat -c %s "$1" 2>/dev/null || stat -f %z "$1" 2>/dev/null || echo 0; }
[ "$(uname -s)" = Linux ] || die "training a LoRA runs on a DGX Spark (Linux, CUDA)"

step "musubi-tuner at ${MUSUBI_COMMIT:0:7} -> $TRAIN/musubi-tuner"
if [ "$(git -C "$TRAIN/musubi-tuner" rev-parse HEAD 2>/dev/null)" != "$MUSUBI_COMMIT" ]; then
  [ "$VERIFY" = 0 ] || die "musubi-tuner is missing or not at ${MUSUBI_COMMIT:0:7}: bash spark/setup-train.sh"
  mkdir -p "$TRAIN"
  [ -d "$TRAIN/musubi-tuner/.git" ] || git clone -q https://github.com/kohya-ss/musubi-tuner "$TRAIN/musubi-tuner"
  git -C "$TRAIN/musubi-tuner" fetch -q origin "$MUSUBI_COMMIT" 2>/dev/null || git -C "$TRAIN/musubi-tuner" fetch -q origin
  git -C "$TRAIN/musubi-tuner" checkout -q "$MUSUBI_COMMIT"
fi
ok "musubi-tuner $(git -C "$TRAIN/musubi-tuner" log -1 --format='%h, %ad' --date=short) (Apache License 2.0)"

step "Its environment: Python 3.12, PyTorch for CUDA 13, musubi-tuner's pinned requirements -> $TRAIN/venv"
PY="$TRAIN/venv/bin/python"
if [ "$VERIFY" = 0 ]; then
  command -v uv >/dev/null || export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null || die "no uv: run oneshot-setup-spark.sh first (it installs uv and the Studio's environment)"
  [ -x "$PY" ] || uv venv "$TRAIN/venv" --python 3.12 --python-preference only-managed --seed -q
  pipi() { UV_TORCH_BACKEND=cu130 uv pip install -q --python "$PY" "$@"; }  # as the Studio's environment (oneshot-setup-cuda.sh)
  "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null || pipi torch torchvision
  pipi -e "$TRAIN/musubi-tuner"
  pipi bitsandbytes || echo "  ! bitsandbytes did not install: the plain AdamW optimizer is used"
fi
[ -x "$PY" ] && [ -x "$TRAIN/venv/bin/accelerate" ] || die "no training environment at $TRAIN/venv: bash spark/setup-train.sh"
"$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || die "PyTorch in $TRAIN/venv does not see the GPU"
"$PY" -c "import musubi_tuner, accelerate, diffusers, transformers, av" || die "musubi-tuner's requirements are missing from $TRAIN/venv"
ok "torch $("$PY" -c 'import torch; print(torch.__version__, "on", torch.cuda.get_device_name(0))')"
"$PY" -c "import triton" 2>/dev/null && ok "triton (the int8 transformer's fused kernels)" \
  || echo "  ! no triton: the int8 transformer runs a slower fallback with the same memory"
# the 8-bit AdamW of the measured recipe, if bitsandbytes runs a step on this GPU; else the plain AdamW
if "$PY" - <<'PYCODE' 2>/dev/null
import torch, bitsandbytes as bnb
p = torch.nn.Parameter(torch.ones(4096, device="cuda"))
opt = bnb.optim.AdamW8bit([p], lr=1e-3)
p.grad = torch.ones_like(p)
opt.step()
torch.cuda.synchronize()
assert torch.isfinite(p).all()
PYCODE
then OPT=adamw8bit; ok "bitsandbytes runs 8-bit AdamW on this GPU"
else OPT=adamw; echo "  ! bitsandbytes does not run on this GPU: the plain AdamW is used (about 3 GB more held while training)"; fi
if [ "$VERIFY" = 0 ]; then echo "$OPT" > "$TRAIN/optimizer"; fi
ok "optimizer: $(cat "$TRAIN/optimizer" 2>/dev/null || echo "$OPT")"

step "MiniMax H3's video encoder in fp16 and ostris's de-distillation adapter (5.5 GB) -> $MODELS/Train-H3"
have() { [ -s "$MODELS/Train-H3/$1" ] && [ "$(size_of "$MODELS/Train-H3/$1")" = "$2" ]; }
if [ "$VERIFY" = 0 ]; then
  echo "  MiniMax H3's files are under the MiniMax H3 Community License (territory limits): https://huggingface.co/MiniMaxAI/MiniMax-H3"
  echo "  ostris's adapter (https://huggingface.co/$ADAPTER_REPO) states no licence of its own: it is made from MiniMax H3."
  HF="$PREFIX/venv/bin/hf"; [ -x "$HF" ] || HF="$(command -v hf || true)"
  [ -n "$HF" ] || die "no hf command (the Studio's environment has one: run oneshot-setup-spark.sh first)"
  T="$MODELS/Train-H3/.download"; mkdir -p "$T" "$MODELS/Train-H3/vae" "$MODELS/Train-H3/adapters"
  if ! have "$VAE_FILE" "$VAE_BYTES"; then
    "$HF" download "$COMFY_REPO" "$VAE_FILE" --revision "$COMFY_REVISION" --local-dir "$T" >/dev/null
    [ "$(sha256sum "$T/$VAE_FILE" | cut -d' ' -f1)" = "$VAE_SHA256" ] || die "the video encoder did not download whole: run this again"
    mv "$T/$VAE_FILE" "$MODELS/Train-H3/$VAE_FILE"
  fi
  if ! have "$ADAPTER_AS" "$ADAPTER_BYTES"; then
    "$HF" download "$ADAPTER_REPO" "$ADAPTER_FILE" --revision "$ADAPTER_REVISION" --local-dir "$T" >/dev/null
    [ "$(sha256sum "$T/$ADAPTER_FILE" | cut -d' ' -f1)" = "$ADAPTER_SHA256" ] || die "the adapter did not download whole: run this again"
    mv "$T/$ADAPTER_FILE" "$MODELS/Train-H3/$ADAPTER_AS"
  fi
  rm -rf "$T"
  if [ ! -s "$MODELS/$DIT" ]; then
    echo "  MiniMax H3's FL2VA transformer (40 GB, what --h3-base installs) is not here: fetching it"
    "$HF" download "$COMFY_REPO" "${DIT#MiniMax-H3-Comfy/}" --revision "$COMFY_REVISION" --local-dir "$MODELS/MiniMax-H3-Comfy" >/dev/null
  fi
fi
have "$VAE_FILE" "$VAE_BYTES" || die "missing: $MODELS/Train-H3/$VAE_FILE (bash spark/setup-train.sh)"
have "$ADAPTER_AS" "$ADAPTER_BYTES" || die "missing: $MODELS/Train-H3/$ADAPTER_AS (bash spark/setup-train.sh)"
[ -s "$MODELS/$DIT" ] || die "missing: $MODELS/$DIT (bash oneshot-setup-spark.sh --h3-base)"
[ -s "$MODELS/FastH3-Comfy/vae/minimax_h3_audio_vae_fp32.safetensors" ] || die "missing: the H3 audio decoder (bash oneshot-setup-spark.sh)"
[ -s "$MODELS/FastH3-Comfy/text_encoders/qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors" ] \
  || [ -s "$MODELS/FastH3-Comfy/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors" ] || die "missing: the H3 text encoder (bash oneshot-setup-spark.sh)"
ok "video encoder (fp16), adapter v2, FL2VA transformer, text encoder and audio decoder"
echo "  Looks -> Train a LoRA is ready. A run holds about 55 GB and the GPU for hours (about 110 s a step at 928x512):"
echo "  ComfyUI is stopped for it and nothing else renders until it ends."
