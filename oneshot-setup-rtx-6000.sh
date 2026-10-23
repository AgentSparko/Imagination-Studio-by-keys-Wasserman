#!/bin/bash
# Imagination Studio on an NVIDIA RTX Pro 6000 Blackwell.
# This chooses the card, then runs oneshot-setup-cuda.sh, the same install
# oneshot-setup-spark.sh runs. Neither entry contains the install.
#   bash oneshot-setup-rtx-6000.sh --all                 # one compute card: that card
#   STUDIO_GPU=1 bash oneshot-setup-rtx-6000.sh --all    # that nvidia-smi index
# With two or more compute GPUs, STUDIO_GPU is required. Indexes are this machine's.
# The same variable, set again, chooses the card when the app starts (scripts/app.sh).
# A second card is another process after this install, with its own STUDIO_GPU, PORT,
# SPARK_COMFY_PORT, STUDIO_HOME and STUDIO_GPU_LOCK. One install is shared.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
. "$HERE/spark/gpu.sh"
studio_pin_gpu
# FLUX3-K and sound are gated on Hugging Face: oneshot-setup-cuda.sh asks before anything is downloaded and leaves out
# a part whose licence this account has not accepted, with what to do (scripts/install-checklist.sh).
# The CUDA installer reads nvidia-smi's first row. This directory's nvidia-smi
# answers those queries for STUDIO_GPU.
export PATH="$HERE/spark/smi-pin:$PATH"
# the checklist names this entry in the command it prints
STUDIO_SETUP_ENTRY=oneshot-setup-rtx-6000.sh exec bash "$HERE/oneshot-setup-cuda.sh" "$@"
