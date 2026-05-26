#!/bin/bash

# nanochat-llava training on VastAI.
# This downloads/caches the HF assets, then trains on The Cauldron.
#
# Run:
# bash runs/vlm_vastai.sh
#
# Fast MFU probe:
# VLM_ITERS=6 VLM_BATCH=256 VLM_STREAM_BUFFER=256 VLM_BATCH_BUFFER=512 VLM_LOG_EVERY=1 VLM_NO_SAVE=1 bash runs/vlm_vastai.sh

# Default intermediate artifacts directory. VastAI usually has /workspace.
export OMP_NUM_THREADS=1
if [ -d /workspace ]; then
    export DATA_ROOT="${DATA_ROOT:-/workspace/nanochat-llava}"
else
    export DATA_ROOT="${DATA_ROOT:-$HOME/.cache/nanochat-llava}"
fi
export NANOCHAT_BASE_DIR="$DATA_ROOT/nanochat"
export HF_HOME="$DATA_ROOT/hf"
export NANOCHAT_SIGLIP_CACHE_DIR="$HF_HOME/siglip"
mkdir -p "$NANOCHAT_BASE_DIR" "$HF_HOME" "$NANOCHAT_SIGLIP_CACHE_DIR" "$DATA_ROOT/checkpoints"

# -----------------------------------------------------------------------------
# Python venv setup with uv

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra vision --extra gpu
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup

WANDB_RUN="${WANDB_RUN:-dummy}"

# -----------------------------------------------------------------------------
# Download/cache model assets.
# The Cauldron itself is streamed by the trainer; HF will cache dataset shards as needed.

export VLM_DATA_REPO="${VLM_DATA_REPO:-HuggingFaceM4/the_cauldron}"
export VLM_DATA_CONFIG="${VLM_DATA_CONFIG:-all}"

python - <<'PY'
import os
from huggingface_hub import snapshot_download

print("Caching nanochat checkpoint...")
snapshot_download("karpathy/nanochat-d32")

print("Caching SigLIP...")
snapshot_download("google/siglip-base-patch16-512", cache_dir=os.environ["NANOCHAT_SIGLIP_CACHE_DIR"])
PY

# -----------------------------------------------------------------------------
# Train projector + nanochat on The Cauldron

VLM_ITERS="${VLM_ITERS:-1000}"
VLM_BATCH="${VLM_BATCH:-24}"
VLM_GRAD_ACCUM="${VLM_GRAD_ACCUM:-1}"
VLM_MAX_BATCH_TOKENS="${VLM_MAX_BATCH_TOKENS:-12000}"
VLM_MAX_EXAMPLES="${VLM_MAX_EXAMPLES:--1}"
VLM_STREAM_BUFFER="${VLM_STREAM_BUFFER:-4096}"
VLM_BATCH_BUFFER="${VLM_BATCH_BUFFER:-0}"
VLM_PREFETCH_BATCHES="${VLM_PREFETCH_BATCHES:-2}"
VLM_LOG_EVERY="${VLM_LOG_EVERY:-10}"
VLM_MFU_WARMUP_STEPS="${VLM_MFU_WARMUP_STEPS:-2}"
VLM_NO_SAVE="${VLM_NO_SAVE:-0}"
VLM_PACK_EXAMPLES="${VLM_PACK_EXAMPLES:-1}"
VLM_PACK_MAX_SEQ_LEN="${VLM_PACK_MAX_SEQ_LEN:-0}"
VLM_SKIP_BAD_IMAGES="${VLM_SKIP_BAD_IMAGES:-1}"
VLM_OUT_DIR="${VLM_OUT_DIR:-$DATA_ROOT/checkpoints/vlm_cauldron}"

SKIP_BAD_IMAGES_FLAG="--skip-bad-images"
if [ "$VLM_SKIP_BAD_IMAGES" = "0" ]; then
    SKIP_BAD_IMAGES_FLAG="--no-skip-bad-images"
fi

SAVE_FLAGS=(--save-every "$VLM_ITERS")
if [ "$VLM_NO_SAVE" = "1" ]; then
    SAVE_FLAGS=(--no-save)
fi

python -m scripts.vlm_train \
    --run "$WANDB_RUN" \
    --hf-repo "$VLM_DATA_REPO" \
    --hf-config "$VLM_DATA_CONFIG" \
    --out-dir "$VLM_OUT_DIR" \
    --device-type cuda \
    --num-iterations "$VLM_ITERS" \
    --device-batch-size "$VLM_BATCH" \
    --grad-accum-steps "$VLM_GRAD_ACCUM" \
    --max-batch-tokens "$VLM_MAX_BATCH_TOKENS" \
    --max-examples "$VLM_MAX_EXAMPLES" \
    --stream-buffer-size "$VLM_STREAM_BUFFER" \
    --batch-buffer-size "$VLM_BATCH_BUFFER" \
    --prefetch-batches "$VLM_PREFETCH_BATCHES" \
    --pack-examples "$VLM_PACK_EXAMPLES" \
    --pack-max-seq-len "$VLM_PACK_MAX_SEQ_LEN" \
    --log-every "$VLM_LOG_EVERY" \
    --mfu-warmup-steps "$VLM_MFU_WARMUP_STEPS" \
    "$SKIP_BAD_IMAGES_FLAG" \
    "${SAVE_FLAGS[@]}" \
    --max-seq-len 2048 \
    --model-step 650 \
    --profile-timing
