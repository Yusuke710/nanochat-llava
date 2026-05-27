# nanochat-llava GPU Runbook

This V0 path starts from `karpathy/nanochat-d32`, freezes SigLIP, and trains a
linear projector plus nanochat on visual-instruction data.

## Setup

```bash
uv sync --extra vision --extra gpu

export DATA_ROOT=${DATA_ROOT:-$HOME/.cache/nanochat-llava}
export NANOCHAT_BASE_DIR=$DATA_ROOT/nanochat
export HF_HOME=$DATA_ROOT/hf
export NANOCHAT_SIGLIP_CACHE_DIR=$HF_HOME/siglip
mkdir -p "$NANOCHAT_BASE_DIR" "$HF_HOME" "$NANOCHAT_SIGLIP_CACHE_DIR"
```

The trainer will link `karpathy/nanochat-d32` into nanochat's checkpoint layout
and cache `google/siglip-base-patch16-512` on first use.

Set the packing knobs for the GPU you are on:

```bash
uv run --extra vision --extra gpu python -c "from nanochat.flash_attention import has_fa3_varlen; print('FA3_VARLEN', has_fa3_varlen())"

# If H100/H200 reports FA3 varlen, require the fast varlen kernel.
export FA3_ARG=--require-fa3-varlen
# Otherwise use: export FA3_ARG=
```

## Local Smoke

```bash
uv run python -m pytest tests/test_vision.py tests/test_vlm_smoke.py -q
```

## Train

```bash
uv run --extra vision --extra gpu python -m scripts.vlm_train \
  --run vlm_finevisionmax_1gpu \
  --hf-repo HuggingFaceM4/FineVisionMax \
  --out-dir "$DATA_ROOT/checkpoints/vlm" \
  --device-type cuda \
  --num-iterations 1000 \
  --device-batch-size 32 \
  --max-seq-len 512 \
  --max-batch-images 96 \
  --num-workers 4 \
  --model-step 650 \
  --save-every 1000 \
  $FA3_ARG
```

`--device-batch-size 32 --max-seq-len 512` builds the nanochat-style fixed
training shape: 32 packed rows of 512 expanded decoder tokens each. The trainer
best-fit packs text-only, single-image, and multi-image examples from an
internal 24x candidate buffer, caps real images with `--max-batch-images`, and
pads row tails with ignored dummy segments, so examples remain separated by
varlen attention boundaries.
The default HF source is the precombined FineVisionMax stream, so the trainer
does not apply an extra local streaming shuffle.

## Modal

```bash
NANOCHAT_MODAL_GPU=H100 uv run --extra vision modal run modal_vlm.py::train \
  --num-iterations 6 \
  --batch-size 32 \
  --max-seq-len 512 \
  --max-batch-images 96 \
  --eval-every -1 \
  --no-save \
  --log-every 1
```

Validated on Modal H100 on 2026-05-27: this same mixed-modality fixed
`32 x 512` shape completed 6 steps with warm-step BF16 MFU `28.31-30.73%`
and peak memory `72866.35MiB`.

## Eval

```bash
uv run --extra vision --extra gpu python -m scripts.vlm_eval \
  --checkpoint-dir "$DATA_ROOT/checkpoints/vlm" \
  --checkpoint-step 1000 \
  --out "$DATA_ROOT/checkpoints/vlm_eval.json" \
  --benchmarks mmstar,scienceqa,chartqa,mmmu,textvqa \
  --limit 24 \
  --max-scan 240
```

## Web Chat

```bash
uv run --extra vision --extra gpu python -m scripts.chat_web \
  --source sft \
  --model-tag d32 \
  --step 650 \
  --device-type cuda \
  --vlm-checkpoint-dir "$DATA_ROOT/checkpoints/vlm" \
  --vlm-checkpoint-step 1000
```

Then open the printed local URL. The image button sends one browser-loaded image
with the chat request; without the VLM flags, text chat continues to work as
before.
