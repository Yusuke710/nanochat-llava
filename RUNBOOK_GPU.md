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

## Local Smoke

```bash
uv run python -m pytest tests/test_vision.py tests/test_vlm_smoke.py -q
```

## Train

```bash
uv run --extra vision --extra gpu python -m scripts.vlm_train \
  --run dummy \
  --hf-repo HuggingFaceM4/FineVisionMax \
  --out-dir "$DATA_ROOT/checkpoints/vlm" \
  --device-type cuda \
  --num-iterations 1000 \
  --device-batch-size 128 \
  --max-batch-tokens 12000 \
  --model-step 650 \
  --save-every 1000 \
  --require-fa3-varlen
```

`--device-batch-size` is the number of candidate image-text examples. The actual
LLM batch is one compact packed row whose examples are separated by FA3 varlen
attention boundaries. The default HF source is the precombined FineVisionMax
stream, so the trainer does not apply an extra local streaming shuffle.

## Modal

```bash
NANOCHAT_MODAL_GPU=H100 uv run --extra vision modal run modal_vlm.py::train \
  --num-iterations 6 \
  --batch-size 768 \
  --max-batch-tokens 18000 \
  --eval-every -1 \
  --no-save \
  --log-every 1
```

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
