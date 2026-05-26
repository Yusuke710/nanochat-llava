# Vast.ai Codex Training Prompt

Use this prompt when starting a Codex run on a single-GPU Vast.ai box.

```text
You are running training for nanochat-llava on a single GPU Vast.ai box.

Repo/commit:
- Work in the nanochat-llava repo.
- Use commit c6e7cf3 or newer.
- Do not add FP8, activation checkpointing, cached image features, profiling code, or framework-style config.
- Keep the code simple and nanochat-style.

Setup:
1. Run:
   uv sync --extra vision --extra gpu

2. Set persistent cache paths:
   export DATA_ROOT=${DATA_ROOT:-/workspace/nanochat-llava-data}
   export NANOCHAT_BASE_DIR=$DATA_ROOT/nanochat
   export HF_HOME=$DATA_ROOT/hf
   export NANOCHAT_SIGLIP_CACHE_DIR=$HF_HOME/siglip
   mkdir -p "$NANOCHAT_BASE_DIR" "$HF_HOME" "$NANOCHAT_SIGLIP_CACHE_DIR"

3. Verify:
   uv run python -m pytest tests/test_vision.py -q

First smoke:
Run 2 steps and confirm train loss, val loss, and checkpoint save work:

uv run --extra vision --extra gpu python -m scripts.vlm_train \
  --run dummy \
  --hf-repo HuggingFaceM4/the_cauldron \
  --hf-config vqav2 \
  --out-dir "$DATA_ROOT/checkpoints/vlm_smoke" \
  --device-type cuda \
  --num-iterations 2 \
  --device-batch-size 8 \
  --max-batch-tokens 2000 \
  --max-examples 64 \
  --eval-every 1 \
  --eval-tokens 2000 \
  --save-every 2 \
  --model-step 650 \
  --require-fa3-varlen

Then run tiny eval:

uv run --extra vision --extra gpu python -m scripts.vlm_eval \
  --checkpoint-dir "$DATA_ROOT/checkpoints/vlm_smoke" \
  --checkpoint-step 2 \
  --out "$DATA_ROOT/checkpoints/vlm_smoke_eval.json" \
  --benchmarks mmstar \
  --limit 1 \
  --max-scan 20

Real first run:
If the smoke passes, run:

uv run --extra vision --extra gpu python -m scripts.vlm_train \
  --run vlm_vqav2_1gpu \
  --hf-repo HuggingFaceM4/the_cauldron \
  --hf-config vqav2 \
  --out-dir "$DATA_ROOT/checkpoints/vlm" \
  --device-type cuda \
  --num-iterations 1000 \
  --device-batch-size 768 \
  --max-batch-tokens 18000 \
  --max-examples 131072 \
  --num-workers 4 \
  --eval-every 200 \
  --eval-tokens 524288 \
  --val-examples 2048 \
  --save-every 1000 \
  --model-step 650 \
  --require-fa3-varlen

If OOM:
- First reduce --max-batch-tokens to 12000.
- Then reduce --device-batch-size.
- Do not add activation checkpointing or feature caching.

After training:
Run VLM eval:

uv run --extra vision --extra gpu python -m scripts.vlm_eval \
  --checkpoint-dir "$DATA_ROOT/checkpoints/vlm" \
  --checkpoint-step 1000 \
  --out "$DATA_ROOT/checkpoints/vlm_eval.json" \
  --benchmarks mmstar,scienceqa,chartqa,mmmu,textvqa \
  --limit 24 \
  --max-scan 240 \
  --print-samples 4

Report:
- final train loss
- validation loss at each eval point
- tokens/sec and MFU after warmup
- peak memory
- eval JSON path and benchmark scores
- 4 sample generations if available
```
