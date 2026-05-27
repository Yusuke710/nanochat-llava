# Vast.ai Codex Training Prompt

Use this prompt when starting a Codex run on a single-GPU Vast.ai box.

```text
You are running training for nanochat-llava on a single GPU Vast.ai box.

Repo/commit:
- Work in the nanochat-llava repo.
- Use the current repo checkout; do not assume older pre-FineVision args.
- Do not add FP8, activation checkpointing, cached image features, profiling code, or framework-style config.
- Keep the code simple and nanochat-style.
- Use the default `HuggingFaceM4/FineVisionMax` stream. Do not add local streaming shuffle, data caps, feature caches, or resume/offset machinery unless asked.

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
   uv run python -m pytest tests/test_vision.py tests/test_vlm_smoke.py -q

4. Check whether this box has FA3 varlen. H100/H200 should; A100 usually will not:
   uv run --extra vision --extra gpu python -c "from nanochat.flash_attention import has_fa3_varlen; print('FA3_VARLEN', has_fa3_varlen())"

   If it prints `True`, use:
   export FA3_ARG=--require-fa3-varlen

   If it prints `False`, use:
   export FA3_ARG=

First smoke:
Run 2 steps and confirm train loss, val loss, and checkpoint save work:

uv run --extra vision --extra gpu python -m scripts.vlm_train \
  --run dummy \
  --hf-repo HuggingFaceM4/FineVisionMax \
  --out-dir "$DATA_ROOT/checkpoints/vlm_smoke" \
  --device-type cuda \
  --num-iterations 2 \
  --device-batch-size 8 \
  --max-seq-len 512 \
  --max-batch-images 16 \
  --eval-every 1 \
  --val-examples 8 \
  --save-every 2 \
  --model-step 650 \
  $FA3_ARG

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
  --run vlm_finevisionmax_1gpu \
  --hf-repo HuggingFaceM4/FineVisionMax \
  --out-dir "$DATA_ROOT/checkpoints/vlm" \
  --device-type cuda \
  --num-iterations 1000 \
  --device-batch-size 32 \
  --max-seq-len 512 \
  --max-batch-images 96 \
  --num-workers 4 \
  --eval-every 200 \
  --val-examples 2048 \
  --save-every 1000 \
  --model-step 650 \
  $FA3_ARG

Notes:
- `--device-batch-size 32 --max-seq-len 512` is the actual fixed tensor shape: 32 packed rows by 512 expanded decoder tokens.
- The trainer best-fit packs text-only, single-image, and multi-image examples from an internal 24x candidate buffer, caps real images with `--max-batch-images`, and pads row tails with ignored dummy segments. Varlen boundaries keep examples from attending across packed boundaries.
- Modal H100 double-check on 2026-05-27: mixed-modality fixed `32 x 512` packing completed 6 steps with warm steps at `28.31-30.73%` BF16 MFU and `72866.35MiB` peak memory.

If OOM:
- First reduce to `--device-batch-size 24 --max-seq-len 512`.
- Then reduce `--max-seq-len` to 384 if needed.
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
