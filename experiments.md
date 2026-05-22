# nanochat-llava v0 run notes

Current source of truth: this top-level snapshot plus `RUNBOOK_GPU.md`.
Older cache/precompute, mem100, fixture, preflight, and streamed-offset paths
were intentionally removed to keep the implementation minimal.

## Current live code snapshot

- `nanochat/vision.py`: `<image>` marker handling, frozen SigLIP base patch-16/512, nanoVLM-style 8x8 pixel-shuffle pooling to 64 visual tokens, linear projector, single-image visual-token insertion, target masking, generation helper, VLM checkpoint helpers, and HF nanochat-d32 linking.
- `nanochat/gpt.py`: thin optional `input_embeds` / `value_token_ids` hook in `GPT.forward`; ordinary text-only `model(idx, targets)` behavior is preserved.
- `nanochat/checkpoint_manager.py`: compatibility patching for old `karpathy/nanochat-d32` checkpoint keys missing from the current GPT module.
- `scripts/vlm_train.py`: two-stage LLaVA trainer. Stage 1 freezes nanochat and SigLIP, trains only the projector. Stage 2 freezes SigLIP, trains projector plus nanochat. HF JSON rows stream by default; images are loaded locally, from a direct HF zip fallback, or downloaded on demand from record URLs / COCO URL template.
- `scripts/vlm_eval.py`: verifier subset runner for MMStar, ScienceQA, ChartQA, MMMU, and TextVQA with optional zero-image controls and stored sample generations.
- `tests/test_vision.py` and `tests/test_vlm_smoke.py`: focused unit tests plus synthetic image-conditioned overfit/control smoke. The smoke now lives in tests, not scripts.
- `modal_vlm.py`: minimal Modal wrapper with `doctor`, `smoke`, `stage1`, `stage2`, and `eval` only. Default GPU is `A100-80GB`; set `NANOCHAT_MODAL_GPU=H100` to switch.
- `RUNBOOK_GPU.md`: external-GPU runbook, streamed-data behavior, Stage 1/Stage 2/eval commands, and go/no-go criteria.

## Pitfalls to avoid

- Do not re-add `vlm_precompute_siglip.py`, online feature caches, `/vol/features`, preflight scripts, resume/offset machinery, mem100 gates, or benchmark report generators unless there is a new explicit reason. They made the code harder to reason about before proving visual learning.
- Keep inline SigLIP for v0. For streamed LLaVA, images are mostly unique, so a repeated-image cache is not aligned with the data path.
- Do not judge success from aggregate benchmark numbers alone. Inspect stored sample generations and zero-image controls.
- Keep Stage 2 starting from the SFT d32 checkpoint and the Stage 1 projector checkpoint. Old non-pixel-shuffle checkpoints are incompatible with the current `12288` projector input dimension.

## Current commands

Local verification:

```bash
uv run python -m pytest tests/test_vision.py tests/test_vlm_smoke.py -q
uv run --extra vision python -m scripts.vlm_train --help
uv run --extra vision python -m scripts.vlm_eval --help
```

Modal smoke and staged run:

```bash
uv run --extra vision modal run modal_vlm.py::doctor
uv run --extra vision modal run modal_vlm.py::smoke

uv run --extra vision modal run modal_vlm.py::stage1 \
  --out-dir /vol/checkpoints/stage1_pixshuffle_250 \
  --num-iterations 250 \
  --batch-size 32 \
  --max-examples 16000

uv run --extra vision modal run modal_vlm.py::eval \
  --checkpoint-dir /vol/checkpoints/stage1_pixshuffle_250 \
  --checkpoint-step 250 \
  --out /vol/bench/stage1_pixshuffle_250.json \
  --benchmarks mmstar,scienceqa,chartqa,mmmu,textvqa \
  --limit 16 \
  --max-scan 240 \
  --print-samples 3 \
  --control

uv run --extra vision modal run modal_vlm.py::stage2 \
  --init-checkpoint-dir /vol/checkpoints/stage1_pixshuffle_250 \
  --init-checkpoint-step 250 \
  --out-dir /vol/checkpoints/stage2_llava_probe \
  --num-iterations 100 \
  --batch-size 24 \
  --max-batch-tokens 12000 \
  --max-examples 4096 \
  --profile-timing

uv run --extra vision modal run modal_vlm.py::eval \
  --checkpoint-dir /vol/checkpoints/stage2_llava_probe \
  --checkpoint-step 100 \
  --out /vol/bench/stage2_llava_probe.json \
  --benchmarks mmstar,scienceqa,chartqa,mmmu,textvqa \
  --limit 16 \
  --max-scan 240 \
  --print-samples 3 \
  --control
```

## Remaining proof

The local code path is ready for a scaled probe, but model-quality success is
not proven until a real GPU run produces Stage 1 and Stage 2 eval JSONs. Compare
scores, zero-image controls, prediction-change rates, and sample generations
before launching longer training.
