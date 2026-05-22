# nanochat-llava v0 run notes

Current source of truth: this top-level snapshot plus `RUNBOOK_GPU.md`.
Older cache/precompute, mem100, fixture, preflight, and streamed-offset paths
were intentionally removed to keep the implementation minimal.

## Current live code snapshot

- `nanochat/vision.py`: `<image>` marker handling, frozen SigLIP base patch-16/512, nanoVLM-style 8x8 pixel-shuffle pooling to 64 visual tokens, linear projector, single-image visual-token insertion, target masking, generation helper, VLM checkpoint helpers, and HF nanochat-d32 linking.
- `nanochat/gpt.py`: thin optional `input_embeds` / `value_token_ids` hook in `GPT.forward`; ordinary text-only `model(idx, targets)` behavior is preserved.
- `nanochat/checkpoint_manager.py`: compatibility patching for old `karpathy/nanochat-d32` checkpoint keys missing from the current GPT module.
- `scripts/vlm_train.py`: two-stage VLM trainer. Stage 1 freezes nanochat and SigLIP, trains only the projector. Stage 2 freezes SigLIP, trains projector plus nanochat. Stage 2 defaults to FineVision `LLaVA_Instruct_150K`, using the nanoVLM-style `images` + `texts` schema with embedded image bytes. Legacy HF JSON rows still stream when `--hf-file` is passed.
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


# nanochat-llava GPU Probe Notes

## 2026-05-22 A100-80GB cheap VLM probe

Repository: `Yusuke710/nanochat-llava` at commit `819bb8a`.

Machine/setup:

- GPU: `NVIDIA A100-SXM4-80GB`, visible memory `79.25 GB`.
- Environment created with `uv sync --extra vision --extra gpu --group dev`.
- Tests passed:
  - `tests/test_vision.py tests/test_vlm_smoke.py`: `17 passed`.
  - CUDA VLM smoke: `1 passed`.
- Data/cache root: `/data/nanochat-llava`.

### Data path notes

Stage 1 initially used streamed metadata from `liuhaotian/LLaVA-Pretrain/blip_laion_cc_sbu_558k_meta.json` and third-party source image URLs. That path was too unreliable for training: many records hit DNS failures, HTTP `403/404/406/410`, disconnects, and corrupt partial images. The run was stopped after step 1.

The successful Stage 1 runs used real LLaVA pretrain data via:

- Metadata: `liuhaotian/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json`.
- Images: `liuhaotian/LLaVA-Pretrain/images.zip`, read directly from the HF cache without extraction.
- `--skip-bad-images` was enabled to bypass occasional corrupt records.

Stage 2 used real LLaVA-Instruct data:

- Metadata: `liuhaotian/LLaVA-Instruct-150K/llava_instruct_150k.json`.
- Images: COCO train2017, downloaded on demand with `http://images.cocodataset.org/train2017/{basename}`.

### Batch sizing

Stage 1 sweep on A100-80GB:

- `--device-batch-size 240` fit in a one-step probe, but `256` OOMed.
- A requested `--device-batch-size 200` full run without token cap OOMed after step 20 on a larger shape.
- Final stable Stage 1 used nominal `--device-batch-size 200` with `--max-batch-tokens 16000`.

Stage 2 sweep:

- With runbook `--max-batch-tokens 12000`, larger device batch sizes are capped by token count and use about `61 GiB`.
- `--device-batch-size 64 --max-batch-tokens 16000` fit in a one-step probe but OOMed during real training after step 1.
- Final stable Stage 2 used `--device-batch-size 64 --max-batch-tokens 12000`.

### Stage 1 projector training

Final command shape:

```bash
python -m scripts.vlm_train \
  --stage 1 \
  --hf-repo liuhaotian/LLaVA-Pretrain \
  --hf-file blip_laion_cc_sbu_558k.json \
  --hf-image-zip images.zip \
  --image-root /data/nanochat-llava/datasets/llava/pretrain_images \
  --out-dir /data/nanochat-llava/checkpoints/stage1_pixshuffle_250_bs200 \
  --device-type cuda \
  --num-iterations 250 \
  --device-batch-size 200 \
  --max-batch-tokens 16000 \
  --max-examples 16000 \
  --max-seq-len 2048 \
  --save-every 250 \
  --model-step 650 \
  --skip-bad-images
```

Training curve:

- Step 1: loss `5.968676`, controls pass.
- Step 20: loss `5.205468`, controls pass.
- Step 80: loss `4.407375`, controls pass.
- Step 150: loss `4.048101`, controls pass.
- Step 190: loss `3.995752`, controls pass.
- Step 230: loss `3.658667`, controls pass.
- Step 250: loss `3.641400`, controls pass.

Final Stage 1 signal:

- Final controls: aligned/shuffled/no-image `2.1813 / 5.4883 / 5.8809`, pass.
- Peak memory: `56860.50 MiB`.
- Total training time: `14.24m`.
- Checkpoint: `/data/nanochat-llava/checkpoints/stage1_pixshuffle_250_bs200/model_000250.pt`.
- Metadata: `/data/nanochat-llava/checkpoints/stage1_pixshuffle_250_bs200/meta_000250.json`.

Stage 1 eval output:

- JSON: `/data/nanochat-llava/bench/stage1_pixshuffle_250_bs200.json`.
- `mmstar`: `0.5000`, zero-image `0.3750`, control pass.
- `scienceqa`: `0.3750`, zero-image `0.5000`, control fail.
- `chartqa`: `0.0625`, zero-image `0.0000`, control pass.
- `mmmu`: `0.2500`, zero-image `0.3750`, control fail.
- `textvqa`: `0.1875`, zero-image `0.0625`, control pass.

Stage 1 qualitative check: not collapsed. It produced non-empty, task-shaped answers, but was still mostly weak multiple-choice/text priors. Some image-conditioned changes were visible, especially in `mmstar`, `chartqa`, and `textvqa`, but grounding was not reliable.

Example Stage 1 samples:

- `chartqa`: predicted `The color of the graph is red`; answer was `Blue`.
- `textvqa`: predicted `This is a dark beer`; accepted answers included `ale`.
- `mmmu`: predicted `C` for one sample where the accepted answer was `C`.

### Stage 2 instruction probe

The first Stage 2 attempt used `--device-batch-size 64 --max-batch-tokens 16000`. It initialized correctly and step 1 was healthy, but OOMed on a larger later batch shape. The successful run used the same nominal batch size with `--max-batch-tokens 12000`.

Final command shape:

```bash
python -m scripts.vlm_train \
  --stage 2 \
  --hf-repo liuhaotian/LLaVA-Instruct-150K \
  --hf-file llava_instruct_150k.json \
  --image-root /data/nanochat-llava/datasets/llava/coco/train2017 \
  --image-url-template 'http://images.cocodataset.org/train2017/{basename}' \
  --init-vlm-checkpoint-dir /data/nanochat-llava/checkpoints/stage1_pixshuffle_250_bs200 \
  --init-vlm-checkpoint-step 250 \
  --out-dir /data/nanochat-llava/checkpoints/stage2_llava_probe_bs64_12k \
  --device-type cuda \
  --num-iterations 100 \
  --device-batch-size 64 \
  --max-batch-tokens 12000 \
  --max-examples 4096 \
  --max-seq-len 2048 \
  --save-every 100 \
  --model-step 650 \
  --profile-timing \
  --skip-bad-images
```

Training curve:

- Step 1: loss `1.796549`, controls pass.
- Step 10: loss `1.675038`, controls pass.
- Step 20: loss `1.606355`, controls pass.
- Step 50: loss `1.544179`, controls pass.
- Step 80: loss `1.484415`, controls pass.
- Step 100: loss `1.461913`, controls pass.

Raw timing excerpt from the Stage 2 run:

```text
step 00040/00100 | loss 1.554684 | samples/sec 0.94 | tokens/sec 334 | bf16_mfu 1.29 | lrm 1.000 | timing data/image+siglip/batch/fwdbwd/optim 0.000/17.084/0.407/4.494/0.424s | controls aligned/s [...]
step 00050/00100 | loss 1.544179 | samples/sec 0.90 | tokens/sec 335 | bf16_mfu 1.30 | lrm 1.000 | timing data/image+siglip/batch/fwdbwd/optim 0.000/15.062/0.371/4.121/0.423s | controls aligned/s [...]
step 00060/00100 | loss 1.513127 | samples/sec 0.87 | tokens/sec 404 | bf16_mfu 1.56 | lrm 0.808 | timing data/image+siglip/batch/fwdbwd/optim 0.000/12.600/0.652/4.653/0.424s | controls aligned/s [...]
```

The old `image+siglip` bucket included image download/open/decode, CPU processor
work, host-to-device transfer, SigLIP forward, and feature pooling. The trainer
now splits this bucket when `--profile-timing` is enabled.

Follow-up 3-step Modal profile with the split timer:

Cold COCO cache, using a fresh image root:

```text
step 00001/00003 | loss 1.688807 | samples/sec 0.29 | tokens/sec 99 | bf16_mfu 0.38 | lrm 1.000 | timing data/image_total/open/download/processor/h2d/siglip/pool/batch/fwdbwd/optim 0.000/13.632/0.123/12.132/0.344/0.043/0.990/0.001/0.601/4.638/53.159s | controls aligned/shuffled/no_image 1.5782/1.6327/1.6678 pass=True
step 00003/00003 | loss 1.668643 | samples/sec 1.18 | tokens/sec 400 | bf16_mfu 1.55 | lrm 0.000 | timing data/image_total/open/download/processor/h2d/siglip/pool/batch/fwdbwd/optim 0.000/11.699/0.105/11.197/0.333/0.020/0.043/0.000/0.535/4.089/0.455s | controls aligned/shuffled/no_image 1.6046/1.6796/1.7804 pass=True
```

Warm COCO cache, rerunning against the same image root:

```text
step 00001/00003 | loss 1.688807 | samples/sec 0.30 | tokens/sec 102 | bf16_mfu 0.40 | lrm 1.000 | timing data/image_total/open/download/processor/h2d/siglip/pool/batch/fwdbwd/optim 0.000/4.265/2.823/0.000/0.313/0.028/1.100/0.000/0.484/5.071/60.169s | controls aligned/shuffled/no_image 1.5808/1.6265/1.6652 pass=True
step 00003/00003 | loss 1.669126 | samples/sec 2.99 | tokens/sec 1012 | bf16_mfu 3.92 | lrm 0.000 | timing data/image_total/open/download/processor/h2d/siglip/pool/batch/fwdbwd/optim 0.000/1.499/1.152/0.000/0.304/0.007/0.036/0.000/0.426/4.263/0.490s | controls aligned/shuffled/no_image 1.6084/1.6772/1.7821 pass=True
```

Interpretation: the cold-cache bottleneck is image download, not SigLIP. On the
steady warm-cache step, SigLIP forward was only `0.036s` versus LLM fwd/bwd
`4.263s`; image open/decode from the Modal volume was still `1.152s`.

After switching Stage 2 to FineVision `LLaVA_Instruct_150K` embedded image bytes
and removing the COCO URL path, a 3-step Modal profile showed:

```text
Loaded 128 records in 6.36s
Rendered 128 usable examples in 0.04s
step 00001/00003 | loss 1.743721 | samples/sec 0.37 | tokens/sec 127 | bf16_mfu 0.49 | lrm 1.000 | timing data/image_total/open/processor/h2d/siglip/pool/batch/fwdbwd/optim 0.000/1.545/0.192/0.274/0.025/1.054/0.000/0.474/4.863/49.754s | controls aligned/shuffled/no_image 1.4990/1.5440/1.6462 pass=True
step 00003/00003 | loss 1.679734 | samples/sec 2.81 | tokens/sec 1199 | bf16_mfu 4.64 | lrm 0.000 | timing data/image_total/open/processor/h2d/siglip/pool/batch/fwdbwd/optim 0.000/0.362/0.131/0.193/0.005/0.032/0.000/0.466/4.720/0.493s | controls aligned/shuffled/no_image 1.4006/1.4298/1.4873 pass=True
Peak memory usage: 63027.15MiB
Total training time: 1.69m
```

On the steady FineVision step, image handling was `0.362s` versus LLM fwd/bwd
`4.720s`, so data/image handling was no longer the step bottleneck. The remaining
data cost is setup-time HF shard streaming, measured here as `6.36s` for 128
records.

Final Stage 2 signal:

- Final controls: aligned/shuffled/no-image `1.4336 / 1.4754 / 1.5098`, pass.
- Peak memory: `64128.95 MiB`.
- Total training time: `35.71m`.
- Checkpoint: `/data/nanochat-llava/checkpoints/stage2_llava_probe_bs64_12k/model_000100.pt`.
- Metadata: `/data/nanochat-llava/checkpoints/stage2_llava_probe_bs64_12k/meta_000100.json`.

Stage 2 eval output:

- JSON: `/data/nanochat-llava/bench/stage2_llava_probe_bs64_12k.json`.
- `mmstar`: `0.4375`, zero-image `0.3750`, control pass.
- `scienceqa`: `0.5625`, zero-image `0.3750`, control pass.
- `chartqa`: `0.0000`, zero-image `0.0000`, control fail.
- `mmmu`: `0.3125`, zero-image `0.4375`, control fail.
- `textvqa`: `0.1875`, zero-image `0.0625`, control pass.

Stage 1 to Stage 2 eval deltas:

| Benchmark | Stage 1 | Stage 2 | Delta | Stage 2 zero-image | Stage 2 control |
| --- | ---: | ---: | ---: | ---: | --- |
| mmstar | 0.5000 | 0.4375 | -0.0625 | 0.3750 | pass |
| scienceqa | 0.3750 | 0.5625 | +0.1875 | 0.3750 | pass |
| chartqa | 0.0625 | 0.0000 | -0.0625 | 0.0000 | fail |
| mmmu | 0.2500 | 0.3125 | +0.0625 | 0.4375 | fail |
| textvqa | 0.1875 | 0.1875 | +0.0000 | 0.0625 | pass |

Qualitative Stage 2 samples:

- `scienceqa`: predicted `C.` for the insulin/nutrients question; accepted answer was `C`, and the zero-image answer was also `C`.
- `scienceqa`: predicted `B. climate`; accepted answer was `B`, while zero-image predicted `A. weather`.
- `chartqa`: predicted `The color of the graph with 56 as the highest value is red`; accepted answer was `Blue`.
- `textvqa`: predicted `This is a beer`; accepted answers included `ale`.

Interpretation:

- The model did not collapse. Both stages produced non-empty, task-shaped outputs.
- Stage 1 established a strong projector-alignment signal: the final aligned control loss was far below shuffled/no-image controls.
- Stage 2 learned during the 100-step probe: loss decreased from `1.80` to `1.46`, controls passed throughout, and ScienceQA improved clearly over Stage 1 and zero-image on this small eval slice.
- ChartQA did not improve. Its generations became more fluent but remained visually/numerically wrong, and the control check failed.
- The benchmark limit was only 16 examples per task, so these numbers are directional rather than statistically stable.
