# nanochat-llava v0 run notes

Current source of truth: this top-level snapshot plus `RUNBOOK_GPU.md`.
Older cache/precompute, mem100, fixture, preflight, and streamed-offset paths
were intentionally removed to keep the implementation minimal.

## Current live code snapshot

- `nanochat/vision.py`: `<image>` marker handling, frozen SigLIP base patch-16/512, nanoVLM-style 8x8 pixel-shuffle pooling to 64 visual tokens, linear projector, single-image visual-token insertion, target masking, generation helper, VLM checkpoint helpers, and HF nanochat-d32 linking.
- `nanochat/gpt.py`: thin optional `input_embeds` / `value_token_ids` hook in `GPT.forward`; ordinary text-only `model(idx, targets)` behavior is preserved.
- `nanochat/checkpoint_manager.py`: compatibility patching for old `karpathy/nanochat-d32` checkpoint keys missing from the current GPT module.
- `scripts/vlm_train.py`: single VLM trainer. It freezes SigLIP, trains projector plus nanochat, and defaults to all configs from `HuggingFaceM4/the_cauldron` with the nanoVLM-style `images` + `texts` schema. HF Datasets handles streaming and shuffle buffering; repo code only renders usable records lazily and batches them. Optional `--eval-every` runs a small VLM benchmark loop during training.
- `scripts/vlm_eval.py`: verifier subset runner for MMStar, ScienceQA, ChartQA, MMMU, and TextVQA. It exposes `evaluate_vlm(...)` for training-time checks, and the CLI evaluates one checkpoint, stores scores and sample generations, and leaves checkpoint-to-checkpoint comparisons outside the script.
- `tests/test_vision.py` and `tests/test_vlm_smoke.py`: focused unit tests plus synthetic image-conditioned overfit/control smoke. The smoke now lives in tests, not scripts.
- `modal_vlm.py`: minimal Modal wrapper with `doctor`, `smoke`, `train`,
  `mfu_probe`, and `eval` only. Default GPU is `A100-80GB`; set
  `NANOCHAT_MODAL_GPU=H100` to switch.
- `RUNBOOK_GPU.md`: external-GPU runbook, streamed-data behavior, train/eval commands, and go/no-go criteria.

## Pitfalls to avoid

- Do not re-add `vlm_precompute_siglip.py`, online feature caches, `/vol/features`, preflight scripts, resume/offset machinery, mem100 gates, or benchmark report generators unless there is a new explicit reason. They made the code harder to reason about before proving visual learning.
- Keep inline SigLIP for v0. For streamed LLaVA, images are mostly unique, so a repeated-image cache is not aligned with the data path.
- Do not judge success from aggregate benchmark numbers alone. Compare separate checkpoint eval JSONs and inspect stored sample generations.
- Training starts directly from the SFT d32 checkpoint unless an explicit VLM checkpoint is passed. Old non-pixel-shuffle checkpoints are incompatible with the current `12288` projector input dimension.

## Current commands

Local verification:

```bash
uv run python -m pytest tests/test_vision.py tests/test_vlm_smoke.py -q
uv run --extra vision python -m scripts.vlm_train --help
uv run --extra vision python -m scripts.vlm_eval --help
```

Modal smoke and training run:

```bash
uv run --extra vision modal run modal_vlm.py::doctor
uv run --extra vision modal run modal_vlm.py::smoke

uv run --extra vision modal run modal_vlm.py::train \
  --out-dir /vol/checkpoints/vlm_cauldron_probe \
  --num-iterations 100 \
  --batch-size 24 \
  --max-batch-tokens 12000 \
  --max-examples 4096 \
  --stream-buffer-size 4096 \
  --prefetch-batches 2 \
  --skip-bad-images \
  --profile-timing

uv run --extra vision modal run modal_vlm.py::eval \
  --checkpoint-dir /vol/checkpoints/vlm_cauldron_probe \
  --checkpoint-step 100 \
  --out /vol/bench/vlm_cauldron_probe.json \
  --benchmarks mmstar,scienceqa,chartqa,mmmu,textvqa \
  --limit 16 \
  --max-scan 240 \
  --print-samples 3
```

## Remaining proof

The local code path is ready for a scaled probe, but model-quality success is
not proven until a real GPU run produces a training-time eval trail or standalone
eval JSONs. Compare scores and sample generations across checkpoints before
launching longer training.

## 2026-05-25 text-only nanochat MFU baseline references

Intent: use published nanochat text-only performance as the LLM-side reference
for VLM training-system optimization, instead of spending GPU time rerunning
text-only pretraining locally.

Sources checked on 2026-05-25:

- `karpathy/nanochat` discussion #710, "Reproducing nanochat in a homelab",
  started 2026-04-16:
  reports the then-current 8xH100 leaderboard speedrun as 1.65 hours, CORE
  0.263, commit `a825e63`, dated 2026-03-14. The post states H100 nanochat
  runs achieve roughly 50-65% MFU and uses 60% MFU in its throughput comparison
  math.
  URL: https://github.com/karpathy/nanochat/discussions/710
- `karpathy/nanochat` discussion #677, "supa beginna nanochat", started
  2026-03-29: reports the 8xH100 speedrun at depth 24 with FP8/Flash Attention
  as 6,612 pretraining steps, about 970K tokens/sec, 59.4% pretraining MFU, and
  50.3% SFT MFU.
  URL: https://github.com/karpathy/nanochat/discussions/677
- `karpathy/nanochat` discussion #8, "$1000 tier nanochat run", started
  2025-10-13: Karpathy's d32 run note shows early pretraining steps at about
  50.9% MFU on 8xH100.
  URL: https://github.com/karpathy/nanochat/discussions/8
- `karpathy/nanochat` discussion #1, initial speedrun announcement, started
  2025-10-13: older d20 8xH100 example logs show steady early steps around
  47-49% MFU and describe this as using almost half of available BF16 compute.
  URL: https://github.com/karpathy/nanochat/discussions/1

Working baseline for VLM MFU work:

- Treat text-only nanochat pretraining on 8xH100 as roughly 60% effective LLM
  MFU for current speedrun-style runs, not as a locally remeasured number.
- The VLM trainer's current `train/mfu` should be interpreted as effective LLM
  MFU under a VLM workload: estimated LLM train FLOPs divided by total VLM step
  wall time. It intentionally does not count frozen SigLIP forward FLOPs or the
  small projector FLOPs.
- Optimization target: keep the VLM metric close to the text-only baseline. If
  the comparable LLM baseline is around 60-66%, a healthy VLM target is roughly
  high-50s to about 60%, with the remaining gap explained by timing breakdowns
  rather than hidden by changing the denominator.
- Prefer transferable pipeline fixes: bounded streaming, async/prefetch,
  cached or precomputed image features when appropriate for the data regime,
  efficient batch construction, fewer unnecessary synchronizations, and clear
  timing attribution. Avoid changes that only overfit a local probe unless they
  are isolated and easy to remove.


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

500-step direct FineVision light eval output:

- Checkpoint: `/data/nanochat-llava/checkpoints/stage2_direct_finevision_probe/model_000500.pt`.
- JSON: `/data/nanochat-llava/bench/stage2_direct_finevision_probe_500.json`.
- Eval command used `--limit 16 --max-scan 240 --control --print-samples 3`.
- `mmstar`: `0.2500`, zero-image `0.1875`, control pass, image-only correct `2`, zero-only correct `1`, changed `14/16`.
- `scienceqa`: `0.6250`, zero-image `0.7500`, control fail, image-only correct `0`, zero-only correct `2`, changed `16/16`, skipped `17`, scanned `33`.
- `chartqa`: `0.0000`, zero-image `0.0625`, control fail, image-only correct `0`, zero-only correct `1`, changed `16/16`.
- `mmmu`: `0.2500`, zero-image `0.3750`, control fail, image-only correct `3`, zero-only correct `5`, changed `13/16`.
- `textvqa`: `0.1250`, zero-image `0.1875`, control fail, image-only correct `1`, zero-only correct `2`, changed `16/16`.

Step 100 to step 500 light eval deltas:

| Benchmark | Step 100 | Step 500 | Delta | Step 500 zero-image | Step 500 control |
| --- | ---: | ---: | ---: | ---: | --- |
| mmstar | 0.1875 | 0.2500 | +0.0625 | 0.1875 | pass |
| scienceqa | 0.6250 | 0.6250 | +0.0000 | 0.7500 | fail |
| chartqa | 0.0000 | 0.0000 | +0.0000 | 0.0625 | fail |
| mmmu | 0.3750 | 0.2500 | -0.1250 | 0.3750 | fail |
| textvqa | 0.0625 | 0.1250 | +0.0625 | 0.1875 | fail |

500-step full MMStar eval:

- Command used the same step-500 checkpoint with `--limit 0 --max-scan 0 --control`.
- The full five-benchmark eval was interrupted before the JSON was written, so this result is from observed stdout, not a completed result artifact.
- `mmstar`: `0.2113`, `n=1500`, skipped `0`, scanned `1500`.
- Zero-image: `0.2193`.
- Control: fail; image-conditioned score was `0.0080` absolute below zero-image.
- Image-only correct: `195`; zero-only correct: `207`.
- Predictions changed on `1267/1500` examples.
- Interpretation: the model uses image features enough to change most predictions, but on full MMStar at step 500 the image-conditioned run did not outperform the zero-image control.

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

## Full FineVision continuation attempt from 500-step checkpoint

Date: 2026-05-22 UTC.

Intent: continue Stage 2 from the 500-step direct FineVision checkpoint and train
over the full `HuggingFaceM4/FineVision` `LLaVA_Instruct_150K` split.

Command shape:

```text
python -m scripts.vlm_train \
  --stage 2 \
  --hf-repo HuggingFaceM4/FineVision \
  --hf-config LLaVA_Instruct_150K \
  --out-dir /data/nanochat-llava/checkpoints/stage2_full_finevision_from500 \
  --init-vlm-checkpoint-dir /data/nanochat-llava/checkpoints/stage2_direct_finevision_probe \
  --init-vlm-checkpoint-step 500 \
  --device-type cuda \
  --num-iterations 8000 \
  --device-batch-size 24 \
  --max-batch-tokens 12000 \
  --max-seq-len 2048 \
  --save-every 500 \
  --model-step 650 \
  --profile-timing
```

Finding: the current trainer is not true training-time streaming. It streams HF
records into `load_records()`, but still appends them into a Python list before
the first training step:

```python
records = []
for rec in iterator:
    records.append(rec)
    if len(records) >= limit:
        break
```

For `--num-iterations 8000`, the implicit record limit is:

```text
device_batch_size * grad_accum_steps * (num_iterations + 1) * 2
= 24 * 1 * 8001 * 2
= 384,048 records
```

Because `LLaVA_Instruct_150K` is smaller than that limit, the trainer attempts to
materialize the full split before logging `step 00001`. During the observed run,
after about 14 minutes the process was still in CPU/HF setup:

```text
PID 5855 elapsed 14:15, CPU ~15.7%, RSS ~20.0 GB
GPU utilization 0%, GPU memory ~27 GB
no files yet in /data/nanochat-llava/checkpoints/stage2_full_finevision_from500
```

The GPU memory was allocated because the base nanochat model had already loaded,
but the training loop had not started. The increasing RSS and `/proc/<pid>/io`
read counters indicated ongoing dataset materialization rather than a CUDA
training hang.

Conclusion: `--max-examples 4096` made the earlier probe start quickly, but a
full-data run currently pays a large up-front dataset materialization cost. To
train the whole LLaVA-Instruct split efficiently, the trainer should be changed
to use true lazy iteration: maintain a bounded shuffled buffer of rendered
examples and draw batches from that buffer, instead of loading all records into
memory before training.

Recommended implementation direction:

- Replace `load_records()` for HF dataset configs with an iterable record source.
- Render and filter examples lazily while filling a bounded shuffle buffer.
- Build batches from the buffer and refill as records are consumed.
- Keep local JSON mode unchanged for small/debug datasets.
- Save enough data cursor/seed metadata to make continuation behavior explicit,
  even if exact HF streaming resume is not initially supported.

## 2026-05-25 VLM training-system instrumentation and overlap pass

Intent: make the existing VLM training path more performance-measurable and
remove obvious CPU/data stalls without introducing a separate benchmark harness.

Changes made:

- Kept `train/mfu` as effective LLM MFU: estimated LLM train FLOPs for useful
  multimodal sequence positions divided by total VLM step wall time. The log now
  prints this as `eff_llm_mfu` to avoid implying full VLM MFU.
- Added `padded_llm_mfu`, `train/padded_tokens_per_sec`, `train/padding_frac`,
  useful tokens, padded tokens, and max sequence length. These are diagnostics
  only; `train/mfu` remains the optimization target so padding waste still shows
  up as lost effective MFU.
- Added `steady_mfu`/`train/steady_mfu`, a warmup-excluded aggregate controlled
  by `--mfu-warmup-steps`, so step 1 optimizer/setup cost is not mistaken for
  steady-state MFU.
- Split profile logs into
  `wait/data/image_total/open/processor/h2d/siglip/pool/batch/fwdbwd/optim`.
  `wait` is foreground wait for the prefetch queue; `data` is dataset iteration,
  rendering, and length-buffer batch selection; `open` is image open/decode;
  `processor` is CPU image transforms; `h2d`, `siglip`, and `pool` remain visible.
- Added bounded rendered-example length buffering for streaming training via
  `--batch-buffer-size` (default `4 * device_batch_size * grad_accum_steps` for
  true HF streaming). This improves padding efficiency without materializing the
  full dataset.
- Added CPU prefetch with `--prefetch-batches` and `--prefetch-processor`.
  Prefetch prepares the next batches, opens images, and optionally runs the
  image processor on CPU while the GPU is working on the current step. The step
  denominator still includes any queue wait; overlapped CPU work is reported in
  the profile counters instead of hidden.
- Split SigLIP preprocessing from GPU encoding in `SigLIPPooledFeatureExtractor`
  so CPU transforms can be prefetched while H2D transfer, SigLIP forward, and
  pooling stay on the measured training step.
- Filter path-only HF image dicts whose local files are missing before they enter
  rendered-example buffers when `--skip-bad-images` is active. This avoids wasting
  batch-buffer capacity on records that will fail during image open.
- `--siglip-use-fast-processor` defaults to enabled. In the current local env,
  Transformers falls back to the slow processor because `torchvision` is not
  installed; installing `torchvision` on GPU hosts is a removable setup-specific
  follow-up if CPU processor time remains material.
- Added `torchvision>=0.24.1` to the GPU extra and Modal image so the fast
  Transformers image processor can be used on GPU hosts instead of silently
  falling back to the slow SigLIP processor.
- Added opt-in `--compile` for `torch.compile(dynamic=True)` on the LLM path.
  Text-only nanochat uses compiled execution, so this is a likely important
  steady-state MFU knob. It intentionally remains opt-in until a GPU run verifies
  compile startup/recompile behavior with variable VLM batch shapes.
- Modal and `RUNBOOK_GPU.md` now expose the prefetch controls.
- Added a low-latency MFU probe path: trainer `--no-save`, Modal
  `mfu_probe`, `--log-every 1`, a small HF stream buffer, and a bounded
  length-aware batch buffer. This is meant to read warmup-excluded `steady_mfu`
  in a few steps instead of waiting for a step-10 run behind a large stream
  prefill.

Validation on local CPU:

```text
uv run python -m pytest tests/test_vision.py tests/test_vlm_smoke.py -q
23 passed in 1.51s

uv run python -m pytest tests/test_vision.py -q
22 passed in 1.57s

uv run python -m pytest -q
36 passed, 10 skipped in 1.46s

uv run --extra vision python -m scripts.vlm_train --help
ok; help includes --batch-buffer-size, --prefetch-batches, --prefetch-processor,
--siglip-use-fast-processor, --mfu-warmup-steps, --no-save, and --compile

uv run --extra vision python -m scripts.vlm_eval --help
ok

uv run python -m py_compile scripts/vlm_train.py nanochat/vision.py modal_vlm.py
ok

uv lock
Resolved 148 packages in 7.79s; added torchvision v0.24.1 variants

uv run --extra vision python - <<'PY'
from transformers import AutoImageProcessor
p = AutoImageProcessor.from_pretrained('google/siglip-base-patch16-512', use_fast=True)
print(type(p).__name__)
PY
SiglipImageProcessor; warning says torchvision is unavailable, so Transformers
falls back to the slow processor in this env
```

Current fast MFU check:

```bash
NANOCHAT_MODAL_GPU=H100 uv run --extra vision modal run modal_vlm.py::mfu_probe
```

What to inspect from that run:

- If `wait` is near zero but `data/open/processor` are nonzero, CPU data work is
  successfully overlapped.
- If `padding_frac` is high and `padded_llm_mfu` is much higher than
  `eff_llm_mfu`, the remaining MFU gap is dominated by batch shape/padding rather
  than image decode or SigLIP.
- If `siglip` is small relative to `fwdbwd`, precomputing SigLIP features is not
  justified for the current mostly-unique streamed data regime.
- If both `eff_llm_mfu` and `padded_llm_mfu` remain low with low `wait` and low
  image/SigLIP timing, the likely limit is the small per-GPU padded token count
  versus nanochat text-only's much larger 65,536-token microbatch/rank. The next
  scalable experiment is to raise `--max-batch-tokens` or change the multimodal
  batch builder to pack multiple short examples per row.

Equivalent local command:

```bash
uv run --extra vision --extra gpu python -m scripts.vlm_train \
  --hf-repo HuggingFaceM4/the_cauldron \
  --hf-config all \
  --out-dir $DATA_ROOT/checkpoints/vlm_cauldron_mfu_probe \
  --device-type cuda \
  --num-iterations 6 \
  --device-batch-size 256 \
  --max-batch-tokens 12000 \
  --max-seq-len 2048 \
  --no-save \
  --log-every 1 \
  --model-step 650 \
  --profile-timing \
  --stream-buffer-size 256 \
  --batch-buffer-size 512 \
  --prefetch-batches 2 \
  --mfu-warmup-steps 2 \
  --skip-bad-images
```

This command intentionally trades some shuffle quality for measurement latency.
It should be treated as a throughput probe, not a quality-training recipe. The
point is to read `steady_mfu` after a few post-warmup steps without spending
minutes filling a large streaming shuffle buffer. If the six-step number is
noisy, rerun the same shape with `--num-iterations 10` or `20`, not with the old
`4096` stream buffer.

Avoided for now:

- No feature cache or SigLIP precompute was reintroduced. The prior FineVision
  timing showed steady image+SigLIP work was small compared with LLM fwd/bwd, and
  streamed image examples are mostly unique.
- No separate benchmark harness was added; the existing trainer remains the
  measurement path.

Modal profile attempts:

```text
Attempt 1 command:
NANOCHAT_MODAL_GPU=H100 uv run --extra vision modal run modal_vlm.py::train
  --out-dir /vol/checkpoints/vlm_cauldron_mfu_probe
  --num-iterations 20 --batch-size 24 --max-batch-tokens 12000
  --max-examples 1024 --stream-buffer-size 4096 --prefetch-batches 2
  --profile-timing

Observed before failure:
Loaded 1,024 records in 180.96s
Rendered 903 usable examples in 0.36s
GPU: NVIDIA H100 80GB HBM3
step 00001/00020 | loss 2.917234 | samples/sec 0.73 |
tokens/sec 109 | eff_llm_mfu 0.13 | padded_llm_mfu 0.35 |
pad 61.9% | timing wait/data/image_total/open/processor/h2d/siglip/pool/batch/fwdbwd/optim
0.689/0.000/1.118/0.189/0.500/0.002/0.427/0.000/0.111/1.160/30.567s
```

The run failed at step 2 because a streamed The Cauldron image dict had no bytes
and its path pointed to the dataset builder filesystem:
`/fsx/m4/datasets/downloads/extracted/.../COCO_train2014_000000570694.jpg`.
This is a data-record issue, not a CUDA failure. The trainer default and Modal/
runbook commands now use `--skip-bad-images`; strict failure remains available
with `--no-skip-bad-images`.

Interpretation from the one completed step is limited because optimizer step 1
included a 30.567s Muon/AdamW optimizer cost. The useful signal is that the new
timing fields are present and already show high padding waste (`61.9%`) and
small SigLIP time (`0.427s`) relative to the first-step optimizer/setup cost.

Attempt 2 used the same materialized 1,024-record subset with the fixed
`--skip-bad-images` default and completed 10 steps on A100-80GB PCIe:

```text
Loaded 1,024 records in 109.25s
Rendered 903 usable examples in 0.35s
GPU: NVIDIA A100 80GB PCIe
step 00010/00010 | loss 1.788896 | samples/sec 4.12 |
tokens/sec 1106 | eff_llm_mfu 4.28 | padded_llm_mfu 12.06 |
pad 64.5% | timing wait/data/image_total/open/processor/h2d/siglip/pool/batch/fwdbwd/optim
0.000/0.000/0.398/0.078/0.289/0.002/0.029/0.000/0.173/2.191/0.515s
Peak memory usage: 63328.30MiB
```

Attempt 3 removed `--max-examples` to exercise true streaming and added
`--batch-buffer-size 128`. Step 1 is ignored as a buffer-fill/warmup step: it
spent about 216s filling the HF shuffle and length buffers. Step 10 is the better
steady-state signal:

```text
GPU: NVIDIA A100-SXM4-80GB
Initialized HF stream in 0.00s | shuffle_buffer=4,096 raw records |
batch_buffer=128 rendered examples
step 00010/00010 | loss 2.735467 | samples/sec 20.97 |
tokens/sec 1804 | eff_llm_mfu 6.98 | padded_llm_mfu 7.23 |
pad 3.4% | timing wait/data/image_total/open/processor/h2d/siglip/pool/batch/fwdbwd/optim
0.004/0.067/0.762/0.182/0.512/0.008/0.059/0.001/0.067/0.508/0.448s
Peak memory usage: 62476.53MiB
```

Interpretation:

- Step 1 is not a valid MFU datapoint. It includes optimizer first-step cost and
  HF/batch-buffer fill. Use `steady_mfu` after warmup; the low-latency probe now
  gets that signal before step 10 by using small buffers and `--log-every 1`.
- The streaming length buffer worked: padding fell from `64.5%` in the
  materialized subset probe to `3.4%` in the true-streaming step-10 probe.
- Data wait was effectively gone at step 10 (`0.004s wait`, `0.067s data`), so
  the dominant bottleneck is not foreground HF iteration.
- SigLIP forward was also not dominant at step 10 (`0.059s`), so a SigLIP feature
  cache is not justified by this evidence.
- CPU image open+processor remained material (`0.182s + 0.512s`) and should be
  revisited with a fast image processor/torchvision on the GPU host if it remains
  high after increasing batch tokens.
- The main remaining MFU gap is tiny useful token count per optimizer step:
  true-streaming step 10 processed only about 1.8K useful tokens on one A100.
  Text-only nanochat uses much larger per-rank token batches, compiled model
  execution, and for the strongest published runs FP8/FA3. The next scalable
  experiment is to raise `--device-batch-size` substantially while keeping
  `--max-batch-tokens` near the memory limit, or to add multimodal packing so
  short VLM examples can fill the token budget without requiring hundreds of
  simultaneous images.
- A follow-up compile probe should use `--compile` and compare only
  warmup-excluded `steady_mfu`, because compile startup/recompile time is not a
  steady-state throughput datapoint.

Continuation on 2026-05-25 after adding the low-latency Modal `mfu_probe`
entrypoint:

```text
Command:
NANOCHAT_MODAL_GPU=H100 uv run --extra vision modal run modal_vlm.py::mfu_probe

Actual trainer command:
python -m scripts.vlm_train --run dummy
  --hf-repo HuggingFaceM4/the_cauldron --hf-config all
  --out-dir /vol/checkpoints/vlm_cauldron_mfu_probe
  --device-type cuda --num-iterations 6 --device-batch-size 256
  --max-batch-tokens 12000 --max-seq-len 2048 --model-step 650
  --stream-buffer-size 256 --prefetch-batches 2 --mfu-warmup-steps 2
  --log-every 1 --no-save --batch-buffer-size 512 --skip-bad-images
  --profile-timing

GPU: NVIDIA H100 80GB HBM3
Pipeline: prefetch_batches=2 prefetch_processor=True batch_buffer_size=512
step 00006/00006 | loss 2.985773 | samples/sec 26.75 |
tokens/sec 3186 | eff_llm_mfu 3.89 | padded_llm_mfu 4.87 |
pad 20.1% | steady_mfu 4.05 (4 steps after warmup) |
timing wait/data/image_total/open/processor/h2d/siglip/pool/batch/fwdbwd/optim
0.004/0.419/1.827/0.895/0.673/0.008/0.247/0.004/0.665/1.893/0.169s
Peak memory usage: 64545.13MiB
Total training time: 3.32m
```

Findings from the H100 six-step probe:

- The Modal wrapper still prints `GPU_TYPE A100-80GB` because that string comes
  from the remote process environment. The authoritative trainer line reported
  `NVIDIA H100 80GB HBM3`, and this is the hardware used for the MFU denominator.
- Step 1 remained a buffer-fill/startup datapoint: `wait=150.390s`,
  `data=146.665s`, and optimizer first-step cost was `31.397s`. The
  warmup-excluded `steady_mfu` is the value to compare.
- After warmup, foreground `wait` was essentially zero (`0.000-0.004s`), so the
  VLM pipeline is not blocked on foreground HF iteration once the prefetch queue
  is warm.
- The non-overlapped vision encoder path was visible but not dominant:
  H2D+SigLIP+pool was roughly `0.15-0.26s` on steady steps, compared with
  LLM fwd/bwd around `1.4-2.9s`. CPU image open/processor work was material but
  mostly overlapped by prefetch; it should remain monitored through `wait`.
- The remaining gap to the published text-only nanochat ~60% MFU baseline is not
  explained by SigLIP or foreground data wait. The dominant issue is that this
  VLM shape feeds only about 3-4K useful tokens/sec to an H100 and uses a small
  per-step padded token budget relative to text-only speedrun settings.

VLM-only token-budget probes, without changing LLM code:

```text
Command:
NANOCHAT_MODAL_GPU=H100 uv run --extra vision modal run modal_vlm.py::mfu_probe \
  --num-iterations 6 --batch-size 512 --max-batch-tokens 24000 \
  --batch-buffer-size 256

Result:
OOM before step 1 at GPT.forward logits.float().
Peak context from error: GPU total 79.18GiB, process 75.94GiB, 3.23GiB free,
attempted 5.79GiB allocation.
```

Interpretation: with the current unchanged LLM forward path, a 24K padded-token
VLM microbatch is over the H100 memory boundary because full-vocab logits are
materialized for the padded multimodal batch. Per user scope, no LLM/loss-path
change was made.

```text
Command:
NANOCHAT_MODAL_GPU=H100 uv run --extra vision modal run modal_vlm.py::mfu_probe \
  --num-iterations 6 --batch-size 384 --max-batch-tokens 16000 \
  --batch-buffer-size 384

GPU: NVIDIA H100 80GB HBM3
step 00006/00006 | loss 1.798027 | samples/sec 12.16 |
tokens/sec 2373 | eff_llm_mfu 2.90 | padded_llm_mfu 10.40 |
pad 72.1% | steady_mfu 2.61 (4 steps after warmup) |
timing wait/data/image_total/open/processor/h2d/siglip/pool/batch/fwdbwd/optim
0.000/0.069/0.529/0.147/0.337/0.003/0.042/0.000/0.149/1.441/0.169s
Peak memory usage: 77475.05MiB
Total training time: 3.65m
```

Interpretation: raising the token budget without a length-aware rendered-example
buffer was a negative result. Padding rose to `62-74%`, peak memory approached
the H100 limit, and effective MFU fell despite `padded_llm_mfu` reaching about
8-10%. For this data shape, VLM-side length bucketing is necessary before larger
token budgets are useful.

Additional instrumentation change: stdout now prints useful/padded token counts
and max sequence length in each training line. These values were already logged
to wandb, but printing them makes future Modal logs self-contained for MFU
attribution.

Post-change local validation:

```text
uv run python -m pytest tests/test_vision.py -q
22 passed in 1.51s

uv run --extra vision python -m scripts.vlm_train --help
ok

uv run python -m py_compile scripts/vlm_train.py modal_vlm.py nanochat/vision.py
ok

uv run python -m pytest -q
36 passed, 10 skipped in 1.47s
```

VLM-side packing experiment, without changing LLM code:

Implementation added:

- `build_multimodal_batch(...)` now accepts `image_counts_per_row` so a packed
  VLM row can contain multiple `<image>` markers while image features remain a
  flat tensor in marker order. Existing one-image-per-row behavior is unchanged
  when `image_counts_per_row` is omitted.
- `scripts/vlm_train.py --pack-examples N` greedily packs up to `N` image-text
  examples into one VLM sequence row after SigLIP feature extraction and before
  `build_multimodal_batch`.
- `--pack-max-seq-len` optionally skips long examples for packed throughput
  probes so a packed row does not unintentionally hit the full 2048-token row
  length. This is opt-in and should be treated as a probe control, not a quality
  training default.

```text
Command:
NANOCHAT_MODAL_GPU=H100 uv run --extra vision modal run modal_vlm.py::mfu_probe \
  --num-iterations 6 --batch-size 128 --max-batch-tokens 12000 \
  --batch-buffer-size 256 --pack-examples 4

GPU: NVIDIA H100 80GB HBM3
step 00006/00006 | loss 2.273963 | samples/sec 10.89 | rows 11 |
tokens/sec 2522 | tokens 10,187/11,187 max_seq 1,017 |
eff_llm_mfu 3.08 | padded_llm_mfu 3.38 | pad 8.9% |
steady_mfu 3.13 (4 steps after warmup) |
timing wait/data/image_total/open/processor/h2d/siglip/pool/batch/fwdbwd/optim
0.000/0.549/2.706/1.272/1.247/0.015/0.167/0.004/0.755/2.925/0.169s
Peak memory usage: 63272.25MiB
```

Interpretation: packing worked mechanically: rows dropped to `11-21`, padded
waste improved, and the stdout token counts made this visible. It did not improve
MFU at the 12K cap because it selected only about 9-10K useful tokens per step
and increased CPU image work per measured step. The un-packed 12K H100 probe
remains better (`steady_mfu 4.05%`).

```text
Command:
NANOCHAT_MODAL_GPU=H100 uv run --extra vision modal run modal_vlm.py::mfu_probe \
  --num-iterations 5 --batch-size 128 --max-batch-tokens 16000 \
  --batch-buffer-size 256 --pack-examples 4

Observed before failure:
step 00003/00005 | tokens 14,030/14,336 max_seq 2,048 |
eff_llm_mfu 2.66 | padded_llm_mfu 2.72 | pad 2.1% |
steady_mfu 2.66 (1 steps after warmup)

Result: OOM during backward on a later step.
```

This showed that packing can reduce padding sharply, but long packed rows near
2048 sequence length are still too memory-heavy under the unchanged LLM forward
path.

```text
Command:
NANOCHAT_MODAL_GPU=H100 uv run --extra vision modal run modal_vlm.py::mfu_probe \
  --num-iterations 4 --batch-size 128 --max-batch-tokens 16000 \
  --batch-buffer-size 256 --pack-examples 4 --pack-max-seq-len 1024

Observed before failure:
step 00002/00004 | samples/sec 17.65 | rows 19 | tokens/sec 2962 |
tokens 12,756/15,219 max_seq 801 | eff_llm_mfu 3.62 |
padded_llm_mfu 4.32 | pad 16.2%

Result: OOM during backward on the next step.
```

Interpretation: even after skipping packed rows above the probe row-length cap,
the 16K packed-token setting still exceeded practical H100 memory for the current
unchanged LLM path. Packing is therefore retained as an opt-in VLM-side control,
but it is not the recommended MFU probe shape yet.

Current best evidence-backed attribution:

- Foreground data wait is not the steady-state bottleneck after prefetch warmup.
- SigLIP forward is visible but not dominant.
- CPU image open/processor can be seconds of work per step but is mostly
  overlapped unless `wait` rises.
- The remaining MFU gap is dominated by VLM batch shape under the unchanged LLM
  path: small useful-token throughput, full-vocab logits memory at larger padded
  token budgets, and long packed rows causing backward OOM. Without changing LLM
  internals, the currently stable H100 VLM effective MFU remains around 4% on
  this The Cauldron probe, far below the ~60% published text-only nanochat
  baseline.

Validation after packing changes:

```text
uv run python -m pytest tests/test_vision.py -q
24 passed in 1.30s

uv run python -m pytest -q
38 passed, 10 skipped in 1.63s

uv run --extra vision python -m scripts.vlm_train --help
ok; help includes --pack-examples, --pack-max-seq-len, and --no-save
```

Gradient accumulation MFU probe:

Implementation adjustment:

- Modal `train`/`mfu_probe` now expose `grad_accum_steps` so the existing trainer
  can measure whether optimizer overhead is a material part of the VLM MFU gap.
- The HF stream shuffle buffer minimum is now one microbatch
  (`device_batch_size`) rather than a full optimizer step
  (`device_batch_size * grad_accum_steps`). This keeps low-latency MFU probes
  from silently turning `--stream-buffer-size 256` into a 1024-record initial
  shuffle buffer when using `grad_accum_steps=4`. Long training defaults using
  the 4096 buffer are unchanged.

```text
Command:
NANOCHAT_MODAL_GPU=H100 uv run --extra vision modal run modal_vlm.py::mfu_probe \
  --num-iterations 4 --grad-accum-steps 4 --batch-size 256 \
  --max-batch-tokens 12000 --batch-buffer-size 512

GPU: NVIDIA H100 80GB HBM3
Pipeline: prefetch_batches=2 prefetch_processor=True batch_buffer_size=512
pack_examples=1 pack_max_seq_len=2048
step 00004/00004 | loss 2.670811 | samples/sec 23.57 | rows 241 |
tokens/sec 3362 | tokens 34,376/47,557 max_seq 211 |
eff_llm_mfu 4.11 | padded_llm_mfu 5.68 | pad 27.7% |
steady_mfu 3.96 (2 steps after warmup) |
timing wait/data/image_total/open/processor/h2d/siglip/pool/batch/fwdbwd/optim
0.013/0.985/6.190/2.994/2.661/0.033/0.496/0.006/1.412/8.090/0.169s
Peak memory usage: 71432.91MiB
```

Interpretation: gradient accumulation amortized the single optimizer step over
roughly 35-40K useful tokens, but it did not improve effective MFU versus the
default H100 probe (`steady_mfu 4.05%`). The per-microbatch LLM forward/backward
and VLM image/batch work dominate, not the optimizer step. `wait` stayed near
zero after warmup, so this does not change the data-wait conclusion.

Post-grad-accum validation:

```text
uv run python -m pytest -q
38 passed, 10 skipped in 1.62s

uv run --extra vision python -m scripts.vlm_train --help
ok

uv run python -m py_compile scripts/vlm_train.py modal_vlm.py nanochat/vision.py
ok
```

Compile MFU probe:

```text
Command:
NANOCHAT_MODAL_GPU=H100 uv run --extra vision modal run modal_vlm.py::mfu_probe \
  --num-iterations 4 --batch-size 256 --max-batch-tokens 12000 \
  --batch-buffer-size 512 --compile-model

GPU: NVIDIA H100 80GB HBM3
Compiled LLM with torch.compile(dynamic=True)
Pipeline: prefetch_batches=2 prefetch_processor=True batch_buffer_size=512
pack_examples=1 pack_max_seq_len=2048
step 00001/00004 | loss 4.645537 | samples/sec 0.25 | rows 125 |
tokens/sec 21 | tokens 10,779/12,000 max_seq 96 | eff_llm_mfu 0.03 |
padded_llm_mfu 0.03 | pad 10.2% | steady_mfu 0.00 (0 steps after warmup) |
timing wait/data/image_total/open/processor/h2d/siglip/pool/batch/fwdbwd/optim
133.176/129.386/4.423/1.193/2.565/0.008/0.657/0.001/0.422/347.995/21.777s

Interrupted after startup proved too slow for the fast MFU probe:
step 00002/00004 | loss 4.130963 | samples/sec 44.60 | rows 105 |
tokens/sec 4418 | tokens 10,402/11,970 max_seq 114 | eff_llm_mfu 5.40 |
padded_llm_mfu 6.21 | pad 13.1% | steady_mfu 0.00 (0 steps after warmup) |
timing wait/data/image_total/open/processor/h2d/siglip/pool/batch/fwdbwd/optim
0.000/0.619/4.361/1.548/2.588/0.008/0.209/0.007/0.425/1.530/0.169s
```

Interpretation: compile is not a good default for a quick VLM MFU probe. The
first visible step paid a huge compile/warmup cost (`fwdbwd 347.995s`,
`optim 21.777s`, plus `wait/data` startup), so it took roughly nine minutes to
reach usable output. The second step suggests compile may improve post-warmup
throughput (`eff_llm_mfu 5.40%` versus the uncompiled H100 steady baseline
`4.05%`), but the run did not reach any warmup-excluded `steady_mfu` sample.
Treat compile as an optional long-run experiment, not as the recommended fast
measurement path.

Current conclusion for the "can VLM get close to LLM MFU?" question:

- The comparable metric is still effective LLM MFU under the VLM workload:
  estimated LLM train FLOPs divided by total VLM step wall time.
- Published text-only nanochat H100 runs are around 50-65% MFU, with a working
  target of about 60%.
- The stable unchanged-LLM VLM probe is around 4% on H100. Packing, larger token
  budgets, gradient accumulation, and compile have not yet produced a validated
  steady-state result close to the text-only baseline.
- The gap is not explained by foreground data wait or frozen SigLIP alone. The
  evidence points to VLM batch shape and memory behavior around the unchanged LLM
  path: many short image-text rows, padded/full-vocab-logits memory pressure, and
  poor scaling when we try to raise the padded token budget.

Selective-loss and fixed-shape compile probes:

```text
Change:
- Added opt-in GPT selective loss for VLM training/eval. This keeps nanochat's
  existing logits softcap and CE semantics, but computes lm_head/CE only at
  supervised target positions when `selective_loss=True`.
- Added optional fixed-shape VLM padding (`--pad-to-max-seq-len`) and
  `--no-selective-loss` for compile probes that need a static full-CE graph.

Validation:
uv run python -m pytest tests/test_vision.py tests/test_vlm_smoke.py -q
27 passed

uv run python -m py_compile nanochat/gpt.py nanochat/vision.py scripts/vlm_train.py modal_vlm.py
ok
```

24K selective-loss probe:

```text
Command:
NANOCHAT_MODAL_GPU=H100 uv run --extra vision modal run modal_vlm.py::mfu_probe \
  --num-iterations 6 --batch-size 512 --max-batch-tokens 24000 \
  --batch-buffer-size 2048

Result:
- The old OOM at GPT logits.float() is gone.
- The run now OOMs on step 2 in transformer MLP activation allocation after
  optimizer state has been allocated.

First logged step:
tokens 23,066/23,925 max_seq 87 | pad 3.6%
fwdbwd 3.630s | optim 40.458s
CUDA OOM: tried to allocate 376MiB with ~79.03GiB in use.
```

Interpretation: selective loss removed the full-vocab logits ceiling, but 24K
padded tokens is still too large for d32 full-LLM finetuning on one H100 once
optimizer state and activations are both resident.

16K selective-loss probe:

```text
Command:
NANOCHAT_MODAL_GPU=H100 uv run --extra vision modal run modal_vlm.py::mfu_probe \
  --num-iterations 6 --batch-size 512 --max-batch-tokens 16000 \
  --batch-buffer-size 2048

GPU: NVIDIA H100 80GB HBM3
step 00006/00006 | rows 121 | tokens/sec 3636 |
tokens 14,502/15,972 max_seq 132 | eff_llm_mfu 4.44 |
padded_llm_mfu 4.89 | pad 9.2% | steady_mfu 4.49 |
timing wait/data/image_total/open/processor/h2d/siglip/pool/batch/fwdbwd/optim
0.000/0.447/2.398/1.092/1.156/0.007/0.138/0.004/0.843/2.821/0.172s
Peak memory usage: 66575.06MiB
```

Interpretation: stable and slightly better than the prior H100 baseline
(`steady_mfu 4.05% -> 4.49%`), but still nowhere near text nanochat. Step-level
variation correlates with sequence length: short `max_seq~88` steps reached
~6.7% MFU, while `max_seq~181` dropped to ~3.7% at similar token counts.

Fixed-shape compile probe, 128x128:

```text
Command:
NANOCHAT_MODAL_GPU=H100 uv run --extra vision modal run modal_vlm.py::mfu_probe \
  --num-iterations 6 --batch-size 128 --max-batch-tokens 0 --max-seq-len 128 \
  --batch-buffer-size 2048 --compile-model --pad-to-max-seq-len --no-selective-loss

Compiled LLM with torch.compile(dynamic=False)
step 00006/00006 | rows 128 | tokens/sec 6086 |
tokens 11,119/16,384 max_seq 128 | eff_llm_mfu 7.43 |
padded_llm_mfu 10.95 | pad 32.1% | steady_mfu 5.94 |
fwdbwd 1.338s | optim 0.172s
Peak memory usage: 63642.55MiB
```

Fixed-shape compile probe, 192x96:

```text
Command:
NANOCHAT_MODAL_GPU=H100 uv run --extra vision modal run modal_vlm.py::mfu_probe \
  --num-iterations 6 --batch-size 192 --max-batch-tokens 0 --max-seq-len 96 \
  --batch-buffer-size 2048 --compile-model --pad-to-max-seq-len --no-selective-loss

Compiled LLM with torch.compile(dynamic=False)
step 00006/00006 | rows 192 | tokens/sec 6812 |
tokens 16,446/18,432 max_seq 96 | eff_llm_mfu 8.32 |
padded_llm_mfu 9.32 | pad 10.8% | steady_mfu 6.69 |
fwdbwd 1.848s | optim 0.175s
Peak memory usage: 68967.92MiB
```

Interpretation: fixed-shape compile is the best throughput direction so far:
validated warmup-excluded effective MFU improves from 4.05% baseline to 6-7%,
with best individual steps around 8% effective / 9-11% padded. This confirms
the text-nanochat style fixed shape helps, but the remaining gap to ~60% is
still about an order of magnitude. The major remaining difference is that VLM is
running small/short multimodal rows through an eager Python multimodal batch
builder and a d32 LLM finetune path, while text nanochat uses a completely fixed
token matrix and a highly tuned compiled training loop.

Follow-up MFU pass after simplifying to nanoVLM-style concatenation:

Implementation changes:

- Removed the slow strict boundary-aware dense SDPA path from the hot training
  route. Per the latest training-system direction, packed rows use ordinary
  causal attention and rely on the model/data format to learn separation.
- Vectorized `build_multimodal_batch(...)`: it now builds a dense token-id tensor,
  runs one `wte(...)` lookup for the whole batch, projects all SigLIP visual
  tokens in one projector call, then overwrites image spans. This removes
  thousands of tiny per-token embedding/projector calls from each VLM step.
- Added `--prefetch-workers` so CPU image open/processor work can prepare
  multiple future batches instead of serializing behind one prefetch thread.
- Added `--pad-to-bucket-lens` for realistic static-shape training. A global
  `--max-seq-len 96` is only a short-bucket probe because 64 of those positions
  are image tokens; bucketed padding lets short examples run at 96 while longer
  examples are retained in larger buckets.
- Bucketed token accounting is used for `--max-batch-tokens`, including packed
  rows, so larger buckets automatically select fewer rows instead of exceeding
  the padded-token budget. `mfu_probe` now exposes `--prefetch-batches` as well
  as `--prefetch-workers`.
- Streaming length-buffer selection is bucket-pure when `--pad-to-bucket-lens`
  is active: it samples a random buffered example and then builds the step from
  examples in that same padding bucket. This preserves the buffer's rough length
  distribution while avoiding accidental cross-bucket batches and extra static
  compile shapes.
- Materialized/local datasets use the same static-shape discipline: each epoch
  shuffles examples, stable-sorts by padding bucket, and stops batches at bucket
  boundaries.
- Added a CPU-only length diagnostic mode:
  `--length-stats-examples N --pad-to-bucket-lens ...`. It renders/tokenizes
  usable one-image examples, reports expanded multimodal length percentiles, and
  estimates rows per bucket at the chosen `--max-batch-tokens` cap. The report
  also prints `text_cap_1img`, which is the non-image token capacity left in each
  bucket after the fixed 64 visual tokens. This is the guardrail against treating
  the 96-token short-bucket result as a realistic full-distribution training
  length.
- Added per-static-bucket steady metrics for bucketed probes. Logs now include
  the current `bucket`/`bucket_steady_mfu` when an optimizer step has one static
  shape, and training prints a final `Bucket steady stats` table. This keeps a
  realistic mixed-bucket run from hiding whether the MFU drop is isolated to
  longer sequence buckets.
- Bucketed batch selection now prefers the longest rows inside the selected
  static bucket, both for streaming buffers and materialized/local datasets. The
  bucket choice still comes from the buffered distribution, but each step wastes
  fewer padded positions inside that bucket.
- Added `--mfu-warmup-bucket-steps` and wired it through Modal. This is meant for
  bucketed static compile probes: the first measured occurrence of each new
  bucket can pay compile/setup cost, so the realistic H100 probe should use
  `--mfu-warmup-bucket-steps 1` and compare steady-state bucket stats after each
  shape has warmed.
- Added `--bucket-selection {sample,cycle,max-tokens}` for bucketed streaming
  batches. Default `sample` keeps training distribution behavior; `cycle` is for
  MFU probes so available static buckets are visited in order and each bucket can
  collect post-warmup measurements.
- Added `--bucket-min-fill-frac` for bucketed streaming batches. The realistic
  MFU probe uses `--bucket-min-fill-frac 0.75` so `cycle` skips a bucket until
  enough rows have accumulated to avoid timing an underfilled static shape.
- Added `--batch-plan-steps`, a CPU-only dry run that renders/selects training
  batches with the same bucket policy, token cap, buffer size, and min-fill gate,
  then reports per-bucket rows, fill, useful/padded tokens, and padding. This
  lets the realistic H100 probe recipe be checked before launching Modal.
- CPU batch-plan smoke on `HuggingFaceM4/the_cauldron/vqav2` with
  `--device-batch-size 8 --max-batch-tokens 512 --pad-to-bucket-lens 96,128,192,256
  --bucket-selection cycle --bucket-min-fill-frac 0.5 --batch-plan-steps 4`
  completed and exited before model/GPU setup. It reported 4 planned steps,
  3.0 rows/step, 448/480 useful/padded tokens per step, and 6.7% padding.
- Batch-plan accounting now reuses the same packed-row grouping helper as
  `pack_example_rows`, so `--batch-plan-steps` also models `--pack-examples`.
  A small `--pack-examples 2 --pack-max-seq-len 256` smoke completed and exited
  before model/GPU setup, reporting 2.0 packed rows/step and 450/480
  useful/padded tokens per step.

Validation:

```text
uv run python -m pytest -q
51 passed, 10 skipped

uv run python -m py_compile scripts/vlm_train.py modal_vlm.py nanochat/vision.py nanochat/gpt.py nanochat/flash_attention.py
ok

uv run --extra vision python -m scripts.vlm_train --help
ok; help includes --pad-to-bucket-lens, --prefetch-workers, and
--length-stats-examples
```

Negative/diagnostic probes:

```text
64 rows x 256 tokens, packed 4 examples/row, static compile:
steady_mfu 4.29

Interpretation: longer 256-token rows were attention/memory heavier and worse
than the 96-token fixed-shape direction.

208 rows x 96 tokens before multi-worker prefetch:
steady_mfu 7.58
best post-warmup steps: eff_llm_mfu 15.86 and 20.94

Interpretation: after vectorizing the batch builder, the LLM path could hit
15-21% MFU, but the single prefetch worker drained and foreground wait hurt the
six-step aggregate.
```

Best H100 probe so far:

```text
Command:
NANOCHAT_MODAL_GPU=H100 uv run --extra vision modal run modal_vlm.py::train \
  --num-iterations 8 --batch-size 224 --max-batch-tokens 0 --max-seq-len 96 \
  --stream-buffer-size 256 --batch-buffer-size 2048 \
  --prefetch-batches 8 --prefetch-workers 4 \
  --mfu-warmup-steps 2 --log-every 1 --no-save \
  --compile-model --pad-to-max-seq-len --no-selective-loss --profile-timing

GPU: NVIDIA H100 80GB HBM3
Pipeline: prefetch_batches=8 prefetch_workers=4 batch_buffer_size=2048
step 00008/00008 | rows 224 | tokens/sec 24710 |
tokens 21,297/21,504 max_seq 96 | eff_llm_mfu 30.18 |
padded_llm_mfu 30.47 | pad 1.0% | steady_mfu 22.96 |
fwdbwd 0.443s | optim 0.179s
Peak memory usage: 76636.18MiB
```

Current comparison:

```text
baseline H100 VLM probe:                 steady_mfu 4.05
16K selective-loss probe:                steady_mfu 4.49
old 192x96 fixed compile:                steady_mfu 6.69
vectorized 192x96 fixed compile:         steady_mfu 9.19
208x96 + 4 prefetch workers:             steady_mfu 17.36
224x96 + 4 prefetch workers:             steady_mfu 22.96
```

Interpretation:

- The strategy works for short buckets: after vectorization and multi-worker CPU
  prefetch, the best individual H100 steps reached roughly 26-30% effective MFU,
  and the warmup-excluded aggregate reached 22.96%.
- `max_seq_len=96` is not a realistic global training max. It leaves about 32
  text-side positions after the fixed 64 image tokens. It should be treated as
  a short-bucket microbenchmark, not as the whole training distribution.
- The next realistic path is bucketed static training, for example
  `--max-seq-len 512 --pad-to-bucket-lens 128,192,256,384,512`, with
  `--max-batch-tokens` set near the H100 memory limit so each bucket uses an
  appropriate row count. This keeps longer examples while preserving the static
  shapes that made the short bucket fast.
- The remaining gap to text-only nanochat is now partly real VLM workload shape:
  longer buckets will have fewer rows and lower MFU, and single-H100 d32
  finetuning is near memory limit by about 21-22K padded tokens with full CE.

## 2026-05-26 realistic bucket diagnostics

Modal remains paused. I added local timing/rate fields to the CPU-only
`--length-stats-examples` and `--batch-plan-steps` reports so future H100 probes
can separate GPU MFU from dataset/rendering setup cost.

CPU-only report startup now uses a tokenizer-only nanochat checkpoint link path.
With the default `--hf-checkpoint karpathy/nanochat-d32`, a tiny length report
fetched only the two tokenizer files before scanning data; full model checkpoint
linking is deferred until real training/eval.

```text
uv run --extra vision python -m scripts.vlm_train \
  --hf-repo HuggingFaceM4/the_cauldron --hf-config vqav2 \
  --max-seq-len 512 --pad-to-bucket-lens 96,128,192,256,384,512 \
  --max-batch-tokens 21504 --length-stats-examples 5 \
  --length-stats-max-records 50 --stream-buffer-size 16 --model-step 650

Fetching 2 files
records_scanned=5 usable_one_image_examples=5
```

An attempted quick scan with `--hf-config all` did not produce a report within a
useful local window and was killed. The Cauldron has 50 configs, so quick MFU
diagnostics should start with named configs before paying the all-config stream
setup cost.

Modal `mfu_probe` now defaults to `--hf-config vqav2` for this reason. Normal
training still defaults to `all`; the probe default is intentionally tuned for
fast, reproducible throughput evidence.

Current config list count:

```text
HuggingFaceM4/the_cauldron configs: 50
```

Length check on `vqav2`:

```text
uv run --extra vision python -m scripts.vlm_train \
  --hf-repo HuggingFaceM4/the_cauldron --hf-config vqav2 \
  --max-seq-len 512 --pad-to-bucket-lens 96,128,192,256,384,512 \
  --max-batch-tokens 21504 --length-stats-examples 100 \
  --length-stats-max-records 1000 --stream-buffer-size 256 --model-step 650

records_scanned=100 usable_one_image_examples=100
scan_elapsed=12.6s usable_examples/sec=7.9
expanded_len min/p50/p80/p90/p95/p99/max/mean 113/129/157/224/299/416/458/152.2
fit_at_max_seq_len_512=100/100 (100.0%)
bucket | count | pct | cumulative | avg_len | avg_pad | text_cap_1img | rows@cap
    96 |     0 |   0.0% |      0 (  0.0%) |     0.0 |    0.0% |            32 | 224
   128 |    49 |  49.0% |     49 ( 49.0%) |   119.5 |    6.6% |            64 | 168
   192 |    38 |  38.0% |     87 ( 87.0%) |   145.5 |   24.2% |           128 | 112
   256 |     5 |   5.0% |     92 ( 92.0%) |   227.4 |   11.2% |           192 | 84
   384 |     6 |   6.0% |     98 ( 98.0%) |   303.0 |   21.1% |           320 | 56
   512 |     2 |   2.0% |    100 (100.0%) |   437.0 |   14.6% |           448 | 42
```

Batch plan on `vqav2` with the realistic 21,504 padded-token cap:

```text
uv run --extra vision python -m scripts.vlm_train \
  --hf-repo HuggingFaceM4/the_cauldron --hf-config vqav2 \
  --device-batch-size 512 --max-batch-tokens 21504 --max-seq-len 512 \
  --stream-buffer-size 256 --batch-buffer-size 2048 \
  --bucket-selection cycle --bucket-min-fill-frac 0.75 \
  --pad-to-bucket-lens 128,192,256,384,512 \
  --batch-plan-steps 8 --model-step 650

planning_elapsed=42.3s records_scanned=2,826 rendered_examples=2,790 rendered_examples/sec=65.9
overall rows/step=103.2 tokens/step=19680/21504 pad=8.5%
bucket | steps | rows avg/min/max | target_rows | avg_fill | tokens/step useful/padded | pad
   128 |     2 | 168.0/168/168 |         168 |   100.0% |   20885/21504   |   2.9%
   192 |     2 | 112.0/112/112 |         112 |   100.0% |   19820/21504   |   7.8%
   256 |     2 |  84.0/ 84/84  |          84 |   100.0% |   19343/21504   |  10.0%
   384 |     1 |  56.0/ 56/56  |          56 |   100.0% |   18827/21504   |  12.4%
   512 |     1 |  42.0/ 42/42  |          42 |   100.0% |   18512/21504   |  13.9%
```

Interpretation: the 96 bucket is not present for this vqav2 sample, so the
realistic static run starts at 128. It should expect most steps at 128/192 and
some at 256-512. The selector can fill the token-capped row counts for those
buckets, so the next MFU question is actual H100 per-bucket throughput, not
batch underfill.

Added `attn_pairs/step` to the batch-plan and train logs. This is not a new MFU
denominator; it is a causal attention-pair estimate for spotting longer buckets
whose attention work is much larger even at the same padded-token cap.

Added sequence-aware MFU diagnostics in the train logs:
`seq_mfu`, `seq_padded_mfu`, `steady_seq_mfu`, `steady_seq_padded_mfu`, and
`bucket_steady_seq_padded_mfu`. At the time this was added, the legacy
`eff_llm_mfu`/`padded_llm_mfu`/`steady_mfu` fields were left unchanged for
comparison. Later selective-aware accounting changed the main MFU fields to use
the sequence-aware path too, with `train/token_estimate_mfu` preserving the old
full-per-token estimate for comparison. The sequence-aware path matters for mixed
buckets because the model's config sequence length can be larger than the current
static bucket.

Updated vqav2 batch-plan output:

```text
planning_elapsed=59.5s records_scanned=2,826 rendered_examples=2,790 rendered_examples/sec=46.9
overall rows/step=103.2 tokens/step=19680/21504 pad=8.5% attn_pairs/step=2.76M
bucket | steps | rows avg/min/max | target_rows | avg_fill | tokens/step useful/padded | pad | attn_pairs/step
   128 |     2 | 168.0/168/168 |         168 |   100.0% |   20885/21504   |   2.9% |           1.39M
   192 |     2 | 112.0/112/112 |         112 |   100.0% |   19820/21504   |   7.8% |           2.08M
   256 |     2 |  84.0/ 84/84  |          84 |   100.0% |   19343/21504   |  10.0% |           2.76M
   384 |     1 |  56.0/ 56/56  |          56 |   100.0% |   18827/21504   |  12.4% |           4.14M
   512 |     1 |  42.0/ 42/42  |          42 |   100.0% |   18512/21504   |  13.9% |           5.52M
```

The 512 bucket has about 4.0x the causal attention-pair count of the 128 bucket
despite the same 21,504 padded-token cap. If the mixed-bucket H100 probe falls
off on long buckets, this is the first local explanation to verify against the
per-bucket `bucket_steady_mfu` and `fwdbwd` timings.

Packed loose-boundary check with `--pack-examples 2`:

```text
uv run --extra vision python -m scripts.vlm_train \
  --hf-repo HuggingFaceM4/the_cauldron --hf-config vqav2 \
  --device-batch-size 512 --max-batch-tokens 21504 --max-seq-len 512 \
  --pack-examples 2 --pack-max-seq-len 512 \
  --stream-buffer-size 256 --batch-buffer-size 1024 \
  --bucket-selection cycle --bucket-min-fill-frac 0.75 \
  --pad-to-bucket-lens 128,192,256,384,512 \
  --batch-plan-steps 4 --model-step 650

planning_elapsed=68.5s records_scanned=2,517 rendered_examples=2,485 rendered_examples/sec=36.3
overall rows/step=49.0 examples/step=98.0 dropped/step=363.8 tokens/step=14567/21504 pad=32.3% attn_pairs/step=4.83M
bucket | steps | rows avg/min/max | examples avg/dropped | target_rows | avg_fill | tokens/step useful/padded | pad | attn_pairs/step
   384 |     2 |  56.0/ 56/56  |  112.0/362.5   |          56 |   100.0% |   14028/21504   |  34.8% |           4.14M
   512 |     2 |  42.0/ 42/42  |   84.0/365.0   |          42 |   100.0% |   15106/21504   |  29.8% |           5.52M
```

Interpretation: loose nanoVLM-style packing is not attractive in this form for
the realistic vqav2 bucket shape. It processes fewer examples per step than the
unpacked bucketed plan (`98.0` vs `103.2`), uses fewer useful tokens
(`14.6K` vs `19.7K`), has much higher padding (`32.3%` vs `8.5%`), and raises
attention pairs (`4.83M` vs `2.76M`). Worse, the current selector consumes many
rendered examples that the packer then cannot fit under the token cap. Training
logs now print packed/selected samples and dropped samples so this cannot be
mistaken for real throughput in an H100 probe. The better near-term MFU path is
the unpacked bucketed static-shape path, not loose packing.

Follow-up implementation: packed training now trims examples that cannot fit
the selected packed rows before image open/processor/SigLIP work. The
`dropped_samples` log still records the gap between selected examples and packed
examples, but those dropped examples should no longer consume vision encoder
time in the actual training path. This does not make loose packing a good MFU
path for vqav2; it only prevents the diagnostic/probe from wasting CPU/GPU image
work on examples that will be discarded.

Current local validation after sequence-aware MFU diagnostics, tokenizer-only
CPU reports, packed-example trimming, and the `mfu_probe` default switch to
`vqav2`:

```text
uv run python -m py_compile nanochat/gpt.py nanochat/vision.py nanochat/flash_attention.py scripts/vlm_train.py modal_vlm.py
ok

uv run python -m pytest -q
55 passed, 10 skipped
```

Added `modal_vlm.py::bucketed_mfu_probe` so the realistic H100 probe has one
reproducible entrypoint instead of requiring a long list of overrides to the
cheap unbucketed `mfu_probe` sanity path.

Current H100 command-builder preview from `build_bucketed_mfu_probe_cmd()`:

```text
python -m scripts.vlm_train --run dummy --hf-repo HuggingFaceM4/the_cauldron
--hf-config vqav2 --out-dir /vol/checkpoints/vlm_cauldron_mfu_probe
--device-type cuda --num-iterations 14 --device-batch-size 512
--grad-accum-steps 1 --max-batch-tokens 21504 --max-seq-len 512
--model-step 650 --stream-buffer-size 256 --bucket-selection cycle
--bucket-min-fill-frac 0.75 --prefetch-batches 8 --prefetch-workers 4
--mfu-warmup-steps 2 --mfu-warmup-bucket-steps 1 --log-every 1
--no-save --compile --batch-buffer-size 4096
--pad-to-bucket-lens 128,192,256,384,512 --no-selective-loss
--skip-bad-images
```

Follow-up cleanup: the realistic bucketed probe now leaves `--profile-timing`
off by default so the target MFU number does not include extra CUDA
synchronization overhead from timing attribution. The cheaper unbucketed
`mfu_probe` still profiles by default for sanity runs, and
`bucketed_mfu_probe --profile-timing` remains available as a separate diagnostic
pass.

Added a boundary-aware packed-row path behind `--boundary-aware-pack`. The packed
batch builder now carries per-token `position_ids`, `segment_ids`,
`segment_starts`, `cu_seqlens`, `max_segment_len`, and flat gather indices.
`GPT.forward` resets RoPE from those position ids, suppresses smear at packed
segment starts, gathers real segment tokens into flattened q/k/v tensors, calls
`flash_attn_varlen_func`, then scatters attention outputs back to the padded row
layout. The wrapper uses the FA3 Hopper varlen API when available and a
per-segment SDPA fallback for CPU/non-Hopper correctness. H100 MFU for this path
is still unproven until Modal is resumed.

Focused validation:

```text
uv run python -m py_compile nanochat/gpt.py nanochat/vision.py nanochat/flash_attention.py scripts/vlm_train.py modal_vlm.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
54 passed, 10 skipped
```

Added `modal_vlm.py::packed_mfu_probe` as the named H100 entrypoint for the
boundary-aware varlen path. Its command-builder defaults match the first packed
probe in the goal prompt: `--boundary-aware-pack`, `--pack-examples 8`,
`--pack-max-seq-len 512`, `--max-batch-tokens 32768`,
`--device-batch-size 512`, `--batch-buffer-size 4096`, four CPU prefetch
workers, selective loss, clean MFU timing, `--require-fa3-varlen`, and no
`--compile` by default. `--require-fa3-varlen` makes the H100 probe fail fast
instead of silently benchmarking the CPU/non-Hopper SDPA fallback. Use
`packed_mfu_probe --compile-model` as the follow-up ablation after the clean
varlen path is stable on H100.

Added `modal_vlm.py::attention_backend`, a cheap GPU backend check that runs
`scripts.vlm_train --attention-backend-report --boundary-aware-pack
--require-fa3-varlen` and exits before checkpoint/model/SigLIP setup. Use it
before the first packed H100 MFU run on a new Modal image to verify that the FA3
varlen kernel is actually active.

CPU-only packed-varlen batch-plan preflights:

```text
batch_size=512:
planning_elapsed=102.0s records_scanned=5,728 rendered_examples=5,632 rendered_examples/sec=55.2
overall rows/step=71.0 examples/step=247.0 dropped/step=265.0 tokens/step=32020/32479 pad=1.4% attn_pairs/step=7.47M

batch_size=256:
planning_elapsed=87.8s records_scanned=4,946 rendered_examples=4,864 rendered_examples/sec=55.4
overall rows/step=71.2 examples/step=245.8 dropped/step=10.2 tokens/step=31044/31457 pad=1.3% attn_pairs/step=6.99M
```

The 256-example selector was initially the better default than 512: it kept
almost the same row count and near-full 32K token cap, but avoided rendering and
then discarding roughly half of the selected examples.

Follow-up measurement cleanup: packed-varlen batch planning and training logs now
count causal attention pairs from original packed segment lengths, not dense
packed-row length. Sequence-aware FLOP diagnostics also use per-segment attention
lengths for `seq_mfu`; `seq_padded_mfu` adds padded-token non-attention matmul
work plus the real varlen segment attention work, instead of pretending the
varlen path ran dense attention over padded packed rows.

Follow-up semantic cleanup: boundary-aware packing now skips the boundary-only
input token between packed examples. The previous example's final token remains
the target of the previous position, but it is no longer processed as an ignored
input token before the next segment. This makes each packed segment's expanded
input length match running that example alone and removes one useless LLM token
per packed boundary. Loose packing keeps dense row attention/accounting.

Short CPU-only preflight after this accounting fix:

```text
uv run --extra vision python -m scripts.vlm_train \
  --hf-repo HuggingFaceM4/the_cauldron --hf-config vqav2 \
  --device-batch-size 256 --max-batch-tokens 32768 --max-seq-len 512 \
  --stream-buffer-size 256 --batch-buffer-size 4096 \
  --pack-examples 8 --pack-max-seq-len 512 --boundary-aware-pack \
  --batch-plan-steps 2 --model-step 650

planning_elapsed=63.0s records_scanned=4,420 rendered_examples=4,352 rendered_examples/sec=69.1
overall rows/step=64.0 examples/step=256.0 dropped/step=0.0 tokens/step=29908/30272 pad=1.2% attn_pairs/step=1.76M
```

CPU-only preflight after the boundary-only input cleanup:

```text
uv run --extra vision python -m scripts.vlm_train \
  --hf-repo HuggingFaceM4/the_cauldron --hf-config vqav2 \
  --device-batch-size 256 --max-batch-tokens 32768 --max-seq-len 512 \
  --stream-buffer-size 256 --batch-buffer-size 4096 \
  --pack-examples 8 --pack-max-seq-len 512 --boundary-aware-pack \
  --batch-plan-steps 2 --model-step 650

planning_elapsed=48.2s records_scanned=4,420 rendered_examples=4,352 rendered_examples/sec=90.2
overall rows/step=64.0 examples/step=256.0 dropped/step=0.0 tokens/step=29716/30080 pad=1.2% attn_pairs/step=1.74M
bucket | steps | rows avg/min/max | examples avg/dropped | target_rows | avg_fill | tokens/step useful/padded | pad | attn_pairs/step
   460 |     1 |  64.0/ 64/64  |  256.0/0.0     |          71 |    90.1% |   28983/29440   |   1.6% |           1.66M
   480 |     1 |  64.0/ 64/64  |  256.0/0.0     |          68 |    94.1% |   30448/30720   |   0.9% |           1.83M
```

Packed selector grid on one 4096-example rendered buffer after the boundary
cleanup:

```text
batch_size=240 rows=60 examples=240 dropped=0 tokens=28463/28560 pad=0.3% bucket=476 target_rows=68 fill=88.2% attn_pairs=1.70M
batch_size=256 rows=64 examples=256 dropped=0 tokens=30441/30720 pad=0.9% bucket=480 target_rows=68 fill=94.1% attn_pairs=1.83M
batch_size=272 rows=68 examples=272 dropped=0 tokens=32457/32640 pad=0.6% bucket=480 target_rows=68 fill=100.0% attn_pairs=1.95M
batch_size=288 rows=64 examples=192 dropped=96 tokens=31628/32768 pad=3.5% bucket=512 target_rows=64 fill=100.0% attn_pairs=2.62M
batch_size=304 rows=88 examples=176 dropped=128 tokens=31248/32560 pad=4.0% bucket=370 target_rows=88 fill=100.0% attn_pairs=2.79M
batch_size=320 rows=80 examples=160 dropped=160 tokens=31242/32480 pad=3.8% bucket=406 target_rows=80 fill=100.0% attn_pairs=3.07M
batch_size=352 rows=64 examples=128 dropped=224 tokens=32751/32768 pad=0.1% bucket=512 target_rows=64 fill=100.0% attn_pairs=4.21M
batch_size=384 rows=64 examples=64 dropped=192 tokens=28289/32768 pad=13.7% bucket=512 target_rows=64 fill=100.0% attn_pairs=6.30M
```

Normal batch-plan CLI verification for the new packed default:

```text
uv run --extra vision python -m scripts.vlm_train \
  --hf-repo HuggingFaceM4/the_cauldron --hf-config vqav2 \
  --device-batch-size 272 --max-batch-tokens 32768 --max-seq-len 512 \
  --stream-buffer-size 256 --batch-buffer-size 4096 \
  --pack-examples 8 --pack-max-seq-len 512 --boundary-aware-pack \
  --batch-plan-steps 2 --model-step 650

planning_elapsed=48.7s records_scanned=4,437 rendered_examples=4,368 rendered_examples/sec=89.6
overall rows/step=68.0 examples/step=272.0 dropped/step=0.0 tokens/step=31631/31960 pad=1.0% attn_pairs/step=1.86M
bucket | steps | rows avg/min/max | examples avg/dropped | target_rows | avg_fill | tokens/step useful/padded | pad | attn_pairs/step
   460 |     1 |  68.0/ 68/68  |  272.0/0.0     |          71 |    95.8% |   30803/31280   |   1.5% |           1.76M
   480 |     1 |  68.0/ 68/68  |  272.0/0.0     |          68 |   100.0% |   32459/32640   |   0.6% |           1.95M
```

Follow-up packed selector fix: streamed packed batches now treat
`--device-batch-size` as a candidate window, not a hard count of examples that
must be consumed. The selector runs the same packer used by training against the
rendered buffer, removes only examples that fit `--max-batch-tokens`, and leaves
non-fitting candidates in the buffer. That makes the original 512-example packed
probe shape viable without the old dropped-example artifact.

Normal batch-plan CLI verification with pack-aware stream selection:

```text
uv run --extra vision python -m scripts.vlm_train \
  --hf-repo HuggingFaceM4/the_cauldron --hf-config vqav2 \
  --device-batch-size 512 --max-batch-tokens 32768 --max-seq-len 512 \
  --stream-buffer-size 256 --batch-buffer-size 4096 \
  --pack-examples 8 --pack-max-seq-len 512 --boundary-aware-pack \
  --batch-plan-steps 2 --model-step 650

planning_elapsed=50.0s records_scanned=4,436 rendered_examples=4,368 rendered_examples/sec=87.4
overall rows/step=69.0 examples/step=276.0 dropped/step=0.0 tokens/step=32346/32700 pad=1.1% attn_pairs/step=1.91M
bucket | steps | rows avg/min/max | examples avg/dropped | target_rows | avg_fill | tokens/step useful/padded | pad | attn_pairs/step
   468 |     1 |  70.0/ 70/70  |  280.0/0.0     |          70 |   100.0% |   32382/32760   |   1.2% |           1.89M
   480 |     1 |  68.0/ 68/68  |  272.0/0.0     |          68 |   100.0% |   32311/32640   |   1.0% |           1.94M
```

Conclusion: `device_batch_size=512` is again the stronger first H100
packed-varlen default once selection is pack-aware. It keeps the 32K packed-token
cap essentially full, processes slightly more useful tokens than the manually
tuned 272-selector run, and reports zero dropped selected examples because
unpacked candidates stay in the rendered buffer instead of being discarded.
The streaming training path also skips the old `trim_examples_to_packable`
repack after pack-aware selection; materialized/debug data still uses the trim
path. This removes one redundant CPU packing pass before image open/processor
work on the packed H100 probe.

Packed `--bucket-selection max-tokens` now evaluates packed candidate windows
and chooses the window with the most useful packed tokens. This is only wired as
the packed MFU probe default; normal training can still use sampled windows for
more ordinary data mixing.
The packer length update was also made incremental instead of recomputing the
packed length from every row member on each placement trial. This preserves the
same greedy row placement while reducing CPU work during candidate-window
selection and row construction.

CPU-only preflight with pack-aware max-token selection:

```text
uv run --extra vision python -m scripts.vlm_train \
  --hf-repo HuggingFaceM4/the_cauldron --hf-config vqav2 \
  --device-batch-size 512 --max-batch-tokens 32768 --max-seq-len 512 \
  --stream-buffer-size 256 --batch-buffer-size 4096 \
  --pack-examples 8 --pack-max-seq-len 512 --boundary-aware-pack \
  --bucket-selection max-tokens \
  --batch-plan-steps 2 --model-step 650

planning_elapsed=50.4s records_scanned=4,444 rendered_examples=4,376 rendered_examples/sec=86.8
overall rows/step=69.5 examples/step=278.0 dropped/step=0.0 tokens/step=32398/32664 pad=0.8% attn_pairs/step=1.90M
bucket | steps | rows avg/min/max | examples avg/dropped | target_rows | avg_fill | tokens/step useful/padded | pad | attn_pairs/step
   468 |     1 |  70.0/ 70/70  |  280.0/0.0     |          70 |   100.0% |   32420/32760   |   1.0% |           1.89M
   472 |     1 |  69.0/ 69/69  |  276.0/0.0     |          69 |   100.0% |   32375/32568   |   0.6% |           1.92M
```

Follow-up cleanup: sequence-aware MFU diagnostics now cache FLOPs/token estimates
by sequence length. This avoids walking model parameters every microbatch while
still logging `seq_*` MFU fields for each static bucket.

Validation after wiring the bucketed probe entrypoint:

```text
uv run python -m py_compile nanochat/gpt.py nanochat/vision.py nanochat/flash_attention.py scripts/vlm_train.py modal_vlm.py
ok

uv run python -m pytest tests/test_vision.py -q
41 passed

uv run python -m pytest -q
55 passed, 10 skipped
```

CPU-only preflight for the new 128-512 bucket recipe:

```text
uv run --extra vision python -m scripts.vlm_train \
  --hf-repo HuggingFaceM4/the_cauldron --hf-config vqav2 \
  --device-batch-size 512 --max-batch-tokens 21504 --max-seq-len 512 \
  --stream-buffer-size 256 --batch-buffer-size 2048 \
  --bucket-selection cycle --bucket-min-fill-frac 0.75 \
  --pad-to-bucket-lens 128,192,256,384,512 \
  --batch-plan-steps 4 --model-step 650

planning_elapsed=39.2s records_scanned=2,444 rendered_examples=2,412 rendered_examples/sec=61.6
overall rows/step=105.0 examples/step=105.0 dropped/step=0.0 tokens/step=20087/21504 pad=6.6% attn_pairs/step=2.59M
bucket | steps | rows avg/min/max | examples avg/dropped | target_rows | avg_fill | tokens/step useful/padded | pad | attn_pairs/step
   128 |     1 | 168.0/168/168 |  168.0/0.0     |         168 |   100.0% |   21075/21504   |   2.0% |           1.39M
   192 |     1 | 112.0/112/112 |  112.0/0.0     |         112 |   100.0% |   20397/21504   |   5.1% |           2.08M
   256 |     1 |  84.0/ 84/84  |   84.0/0.0     |          84 |   100.0% |   20050/21504   |   6.8% |           2.76M
   384 |     1 |  56.0/ 56/56  |   56.0/0.0     |          56 |   100.0% |   18827/21504   |  12.4% |           4.14M
```

This confirms the realistic recipe does not depend on the 96-token diagnostic
bucket. The first four cycled buckets fill their token-capped row counts with no
dropped examples; a longer H100 probe should use per-bucket MFU to determine
whether 384/512 sequence buckets are the next real compute bottleneck.

Added `--bucket-cycle-repeat` for `--bucket-selection cycle`. This is primarily
for bucketed MFU probes with gradient accumulation: prefetch prepares
microbatches ahead of the training loop, so repeating each selected bucket N
times in the selector is the simple way to keep all N accumulation microbatches
in one optimizer step on the same static bucket. `bucketed_mfu_probe` now exposes
`grad_accum_steps`; when it is greater than 1, the command builder sets
`--bucket-cycle-repeat` to the same value unless explicitly overridden. This
keeps `bucket_steady_mfu` and `bucket_steady_seq_padded_mfu` valid for optimizer
amortization experiments.

CPU-only preflight for the grad-accumulation bucket repeat path:

```text
uv run --extra vision python -m scripts.vlm_train \
  --hf-repo HuggingFaceM4/the_cauldron --hf-config vqav2 \
  --device-batch-size 512 --grad-accum-steps 2 \
  --max-batch-tokens 21504 --max-seq-len 512 \
  --stream-buffer-size 256 --batch-buffer-size 4096 \
  --bucket-selection cycle --bucket-min-fill-frac 0.75 \
  --bucket-cycle-repeat 2 \
  --pad-to-bucket-lens 128,192,256,384,512 \
  --batch-plan-steps 6 --model-step 650

planning_elapsed=64.4s records_scanned=4,821 rendered_examples=4,740 rendered_examples/sec=73.6
overall rows/step=121.3 examples/step=121.3 dropped/step=0.0 tokens/step=20503/21504 pad=4.7% attn_pairs/step=2.08M
bucket | steps | rows avg/min/max | examples avg/dropped | target_rows | avg_fill | tokens/step useful/padded | pad | attn_pairs/step
   128 |     2 | 168.0/168/168 |  168.0/0.0     |         168 |   100.0% |   21116/21504   |   1.8% |           1.39M
   192 |     2 | 112.0/112/112 |  112.0/0.0     |         112 |   100.0% |   20490/21504   |   4.7% |           2.08M
   256 |     2 |  84.0/ 84/84  |   84.0/0.0     |          84 |   100.0% |   19904/21504   |   7.4% |           2.76M
```

This verifies that a future `bucketed_mfu_probe --grad-accum-steps 2` can
amortize optimizer work while still producing same-bucket optimizer steps for
per-bucket MFU diagnostics.

Follow-up formatter cleanup: `--batch-plan-steps` now reports optimizer-step
groups when `--grad-accum-steps > 1`, so the dry run matches the unit of MFU
logging. This prevents confusing a good microbatch plan with a mixed-shape
optimizer step.

```text
uv run --extra vision python -m scripts.vlm_train \
  --hf-repo HuggingFaceM4/the_cauldron --hf-config vqav2 \
  --device-batch-size 512 --grad-accum-steps 2 \
  --max-batch-tokens 21504 --max-seq-len 512 \
  --stream-buffer-size 256 --batch-buffer-size 4096 \
  --bucket-selection cycle --bucket-min-fill-frac 0.75 \
  --bucket-cycle-repeat 2 \
  --pad-to-bucket-lens 128,192,256,384,512 \
  --batch-plan-steps 4 --model-step 650

planning_elapsed=64.8s records_scanned=4,619 rendered_examples=4,544 rendered_examples/sec=70.1
overall rows/step=140.0 examples/step=140.0 dropped/step=0.0 tokens/step=20803/21504 pad=3.3% attn_pairs/step=1.73M
optimizer_steps complete=2/2 incomplete=0 same_bucket=2 mixed_bucket=0 tokens/optimizer_step=41606/43008 pad=3.3% attn_pairs/optimizer_step=3.46M
optimizer bucket | steps | rows/step | examples/step | tokens/optimizer_step useful/padded | pad | attn_pairs/optimizer_step
             128 |     1 |     336.0 |         336.0 |   42231/43008   |   1.8% |                    2.77M
             192 |     1 |     224.0 |         224.0 |   40980/43008   |   4.7% |                    4.15M
```

Validation after adding `--bucket-cycle-repeat`:

```text
uv run python -m py_compile nanochat/gpt.py nanochat/vision.py nanochat/flash_attention.py scripts/vlm_train.py modal_vlm.py
ok

uv run python -m pytest -q
56 passed, 10 skipped
```

Added optimizer timing split for the next H100 profile. The existing `optim`
field remains the total optimizer step time for backward comparison, and
`--profile-timing` now appends `projector_optim` and `llm_optim` to the printed
timing line. These are also logged as `timing/optim_projector_sec` and
`timing/optim_llm_sec`. This will show whether a grad-accumulation MFU change is
amortizing projector AdamW, the LLM Muon/AdamW optimizer, or neither.

Hot-path cleanup: `build_multimodal_batch` now materializes padded
`value_token_ids` and `targets` with one tensor creation each instead of creating
two small device tensors per row while filling the batch. The semantics are
unchanged, but bucketed H100 probes with 100-200 rows avoid hundreds of tiny
row-level tensor creations before the single vectorized embedding lookup.
The same path now writes projected visual-token embeddings with one advanced
indexing assignment across all image spans instead of one slice assignment per
image. A packed-row unit test verifies two images preserve feature ordering
exactly.
Static bucket padding also moved into `build_multimodal_batch`: after the
builder computes expanded row lengths, it chooses the bucket length directly.
The trainer no longer scans packed rows once to compute `pad_to_len` and then
again inside the batch builder to expand them.
The full `input_embeds` clone before image insertion was removed as well:
projected image features are written directly into the embedding output. A
gradient test checks that the inserted image span still backpropagates into the
projector, and the VLM smoke test exercises the actual training-loop backward
path.

Timing-path cleanup: the trainer now synchronizes once before entering the
training loop, then uses the per-step loss scalar read to synchronize before
measuring `dt`. This removes the redundant explicit start/end CUDA
synchronizations around every optimizer step while preserving warmup-excluded
MFU timing semantics.

Validation after optimizer timing split and multimodal batch materialization
cleanup:

```text
uv run python -m py_compile nanochat/gpt.py nanochat/vision.py nanochat/flash_attention.py scripts/vlm_train.py modal_vlm.py
ok

uv run python -m pytest -q
58 passed, 10 skipped
```

Validation after vectorized visual-token writes:

```text
uv run python -m py_compile nanochat/gpt.py nanochat/vision.py nanochat/flash_attention.py scripts/vlm_train.py modal_vlm.py
ok

uv run python -m pytest -q
58 passed, 10 skipped
```

Validation after moving static bucket padding into `build_multimodal_batch`:

```text
uv run python -m py_compile nanochat/gpt.py nanochat/vision.py nanochat/flash_attention.py scripts/vlm_train.py modal_vlm.py
ok

uv run python -m pytest -q
59 passed, 10 skipped
```

Validation after removing the full embedding clone before image insertion:

```text
uv run python -m pytest tests/test_vision.py tests/test_vlm_smoke.py -q
47 passed

uv run python -m py_compile nanochat/gpt.py nanochat/vision.py nanochat/flash_attention.py scripts/vlm_train.py modal_vlm.py
ok

uv run python -m pytest -q
60 passed, 10 skipped
```

Validation after removing redundant per-step timing synchronizations:

```text
uv run python -m py_compile nanochat/gpt.py nanochat/vision.py nanochat/flash_attention.py scripts/vlm_train.py modal_vlm.py
ok

uv run python -m pytest -q
60 passed, 10 skipped
```

Packed selector cleanup: `--bucket-selection max-tokens` still chooses the
packed candidate window with the most useful processed tokens first, but its
tie-breaker now uses the same attention-pair estimate as the boundary-aware
varlen training path. For `--boundary-aware-pack`, that means original segment
lengths rather than packed row lengths; for loose packing, it remains dense row
attention. This avoids preferring a window with the same useful/padded token
count but much worse segment-level causal attention work.

Focused validation:

```text
uv run python -m pytest tests/test_vision.py::test_stream_packed_max_token_selection_tiebreaks_by_segment_attention tests/test_vision.py::test_stream_packed_selection_can_choose_max_token_window -q
2 passed

uv run python -m py_compile scripts/vlm_train.py tests/test_vision.py
ok

uv run python -m py_compile nanochat/gpt.py nanochat/vision.py nanochat/flash_attention.py scripts/vlm_train.py modal_vlm.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
64 passed, 10 skipped

uv run python -m pytest -q
72 passed, 10 skipped
```

Added compact flattened boundary-aware packing behind
`--flatten-packed-batch`. The selector still builds ordinary packed rows to stay
under `--pack-max-seq-len` and `--max-batch-tokens`, but after image encoding the
training batch concatenates those rows into one compact varlen sequence. For the
boundary-aware path this preserves per-example `cu_seqlens`, RoPE reset, and
smear reset, while avoiding per-layer q/k/v gather and output scatter when the
tokens are already in varlen order. It also removes the remaining packed-row
padding from the LLM input.

CPU-only compact packed preflight:

```text
uv run --extra vision python -m scripts.vlm_train \
  --hf-repo HuggingFaceM4/the_cauldron --hf-config vqav2 \
  --device-batch-size 512 --max-batch-tokens 32768 --max-seq-len 512 \
  --stream-buffer-size 256 --batch-buffer-size 4096 \
  --pack-examples 8 --pack-max-seq-len 512 --boundary-aware-pack \
  --flatten-packed-batch --bucket-selection max-tokens \
  --batch-plan-steps 2 --model-step 650

planning_elapsed=46.7s records_scanned=4,444 rendered_examples=4,376 rendered_examples/sec=93.7
overall rows/step=1.0 examples/step=278.0 dropped/step=0.0 tokens/step=32398/32398 pad=0.0% attn_pairs/step=1.90M
bucket | steps | rows avg/min/max | examples avg/dropped | target_rows | avg_fill | tokens/step useful/padded | pad | attn_pairs/step
 32375 |     1 |   1.0/  1/1   |  276.0/0.0     |           1 |   100.0% |   32375/32375   |   0.0% |           1.92M
 32420 |     1 |   1.0/  1/1   |  280.0/0.0     |           1 |   100.0% |   32420/32420   |   0.0% |           1.89M
```

Validation after compact packed mode:

```text
uv run python -m py_compile nanochat/gpt.py nanochat/vision.py nanochat/flash_attention.py scripts/vlm_train.py modal_vlm.py tests/test_vision.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
66 passed, 10 skipped

uv run python -m pytest -q
74 passed, 10 skipped
```

Follow-up compact selector cleanup: once the final batch is flattened, the real
token budget is total useful tokens rather than dense packed-row padding. The
packer and stream selector now use that compact budget under
`--flatten-packed-batch`. A naive exact-fill selector briefly reached
`tokens/step=32762/32762`, but did so by picking much longer segments and raised
`attn_pairs/step` to 2.81M. The final selector treats compact fills within a
512-token band as equivalent, then picks the lower segment-attention window.

CPU-only compact packed preflight after the compact-budget and attention-aware
fill-band selector:

```text
uv run --extra vision python -m scripts.vlm_train \
  --hf-repo HuggingFaceM4/the_cauldron --hf-config vqav2 \
  --device-batch-size 512 --max-batch-tokens 32768 --max-seq-len 512 \
  --stream-buffer-size 256 --batch-buffer-size 4096 \
  --pack-examples 8 --pack-max-seq-len 512 --boundary-aware-pack \
  --flatten-packed-batch --bucket-selection max-tokens \
  --batch-plan-steps 2 --model-step 650

planning_elapsed=57.5s records_scanned=4,447 rendered_examples=4,379 rendered_examples/sec=76.2
overall rows/step=1.0 examples/step=281.0 dropped/step=0.0 tokens/step=32747/32747 pad=0.0% attn_pairs/step=1.92M
bucket | steps | rows avg/min/max | examples avg/dropped | target_rows | avg_fill | tokens/step useful/padded | pad | attn_pairs/step
 32729 |     1 |   1.0/  1/1   |  279.0/0.0     |           1 |   100.0% |   32729/32729   |   0.0% |           1.94M
 32765 |     1 |   1.0/  1/1   |  283.0/0.0     |           1 |   100.0% |   32765/32765   |   0.0% |           1.91M
```

Added segment diagnostics to the batch planner and train logs so compact varlen
runs report the flattened row length separately from the real attention
segment lengths. The same compact preflight now prints:

```text
overall rows/step=1.0 examples/step=281.0 dropped/step=0.0 tokens/step=32747/32747 pad=0.0% attn_pairs/step=1.92M segments/step=281.0 max_segment=118
bucket | steps | rows avg/min/max | examples avg/dropped | segments avg/max | target_rows | avg_fill | tokens/step useful/padded | pad | attn_pairs/step
 32729 |     1 |   1.0/  1/1   |  279.0/0.0     |  279.0/118     |           1 |   100.0% |   32729/32729   |   0.0% |           1.94M
 32765 |     1 |   1.0/  1/1   |  283.0/0.0     |  283.0/117     |           1 |   100.0% |   32765/32765   |   0.0% |           1.91M
```

This matters for H100 interpretation: `max_seq` in the training line will be the
compact flattened length, while `max_segment` is the attention kernel's maximum
causal sequence length.

Focused validation after compact-budget selection:

```text
uv run python -m pytest tests/test_vision.py::test_compact_packed_max_token_selection_prefers_lower_attention_near_full tests/test_vision.py::test_compact_packed_budget_uses_total_tokens_not_dense_rows tests/test_vision.py::test_stream_packed_max_token_selection_tiebreaks_by_segment_attention -q
3 passed

uv run python -m py_compile nanochat/gpt.py nanochat/vision.py nanochat/flash_attention.py scripts/vlm_train.py modal_vlm.py tests/test_vision.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
68 passed, 10 skipped

uv run python -m pytest -q
76 passed, 10 skipped
```

Packed probe realism update: `96` was only a short-bucket diagnostic, and even
the compact packed `max_segment=118` should not be interpreted as a dataset cap.
The dedicated packed MFU probe now defaults to `--max-seq-len 1024` and
`--pack-max-seq-len 1024` while keeping `--max-batch-tokens 32768`,
`--boundary-aware-pack`, and `--flatten-packed-batch`. This admits longer
examples for realistic probes/training without changing the compact useful-token
budget.

CPU-only compact packed preflight with the 1024-token cap:

```text
uv run --extra vision python -m scripts.vlm_train \
  --hf-repo HuggingFaceM4/the_cauldron --hf-config vqav2 \
  --device-batch-size 512 --max-batch-tokens 32768 --max-seq-len 1024 \
  --stream-buffer-size 256 --batch-buffer-size 4096 \
  --pack-examples 8 --pack-max-seq-len 1024 --boundary-aware-pack \
  --flatten-packed-batch --bucket-selection max-tokens \
  --batch-plan-steps 2 --model-step 650

planning_elapsed=53.2s records_scanned=4,382 rendered_examples=4,379 rendered_examples/sec=82.2
overall rows/step=1.0 examples/step=281.0 dropped/step=0.0 tokens/step=32752/32752 pad=0.0% attn_pairs/step=1.93M segments/step=281.0 max_segment=118
bucket | steps | rows avg/min/max | examples avg/dropped | segments avg/max | target_rows | avg_fill | tokens/step useful/padded | pad | attn_pairs/step
 32735 |     1 |   1.0/  1/1   |  279.0/0.0     |  279.0/118     |           1 |   100.0% |   32735/32735   |   0.0% |           1.94M
 32768 |     1 |   1.0/  1/1   |  283.0/0.0     |  283.0/117     |           1 |   100.0% |   32768/32768   |   0.0% |           1.91M
```

Compact training-loop cleanup: the flattened boundary-aware path now bypasses
the second intermediate packed-row construction when `--pack-fixed-rows` is not
used. Selected examples are concatenated directly into the single compact varlen
row, each original example remains its own segment, and in-order image features
are reused without an extra `index_select`. The older packed-row path remains
for padded row-major varlen coverage and explicit fixed-row experiments.
Added a model-level regression test that compares direct compact batches against
the old repack-then-flatten path and verifies the boundary-aware varlen loss
matches.
Added a compact budget-agreement test so the selector's useful-token cap and
the direct compact constructor cannot drift and silently drop examples after
selection.
The direct compact regression also asserts `sum(segment_lengths)` equals useful
tokens and `attention_pairs` equals the causal segment-pair sum, matching the
fields used by the training loop for varlen FLOP/MFU diagnostics.
Packed-row and direct-compact feature selection now use per-example image
feature spans instead of assuming one feature row per example. This preserves
feature order for packed rows that contain examples with more than one image
marker, while keeping the one-image Cauldron path unchanged.
The Modal `train`, `mfu_probe`, and `packed_mfu_probe` entrypoints now expose
`flatten_packed_batch` directly, so the compact path and its ablation switch are
reachable from Modal instead of only from the command builders.

Compact hot-path metadata cleanup: compact varlen batches no longer materialize
the dense `segment_ids` tensor. Varlen attention uses `cu_seqlens`, RoPE uses
`position_ids`, and smear reset uses `segment_starts`, so `segment_ids` was only
extra metadata in the compact H100 path. The padded row-major correctness path
still returns `segment_ids` for diagnostics and dense fallback coverage.

Current validation after direct compact cleanup:

```text
uv run python -m pytest tests/test_vision.py::test_direct_compact_batch_loss_matches_repack_then_flatten_path tests/test_vision.py::test_flatten_examples_as_compact_batch_keeps_segments_and_feature_order tests/test_vision.py::test_compact_boundary_aware_packed_batch_matches_padded_rows -q
3 passed

uv run python -m py_compile nanochat/gpt.py nanochat/vision.py nanochat/flash_attention.py scripts/vlm_train.py modal_vlm.py tests/test_vision.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
72 passed, 10 skipped

uv run python -m pytest -q
80 passed, 10 skipped
```

Local follow-up while Modal is paused: added an opt-in chunked full-CE path via
`--loss-chunk-size`. This is for static/bucketed compile probes that want
`--no-selective-loss` semantics without materializing the entire `[B,T,V]`
logits tensor. The path keeps the existing nanochat softcap and PyTorch
ignore-index CE behavior; it only chunks the lm-head/loss computation. Defaults
remain unchanged, and the compact packed probe still uses selective VLM loss.
The Modal `train`, `mfu_probe`, `bucketed_mfu_probe`, and `packed_mfu_probe`
wrappers now expose the flag, with `bucketed_mfu_probe --loss-chunk-size 4096`
as the intended OOM ablation for static full-CE H100 runs.

Also extended the compact budget guard so `trim_examples_to_packable` and the
direct compact constructor agree on which materialized examples survive a
compact useful-token cap before image processing.

Validation after chunked full-CE and compact trim guard:

```text
uv run python -m pytest tests/test_vision.py::test_gpt_selective_loss_matches_ignore_index_path tests/test_vision.py::test_direct_compact_batch_matches_compact_packer_budget tests/test_vision.py::test_modal_command_builders -q
3 passed

uv run python -m py_compile nanochat/gpt.py scripts/vlm_train.py modal_vlm.py tests/test_vision.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
72 passed, 10 skipped

uv run python -m pytest -q
80 passed, 10 skipped

git diff --check
ok
```

2026-05-26 local batch-plan segment percentile diagnostics:

- Batch-plan rows now carry the exact expanded segment lengths used for
  boundary-aware attention accounting, and `format_batch_plan` reports
  `p50_segment` and `p90_segment` alongside `avg_segment` and `max_segment`.
- This directly addresses the 96-token realism concern: future 32K/65K CPU
  preflights and Modal probes can show whether a high max segment is just one
  outlier or whether the median/p90 selected examples are genuinely longer.
- The runbook now asks H100 probe comparisons to include
  `avg_segment`/`p50_segment`/`p90_segment`/`max_segment`, so MFU evidence is
  tied to the actual selected sequence distribution.

Validation after segment percentile diagnostics:

```text
uv run python -m pytest tests/test_vision.py::test_batch_plan_summary_reports_bucket_fill_and_padding tests/test_vision.py::test_batch_plan_row_models_packed_examples tests/test_vision.py::test_direct_compact_plan_matches_flattened_runtime_after_skips -q
3 passed

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
98 passed, 10 skipped

uv run python -m py_compile nanochat/gpt.py nanochat/vision.py nanochat/flash_attention.py scripts/vlm_train.py modal_vlm.py tests/test_vision.py
ok

uv run python -m pytest -q
106 passed, 10 skipped

git diff --check
ok
```

2026-05-26 local training-log segment percentile diagnostics:

- Extended the actual training metrics path to report step, steady, and
  bucket-steady segment percentiles, not only CPU batch-plan percentiles.
- The stdout line now includes `p50_segment`, `p90_segment`,
  `steady_p50_segment`, and `steady_p90_segment`; W&B gets
  `train/p50_segment_len`, `train/p90_segment_len`,
  `train/steady_p50_segment_len`, `train/steady_p90_segment_len`,
  `train/bucket_steady_p50_segment_len`, and
  `train/bucket_steady_p90_segment_len`.
- Steady percentiles use a compact length-count histogram, so longer training
  runs do not retain every segment length in memory. This keeps future H100
  `steady_mfu` evidence tied to the actual selected sequence distribution.

Validation after training-log segment percentiles:

```text
uv run python -m pytest tests/test_vision.py::test_bucket_steady_metrics_accumulate_by_static_shape tests/test_vision.py::test_batch_plan_summary_reports_bucket_fill_and_padding -q
2 passed

uv run python -m py_compile scripts/vlm_train.py tests/test_vision.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
98 passed, 10 skipped

uv run python -m py_compile nanochat/gpt.py nanochat/vision.py nanochat/flash_attention.py scripts/vlm_train.py modal_vlm.py tests/test_vision.py
ok

uv run python -m pytest -q
106 passed, 10 skipped

git diff --check
ok
```

2026-05-26 local compact-varlen MFU bucket guard:

- Added `static_mfu_step_bucket(...)` so compact varlen probes do not treat each
  slightly different flattened row length as a static bucket. This prevents
  `--mfu-warmup-bucket-steps` from accidentally excluding every compact packed
  step when total useful tokens vary by a few tokens around 32K/65K.
- Dense/static bucketed probes still use per-bucket warmup, which is where
  first-use compile/setup cost matters.
- The runbook now calls out that compact varlen probes should compare global
  warmup-excluded `steady_mfu` plus segment percentiles, not fake per-length
  bucket stats.

Validation after compact-varlen MFU bucket guard:

```text
uv run python -m pytest tests/test_vision.py::test_mfu_step_counting_respects_global_and_bucket_warmups tests/test_vision.py::test_compact_varlen_steps_do_not_create_static_mfu_buckets tests/test_vision.py::test_bucket_steady_metrics_accumulate_by_static_shape -q
3 passed

uv run python -m py_compile scripts/vlm_train.py tests/test_vision.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
99 passed, 10 skipped

uv run python -m py_compile nanochat/gpt.py nanochat/vision.py nanochat/flash_attention.py scripts/vlm_train.py modal_vlm.py tests/test_vision.py
ok

uv run python -m pytest -q
107 passed, 10 skipped

git diff --check
ok
```

2026-05-26 local final bucket-summary segment percentile wiring:

- The final `Bucket steady stats` block now uses a tested
  `format_bucket_steady_line(...)` helper and prints `p50_segment` and
  `p90_segment` next to `avg_segment`/`max_segment`.
- This matters for static bucketed H100 probes: the end-of-run summary copied
  into notes now carries the same segment-realism evidence as the per-step logs
  and W&B metrics.

Validation after final bucket-summary percentile wiring:

```text
uv run python -m pytest tests/test_vision.py::test_bucket_steady_metrics_accumulate_by_static_shape -q
1 passed

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
99 passed, 10 skipped

uv run python -m py_compile nanochat/gpt.py nanochat/vision.py nanochat/flash_attention.py scripts/vlm_train.py modal_vlm.py tests/test_vision.py
ok

uv run python -m pytest -q
107 passed, 10 skipped

git diff --check
ok
```

2026-05-26 local compact visual-token scatter cleanup:

- Replaced the compact B=1 visual-token insertion path in
  `build_multimodal_batch` with a flat `index_copy_` over the image-token
  positions. This avoids the 2D advanced-indexing scatter in the exact compact
  varlen shape used by the 32K/65K packed probes.
- The change preserves projector gradients, feature ordering, compact boundary
  metadata, and packed-vs-separate loss equivalence.

Validation after compact visual-token scatter cleanup:

```text
uv run python -m pytest tests/test_vision.py::test_multimodal_batch_backward_reaches_projector_after_image_insert tests/test_vision.py::test_direct_compact_batch_loss_matches_separate_examples tests/test_vision.py::test_compact_boundary_aware_packed_batch_matches_padded_rows tests/test_vision.py::test_flatten_examples_as_compact_batch_keeps_segments_and_feature_order -q
4 passed

uv run python -m py_compile nanochat/vision.py tests/test_vision.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
99 passed, 10 skipped

uv run python -m py_compile nanochat/gpt.py nanochat/vision.py nanochat/flash_attention.py scripts/vlm_train.py modal_vlm.py tests/test_vision.py
ok

uv run python -m pytest -q
107 passed, 10 skipped

git diff --check
ok
```

2026-05-26 local real-data 65K max-compute p50/p90 batch-plan check:

- Reran the compute-heavy compact packed CPU preflight after adding segment
  percentiles to the batch-plan output. Modal stayed paused.
- This confirms the `max-compute` selector is not just one long outlier: the
  selected 65K step had `p50_segment=687`, `p90_segment=903`, and
  `max_segment=1024`.

```text
uv run python -m scripts.vlm_train --device-type cpu --hf-repo HuggingFaceM4/the_cauldron --hf-config vqav2 --device-batch-size 1024 --max-batch-tokens 65536 --max-seq-len 1024 --stream-buffer-size 256 --batch-buffer-size 8192 --bucket-selection max-compute --pack-examples 16 --pack-max-seq-len 1024 --boundary-aware-pack --flatten-packed-batch --batch-plan-steps 1 --model-step 650

Batch plan source=stream:HuggingFaceM4/the_cauldron/vqav2
steps=1 batch_size=1,024 max_batch_tokens=65,536 grad_accum_steps=1 bucket_lens=none bucket_selection=max-compute bucket_min_fill_frac=0 bucket_cycle_repeat=1 pack_examples=16 boundary_aware_pack=True flatten_packed_batch=True
planning_elapsed=120.3s records_scanned=8,203 rendered_examples=8,192 rendered_examples/sec=68.1
overall rows/step=1.0 examples/step=91.0 dropped/step=0.0 tokens/step=65389/65389 pad=0.0% attn_pairs/step=24.18M segments/step=91.0 avg_segment=718.6 p50_segment=687 p90_segment=903 max_segment=1024 near_cap/step=4.0 cap_hits/step=1.0
bucket | steps | rows avg/min/max | examples avg/dropped | segments avg | segment_len avg/p50/p90/max | near_cap/cap avg | target_rows | avg_fill | tokens/step useful/padded | pad | attn_pairs/step
 65389 |     1 |   1.0/  1/1   |   91.0/0.0     |         91.0 |     718.6/687/903/1024    |     4.0/1.0     |           1 |   100.0% |   65389/65389   |   0.0% |          24.18M
```

2026-05-26 local real-data 65K leaky p50/p90 batch-plan check:

- Reran the semantics-relaxed nanoVLM-style leaky packed CPU preflight with the
  same p50/p90 diagnostics. Modal stayed paused.
- This gives the fallback H100 ablation a directly comparable segment
  distribution: dense 1024-token rows, `p50_segment=993`, `p90_segment=1014`,
  and 33.59M attention pairs/step.
- This remains a diagnostic only; it uses `--allow-leaky-pack` and does not
  preserve strict cross-example attention boundaries.

```text
uv run python -m scripts.vlm_train --device-type cpu --hf-repo HuggingFaceM4/the_cauldron --hf-config vqav2 --device-batch-size 1024 --max-batch-tokens 65536 --max-seq-len 1024 --stream-buffer-size 256 --batch-buffer-size 8192 --bucket-selection max-tokens --pack-examples 16 --pack-max-seq-len 1024 --allow-leaky-pack --pad-to-bucket-lens 1024 --batch-plan-steps 1 --model-step 650

Batch plan source=stream:HuggingFaceM4/the_cauldron/vqav2
steps=1 batch_size=1,024 max_batch_tokens=65,536 grad_accum_steps=1 bucket_lens=[1024] bucket_selection=max-tokens bucket_min_fill_frac=0 bucket_cycle_repeat=1 pack_examples=16 boundary_aware_pack=False flatten_packed_batch=False
planning_elapsed=120.4s records_scanned=8,203 rendered_examples=8,192 rendered_examples/sec=68.1
overall rows/step=64.0 examples/step=448.0 dropped/step=0.0 tokens/step=63768/65536 pad=2.7% attn_pairs/step=33.59M segments/step=64.0 avg_segment=996.4 p50_segment=993 p90_segment=1014 max_segment=1021 near_cap/step=64.0 cap_hits/step=0.0
bucket | steps | rows avg/min/max | examples avg/dropped | segments avg | segment_len avg/p50/p90/max | near_cap/cap avg | target_rows | avg_fill | tokens/step useful/padded | pad | attn_pairs/step
  1024 |     1 |  64.0/ 64/64  |  448.0/0.0     |         64.0 |     996.4/993/1014/1021    |    64.0/0.0     |          64 |   100.0% |   63768/65536   |   2.7% |          33.59M
```

2026-05-26 local direct compact shape guard:

- Added `test_direct_compact_plan_matches_flattened_runtime_after_skips`.
  This covers the realistic compact varlen path after the 96-token diagnostic
  discussion: an over-length example is skipped, a later example is skipped by
  the compact useful-token budget, selected image features keep their original
  spans, and `batch_plan_row` reports the same expanded segment lengths,
  useful tokens, and boundary-aware attention pairs that
  `build_multimodal_batch(..., compact_varlen_indices=True)` actually sends to
  the model.
- This is a guard against validating future 32K/65K packed probes with a
  virtual repacked shape. The plan now has a test tying CPU preflight accounting
  to the flattened runtime batch representation.

Validation after direct compact shape guard:

```text
uv run python -m pytest tests/test_vision.py::test_direct_compact_plan_matches_flattened_runtime_after_skips tests/test_vision.py::test_compact_trim_preserves_selected_order_without_virtual_repack tests/test_vision.py::test_direct_compact_batch_matches_compact_packer_budget -q
3 passed

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
98 passed, 10 skipped

uv run python -m py_compile nanochat/gpt.py nanochat/vision.py nanochat/flash_attention.py scripts/vlm_train.py modal_vlm.py tests/test_vision.py
ok

uv run python -m pytest -q
106 passed, 10 skipped

git diff --check
ok
```

2026-05-26 local max-compute packed selector ablation:

- Added `--bucket-selection max-compute` for packed/bucketed selection. For
  compact packed batches it keeps the same near-full useful-token fill bands as
  `max-tokens`, but when candidate windows are similarly full it prefers more
  segment attention work instead of less. The intent is to test a more
  text-like LLM workload per frozen SigLIP/image example, without replacing the
  existing max-token stress probe or the random representative probe.
- CPU-only real-data batch plan on `HuggingFaceM4/the_cauldron/vqav2` with the
  same large H100 shape filled the 65K useful-token budget with much longer
  segments than the max-token stress plan:

```text
uv run python -m scripts.vlm_train --device-type cpu --hf-repo HuggingFaceM4/the_cauldron --hf-config vqav2 --device-batch-size 1024 --max-batch-tokens 65536 --max-seq-len 1024 --stream-buffer-size 256 --batch-buffer-size 8192 --bucket-selection max-compute --pack-examples 16 --pack-max-seq-len 1024 --boundary-aware-pack --flatten-packed-batch --batch-plan-steps 2 --model-step 650

Batch plan source=stream:HuggingFaceM4/the_cauldron/vqav2
steps=2 batch_size=1,024 max_batch_tokens=65,536 grad_accum_steps=1 bucket_lens=none bucket_selection=max-compute bucket_min_fill_frac=0 bucket_cycle_repeat=1 pack_examples=16 boundary_aware_pack=True flatten_packed_batch=True
planning_elapsed=115.5s records_scanned=8,294 rendered_examples=8,283 rendered_examples/sec=71.7
overall rows/step=1.0 examples/step=116.0 dropped/step=0.0 tokens/step=65462/65462 pad=0.0% attn_pairs/step=19.80M segments/step=116.0 avg_segment=564.3 max_segment=1024 near_cap/step=2.0 cap_hits/step=0.5
```

- Compared with the prior 65K plans, `max-compute` sits between semantic
  compact packing and the leaky dense-row ablation: it is still boundary-aware
  and compact, but it trades fewer image examples/segments per step for about
  5x the attention pairs of the max-token stress selector. The future H100
  question is whether this extra LLM work and lower image/segment count improve
  warmup-excluded `steady_mfu`.
- Added named Modal wrappers for that exact future ablation:
  `packed_large_compute_batch_plan`, `packed_large_compute_mfu_probe`, and
  `packed_large_compute_profile_mfu_probe`. They force
  `--bucket-selection max-compute` while keeping the same 65K compact
  boundary-aware packed defaults as `packed_large_mfu_probe`.
- Simplified the default dynamic compact path so it trims selected examples
  directly by per-example `expanded_len` and total useful-token budget instead of
  doing a second virtual packed-row pass. Dense/static/fixed-row experiments
  still use the row packer. This keeps the H100 compact path closer to the
  actual varlen batch: selected examples become independent segments in one
  compact row, in selected order.

Validation after adding `max-compute`:

```text
uv run python -m pytest tests/test_vision.py::test_stream_packed_max_token_selection_tiebreaks_by_segment_attention tests/test_vision.py::test_stream_packed_max_compute_selection_prefers_more_attention_near_full tests/test_vision.py::test_modal_command_builders -q
3 passed

uv run python -m pytest tests/test_vision.py::test_compact_trim_preserves_selected_order_without_virtual_repack tests/test_vision.py::test_direct_compact_batch_matches_compact_packer_budget tests/test_vision.py::test_stream_packed_max_compute_selection_prefers_more_attention_near_full -q
3 passed

uv run python -m py_compile scripts/vlm_train.py tests/test_vision.py modal_vlm.py
ok

uv run python -m scripts.vlm_train --device-type cpu --attention-backend-report --bucket-selection max-compute
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
97 passed, 10 skipped

uv run python -m pytest -q
105 passed, 10 skipped

uv run python -m py_compile nanochat/gpt.py nanochat/vision.py nanochat/flash_attention.py scripts/vlm_train.py modal_vlm.py tests/test_vision.py
ok

git diff --check
ok
```

2026-05-26 packed-probe realism note:

- `96` expanded tokens is not a realistic global VLM training length. With one
  image fixed at 64 visual tokens, it leaves only about 32 non-image positions
  for role markers, prompt text, answer text, and EOS. It should remain a
  short-shape diagnostic bucket only.
- The current packed MFU path should be judged from the 1024-cap probes, not
  the old 96-token bucket. I added regression coverage that the large packed
  MFU probes and matching CPU batch-plan probes default to `--max-seq-len 1024`
  and `--pack-max-seq-len 1024`, with `--max-batch-tokens 65536`.
- The `vqav2` max-token selector can still produce short average segments
  because it deliberately fills the 65K useful-token budget with cheap examples.
  That is an MFU stress shape. The matching random-selector probe is the
  realism ablation; the local dry run saw `max_segment=1000`, confirming longer
  examples are admitted under the same 1024 cap.

Measurement-integrity cleanup: training now counts processed examples from
boundary segment metadata when it is available instead of using image marker
counts. This leaves the one-image Cauldron path unchanged, but keeps
`samples/sec` and `dropped_samples` honest for future packed rows containing
multi-image examples. Added a focused regression for this helper.

Validation after packed-sample accounting cleanup:

```text
uv run python -m pytest tests/test_vision.py::test_packed_example_count_uses_segments_not_image_markers tests/test_vision.py::test_flatten_examples_as_compact_batch_keeps_segments_and_feature_order -q
2 passed

uv run python -m py_compile nanochat/gpt.py nanochat/vision.py nanochat/flash_attention.py scripts/vlm_train.py modal_vlm.py tests/test_vision.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
73 passed, 10 skipped

uv run python -m pytest -q
81 passed, 10 skipped

git diff --check
ok
```

Segment-realism metric cleanup: batch-plan output and training logs now include
`avg_segment` in addition to segment count and `max_segment`. This makes future
H100 packed-MFU probes auditable: `max_seq` can be a 32K compact flattened row,
while `avg_segment`/`max_segment` show the actual per-example causal sequence
lengths seen by varlen attention. This directly distinguishes realistic longer
examples from a selector that filled the token budget mostly with short VQA
segments.

Validation after adding the average segment metric:

```text
uv run python -m pytest tests/test_vision.py::test_batch_plan_summary_reports_bucket_fill_and_padding tests/test_vision.py::test_packed_example_count_uses_segments_not_image_markers -q
2 passed

uv run python -m py_compile nanochat/gpt.py nanochat/vision.py nanochat/flash_attention.py scripts/vlm_train.py modal_vlm.py tests/test_vision.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
73 passed, 10 skipped

uv run python -m pytest -q
81 passed, 10 skipped

git diff --check
ok
```

CPU-only compact packed preflight after adding `avg_segment`:

```text
uv run --extra vision python -m scripts.vlm_train \
  --hf-repo HuggingFaceM4/the_cauldron --hf-config vqav2 \
  --device-batch-size 512 --max-batch-tokens 32768 --max-seq-len 1024 \
  --stream-buffer-size 256 --batch-buffer-size 4096 \
  --pack-examples 8 --pack-max-seq-len 1024 --boundary-aware-pack \
  --flatten-packed-batch --bucket-selection max-tokens \
  --batch-plan-steps 1 --model-step 650

planning_elapsed=46.8s records_scanned=4,099 rendered_examples=4,096 rendered_examples/sec=87.5
overall rows/step=1.0 examples/step=283.0 dropped/step=0.0 tokens/step=32768/32768 pad=0.0% attn_pairs/step=1.91M segments/step=283.0 avg_segment=115.8 max_segment=117
bucket | steps | rows avg/min/max | examples avg/dropped | segments avg | segment_len avg/max | target_rows | avg_fill | tokens/step useful/padded | pad | attn_pairs/step
 32768 |     1 |   1.0/  1/1   |  283.0/0.0     |        283.0 |     115.8/117     |           1 |   100.0% |   32768/32768   |   0.0% |           1.91M
```

Interpretation: the packed selector fills the 32K useful-token budget with zero
padding and no dropped examples, but this vqav2 probe is still dominated by
short segments. That is acceptable for a low-latency MFU smoke, but a final
realism claim needs either a mixed/full Cauldron preflight or an H100 profile
whose `avg_segment`/`max_segment` distribution is representative of the intended
training mixture.

Attempted a CPU-only `--hf-config all` compact packed preflight with
`--batch-buffer-size 2048`, but it did not produce a planned batch after more
than two minutes past tokenizer setup, matching the earlier warning that
all-config Cauldron can spend too long resolving/filling for quick MFU evidence.
Stopped it and probed named configs instead.

Named-config length scans at `--max-seq-len 1024`:

```text
localized_narratives: records_scanned=200 usable=200 elapsed=18.1s expanded_len min/p50/p80/p90/p95/p99/max/mean 86/116/131/139/148/166/169/118.2
textcaps: records_scanned=200 usable=200 elapsed=29.0s expanded_len min/p50/p80/p90/p95/p99/max/mean 83/91/94/97/99/103/109/91.3
docvqa: records_scanned=200 usable=200 elapsed=41.2s expanded_len min/p50/p80/p90/p95/p99/max/mean 84/149/225/259/293/342/360/168.1
```

`docvqa` is a better short local realism check than `vqav2`, but the packed
`max-tokens` selector still prefers shortish segments for throughput:

```text
uv run --extra vision python -m scripts.vlm_train \
  --hf-repo HuggingFaceM4/the_cauldron --hf-config docvqa \
  --device-batch-size 512 --max-batch-tokens 32768 --max-seq-len 1024 \
  --stream-buffer-size 256 --batch-buffer-size 1024 \
  --pack-examples 8 --pack-max-seq-len 1024 --boundary-aware-pack \
  --flatten-packed-batch --bucket-selection max-tokens \
  --batch-plan-steps 1 --model-step 650

planning_elapsed=116.9s records_scanned=1,024 rendered_examples=1,024 rendered_examples/sec=8.8
overall rows/step=1.0 examples/step=257.0 dropped/step=0.0 tokens/step=32689/32689 pad=0.0% attn_pairs/step=2.11M segments/step=257.0 avg_segment=127.2 max_segment=143
bucket | steps | rows avg/min/max | examples avg/dropped | segments avg | segment_len avg/max | target_rows | avg_fill | tokens/step useful/padded | pad | attn_pairs/step
 32689 |     1 |   1.0/  1/1   |  257.0/0.0     |        257.0 |     127.2/143     |           1 |   100.0% |   32689/32689   |   0.0% |           2.11M
```

Added a packed `--bucket-selection random` mode for the representative-selection
ablation. Unlike `sample`, which picks a contiguous length-sorted window, the
new mode samples across the rendered buffer under the same useful-token cap. On
the same `docvqa` preflight shape it still fills the compact batch, but keeps
the segment distribution much closer to the source length scan:

```text
uv run --extra vision python -m scripts.vlm_train \
  --hf-repo HuggingFaceM4/the_cauldron --hf-config docvqa \
  --device-batch-size 512 --max-batch-tokens 32768 --max-seq-len 1024 \
  --stream-buffer-size 256 --batch-buffer-size 1024 \
  --pack-examples 8 --pack-max-seq-len 1024 --boundary-aware-pack \
  --flatten-packed-batch --bucket-selection random \
  --batch-plan-steps 1 --model-step 650

planning_elapsed=121.4s records_scanned=1,024 rendered_examples=1,024 rendered_examples/sec=8.4
overall rows/step=1.0 examples/step=201.0 dropped/step=0.0 tokens/step=32754/32754 pad=0.0% attn_pairs/step=3.04M segments/step=201.0 avg_segment=163.0 max_segment=346
bucket | steps | rows avg/min/max | examples avg/dropped | segments avg | segment_len avg/max | target_rows | avg_fill | tokens/step useful/padded | pad | attn_pairs/step
 32754 |     1 |   1.0/  1/1   |  201.0/0.0     |        201.0 |     163.0/346     |           1 |   100.0% |   32754/32754   |   0.0% |           3.04M
```

Interpretation: compact packing itself is doing what we want for MFU, with both
throughput and representative selectors filling the useful-token budget without
padding or drops. `max-tokens` remains the right low-latency throughput smoke;
`random` is the better realism ablation and should be part of the H100 grid once
Modal is unpaused.

Added a named `packed_random_mfu_probe` Modal entrypoint and matching
`build_packed_random_mfu_probe_cmd()` helper. Modal doctor now prints both
`packed_mfu_probe_preview` and `packed_random_mfu_probe_preview`, so the
throughput and representative packed H100 probes are visible before launch and
can be invoked without remembering the `--bucket-selection random` override.

Validation after exposing the named random packed probe:

```text
uv run python -m pytest tests/test_vision.py::test_modal_command_builders -q
1 passed

uv run python -m py_compile nanochat/gpt.py nanochat/vision.py nanochat/flash_attention.py scripts/vlm_train.py modal_vlm.py tests/test_vision.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
74 passed, 10 skipped

uv run python -m pytest -q
82 passed, 10 skipped
```

Local follow-up while Modal is paused: added a warmup-excluded steady timing
summary for `--profile-timing` runs. The per-step timing line is still useful,
but H100 MFU decisions should use the aggregate steady summary because it
matches `steady_mfu`'s warmup exclusion and reports where wall time went across
data/image preprocessing, SigLIP, forward/backward, and optimizer work.

Validation after adding the steady timing summary:

```text
uv run python -m pytest tests/test_vision.py::test_profile_summary_reports_warmup_excluded_timing_percentages tests/test_vision.py::test_profile_includes_split_optimizer_timing_keys -q
2 passed

uv run python -m py_compile nanochat/gpt.py nanochat/vision.py nanochat/flash_attention.py scripts/vlm_train.py modal_vlm.py tests/test_vision.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
75 passed, 10 skipped

uv run python -m pytest -q
83 passed, 10 skipped

git diff --check
ok
```

Local MFU accounting cleanup while Modal remains paused: the main
`eff_llm_mfu`/`padded_llm_mfu`/`steady_mfu` fields now use the sequence-aware
FLOP path instead of the older config-length-per-token estimate. For selective
VLM loss, this charges the lm-head only for supervised target positions because
image, user/prompt, and pad positions do not run the lm-head matmul. The logs
also report `loss_tokens`, `lm_head useful/padded` counts, and auxiliary
`train/token_estimate_mfu` and `train/steady_token_estimate_mfu` fields for
comparison with older probes.

This correction is important for integrity: compact packing should improve MFU
by feeding more real transformer work to the GPU, not by counting full-vocab
logit work that selective loss deliberately skipped.

Validation after selective-aware MFU accounting:

```text
uv run python -m pytest tests/test_vision.py::test_varlen_step_flop_estimate_counts_segment_attention_and_padded_matmuls tests/test_vision.py::test_step_flop_estimate_charges_selective_lm_head_only_on_loss_tokens tests/test_vision.py::test_bucket_steady_metrics_accumulate_by_static_shape -q
3 passed

uv run python -m py_compile nanochat/gpt.py nanochat/vision.py nanochat/flash_attention.py scripts/vlm_train.py modal_vlm.py tests/test_vision.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
76 passed, 10 skipped

uv run python -m pytest -q
84 passed, 10 skipped
```

Follow-up: added warmup-excluded
`train/steady_token_estimate_mfu`/`train/steady_token_estimate_padded_mfu` so old
full-lm-head-per-token logs can be compared with the same warmup semantics as
the corrected `steady_mfu`.

Validation after adding steady token-estimate metrics:

```text
uv run python -m pytest tests/test_vision.py::test_step_flop_estimate_charges_selective_lm_head_only_on_loss_tokens tests/test_vision.py::test_bucket_steady_metrics_accumulate_by_static_shape -q
2 passed

uv run python -m py_compile nanochat/gpt.py nanochat/vision.py nanochat/flash_attention.py scripts/vlm_train.py modal_vlm.py tests/test_vision.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
76 passed, 10 skipped

uv run python -m pytest -q
84 passed, 10 skipped
```

Semantic guard cleanup: `--pack-examples > 1` now requires
`--boundary-aware-pack` unless the run explicitly passes `--allow-leaky-pack`.
This prevents the old dense-attention packed-row path from being used
accidentally as the final MFU training recipe, while keeping a clearly marked
diagnostic ablation available. The Modal train and MFU-probe command builders
expose `allow_leaky_pack=False` for that explicit opt-in.

Validation after the semantic packing guard:

```text
uv run python -m pytest tests/test_vision.py::test_pack_examples_requires_boundary_aware_or_explicit_leaky_opt_in tests/test_vision.py::test_modal_command_builders -q
2 passed

uv run python -m py_compile nanochat/gpt.py nanochat/vision.py nanochat/flash_attention.py scripts/vlm_train.py modal_vlm.py tests/test_vision.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
77 passed, 10 skipped

uv run python -m pytest -q
85 passed, 10 skipped
```

Direct compact correctness follow-up: added a regression that compares the final
`--boundary-aware-pack --flatten-packed-batch` compact path directly against
running each example separately. This covers the exact H100 recipe rather than
only proving padded varlen rows and compact-vs-padded equivalence. The test also
asserts the compact metadata shape: no dense `segment_ids`, no gather/scatter
`varlen_indices`, correct `cu_seqlens`, segment starts, and RoPE reset at the
second segment.

Validation after the direct compact-vs-separate regression:

```text
uv run python -m pytest tests/test_vision.py::test_direct_compact_batch_loss_matches_separate_examples tests/test_vision.py::test_direct_compact_batch_loss_matches_repack_then_flatten_path tests/test_vision.py::test_boundary_aware_packed_multimodal_loss_matches_separate_examples -q
3 passed

uv run python -m py_compile nanochat/gpt.py nanochat/vision.py nanochat/flash_attention.py scripts/vlm_train.py modal_vlm.py tests/test_vision.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
78 passed, 10 skipped

uv run python -m pytest -q
86 passed, 10 skipped
```

Modal probe cleanup while H100 runs remain paused: added a named
`packed_large_mfu_probe` entrypoint and `build_packed_large_mfu_probe_cmd()`.
This is the 65K useful-token stress probe from the original grid, using the same
compact boundary-aware recipe as `packed_mfu_probe` but with
`--device-batch-size 1024`, `--max-batch-tokens 65536`, `--pack-examples 16`,
and `--batch-buffer-size 8192`. Modal doctor now prints the large probe preview
alongside the default 32K throughput probe and the random representative probe.

Validation after adding the large packed probe entrypoint:

```text
uv run python -m pytest tests/test_vision.py::test_modal_command_builders -q
1 passed

uv run python -m py_compile nanochat/gpt.py nanochat/vision.py nanochat/flash_attention.py scripts/vlm_train.py modal_vlm.py tests/test_vision.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
78 passed, 10 skipped

uv run python -m pytest -q
86 passed, 10 skipped
```

Modal command-builder cleanup: `build_packed_random_mfu_probe_cmd()` now forces
`bucket_selection="random"` through kwargs before delegating to the packed probe
builder. This avoids a duplicate-key crash if future tooling passes an inherited
`bucket_selection` value while still preserving the named random probe's
semantics.

Validation after the random-probe builder cleanup:

```text
uv run python -m pytest tests/test_vision.py::test_modal_command_builders -q
1 passed

uv run python -m py_compile nanochat/gpt.py nanochat/vision.py nanochat/flash_attention.py scripts/vlm_train.py modal_vlm.py tests/test_vision.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
78 passed, 10 skipped

uv run python -m pytest -q
86 passed, 10 skipped
```

Added a named `packed_profile_mfu_probe` Modal entrypoint and
`build_packed_profile_mfu_probe_cmd()`. It forces `--profile-timing` on the same
32K compact boundary-aware packed recipe, so if the clean H100 probe misses the
target the follow-up run produces warmup-excluded `Steady timing totals` without
manually reassembling flags.

Validation after adding the profile packed probe:

```text
uv run python -m pytest tests/test_vision.py::test_modal_command_builders -q
1 passed

uv run python -m py_compile nanochat/gpt.py nanochat/vision.py nanochat/flash_attention.py scripts/vlm_train.py modal_vlm.py tests/test_vision.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
78 passed, 10 skipped

uv run python -m pytest -q
86 passed, 10 skipped
```

Added a named `packed_batch_plan` Modal entrypoint and
`build_packed_batch_plan_cmd()`. It runs the CPU-only `--batch-plan-steps` path
with the same default 32K compact packed selector shape as `packed_mfu_probe`
and intentionally omits `--require-fa3-varlen`, because it validates fill,
segments, padding, drops, and attention-pair estimates rather than the H100
attention backend. This makes the pre-H100 selector dry run reproducible from
the same Modal wrapper.

Validation after adding the packed batch-plan entrypoint:

```text
uv run python -m pytest tests/test_vision.py::test_modal_command_builders -q
1 passed

uv run python -m py_compile nanochat/gpt.py nanochat/vision.py nanochat/flash_attention.py scripts/vlm_train.py modal_vlm.py tests/test_vision.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
78 passed, 10 skipped

uv run python -m pytest -q
86 passed, 10 skipped
```

Added `packed_random_batch_plan` and `build_packed_random_batch_plan_cmd()` so
the representative random selector has the same CPU-only fill/segment/drops
preflight as the default max-token throughput selector. The helper forces
`bucket_selection="random"` even if inherited kwargs include a different
selection mode.

Validation after adding the random packed batch-plan entrypoint:

```text
uv run python -m pytest tests/test_vision.py::test_modal_command_builders -q
1 passed

uv run python -m py_compile nanochat/gpt.py nanochat/vision.py nanochat/flash_attention.py scripts/vlm_train.py modal_vlm.py tests/test_vision.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
78 passed, 10 skipped

uv run python -m pytest -q
86 passed, 10 skipped
```

Added `packed_large_batch_plan` and `build_packed_large_batch_plan_cmd()` so the
65K useful-token stress probe also has a CPU-only selector preflight before an
H100 run. It reuses the normal compact packed batch-plan path but defaults to
the large probe shape: `--device-batch-size 1024`, `--max-batch-tokens 65536`,
`--batch-buffer-size 8192`, `--pack-examples 16`, `--max-seq-len 1024`, and
`--pack-max-seq-len 1024`. The dry run intentionally omits
`--require-fa3-varlen` because it checks fill, segment lengths, drops, and
attention-pair estimates without loading the CUDA attention backend. The GPU
runbook now calls this out as the preflight immediately before
`packed_large_mfu_probe`.

Validation after adding the large packed batch-plan entrypoint:

```text
uv run python -m pytest tests/test_vision.py::test_modal_command_builders -q
1 passed

uv run python -m py_compile nanochat/gpt.py nanochat/vision.py nanochat/flash_attention.py scripts/vlm_train.py modal_vlm.py tests/test_vision.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
78 passed, 10 skipped

uv run python -m pytest -q
86 passed, 10 skipped

git diff --check
ok
```

Local CPU preflight for the new 65K packed batch-plan shape, with Modal still
paused:

```text
uv run --extra vision python -m scripts.vlm_train \
  --device-type cpu \
  --hf-repo HuggingFaceM4/the_cauldron \
  --hf-config vqav2 \
  --device-batch-size 1024 \
  --max-batch-tokens 65536 \
  --max-seq-len 1024 \
  --stream-buffer-size 256 \
  --batch-buffer-size 8192 \
  --bucket-selection max-tokens \
  --pack-examples 16 \
  --pack-max-seq-len 1024 \
  --boundary-aware-pack \
  --flatten-packed-batch \
  --batch-plan-steps 2 \
  --model-step 650

Batch plan source=stream:HuggingFaceM4/the_cauldron/vqav2
steps=2 batch_size=1,024 max_batch_tokens=65,536 grad_accum_steps=1 bucket_lens=none bucket_selection=max-tokens bucket_min_fill_frac=0 bucket_cycle_repeat=1 pack_examples=16 boundary_aware_pack=True flatten_packed_batch=True
planning_elapsed=95.8s records_scanned=8,769 rendered_examples=8,757 rendered_examples/sec=91.4
overall rows/step=1.0 examples/step=561.5 dropped/step=0.0 tokens/step=65519/65519 pad=0.0% attn_pairs/step=3.86M segments/step=561.5 avg_segment=116.7 max_segment=118
bucket | steps | rows avg/min/max | examples avg/dropped | segments avg | segment_len avg/max | target_rows | avg_fill | tokens/step useful/padded | pad | attn_pairs/step
 65504 |     1 |   1.0/  1/1   |  558.0/0.0     |        558.0 |     117.4/118     |           1 |   100.0% |   65504/65504   |   0.0% |           3.88M
 65534 |     1 |   1.0/  1/1   |  565.0/0.0     |        565.0 |     116.0/117     |           1 |   100.0% |   65534/65534   |   0.0% |           3.83M
```

Interpretation: the 65K compact shape is mechanically healthy for the throughput
stress path: it reaches the useful-token budget with no row padding and no
dropped examples. The same caveat as the 32K `vqav2` preflight applies: the
`max-tokens` selector finds very short segments (`avg_segment=116.7`,
`max_segment=118`), so this is evidence for an MFU stress run, not proof of a
representative long-text distribution. Use the random selector or a longer
config such as `docvqa` when checking realism.

Added `packed_large_random_mfu_probe` and
`build_packed_large_random_mfu_probe_cmd()` plus the matching CPU-only
`packed_large_random_batch_plan` wrapper. This keeps the 65K compact packed
shape from `packed_large_mfu_probe` but forces `--bucket-selection random`, so
the representative long-segment ablation can be launched and dry-run without
manual flag edits. Modal doctor now prints both the H100 probe preview and the
CPU batch-plan preview.

Validation after adding the large random packed entrypoints:

```text
uv run python -m pytest tests/test_vision.py::test_modal_command_builders -q
1 passed

uv run python -m py_compile nanochat/gpt.py nanochat/vision.py nanochat/flash_attention.py scripts/vlm_train.py modal_vlm.py tests/test_vision.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
78 passed, 10 skipped

uv run python -m pytest -q
86 passed, 10 skipped
```

Local CPU preflight for the 65K random packed batch-plan shape, with Modal still
paused:

```text
uv run --extra vision python -m scripts.vlm_train \
  --device-type cpu \
  --hf-repo HuggingFaceM4/the_cauldron \
  --hf-config vqav2 \
  --device-batch-size 1024 \
  --max-batch-tokens 65536 \
  --max-seq-len 1024 \
  --stream-buffer-size 256 \
  --batch-buffer-size 8192 \
  --bucket-selection random \
  --pack-examples 16 \
  --pack-max-seq-len 1024 \
  --boundary-aware-pack \
  --flatten-packed-batch \
  --batch-plan-steps 2 \
  --model-step 650

Batch plan source=stream:HuggingFaceM4/the_cauldron/vqav2
steps=2 batch_size=1,024 max_batch_tokens=65,536 grad_accum_steps=1 bucket_lens=none bucket_selection=random bucket_min_fill_frac=0 bucket_cycle_repeat=1 pack_examples=16 boundary_aware_pack=True flatten_packed_batch=True
planning_elapsed=88.6s records_scanned=8,590 rendered_examples=8,578 rendered_examples/sec=96.8
overall rows/step=1.0 examples/step=387.5 dropped/step=0.0 tokens/step=65479/65479 pad=0.0% attn_pairs/step=7.54M segments/step=387.5 avg_segment=169.0 max_segment=1000
bucket | steps | rows avg/min/max | examples avg/dropped | segments avg | segment_len avg/max | target_rows | avg_fill | tokens/step useful/padded | pad | attn_pairs/step
 65445 |     1 |   1.0/  1/1   |  389.0/0.0     |        389.0 |     168.2/979     |           1 |   100.0% |   65445/65445   |   0.0% |           7.47M
 65513 |     1 |   1.0/  1/1   |  386.0/0.0     |        386.0 |     169.7/1000    |           1 |   100.0% |   65513/65513   |   0.0% |           7.61M
```

Interpretation: the 65K random path also keeps the useful-token budget full with
zero padding and zero dropped examples. Compared with the max-token 65K
preflight, it processes fewer examples per step, raises attention-pair work from
3.86M to 7.54M, and exposes much longer segments (`max_segment=1000`). This is
the right companion to the throughput stress probe for judging whether a high
MFU number survives more realistic sequence lengths.

Added near-cap segment diagnostics to CPU batch-plan summaries. The report now
prints `near_cap/step` for selected segments at or above 95% of the active
segment cap and `cap_hits/step` for segments exactly at the cap. This keeps
long-segment packed preflights auditable: a high `max_segment` can now be
distinguished from systematic cap pressure.

Validation after adding near-cap diagnostics:

```text
uv run python -m pytest tests/test_vision.py::test_batch_plan_summary_reports_bucket_fill_and_padding tests/test_vision.py::test_batch_plan_row_reports_near_cap_segments tests/test_vision.py::test_batch_plan_summary_reports_optimizer_step_groups -q
3 passed

uv run python -m py_compile scripts/vlm_train.py tests/test_vision.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
79 passed, 10 skipped

uv run python -m pytest -q
87 passed, 10 skipped
```

Rerun of the 65K random packed preflight with near-cap diagnostics:

```text
uv run --extra vision python -m scripts.vlm_train \
  --device-type cpu \
  --hf-repo HuggingFaceM4/the_cauldron \
  --hf-config vqav2 \
  --device-batch-size 1024 \
  --max-batch-tokens 65536 \
  --max-seq-len 1024 \
  --stream-buffer-size 256 \
  --batch-buffer-size 8192 \
  --bucket-selection random \
  --pack-examples 16 \
  --pack-max-seq-len 1024 \
  --boundary-aware-pack \
  --flatten-packed-batch \
  --batch-plan-steps 2 \
  --model-step 650

Batch plan source=stream:HuggingFaceM4/the_cauldron/vqav2
steps=2 batch_size=1,024 max_batch_tokens=65,536 grad_accum_steps=1 bucket_lens=none bucket_selection=random bucket_min_fill_frac=0 bucket_cycle_repeat=1 pack_examples=16 boundary_aware_pack=True flatten_packed_batch=True
planning_elapsed=85.6s records_scanned=8,590 rendered_examples=8,578 rendered_examples/sec=100.2
overall rows/step=1.0 examples/step=387.5 dropped/step=0.0 tokens/step=65479/65479 pad=0.0% attn_pairs/step=7.54M segments/step=387.5 avg_segment=169.0 max_segment=1000 near_cap/step=1.0 cap_hits/step=0.0
bucket | steps | rows avg/min/max | examples avg/dropped | segments avg | segment_len avg/max | near_cap/cap avg | target_rows | avg_fill | tokens/step useful/padded | pad | attn_pairs/step
 65445 |     1 |   1.0/  1/1   |  389.0/0.0     |        389.0 |     168.2/979     |     1.0/0.0     |           1 |   100.0% |   65445/65445   |   0.0% |           7.47M
 65513 |     1 |   1.0/  1/1   |  386.0/0.0     |        386.0 |     169.7/1000    |     1.0/0.0     |           1 |   100.0% |   65513/65513   |   0.0% |           7.61M
```

Interpretation: the long-tail random preflight is not saturating the 1024-token
cap. It selected about one near-cap segment per step and no exact cap hits, so
the 65K random ablation is a realistic long-segment stress case without being
dominated by examples clipped to the configured maximum.

Extended the same near-cap counters into the actual training loop. Step logs now
print `near_cap` and `cap_hits`, and W&B receives `train/near_cap_segments` and
`train/cap_segments`. This makes the future H100 `steady_mfu` probe auditable
without separately rerunning a CPU batch-plan command for the exact selected
batches.

Validation after adding train-log near-cap counters:

```text
uv run python -m pytest tests/test_vision.py::test_batch_plan_row_reports_near_cap_segments tests/test_vision.py::test_count_near_cap_segments_matches_batch_plan_threshold -q
2 passed

uv run python -m py_compile scripts/vlm_train.py tests/test_vision.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
80 passed, 10 skipped

uv run python -m pytest -q
88 passed, 10 skipped
```

Added a named `packed_large_profile_mfu_probe` Modal entrypoint and
`build_packed_large_profile_mfu_probe_cmd()`. It forces `--profile-timing` on the
same 65K compact boundary-aware shape as `packed_large_mfu_probe`
(`--device-batch-size 1024`, `--max-batch-tokens 65536`,
`--batch-buffer-size 8192`, `--pack-examples 16`). This gives the large stress
probe a matching attribution run without manually reassembling flags. Modal
doctor now prints the large profile preview.

Validation after adding the large packed profile entrypoint:

```text
uv run python -m pytest tests/test_vision.py::test_modal_command_builders -q
1 passed

uv run python -m py_compile modal_vlm.py tests/test_vision.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
80 passed, 10 skipped

uv run python -m pytest -q
88 passed, 10 skipped
```

Extended warmup-excluded steady diagnostics to include the same segment-shape
information as each step. The H100 log and W&B now expose:
`train/steady_attention_pairs_per_step`, `train/steady_segments_per_step`,
`train/steady_avg_segment_len`, `train/steady_max_segment_len`,
`train/steady_near_cap_segments_per_step`, and
`train/steady_cap_segments_per_step`. Static-bucket steady metrics also carry
the matching `bucket_steady_*` segment fields. This aligns the segment/cap
distribution with the exact same warmup-excluded window used by `steady_mfu`.

Validation after adding steady segment diagnostics:

```text
uv run python -m pytest tests/test_vision.py::test_bucket_steady_metrics_accumulate_by_static_shape tests/test_vision.py::test_count_near_cap_segments_matches_batch_plan_threshold -q
2 passed

uv run python -m py_compile scripts/vlm_train.py tests/test_vision.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
80 passed, 10 skipped

uv run python -m pytest -q
88 passed, 10 skipped
```

Added the missing `packed_large_random_profile_mfu_probe` Modal entrypoint. The
helper/doctor preview already forced `--bucket-selection random` and
`--profile-timing`; the new entrypoint exposes the same 65K compact packed shape
as `packed_large_random_mfu_probe` without requiring manual flag edits. This is
the profiler companion for the representative random selector, where local
preflight showed `avg_segment=169.0`, `max_segment=1000`, and
`attn_pairs/step=7.54M`.

Validation after adding the large random packed profile entrypoint:

```text
uv run python -m pytest tests/test_vision.py::test_modal_command_builders -q
1 passed

uv run python -m py_compile modal_vlm.py tests/test_vision.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
80 passed, 10 skipped

uv run python -m pytest -q
88 passed, 10 skipped
```

Tightened profile attribution for the next H100 MFU run. `--profile-timing` now
reports a separate `pack` bucket between SigLIP/pooling and multimodal tensor
construction. Previously the CPU work that builds compact packed rows was not
visible in the warmup-excluded `Steady timing totals`; now a low `steady_mfu`
can distinguish dataset/render, image/SigLIP, packed-row construction,
`build_multimodal_batch`, transformer forward/backward, and optimizer time.

Validation after adding packed-row timing:

```text
uv run python -m pytest tests/test_vision.py::test_profile_summary_reports_pack_timing -q
1 passed

uv run python -m py_compile scripts/vlm_train.py tests/test_vision.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
81 passed, 10 skipped

uv run python -m pytest -q
89 passed, 10 skipped
```

Added residual `other` timing to `format_profile_summary()`. The calculation
uses top-level non-overlapping buckets (`data_wait`, `data`, `image_siglip`,
`pack`, `batch`, `fwdbwd`, `optim`) and reports any remaining warmup-excluded
wall time. This is useful for H100 bottleneck attribution: if `steady_mfu` is
low and the named buckets do not explain the wall time, `other` will make that
visible instead of hiding it in the denominator.

Validation after adding residual timing:

```text
uv run python -m pytest tests/test_vision.py::test_profile_summary_reports_pack_timing tests/test_vision.py::test_profile_summary_reports_warmup_excluded_timing_percentages -q
2 passed

uv run python -m py_compile scripts/vlm_train.py tests/test_vision.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
81 passed, 10 skipped

uv run python -m pytest -q
89 passed, 10 skipped
```

Tightened `pack_example_groups()` so the single-row path (`max_images_per_row <=
1`) now respects `max_seq_len`, `max_batch_tokens`, and `fixed_rows` instead of
returning every example unchanged. The general packed path also rejects a first
row that already exceeds `--max-batch-tokens`. This keeps CPU batch plans and
future packed probes from silently violating the token-budget invariant in edge
cases, which matters for credible MFU denominator/accounting.

Validation after tightening pack budget invariants:

```text
uv run python -m pytest tests/test_vision.py::test_pack_example_groups_single_rows_respect_caps tests/test_vision.py::test_pack_example_groups_fixed_rows_do_not_add_empty_row_boundary tests/test_vision.py::test_stream_batch_can_select_only_packable_examples -q
3 passed

uv run python -m py_compile scripts/vlm_train.py tests/test_vision.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
82 passed, 10 skipped

uv run python -m pytest -q
90 passed, 10 skipped
```

Tightened profile attribution for packed probes. In `--profile-timing` mode the
trainer now synchronizes immediately after compact row packing, so GPU feature
reordering caused by packing is charged to the `pack` bucket instead of slipping
into the following multimodal tensor construction bucket. The residual timing
calculation is also exposed as helpers and logged to W&B as `timing/other_sec`,
`timing/other_frac`, `timing/steady_other_sec`, and
`timing/steady_other_frac`, matching the stdout `Steady timing totals` residual.

Validation after tightening profile attribution:

```text
uv run python -m pytest tests/test_vision.py::test_profile_summary_reports_pack_timing tests/test_vision.py::test_profile_summary_reports_warmup_excluded_timing_percentages -q
2 passed

uv run python -m py_compile scripts/vlm_train.py tests/test_vision.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
82 passed, 10 skipped

uv run python -m pytest -q
90 passed, 10 skipped
```

2026-05-26 local packing-selector cleanup while Modal is paused:

- `96` remains only a diagnostic short bucket. The realistic MFU path should use
  the length distribution and packed/bucketed caps such as 512/1024, not a global
  96-token sequence length.
- Tightened the packed stream selector and direct compact flattener so
  `--max-batch-tokens` is respected even for the first candidate segment. If a
  sampled/fallback packed window is over cap but a later buffered example fits,
  the selector now chooses the fit example instead of sending an unfit one to
  image open/processor/SigLIP only to drop it during packing.
- `trim_examples_to_packable()` now also filters a single selected packed
  example when it violates the pack length/token cap, so stream-packed training
  can discard impossible selections before vision work.

Focused validation:

```text
uv run python -m pytest tests/test_vision.py::test_pack_trim_filters_examples_before_vision_work tests/test_vision.py::test_stream_packed_selection_skips_unfit_fallback_windows tests/test_vision.py::test_stream_packed_random_compact_selection_respects_first_token_cap tests/test_vision.py::test_flatten_examples_as_compact_batch_keeps_segments_and_feature_order -q
4 passed

uv run python -m py_compile scripts/vlm_train.py tests/test_vision.py
ok
```

2026-05-26 local no-sync cleanup for the boundary-aware packed path:

- Removed a CUDA synchronization point from `GPT.forward(...)` on packed VLM
  batches. The rotary cache bounds check now runs only for CPU `position_ids`;
  H100 packed training no longer performs `int(position_ids.max())` every
  forward.
- `build_multimodal_batch(...)` now carries CPU-side `token_count`,
  `padded_token_count`, and `supervised_target_count` on `MultimodalBatch`.
  `vlm_train.py` uses those for MFU/logit FLOP accounting instead of doing CUDA
  reductions on `batch.lengths.sum()` and `(batch.targets != -1).sum()` each
  microstep.
- Removed the small `valid.any()` Python branch when inserting projected image
  features; empty/truncated visual spans are handled by empty tensor indexing.

Validation after removing these sync points:

```text
uv run python -m pytest tests/test_vision.py::test_visual_token_insertion_and_target_masking tests/test_vision.py::test_multimodal_batch_can_pad_to_fixed_len tests/test_vision.py::test_multimodal_batch_can_pad_to_static_bucket tests/test_vision.py::test_multimodal_batch_counts_survive_truncated_image_span tests/test_vision.py::test_multimodal_batch_allows_packed_image_rows -q
5 passed

uv run python -m py_compile nanochat/gpt.py nanochat/vision.py scripts/vlm_train.py tests/test_vision.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
85 passed, 10 skipped

uv run python -m pytest -q
93 passed, 10 skipped
```

2026-05-26 local materialized packed-selector cleanup:

- Added a buffered packed selector for materialized/non-stream datasets. When
  `--pack-examples > 1`, local JSON and `--max-examples` training now select
  packable examples from a rendered-example buffer with the same
  `_choose_stream_packed_indices(...)` policy used by the HF streaming path,
  instead of taking a generic batch and trimming it after selection.
- This keeps CPU batch plans and local/debug packed runs closer to the H100
  streaming recipe: examples that cannot fit `--pack-max-seq-len` or
  `--max-batch-tokens` are avoided before image open/processor/SigLIP work when
  a packable buffered example exists.

Validation after materialized packed-selector cleanup:

```text
uv run python -m pytest tests/test_vision.py::test_materialized_bucket_batches_stop_at_bucket_boundaries tests/test_vision.py::test_materialized_packed_batch_uses_packable_buffer_before_vision_work tests/test_vision.py::test_stream_batch_can_select_only_packable_examples tests/test_vision.py::test_stream_packed_selection_skips_unfit_fallback_windows -q
4 passed

uv run python -m py_compile scripts/vlm_train.py tests/test_vision.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
86 passed, 10 skipped

uv run python -m pytest -q
94 passed, 10 skipped
```

2026-05-26 local full-CE allocation cleanup:

- `build_multimodal_batch(...)` now has `return_loss_indices`. The default still
  returns precomputed selective-loss indices for VLM selective CE, but the train
  loop passes `return_loss_indices=False` when `--no-selective-loss` is active.
- Full-CE fixed-shape/compile probes still receive complete target tensors and
  the same `supervised_target_count` logging, but they no longer allocate the
  extra `loss_indices`/`loss_targets` tensors that only selective loss uses.

Validation after skipping selective-loss tensors for full CE:

```text
uv run python -m pytest tests/test_vision.py::test_visual_token_insertion_and_target_masking tests/test_vision.py::test_multimodal_batch_can_skip_selective_loss_indices tests/test_vision.py::test_gpt_selective_loss_matches_ignore_index_path tests/test_vision.py::test_materialized_packed_batch_uses_packable_buffer_before_vision_work -q
4 passed

uv run python -m py_compile nanochat/vision.py scripts/vlm_train.py tests/test_vision.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
87 passed, 10 skipped

uv run python -m pytest -q
95 passed, 10 skipped
```

2026-05-26 local target-index construction cleanup:

- `build_multimodal_batch(...)` now records supervised label positions while it
  constructs each shifted target row, instead of scanning all target rows a
  second time afterward. Truncation filters those per-row positions so
  `loss_indices` and `loss_targets` still match the post-truncation tensors.
- Boundary-aware packed correctness tests now pass the precomputed
  `loss_indices`/`loss_targets` through `GPT.forward(...)`, covering the same
  selective-loss path used by packed training.

Validation after collecting loss indices during construction:

```text
uv run python -m pytest tests/test_vision.py::test_visual_token_insertion_and_target_masking tests/test_vision.py::test_multimodal_batch_counts_survive_truncated_image_span tests/test_vision.py::test_boundary_aware_packed_multimodal_loss_matches_separate_examples tests/test_vision.py::test_direct_compact_batch_loss_matches_repack_then_flatten_path -q
4 passed

uv run python -m py_compile nanochat/vision.py tests/test_vision.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
87 passed, 10 skipped

uv run python -m pytest -q
95 passed, 10 skipped
```

2026-05-26 local selective-loss index cleanup:

- `build_multimodal_batch(...)` now precomputes `loss_indices` and
  `loss_targets` for supervised VLM labels while it is already constructing the
  shifted target rows.
- `GPT.forward(..., selective_loss=True)` can consume those precomputed tensors.
  The VLM training path passes them through, avoiding the per-forward GPU target
  scan/boolean mask used to discover valid labels inside the packed H100 path.
- The fallback selective-loss behavior is unchanged: callers that do not provide
  precomputed indices still use the existing `targets != -1` path, and
  `loss_reduction="none"`, `"sum"`, and `"mean"` continue to match full
  ignore-index CE.

Validation after adding precomputed selective-loss indices:

```text
uv run python -m pytest tests/test_vision.py::test_visual_token_insertion_and_target_masking tests/test_vision.py::test_gpt_selective_loss_matches_ignore_index_path tests/test_vision.py::test_boundary_aware_packed_multimodal_loss_matches_separate_examples tests/test_vision.py::test_direct_compact_batch_loss_matches_separate_examples -q
4 passed

uv run python -m py_compile nanochat/gpt.py nanochat/vision.py scripts/vlm_train.py tests/test_vision.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
85 passed, 10 skipped

uv run python -m pytest -q
93 passed, 10 skipped
```

2026-05-26 local sparse-target selective-loss cleanup:

- Selective VLM training now calls `build_multimodal_batch(...)` with
  `return_targets=False`, so packed 32K/65K steps no longer build, allocate, or
  transfer the dense `[B, T]` target tensor when precomputed
  `loss_indices`/`loss_targets` are sufficient.
- `GPT.forward(..., selective_loss=True)` can compute the same softcapped CE
  from `loss_indices` and `loss_targets` even when `targets=None`. The dense
  target path remains the default for full CE, chunked CE, eval, and callers
  that need `loss_reduction="none"` with explicit ignore-index targets.

Validation after making selective training target-sparse:

```text
uv run python -m pytest tests/test_vision.py::test_multimodal_batch_can_skip_dense_targets_for_selective_loss tests/test_vision.py::test_gpt_selective_loss_matches_ignore_index_path tests/test_vision.py::test_boundary_aware_packed_multimodal_loss_matches_separate_examples -q
3 passed

uv run python -m py_compile nanochat/gpt.py nanochat/vision.py scripts/vlm_train.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
88 passed, 10 skipped

uv run python -m pytest -q
96 passed, 10 skipped

git diff --check
ok
```

2026-05-26 local full-span image-insertion cleanup:

- `build_multimodal_batch(...)` now uses a direct full-span assignment for the
  common packed case where every projected image contributes all 64 visual
  tokens. The clipped boolean-mask insertion path remains for truncated rows.
- This removes the per-batch clipping mask and feature-offset gather tensors
  from realistic untruncated compact packed batches, while preserving the
  existing projected-feature ordering and truncation behavior.

Validation after the full-span image insertion cleanup:

```text
uv run python -m pytest tests/test_vision.py::test_multimodal_batch_allows_packed_image_rows tests/test_vision.py::test_multimodal_batch_counts_survive_truncated_image_span tests/test_vision.py::test_compact_boundary_aware_packed_batch_matches_padded_rows tests/test_vision.py::test_direct_compact_batch_loss_matches_repack_then_flatten_path -q
4 passed

uv run python -m py_compile nanochat/vision.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
88 passed, 10 skipped

uv run python -m pytest -q
96 passed, 10 skipped
```

2026-05-26 local sparse smear-boundary cleanup:

- Packed VLM training now asks `build_multimodal_batch(...)` for sparse
  `segment_start_indices` instead of the dense `[B, T]` `segment_starts` mask.
  `GPT.forward(...)` uses those indices to zero smear only at packed segment
  starts.
- The default dense `segment_starts` path remains for compatibility and
  correctness tests. The sparse path is what the realistic packed 32K/65K
  selective training loop now uses.

Validation after sparse smear-boundary metadata:

```text
uv run python -m pytest tests/test_vision.py::test_sparse_segment_start_indices_match_dense_smear_mask tests/test_vision.py::test_boundary_aware_packed_multimodal_loss_matches_separate_examples tests/test_vision.py::test_compact_boundary_aware_packed_batch_matches_padded_rows -q
3 passed

uv run python -m py_compile nanochat/gpt.py nanochat/vision.py scripts/vlm_train.py tests/test_vision.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
89 passed, 10 skipped

uv run python -m pytest -q
97 passed, 10 skipped
```

2026-05-26 local prefix image-feature selection cleanup:

- `_select_image_features(...)` now returns a prefix view when the packed batch
  keeps image features `[0:n]` from the SigLIP output. It still returns the
  exact original tensor when all features are selected, and still uses
  `index_select` when packing reorders or skips the middle of the feature list.
- This avoids an unnecessary feature copy in compact packed batches that keep a
  prefix under the token budget, while preserving multi-image feature ordering.

Validation after prefix feature selection:

```text
uv run python -m pytest tests/test_vision.py::test_direct_compact_batch_matches_compact_packer_budget tests/test_vision.py::test_flatten_examples_as_compact_batch_keeps_segments_and_feature_order -q
2 passed

uv run python -m py_compile scripts/vlm_train.py tests/test_vision.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
89 passed, 10 skipped

uv run python -m pytest -q
97 passed, 10 skipped
```

2026-05-26 local compact position-id derivation cleanup:

- Compact varlen packed batches now derive RoPE `position_ids` from
  `segment_lengths`/`cu_seqlens` with tensor ops instead of appending one Python
  position value per expanded token while constructing the batch.
- Padded and truncated batches keep the existing explicit position-row path.
  The compact path still resets RoPE positions at every packed segment boundary.

Validation after compact position-id derivation:

```text
uv run python -m pytest tests/test_vision.py::test_direct_compact_batch_loss_matches_separate_examples tests/test_vision.py::test_compact_boundary_aware_packed_batch_matches_padded_rows tests/test_vision.py::test_direct_compact_batch_loss_matches_repack_then_flatten_path -q
3 passed

uv run python -m py_compile nanochat/vision.py tests/test_vision.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
89 passed, 10 skipped

uv run python -m pytest -q
97 passed, 10 skipped
```

2026-05-26 local compact single-row image-insertion cleanup:

- Full-span visual-token insertion now has a single-row fast path for compact
  packed batches. The common `flatten_packed_batch=True` training case assigns
  projected image tokens with only the positions tensor, avoiding the extra
  row-index tensor needed by dense multi-row batches.
- Multi-row full-span insertion and truncated clipped insertion keep their
  previous paths. Tests now explicitly check compact visual-span feature
  ordering after insertion.

Validation after compact single-row image insertion:

```text
uv run python -m pytest tests/test_vision.py::test_multimodal_batch_allows_packed_image_rows tests/test_vision.py::test_direct_compact_batch_loss_matches_separate_examples tests/test_vision.py::test_compact_boundary_aware_packed_batch_matches_padded_rows -q
3 passed

uv run python -m py_compile nanochat/vision.py tests/test_vision.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
89 passed, 10 skipped

uv run python -m pytest -q
97 passed, 10 skipped
```

2026-05-26 local realistic 65K packed-plan guardrail:

- Added a deterministic batch-plan test for the packed-large shape:
  `max_batch_tokens=65,536`, `max_seq_len=1024`, `pack_examples=16`,
  `boundary_aware_pack=True`, `flatten_packed_batch=True`.
- The synthetic plan fills the full 65,536-token compact budget with 128
  length-512 segments in one compact row, and verifies attention pairs are
  computed per segment rather than as one giant 65K causal sequence. This keeps
  local coverage aimed at the realistic packed path instead of the old 96-token
  diagnostic.

Validation after the 65K packed-plan guardrail:

```text
uv run python -m pytest tests/test_vision.py::test_packed_large_batch_plan_fills_realistic_token_budget tests/test_vision.py::test_flattened_packed_batch_plan_reports_compact_tokens tests/test_vision.py::test_compact_packed_budget_uses_total_tokens_not_dense_rows -q
3 passed

uv run python -m py_compile tests/test_vision.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
90 passed, 10 skipped

uv run python -m pytest -q
98 passed, 10 skipped
```

2026-05-26 local packed lengths-tensor cleanup:

- `build_multimodal_batch(...)` now has `return_lengths`. Boundary-aware packed
  training passes `return_lengths=False` because the train loop already uses
  CPU-side `token_count`/`padded_token_count` and segment metadata for MFU
  accounting.
- Dense/non-boundary training still requests `lengths`, because that path uses
  row lengths for near-cap diagnostics. Eval and generation keep the default
  tensor behavior.

Validation after skipping the unused packed lengths tensor:

```text
uv run python -m pytest tests/test_vision.py::test_multimodal_batch_can_skip_lengths_tensor_for_training_counts tests/test_vision.py::test_boundary_aware_packed_multimodal_loss_matches_separate_examples tests/test_vision.py::test_direct_compact_batch_loss_matches_separate_examples -q
3 passed

uv run python -m py_compile nanochat/vision.py scripts/vlm_train.py tests/test_vision.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
91 passed, 10 skipped

uv run python -m pytest -q
99 passed, 10 skipped
```

2026-05-26 local compact smear-index derivation cleanup:

- Compact varlen batches now derive sparse `segment_start_indices` from
  `segment_lengths`/prefix sums instead of collecting per-token start indices
  while walking the packed token row.
- Dense packed rows keep the previous explicit start-index path, and compact
  correctness tests compare the derived sparse indices against the dense
  `segment_starts` smear mask.

Validation after compact smear-index derivation:

```text
uv run python -m pytest tests/test_vision.py::test_compact_segment_start_indices_derive_from_segment_lengths tests/test_vision.py::test_sparse_segment_start_indices_match_dense_smear_mask tests/test_vision.py::test_direct_compact_batch_loss_matches_separate_examples -q
3 passed

uv run python -m py_compile nanochat/vision.py tests/test_vision.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
92 passed, 10 skipped

uv run python -m pytest -q
100 passed, 10 skipped
```

2026-05-26 local real-data packed-large batch-plan check:

- Ran the CPU-only packed-large batch plan against cached/local
  `HuggingFaceM4/the_cauldron/vqav2` streaming data with the realistic packed
  probe settings: `batch_size=1024`, `max_batch_tokens=65,536`,
  `max_seq_len=1024`, `pack_examples=16`, `boundary_aware_pack=True`,
  `flatten_packed_batch=True`, `bucket_selection=max-tokens`.
- Result: two planned optimizer steps filled compact rows at 65,504 and 65,534
  useful tokens with 0.0% padding. Average examples/segments per step was
  561.5, average segment length 116.7, max segment length 118, and attention
  pairs were ~3.86M/step. This confirms the realistic packed path can produce
  large 65K useful-token steps locally on real VQA data; it is not relying on
  the old 96-token diagnostic shape.

Command and output summary:

```text
uv run python -m scripts.vlm_train --device-type cpu --hf-repo HuggingFaceM4/the_cauldron --hf-config vqav2 --device-batch-size 1024 --max-batch-tokens 65536 --max-seq-len 1024 --stream-buffer-size 256 --batch-buffer-size 8192 --bucket-selection max-tokens --pack-examples 16 --pack-max-seq-len 1024 --boundary-aware-pack --flatten-packed-batch --batch-plan-steps 2 --model-step 650

Batch plan source=stream:HuggingFaceM4/the_cauldron/vqav2
steps=2 batch_size=1,024 max_batch_tokens=65,536 grad_accum_steps=1 bucket_lens=none bucket_selection=max-tokens bucket_min_fill_frac=0 bucket_cycle_repeat=1 pack_examples=16 boundary_aware_pack=True flatten_packed_batch=True
planning_elapsed=117.9s records_scanned=8,769 rendered_examples=8,757 rendered_examples/sec=74.3
overall rows/step=1.0 examples/step=561.5 dropped/step=0.0 tokens/step=65519/65519 pad=0.0% attn_pairs/step=3.86M segments/step=561.5 avg_segment=116.7 max_segment=118 near_cap/step=0.0 cap_hits/step=0.0
```

2026-05-26 local real-data packed-large random batch-plan check:

- Ran the same CPU-only packed-large batch plan against cached/local
  `HuggingFaceM4/the_cauldron/vqav2`, but with `bucket_selection=random`, which
  matches the packed-large-random H100 probe variant.
- Result: two planned optimizer steps still filled compact rows near the 65,536
  token cap, at 65,445 and 65,513 useful tokens with 0.0% padding. Average
  examples/segments per step was 387.5, average segment length 169.0, max
  segment length 1000, and attention pairs were ~7.54M/step.
- Compared with the `max-tokens` selector above, random fills the token budget
  but produces fewer/longer segments and roughly 2x attention pairs. That makes
  `max-tokens` the better default for MFU probes, while `random` remains a
  useful stress/quality ablation.

Command and output summary:

```text
uv run python -m scripts.vlm_train --device-type cpu --hf-repo HuggingFaceM4/the_cauldron --hf-config vqav2 --device-batch-size 1024 --max-batch-tokens 65536 --max-seq-len 1024 --stream-buffer-size 256 --batch-buffer-size 8192 --bucket-selection random --pack-examples 16 --pack-max-seq-len 1024 --boundary-aware-pack --flatten-packed-batch --batch-plan-steps 2 --model-step 650

Batch plan source=stream:HuggingFaceM4/the_cauldron/vqav2
steps=2 batch_size=1,024 max_batch_tokens=65,536 grad_accum_steps=1 bucket_lens=none bucket_selection=random bucket_min_fill_frac=0 bucket_cycle_repeat=1 pack_examples=16 boundary_aware_pack=True flatten_packed_batch=True
planning_elapsed=113.6s records_scanned=8,590 rendered_examples=8,578 rendered_examples/sec=75.5
overall rows/step=1.0 examples/step=387.5 dropped/step=0.0 tokens/step=65479/65479 pad=0.0% attn_pairs/step=7.54M segments/step=387.5 avg_segment=169.0 max_segment=1000 near_cap/step=1.0 cap_hits/step=0.0
```

2026-05-26 local return-lengths consistency cleanup:

- The clipped visual-span fallback now uses a local temporary lengths tensor
  instead of reusing `lengths_tensor`, so `return_lengths=False` consistently
  returns `batch.lengths is None` even when a truncated image span needs row
  lengths for insertion.
- This keeps the packed training path's skipped-lengths behavior precise while
  preserving clipped image insertion correctness.

Validation after return-lengths consistency cleanup:

```text
uv run python -m pytest tests/test_vision.py::test_multimodal_batch_can_skip_lengths_tensor_for_training_counts tests/test_vision.py::test_multimodal_batch_can_skip_lengths_tensor_with_truncated_image_span tests/test_vision.py::test_multimodal_batch_counts_survive_truncated_image_span -q
3 passed

uv run python -m py_compile nanochat/vision.py tests/test_vision.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
93 passed, 10 skipped

uv run python -m pytest -q
101 passed, 10 skipped
```

2026-05-26 local SigLIP pooled-feature dtype cleanup:

- `pool_siglip_features` now preserves the SigLIP encoder activation dtype
  instead of forcing pooled features to fp32. On H100 bf16 probes, the training
  path was immediately casting these pooled features back to the GPT embedding
  dtype before the projector, so the fp32 promotion only added a large transient
  feature tensor between SigLIP and `VisionProjector`.
- For the packed-large real-data plan above, hundreds of images can be present
  in one 65K-token optimizer step; keeping pooled features in bf16 avoids that
  avoidable inter-stage memory bandwidth and allocator pressure without changing
  the projector input dtype used by the packed training path.

Validation after pooled-feature dtype cleanup:

```text
uv run python -m pytest tests/test_vision.py::test_pool_siglip_features_uses_nanovlm_pixel_shuffle tests/test_vision.py::test_pool_siglip_features_preserves_encoder_dtype tests/test_vision.py::test_projector_forward_shape_and_dtype -q
3 passed

uv run python -m py_compile nanochat/vision.py tests/test_vision.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
94 passed, 10 skipped

uv run python -m py_compile nanochat/gpt.py nanochat/vision.py nanochat/flash_attention.py scripts/vlm_train.py modal_vlm.py
ok

uv run python -m pytest -q
102 passed, 10 skipped

git diff --check
ok
```

2026-05-26 local leaky packed-large ablation setup:

- Added a dedicated nanoVLM-style leaky packed-large Modal probe and matching
  CPU batch-plan entrypoint:
  `modal_vlm.py::leaky_packed_large_mfu_probe` and
  `modal_vlm.py::leaky_packed_large_batch_plan`.
- This is intentionally not the default semantic recipe. It uses
  `--allow-leaky-pack`, dense packed rows, `--pack-examples 16`,
  `--max-seq-len 1024`, `--pad-to-bucket-lens 1024`, and the same 65K token cap
  as the boundary-aware large probe. It deliberately does not use
  `--boundary-aware-pack`, `--flatten-packed-batch`, or `--require-fa3-varlen`.
- The purpose is to make the user's nanoVLM hypothesis directly testable after
  Modal is resumed: if boundary-aware varlen attention spends too much time on
  hundreds of short segments, this relaxed dense-row ablation can show whether
  H100 MFU moves closer to text nanochat when rows are long and static.

Validation after leaky packed-large ablation setup:

```text
uv run python -m pytest tests/test_vision.py::test_modal_command_builders -q
1 passed

uv run python -m py_compile modal_vlm.py tests/test_vision.py
ok

uv run python -m pytest tests/test_vision.py::test_modal_command_builders tests/test_vision.py::test_pack_examples_requires_boundary_aware_or_explicit_leaky_opt_in -q
2 passed

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
94 passed, 10 skipped

uv run python -m py_compile nanochat/gpt.py nanochat/vision.py nanochat/flash_attention.py scripts/vlm_train.py modal_vlm.py
ok

uv run python -m pytest -q
102 passed, 10 skipped

git diff --check
ok
```

2026-05-26 local real-data leaky packed-large batch-plan check:

- Ran the CPU-only leaky packed-large batch plan against cached/local
  `HuggingFaceM4/the_cauldron/vqav2` with the H100 ablation settings:
  `batch_size=1024`, `max_batch_tokens=65,536`, `max_seq_len=1024`,
  `pack_examples=16`, `allow_leaky_pack=True`, `pad_to_bucket_lens=1024`,
  `boundary_aware_pack=False`, and `flatten_packed_batch=False`.
- Result: two planned optimizer steps filled the dense static shape at exactly
  64 rows/step, with 63,830 useful tokens out of 65,536 padded tokens and 2.6%
  padding. Average processed examples per step was 416.0.
- The important caveat is attention work: this leaky dense-row shape has
  ~33.59M attention pairs/step, versus ~3.86M/step for the boundary-aware 65K
  max-tokens compact plan above. So the leaky ablation may improve kernel shape
  and static-row utilization, but it is not less attention work; the H100
  comparison should be interpreted as "static dense row efficiency vs much more
  attention FLOPs plus semantic leakage."

Command and output summary:

```text
uv run python -m scripts.vlm_train --device-type cpu --hf-repo HuggingFaceM4/the_cauldron --hf-config vqav2 --device-batch-size 1024 --max-batch-tokens 65536 --max-seq-len 1024 --stream-buffer-size 256 --batch-buffer-size 8192 --bucket-selection max-tokens --pack-examples 16 --pack-max-seq-len 1024 --allow-leaky-pack --pad-to-bucket-lens 1024 --batch-plan-steps 2 --model-step 650

Batch plan source=stream:HuggingFaceM4/the_cauldron/vqav2
steps=2 batch_size=1,024 max_batch_tokens=65,536 grad_accum_steps=1 bucket_lens=[1024] bucket_selection=max-tokens bucket_min_fill_frac=0 bucket_cycle_repeat=1 pack_examples=16 boundary_aware_pack=False flatten_packed_batch=False
planning_elapsed=116.1s records_scanned=8,652 rendered_examples=8,640 rendered_examples/sec=74.4
overall rows/step=64.0 examples/step=416.0 dropped/step=0.0 tokens/step=63830/65536 pad=2.6% attn_pairs/step=33.59M segments/step=64.0 avg_segment=997.3 max_segment=1024 near_cap/step=58.5 cap_hits/step=7.0
```

Validation after leaky packed-large real-data batch-plan note:

```text
uv run python -m pytest tests/test_vision.py::test_modal_command_builders tests/test_vision.py::test_pack_examples_requires_boundary_aware_or_explicit_leaky_opt_in -q
2 passed

uv run python -m pytest -q
102 passed, 10 skipped

git diff --check
ok
```

2026-05-26 local visual-placeholder embedding cleanup:

- `build_multimodal_batch` now builds `input_embeds` for multimodal rows by
  embedding only non-visual positions and then writing projected image features
  into the visual spans. The fallback `value_token_ids` are still kept for
  value-embedding lookups, so model semantics are unchanged, but the batch
  builder no longer performs embedding lookups for visual placeholder tokens
  that are immediately overwritten.
- This is most relevant for the realistic packed probes: in 65K-token batches,
  hundreds of images can mean tens of thousands of visual positions per step.
  Skipping placeholder embedding work reduces batch-construction memory traffic
  without changing targets, RoPE positions, segment boundaries, or the projected
  visual features seen by the LLM.

Validation after visual-placeholder embedding cleanup:

```text
uv run python -m pytest tests/test_vision.py::test_visual_token_insertion_and_target_masking tests/test_vision.py::test_multimodal_batch_allows_packed_image_rows tests/test_vision.py::test_multimodal_batch_backward_reaches_projector_after_image_insert tests/test_vision.py::test_direct_compact_batch_loss_matches_separate_examples -q
4 passed

uv run python -m py_compile nanochat/vision.py tests/test_vision.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
94 passed, 10 skipped

uv run python -m py_compile nanochat/gpt.py nanochat/vision.py nanochat/flash_attention.py scripts/vlm_train.py modal_vlm.py
ok

uv run python -m pytest -q
102 passed, 10 skipped

git diff --check
ok
```

2026-05-26 local dense-path boundary metadata cleanup:

- `build_multimodal_batch` now accepts `return_boundary_metadata=False`. When
  disabled, it skips unused RoPE position IDs, segment IDs, sparse smear-boundary
  indices, `cu_seqlens`, varlen indices, segment-length lists, and attention-pair
  metadata.
- The VLM training loop passes this flag whenever `--boundary-aware-pack` is off.
  Boundary-aware packed runs still build and use the full metadata. Dense and
  leaky packed runs keep their existing `value_token_ids`, targets, lengths, and
  input embeddings, but avoid constructing tensors they never pass into
  `GPT.forward`.
- This is most useful for the new leaky packed-large ablation and any dense
  static bucket probe, where `max_seq_len=1024` and 64 rows/step can otherwise
  spend batch-construction time on boundary metadata that is intentionally unused.

Validation after dense-path boundary metadata cleanup:

```text
uv run python -m pytest tests/test_vision.py::test_multimodal_batch_can_skip_boundary_metadata_for_dense_training tests/test_vision.py::test_boundary_aware_packed_multimodal_loss_matches_separate_examples tests/test_vision.py::test_direct_compact_batch_loss_matches_separate_examples tests/test_vision.py::test_modal_command_builders -q
4 passed

uv run python -m py_compile nanochat/vision.py scripts/vlm_train.py tests/test_vision.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
95 passed, 10 skipped

uv run python -m py_compile nanochat/gpt.py nanochat/vision.py nanochat/flash_attention.py scripts/vlm_train.py modal_vlm.py
ok

uv run python -m pytest -q
103 passed, 10 skipped

git diff --check
ok
```

2026-05-26 local full-CE sparse-loss-list cleanup:

- `build_multimodal_batch` now only builds sparse `loss_indices`/`loss_targets`
  Python lists when `return_loss_indices=True`. Full-CE static probes pass
  `return_loss_indices=False`, so they keep dense targets but skip the redundant
  sparse-list construction.
- `supervised_target_count` is counted directly as targets are emitted and now
  stays correct even when `return_loss_indices=False` and `max_seq_len` clips the
  row. This preserves logging/accounting without paying for sparse tensors that
  the full-CE path never passes to `GPT.forward`.
- This mainly helps fixed-shape or leaky/static ablations that use
  `--no-selective-loss`; boundary-aware selective-loss runs still build the
  sparse loss tensors they need.

Validation after full-CE sparse-loss-list cleanup:

```text
uv run python -m pytest tests/test_vision.py::test_multimodal_batch_can_skip_selective_loss_indices tests/test_vision.py::test_multimodal_batch_can_skip_dense_targets_for_selective_loss tests/test_vision.py::test_multimodal_batch_can_skip_boundary_metadata_for_dense_training tests/test_vision.py::test_boundary_aware_packed_multimodal_loss_matches_separate_examples -q
4 passed

uv run python -m py_compile nanochat/vision.py tests/test_vision.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
95 passed, 10 skipped

uv run python -m py_compile nanochat/gpt.py nanochat/vision.py nanochat/flash_attention.py scripts/vlm_train.py modal_vlm.py tests/test_vision.py
ok

uv run python -m pytest -q
103 passed, 10 skipped

git diff --check
ok
```

2026-05-26 local visual-embedding splice simplification:

- `build_multimodal_batch` now builds `input_embeds` with one normal embedding
  lookup over `value_token_ids`, then overwrites visual spans with projected
  SigLIP features. This removes the previous boolean `visual_mask` allocation and
  irregular non-visual embedding gather/scatter from every VLM batch.
- The existing gradient regression still passes: a loss on visual positions
  reaches the projector and does not create fallback-token embedding gradients.
  Multi-image ordering and truncated visual-span handling are unchanged.
- This is a small batch-construction simplification, not a substitute for the
  missing H100 `steady_mfu` proof. It should help the packed/large probes by
  keeping the multimodal splice closer to the simple VLM implementations used in
  other repos.

Validation after visual-embedding splice simplification:

```text
uv run python -m pytest tests/test_vision.py::test_multimodal_batch_backward_reaches_projector_after_image_insert tests/test_vision.py::test_visual_token_insertion_and_target_masking tests/test_vision.py::test_multimodal_batch_allows_packed_image_rows tests/test_vision.py::test_multimodal_batch_counts_survive_truncated_image_span tests/test_vision.py::test_direct_compact_batch_loss_matches_separate_examples -q
5 passed

uv run python -m py_compile nanochat/vision.py tests/test_vision.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
95 passed, 10 skipped

uv run python -m py_compile nanochat/gpt.py nanochat/vision.py nanochat/flash_attention.py scripts/vlm_train.py modal_vlm.py tests/test_vision.py
ok

uv run python -m pytest -q
103 passed, 10 skipped

git diff --check
ok
```

2026-05-26 local compact smear-index metadata cleanup:

- In the common compact packed path (`--boundary-aware-pack
  --flatten-packed-batch` with one flattened row), `build_multimodal_batch` now
  derives sparse smear reset indices from `cu_seqlens[1:-1] - 1` instead of
  walking segment lengths once for smear indices and again for `cu_seqlens`.
- This preserves the same boundary-aware attention, RoPE reset, and smear reset
  semantics, but removes one redundant Python segment walk from the H100 compact
  packed probe path.
- The compact segment-start regression now explicitly asserts that the sparse
  reset indices equal the `cu_seqlens`-derived form.

Validation after compact smear-index metadata cleanup:

```text
uv run python -m pytest tests/test_vision.py::test_compact_segment_start_indices_derive_from_segment_lengths tests/test_vision.py::test_sparse_segment_start_indices_match_dense_smear_mask tests/test_vision.py::test_compact_boundary_aware_packed_batch_matches_padded_rows tests/test_vision.py::test_direct_compact_batch_loss_matches_separate_examples -q
4 passed

uv run python -m py_compile nanochat/vision.py tests/test_vision.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
95 passed, 10 skipped

uv run python -m pytest -q
103 passed, 10 skipped

uv run python -m py_compile nanochat/gpt.py nanochat/vision.py nanochat/flash_attention.py scripts/vlm_train.py modal_vlm.py tests/test_vision.py
ok

git diff --check
ok
```

2026-05-26 local targetless selective-loss cleanup:

- `GPT.forward(..., selective_loss=True, loss_indices=..., loss_targets=...)`
  no longer allocates a dummy flat target tensor for `loss_reduction="none"` when
  dense `targets` are omitted. It now allocates the output loss vector directly.
- This preserves nanochat's softcapped CE semantics and the existing
  sparse-only selective-loss regression, while keeping the memory-efficient VLM
  loss path free of unnecessary target tensors.

Validation after targetless selective-loss cleanup:

```text
uv run python -m pytest tests/test_vision.py::test_gpt_selective_loss_matches_ignore_index_path tests/test_vision.py::test_multimodal_batch_can_skip_dense_targets_for_selective_loss tests/test_vision.py::test_boundary_aware_packed_multimodal_loss_matches_separate_examples -q
3 passed

uv run python -m py_compile nanochat/gpt.py tests/test_vision.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
95 passed, 10 skipped

uv run python -m pytest -q
103 passed, 10 skipped

uv run python -m py_compile nanochat/gpt.py nanochat/vision.py nanochat/flash_attention.py scripts/vlm_train.py modal_vlm.py tests/test_vision.py
ok

git diff --check
ok
```

2026-05-26 local image-marker pre-scan cleanup:

- `build_multimodal_batch` no longer calls `count_image_tokens(row[:-1])` before
  expanding each row. The expansion loop already visits every token, so this was
  a redundant full row scan on large compact packed batches.
- Over-counts are now caught at the image marker before consuming another row's
  feature, and under-counts are still caught by the end-of-row
  `consumed ... expected ...` assertion. This keeps the same image-feature
  ordering invariant while reducing Python work in the batch builder.

Validation after image-marker pre-scan cleanup:

```text
uv run python -m pytest tests/test_vision.py::test_visual_token_insertion_and_target_masking tests/test_vision.py::test_multimodal_batch_allows_packed_image_rows tests/test_vision.py::test_multimodal_batch_counts_survive_truncated_image_span tests/test_vision.py::test_direct_compact_batch_loss_matches_separate_examples -q
4 passed

uv run python -m py_compile nanochat/vision.py tests/test_vision.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
95 passed, 10 skipped

uv run python -m pytest -q
103 passed, 10 skipped

uv run python -m py_compile nanochat/gpt.py nanochat/vision.py nanochat/flash_attention.py scripts/vlm_train.py modal_vlm.py tests/test_vision.py
ok

git diff --check
ok
```

2026-05-26 local per-image feature-shape check cleanup:

- `build_multimodal_batch` now checks the visual-token dimension once with
  `image_features.size(1) == VISION_TOKENS` before row expansion, instead of
  indexing `image_features[feature_cursor]` for every `<image>` marker only to
  assert that each feature row has 64 visual tokens.
- This removes hundreds of tiny tensor view/index operations from large compact
  packed batches while preserving the same feature-count and feature-order
  assertions.

Validation after per-image feature-shape check cleanup:

```text
uv run python -m pytest tests/test_vision.py::test_visual_token_insertion_and_target_masking tests/test_vision.py::test_multimodal_batch_allows_packed_image_rows tests/test_vision.py::test_multimodal_batch_backward_reaches_projector_after_image_insert tests/test_vision.py::test_flatten_examples_as_compact_batch_keeps_segments_and_feature_order tests/test_vision.py::test_direct_compact_batch_loss_matches_separate_examples -q
5 passed

uv run python -m py_compile nanochat/vision.py tests/test_vision.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
95 passed, 10 skipped

uv run python -m pytest -q
103 passed, 10 skipped

uv run python -m py_compile nanochat/gpt.py nanochat/vision.py nanochat/flash_attention.py scripts/vlm_train.py modal_vlm.py tests/test_vision.py
ok

git diff --check
ok
```

2026-05-26 local untruncated visual-span check cleanup:

- `build_multimodal_batch` now treats `max_seq_len is None` as a guaranteed
  full-span visual insertion case, avoiding a Python scan over every visual span
  in the normal untruncated compact-packed path.
- The existing truncated-span path still checks every span against the retained
  row length, so partially truncated image spans continue to be skipped instead
  of writing outside the retained sequence.

Validation after untruncated visual-span check cleanup:

```text
uv run python -m pytest tests/test_vision.py::test_visual_token_insertion_and_target_masking tests/test_vision.py::test_multimodal_batch_counts_survive_truncated_image_span tests/test_vision.py::test_multimodal_batch_can_skip_lengths_tensor_with_truncated_image_span tests/test_vision.py::test_direct_compact_batch_loss_matches_separate_examples -q
4 passed

uv run python -m py_compile nanochat/vision.py tests/test_vision.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
99 passed, 10 skipped

uv run python -m py_compile nanochat/gpt.py nanochat/vision.py nanochat/flash_attention.py scripts/vlm_train.py modal_vlm.py tests/test_vision.py
ok

git diff --check
ok

uv run python -m pytest -q
107 passed, 10 skipped
```

2026-05-26 Modal H100 strict-varlen vs leaky packed attention check:

- The 65K strict boundary-aware compact varlen probe reached H100 with FA3
  varlen available, then aborted during the first transformer pass before any
  optimizer-step metrics. The failed shape was `--device-batch-size 1024`,
  `--max-batch-tokens 65536`, `--pack-examples 16`,
  `--pack-max-seq-len 1024`, `--boundary-aware-pack`,
  `--flatten-packed-batch`, and `--require-fa3-varlen`.
- The 32K strict compact varlen probe also aborted before step metrics after
  setup. Setup allocation was about 11.3 GiB, so the failure happens on the
  first real packed training batch rather than model load.
- The known-stable 16K token budget does run. Clean no-profile H100 run:

```text
NANOCHAT_MODAL_GPU=H100 uv run --extra vision modal run modal_vlm.py::mfu_probe \
  --num-iterations 6 --batch-size 512 --max-batch-tokens 16000 \
  --max-seq-len 1024 --batch-buffer-size 2048 \
  --bucket-selection max-tokens --prefetch-batches 4 --prefetch-workers 2 \
  --pack-examples 8 --pack-max-seq-len 1024 \
  --boundary-aware-pack --flatten-packed-batch --require-fa3-varlen \
  --no-profile-timing
```

Result:

```text
GPU: NVIDIA H100 80GB HBM3
FA3 varlen: has_fa3_varlen=True, varlen_backend=fa3
steady_mfu=22.18 over 4 post-warmup steps
steady_padded_mfu=22.18
steady_attn_pairs/token=61.49
steady_avg_segment=121.9 p50_segment=122 p90_segment=127 max_segment=128
tokens/sec on post-warmup steps ~= 22.3K
peak_memory=63,593 MiB
```

The process printed all metrics, then the Modal subprocess exited nonzero with
`PyGILState_Release` during shutdown. Treat the step metrics as valid but the
entrypoint exit status as dirty.

Matched no-profile leaky dense packed ablation:

```text
NANOCHAT_MODAL_GPU=H100 uv run --extra vision modal run modal_vlm.py::mfu_probe \
  --num-iterations 6 --batch-size 512 --max-batch-tokens 16000 \
  --max-seq-len 1024 --batch-buffer-size 2048 \
  --bucket-selection max-tokens --prefetch-batches 4 --prefetch-workers 2 \
  --pack-examples 8 --pack-max-seq-len 1024 \
  --allow-leaky-pack --pad-to-bucket-lens 1024 \
  --no-profile-timing
```

Result:

```text
GPU: NVIDIA H100 80GB HBM3
steady_mfu=22.47 over 4 post-warmup steps
steady_padded_mfu=23.37
bucket steady tokens/sec=21.0K
pad=3.9%
steady_attn_pairs/token=533.18
steady_avg_segment=984.3 p50_segment=1015 p90_segment=1024 max_segment=1024
peak_memory=62,786 MiB
```

Interpretation:

- On identical H100 class and the same 16K token cap, leaky dense attention
  reports slightly higher MFU (`22.47%` vs `22.18%`) because it performs and
  counts much more cross-example attention work (`533` vs `61`
  attention-pairs/token).
- Strict compact varlen is slightly better on useful token throughput
  (`~22.3K` vs `21.0K` tokens/sec), has 0% padding, and preserves the intended
  packed semantics.
- Therefore naive/leaky attention is not a real win for this 16K shape. It
  inflates attention work and the MFU numerator, but does not process useful
  VLM tokens faster.
- The current hard blocker for higher MFU is memory/activation headroom: strict
  32K/65K compact varlen does not reach step metrics yet. We are not using
  activation checkpointing; the next real work should stay simple and focus on
  token-budget headroom, static-shape efficiency, or other common VLM packing
  levers rather than relaxing boundaries.

Profiled 16K sanity pair:

```text
strict varlen profile: steady_mfu=20.81, peak_memory=63,593 MiB
leaky dense profile:  steady_mfu=21.74, peak_memory=62,786 MiB
```

2026-05-26 Cauldron/VQAv2 expanded length check:

- The current Modal MFU probes use `HuggingFaceM4/the_cauldron` with
  `hf_config=vqav2`, so the directly relevant length scan is VQAv2 inside The
  Cauldron.
- A 1,000 usable one-image VQAv2 scan with the repo tokenizer and 64-token image
  expansion found max expanded length `1,099`. This is above the current
  `--pack-max-seq-len 1024`, so the 1,024 cap is not a true dataset maximum.

```text
uv run python -m scripts.vlm_train --device-type cpu \
  --hf-repo HuggingFaceM4/the_cauldron --hf-config vqav2 \
  --max-seq-len 16384 --length-stats-examples 1000 \
  --length-stats-max-records 2000 --stream-buffer-size 0 --model-step 650

records_scanned=1,000 usable_one_image_examples=1,000
expanded_len min/p50/p80/p90/p95/p99/max/mean
108/132/177/225/292/518/1,099/159.8
fit_at_max_seq_len_16,384=1,000/1,000 (100.0%)
```

- An attempted all-config Cauldron scan did not finish quickly enough to support
  an exact all-50-config maximum. Do not cite `1,099` as the global Cauldron
  maximum; it is the measured max for the first 1,000 usable VQAv2 examples.

2026-05-26 local per-step GPU memory telemetry:

- Added `cuda_memory_stats_mib(...)` and wired setup, per-step, and final CUDA
  memory stats into the trainer logs.
- Per-step W&B keys now include `gpu/allocated_mib`, `gpu/reserved_mib`,
  `gpu/max_allocated_mib`, and `gpu/max_reserved_mib`; setup and final logs also
  record allocated/reserved memory. Stdout prints
  `mem alloc/peak/reserved ... MiB` on the training line.
- This gives the next H100 packed probes memory headroom/OOM context next to
  `steady_mfu`, segment percentiles, and timing. It does not change the MFU
  denominator or training behavior.

Validation after per-step GPU memory telemetry:

```text
uv run python -m pytest tests/test_vision.py::test_cuda_memory_stats_cpu_reports_zeroes tests/test_vision.py::test_profile_includes_split_optimizer_timing_keys tests/test_vision.py::test_profile_summary_reports_warmup_excluded_timing_percentages -q
3 passed

uv run python -m py_compile nanochat/gpt.py nanochat/vision.py nanochat/flash_attention.py scripts/vlm_train.py modal_vlm.py tests/test_vision.py
ok

git diff --check
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
100 passed, 10 skipped

uv run python -m pytest -q
108 passed, 10 skipped
```

2026-05-26 local profile-timing batch sync fix:

- The `--profile-timing` path now synchronizes before recording the `batch`
  bucket after `build_multimodal_batch(...)` returns.
- This charges asynchronous multimodal tensor construction, visual-token scatter,
  and post-projector batch work to `batch` instead of leaving it in the residual
  `other` bucket before forward/backward timing starts.
- Normal non-profile training is unchanged. The extra synchronization is only in
  profile runs used for H100 bottleneck attribution.

Validation after profile-timing batch sync fix:

```text
uv run python -m py_compile scripts/vlm_train.py
ok

uv run python -m pytest tests/test_vision.py::test_visual_token_insertion_and_target_masking tests/test_vision.py::test_profile_summary_reports_pack_timing tests/test_vision.py::test_profile_includes_split_optimizer_timing_keys tests/test_vision.py::test_profile_summary_reports_warmup_excluded_timing_percentages -q
4 passed

git diff --check
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
99 passed, 10 skipped

uv run python -m pytest -q
107 passed, 10 skipped
```

2026-05-26 Modal H100 no-checkpoint strict packed token-budget bracket:

- Confirmed this path has no activation checkpointing. This matches the
  text-only nanochat training style and keeps the implementation simple.
- Re-ran strict boundary-aware compact varlen with `max_seq_len=2048` and
  `pack_max_seq_len=2048` so the VQAv2 measured max of `1,099` expanded tokens
  is not clipped by a `1024` cap.
- Stable 20K useful-token command:

```text
NANOCHAT_MODAL_GPU=H100 uv run --extra vision modal run modal_vlm.py::mfu_probe \
  --num-iterations 6 --batch-size 768 --max-batch-tokens 20000 \
  --max-seq-len 2048 --batch-buffer-size 4096 \
  --bucket-selection max-tokens --prefetch-batches 4 --prefetch-workers 2 \
  --pack-examples 8 --pack-max-seq-len 2048 \
  --boundary-aware-pack --flatten-packed-batch --require-fa3-varlen \
  --no-profile-timing
```

Result:

```text
GPU: NVIDIA H100 80GB HBM3
FA3 varlen: has_fa3_varlen=True, varlen_backend=fa3, cuda_capability=sm90
Params total/trainable: 4,051,700,826/1,904,214,106
Estimated LLM FLOPs/token: 1.207961e+10 | Peak BF16 FLOPS: 9.89e+14
steady_mfu=25.26 over 4 post-warmup steps
steady_padded_mfu=25.26
peak_memory=75,398 MiB
step 3: tokens=20,000 segments=170 avg_segment=117.6 attn_pairs/token=59.32 eff_mfu=23.34
step 4: tokens=19,999 segments=20 avg_segment=1000.0 attn_pairs/token=530.76 eff_mfu=29.71
step 5: tokens=19,971 segments=169 avg_segment=118.2 attn_pairs/token=59.59 eff_mfu=24.31
step 6: tokens=19,996 segments=169 avg_segment=118.3 attn_pairs/token=59.66 eff_mfu=24.26
```

- 22K useful-token command with the same simple strict path:

```text
NANOCHAT_MODAL_GPU=H100 uv run --extra vision modal run modal_vlm.py::mfu_probe \
  --num-iterations 6 --batch-size 768 --max-batch-tokens 22000 \
  --max-seq-len 2048 --batch-buffer-size 4096 \
  --bucket-selection max-tokens --prefetch-batches 4 --prefetch-workers 2 \
  --pack-examples 8 --pack-max-seq-len 2048 \
  --boundary-aware-pack --flatten-packed-batch --require-fa3-varlen \
  --no-profile-timing
```

Result:

```text
GPU: NVIDIA H100 80GB HBM3
step 1: tokens=21,900 peak_memory=70,927 MiB eff_mfu=0.36
step 2: tokens=21,928 peak_memory=78,833 MiB eff_mfu=21.72
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 470.00 MiB.
GPU total=79.18 GiB, free=256.31 MiB, process in use=78.91 GiB.
PyTorch allocated=76.54 GiB, reserved but unallocated=1.63 GiB.
```

Interpretation:

- The best correct no-checkpoint BVLM number so far is `25.26%` steady MFU at
  20K useful tokens on H100.
- The next token-budget step, 22K, is right at the 80 GB memory edge and OOMs
  during backward. That brackets the current simple strict path: the remaining
  gap to text-only nanochat is not image fetching first; it is mostly that BVLM
  cannot yet run the text nanochat-sized token batch.
- Step 4 in the 20K run reached `29.71%` MFU when the packed batch happened to
  contain longer segments (`avg_segment=1000.0`, `attn_pairs/token=530.76`).
  The other steady steps were dominated by many short segments around 118
  tokens (`attn_pairs/token ~= 59`). This points to a second gap after memory:
  strict VLM varlen batches are often less kernel- and FLOP-dense than
  text-only fixed `[B,T]` batches even at the same useful-token count.

Follow-up strict max-compute diagnostic at 20K useful tokens:

```text
NANOCHAT_MODAL_GPU=H100 uv run --extra vision modal run modal_vlm.py::mfu_probe \
  --num-iterations 6 --batch-size 768 --max-batch-tokens 20000 \
  --max-seq-len 2048 --batch-buffer-size 4096 \
  --bucket-selection max-compute --prefetch-batches 4 --prefetch-workers 2 \
  --pack-examples 8 --pack-max-seq-len 2048 \
  --boundary-aware-pack --flatten-packed-batch --require-fa3-varlen \
  --no-profile-timing
```

Result:

```text
GPU: NVIDIA H100 80GB HBM3
steady_mfu=25.41 over 4 post-warmup steps
steady_padded_mfu=25.41
steady_attn_pairs/token=163.11
steady_avg_segment=273.9 steady_p50_segment=240 steady_p90_segment=467 steady_max_segment=1,585
peak_memory=75,286 MiB
```

This is essentially tied with the max-token 20K run (`25.26%`). Selecting
compute-heavier packed windows alone does not close the gap to text-only
nanochat. It raises attention intensity but the useful-token cap and non-static
VLM step still dominate.

Local allocator parity with text nanochat:

- Added `os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")`
  to the VLM trainer before importing/initializing CUDA-heavy modules. This
  mirrors the allocator setup already used in `scripts/base_train.py` and
  `scripts/chat_sft.py`.
- Validation:

```text
uv run python -m py_compile scripts/vlm_train.py
ok

uv run python -m pytest tests/test_vision.py::test_modal_command_builders tests/test_vision.py::test_cuda_memory_stats_cpu_reports_zeroes -q
2 passed

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
100 passed, 10 skipped

git diff --check
ok
```

22K strict rerun after allocator parity:

```text
NANOCHAT_MODAL_GPU=H100 uv run --extra vision modal run modal_vlm.py::mfu_probe \
  --num-iterations 6 --batch-size 768 --max-batch-tokens 22000 \
  --max-seq-len 2048 --batch-buffer-size 4096 \
  --bucket-selection max-tokens --prefetch-batches 4 --prefetch-workers 2 \
  --pack-examples 8 --pack-max-seq-len 2048 \
  --boundary-aware-pack --flatten-packed-batch --require-fa3-varlen \
  --no-profile-timing
```

Result:

```text
GPU: NVIDIA H100 NVL
Peak BF16 FLOPS: 8.35e+14
steady_mfu=22.51 over 4 post-warmup steps
steady_padded_mfu=22.51
steady_attn_pairs/token=60.52
steady_avg_segment=120.0 steady_p50_segment=120 steady_p90_segment=122 steady_max_segment=123
peak_memory=78,967 MiB allocated, 84,404 MiB reserved
```

This shows 22K can run when Modal lands on H100 NVL memory/headroom, but it is
not an apples-to-apples improvement over the `H100 80GB HBM3` 20K baseline.
The measured MFU is lower because this batch stream is short-segment dominated
and the hardware denominator differs. The important signal is that 22K is still
memory/headroom-limited on the 80GB-HBM3 run, not data-loader-limited.

2026-05-26 correction: strict-boundary segments vs Karpathy-style text packing:

- The earlier "many short segments" explanation only applies to the strict
  boundary-aware compact varlen path. It is not a claim that short records are
  inherently inefficient.
- nanochat text packing fills fixed dense `[B, T]` rows. Its pretraining loader
  best-fits documents into full rows, and its SFT loader best-fits conversations
  into rows while masking loss. Neither path carries block-diagonal attention
  metadata, RoPE resets, or smear resets for every packed record. The attention
  kernel sees ordinary dense rows.
- Therefore, if we intentionally allow Karpathy-style packed-row semantics for
  VLM, short VQA examples plus 64 image tokens should also become dense 2048
  token rows. That is a valid throughput ablation, but it is not the same as the
  strict boundary-aware training path required by the original goal prompt.

Karpathy-style dense/leaky VLM packing at 20K on H100 80GB HBM3:

```text
NANOCHAT_MODAL_GPU=H100 uv run --extra vision modal run modal_vlm.py::mfu_probe \
  --num-iterations 6 --batch-size 768 --max-batch-tokens 20000 \
  --max-seq-len 2048 --batch-buffer-size 4096 \
  --bucket-selection max-tokens --prefetch-batches 4 --prefetch-workers 2 \
  --pack-examples 16 --pack-max-seq-len 2048 \
  --allow-leaky-pack --pad-to-bucket-lens 2048 \
  --no-profile-timing
```

Result:

```text
GPU: NVIDIA H100 80GB HBM3
boundary_aware_pack=False flatten_packed_batch=False pad_buckets=[2048]
steady_mfu=26.98 over 4 post-warmup steps
steady_padded_mfu=27.57
steady_attn_pairs/token=1047.23
steady_avg_segment=2003.6 steady_p50_segment=2009 steady_p90_segment=2048 steady_max_segment=2048
pad=2.2%
peak_memory=70,507 MiB
```

This supports the user's correction: when we allow normal dense text-style
packing, the short-segment issue disappears and rows are near 2048 tokens. The
MFU improves over strict 20K (`25.26%`) but only to `26.98%`, so the remaining
gap is not caused primarily by short sentences.

Karpathy-style dense/leaky VLM packing at 22.5K on H100 80GB HBM3:

```text
NANOCHAT_MODAL_GPU=H100 uv run --extra vision modal run modal_vlm.py::mfu_probe \
  --num-iterations 6 --batch-size 768 --max-batch-tokens 22528 \
  --max-seq-len 2048 --batch-buffer-size 4096 \
  --bucket-selection max-tokens --prefetch-batches 4 --prefetch-workers 2 \
  --pack-examples 16 --pack-max-seq-len 2048 \
  --allow-leaky-pack --pad-to-bucket-lens 2048 \
  --no-profile-timing
```

Result:

```text
GPU: NVIDIA H100 80GB HBM3
step 1: tokens=21,955/22,528 rows=11 attn_pairs/token=1051.24 peak_memory=72,369 MiB eff_mfu=0.53
torch.OutOfMemoryError: CUDA out of memory during selective lm_head/logit softcap on step 2.
Tried to allocate 620.00 MiB. GPU total=79.18 GiB, free=128.31 MiB.
PyTorch allocated=77.66 GiB, reserved but unallocated=662.46 MiB.
```

Karpathy-style dense/leaky VLM packing at 32K on H100 80GB HBM3:

```text
NANOCHAT_MODAL_GPU=H100 uv run --extra vision modal run modal_vlm.py::mfu_probe \
  --num-iterations 6 --batch-size 768 --max-batch-tokens 32768 \
  --max-seq-len 2048 --batch-buffer-size 4096 \
  --bucket-selection max-tokens --prefetch-batches 4 --prefetch-workers 2 \
  --pack-examples 16 --pack-max-seq-len 2048 \
  --allow-leaky-pack --pad-to-bucket-lens 2048 \
  --no-profile-timing
```

Result:

```text
GPU: NVIDIA H100 80GB HBM3
torch.OutOfMemoryError: CUDA out of memory in first transformer pass, before step metrics.
Tried to allocate 64.00 MiB. GPU total=79.18 GiB, free=40.31 MiB.
PyTorch allocated=78.45 GiB.
```

Karpathy-style dense/leaky VLM packing at 24K on H100 NVL:

```text
GPU: NVIDIA H100 NVL
Peak BF16 FLOPS: 8.35e+14
steady_mfu=25.98 over 4 post-warmup steps
steady_padded_mfu=26.64
steady_attn_pairs/token=1051.10
steady_avg_segment=1996.2 steady_p50_segment=2030 steady_p90_segment=2048 steady_max_segment=2048
pad=2.5%
peak_memory=87,117 MiB
```

This is not comparable to the H100 80GB HBM3 denominator or memory limit, but it
confirms that dense text-style rows need substantially more than 80GB once the
token budget gets into the mid-20K range.

20K strict boundary-aware profile timing:

```text
NANOCHAT_MODAL_GPU=H100 uv run --extra vision modal run modal_vlm.py::mfu_probe \
  --num-iterations 6 --batch-size 768 --max-batch-tokens 20000 \
  --max-seq-len 2048 --batch-buffer-size 4096 \
  --bucket-selection max-tokens --prefetch-batches 4 --prefetch-workers 2 \
  --pack-examples 8 --pack-max-seq-len 2048 \
  --boundary-aware-pack --flatten-packed-batch --require-fa3-varlen \
  --profile-timing
```

Result:

```text
GPU: NVIDIA H100 80GB HBM3
steady_mfu=24.90 over 4 post-warmup steps
peak_memory=75,398 MiB
Steady timing totals wall=3.257s
data_wait=0.002s/0.1%
pack=0.004s/0.1%
batch=0.031s/0.9%
batch_projector=0.008s/0.3%
fwdbwd=1.981s/60.8%
optim=0.693s/21.3%
```

The profile also prints image preprocessing/SigLIP totals greater than wall time
because those counters include overlapped prefetch-worker work. The important
stall metric is `data_wait`, which is only `0.1%`. So the current MFU gap is not
image fetching. The serialized step time is dominated by LLM forward/backward
and optimizer, while memory prevents increasing the H100 80GB token budget to
nanochat's `32 x 2048 = 65,536` text microbatch.

Why same-H100 MFU is still lower than text nanochat:

- The MFU denominator already adjusts for GPU count, so the gap is not explained
  by `1xH100` versus `8xH100` arithmetic.
- The per-GPU workload and memory layout are not equivalent. Text nanochat
  trains dense `32 x 2048 = 65,536` tokens per GPU with a fixed static shape.
  The current VLM H100 80GB runs fit only about 20K useful tokens in either
  strict varlen or dense text-style packing before hitting the memory edge.
- Text nanochat distributed training uses `DistMuonAdamW`, whose optimizer state
  is sharded across ranks. The VLM trainer currently asserts single-GPU training
  and therefore uses single-GPU `MuonAdamW`, with all optimizer state on one
  H100. MFU normalizes compute by GPU count, but it does not make the optimizer
  memory footprint per GPU equivalent.
- Text nanochat compiles a fixed-shape model with `torch.compile(dynamic=False)`.
  The VLM selective-loss compile path currently uses `dynamic=True`, which did
  not reach a first training step within several minutes in the 20K dense probe
  before the run was stopped.
- Published strongest nanochat text references also include FP8/FA3 speedrun
  settings. The VLM path here is still BF16.

Selective-loss chunking follow-up:

- Added `loss_chunk_size` support to the selective-loss path in `GPT.forward`.
  This preserves the same lm_head, logit softcap, and CE semantics, but computes
  supervised-target logits in smaller chunks.
- Local validation:

```text
uv run python -m pytest tests/test_vision.py::test_gpt_selective_loss_matches_ignore_index_path tests/test_vision.py::test_modal_command_builders -q
2 passed

uv run python -m py_compile nanochat/gpt.py nanochat/vision.py nanochat/flash_attention.py scripts/vlm_train.py modal_vlm.py tests/test_vision.py
ok

git diff --check
ok
```

22.5K dense/leaky rerun with selective loss chunking:

```text
NANOCHAT_MODAL_GPU=H100 uv run --extra vision modal run modal_vlm.py::mfu_probe \
  --num-iterations 6 --batch-size 768 --max-batch-tokens 22528 \
  --max-seq-len 2048 --batch-buffer-size 4096 \
  --bucket-selection max-tokens --prefetch-batches 4 --prefetch-workers 2 \
  --pack-examples 16 --pack-max-seq-len 2048 \
  --allow-leaky-pack --pad-to-bucket-lens 2048 \
  --loss-chunk-size 512 \
  --no-profile-timing
```

Result:

```text
GPU: NVIDIA H100 80GB HBM3
step 1: tokens=21,955/22,528 rows=11 peak_memory=72,843 MiB
torch.OutOfMemoryError: CUDA out of memory on step 2 inside lm_head weight cast.
Tried to allocate 256.00 MiB. GPU total=79.18 GiB, free=14.31 MiB.
PyTorch allocated=77.82 GiB, reserved but unallocated=608.50 MiB.
```

This shows the previous full supervised-logits tensor was not the only memory
blocker. At 22.5K dense tokens, the model has essentially no headroom left even
for the bf16 cast of the full `lm_head` weight used by the custom `Linear`
wrapper. Smaller label chunks do not solve that because each chunk still needs
the full vocabulary projection matrix.

20K dense/leaky compile probe:

```text
NANOCHAT_MODAL_GPU=H100 uv run --extra vision modal run modal_vlm.py::mfu_probe \
  --num-iterations 8 --batch-size 768 --max-batch-tokens 20000 \
  --max-seq-len 2048 --batch-buffer-size 4096 \
  --bucket-selection max-tokens --prefetch-batches 4 --prefetch-workers 2 \
  --pack-examples 16 --pack-max-seq-len 2048 \
  --allow-leaky-pack --pad-to-bucket-lens 2048 \
  --compile-model \
  --no-profile-timing
```

Result:

```text
GPU: NVIDIA H100 80GB HBM3
Compiled LLM with torch.compile(dynamic=True)
Stopped after about 3.7 minutes without first step metrics.
```

This is a negative result for the current selective-loss compile path. It does
not reproduce nanochat text's `dynamic=False` fixed-shape compile behavior. A
real compile-based MFU improvement likely needs a separate static trunk compile
path or a static-shaped loss interface, not simply passing `--compile-model` to
the current dynamic selective-loss graph.

2026-05-26 local projector-forward timing split:

- `--profile-timing` now records `batch_projector` separately from the broader
  `batch` bucket. The broader bucket still measures the full multimodal batch
  construction wall time, while `batch_projector` is a sub-bucket for the
  trainable projector forward on pooled SigLIP features.
- The normal training path is unchanged; the extra synchronization only happens
  when profile timing is enabled.
- This makes future H100 attribution sharper: a low packed `steady_mfu` can now
  distinguish Python/tensor batch construction from projector-forward overhead
  before blaming FA3 varlen attention or the LLM trunk.

Validation after projector-forward timing split:

```text
uv run python -m pytest tests/test_vision.py::test_visual_token_insertion_and_target_masking tests/test_vision.py::test_profile_summary_reports_pack_timing tests/test_vision.py::test_profile_includes_split_optimizer_timing_keys -q
3 passed

uv run python -m py_compile nanochat/gpt.py nanochat/vision.py nanochat/flash_attention.py scripts/vlm_train.py modal_vlm.py tests/test_vision.py
ok

git diff --check
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
99 passed, 10 skipped

uv run python -m pytest -q
107 passed, 10 skipped
```

2026-05-26 local explicit MFU metric aliases:

- Added explicit W&B aliases for the goal prompt's MFU terms without changing
  the underlying calculation:
  - `train/eff_llm_mfu` mirrors the existing useful-token `train/mfu`.
  - `train/padded_llm_mfu` mirrors `train/padded_mfu`.
  - `train/steady_eff_llm_mfu` mirrors `train/steady_mfu`.
  - `train/steady_padded_llm_mfu` mirrors `train/steady_padded_mfu`.
  - Bucket-steady aliases mirror the existing bucket steady values.
- This keeps old dashboards compatible while making future H100 evidence easier
  to compare directly against the success criteria.

Validation after explicit MFU metric aliases:

```text
uv run python -m py_compile scripts/vlm_train.py
ok

uv run python -m pytest tests/test_vision.py::test_bucket_steady_metrics_accumulate_by_static_shape tests/test_vision.py::test_flattened_packed_batch_plan_reports_compact_tokens -q
2 passed

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
99 passed, 10 skipped

uv run python -m py_compile nanochat/gpt.py nanochat/vision.py nanochat/flash_attention.py scripts/vlm_train.py modal_vlm.py tests/test_vision.py
ok

git diff --check
ok

uv run python -m pytest -q
107 passed, 10 skipped
```

2026-05-26 local attention-intensity diagnostic:

- Added `attn_pairs/token` to CPU batch-plan output, optimizer-step summaries,
  training stdout, W&B metrics, and bucket-steady summaries.
- This is the attention-work intensity number needed to interpret future H100
  packed probes. A 65K useful-token batch with many short segments can have far
  less causal attention work per token than a 65K batch with long segments, so
  `tokens/sec` and total `attn_pairs/step` alone are not enough to explain
  `steady_mfu`.
- New W&B keys include `train/attention_pairs_per_token`,
  `train/steady_attention_pairs_per_token`, and
  `train/bucket_steady_attention_pairs_per_token`.

Validation after attention-intensity diagnostic:

```text
uv run python -m pytest tests/test_vision.py::test_bucket_steady_metrics_accumulate_by_static_shape tests/test_vision.py::test_flattened_packed_batch_plan_reports_compact_tokens -q
2 passed

uv run python -m py_compile scripts/vlm_train.py tests/test_vision.py
ok

uv run python -m pytest tests/test_attention_fallback.py tests/test_vision.py tests/test_vlm_smoke.py -q
99 passed, 10 skipped

uv run python -m py_compile nanochat/gpt.py nanochat/vision.py nanochat/flash_attention.py scripts/vlm_train.py modal_vlm.py tests/test_vision.py
ok

git diff --check
ok

uv run python -m pytest -q
107 passed, 10 skipped
```

2026-05-26 Modal strict-packed FP8 probe:

- Added a default-off `--fp8 --fp8-recipe tensorwise` path that reuses
  nanochat's existing FP8 linear wrapper for eligible LLM `nn.Linear` modules.
  This covers transformer attention projections (`q/k/v/o`) and MLP projections
  (`c_fc/c_proj`). It does not make FA3 varlen attention, softmax, norms,
  embeddings, SigLIP, or the vision projector FP8.
- The VLM selective-loss path has arbitrary supervised-token counts, so
  `nanochat/fp8.py` now pads only the internal FP8 matmul rows to a multiple of
  16 and slices the result back. This preserves sequence boundaries and loss
  semantics.
- Result on strict boundary-aware 20K packed H100 probe:
  - FP8 all eligible LLM linears including `lm_head`: converted 193/210
    linears, `steady_mfu=13.99`, peak memory 69,969 MiB.
  - FP8 transformer linears with selective-loss `lm_head` left BF16: converted
    192/210 linears, `steady_mfu=14.71`, peak memory 70,100 MiB.
  - BF16 strict-packed 20K baseline remains much better at
    `steady_mfu=25.26`.
- Conclusion: FP8 is not the current MFU lever for BVLM. The minimal tensorwise
  FP8 wrapper adds scale/cast/padding overhead on these strict-varlen shapes,
  while the VLM only adds SigLIP/projector work on top of the mostly unchanged
  LLM trunk. Keep FP8 default-off unless a fused/static compile path changes
  the economics.

2026-05-26 Modal strict-packed BF16 grad-accum probe:

- Tried the same strict boundary-aware 20K packed H100 probe with
  `--grad-accum-steps 2` to amortize optimizer overhead.
- Stopped after several minutes without first-step metrics. Treat this as a
  negative/blocked probe for now, not evidence of improvement.

2026-05-26 Modal strict-packed memory investigation:

- Added a default-off `--siglip-forward-batch-size` knob for frozen SigLIP
  inference chunking. A 22K strict boundary-aware H100 probe with
  `--siglip-forward-batch-size 32` still OOMed in LLM backward after step 2:
  setup allocation was 11,289.65 MiB, step-2 peak was 78,833 MiB, and PyTorch
  failed on a 470 MiB allocation. This is essentially the same failure mode as
  the unchunked 22K run, so frozen SigLIP activation/feature memory is not the
  decisive peak-memory blocker.
- The base nanochat checkpoint used for VLM initialization is missing all
  value-embedding keys. The compatibility loader patches
  2,147,483,648 `value_embeds.*` parameters and 3,072 value-gate parameters to
  zero. Because the current VLM path freezes those tensors and the gates are
  zero, the value-embedding path is exactly dead for this checkpoint.
- Added a default-off `--drop-zero-value-embeds` path that records the loader's
  compatibility metadata and then removes the dead value-embedding modules and
  zero gates before optimizer construction when all expected value embeddings
  were compatibility-patched. This reduced VLM setup allocation from about
  11.3 GiB to 7.2 GiB.
- 22K strict boundary-aware packed BF16 with `--drop-zero-value-embeds` now
  completes:
  - 4-step probe: peak 73,528.24 MiB, `steady_mfu=24.91` over 2 post-warmup
    steps.
  - 6-step probe: peak 73,535.55 MiB, `steady_mfu=23.55` over 4 post-warmup
    steps, `steady_attn_pairs/token=60.46`.
- 24K strict boundary-aware packed BF16 with max-token bucket selection and
  `--drop-zero-value-embeds` completed a 4-step probe, but was near the memory
  edge: peak 78,495.74 MiB and `steady_mfu=22.21`.
- 24K strict boundary-aware packed BF16 with max-compute bucket selection and
  `--drop-zero-value-embeds` OOMed after the first step. The first step had
  23,973 tokens, 27 segments, average segment length 887.9, p90 1024, max
  1535, and `attn_pairs/token=468.94`; the next backward failed on a 986 MiB
  allocation with only 624 MiB free.

Current conclusion:

- The original 22K ceiling was partly self-inflicted memory waste from dead
  value-embedding tables. Dropping them is correct for the patched-zero base
  checkpoint and raises the stable strict-packed ceiling from roughly 20K to
  22K useful tokens/step, with some 24K short-segment batches fitting.
- The remaining 65K-vs-22K gap is not explained by SigLIP pooled features or
  frozen SigLIP forward activations. Those are hundreds of MiB at the observed
  184-202 images/step scale, while the failure happens in the trainable LLM
  backward near 73-79 GiB peak.
- The VLM strict-packed path is still not equivalent to nanochat text's 65K
  token/rank hot path. Text training uses fixed `[device_batch_size,
  max_seq_len]` dense tensors and `torch.compile(dynamic=False)`. The current
  VLM path uses dynamic flattened varlen batches, FA3 varlen boundaries,
  selective supervised-token loss, single-GPU optimizer state for these probes,
  and BF16 linears. Proper packing fixed wasted pad tokens, but it did not
  make the memory and compiler regime identical to the dense text run.

2026-05-26 simplification handoff:

- Keep strict boundary-aware varlen FlashAttention 3. It is the best-correct
  attention path found: segment boundaries are real, RoPE resets per segment,
  smear does not cross segments, and H100 uses FA3 varlen instead of the dense
  SDPA fallback.
- Keep the minimal memory-safe selective VLM loss. It fixed the original full
  logits OOM without changing softcapped CE semantics.
- Do not carry FP8 into the minimal V0 branch. The tested tensorwise FP8 path
  lowered strict-packed MFU from the BF16 baseline instead of improving it.
- Do not carry SigLIP forward chunking into the minimal V0 branch. Chunking
  did not change the 22K OOM failure mode; the peak failure was in trainable
  LLM backward, not frozen SigLIP inference.
- Do not carry broad profiling/MFU-grid machinery into minimal V0. The useful
  findings are now recorded here; the experiment branch keeps the knobs and
  instrumentation for later analysis.
