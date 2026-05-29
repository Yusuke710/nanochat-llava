# nanochat-llava v0 run notes

Current source of truth: this top-level snapshot plus `RUNBOOK_GPU.md`.
Older cache/precompute, mem100, fixture, preflight, and streamed-offset paths
were intentionally removed to keep the implementation minimal.

## Current live code snapshot

- `nanochat/vision.py`: `<image>` marker handling, frozen SigLIP base patch-16/512, nanoVLM-style 8x8 pixel-shuffle pooling to 64 visual tokens, linear projector, vectorized visual-token insertion, target masking, generation helper, VLM checkpoint helpers, and HF nanochat-d32 linking.
- `nanochat/gpt.py`: thin optional `input_embeds` / `value_token_ids` / varlen-boundary / target-index hooks in `GPT.forward`; ordinary text-only `model(idx, targets)` behavior is preserved.
- `nanochat/checkpoint_manager.py`: compatibility patching for old `karpathy/nanochat-d32` checkpoint keys missing from the current GPT module.
- `scripts/vlm_train.py`: one visual-instruction trainer. It starts from `karpathy/nanochat-d32`, freezes SigLIP, trains the linear projector plus nanochat, and uses a fixed nanochat-style packed shape controlled by `--device-batch-size` and `--max-seq-len`. It supports mixed text-only, single-image, and multi-image JSON/HF rows with `--max-batch-images`, and keeps SigLIP inline in the training step.
- `scripts/vlm_eval.py`: verifier subset runner for MMStar, ScienceQA, ChartQA, MMMU, and TextVQA. It exposes `evaluate_vlm(...)` for training-time checks, and the CLI evaluates one checkpoint, stores scores and sample generations, and leaves checkpoint-to-checkpoint comparisons outside the script.
- `tests/test_vision.py` and `tests/test_vlm_smoke.py`: focused unit tests plus synthetic image-conditioned overfit/control smoke. The smoke now lives in tests, not scripts.
- `modal_vlm.py`: minimal Modal wrapper with `doctor`, `smoke`, `train`, and `eval` only. Default GPU is `A100-80GB`; set `NANOCHAT_MODAL_GPU=H100` to switch.
- `RUNBOOK_GPU.md`: external-GPU runbook with one train command, one MFU-probe-shaped Modal command, and one eval command.

## Current VLM format and data mix

- Design choice: keep `<image>` as a human-readable data marker only. It is not added to the nanochat tokenizer. The renderer converts it to the internal sentinel `IMAGE_TOKEN_ID = -200`, and batching expands each sentinel to 64 projected SigLIP embeddings before calling the existing GPT path.
- This is the nanochat-native extension: no tokenizer resize, no checkpoint surgery, one small multimodal hook, and ordinary text-only `GPT.forward(idx, targets)` remains unchanged.
- Comparison note: nanoVLM owns tokenizer-level image placeholders and replaces their embeddings with projected vision features; InternVL expands an image marker into start/context/end placeholder tokens and replaces the context-token embeddings. For nanochat-llava v0, the simpler one-marker plus internal-sentinel design is the intended path.
- FineVisionMax is mixed text/image data, not purely image-conditioned data. The current loader supports text-only, single-image, and multi-image rows, with `--max-batch-images` capping real images per microbatch.
- Hugging Face cached dataset-server statistics for `HuggingFaceM4/FineVisionMax` report a partial 26,675-row stats sample with `texts` present on every row, mean `3.655` text entries per row, and `images` mean `0.714` images per row, median `1`, max `51`. Average item ratio in that stats sample is therefore image:text = `0.714:3.655`, about `1:5.12`.
- Existing first-10k training-shape probe: `6,188` usable single-image rendered rows and `3,812` non-single-image/no-image rows, i.e. `61.88%` single-image renderable and `38.12%` other under that older probe. Source for cached stats: https://datasets-server.huggingface.co/statistics?dataset=HuggingFaceM4/FineVisionMax&config=default&split=train.

## Current CPU/GPU dataflow

- CPU DataLoader side: stream/load records, normalize conversations, tokenize text, convert literal `<image>` to sentinel token id `-200`, build loss masks, decode images, run the SigLIP image processor, and pack fixed-shape rows.
- CPU to GPU boundary: packed token/mask metadata and CPU `pixel_values` are handed to the training step; tensors used by the model are created on or moved to the GPU.
- GPU vision side: frozen SigLIP consumes `pixel_values`, then the linear projector maps each image to 64 nanochat-width visual embeddings.
- GPU multimodal side: `build_multimodal_batch` has CPU-side Python control flow, but the token-id tensors, token embedding lookup, visual embedding insertion, and final `input_embeds` live on GPU. SigLIP features do not bounce back to CPU.
- GPU language side: nanochat GPT receives the constructed multimodal `input_embeds` plus targets and runs the normal forward/backward path.

## Current VLM attention mental model

- There are two different "image token" stages. In the vision encoder, image
  patch tokens use bidirectional/full visual self-attention, so each exported
  visual feature is already image-contextualized before it reaches nanochat.
- After projection, the 64 visual features are inserted into the GPT input
  stream as prefix embeddings. From that point on, nanochat uses the normal
  causal decoder attention path. We do not build a modality-specific mask where
  projected image tokens attend bidirectionally inside the LLM.
- For a sequence like `IMG1 IMG2 IMG3 TXT1 TXT2 TXT3`, the LLM-side mask is
  causal: `IMG1` sees only `IMG1`, `IMG2` sees `IMG1 IMG2`, `IMG3` sees
  `IMG1 IMG2 IMG3`, and text tokens see all previous image/text tokens. The
  practical effect is that bidirectional image understanding happens in SigLIP,
  then causal decoder attention carries that visual context into text decoding.
- The training loss mask is separate from the attention mask. Image positions
  and prompt/user positions can be ignored as supervised targets while still
  remaining visible as causal context for later tokens.
- This matches the baseline pattern in nanoVLM and InternVL: nanoVLM replaces
  image placeholder token embeddings and calls a causal language decoder after a
  non-causal ViT image encoder
  ([VLM forward](https://github.com/huggingface/nanoVLM/blob/main/models/vision_language_model.py#L62-L72),
  [ViT non-causal attention](https://github.com/huggingface/nanoVLM/blob/main/models/vision_transformer.py#L81-L85),
  [LM causal prefill](https://github.com/huggingface/nanoVLM/blob/main/models/language_model.py#L260-L278)).
  InternVL similarly replaces `<IMG_CONTEXT>` embeddings, then calls a causal
  LLM; its vision encoder uses `causal=False`, while InternLM2 FlashAttention
  uses `causal=True` for multi-token prefill/training and only disables causal
  masking for single-token KV-cache decode
  ([InternVL embedding replacement](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/internvl/model/internvl_chat/modeling_internvl_chat.py#L162-L203),
  [InternViT causal=False](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/internvl/model/internvl_chat/modeling_intern_vit.py#L239-L241),
  [InternLM2 causal flag](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/internvl/model/internlm2/modeling_internlm2.py#L523-L573)).

## Batch and token budget target

Reference VLM sample batch sizes:

- nanoVLM-222M used global batch size `256` samples/update.
- InternVL-Chat V1.2 SFT used global batch size `512` samples/update.
- InternVL2/InternVL2.5 fine-tune scripts commonly use global batch size `128` samples/update.

Current nanochat-llava H100 default shape:

```text
device_batch_size = 32
num_gpus = 1
max_seq_len = 512
```

Gradient accumulation to match a reference global sample batch:

```text
global_batch_samples = device_batch_size * num_gpus * grad_accum_steps
grad_accum_steps = reference_global_batch_samples / (device_batch_size * num_gpus)
```

For nanoVLM-222M's global batch:

```text
grad_accum_steps = 256 / (32 * 1) = 8
```

Tokens per optimizer update with that setting:

```text
tokens_per_update = device_batch_size * max_seq_len * grad_accum_steps
tokens_per_update = 32 * 512 * 8 = 131,072
```

Optimizer steps for about `1B` multimodal input tokens:

```text
num_iterations = 1,000,000,000 / 131,072 = 7,629.39
```

Use:

```bash
--grad-accum-steps 8 --num-iterations 7630
```

This matches nanoVLM-222M's reported global sample batch, not its approximate
token batch. If matching nanoVLM-222M's approximate token batch instead, use
`256 * 128 = 32,768` tokens/update, which maps to
`32,768 / (32 * 512) = 2` grad accumulation steps on the current shape.

## Pitfalls to avoid

- Do not re-add `vlm_precompute_siglip.py`, `/vol/features`, preflight scripts, resume/offset machinery, mem100 gates, benchmark report generators, FP8 probes, profiling grids, or frozen-feature training shortcuts unless there is a new explicit reason. They made the code harder to reason about before proving visual learning.
- Keep inline SigLIP for v0 so the main path reflects real training.
- Do not judge success from aggregate benchmark numbers alone. Compare separate checkpoint eval JSONs and inspect stored sample generations.
- The old Stage 1/Stage 2 split lives only in the experiment branch. Main uses one visual-instruction path.
- Keep training-time loss eval cheap and separate from standalone benchmark eval. Main packs a small held-out VLM pool through the same fixed training shape; `vlm_eval.py` remains the heavier MMStar/ScienceQA/ChartQA/MMMU/TextVQA verifier.

## Current commands

Local verification:

```bash
uv run python -m pytest tests/test_vision.py tests/test_vlm_smoke.py -q
uv run --extra vision python -m scripts.vlm_train --help
uv run --extra vision python -m scripts.vlm_eval --help
```

Modal smoke, MFU-shaped probe, and eval:

```bash
uv run --extra vision modal run modal_vlm.py::doctor
uv run --extra vision modal run modal_vlm.py::smoke

NANOCHAT_MODAL_GPU=H100 uv run --extra vision modal run modal_vlm.py::train \
  --num-iterations 6 \
  --batch-size 32 \
  --max-seq-len 512 \
  --max-batch-images 96 \
  --eval-every -1 \
  --no-save \
  --log-every 1

uv run --extra vision modal run modal_vlm.py::eval \
  --checkpoint-dir /vol/checkpoints/vlm \
  --checkpoint-step 1000 \
  --out /vol/checkpoints/vlm_eval.json \
  --benchmarks mmstar,scienceqa,chartqa,mmmu,textvqa \
  --limit 24 \
  --max-scan 240
```

## Remaining proof

The local code path is ready for a longer visual-instruction run, but
model-quality success is not proven until a real GPU run produces standalone VLM
eval JSONs. Compare scores and sample generations across checkpoints before
launching longer training.


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
work, host-to-device transfer, SigLIP forward, and feature pooling. That
profiling flag belonged to the experiment branch; main keeps only the clean
training loop.

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

## Main branch simplification and MFU check, 2026-05-26

The larger MFU/profiling branch was snapshotted as `experiment/mfu-varlen-fa3`
at `55a4f21`. Main was then simplified back to a nanochat-shaped V0 path:
frozen SigLIP, projector, nanochat training, and strict varlen FlashAttention
boundaries. The SigLIP batch/chunk knob, VLM FP8 experiments, profile timers,
and MFU grid machinery were removed from the main training script.

Short Modal H100 checks on the then-current streamed VLM dataset:

```text
max_batch_tokens=12000, steps=4
step 00004/00004 | loss 1.530354 | samples/sec 14.82 | tokens/sec 2444 | bf16_mfu 2.99
Peak memory usage: 64526.58MiB

max_batch_tokens=16000, steps=6
step 00002/00006 | loss 1.578618 | samples/sec 15.38 | tokens/sec 2456 | bf16_mfu 3.00
step 00003/00006 | loss 1.544003 | samples/sec 15.50 | tokens/sec 2454 | bf16_mfu 3.00
step 00004/00006 | loss 1.487748 | samples/sec 15.17 | tokens/sec 2384 | bf16_mfu 2.91
step 00005/00006 | loss 1.486309 | samples/sec 16.29 | tokens/sec 2502 | bf16_mfu 3.06
step 00006/00006 | loss 1.448560 | samples/sec 14.53 | tokens/sec 2460 | bf16_mfu 3.00
Peak memory usage: 78495.79MiB
```

Interpretation:

- The simplified main branch compiles and the strict varlen FA3 path trains on
  H100.
- Increasing the packed-token cap from 12K to 16K did not materially raise
  useful tokens/sec; it mostly consumed the remaining H100 memory.
- The first simplified rewrite was too naive: it used full dense logits for all
  ignored visual/prompt positions and constructed multimodal batches with many
  tiny `wte(token)` and per-image projector calls.

Follow-up fixes kept the same CE objective but made the implementation closer to
the experiment branch's efficient path:

- Added target-only logits via `loss_indices` / `loss_targets`. This is the same
  ignore-index cross-entropy as nanochat, but it skips the vocab projection for
  labels that are ignored.
- Added optional frozen SigLIP feature caching for finite runs. This moves image
  encoding out of the measured training step when SigLIP is frozen.
- Replaced the naive multimodal builder with a vectorized version: one embedding
  lookup for `value_token_ids`, one projector call for all visual features, and
  indexed insertion of the 64 visual embeddings.

Data-pipeline reference from existing VLM repos:

- nanoVLM keeps the path simple: dataset workers process PIL images with the
  image processor, the DataLoader uses `num_workers` and `pin_memory`, and the
  model forward concatenates image tensors and runs the vision encoder on the
  model device. Relevant code:
  [dataset image processing](https://github.com/huggingface/nanoVLM/blob/main/data/datasets.py#L62-L77),
  [DataLoader workers/pin memory](https://github.com/huggingface/nanoVLM/blob/main/train.py#L213-L223),
  [model-side image H2D/vision encoder](https://github.com/huggingface/nanoVLM/blob/main/models/vision_language_model.py#L51-L68).
- InternVL uses the same split at larger scale: the dataset opens images with
  TCS/PIL, applies CPU transforms, and returns stacked `pixel_values`; the
  collator concatenates `pixel_values`; Trainer/Accelerate moves the batch to
  the device; the model forward calls `extract_feature(pixel_values)`, which
  runs the vision model and projector before inserting visual embeddings into
  the LLM input embeddings. Relevant code:
  [image open/CPU transform](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/internvl/train/internvl_chat_finetune.py#L401-L442),
  [pixel_values collator](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/internvl/patch/pad_data_collator.py#L98-L116),
  [DataLoader workers/pin memory](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/internvl/patch/train_dataloader_patch.py#L33-L48),
  [model forward](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/internvl/model/internvl_chat/modeling_internvl_chat.py#L143-L166),
  [vision encoder/projector](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/internvl/model/internvl_chat/modeling_internvl_chat.py#L273-L291).

Modal H100 follow-up on the simplified main branch:

```text
target-only CE + vectorized multimodal construction + inline SigLIP
max_batch_tokens=18000, steps=6
step 00002/00006 | tokens/sec 6989 | bf16_mfu 8.03
step 00003/00006 | tokens/sec 7228 | bf16_mfu 8.30
step 00004/00006 | tokens/sec 7733 | bf16_mfu 8.89
step 00005/00006 | tokens/sec 7136 | bf16_mfu 8.20
step 00006/00006 | tokens/sec 6956 | bf16_mfu 7.99
Peak memory usage: 69524.90MiB

target-only CE + vectorized multimodal construction + DataLoader-worker image preprocessing + inline SigLIP
max_batch_tokens=18000, steps=6, num_workers=4
step 00001/00006 | tokens/sec 560 | bf16_mfu 0.64
step 00002/00006 | tokens/sec 21475 | bf16_mfu 24.67
step 00003/00006 | tokens/sec 20912 | bf16_mfu 24.03
step 00004/00006 | tokens/sec 24382 | bf16_mfu 28.02
step 00005/00006 | tokens/sec 23961 | bf16_mfu 27.53
step 00006/00006 | tokens/sec 23870 | bf16_mfu 27.41
Peak memory usage: 69524.90MiB

target-only CE + diagnostic precomputed features, but naive per-token/per-image batch construction
max_batch_tokens=20000, steps=6
step 00002/00006 | tokens/sec 3705 | bf16_mfu 5.04
step 00006/00006 | tokens/sec 3635 | bf16_mfu 4.95
Peak memory usage: 74301.75MiB

target-only CE + diagnostic precomputed features + vectorized multimodal construction
max_batch_tokens=20000, steps=6
step 00002/00006 | tokens/sec 23723 | bf16_mfu 27.25
step 00003/00006 | tokens/sec 21629 | bf16_mfu 24.85
step 00004/00006 | tokens/sec 23708 | bf16_mfu 27.24
step 00005/00006 | tokens/sec 25121 | bf16_mfu 28.86
Result: OOM on step 6 near the 80GB memory edge.

target-only CE + diagnostic precomputed features + vectorized multimodal construction
max_batch_tokens=18000, steps=6
step 00002/00006 | tokens/sec 21642 | bf16_mfu 24.86
step 00003/00006 | tokens/sec 21222 | bf16_mfu 24.38
step 00004/00006 | tokens/sec 22730 | bf16_mfu 26.12
step 00005/00006 | tokens/sec 22248 | bf16_mfu 25.56
step 00006/00006 | tokens/sec 22061 | bf16_mfu 25.34
Peak memory usage: 69211.61MiB
```

Modal H100 double-check on current checkout, 2026-05-27:

```text
FineVisionMax streaming data shape, first 10,000 train rows:
usable single-image rendered rows: 6,188; skipped non-single-image/no-image rows: 3,812
all_text_words: mean 199.9, median 108, p90 407, p95 530, p99 1485, max 14739
all_text_nanochat_tokens: mean 418.2, median 183, p90 986, p95 1620, p99 3207, max 22992
assistant_nanochat_tokens: mean 316.0, median 135, p90 827, p95 1365, p99 2576, max 6678
rendered_conversation_tokens_with_specials: mean 436.7, median 194, p90 1014, p95 1659, p99 3278, max 25024
expanded_decoder_input_len_with_64_image_tokens: mean 498.7, median 256, p90 1076, p95 1721, p99 3340, max 25086

target-only CE + vectorized multimodal construction + DataLoader-worker image preprocessing + inline SigLIP
batch_size=768, max_batch_tokens=18000, steps=6, num_workers=4
step 00002/00006 | tokens/sec 24279 | bf16_mfu 28.84
Result: OOM on step 3 during backward. Memory at failure: 77.46GiB in use, 1.71GiB free.

target-only CE + vectorized multimodal construction + DataLoader-worker image preprocessing + inline SigLIP
batch_size=768, max_batch_tokens=16000, steps=6, num_workers=4
step 00001/00006 | tokens/sec 386 | bf16_mfu 0.46
step 00002/00006 | tokens/sec 22392 | bf16_mfu 26.63
step 00003/00006 | tokens/sec 22402 | bf16_mfu 26.86
step 00004/00006 | tokens/sec 23191 | bf16_mfu 27.67
step 00005/00006 | tokens/sec 23640 | bf16_mfu 28.18
step 00006/00006 | tokens/sec 24112 | bf16_mfu 28.74
Peak memory usage: 73007.99MiB

target-only CE + fixed packed rows + vectorized multimodal construction + DataLoader-worker image preprocessing + inline SigLIP
batch_size=32, max_seq_len=512, internal candidate buffer=768, steps=6, num_workers=4
step 00001/00006 | samples/sec 0.47 | tokens/sec 242 | target_tokens 11,027 | bf16_mfu 0.29
step 00002/00006 | samples/sec 47.73 | tokens/sec 24440 | target_tokens 10,034 | bf16_mfu 29.08
step 00003/00006 | samples/sec 48.93 | tokens/sec 25054 | target_tokens 10,827 | bf16_mfu 29.91
step 00004/00006 | samples/sec 49.18 | tokens/sec 25181 | target_tokens 10,714 | bf16_mfu 30.05
step 00005/00006 | samples/sec 64.70 | tokens/sec 24094 | target_tokens 9,725 | bf16_mfu 28.63
step 00006/00006 | samples/sec 79.52 | tokens/sec 24128 | target_tokens 9,176 | bf16_mfu 28.61
Peak memory usage: 73845.14MiB

target-only CE + fixed packed rows + vectorized multimodal construction + DataLoader-worker image preprocessing + inline SigLIP
batch_size=64, max_seq_len=512, internal candidate buffer=1536, steps=6, num_workers=4
Model/logged setup: `karpathy/nanochat-d32` config is 32 layers, 2048 hidden, 65,536 vocab; params total/trainable: 4,051,700,826 / 1,904,214,106.
Result: OOM before finishing step 1. Memory at failure: 78.82GiB in use, 356.25MiB free. Failure allocation was a 512MiB MLP activation allocation.

target-only CE + mixed text/single-image/multi-image fixed packed rows + vectorized multimodal construction + DataLoader-worker image preprocessing + inline SigLIP
batch_size=32, max_seq_len=512, max_batch_images=96, internal candidate buffer=768, steps=6, num_workers=4
step 00001/00006 | samples/sec 0.75 | tokens/sec 301 | target_tokens 9,351 | bf16_mfu 0.36
step 00002/00006 | samples/sec 56.22 | tokens/sec 25587 | target_tokens 8,621 | bf16_mfu 30.26
step 00003/00006 | samples/sec 59.81 | tokens/sec 25788 | target_tokens 8,063 | bf16_mfu 30.43
step 00004/00006 | samples/sec 48.77 | tokens/sec 24213 | target_tokens 9,212 | bf16_mfu 28.71
step 00005/00006 | samples/sec 74.12 | tokens/sec 23812 | target_tokens 9,860 | bf16_mfu 28.31
step 00006/00006 | samples/sec 64.94 | tokens/sec 25949 | target_tokens 8,873 | bf16_mfu 30.73
Peak memory usage: 72866.35MiB
```

Interpretation:

- The `3%` result was a regression in the simplified main implementation, not a
  limit of varlen FA3.
- Target-only CE alone did not restore MFU because the naive multimodal builder
  still launched thousands of tiny GPU ops per step.
- Vectorized multimodal construction plus inline SigLIP raises the realistic
  main-branch path from about `3%` to about `8%` MFU at an 18K token cap.
- Moving image open/decode and HF CPU processing into DataLoader workers raises
  the same realistic inline-SigLIP path to about `24-28%` MFU after the first
  worker warmup step. This matches nanoVLM/InternVL's simple pattern: workers
  return pinned CPU `pixel_values`, and the main process runs H2D plus SigLIP.
  The current stable H100 default is the fixed `--device-batch-size 32
  --max-seq-len 512 --max-batch-images 96` shape; after mixed text/multi-image
  support it ran at `28.31-30.73%` BF16 MFU. Doubling to `64 x 512` OOMed
  before step 1 on the current d32 model. The older compact
  `max_batch_tokens=16000` path was removed from live code after this
  comparison; it was slightly slower at `26.63-28.74%`, while `18000` was close
  enough to the 80GB edge to OOM on later random batch shapes.
- The train script now keeps benchmark/generation eval separate from cheap
  validation loss. `vlm_train.py` can run target-token validation CE every
  `--eval-every` steps over a small held-out visual-instruction split using the
  same worker `pixel_values` path. `vlm_eval.py` remains the heavier
  MMStar/ScienceQA/ChartQA/MMMU/TextVQA verifier, analogous to CORE/ChatCORE.
- Diagnostic precomputed features isolated the projector+LLM path and restored
  that path to the experiment branch regime: roughly `24-26%` steady at an 18K
  token cap, with the 20K cap reaching `25-29%` before OOM.
- The remaining gap to text-only nanochat is still the old one: the 80GB VLM path
  fits about 16K packed-row tokens reliably with this model and no checkpointing,
  not the much larger per-rank text batch used in the nanochat speedrun. With
  inline SigLIP, there is also real vision-encoder time inside every optimizer
  step.
- The diagnostic precomputed-feature shortcut was useful for attribution but is
  not part of main because real v0 training keeps SigLIP inline.

## Vast.ai A100 80GB run and benchmark drift, 2026-05-26

Repo state: `0de2303aa6a22468624f6e59a5a39676365aabe1`
(`Add Vast.ai Codex training prompt`), run on a single A100 80GB Vast.ai box.
The A100 run could not use `--require-fa3-varlen`: the repo reports
`HAS_FA3=False`, `USE_FA3=False`, and `has_fa3_varlen=False` because the FA3
path is Hopper/SM90-only. Training below used the SDPA varlen fallback with the
same batch/token settings otherwise. After installing the dev group,
`uv run python -m pytest tests/test_vision.py -q` passed with `22 passed`.

Data setup and training timing:

- Persistent data root: `/workspace/nanochat-llava-data`
- VQAv2 load/render time before training: `255.20s`
- Split: `80,717` train records and `2,048` validation records
- Real run output: `/workspace/nanochat-llava-data/checkpoints/vlm_latest_0de2303`
- Final checkpoint: `model_001000.pt`
- Total training time: `69.75m`

Training summary:

| metric | value |
| --- | ---: |
| final train loss, step 1000 | `0.475191` |
| final raw loss, step 1000 | `0.519070` |
| final tokens/sec | `4699.15` |
| final BF16 MFU | `17.12%` |
| best observed BF16 MFU | about `18%` |
| peak memory | `71635.30MiB` |

Validation loss:

| step | val loss |
| ---: | ---: |
| 200 | `0.735346` |
| 400 | `0.653892` |
| 600 | `0.611143` |
| 800 | `0.595511` |
| 1000 | `0.586254` |

The A100 result is below the Modal H100 `24-28%` MFU result above. That is
expected for this run because it did not use the H100/FA3 path. The improved
DataLoader path still matters on A100, but this hardware/software path topped
out around `17-18%` MFU with inline SigLIP and no activation checkpointing.

Step 2 versus step 1000 verifier scores, larger sample run:

| benchmark | step 2 | step 1000 | delta | n |
| --- | ---: | ---: | ---: | ---: |
| MMStar | `0.3300` | `0.2800` | `-0.0500` | 200 |
| ScienceQA | `0.4350` | `0.3050` | `-0.1300` | 200 |
| ChartQA | `0.0550` | `0.1250` | `+0.0700` | 200 |
| MMMU Accounting | `0.2333` | `0.2667` | `+0.0333` | 30 |
| TextVQA | `0.0450` | `0.1400` | `+0.0950` | 200 |
| Mean | `0.2197` | `0.2233` | `+0.0037` | - |

JSON outputs:

- Step 2:
  `/workspace/nanochat-llava-data/checkpoints/vlm_latest_smoke_0de2303_eval_limit200.json`
- Step 1000:
  `/workspace/nanochat-llava-data/checkpoints/vlm_latest_0de2303_eval_limit200.json`

Proper MMMU all-config run:

- Configs: all 30 MMMU validation subjects, `30` examples each, `900` total
- Step 2: `0.2489`, `224/900`
- Step 1000: `0.2744`, `247/900`
- Delta: `+0.0256`, `+23` correct
- Step 2 JSON:
  `/workspace/nanochat-llava-data/checkpoints/vlm_latest_smoke_0de2303_mmmu_all.json`
- Step 1000 JSON:
  `/workspace/nanochat-llava-data/checkpoints/vlm_latest_0de2303_mmmu_all.json`

Largest MMMU gains were History `+0.300`, Art_Theory `+0.200`,
Basic_Medical_Science `+0.200`, Finance `+0.167`, and
Architecture_and_Engineering `+0.133`. Largest regressions were Agriculture
`-0.167`, Marketing `-0.167`, Energy_and_Power `-0.133`, and
Diagnostics_and_Laboratory_Medicine `-0.100`.

Paired diagnostic, same records at step 2 and step 1000:

| benchmark | n | step 2 | step 1000 | correct to wrong | wrong to correct | both wrong | both correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MMStar | 200 | `0.3300` | `0.2800` | 31 | 21 | 113 | 35 |
| ScienceQA | 200 | `0.4350` | `0.3050` | 54 | 28 | 85 | 33 |
| MMMU all | 900 | `0.2489` | `0.2744` | 96 | 119 | 557 | 128 |

Paired diagnostic JSON:
`/workspace/nanochat-llava-data/checkpoints/paired_compare_latest_step2_vs_1000.json`

Interpretation:

- Training clearly improved the held-out VQAv2 validation loss from `0.735346`
  at step 200 to `0.586254` at step 1000.
- The verifier benchmark movement is mixed. Step 1000 improved ChartQA,
  TextVQA, and MMMU, but regressed MMStar and ScienceQA.
- The MMStar and ScienceQA decreases are not only sample noise in this paired
  run: MMStar had `31` correct-to-wrong flips versus `21` wrong-to-correct
  flips, and ScienceQA had `54` correct-to-wrong flips versus `28`
  wrong-to-correct flips.
- The likely cause is task drift from VQAv2-only finetuning. Step 2 still has
  more base SFT multiple-choice and science-prior behavior. After VQAv2
  training, answers become more VQA-style and short/free-form, which helps
  some visual QA cases but hurts option-letter stability and some science-prior
  questions.
- MMMU improves modestly overall when evaluated across all subjects, but most
  examples are still hard failures: `557/900` are wrong at both step 2 and step
  1000. This is expected because VQAv2 does not teach the domain-specific
  diagram, table, formula, and professional-subject reasoning needed for MMMU.
- MMMU verifier scores should be treated as rough diagnostics, not leaderboard
  numbers. The current matcher can credit an initial option letter even when
  the following generated text is semantically inconsistent, and can sometimes
  credit answer text even when the emitted option letter is wrong.

## Vast.ai A100 80GB FineVisionMax stream run, 2026-05-27

Repo state: `269e488374991d2d27f2dfc92619b8ea978dede6`, run on a single
A100-SXM4-80GB Vast.ai box. The repo again reported `has_fa3_varlen=False`, so
training used the SDPA varlen fallback. Setup used the default
`HuggingFaceM4/FineVisionMax` stream, not a local JSON/VQAv2 conversion.
Focused verification passed with
`uv run python -m pytest tests/test_vision.py tests/test_vlm_smoke.py -q`:
`27 passed`.

Smoke and upload:

- Smoke checkpoint: `/workspace/nanochat-llava-data/checkpoints/vlm_smoke`
- Smoke eval JSON: `/workspace/nanochat-llava-data/checkpoints/vlm_smoke_eval.json`
- Smoke eval: `mmstar=1.0000`, `n=1`
- Final checkpoint: `/workspace/nanochat-llava-data/checkpoints/vlm/model_001000.pt`
- Metadata: `/workspace/nanochat-llava-data/checkpoints/vlm/meta_001000.json`
- Hugging Face upload:
  `Yusuke710/nanochat-llava-finevisionmax-1gpu`

Training summary:

| metric | value |
| --- | ---: |
| final train loss, step 1000 | `1.298515` |
| final raw loss, step 1000 | `1.319070` |
| final tokens/sec | `4688.74` |
| final BF16 MFU | `17.54%` |
| warm tokens/sec range | about `4.7k-5.9k` |
| warm BF16 MFU range | about `17-22%` |
| peak memory | `75646.67MiB` |
| total training time | `57.99m` |

Validation loss:

| step | val loss |
| ---: | ---: |
| 200 | `1.409743` |
| 400 | `1.346387` |
| 600 | `1.311720` |
| 800 | `1.291577` |
| 1000 | `1.281523` |

The FineVisionMax validation loss scale is not comparable to the earlier
VQAv2-only run because the held-out stream is mixed text-only, single-image,
and multi-image data with different target distributions. Throughput is
comparable to the previous A100 run, while peak memory rose from
`71635.30MiB` to `75646.67MiB`, consistent with the mixed-modality packing and
higher image cap.

Step 1000 verifier scores, matching the prior `limit=200` suite:

| benchmark | VQAv2 step 1000 | FineVisionMax step 1000 | delta | n |
| --- | ---: | ---: | ---: | ---: |
| MMStar | `0.2800` | `0.1750` | `-0.1050` | 200 |
| ScienceQA | `0.3050` | `0.4421` | `+0.1371` | 95 |
| ChartQA | `0.1250` | `0.0750` | `-0.0500` | 200 |
| MMMU Accounting | `0.2667` | `0.2667` | `+0.0000` | 30 |
| TextVQA | `0.1400` | `0.1200` | `-0.0200` | 200 |
| Mean | `0.2233` | `0.2158` | `-0.0076` | - |

JSON output:
`/workspace/nanochat-llava-data/checkpoints/vlm_finevisionmax_1gpu_eval_limit200.json`

ScienceQA caveat: this run scanned 200 records but evaluated only 95 image
records because 105 records raised `KeyError: 'no PIL image field found'`.
Treat the ScienceQA delta as directional, not a paired comparison with the
older `n=200` row.

Sample generations:

- `mmstar`: predicted `D` for the suitcase/object-relationship example; answer
  was `A`. It did get the guitar-theme and trees examples correct.
- `scienceqa`: predicted `B`, `C`, and `B` correctly for the first three
  printed image examples, then missed the New Zealand falcon feet question.
- `chartqa`: still weak on chart grounding, e.g. predicted `red` where the
  answer was `Blue`, and `2013` where the answer was `2018`.
- `mmmu`: Accounting remained essentially unchanged from the VQAv2 run:
  `0.2667`, `8/30`.
- `textvqa`: produced more OCR-shaped strings than the early runs, but the
  printed samples were still wrong: `DROPPAGE` for `copenhagen`, `Sulfury` for
  `ale`, and `Mason's` for `bowmore`.

Interpretation:

- Switching from VQAv2-only to FineVisionMax removed the clear VQAv2-specialized
  gains on ChartQA/TextVQA seen in the previous run and further hurt MMStar on
  this sample.
- ScienceQA improved on the evaluated image subset, which is consistent with
  FineVisionMax being broader and less pure-VQA-style than the earlier data.
  Because the evaluated ScienceQA subset changed to 95 usable image rows, this
  needs a paired diagnostic before treating it as a real regression fix.
- The aggregate mean is effectively flat versus the previous VQAv2 step-1000
  verifier result (`0.2158` versus `0.2233`). The more important signal is task
  reshuffling: broader FineVisionMax helped the science slice here, did not move
  MMMU Accounting, and reduced MMStar/ChartQA/TextVQA on this limited verifier.
