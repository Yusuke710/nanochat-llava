# nanochat-llava external GPU runbook

This is the current short runbook for a rented GPU machine. It avoids Modal and
uses streamed HF JSON plus on-demand image downloads where possible.

## Setup

Use A100-80GB if available. A100-40GB may need smaller batch sizes; 24GB GPUs are
unlikely to fit the full `karpathy/nanochat-d32` VLM stage 2 path.

```bash
cd /path/to/nanochat-llava
uv sync --extra vision --extra gpu

export DATA_ROOT=/data/nanochat-llava
export NANOCHAT_BASE_DIR=$DATA_ROOT/nanochat
export HF_HOME=$DATA_ROOT/hf
export NANOCHAT_SIGLIP_CACHE_DIR=$HF_HOME/siglip
mkdir -p $DATA_ROOT/{nanochat,hf,datasets/llava,checkpoints,bench,logs}

uv run --extra vision --extra gpu python - <<'PY'
import torch
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_properties(0).total_memory / 1024**3, "GB")
PY
```

## Data download behavior

Training metadata is streamed from Hugging Face by default. Do not pass
`--no-stream-hf-data` for the cheap probe path.

- Stage 1 default: streams `blip_laion_cc_sbu_558k_meta.json`, then downloads
  only the source images that are actually used by the run.
- Stage 1 fallback: `--hf-image-zip images.zip` downloads the large HF image zip
  once into the HF cache, but reads images directly from the zip without
  extracting them.
- Stage 2 default: streams `llava_instruct_150k.json`, then downloads only the
  referenced COCO train2017 images.

The first run still downloads model weights once: `karpathy/nanochat-d32` and
`google/siglip-base-patch16-512`.

## Cheap smoke

```bash
VLM_SMOKE_DEVICE=cuda uv run --extra vision --extra gpu python -m pytest tests/test_vlm_smoke.py -q
```

## Stage 1 projector alignment

Default no-upfront-download path. This streams the LLaVA-Pretrain meta JSON and
downloads only referenced source images. Dead or slow image URLs are skipped.

```bash
uv run --extra vision --extra gpu python -m scripts.vlm_train \
  --stage 1 \
  --hf-repo liuhaotian/LLaVA-Pretrain \
  --hf-file blip_laion_cc_sbu_558k_meta.json \
  --image-root $DATA_ROOT/datasets/llava/pretrain_images \
  --out-dir $DATA_ROOT/checkpoints/stage1_pixshuffle_250 \
  --device-type cuda \
  --num-iterations 250 \
  --device-batch-size 32 \
  --max-examples 16000 \
  --max-seq-len 2048 \
  --save-every 250 \
  --model-step 650 \
  --skip-bad-images
```

If source URLs are too slow or too many are dead, use the reliable HF zip
fallback. It downloads `images.zip` once into the HF cache and reads from it
without extracting.

```bash
uv run --extra vision --extra gpu python -m scripts.vlm_train \
  --stage 1 \
  --hf-repo liuhaotian/LLaVA-Pretrain \
  --hf-file blip_laion_cc_sbu_558k.json \
  --hf-image-zip images.zip \
  --image-root $DATA_ROOT/datasets/llava/pretrain_images \
  --out-dir $DATA_ROOT/checkpoints/stage1_pixshuffle_250 \
  --device-type cuda \
  --num-iterations 250 \
  --device-batch-size 32 \
  --max-examples 16000 \
  --max-seq-len 2048 \
  --save-every 250 \
  --model-step 650
```

## Benchmark Stage 1 baseline

```bash
uv run --extra vision --extra gpu python -m scripts.vlm_eval \
  --checkpoint-dir $DATA_ROOT/checkpoints/stage1_pixshuffle_250 \
  --checkpoint-step 250 \
  --benchmarks mmstar,scienceqa,chartqa,mmmu,textvqa \
  --limit 16 \
  --max-scan 240 \
  --print-samples 3 \
  --control \
  --out $DATA_ROOT/bench/stage1_pixshuffle_250.json \
  --device-type cuda \
  --model-step 650
```

## Stage 2 visual instruction probe

This streams LLaVA-Instruct JSON and downloads only referenced COCO images.

```bash
uv run --extra vision --extra gpu python -m scripts.vlm_train \
  --stage 2 \
  --hf-repo liuhaotian/LLaVA-Instruct-150K \
  --hf-file llava_instruct_150k.json \
  --image-root $DATA_ROOT/datasets/llava/coco/train2017 \
  --image-url-template 'http://images.cocodataset.org/train2017/{basename}' \
  --init-vlm-checkpoint-dir $DATA_ROOT/checkpoints/stage1_pixshuffle_250 \
  --init-vlm-checkpoint-step 250 \
  --out-dir $DATA_ROOT/checkpoints/stage2_llava_probe \
  --device-type cuda \
  --num-iterations 100 \
  --device-batch-size 24 \
  --max-batch-tokens 12000 \
  --max-examples 4096 \
  --max-seq-len 2048 \
  --save-every 100 \
  --model-step 650 \
  --profile-timing
```

## Benchmark Stage 2

```bash
uv run --extra vision --extra gpu python -m scripts.vlm_eval \
  --checkpoint-dir $DATA_ROOT/checkpoints/stage2_llava_probe \
  --checkpoint-step 100 \
  --benchmarks mmstar,scienceqa,chartqa,mmmu,textvqa \
  --limit 16 \
  --max-scan 240 \
  --print-samples 3 \
  --control \
  --out $DATA_ROOT/bench/stage2_llava_probe.json \
  --device-type cuda \
  --model-step 650
```

Compare `stage1_pixshuffle_250.json` and `stage2_llava_probe.json`. The first
useful signal is whether Stage 2 improves the subset scores and whether sample
generations are image-relevant.

## Go/no-go criteria

Use this cheap probe as a gate before spending on longer LLaVA training:

- Continue if Stage 2 logs show stable loss, nonzero `samples/sec`/`tokens/sec`,
  reasonable peak memory headroom, and at least some benchmark/sample evidence
  that image-conditioned answers differ from zero-image answers in a useful way.
- Stop and inspect before scaling if loss is flat or exploding, controls usually
  fail, Stage 2 score is not better than Stage 1 on any verifier subset, or
  sample generations look like text-only priors/gibberish rather than image
  answers.
- Do not treat one noisy benchmark number as proof. Check the printed samples
  and the `stage2_zero` / `changed` columns from the inspection command.

## Inspect results

Use this after both benchmark JSON files exist:

```bash
uv run --extra vision --extra gpu python - <<'PY'
import json
import os
from pathlib import Path

bench = Path(os.environ["DATA_ROOT"]) / "bench"
before = json.loads((bench / "stage1_pixshuffle_250.json").read_text())
after = json.loads((bench / "stage2_llava_probe.json").read_text())

print("benchmark,stage1,stage2,delta,stage2_zero,changed")
for key in sorted(after["benchmarks"]):
    b = before["benchmarks"].get(key, {})
    a = after["benchmarks"][key]
    before_score = float(b.get("score", 0.0))
    after_score = float(a.get("score", 0.0))
    zero = a.get("zero_image_score", "")
    changed = a.get("prediction_changed_rate", "")
    print(f"{key},{before_score:.4f},{after_score:.4f},{after_score-before_score:+.4f},{zero},{changed}")

print("\nStage 2 sample generations:")
for key, row in after["benchmarks"].items():
    for sample in row.get("samples", [])[:3]:
        print(f"\n[{key} #{sample['index']}]")
        print("prompt:", sample.get("prompt", ""))
        print("pred:", sample.get("prediction", ""))
        if "zero_image_prediction" in sample:
            print("zero:", sample.get("zero_image_prediction", ""))
        print("answers:", sample.get("answers", []))
PY
```

## Timing notes

- Stage 1 250 steps at batch 32 was about 4 minutes when image reads were warm
  or not dominated by third-party URLs. Cold URL mode can be slower.
- Stage 2 100 steps at batch 24 is expected to be tens of minutes on A100-80GB.
- Use `--profile-timing` output for actual ETA: logs include `samples/sec`,
  `tokens/sec`, `bf16_mfu`, peak memory, and image/SigLIP timing.
