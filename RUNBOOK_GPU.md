# nanochat-llava external GPU runbook

This is the current short runbook for a rented GPU machine. It avoids Modal and
uses streamed HF metadata/shards with local or embedded image bytes.

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

Training metadata is streamed from Hugging Face by default.

- Stage 1 default: streams `blip_laion_cc_sbu_558k.json` and reads images
  directly from the HF `images.zip` cache file without extracting it.
- Stage 2 default: streams `HuggingFaceM4/FineVision` config
  `LLaVA_Instruct_150K`. Image bytes come from the HF dataset shards, so there
  are no per-sample COCO URL downloads.

The first run still downloads model weights once: `karpathy/nanochat-d32` and
`google/siglip-base-patch16-512`.

## Cheap smoke

```bash
VLM_SMOKE_DEVICE=cuda uv run --extra vision --extra gpu python -m pytest tests/test_vlm_smoke.py -q
```

## Stage 1 projector alignment

This streams LLaVA-Pretrain metadata and reads images directly from the HF
`images.zip` cache file without extracting it.

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
  --model-step 650 \
  --skip-bad-images
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
  --out $DATA_ROOT/bench/stage1_pixshuffle_250.json \
  --device-type cuda \
  --model-step 650
```

## Stage 2 visual instruction probe

This streams the FineVision-packaged LLaVA-Instruct subset. It follows the
nanoVLM-style `images` + `texts` schema and avoids per-image COCO URL fetches.
This direct run skips Stage 1 projector alignment: nanochat starts from
`karpathy/nanochat-d32`, the projector starts random, SigLIP stays frozen, and
Stage 2 trains projector plus nanochat.

```bash
uv run --extra vision --extra gpu python -m scripts.vlm_train \
  --stage 2 \
  --hf-repo HuggingFaceM4/FineVision \
  --hf-config LLaVA_Instruct_150K \
  --out-dir $DATA_ROOT/checkpoints/stage2_direct_finevision_probe \
  --device-type cuda \
  --num-iterations 100 \
  --device-batch-size 24 \
  --max-batch-tokens 12000 \
  --max-examples 4096 \
  --max-seq-len 2048 \
  --save-every 100 \
  --model-step 650 \
  --profile-timing \
  --eval-every 100 \
  --eval-limit 16 \
  --eval-max-scan 240 \
  --eval-print-samples 3
```

## Benchmark Stage 2

```bash
uv run --extra vision --extra gpu python -m scripts.vlm_eval \
  --checkpoint-dir $DATA_ROOT/checkpoints/stage2_direct_finevision_probe \
  --checkpoint-step 100 \
  --benchmarks mmstar,scienceqa,chartqa,mmmu,textvqa \
  --limit 16 \
  --max-scan 240 \
  --print-samples 3 \
  --out $DATA_ROOT/bench/stage2_direct_finevision_probe.json \
  --device-type cuda \
  --model-step 650
```

Compare this eval against the text-only/base baseline or the previous Stage 1/2
probe if available. The first useful signal is whether the subset scores improve
and whether sample generations are image-relevant.

## Scaled Training Tracking

Karpathy-style tracking for a longer Stage 2 run is:

- `val_bpb`: held-out VLM answer-token BPB on a local validation JSON.
- `eval/mean_score`: benchmark generation accuracy every 2000 steps, capped at
  500 examples per benchmark.

Add these args to the Stage 2 command when the held-out set is available on the
machine:

```bash
  --val-json stage2_memorization_set/heldout.json \
  --val-image-root stage2_memorization_set \
  --val-bpb-every 200 \
  --val-bpb-examples 100 \
  --eval-every 2000 \
  --eval-limit 500 \
  --eval-max-scan 4000 \
  --eval-print-samples 3
```

## Go/no-go criteria

Use this cheap probe as a gate before spending on longer LLaVA training:

- Continue if Stage 2 logs show stable loss, nonzero `samples/sec`/`tokens/sec`,
  reasonable peak memory headroom, improved scores versus an earlier checkpoint,
  and sample generations that are image-relevant.
- Stop and inspect before scaling if loss is flat or exploding, the later
  checkpoint is not better than the earlier checkpoint on any verifier subset,
  or sample generations look like text-only priors/gibberish rather than image
  answers.
- Do not treat one noisy benchmark number as proof. Run eval on two checkpoints
  such as step 100 and step 500, then compare scores and printed samples.

## Inspect results

Use this after both benchmark JSON files exist:

```bash
uv run --extra vision --extra gpu python - <<'PY'
import json
import os
from pathlib import Path

bench = Path(os.environ["DATA_ROOT"]) / "bench"
before = json.loads((bench / "stage2_direct_finevision_step100.json").read_text())
after = json.loads((bench / "stage2_direct_finevision_step500.json").read_text())

print("benchmark,before,after,delta")
for key in sorted(after["benchmarks"]):
    b = before["benchmarks"].get(key, {})
    a = after["benchmarks"][key]
    before_score = float(b.get("score", 0.0))
    after_score = float(a.get("score", 0.0))
    print(f"{key},{before_score:.4f},{after_score:.4f},{after_score-before_score:+.4f}")

print("\nLatest sample generations:")
for key, row in after["benchmarks"].items():
    for sample in row.get("samples", [])[:3]:
        print(f"\n[{key} #{sample['index']}]")
        print("prompt:", sample.get("prompt", ""))
        print("pred:", sample.get("prediction", ""))
        print("answers:", sample.get("answers", []))
PY
```

## Timing notes

- Stage 1 250 steps at batch 32 was about 4 minutes when image reads were warm
  or not dominated by third-party URLs. Cold URL mode can be slower.
- Stage 2 100 steps at batch 24 is expected to be tens of minutes on A100-80GB.
- Use `--profile-timing` output for actual ETA: logs include `samples/sec`,
  `tokens/sec`, `bf16_mfu`, peak memory, and image/SigLIP timing.
