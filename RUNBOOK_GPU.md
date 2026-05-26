# nanochat-llava external GPU runbook

This is the current short runbook for a rented GPU machine. It avoids Modal and
uses streamed Hugging Face image-text shards with embedded image bytes.

## Setup

Use A100-80GB if available. A100-40GB may need smaller batch sizes; 24GB GPUs are
unlikely to fit the full `karpathy/nanochat-d32` VLM training path.

```bash
cd /path/to/nanochat-llava
uv sync --extra vision --extra gpu

export DATA_ROOT=/data/nanochat-llava
export NANOCHAT_BASE_DIR=$DATA_ROOT/nanochat
export HF_HOME=$DATA_ROOT/hf
export NANOCHAT_SIGLIP_CACHE_DIR=$HF_HOME/siglip
mkdir -p $DATA_ROOT/{nanochat,hf,checkpoints,bench,logs}

uv run --extra vision --extra gpu python - <<'PY'
import torch
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_properties(0).total_memory / 1024**3, "GB")
PY
```

## Data download behavior

Training rows are streamed from Hugging Face by default.

- Training default: streams all configs from `HuggingFaceM4/the_cauldron`.

Image bytes come from the HF dataset shards, so there are no per-sample COCO URL
downloads and no separate image zip extraction step.

For probes, `--max-examples` materializes a fixed small subset. For longer runs,
omit `--max-examples`; training then uses a bounded HF streaming shuffle buffer
controlled by `--stream-buffer-size`, a rendered-example length buffer controlled
by `--batch-buffer-size`, and CPU prefetch controlled by `--prefetch-batches`.

The first run still downloads model weights once: `karpathy/nanochat-d32` and
`google/siglip-base-patch16-512`.

## Cheap smoke

```bash
VLM_SMOKE_DEVICE=cuda uv run --extra vision --extra gpu python -m pytest tests/test_vlm_smoke.py -q
```

## Visual instruction probe

This streams The Cauldron. It follows the nanoVLM-style `images` + `texts`
schema and avoids per-image COCO URL fetches. Nanochat starts from
`karpathy/nanochat-d32`, the projector starts random, SigLIP stays frozen, and
the trainer updates projector plus nanochat.

```bash
uv run --extra vision --extra gpu python -m scripts.vlm_train \
  --hf-repo HuggingFaceM4/the_cauldron \
  --hf-config all \
  --out-dir $DATA_ROOT/checkpoints/vlm_cauldron_probe \
  --device-type cuda \
  --num-iterations 100 \
  --device-batch-size 24 \
  --max-batch-tokens 12000 \
  --max-examples 4096 \
  --max-seq-len 2048 \
  --save-every 100 \
  --model-step 650 \
  --profile-timing \
  --stream-buffer-size 4096 \
  --prefetch-batches 2 \
  --skip-bad-images \
  --eval-every 100 \
  --eval-examples 100 \
  --vlm-eval-every 100 \
  --vlm-eval-max-per-benchmark 16 \
  --vlm-eval-max-scan 240 \
  --vlm-eval-print-samples 3
```

## Benchmark

```bash
uv run --extra vision --extra gpu python -m scripts.vlm_eval \
  --checkpoint-dir $DATA_ROOT/checkpoints/vlm_cauldron_probe \
  --checkpoint-step 100 \
  --benchmarks mmstar,scienceqa,chartqa,mmmu,textvqa \
  --limit 16 \
  --max-scan 240 \
  --print-samples 3 \
  --out $DATA_ROOT/bench/vlm_cauldron_probe.json \
  --device-type cuda \
  --model-step 650
```

Compare this eval against the text-only/base baseline or an earlier checkpoint.
The first useful signal is whether the subset scores improve and whether sample
generations are image-relevant.

## Train-Time Eval

`vlm_train.py` follows the same split as nanochat SFT:

- `--eval-every`: fixed held-out BPB/loss reserved from the same HF stream.
- `--vlm-eval-every`: fixed benchmark generation subset from MMStar, ScienceQA,
  ChartQA, MMMU, and TextVQA.

For scaled tracking with roughly 500 benchmark examples every 2000 steps:

```bash
  --eval-every 200 \
  --eval-examples 100 \
  --stream-buffer-size 4096 \
  --prefetch-batches 2 \
  --skip-bad-images \
  --vlm-eval-every 2000 \
  --vlm-eval-max-per-benchmark 100 \
  --vlm-eval-max-scan 2000 \
  --vlm-eval-print-samples 3
```

## Length/Bucket Check

Before a realistic MFU probe, measure the expanded multimodal length
distribution. This is CPU-only: it tokenizes streamed records and exits before
loading SigLIP or the train model. With the default `--hf-checkpoint`, this mode
links only the tokenizer files from the nanochat checkpoint; full model
checkpoint linking is deferred until real training/eval.

```bash
uv run --extra vision --extra gpu python -m scripts.vlm_train \
  --hf-repo HuggingFaceM4/the_cauldron \
  --hf-config vqav2 \
  --max-seq-len 512 \
  --pad-to-bucket-lens 96,128,192,256,384,512 \
  --max-batch-tokens 21504 \
  --length-stats-examples 2000 \
  --stream-buffer-size 256 \
  --model-step 650
```

Interpret `96` as a diagnostic bucket, not a default training length. One image
expands to 64 LLM input positions, so a global `--max-seq-len 96` leaves only 32
non-image positions for text/chat tokens and role markers. The length report
prints this as `text_cap_1img`; only include the 96 bucket in a realistic probe
if the length report shows meaningful traffic there.

Before launching the H100 probe, also dry-run the actual batch selector. This is
CPU-only and exits before loading SigLIP or the train model.

```bash
uv run --extra vision --extra gpu python -m scripts.vlm_train \
  --hf-repo HuggingFaceM4/the_cauldron \
  --hf-config vqav2 \
  --device-batch-size 512 \
  --max-batch-tokens 21504 \
  --max-seq-len 512 \
  --stream-buffer-size 256 \
  --batch-buffer-size 4096 \
  --bucket-selection cycle \
  --bucket-min-fill-frac 0.75 \
  --pad-to-bucket-lens 128,192,256,384,512 \
  --batch-plan-steps 24 \
  --model-step 650
```

The batch plan should show which buckets are visited, average rows per bucket,
examples per step, dropped selected examples, fill against each bucket's row
cap, padding, `attn_pairs/step`, `planning_elapsed`, raw `records_scanned`,
rendered examples, and rendered-example/sec. `attn_pairs/step` is a causal
attention-pair estimate; it usually grows with the bucket length even when the
padded-token cap is fixed. With `--boundary-aware-pack`, `attn_pairs/step` is
computed from the original packed segment lengths rather than the dense packed
row length. If the plan is underfilled, increase
`--batch-buffer-size`, relax `--bucket-min-fill-frac`, or retune bucket edges
before paying for a GPU probe. If testing `--pack-examples`, include the same
packing flags in this dry run; the report uses the same packed-row length
accounting as training. Treat nonzero `dropped/step` as a red flag for a training
run, because those selected examples were not processed by the packed rows. The
trainer trims dropped packed examples before image open/processor/SigLIP work,
so high `dropped/step` mainly means the packed selection policy is wasting
dataset/rendering bandwidth and the packing shape is probably a poor MFU
direction.

When `--grad-accum-steps > 1`, the batch plan also prints an `optimizer_steps`
line. For a clean bucketed MFU probe this should show `same_bucket` equal to the
number of complete optimizer groups and `mixed_bucket=0`; otherwise the probe is
mixing static sequence shapes inside one measured optimizer step.

Use `--hf-config all` for a later full-distribution audit after the named-config
probe is healthy. Local diagnostics showed the all-config Cauldron stream can
spend a long time resolving and filling before it produces useful throughput
evidence.

## MFU Probe

For a throughput-only MFU check, use the same trainer path but skip eval and use
a larger example batch so the LLM token budget is closer to full. The first pass
keeps the HF shuffle buffer small, logs every step, disables checkpoint writes,
and reads `steady_mfu` after four post-warmup steps instead of waiting for step
10 behind a large stream prefill.

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

On Modal, `mfu_probe` is the cheap unbucketed sanity entrypoint:

```bash
NANOCHAT_MODAL_GPU=H100 uv run --extra vision modal run modal_vlm.py::mfu_probe
```

The Modal `mfu_probe` default uses `--hf-config vqav2` for a low-latency,
reproducible throughput probe. Use `--hf-config all` only after a named-config
probe is healthy; local diagnostics showed the all-config stream can spend a
long time resolving/filling before producing useful MFU evidence.

For the current realistic static-shape path, use the bucketed H100 probe after
the length check:

```bash
NANOCHAT_MODAL_GPU=H100 uv run --extra vision modal run modal_vlm.py::bucketed_mfu_probe
```

At a 21,504 padded-token cap, the expected row ceilings are 168 rows at bucket
128, 112 at 192, 84 at 256, 56 at 384, and 42 at 512. Compare
warmup-excluded aggregate `steady_mfu`, per-step `bucket_steady_mfu`, and the
final `Bucket steady stats` table before increasing the token cap. The per-bucket
numbers are the useful signal for whether the remaining MFU loss is only in
longer sequence buckets. The bucketed selector chooses dense rows within the
selected bucket, so high `pad` within a bucket usually means the bucket edges
need retuning or the buffer is too small. `--mfu-warmup-bucket-steps 1` excludes
the first measured occurrence of a new static bucket from steady MFU, which keeps
first-time compile/setup cost out of the mixed-bucket steady-state metric.
`steady_mfu` and `padded_llm_mfu` now use the same sequence-aware FLOP accounting
as the `seq_*` fields. They use the actual bucket or segment length for the
attention term and, under selective loss, charge lm-head FLOPs only for supervised
loss tokens. Use `train/token_estimate_mfu` only when comparing against older log
lines that used the full-lm-head-per-token estimate; use
`train/steady_token_estimate_mfu` for the matching warmup-excluded comparison.
`--bucket-selection cycle` makes the probe visit available static buckets in
order; `--bucket-min-fill-frac 0.75` keeps the probe from timing an underfilled
bucket before enough rows for that bucket have accumulated. Use the default
`sample` policy and zero minimum fill for normal training.

`bucketed_mfu_probe` defaults to `--compile-model`, `--no-selective-loss`,
`--prefetch-batches 8`, `--prefetch-workers 4`, `--batch-buffer-size 4096`,
`--mfu-warmup-bucket-steps 1`, and
`--pad-to-bucket-lens 128,192,256,384,512`. If infrastructure has changed, run
the cheap `mfu_probe` sanity check first. For MFU comparison, use the bucketed
entrypoint so the logs are directly comparable to the current 21,504-token
static-shape plan.
If a full-CE static probe OOMs from materializing logits, keep
`--no-selective-loss` and add `--loss-chunk-size 4096` or `8192`. This preserves
the same ignore-index CE and softcap semantics while computing the lm head in
chunks.

The bucketed probe does not enable `--profile-timing` by default. That keeps
extra CUDA synchronizations out of the target MFU number. Run
`bucketed_mfu_probe --profile-timing` only as a separate attribution pass when
the clean probe still shows a bottleneck.

If optimizer overhead is still visible after the bucketed probe, try
`--grad-accum-steps 2` on `bucketed_mfu_probe`. The entrypoint automatically
sets `--bucket-cycle-repeat` to the accumulation count unless explicitly
overridden, so each optimizer step consumes repeated microbatches from one static
bucket instead of mixing bucket shapes and losing per-bucket MFU accounting.

Do not raise `--max-batch-tokens` without watching `tokens useful/padded`,
`pad`, and peak memory. A 24K padded-token H100 probe exceeded memory in the
unchanged LLM logits path, and a 16K probe without length bucketing wasted more
than 60% of padded tokens. Keep `--batch-buffer-size` above the device batch size
when comparing larger token budgets so the VLM batcher can group similar lengths.

`--pack-examples N` is available as an opt-in VLM-side experiment that packs
multiple short image-text examples into one sequence row. It does not change LLM
code. Early H100 probes showed it reduces row count and padding, but did not beat
the default 12K probe and 16K packed-token probes still OOMed, so keep it
experimental. If using it, also consider `--pack-max-seq-len` and inspect
`rows`, useful/padded tokens, `max_seq`, and peak memory.

`--boundary-aware-pack` makes packed rows semantically correct by resetting RoPE
positions at packed example boundaries, preventing smear across boundaries, and
flattening real segment tokens through a varlen attention wrapper. On H100 with
the FA3 kernel available, this calls `flash_attn_varlen_func`; on CPU/non-Hopper
it falls back to SDPA per segment for correctness. H100 MFU for this path is
still unproven until Modal is resumed.

Boundary-aware packing skips the boundary-only input token that would otherwise
appear between two concatenated examples. The final token of the previous
example is still used as the previous target, but it is not processed as an
extra ignored input, so segment lengths match running each example alone. Loose
packing without `--boundary-aware-pack` keeps ordinary dense row attention and
now requires `--allow-leaky-pack`; treat it only as a semantics-relaxed
diagnostic ablation, not a training recipe.

For streamed packed batches, the selector is pack-aware: `--device-batch-size`
is a candidate window, not the number of examples that must be consumed. The
selector removes only examples that fit the packed `--max-batch-tokens` budget
and leaves the rest in the rendered buffer for later batches. This is why the
packed H100 probe can use a 512-example candidate window without the old
hundreds-of-examples dropped/step behavior.
With `--bucket-selection max-tokens`, candidate windows are ranked first by
useful packed tokens and then by padding and the same attention-pair estimate
used by training. Boundary-aware windows use original segment lengths for this
tie-breaker, not dense packed row lengths.

The packed H100 probe also uses `--flatten-packed-batch`. The selector still
forms short packed rows under `--pack-max-seq-len`, but the actual training batch
concatenates those rows into one compact varlen sequence after image encoding.
That keeps strict `cu_seqlens` boundaries while avoiding packed-row padding and
the per-layer q/k/v gather plus output scatter used by padded row-major varlen
batches.
In this compact mode, `--max-batch-tokens` is enforced on total useful tokens.
Near-full candidate windows are grouped into 512-token fill bands, then the
selector prefers lower segment-attention work inside the band; this avoids
trading a tiny token-fill gain for much longer packed segments.
Compact varlen probes do not treat the flattened row length as a static MFU
bucket, because that length can vary by a few tokens every step. Per-bucket
warmup is therefore only meaningful for dense/static bucketed probes.
Use `packed_random_mfu_probe` as the representative-selection ablation. It is
the same packed recipe with `--bucket-selection random`, which samples across the
rendered buffer under the same useful-token cap, so `avg_segment`/`max_segment`
should better match the source mixture at the cost of more attention work than
the throughput smoke.
Use `packed_mfu_probe --bucket-selection max-compute` as the compute-heavy
selector ablation. It keeps the same near-full useful-token fill bands as
`max-tokens`, but when candidate windows are similarly full it prefers the
window with more segment attention work. This usually means fewer, longer
examples per 32K/65K step, which can improve effective LLM MFU if frozen SigLIP,
data, or varlen launch overhead scales more with image/segment count than with
LLM FLOPs.
Batch-plan logs report segment count, `avg_segment`, `p50_segment`,
`p90_segment`, `max_segment`, and `attn_pairs/token`. For compact packed probes,
`max_seq` is the flattened row length, while these segment metrics describe the
real per-example causal sequences seen by the FA3 varlen kernel. Training logs
also print `avg_segment`, `p50_segment`, `p90_segment`, `max_segment`,
`near_cap`, `cap_hits`, and `attn_pairs/token`, with matching W&B keys
`train/avg_segment_len`, `train/p50_segment_len`, `train/p90_segment_len`,
`train/max_segment_len`, `train/near_cap_segments`, `train/cap_segments`, and
`train/attention_pairs_per_token`. Useful-token MFU is logged both under the
legacy key `train/mfu` and the explicit key `train/eff_llm_mfu`; padded-token
MFU is logged as both `train/padded_mfu` and `train/padded_llm_mfu`. The
warmup-excluded window has matching steady fields such as
`train/steady_eff_llm_mfu`, `train/steady_padded_llm_mfu`,
`train/steady_avg_segment_len`,
`train/steady_p50_segment_len`, `train/steady_p90_segment_len`,
`train/steady_max_segment_len`, `train/steady_attention_pairs_per_token`,
`train/steady_near_cap_segments_per_step`, and
`train/steady_cap_segments_per_step`, so compare those against `steady_mfu`.

SigLIP pooled features preserve the encoder dtype. On H100 this means pooled
features stay bf16 until the projector, avoiding a large temporary fp32 feature
tensor in packed 32K/65K-token probes.

The dedicated Modal entrypoint for this path is:

```bash
NANOCHAT_MODAL_GPU=H100 uv run --extra vision modal run modal_vlm.py::packed_mfu_probe
```

Before the first packed MFU run, dry-run the packed selector with the same
defaults:

```bash
uv run --extra vision modal run modal_vlm.py::packed_batch_plan
```

This is CPU-only and reports useful tokens, padding, dropped examples, segment
count, `avg_segment`, `p50_segment`, `p90_segment`, `max_segment`, and
`attn_pairs/step` plus `attn_pairs/token` for the compact packed batch. It also
reports `near_cap/step` and `cap_hits/step` so long-segment probes show whether
selected examples are pressing against `--pack-max-seq-len`.
Use `modal_vlm.py::packed_random_batch_plan` for the matching dry run before
`packed_random_mfu_probe`. On a new image or GPU, also verify the
attention backend without loading nanochat or SigLIP:

```bash
NANOCHAT_MODAL_GPU=H100 uv run --extra vision modal run modal_vlm.py::attention_backend
```

It defaults to `--boundary-aware-pack`, `--pack-examples 8`,
`--max-seq-len 1024`, `--pack-max-seq-len 1024`,
`--max-batch-tokens 32768`, `--device-batch-size 512`,
`--batch-buffer-size 4096`, `--bucket-selection max-tokens`,
`--flatten-packed-batch`, `--prefetch-batches 8`, `--prefetch-workers 4`, and
clean MFU timing with no `--profile-timing`. It also passes
`--require-fa3-varlen`, so the run fails instead of silently using the SDPA
fallback if the H100 FA3 varlen kernel is unavailable. It intentionally leaves
`--compile` off for the first varlen proof; if the clean probe is stable, try
`packed_mfu_probe --compile-model` as the next H100 ablation.
The 1024-token cap is for dataset realism, not a larger dense row: compact mode
still enforces `--max-batch-tokens` on useful flattened tokens.
For the default dynamic-row compact path, training concatenates the selected
examples directly into the final varlen row; intermediate packed rows are only
used by the selector and by explicit `--pack-fixed-rows` experiments.
If this 32K-token proof is stable and memory has headroom, the next stress probe
is:

```bash
uv run --extra vision modal run modal_vlm.py::packed_large_batch_plan
NANOCHAT_MODAL_GPU=H100 uv run --extra vision modal run modal_vlm.py::packed_large_mfu_probe
```

Run the CPU-only `packed_large_batch_plan` first. It uses the same 65K useful
token budget, `--device-batch-size 1024`, `--pack-examples 16`, and
`--batch-buffer-size 8192` shape as the large H100 probe, so it catches underfill
or unexpectedly long segment distributions before spending H100 time.

That uses the same compact boundary-aware recipe with `--max-batch-tokens 65536`,
`--device-batch-size 1024`, `--pack-examples 16`, and `--batch-buffer-size 8192`.
For the representative 65K ablation, use the matching random selector pair:

```bash
uv run --extra vision modal run modal_vlm.py::packed_large_random_batch_plan
NANOCHAT_MODAL_GPU=H100 uv run --extra vision modal run modal_vlm.py::packed_large_random_mfu_probe
```

That keeps the large compact shape but forces `--bucket-selection random`, so
`avg_segment`/`p50_segment`/`p90_segment`/`max_segment` should be closer to the
source distribution than the throughput-oriented max-token stress probe.

For a compute-heavy 65K ablation, reuse the large packed entrypoints with
the named compute-heavy entrypoints:

```bash
uv run --extra vision modal run modal_vlm.py::packed_large_compute_batch_plan
NANOCHAT_MODAL_GPU=H100 uv run --extra vision modal run modal_vlm.py::packed_large_compute_mfu_probe
```

This keeps the same compact boundary-aware recipe and token cap as
`packed_large_mfu_probe`, but among near-full candidate windows it prefers
longer segment attention work. Compare its `samples`, `segments`, `avg_segment`,
`p50_segment`, `p90_segment`, `max_segment`, SigLIP/profile timing, and `steady_mfu` against both
`max-tokens` and `random`.

If boundary-aware varlen attention is the bottleneck, run the semantics-relaxed
nanoVLM-style ablation:

```bash
uv run --extra vision modal run modal_vlm.py::leaky_packed_large_batch_plan
NANOCHAT_MODAL_GPU=H100 uv run --extra vision modal run modal_vlm.py::leaky_packed_large_mfu_probe
```

This uses dense packed rows with `--allow-leaky-pack`, `--pack-examples 16`,
`--max-seq-len 1024`, `--pad-to-bucket-lens 1024`, and a 65K padded-token cap.
It intentionally does not pass `--boundary-aware-pack`, `--flatten-packed-batch`,
or `--require-fa3-varlen`. Treat it as a throughput ablation for the user's
"attention can learn unrelated examples" hypothesis, not as the default semantic
training recipe. A local `vqav2` dry run filled 64 dense rows/step at 2.6%
padding, but used ~33.6M attention pairs/step versus ~3.9M for the
boundary-aware compact 65K max-token plan, so interpret the H100 result as a
static dense-row efficiency tradeoff against much more attention work.

If the clean 32K probe misses the target and needs bottleneck attribution, run:

```bash
NANOCHAT_MODAL_GPU=H100 uv run --extra vision modal run modal_vlm.py::packed_profile_mfu_probe
```

If the 32K proof works but the 65K stress probe misses the target or changes
shape enough to need attribution, profile the matching large shape:

```bash
NANOCHAT_MODAL_GPU=H100 uv run --extra vision modal run modal_vlm.py::packed_large_profile_mfu_probe
```

If the representative 65K random probe misses the target, profile that same
random-selection shape:

```bash
NANOCHAT_MODAL_GPU=H100 uv run --extra vision modal run modal_vlm.py::packed_large_random_profile_mfu_probe
```

If the compute-heavy 65K probe misses the target, profile that exact
`max-compute` shape:

```bash
NANOCHAT_MODAL_GPU=H100 uv run --extra vision modal run modal_vlm.py::packed_large_compute_profile_mfu_probe
```

When using `--profile-timing`, prefer the final `Steady timing totals` line over
individual step timings. It uses the same warmup exclusion as `steady_mfu`, so a
low MFU result can be attributed to data/image preprocessing, SigLIP,
packed row construction, projector forward inside batch construction,
forward/backward, optimizer time, or remaining
unattributed wall time without mixing in startup effects.

`--grad-accum-steps` is exposed for MFU probes, but the first H100 check at
`grad_accum_steps=4` did not improve `steady_mfu`; it mainly repeated the same
microbatch work four times and amortized only a small optimizer cost.

## Go/no-go criteria

Use this cheap probe as a gate before spending on longer VLM training:

- Continue if training logs show stable loss, nonzero `samples/sec`/`tokens/sec`,
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
before = json.loads((bench / "vlm_cauldron_step100.json").read_text())
after = json.loads((bench / "vlm_cauldron_step500.json").read_text())

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

- 100 steps at batch 24 is expected to be tens of minutes on A100-80GB.
- Use `--profile-timing` output for actual ETA and bottleneck attribution: logs
  include `samples/sec`, useful `tokens/sec`, `eff_llm_mfu` (`train/mfu`),
  useful/padded token counts, max sequence length, `padded_llm_mfu`,
  warmup-excluded `steady_mfu`, padding percentage, current allocated/reserved
  memory, peak allocated/reserved memory, prefetch wait, dataset/render time,
  image open/decode, processor, H2D, SigLIP, packed row construction,
  multimodal tensor construction, projector forward inside batch construction,
  LLM forward/backward, total optimizer timing, split projector/LLM optimizer
  timing, and remaining unattributed wall time. The `batch` bucket synchronizes
  before recording when `--profile-timing` is set, so asynchronous multimodal
  tensor/scatter work is charged there instead of leaking into `other`. W&B
  receives the same `timing/*_sec` fields, plus `timing/other_sec`,
  `timing/other_frac`, `timing/steady_other_sec`, and
  `timing/steady_other_frac`; per-step GPU memory keys are
  `gpu/allocated_mib`, `gpu/reserved_mib`, `gpu/max_allocated_mib`, and
  `gpu/max_reserved_mib`.
- `eff_llm_mfu` is the optimization target. It is estimated LLM train FLOPs over
  total VLM step wall time and does not include frozen SigLIP/projector FLOPs.
  For selective VLM loss it charges the lm-head only for supervised loss tokens,
  since image/user/pad positions do not run that matmul. The auxiliary
  `train/token_estimate_mfu` and `train/steady_token_estimate_mfu` fields keep
  the older full-lm-head-per-token estimate for comparison.
- Step 1 is not a valid steady-state MFU datapoint. Use warmup-excluded
  `steady_mfu`; for the fast probe, `--log-every 1` makes this visible before
  step 10.
- `--compile` enables `torch.compile(dynamic=True)` for the LLM path. It adds
  major startup cost, so only compare MFU after warmup and only use it when the
  run is long enough to amortize compile time.
- The GPU extra installs `torchvision` so Transformers can use the fast SigLIP
  image processor when available; without it the processor timing may overstate
  CPU transform cost.
