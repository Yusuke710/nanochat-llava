"""
Minimal Modal entrypoints for nanochat-llava v0.

Default GPU is A100-80GB. Switch to H100 with:
NANOCHAT_MODAL_GPU=H100 modal run modal_vlm.py::train
"""

from __future__ import annotations

import os
import shlex
import subprocess

import modal


APP_NAME = "nanochat-llava-v0"
GPU_TYPE = os.environ.get("NANOCHAT_MODAL_GPU", "A100-80GB")
DEFAULT_HF_REPO = "HuggingFaceM4/the_cauldron"
DEFAULT_TRAIN_HF_CONFIG = "all"
DEFAULT_PROBE_HF_CONFIG = "vqav2"
DEFAULT_BUCKETED_PROBE_LENS = "128,192,256,384,512"
DEFAULT_PACKED_PROBE_BATCH_SIZE = 512
DEFAULT_PACKED_PROBE_MAX_BATCH_TOKENS = 32768
DEFAULT_PACKED_PROBE_MAX_SEQ_LEN = 1024
DEFAULT_PACKED_PROBE_EXAMPLES = 8
DEFAULT_PACKED_LARGE_PROBE_BATCH_SIZE = 1024
DEFAULT_PACKED_LARGE_PROBE_MAX_BATCH_TOKENS = 65536
DEFAULT_PACKED_LARGE_PROBE_BATCH_BUFFER_SIZE = 8192
DEFAULT_PACKED_LARGE_PROBE_EXAMPLES = 16
VOLUME_DIRS = ["/vol/datasets", "/vol/checkpoints", "/vol/logs", "/vol/bench", "/vol/nanochat", "/vol/hf"]
VOL = modal.Volume.from_name("nanochat-llava-v0", create_if_missing=True)

SECRETS = [modal.Secret.from_name(os.environ.get("NANOCHAT_MODAL_HF_SECRET", "huggingface-secret"))]
if os.environ.get("NANOCHAT_MODAL_WANDB_SECRET"):
    SECRETS.append(modal.Secret.from_name(os.environ["NANOCHAT_MODAL_WANDB_SECRET"]))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.9.1",
        "torchvision>=0.24.1",
        "datasets>=4.0.0",
        "transformers>=4.57.3,<5",
        "tokenizers>=0.22.0",
        "tiktoken>=0.11.0",
        "rustbpe>=0.1.0",
        "pillow>=11.0.0",
        "huggingface-hub>=0.34.0",
        "modal>=1.0.0",
        "wandb>=0.21.3",
        "kernels>=0.11.7",
        "pytest>=8.0.0",
    )
    .add_local_dir("nanochat", remote_path="/root/nanochat-llava/nanochat", copy=True)
    .add_local_dir("scripts", remote_path="/root/nanochat-llava/scripts", copy=True)
    .add_local_dir("tasks", remote_path="/root/nanochat-llava/tasks", copy=True)
    .add_local_dir("tests", remote_path="/root/nanochat-llava/tests", copy=True)
    .add_local_file("pyproject.toml", remote_path="/root/nanochat-llava/pyproject.toml", copy=True)
)

app = modal.App(APP_NAME, image=image)


def build_train_cmd(
    init_checkpoint_dir: str = "",
    init_checkpoint_step: int = 0,
    out_dir: str = "/vol/checkpoints/vlm",
    num_iterations: int = 1000,
    batch_size: int = 24,
    grad_accum_steps: int = 1,
    max_batch_tokens: int = 12000,
    max_seq_len: int = 2048,
    max_examples: int = -1,
    run: str = "dummy",
    model_step: int = 650,
    profile_timing: bool = False,
    hf_repo: str = DEFAULT_HF_REPO,
    hf_config: str = DEFAULT_TRAIN_HF_CONFIG,
    stream_buffer_size: int = 4096,
    batch_buffer_size: int = 0,
    bucket_selection: str = "sample",
    bucket_min_fill_frac: float = 0.0,
    bucket_cycle_repeat: int = 1,
    prefetch_batches: int = 2,
    prefetch_workers: int = 1,
    prefetch_processor: bool = True,
    siglip_forward_batch_size: int = 0,
    skip_bad_images: bool = True,
    compile_model: bool = False,
    fp8: bool = False,
    fp8_recipe: str = "tensorwise",
    drop_zero_value_embeds: bool = False,
    mfu_warmup_steps: int = 2,
    mfu_warmup_bucket_steps: int = 0,
    log_every: int = 10,
    no_save: bool = False,
    pack_examples: int = 1,
    pack_max_seq_len: int = 0,
    pack_fixed_rows: int = 0,
    boundary_aware_pack: bool = False,
    flatten_packed_batch: bool = False,
    allow_leaky_pack: bool = False,
    require_fa3_varlen: bool = False,
    pad_to_max_seq_len: bool = False,
    pad_to_bucket_lens: str = "",
    selective_loss: bool = True,
    loss_chunk_size: int = 0,
):
    cmd = [
        "python",
        "-m",
        "scripts.vlm_train",
        "--run",
        run,
        "--hf-repo",
        hf_repo,
        "--hf-config",
        hf_config,
        "--out-dir",
        out_dir,
        "--device-type",
        "cuda",
        "--num-iterations",
        str(num_iterations),
        "--device-batch-size",
        str(batch_size),
        "--grad-accum-steps",
        str(grad_accum_steps),
        "--max-batch-tokens",
        str(max_batch_tokens),
        "--max-seq-len",
        str(max_seq_len),
        "--model-step",
        str(model_step),
        "--stream-buffer-size",
        str(stream_buffer_size),
        "--bucket-selection",
        bucket_selection,
        "--bucket-min-fill-frac",
        str(bucket_min_fill_frac),
        "--bucket-cycle-repeat",
        str(bucket_cycle_repeat),
        "--prefetch-batches",
        str(prefetch_batches),
        "--prefetch-workers",
        str(prefetch_workers),
        "--mfu-warmup-steps",
        str(mfu_warmup_steps),
        "--mfu-warmup-bucket-steps",
        str(mfu_warmup_bucket_steps),
        "--log-every",
        str(log_every),
    ]
    if no_save:
        cmd += ["--no-save"]
    else:
        cmd += ["--save-every", str(num_iterations)]
    if compile_model:
        cmd += ["--compile"]
    if fp8:
        cmd += ["--fp8", "--fp8-recipe", fp8_recipe]
    if drop_zero_value_embeds:
        cmd += ["--drop-zero-value-embeds"]
    if siglip_forward_batch_size > 0:
        cmd += ["--siglip-forward-batch-size", str(siglip_forward_batch_size)]
    if batch_buffer_size > 0:
        cmd += ["--batch-buffer-size", str(batch_buffer_size)]
    if pack_examples > 1:
        cmd += ["--pack-examples", str(pack_examples)]
    if pack_max_seq_len > 0:
        cmd += ["--pack-max-seq-len", str(pack_max_seq_len)]
    if pack_fixed_rows > 0:
        cmd += ["--pack-fixed-rows", str(pack_fixed_rows)]
    if boundary_aware_pack:
        cmd += ["--boundary-aware-pack"]
    if flatten_packed_batch:
        cmd += ["--flatten-packed-batch"]
    if allow_leaky_pack:
        cmd += ["--allow-leaky-pack"]
    if require_fa3_varlen:
        cmd += ["--require-fa3-varlen"]
    if pad_to_max_seq_len:
        cmd += ["--pad-to-max-seq-len"]
    if pad_to_bucket_lens:
        cmd += ["--pad-to-bucket-lens", pad_to_bucket_lens]
    if not selective_loss:
        cmd += ["--no-selective-loss"]
    if loss_chunk_size > 0:
        cmd += ["--loss-chunk-size", str(loss_chunk_size)]
    if not prefetch_processor:
        cmd += ["--no-prefetch-processor"]
    if skip_bad_images:
        cmd += ["--skip-bad-images"]
    else:
        cmd += ["--no-skip-bad-images"]
    if init_checkpoint_dir:
        cmd += ["--init-vlm-checkpoint-dir", init_checkpoint_dir, "--init-vlm-checkpoint-step", str(init_checkpoint_step)]
    if max_examples > 0:
        cmd += ["--max-examples", str(max_examples)]
    if profile_timing:
        cmd += ["--profile-timing"]
    return cmd


def build_mfu_probe_cmd(
    out_dir: str = "/vol/checkpoints/vlm_cauldron_mfu_probe",
    num_iterations: int = 6,
    batch_size: int = 256,
    grad_accum_steps: int = 1,
    max_batch_tokens: int = 12000,
    max_seq_len: int = 2048,
    run: str = "dummy",
    model_step: int = 650,
    hf_repo: str = DEFAULT_HF_REPO,
    hf_config: str = DEFAULT_PROBE_HF_CONFIG,
    stream_buffer_size: int = 256,
    batch_buffer_size: int = 512,
    bucket_selection: str = "sample",
    bucket_min_fill_frac: float = 0.0,
    bucket_cycle_repeat: int = 1,
    prefetch_batches: int = 2,
    prefetch_workers: int = 1,
    siglip_forward_batch_size: int = 0,
    compile_model: bool = False,
    fp8: bool = False,
    fp8_recipe: str = "tensorwise",
    drop_zero_value_embeds: bool = False,
    pack_examples: int = 1,
    pack_max_seq_len: int = 0,
    pack_fixed_rows: int = 0,
    boundary_aware_pack: bool = False,
    flatten_packed_batch: bool = False,
    allow_leaky_pack: bool = False,
    require_fa3_varlen: bool = False,
    pad_to_max_seq_len: bool = False,
    pad_to_bucket_lens: str = "",
    selective_loss: bool = True,
    loss_chunk_size: int = 0,
    mfu_warmup_bucket_steps: int = 0,
    profile_timing: bool = True,
):
    return build_train_cmd(
        out_dir=out_dir,
        num_iterations=num_iterations,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        max_batch_tokens=max_batch_tokens,
        max_seq_len=max_seq_len,
        run=run,
        model_step=model_step,
        profile_timing=profile_timing,
        hf_repo=hf_repo,
        hf_config=hf_config,
        stream_buffer_size=stream_buffer_size,
        batch_buffer_size=batch_buffer_size,
        bucket_selection=bucket_selection,
        bucket_min_fill_frac=bucket_min_fill_frac,
        bucket_cycle_repeat=bucket_cycle_repeat,
        prefetch_batches=prefetch_batches,
        prefetch_workers=prefetch_workers,
        siglip_forward_batch_size=siglip_forward_batch_size,
        skip_bad_images=True,
        compile_model=compile_model,
        fp8=fp8,
        fp8_recipe=fp8_recipe,
        drop_zero_value_embeds=drop_zero_value_embeds,
        mfu_warmup_steps=2,
        mfu_warmup_bucket_steps=mfu_warmup_bucket_steps,
        log_every=1,
        no_save=True,
        pack_examples=pack_examples,
        pack_max_seq_len=pack_max_seq_len,
        pack_fixed_rows=pack_fixed_rows,
        boundary_aware_pack=boundary_aware_pack,
        flatten_packed_batch=flatten_packed_batch,
        allow_leaky_pack=allow_leaky_pack,
        require_fa3_varlen=require_fa3_varlen,
        pad_to_max_seq_len=pad_to_max_seq_len,
        pad_to_bucket_lens=pad_to_bucket_lens,
        selective_loss=selective_loss,
        loss_chunk_size=loss_chunk_size,
    )


def build_bucketed_mfu_probe_cmd(
    out_dir: str = "/vol/checkpoints/vlm_cauldron_mfu_probe",
    num_iterations: int = 14,
    batch_size: int = 512,
    grad_accum_steps: int = 1,
    max_batch_tokens: int = 21504,
    max_seq_len: int = 512,
    run: str = "dummy",
    model_step: int = 650,
    hf_repo: str = DEFAULT_HF_REPO,
    hf_config: str = DEFAULT_PROBE_HF_CONFIG,
    stream_buffer_size: int = 256,
    batch_buffer_size: int = 4096,
    bucket_selection: str = "max-tokens",
    prefetch_batches: int = 8,
    prefetch_workers: int = 4,
    compile_model: bool = True,
    fp8: bool = False,
    fp8_recipe: str = "tensorwise",
    pad_to_bucket_lens: str = DEFAULT_BUCKETED_PROBE_LENS,
    bucket_cycle_repeat: int = 0,
    profile_timing: bool = False,
    boundary_aware_pack: bool = False,
    require_fa3_varlen: bool = False,
    loss_chunk_size: int = 0,
):
    repeat = bucket_cycle_repeat if bucket_cycle_repeat > 0 else max(1, grad_accum_steps)
    return build_mfu_probe_cmd(
        out_dir=out_dir,
        num_iterations=num_iterations,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        max_batch_tokens=max_batch_tokens,
        max_seq_len=max_seq_len,
        run=run,
        model_step=model_step,
        hf_repo=hf_repo,
        hf_config=hf_config,
        stream_buffer_size=stream_buffer_size,
        batch_buffer_size=batch_buffer_size,
        bucket_selection="cycle",
        bucket_min_fill_frac=0.75,
        bucket_cycle_repeat=repeat,
        prefetch_batches=prefetch_batches,
        prefetch_workers=prefetch_workers,
        compile_model=compile_model,
        fp8=fp8,
        fp8_recipe=fp8_recipe,
        mfu_warmup_bucket_steps=1,
        pad_to_bucket_lens=pad_to_bucket_lens,
        selective_loss=False,
        loss_chunk_size=loss_chunk_size,
        profile_timing=profile_timing,
        boundary_aware_pack=boundary_aware_pack,
        require_fa3_varlen=require_fa3_varlen,
    )


def build_packed_mfu_probe_cmd(
    out_dir: str = "/vol/checkpoints/vlm_cauldron_packed_mfu_probe",
    num_iterations: int = 10,
    batch_size: int = DEFAULT_PACKED_PROBE_BATCH_SIZE,
    grad_accum_steps: int = 1,
    max_batch_tokens: int = DEFAULT_PACKED_PROBE_MAX_BATCH_TOKENS,
    max_seq_len: int = DEFAULT_PACKED_PROBE_MAX_SEQ_LEN,
    run: str = "dummy",
    model_step: int = 650,
    hf_repo: str = DEFAULT_HF_REPO,
    hf_config: str = DEFAULT_PROBE_HF_CONFIG,
    stream_buffer_size: int = 256,
    batch_buffer_size: int = 4096,
    bucket_selection: str = "max-tokens",
    prefetch_batches: int = 8,
    prefetch_workers: int = 4,
    compile_model: bool = False,
    fp8: bool = False,
    fp8_recipe: str = "tensorwise",
    pack_examples: int = DEFAULT_PACKED_PROBE_EXAMPLES,
    pack_max_seq_len: int = DEFAULT_PACKED_PROBE_MAX_SEQ_LEN,
    profile_timing: bool = False,
    require_fa3_varlen: bool = True,
    flatten_packed_batch: bool = True,
    loss_chunk_size: int = 0,
):
    return build_mfu_probe_cmd(
        out_dir=out_dir,
        num_iterations=num_iterations,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        max_batch_tokens=max_batch_tokens,
        max_seq_len=max_seq_len,
        run=run,
        model_step=model_step,
        hf_repo=hf_repo,
        hf_config=hf_config,
        stream_buffer_size=stream_buffer_size,
        batch_buffer_size=batch_buffer_size,
        bucket_selection=bucket_selection,
        prefetch_batches=prefetch_batches,
        prefetch_workers=prefetch_workers,
        compile_model=compile_model,
        fp8=fp8,
        fp8_recipe=fp8_recipe,
        pack_examples=pack_examples,
        pack_max_seq_len=pack_max_seq_len,
        boundary_aware_pack=True,
        flatten_packed_batch=flatten_packed_batch,
        require_fa3_varlen=require_fa3_varlen,
        selective_loss=True,
        loss_chunk_size=loss_chunk_size,
        profile_timing=profile_timing,
    )


def build_packed_random_mfu_probe_cmd(**kwargs):
    kwargs.setdefault("out_dir", "/vol/checkpoints/vlm_cauldron_packed_random_mfu_probe")
    kwargs["bucket_selection"] = "random"
    return build_packed_mfu_probe_cmd(**kwargs)


def build_packed_large_mfu_probe_cmd(**kwargs):
    kwargs.setdefault("out_dir", "/vol/checkpoints/vlm_cauldron_packed_large_mfu_probe")
    kwargs.setdefault("batch_size", DEFAULT_PACKED_LARGE_PROBE_BATCH_SIZE)
    kwargs.setdefault("max_batch_tokens", DEFAULT_PACKED_LARGE_PROBE_MAX_BATCH_TOKENS)
    kwargs.setdefault("batch_buffer_size", DEFAULT_PACKED_LARGE_PROBE_BATCH_BUFFER_SIZE)
    kwargs.setdefault("pack_examples", DEFAULT_PACKED_LARGE_PROBE_EXAMPLES)
    return build_packed_mfu_probe_cmd(**kwargs)


def build_packed_large_random_mfu_probe_cmd(**kwargs):
    kwargs.setdefault("out_dir", "/vol/checkpoints/vlm_cauldron_packed_large_random_mfu_probe")
    kwargs["bucket_selection"] = "random"
    return build_packed_large_mfu_probe_cmd(**kwargs)


def build_packed_large_compute_mfu_probe_cmd(**kwargs):
    kwargs.setdefault("out_dir", "/vol/checkpoints/vlm_cauldron_packed_large_compute_mfu_probe")
    kwargs["bucket_selection"] = "max-compute"
    return build_packed_large_mfu_probe_cmd(**kwargs)


def build_packed_profile_mfu_probe_cmd(**kwargs):
    kwargs.setdefault("out_dir", "/vol/checkpoints/vlm_cauldron_packed_profile_mfu_probe")
    kwargs["profile_timing"] = True
    return build_packed_mfu_probe_cmd(**kwargs)


def build_packed_large_profile_mfu_probe_cmd(**kwargs):
    kwargs.setdefault("out_dir", "/vol/checkpoints/vlm_cauldron_packed_large_profile_mfu_probe")
    kwargs["profile_timing"] = True
    return build_packed_large_mfu_probe_cmd(**kwargs)


def build_packed_large_random_profile_mfu_probe_cmd(**kwargs):
    kwargs.setdefault("out_dir", "/vol/checkpoints/vlm_cauldron_packed_large_random_profile_mfu_probe")
    kwargs["profile_timing"] = True
    return build_packed_large_random_mfu_probe_cmd(**kwargs)


def build_packed_large_compute_profile_mfu_probe_cmd(**kwargs):
    kwargs.setdefault("out_dir", "/vol/checkpoints/vlm_cauldron_packed_large_compute_profile_mfu_probe")
    kwargs["profile_timing"] = True
    return build_packed_large_compute_mfu_probe_cmd(**kwargs)


def build_leaky_packed_large_mfu_probe_cmd(
    out_dir: str = "/vol/checkpoints/vlm_cauldron_leaky_packed_large_mfu_probe",
    num_iterations: int = 10,
    batch_size: int = DEFAULT_PACKED_LARGE_PROBE_BATCH_SIZE,
    grad_accum_steps: int = 1,
    max_batch_tokens: int = DEFAULT_PACKED_LARGE_PROBE_MAX_BATCH_TOKENS,
    max_seq_len: int = DEFAULT_PACKED_PROBE_MAX_SEQ_LEN,
    run: str = "dummy",
    model_step: int = 650,
    hf_repo: str = DEFAULT_HF_REPO,
    hf_config: str = DEFAULT_PROBE_HF_CONFIG,
    stream_buffer_size: int = 256,
    batch_buffer_size: int = DEFAULT_PACKED_LARGE_PROBE_BATCH_BUFFER_SIZE,
    bucket_selection: str = "max-tokens",
    prefetch_batches: int = 8,
    prefetch_workers: int = 4,
    compile_model: bool = False,
    fp8: bool = False,
    fp8_recipe: str = "tensorwise",
    pack_examples: int = DEFAULT_PACKED_LARGE_PROBE_EXAMPLES,
    pack_max_seq_len: int = DEFAULT_PACKED_PROBE_MAX_SEQ_LEN,
    pad_to_bucket_lens: str = "",
    profile_timing: bool = False,
    loss_chunk_size: int = 0,
):
    return build_mfu_probe_cmd(
        out_dir=out_dir,
        num_iterations=num_iterations,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        max_batch_tokens=max_batch_tokens,
        max_seq_len=max_seq_len,
        run=run,
        model_step=model_step,
        hf_repo=hf_repo,
        hf_config=hf_config,
        stream_buffer_size=stream_buffer_size,
        batch_buffer_size=batch_buffer_size,
        bucket_selection=bucket_selection,
        prefetch_batches=prefetch_batches,
        prefetch_workers=prefetch_workers,
        compile_model=compile_model,
        fp8=fp8,
        fp8_recipe=fp8_recipe,
        pack_examples=pack_examples,
        pack_max_seq_len=pack_max_seq_len,
        boundary_aware_pack=False,
        flatten_packed_batch=False,
        allow_leaky_pack=True,
        require_fa3_varlen=False,
        pad_to_bucket_lens=pad_to_bucket_lens or str(max_seq_len),
        selective_loss=True,
        loss_chunk_size=loss_chunk_size,
        profile_timing=profile_timing,
    )


def build_packed_batch_plan_cmd(
    batch_plan_steps: int = 2,
    batch_size: int = DEFAULT_PACKED_PROBE_BATCH_SIZE,
    max_batch_tokens: int = DEFAULT_PACKED_PROBE_MAX_BATCH_TOKENS,
    max_seq_len: int = DEFAULT_PACKED_PROBE_MAX_SEQ_LEN,
    model_step: int = 650,
    hf_repo: str = DEFAULT_HF_REPO,
    hf_config: str = DEFAULT_PROBE_HF_CONFIG,
    stream_buffer_size: int = 256,
    batch_buffer_size: int = 4096,
    bucket_selection: str = "max-tokens",
    pack_examples: int = DEFAULT_PACKED_PROBE_EXAMPLES,
    pack_max_seq_len: int = DEFAULT_PACKED_PROBE_MAX_SEQ_LEN,
):
    return [
        "python",
        "-m",
        "scripts.vlm_train",
        "--device-type",
        "cpu",
        "--hf-repo",
        hf_repo,
        "--hf-config",
        hf_config,
        "--device-batch-size",
        str(batch_size),
        "--max-batch-tokens",
        str(max_batch_tokens),
        "--max-seq-len",
        str(max_seq_len),
        "--stream-buffer-size",
        str(stream_buffer_size),
        "--batch-buffer-size",
        str(batch_buffer_size),
        "--bucket-selection",
        bucket_selection,
        "--pack-examples",
        str(pack_examples),
        "--pack-max-seq-len",
        str(pack_max_seq_len),
        "--boundary-aware-pack",
        "--flatten-packed-batch",
        "--batch-plan-steps",
        str(batch_plan_steps),
        "--model-step",
        str(model_step),
    ]


def build_packed_random_batch_plan_cmd(**kwargs):
    kwargs["bucket_selection"] = "random"
    return build_packed_batch_plan_cmd(**kwargs)


def build_packed_large_batch_plan_cmd(**kwargs):
    kwargs.setdefault("batch_size", DEFAULT_PACKED_LARGE_PROBE_BATCH_SIZE)
    kwargs.setdefault("max_batch_tokens", DEFAULT_PACKED_LARGE_PROBE_MAX_BATCH_TOKENS)
    kwargs.setdefault("batch_buffer_size", DEFAULT_PACKED_LARGE_PROBE_BATCH_BUFFER_SIZE)
    kwargs.setdefault("pack_examples", DEFAULT_PACKED_LARGE_PROBE_EXAMPLES)
    return build_packed_batch_plan_cmd(**kwargs)


def build_packed_large_random_batch_plan_cmd(**kwargs):
    kwargs["bucket_selection"] = "random"
    return build_packed_large_batch_plan_cmd(**kwargs)


def build_packed_large_compute_batch_plan_cmd(**kwargs):
    kwargs["bucket_selection"] = "max-compute"
    return build_packed_large_batch_plan_cmd(**kwargs)


def build_leaky_packed_large_batch_plan_cmd(
    batch_plan_steps: int = 2,
    batch_size: int = DEFAULT_PACKED_LARGE_PROBE_BATCH_SIZE,
    max_batch_tokens: int = DEFAULT_PACKED_LARGE_PROBE_MAX_BATCH_TOKENS,
    max_seq_len: int = DEFAULT_PACKED_PROBE_MAX_SEQ_LEN,
    model_step: int = 650,
    hf_repo: str = DEFAULT_HF_REPO,
    hf_config: str = DEFAULT_PROBE_HF_CONFIG,
    stream_buffer_size: int = 256,
    batch_buffer_size: int = DEFAULT_PACKED_LARGE_PROBE_BATCH_BUFFER_SIZE,
    bucket_selection: str = "max-tokens",
    pack_examples: int = DEFAULT_PACKED_LARGE_PROBE_EXAMPLES,
    pack_max_seq_len: int = DEFAULT_PACKED_PROBE_MAX_SEQ_LEN,
    pad_to_bucket_lens: str = "",
):
    return [
        "python",
        "-m",
        "scripts.vlm_train",
        "--device-type",
        "cpu",
        "--hf-repo",
        hf_repo,
        "--hf-config",
        hf_config,
        "--device-batch-size",
        str(batch_size),
        "--max-batch-tokens",
        str(max_batch_tokens),
        "--max-seq-len",
        str(max_seq_len),
        "--stream-buffer-size",
        str(stream_buffer_size),
        "--batch-buffer-size",
        str(batch_buffer_size),
        "--bucket-selection",
        bucket_selection,
        "--pack-examples",
        str(pack_examples),
        "--pack-max-seq-len",
        str(pack_max_seq_len),
        "--allow-leaky-pack",
        "--pad-to-bucket-lens",
        pad_to_bucket_lens or str(max_seq_len),
        "--batch-plan-steps",
        str(batch_plan_steps),
        "--model-step",
        str(model_step),
    ]


def build_attention_backend_cmd(require_fa3_varlen: bool = True):
    cmd = ["python", "-m", "scripts.vlm_train", "--device-type", "cuda", "--attention-backend-report"]
    if require_fa3_varlen:
        cmd += ["--boundary-aware-pack", "--require-fa3-varlen"]
    return cmd


def build_eval_cmd(
    checkpoint_dir: str = "/vol/checkpoints/vlm",
    checkpoint_step: int = 1000,
    out: str = "/vol/bench/vlm_eval.json",
    benchmarks: str = "mmstar,scienceqa,chartqa,mmmu,textvqa",
    mmmu_configs: str = "Accounting",
    limit: int = 32,
    max_scan: int = 0,
    print_samples: int = 0,
    model_step: int = 650,
):
    cmd = [
        "python",
        "-m",
        "scripts.vlm_eval",
        "--benchmarks",
        benchmarks,
        "--mmmu-configs",
        mmmu_configs,
        "--checkpoint-dir",
        checkpoint_dir,
        "--checkpoint-step",
        str(checkpoint_step),
        "--limit",
        str(limit),
        "--out",
        out,
        "--model-step",
        str(model_step),
    ]
    if max_scan > 0:
        cmd += ["--max-scan", str(max_scan)]
    if print_samples > 0:
        cmd += ["--print-samples", str(print_samples)]
    return cmd


def _ensure_volume_dirs():
    for path in VOLUME_DIRS:
        os.makedirs(path, exist_ok=True)


def _run(args):
    env = os.environ.copy()
    env.setdefault("NANOCHAT_BASE_DIR", "/vol/nanochat")
    env.setdefault("HF_HOME", "/vol/hf")
    env.setdefault("NANOCHAT_SIGLIP_CACHE_DIR", "/vol/hf/siglip")
    env.setdefault("WANDB_DIR", "/vol/logs/wandb")
    _ensure_volume_dirs()
    print("GPU_TYPE", GPU_TYPE)
    print("+", " ".join(shlex.quote(a) for a in args))
    subprocess.run(args, cwd="/root/nanochat-llava", env=env, check=True)
    VOL.commit()


def build_doctor_summary():
    return {
        "app": APP_NAME,
        "gpu": GPU_TYPE,
        "volume_dirs": VOLUME_DIRS,
        "train_preview": build_train_cmd(num_iterations=1, batch_size=1, max_examples=1),
        "mfu_probe_preview": build_mfu_probe_cmd(),
        "bucketed_mfu_probe_preview": build_bucketed_mfu_probe_cmd(),
        "packed_mfu_probe_preview": build_packed_mfu_probe_cmd(),
        "packed_random_mfu_probe_preview": build_packed_random_mfu_probe_cmd(),
        "packed_large_mfu_probe_preview": build_packed_large_mfu_probe_cmd(),
        "packed_large_random_mfu_probe_preview": build_packed_large_random_mfu_probe_cmd(),
        "packed_large_compute_mfu_probe_preview": build_packed_large_compute_mfu_probe_cmd(),
        "packed_profile_mfu_probe_preview": build_packed_profile_mfu_probe_cmd(),
        "packed_large_profile_mfu_probe_preview": build_packed_large_profile_mfu_probe_cmd(),
        "packed_large_random_profile_mfu_probe_preview": build_packed_large_random_profile_mfu_probe_cmd(),
        "packed_large_compute_profile_mfu_probe_preview": build_packed_large_compute_profile_mfu_probe_cmd(),
        "leaky_packed_large_mfu_probe_preview": build_leaky_packed_large_mfu_probe_cmd(),
        "packed_batch_plan_preview": build_packed_batch_plan_cmd(),
        "packed_random_batch_plan_preview": build_packed_random_batch_plan_cmd(),
        "packed_large_batch_plan_preview": build_packed_large_batch_plan_cmd(),
        "packed_large_random_batch_plan_preview": build_packed_large_random_batch_plan_cmd(),
        "packed_large_compute_batch_plan_preview": build_packed_large_compute_batch_plan_cmd(),
        "leaky_packed_large_batch_plan_preview": build_leaky_packed_large_batch_plan_cmd(),
        "attention_backend_preview": build_attention_backend_cmd(),
        "eval_preview": build_eval_cmd(limit=1, max_scan=2, benchmarks="mmstar"),
    }


@app.function(volumes={"/vol": VOL}, timeout=10 * 60)
def doctor():
    summary = build_doctor_summary()
    print("Modal doctor")
    print("APP_NAME", summary["app"])
    print("GPU_TYPE", summary["gpu"])
    print("VOLUME_DIRS", ",".join(summary["volume_dirs"]))
    for key in [
        "train_preview",
        "mfu_probe_preview",
        "bucketed_mfu_probe_preview",
        "packed_mfu_probe_preview",
        "packed_random_mfu_probe_preview",
        "packed_large_mfu_probe_preview",
        "packed_large_random_mfu_probe_preview",
        "packed_large_compute_mfu_probe_preview",
        "packed_profile_mfu_probe_preview",
        "packed_large_profile_mfu_probe_preview",
        "packed_large_random_profile_mfu_probe_preview",
        "packed_large_compute_profile_mfu_probe_preview",
        "leaky_packed_large_mfu_probe_preview",
        "packed_batch_plan_preview",
        "packed_random_batch_plan_preview",
        "packed_large_batch_plan_preview",
        "packed_large_random_batch_plan_preview",
        "packed_large_compute_batch_plan_preview",
        "leaky_packed_large_batch_plan_preview",
        "attention_backend_preview",
        "eval_preview",
    ]:
        print(key, " ".join(shlex.quote(arg) for arg in summary[key]))
    for module in ["scripts.vlm_train", "scripts.vlm_eval"]:
        subprocess.run(["python", "-m", module, "--help"], cwd="/root/nanochat-llava", check=True, stdout=subprocess.DEVNULL)
        print("help_ok", module)
    VOL.commit()


@app.function(gpu=GPU_TYPE, volumes={"/vol": VOL}, secrets=SECRETS, timeout=30 * 60)
def smoke():
    _run(["python", "-m", "pytest", "tests/test_vlm_smoke.py", "-q"])


@app.function(gpu=GPU_TYPE, volumes={"/vol": VOL}, secrets=SECRETS, timeout=10 * 60)
def attention_backend(require_fa3_varlen: bool = True):
    _run(build_attention_backend_cmd(require_fa3_varlen=require_fa3_varlen))


@app.function(volumes={"/vol": VOL}, secrets=SECRETS, timeout=2 * 60 * 60)
def packed_batch_plan(
    batch_plan_steps: int = 2,
    batch_size: int = DEFAULT_PACKED_PROBE_BATCH_SIZE,
    max_batch_tokens: int = DEFAULT_PACKED_PROBE_MAX_BATCH_TOKENS,
    max_seq_len: int = DEFAULT_PACKED_PROBE_MAX_SEQ_LEN,
    model_step: int = 650,
    hf_repo: str = DEFAULT_HF_REPO,
    hf_config: str = DEFAULT_PROBE_HF_CONFIG,
    stream_buffer_size: int = 256,
    batch_buffer_size: int = 4096,
    bucket_selection: str = "max-tokens",
    pack_examples: int = DEFAULT_PACKED_PROBE_EXAMPLES,
    pack_max_seq_len: int = DEFAULT_PACKED_PROBE_MAX_SEQ_LEN,
):
    _run(build_packed_batch_plan_cmd(
        batch_plan_steps=batch_plan_steps,
        batch_size=batch_size,
        max_batch_tokens=max_batch_tokens,
        max_seq_len=max_seq_len,
        model_step=model_step,
        hf_repo=hf_repo,
        hf_config=hf_config,
        stream_buffer_size=stream_buffer_size,
        batch_buffer_size=batch_buffer_size,
        bucket_selection=bucket_selection,
        pack_examples=pack_examples,
        pack_max_seq_len=pack_max_seq_len,
    ))


@app.function(volumes={"/vol": VOL}, secrets=SECRETS, timeout=2 * 60 * 60)
def packed_random_batch_plan(
    batch_plan_steps: int = 2,
    batch_size: int = DEFAULT_PACKED_PROBE_BATCH_SIZE,
    max_batch_tokens: int = DEFAULT_PACKED_PROBE_MAX_BATCH_TOKENS,
    max_seq_len: int = DEFAULT_PACKED_PROBE_MAX_SEQ_LEN,
    model_step: int = 650,
    hf_repo: str = DEFAULT_HF_REPO,
    hf_config: str = DEFAULT_PROBE_HF_CONFIG,
    stream_buffer_size: int = 256,
    batch_buffer_size: int = 4096,
    pack_examples: int = DEFAULT_PACKED_PROBE_EXAMPLES,
    pack_max_seq_len: int = DEFAULT_PACKED_PROBE_MAX_SEQ_LEN,
):
    _run(build_packed_random_batch_plan_cmd(
        batch_plan_steps=batch_plan_steps,
        batch_size=batch_size,
        max_batch_tokens=max_batch_tokens,
        max_seq_len=max_seq_len,
        model_step=model_step,
        hf_repo=hf_repo,
        hf_config=hf_config,
        stream_buffer_size=stream_buffer_size,
        batch_buffer_size=batch_buffer_size,
        pack_examples=pack_examples,
        pack_max_seq_len=pack_max_seq_len,
    ))


@app.function(volumes={"/vol": VOL}, secrets=SECRETS, timeout=2 * 60 * 60)
def packed_large_batch_plan(
    batch_plan_steps: int = 2,
    batch_size: int = DEFAULT_PACKED_LARGE_PROBE_BATCH_SIZE,
    max_batch_tokens: int = DEFAULT_PACKED_LARGE_PROBE_MAX_BATCH_TOKENS,
    max_seq_len: int = DEFAULT_PACKED_PROBE_MAX_SEQ_LEN,
    model_step: int = 650,
    hf_repo: str = DEFAULT_HF_REPO,
    hf_config: str = DEFAULT_PROBE_HF_CONFIG,
    stream_buffer_size: int = 256,
    batch_buffer_size: int = DEFAULT_PACKED_LARGE_PROBE_BATCH_BUFFER_SIZE,
    bucket_selection: str = "max-tokens",
    pack_examples: int = DEFAULT_PACKED_LARGE_PROBE_EXAMPLES,
    pack_max_seq_len: int = DEFAULT_PACKED_PROBE_MAX_SEQ_LEN,
):
    _run(build_packed_large_batch_plan_cmd(
        batch_plan_steps=batch_plan_steps,
        batch_size=batch_size,
        max_batch_tokens=max_batch_tokens,
        max_seq_len=max_seq_len,
        model_step=model_step,
        hf_repo=hf_repo,
        hf_config=hf_config,
        stream_buffer_size=stream_buffer_size,
        batch_buffer_size=batch_buffer_size,
        bucket_selection=bucket_selection,
        pack_examples=pack_examples,
        pack_max_seq_len=pack_max_seq_len,
    ))


@app.function(volumes={"/vol": VOL}, secrets=SECRETS, timeout=2 * 60 * 60)
def packed_large_random_batch_plan(
    batch_plan_steps: int = 2,
    batch_size: int = DEFAULT_PACKED_LARGE_PROBE_BATCH_SIZE,
    max_batch_tokens: int = DEFAULT_PACKED_LARGE_PROBE_MAX_BATCH_TOKENS,
    max_seq_len: int = DEFAULT_PACKED_PROBE_MAX_SEQ_LEN,
    model_step: int = 650,
    hf_repo: str = DEFAULT_HF_REPO,
    hf_config: str = DEFAULT_PROBE_HF_CONFIG,
    stream_buffer_size: int = 256,
    batch_buffer_size: int = DEFAULT_PACKED_LARGE_PROBE_BATCH_BUFFER_SIZE,
    pack_examples: int = DEFAULT_PACKED_LARGE_PROBE_EXAMPLES,
    pack_max_seq_len: int = DEFAULT_PACKED_PROBE_MAX_SEQ_LEN,
):
    _run(build_packed_large_random_batch_plan_cmd(
        batch_plan_steps=batch_plan_steps,
        batch_size=batch_size,
        max_batch_tokens=max_batch_tokens,
        max_seq_len=max_seq_len,
        model_step=model_step,
        hf_repo=hf_repo,
        hf_config=hf_config,
        stream_buffer_size=stream_buffer_size,
        batch_buffer_size=batch_buffer_size,
        pack_examples=pack_examples,
        pack_max_seq_len=pack_max_seq_len,
    ))


@app.function(volumes={"/vol": VOL}, secrets=SECRETS, timeout=2 * 60 * 60)
def packed_large_compute_batch_plan(
    batch_plan_steps: int = 2,
    batch_size: int = DEFAULT_PACKED_LARGE_PROBE_BATCH_SIZE,
    max_batch_tokens: int = DEFAULT_PACKED_LARGE_PROBE_MAX_BATCH_TOKENS,
    max_seq_len: int = DEFAULT_PACKED_PROBE_MAX_SEQ_LEN,
    model_step: int = 650,
    hf_repo: str = DEFAULT_HF_REPO,
    hf_config: str = DEFAULT_PROBE_HF_CONFIG,
    stream_buffer_size: int = 256,
    batch_buffer_size: int = DEFAULT_PACKED_LARGE_PROBE_BATCH_BUFFER_SIZE,
    pack_examples: int = DEFAULT_PACKED_LARGE_PROBE_EXAMPLES,
    pack_max_seq_len: int = DEFAULT_PACKED_PROBE_MAX_SEQ_LEN,
):
    _run(build_packed_large_compute_batch_plan_cmd(
        batch_plan_steps=batch_plan_steps,
        batch_size=batch_size,
        max_batch_tokens=max_batch_tokens,
        max_seq_len=max_seq_len,
        model_step=model_step,
        hf_repo=hf_repo,
        hf_config=hf_config,
        stream_buffer_size=stream_buffer_size,
        batch_buffer_size=batch_buffer_size,
        pack_examples=pack_examples,
        pack_max_seq_len=pack_max_seq_len,
    ))


@app.function(volumes={"/vol": VOL}, secrets=SECRETS, timeout=2 * 60 * 60)
def leaky_packed_large_batch_plan(
    batch_plan_steps: int = 2,
    batch_size: int = DEFAULT_PACKED_LARGE_PROBE_BATCH_SIZE,
    max_batch_tokens: int = DEFAULT_PACKED_LARGE_PROBE_MAX_BATCH_TOKENS,
    max_seq_len: int = DEFAULT_PACKED_PROBE_MAX_SEQ_LEN,
    model_step: int = 650,
    hf_repo: str = DEFAULT_HF_REPO,
    hf_config: str = DEFAULT_PROBE_HF_CONFIG,
    stream_buffer_size: int = 256,
    batch_buffer_size: int = DEFAULT_PACKED_LARGE_PROBE_BATCH_BUFFER_SIZE,
    bucket_selection: str = "max-tokens",
    pack_examples: int = DEFAULT_PACKED_LARGE_PROBE_EXAMPLES,
    pack_max_seq_len: int = DEFAULT_PACKED_PROBE_MAX_SEQ_LEN,
    pad_to_bucket_lens: str = "",
):
    _run(build_leaky_packed_large_batch_plan_cmd(
        batch_plan_steps=batch_plan_steps,
        batch_size=batch_size,
        max_batch_tokens=max_batch_tokens,
        max_seq_len=max_seq_len,
        model_step=model_step,
        hf_repo=hf_repo,
        hf_config=hf_config,
        stream_buffer_size=stream_buffer_size,
        batch_buffer_size=batch_buffer_size,
        bucket_selection=bucket_selection,
        pack_examples=pack_examples,
        pack_max_seq_len=pack_max_seq_len,
        pad_to_bucket_lens=pad_to_bucket_lens,
    ))


@app.function(gpu=GPU_TYPE, volumes={"/vol": VOL}, secrets=SECRETS, timeout=24 * 60 * 60)
def train(
    init_checkpoint_dir: str = "",
    init_checkpoint_step: int = 0,
    out_dir: str = "/vol/checkpoints/vlm",
    num_iterations: int = 1000,
    batch_size: int = 24,
    grad_accum_steps: int = 1,
    max_batch_tokens: int = 12000,
    max_seq_len: int = 2048,
    max_examples: int = -1,
    run: str = "dummy",
    model_step: int = 650,
    profile_timing: bool = False,
    hf_repo: str = DEFAULT_HF_REPO,
    hf_config: str = DEFAULT_TRAIN_HF_CONFIG,
    stream_buffer_size: int = 4096,
    batch_buffer_size: int = 0,
    bucket_selection: str = "sample",
    bucket_min_fill_frac: float = 0.0,
    bucket_cycle_repeat: int = 1,
    prefetch_batches: int = 2,
    prefetch_workers: int = 1,
    prefetch_processor: bool = True,
    siglip_forward_batch_size: int = 0,
    skip_bad_images: bool = True,
    compile_model: bool = False,
    fp8: bool = False,
    fp8_recipe: str = "tensorwise",
    drop_zero_value_embeds: bool = False,
    mfu_warmup_steps: int = 2,
    mfu_warmup_bucket_steps: int = 0,
    log_every: int = 10,
    no_save: bool = False,
    pack_examples: int = 1,
    pack_max_seq_len: int = 0,
    pack_fixed_rows: int = 0,
    boundary_aware_pack: bool = False,
    flatten_packed_batch: bool = False,
    allow_leaky_pack: bool = False,
    require_fa3_varlen: bool = False,
    pad_to_max_seq_len: bool = False,
    pad_to_bucket_lens: str = "",
    selective_loss: bool = True,
    loss_chunk_size: int = 0,
):
    _run(build_train_cmd(
        init_checkpoint_dir=init_checkpoint_dir,
        init_checkpoint_step=init_checkpoint_step,
        out_dir=out_dir,
        num_iterations=num_iterations,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        max_batch_tokens=max_batch_tokens,
        max_seq_len=max_seq_len,
        max_examples=max_examples,
        run=run,
        model_step=model_step,
        profile_timing=profile_timing,
        hf_repo=hf_repo,
        hf_config=hf_config,
        stream_buffer_size=stream_buffer_size,
        batch_buffer_size=batch_buffer_size,
        bucket_selection=bucket_selection,
        bucket_min_fill_frac=bucket_min_fill_frac,
        bucket_cycle_repeat=bucket_cycle_repeat,
        prefetch_batches=prefetch_batches,
        prefetch_workers=prefetch_workers,
        prefetch_processor=prefetch_processor,
        siglip_forward_batch_size=siglip_forward_batch_size,
        skip_bad_images=skip_bad_images,
        compile_model=compile_model,
        fp8=fp8,
        fp8_recipe=fp8_recipe,
        drop_zero_value_embeds=drop_zero_value_embeds,
        mfu_warmup_steps=mfu_warmup_steps,
        mfu_warmup_bucket_steps=mfu_warmup_bucket_steps,
        log_every=log_every,
        no_save=no_save,
        pack_examples=pack_examples,
        pack_max_seq_len=pack_max_seq_len,
        pack_fixed_rows=pack_fixed_rows,
        boundary_aware_pack=boundary_aware_pack,
        flatten_packed_batch=flatten_packed_batch,
        allow_leaky_pack=allow_leaky_pack,
        require_fa3_varlen=require_fa3_varlen,
        pad_to_max_seq_len=pad_to_max_seq_len,
        pad_to_bucket_lens=pad_to_bucket_lens,
        selective_loss=selective_loss,
        loss_chunk_size=loss_chunk_size,
    ))


@app.function(gpu=GPU_TYPE, volumes={"/vol": VOL}, secrets=SECRETS, timeout=2 * 60 * 60)
def mfu_probe(
    out_dir: str = "/vol/checkpoints/vlm_cauldron_mfu_probe",
    num_iterations: int = 6,
    batch_size: int = 256,
    grad_accum_steps: int = 1,
    max_batch_tokens: int = 12000,
    max_seq_len: int = 2048,
    run: str = "dummy",
    model_step: int = 650,
    hf_repo: str = DEFAULT_HF_REPO,
    hf_config: str = DEFAULT_PROBE_HF_CONFIG,
    stream_buffer_size: int = 256,
    batch_buffer_size: int = 512,
    bucket_selection: str = "sample",
    bucket_min_fill_frac: float = 0.0,
    bucket_cycle_repeat: int = 1,
    prefetch_batches: int = 2,
    prefetch_workers: int = 1,
    siglip_forward_batch_size: int = 0,
    compile_model: bool = False,
    fp8: bool = False,
    fp8_recipe: str = "tensorwise",
    drop_zero_value_embeds: bool = False,
    mfu_warmup_bucket_steps: int = 0,
    pack_examples: int = 1,
    pack_max_seq_len: int = 0,
    pack_fixed_rows: int = 0,
    boundary_aware_pack: bool = False,
    flatten_packed_batch: bool = False,
    allow_leaky_pack: bool = False,
    require_fa3_varlen: bool = False,
    pad_to_max_seq_len: bool = False,
    pad_to_bucket_lens: str = "",
    selective_loss: bool = True,
    loss_chunk_size: int = 0,
    profile_timing: bool = True,
):
    _run(build_mfu_probe_cmd(
        out_dir=out_dir,
        num_iterations=num_iterations,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        max_batch_tokens=max_batch_tokens,
        max_seq_len=max_seq_len,
        run=run,
        model_step=model_step,
        hf_repo=hf_repo,
        hf_config=hf_config,
        stream_buffer_size=stream_buffer_size,
        batch_buffer_size=batch_buffer_size,
        bucket_selection=bucket_selection,
        bucket_min_fill_frac=bucket_min_fill_frac,
        bucket_cycle_repeat=bucket_cycle_repeat,
        prefetch_batches=prefetch_batches,
        prefetch_workers=prefetch_workers,
        siglip_forward_batch_size=siglip_forward_batch_size,
        compile_model=compile_model,
        fp8=fp8,
        fp8_recipe=fp8_recipe,
        drop_zero_value_embeds=drop_zero_value_embeds,
        mfu_warmup_bucket_steps=mfu_warmup_bucket_steps,
        pack_examples=pack_examples,
        pack_max_seq_len=pack_max_seq_len,
        pack_fixed_rows=pack_fixed_rows,
        boundary_aware_pack=boundary_aware_pack,
        flatten_packed_batch=flatten_packed_batch,
        allow_leaky_pack=allow_leaky_pack,
        require_fa3_varlen=require_fa3_varlen,
        pad_to_max_seq_len=pad_to_max_seq_len,
        pad_to_bucket_lens=pad_to_bucket_lens,
        selective_loss=selective_loss,
        loss_chunk_size=loss_chunk_size,
        profile_timing=profile_timing,
    ))


@app.function(gpu=GPU_TYPE, volumes={"/vol": VOL}, secrets=SECRETS, timeout=4 * 60 * 60)
def bucketed_mfu_probe(
    out_dir: str = "/vol/checkpoints/vlm_cauldron_mfu_probe",
    num_iterations: int = 14,
    batch_size: int = 512,
    grad_accum_steps: int = 1,
    max_batch_tokens: int = 21504,
    max_seq_len: int = 512,
    run: str = "dummy",
    model_step: int = 650,
    hf_repo: str = DEFAULT_HF_REPO,
    hf_config: str = DEFAULT_PROBE_HF_CONFIG,
    stream_buffer_size: int = 256,
    batch_buffer_size: int = 4096,
    bucket_selection: str = "max-tokens",
    prefetch_batches: int = 8,
    prefetch_workers: int = 4,
    compile_model: bool = True,
    fp8: bool = False,
    fp8_recipe: str = "tensorwise",
    pad_to_bucket_lens: str = DEFAULT_BUCKETED_PROBE_LENS,
    bucket_cycle_repeat: int = 0,
    profile_timing: bool = False,
    boundary_aware_pack: bool = False,
    require_fa3_varlen: bool = False,
    loss_chunk_size: int = 0,
):
    _run(build_bucketed_mfu_probe_cmd(
        out_dir=out_dir,
        num_iterations=num_iterations,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        max_batch_tokens=max_batch_tokens,
        max_seq_len=max_seq_len,
        run=run,
        model_step=model_step,
        hf_repo=hf_repo,
        hf_config=hf_config,
        stream_buffer_size=stream_buffer_size,
        batch_buffer_size=batch_buffer_size,
        bucket_selection=bucket_selection,
        prefetch_batches=prefetch_batches,
        prefetch_workers=prefetch_workers,
        compile_model=compile_model,
        fp8=fp8,
        fp8_recipe=fp8_recipe,
        pad_to_bucket_lens=pad_to_bucket_lens,
        bucket_cycle_repeat=bucket_cycle_repeat,
        profile_timing=profile_timing,
        boundary_aware_pack=boundary_aware_pack,
        require_fa3_varlen=require_fa3_varlen,
        loss_chunk_size=loss_chunk_size,
    ))


@app.function(gpu=GPU_TYPE, volumes={"/vol": VOL}, secrets=SECRETS, timeout=4 * 60 * 60)
def packed_mfu_probe(
    out_dir: str = "/vol/checkpoints/vlm_cauldron_packed_mfu_probe",
    num_iterations: int = 10,
    batch_size: int = DEFAULT_PACKED_PROBE_BATCH_SIZE,
    grad_accum_steps: int = 1,
    max_batch_tokens: int = DEFAULT_PACKED_PROBE_MAX_BATCH_TOKENS,
    max_seq_len: int = DEFAULT_PACKED_PROBE_MAX_SEQ_LEN,
    run: str = "dummy",
    model_step: int = 650,
    hf_repo: str = DEFAULT_HF_REPO,
    hf_config: str = DEFAULT_PROBE_HF_CONFIG,
    stream_buffer_size: int = 256,
    batch_buffer_size: int = 4096,
    bucket_selection: str = "max-tokens",
    prefetch_batches: int = 8,
    prefetch_workers: int = 4,
    compile_model: bool = False,
    fp8: bool = False,
    fp8_recipe: str = "tensorwise",
    pack_examples: int = DEFAULT_PACKED_PROBE_EXAMPLES,
    pack_max_seq_len: int = DEFAULT_PACKED_PROBE_MAX_SEQ_LEN,
    profile_timing: bool = False,
    require_fa3_varlen: bool = True,
    flatten_packed_batch: bool = True,
    loss_chunk_size: int = 0,
):
    _run(build_packed_mfu_probe_cmd(
        out_dir=out_dir,
        num_iterations=num_iterations,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        max_batch_tokens=max_batch_tokens,
        max_seq_len=max_seq_len,
        run=run,
        model_step=model_step,
        hf_repo=hf_repo,
        hf_config=hf_config,
        stream_buffer_size=stream_buffer_size,
        batch_buffer_size=batch_buffer_size,
        bucket_selection=bucket_selection,
        prefetch_batches=prefetch_batches,
        prefetch_workers=prefetch_workers,
        compile_model=compile_model,
        fp8=fp8,
        fp8_recipe=fp8_recipe,
        pack_examples=pack_examples,
        pack_max_seq_len=pack_max_seq_len,
        profile_timing=profile_timing,
        require_fa3_varlen=require_fa3_varlen,
        flatten_packed_batch=flatten_packed_batch,
        loss_chunk_size=loss_chunk_size,
    ))


@app.function(gpu=GPU_TYPE, volumes={"/vol": VOL}, secrets=SECRETS, timeout=4 * 60 * 60)
def packed_random_mfu_probe(
    out_dir: str = "/vol/checkpoints/vlm_cauldron_packed_random_mfu_probe",
    num_iterations: int = 10,
    batch_size: int = DEFAULT_PACKED_PROBE_BATCH_SIZE,
    grad_accum_steps: int = 1,
    max_batch_tokens: int = DEFAULT_PACKED_PROBE_MAX_BATCH_TOKENS,
    max_seq_len: int = DEFAULT_PACKED_PROBE_MAX_SEQ_LEN,
    run: str = "dummy",
    model_step: int = 650,
    hf_repo: str = DEFAULT_HF_REPO,
    hf_config: str = DEFAULT_PROBE_HF_CONFIG,
    stream_buffer_size: int = 256,
    batch_buffer_size: int = 4096,
    prefetch_batches: int = 8,
    prefetch_workers: int = 4,
    compile_model: bool = False,
    fp8: bool = False,
    fp8_recipe: str = "tensorwise",
    pack_examples: int = DEFAULT_PACKED_PROBE_EXAMPLES,
    pack_max_seq_len: int = DEFAULT_PACKED_PROBE_MAX_SEQ_LEN,
    profile_timing: bool = False,
    require_fa3_varlen: bool = True,
    flatten_packed_batch: bool = True,
    loss_chunk_size: int = 0,
):
    _run(build_packed_random_mfu_probe_cmd(
        out_dir=out_dir,
        num_iterations=num_iterations,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        max_batch_tokens=max_batch_tokens,
        max_seq_len=max_seq_len,
        run=run,
        model_step=model_step,
        hf_repo=hf_repo,
        hf_config=hf_config,
        stream_buffer_size=stream_buffer_size,
        batch_buffer_size=batch_buffer_size,
        prefetch_batches=prefetch_batches,
        prefetch_workers=prefetch_workers,
        compile_model=compile_model,
        fp8=fp8,
        fp8_recipe=fp8_recipe,
        pack_examples=pack_examples,
        pack_max_seq_len=pack_max_seq_len,
        profile_timing=profile_timing,
        require_fa3_varlen=require_fa3_varlen,
        flatten_packed_batch=flatten_packed_batch,
        loss_chunk_size=loss_chunk_size,
    ))


@app.function(gpu=GPU_TYPE, volumes={"/vol": VOL}, secrets=SECRETS, timeout=4 * 60 * 60)
def packed_large_mfu_probe(
    out_dir: str = "/vol/checkpoints/vlm_cauldron_packed_large_mfu_probe",
    num_iterations: int = 10,
    batch_size: int = DEFAULT_PACKED_LARGE_PROBE_BATCH_SIZE,
    grad_accum_steps: int = 1,
    max_batch_tokens: int = DEFAULT_PACKED_LARGE_PROBE_MAX_BATCH_TOKENS,
    max_seq_len: int = DEFAULT_PACKED_PROBE_MAX_SEQ_LEN,
    run: str = "dummy",
    model_step: int = 650,
    hf_repo: str = DEFAULT_HF_REPO,
    hf_config: str = DEFAULT_PROBE_HF_CONFIG,
    stream_buffer_size: int = 256,
    batch_buffer_size: int = DEFAULT_PACKED_LARGE_PROBE_BATCH_BUFFER_SIZE,
    bucket_selection: str = "max-tokens",
    prefetch_batches: int = 8,
    prefetch_workers: int = 4,
    compile_model: bool = False,
    fp8: bool = False,
    fp8_recipe: str = "tensorwise",
    pack_examples: int = DEFAULT_PACKED_LARGE_PROBE_EXAMPLES,
    pack_max_seq_len: int = DEFAULT_PACKED_PROBE_MAX_SEQ_LEN,
    profile_timing: bool = False,
    require_fa3_varlen: bool = True,
    flatten_packed_batch: bool = True,
    loss_chunk_size: int = 0,
):
    _run(build_packed_large_mfu_probe_cmd(
        out_dir=out_dir,
        num_iterations=num_iterations,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        max_batch_tokens=max_batch_tokens,
        max_seq_len=max_seq_len,
        run=run,
        model_step=model_step,
        hf_repo=hf_repo,
        hf_config=hf_config,
        stream_buffer_size=stream_buffer_size,
        batch_buffer_size=batch_buffer_size,
        bucket_selection=bucket_selection,
        prefetch_batches=prefetch_batches,
        prefetch_workers=prefetch_workers,
        compile_model=compile_model,
        fp8=fp8,
        fp8_recipe=fp8_recipe,
        pack_examples=pack_examples,
        pack_max_seq_len=pack_max_seq_len,
        profile_timing=profile_timing,
        require_fa3_varlen=require_fa3_varlen,
        flatten_packed_batch=flatten_packed_batch,
        loss_chunk_size=loss_chunk_size,
    ))


@app.function(gpu=GPU_TYPE, volumes={"/vol": VOL}, secrets=SECRETS, timeout=4 * 60 * 60)
def leaky_packed_large_mfu_probe(
    out_dir: str = "/vol/checkpoints/vlm_cauldron_leaky_packed_large_mfu_probe",
    num_iterations: int = 10,
    batch_size: int = DEFAULT_PACKED_LARGE_PROBE_BATCH_SIZE,
    grad_accum_steps: int = 1,
    max_batch_tokens: int = DEFAULT_PACKED_LARGE_PROBE_MAX_BATCH_TOKENS,
    max_seq_len: int = DEFAULT_PACKED_PROBE_MAX_SEQ_LEN,
    run: str = "dummy",
    model_step: int = 650,
    hf_repo: str = DEFAULT_HF_REPO,
    hf_config: str = DEFAULT_PROBE_HF_CONFIG,
    stream_buffer_size: int = 256,
    batch_buffer_size: int = DEFAULT_PACKED_LARGE_PROBE_BATCH_BUFFER_SIZE,
    bucket_selection: str = "max-tokens",
    prefetch_batches: int = 8,
    prefetch_workers: int = 4,
    compile_model: bool = False,
    fp8: bool = False,
    fp8_recipe: str = "tensorwise",
    pack_examples: int = DEFAULT_PACKED_LARGE_PROBE_EXAMPLES,
    pack_max_seq_len: int = DEFAULT_PACKED_PROBE_MAX_SEQ_LEN,
    pad_to_bucket_lens: str = "",
    profile_timing: bool = False,
    loss_chunk_size: int = 0,
):
    _run(build_leaky_packed_large_mfu_probe_cmd(
        out_dir=out_dir,
        num_iterations=num_iterations,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        max_batch_tokens=max_batch_tokens,
        max_seq_len=max_seq_len,
        run=run,
        model_step=model_step,
        hf_repo=hf_repo,
        hf_config=hf_config,
        stream_buffer_size=stream_buffer_size,
        batch_buffer_size=batch_buffer_size,
        bucket_selection=bucket_selection,
        prefetch_batches=prefetch_batches,
        prefetch_workers=prefetch_workers,
        compile_model=compile_model,
        fp8=fp8,
        fp8_recipe=fp8_recipe,
        pack_examples=pack_examples,
        pack_max_seq_len=pack_max_seq_len,
        pad_to_bucket_lens=pad_to_bucket_lens,
        profile_timing=profile_timing,
        loss_chunk_size=loss_chunk_size,
    ))


@app.function(gpu=GPU_TYPE, volumes={"/vol": VOL}, secrets=SECRETS, timeout=4 * 60 * 60)
def packed_large_random_mfu_probe(
    out_dir: str = "/vol/checkpoints/vlm_cauldron_packed_large_random_mfu_probe",
    num_iterations: int = 10,
    batch_size: int = DEFAULT_PACKED_LARGE_PROBE_BATCH_SIZE,
    grad_accum_steps: int = 1,
    max_batch_tokens: int = DEFAULT_PACKED_LARGE_PROBE_MAX_BATCH_TOKENS,
    max_seq_len: int = DEFAULT_PACKED_PROBE_MAX_SEQ_LEN,
    run: str = "dummy",
    model_step: int = 650,
    hf_repo: str = DEFAULT_HF_REPO,
    hf_config: str = DEFAULT_PROBE_HF_CONFIG,
    stream_buffer_size: int = 256,
    batch_buffer_size: int = DEFAULT_PACKED_LARGE_PROBE_BATCH_BUFFER_SIZE,
    prefetch_batches: int = 8,
    prefetch_workers: int = 4,
    compile_model: bool = False,
    fp8: bool = False,
    fp8_recipe: str = "tensorwise",
    pack_examples: int = DEFAULT_PACKED_LARGE_PROBE_EXAMPLES,
    pack_max_seq_len: int = DEFAULT_PACKED_PROBE_MAX_SEQ_LEN,
    profile_timing: bool = False,
    require_fa3_varlen: bool = True,
    flatten_packed_batch: bool = True,
    loss_chunk_size: int = 0,
):
    _run(build_packed_large_random_mfu_probe_cmd(
        out_dir=out_dir,
        num_iterations=num_iterations,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        max_batch_tokens=max_batch_tokens,
        max_seq_len=max_seq_len,
        run=run,
        model_step=model_step,
        hf_repo=hf_repo,
        hf_config=hf_config,
        stream_buffer_size=stream_buffer_size,
        batch_buffer_size=batch_buffer_size,
        prefetch_batches=prefetch_batches,
        prefetch_workers=prefetch_workers,
        compile_model=compile_model,
        fp8=fp8,
        fp8_recipe=fp8_recipe,
        pack_examples=pack_examples,
        pack_max_seq_len=pack_max_seq_len,
        profile_timing=profile_timing,
        require_fa3_varlen=require_fa3_varlen,
        flatten_packed_batch=flatten_packed_batch,
        loss_chunk_size=loss_chunk_size,
    ))


@app.function(gpu=GPU_TYPE, volumes={"/vol": VOL}, secrets=SECRETS, timeout=4 * 60 * 60)
def packed_large_compute_mfu_probe(
    out_dir: str = "/vol/checkpoints/vlm_cauldron_packed_large_compute_mfu_probe",
    num_iterations: int = 10,
    batch_size: int = DEFAULT_PACKED_LARGE_PROBE_BATCH_SIZE,
    grad_accum_steps: int = 1,
    max_batch_tokens: int = DEFAULT_PACKED_LARGE_PROBE_MAX_BATCH_TOKENS,
    max_seq_len: int = DEFAULT_PACKED_PROBE_MAX_SEQ_LEN,
    run: str = "dummy",
    model_step: int = 650,
    hf_repo: str = DEFAULT_HF_REPO,
    hf_config: str = DEFAULT_PROBE_HF_CONFIG,
    stream_buffer_size: int = 256,
    batch_buffer_size: int = DEFAULT_PACKED_LARGE_PROBE_BATCH_BUFFER_SIZE,
    prefetch_batches: int = 8,
    prefetch_workers: int = 4,
    compile_model: bool = False,
    fp8: bool = False,
    fp8_recipe: str = "tensorwise",
    pack_examples: int = DEFAULT_PACKED_LARGE_PROBE_EXAMPLES,
    pack_max_seq_len: int = DEFAULT_PACKED_PROBE_MAX_SEQ_LEN,
    profile_timing: bool = False,
    require_fa3_varlen: bool = True,
    flatten_packed_batch: bool = True,
    loss_chunk_size: int = 0,
):
    _run(build_packed_large_compute_mfu_probe_cmd(
        out_dir=out_dir,
        num_iterations=num_iterations,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        max_batch_tokens=max_batch_tokens,
        max_seq_len=max_seq_len,
        run=run,
        model_step=model_step,
        hf_repo=hf_repo,
        hf_config=hf_config,
        stream_buffer_size=stream_buffer_size,
        batch_buffer_size=batch_buffer_size,
        prefetch_batches=prefetch_batches,
        prefetch_workers=prefetch_workers,
        compile_model=compile_model,
        fp8=fp8,
        fp8_recipe=fp8_recipe,
        pack_examples=pack_examples,
        pack_max_seq_len=pack_max_seq_len,
        profile_timing=profile_timing,
        require_fa3_varlen=require_fa3_varlen,
        flatten_packed_batch=flatten_packed_batch,
        loss_chunk_size=loss_chunk_size,
    ))


@app.function(gpu=GPU_TYPE, volumes={"/vol": VOL}, secrets=SECRETS, timeout=4 * 60 * 60)
def packed_profile_mfu_probe(
    out_dir: str = "/vol/checkpoints/vlm_cauldron_packed_profile_mfu_probe",
    num_iterations: int = 10,
    batch_size: int = DEFAULT_PACKED_PROBE_BATCH_SIZE,
    grad_accum_steps: int = 1,
    max_batch_tokens: int = DEFAULT_PACKED_PROBE_MAX_BATCH_TOKENS,
    max_seq_len: int = DEFAULT_PACKED_PROBE_MAX_SEQ_LEN,
    run: str = "dummy",
    model_step: int = 650,
    hf_repo: str = DEFAULT_HF_REPO,
    hf_config: str = DEFAULT_PROBE_HF_CONFIG,
    stream_buffer_size: int = 256,
    batch_buffer_size: int = 4096,
    bucket_selection: str = "max-tokens",
    prefetch_batches: int = 8,
    prefetch_workers: int = 4,
    compile_model: bool = False,
    fp8: bool = False,
    fp8_recipe: str = "tensorwise",
    pack_examples: int = DEFAULT_PACKED_PROBE_EXAMPLES,
    pack_max_seq_len: int = DEFAULT_PACKED_PROBE_MAX_SEQ_LEN,
    require_fa3_varlen: bool = True,
    flatten_packed_batch: bool = True,
    loss_chunk_size: int = 0,
):
    _run(build_packed_profile_mfu_probe_cmd(
        out_dir=out_dir,
        num_iterations=num_iterations,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        max_batch_tokens=max_batch_tokens,
        max_seq_len=max_seq_len,
        run=run,
        model_step=model_step,
        hf_repo=hf_repo,
        hf_config=hf_config,
        stream_buffer_size=stream_buffer_size,
        batch_buffer_size=batch_buffer_size,
        bucket_selection=bucket_selection,
        prefetch_batches=prefetch_batches,
        prefetch_workers=prefetch_workers,
        compile_model=compile_model,
        fp8=fp8,
        fp8_recipe=fp8_recipe,
        pack_examples=pack_examples,
        pack_max_seq_len=pack_max_seq_len,
        require_fa3_varlen=require_fa3_varlen,
        flatten_packed_batch=flatten_packed_batch,
        loss_chunk_size=loss_chunk_size,
    ))


@app.function(gpu=GPU_TYPE, volumes={"/vol": VOL}, secrets=SECRETS, timeout=4 * 60 * 60)
def packed_large_profile_mfu_probe(
    out_dir: str = "/vol/checkpoints/vlm_cauldron_packed_large_profile_mfu_probe",
    num_iterations: int = 10,
    batch_size: int = DEFAULT_PACKED_LARGE_PROBE_BATCH_SIZE,
    grad_accum_steps: int = 1,
    max_batch_tokens: int = DEFAULT_PACKED_LARGE_PROBE_MAX_BATCH_TOKENS,
    max_seq_len: int = DEFAULT_PACKED_PROBE_MAX_SEQ_LEN,
    run: str = "dummy",
    model_step: int = 650,
    hf_repo: str = DEFAULT_HF_REPO,
    hf_config: str = DEFAULT_PROBE_HF_CONFIG,
    stream_buffer_size: int = 256,
    batch_buffer_size: int = DEFAULT_PACKED_LARGE_PROBE_BATCH_BUFFER_SIZE,
    bucket_selection: str = "max-tokens",
    prefetch_batches: int = 8,
    prefetch_workers: int = 4,
    compile_model: bool = False,
    fp8: bool = False,
    fp8_recipe: str = "tensorwise",
    pack_examples: int = DEFAULT_PACKED_LARGE_PROBE_EXAMPLES,
    pack_max_seq_len: int = DEFAULT_PACKED_PROBE_MAX_SEQ_LEN,
    require_fa3_varlen: bool = True,
    flatten_packed_batch: bool = True,
    loss_chunk_size: int = 0,
):
    _run(build_packed_large_profile_mfu_probe_cmd(
        out_dir=out_dir,
        num_iterations=num_iterations,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        max_batch_tokens=max_batch_tokens,
        max_seq_len=max_seq_len,
        run=run,
        model_step=model_step,
        hf_repo=hf_repo,
        hf_config=hf_config,
        stream_buffer_size=stream_buffer_size,
        batch_buffer_size=batch_buffer_size,
        bucket_selection=bucket_selection,
        prefetch_batches=prefetch_batches,
        prefetch_workers=prefetch_workers,
        compile_model=compile_model,
        fp8=fp8,
        fp8_recipe=fp8_recipe,
        pack_examples=pack_examples,
        pack_max_seq_len=pack_max_seq_len,
        require_fa3_varlen=require_fa3_varlen,
        flatten_packed_batch=flatten_packed_batch,
        loss_chunk_size=loss_chunk_size,
    ))


@app.function(gpu=GPU_TYPE, volumes={"/vol": VOL}, secrets=SECRETS, timeout=4 * 60 * 60)
def packed_large_random_profile_mfu_probe(
    out_dir: str = "/vol/checkpoints/vlm_cauldron_packed_large_random_profile_mfu_probe",
    num_iterations: int = 10,
    batch_size: int = DEFAULT_PACKED_LARGE_PROBE_BATCH_SIZE,
    grad_accum_steps: int = 1,
    max_batch_tokens: int = DEFAULT_PACKED_LARGE_PROBE_MAX_BATCH_TOKENS,
    max_seq_len: int = DEFAULT_PACKED_PROBE_MAX_SEQ_LEN,
    run: str = "dummy",
    model_step: int = 650,
    hf_repo: str = DEFAULT_HF_REPO,
    hf_config: str = DEFAULT_PROBE_HF_CONFIG,
    stream_buffer_size: int = 256,
    batch_buffer_size: int = DEFAULT_PACKED_LARGE_PROBE_BATCH_BUFFER_SIZE,
    prefetch_batches: int = 8,
    prefetch_workers: int = 4,
    compile_model: bool = False,
    fp8: bool = False,
    fp8_recipe: str = "tensorwise",
    pack_examples: int = DEFAULT_PACKED_LARGE_PROBE_EXAMPLES,
    pack_max_seq_len: int = DEFAULT_PACKED_PROBE_MAX_SEQ_LEN,
    require_fa3_varlen: bool = True,
    flatten_packed_batch: bool = True,
    loss_chunk_size: int = 0,
):
    _run(build_packed_large_random_profile_mfu_probe_cmd(
        out_dir=out_dir,
        num_iterations=num_iterations,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        max_batch_tokens=max_batch_tokens,
        max_seq_len=max_seq_len,
        run=run,
        model_step=model_step,
        hf_repo=hf_repo,
        hf_config=hf_config,
        stream_buffer_size=stream_buffer_size,
        batch_buffer_size=batch_buffer_size,
        prefetch_batches=prefetch_batches,
        prefetch_workers=prefetch_workers,
        compile_model=compile_model,
        fp8=fp8,
        fp8_recipe=fp8_recipe,
        pack_examples=pack_examples,
        pack_max_seq_len=pack_max_seq_len,
        require_fa3_varlen=require_fa3_varlen,
        flatten_packed_batch=flatten_packed_batch,
        loss_chunk_size=loss_chunk_size,
    ))


@app.function(gpu=GPU_TYPE, volumes={"/vol": VOL}, secrets=SECRETS, timeout=4 * 60 * 60)
def packed_large_compute_profile_mfu_probe(
    out_dir: str = "/vol/checkpoints/vlm_cauldron_packed_large_compute_profile_mfu_probe",
    num_iterations: int = 10,
    batch_size: int = DEFAULT_PACKED_LARGE_PROBE_BATCH_SIZE,
    grad_accum_steps: int = 1,
    max_batch_tokens: int = DEFAULT_PACKED_LARGE_PROBE_MAX_BATCH_TOKENS,
    max_seq_len: int = DEFAULT_PACKED_PROBE_MAX_SEQ_LEN,
    run: str = "dummy",
    model_step: int = 650,
    hf_repo: str = DEFAULT_HF_REPO,
    hf_config: str = DEFAULT_PROBE_HF_CONFIG,
    stream_buffer_size: int = 256,
    batch_buffer_size: int = DEFAULT_PACKED_LARGE_PROBE_BATCH_BUFFER_SIZE,
    prefetch_batches: int = 8,
    prefetch_workers: int = 4,
    compile_model: bool = False,
    fp8: bool = False,
    fp8_recipe: str = "tensorwise",
    pack_examples: int = DEFAULT_PACKED_LARGE_PROBE_EXAMPLES,
    pack_max_seq_len: int = DEFAULT_PACKED_PROBE_MAX_SEQ_LEN,
    require_fa3_varlen: bool = True,
    flatten_packed_batch: bool = True,
    loss_chunk_size: int = 0,
):
    _run(build_packed_large_compute_profile_mfu_probe_cmd(
        out_dir=out_dir,
        num_iterations=num_iterations,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        max_batch_tokens=max_batch_tokens,
        max_seq_len=max_seq_len,
        run=run,
        model_step=model_step,
        hf_repo=hf_repo,
        hf_config=hf_config,
        stream_buffer_size=stream_buffer_size,
        batch_buffer_size=batch_buffer_size,
        prefetch_batches=prefetch_batches,
        prefetch_workers=prefetch_workers,
        compile_model=compile_model,
        fp8=fp8,
        fp8_recipe=fp8_recipe,
        pack_examples=pack_examples,
        pack_max_seq_len=pack_max_seq_len,
        require_fa3_varlen=require_fa3_varlen,
        flatten_packed_batch=flatten_packed_batch,
        loss_chunk_size=loss_chunk_size,
    ))


@app.function(gpu=GPU_TYPE, volumes={"/vol": VOL}, secrets=SECRETS, timeout=24 * 60 * 60)
def eval(
    checkpoint_dir: str = "/vol/checkpoints/vlm",
    checkpoint_step: int = 1000,
    out: str = "/vol/bench/vlm_eval.json",
    benchmarks: str = "mmstar,scienceqa,chartqa,mmmu,textvqa",
    mmmu_configs: str = "Accounting",
    limit: int = 32,
    max_scan: int = 0,
    print_samples: int = 0,
    model_step: int = 650,
):
    _run(build_eval_cmd(checkpoint_dir, checkpoint_step, out, benchmarks, mmmu_configs, limit, max_scan, print_samples, model_step))
