"""
Minimal Modal entrypoints for nanochat-llava.

Default GPU is A100-80GB. Switch with:
NANOCHAT_MODAL_GPU=H100 modal run modal_vlm.py::train
"""

from __future__ import annotations

import os
import subprocess

import modal


APP_NAME = "nanochat-llava-v0"
GPU_TYPE = os.environ.get("NANOCHAT_MODAL_GPU", "A100-80GB")
VOL = modal.Volume.from_name("nanochat-llava-v0", create_if_missing=True)
VOLUME_DIRS = ["/vol/checkpoints", "/vol/logs", "/vol/nanochat", "/vol/hf"]


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.9.1",
        "datasets>=4.0.0",
        "transformers>=4.57.3,<5",
        "tokenizers>=0.22.0",
        "tiktoken>=0.11.0",
        "rustbpe>=0.1.0",
        "huggingface_hub>=0.36.0",
        "safetensors>=0.6.0",
        "Pillow>=11.0.0",
        "wandb>=0.19.0",
        "numpy>=2.0.0",
        "tqdm>=4.67.0",
        "pytest>=8.0.0",
        "kernels>=0.9.0",
    )
    .env({
        "NANOCHAT_BASE_DIR": "/vol/nanochat",
        "HF_HOME": "/vol/hf",
        "NANOCHAT_SIGLIP_CACHE_DIR": "/vol/hf/siglip",
    })
    .add_local_dir("nanochat", "/nanochat", copy=True)
    .add_local_dir("scripts", "/scripts", copy=True)
    .add_local_dir("tasks", "/tasks", copy=True)
    .add_local_dir("tests", "/tests", copy=True)
    .add_local_file("pyproject.toml", "/pyproject.toml", copy=True)
)


app = modal.App(APP_NAME, image=image)


def build_train_cmd(
    out_dir: str = "/vol/checkpoints/vlm",
    num_iterations: int = 1000,
    batch_size: int = 32,
    max_seq_len: int = 512,
    max_batch_images: int = 96,
    run: str = "dummy",
    model_step: int = 650,
    hf_repo: str = "HuggingFaceM4/FineVisionMax",
    no_save: bool = False,
    require_fa3_varlen: bool = False,
    log_every: int = 10,
    num_workers: int = 4,
    eval_every: int = 200,
):
    cmd = [
        "python",
        "-m",
        "scripts.vlm_train",
        "--run",
        run,
        "--hf-repo",
        hf_repo,
        "--out-dir",
        out_dir,
        "--device-type",
        "cuda",
        "--num-iterations",
        str(num_iterations),
        "--device-batch-size",
        str(batch_size),
        "--max-seq-len",
        str(max_seq_len),
        "--max-batch-images",
        str(max_batch_images),
        "--model-step",
        str(model_step),
        "--log-every",
        str(log_every),
        "--num-workers",
        str(num_workers),
        "--eval-every",
        str(eval_every),
        "--skip-bad-images",
    ]
    if no_save:
        cmd += ["--no-save"]
    else:
        cmd += ["--save-every", str(num_iterations)]
    if require_fa3_varlen:
        cmd += ["--require-fa3-varlen"]
    return cmd


def build_eval_cmd(
    checkpoint_dir: str = "/vol/checkpoints/vlm",
    checkpoint_step: int = 1000,
    out: str = "/vol/checkpoints/vlm_eval.json",
    benchmarks: str = "mmstar,scienceqa,chartqa,mmmu,textvqa",
    limit: int = 24,
    max_scan: int = 240,
):
    return [
        "python",
        "-m",
        "scripts.vlm_eval",
        "--checkpoint-dir",
        checkpoint_dir,
        "--checkpoint-step",
        str(checkpoint_step),
        "--out",
        out,
        "--benchmarks",
        benchmarks,
        "--limit",
        str(limit),
        "--max-scan",
        str(max_scan),
    ]


def _run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd="/", check=True)


@app.function(gpu=GPU_TYPE, volumes={"/vol": VOL}, timeout=60 * 30)
def doctor():
    print("GPU_TYPE", GPU_TYPE, flush=True)
    _run(build_train_cmd(num_iterations=1, batch_size=1, eval_every=-1, no_save=True))


@app.function(gpu=GPU_TYPE, volumes={"/vol": VOL}, timeout=60 * 20)
def smoke():
    _run(["python", "-m", "pytest", "tests/test_vlm_smoke.py", "-q"])


@app.function(gpu=GPU_TYPE, volumes={"/vol": VOL}, timeout=60 * 60 * 24)
def train(
    out_dir: str = "/vol/checkpoints/vlm",
    num_iterations: int = 1000,
    batch_size: int = 32,
    max_seq_len: int = 512,
    max_batch_images: int = 96,
    run: str = "dummy",
    model_step: int = 650,
    hf_repo: str = "HuggingFaceM4/FineVisionMax",
    no_save: bool = False,
    require_fa3_varlen: bool = True,
    log_every: int = 10,
    num_workers: int = 4,
    eval_every: int = 200,
):
    _run(build_train_cmd(
        out_dir=out_dir,
        num_iterations=num_iterations,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        max_batch_images=max_batch_images,
        run=run,
        model_step=model_step,
        hf_repo=hf_repo,
        no_save=no_save,
        require_fa3_varlen=require_fa3_varlen,
        log_every=log_every,
        num_workers=num_workers,
        eval_every=eval_every,
    ))


@app.function(gpu=GPU_TYPE, volumes={"/vol": VOL}, timeout=60 * 60 * 6)
def eval(
    checkpoint_dir: str = "/vol/checkpoints/vlm",
    checkpoint_step: int = 1000,
    out: str = "/vol/checkpoints/vlm_eval.json",
    benchmarks: str = "mmstar,scienceqa,chartqa,mmmu,textvqa",
    limit: int = 24,
    max_scan: int = 240,
):
    _run(build_eval_cmd(checkpoint_dir, checkpoint_step, out, benchmarks, limit, max_scan))
