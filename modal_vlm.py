"""
Minimal Modal entrypoints for nanochat-llava v0.

Default GPU is A100-80GB. Switch to H100 with:
NANOCHAT_MODAL_GPU=H100 modal run modal_vlm.py::stage1
"""

from __future__ import annotations

import os
import shlex
import subprocess

import modal


APP_NAME = "nanochat-llava-v0"
GPU_TYPE = os.environ.get("NANOCHAT_MODAL_GPU", "A100-80GB")
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


def build_stage1_cmd(
    out_dir: str = "/vol/checkpoints/stage1",
    num_iterations: int = 1000,
    batch_size: int = 32,
    max_batch_tokens: int = 0,
    max_examples: int = -1,
    run: str = "dummy",
    model_step: int = 650,
    image_root: str = "/vol/datasets/llava/pretrain_images",
):
    cmd = [
        "python",
        "-m",
        "scripts.vlm_train",
        "--run",
        run,
        "--stage",
        "1",
        "--hf-repo",
        "liuhaotian/LLaVA-Pretrain",
        "--hf-file",
        "blip_laion_cc_sbu_558k.json",
        "--hf-image-zip",
        "images.zip",
        "--image-root",
        image_root,
        "--out-dir",
        out_dir,
        "--device-type",
        "cuda",
        "--num-iterations",
        str(num_iterations),
        "--device-batch-size",
        str(batch_size),
        "--max-seq-len",
        "2048",
        "--save-every",
        str(num_iterations),
        "--model-step",
        str(model_step),
        "--skip-bad-images",
    ]
    if max_examples > 0:
        cmd += ["--max-examples", str(max_examples)]
    if max_batch_tokens > 0:
        cmd += ["--max-batch-tokens", str(max_batch_tokens)]
    return cmd


def build_stage2_cmd(
    init_checkpoint_dir: str = "",
    init_checkpoint_step: int = 0,
    out_dir: str = "/vol/checkpoints/stage2",
    num_iterations: int = 1000,
    batch_size: int = 24,
    max_batch_tokens: int = 12000,
    max_examples: int = -1,
    run: str = "dummy",
    model_step: int = 650,
    profile_timing: bool = False,
    hf_repo: str = "HuggingFaceM4/FineVision",
    hf_config: str = "LLaVA_Instruct_150K",
):
    cmd = [
        "python",
        "-m",
        "scripts.vlm_train",
        "--run",
        run,
        "--stage",
        "2",
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
        "--max-batch-tokens",
        str(max_batch_tokens),
        "--max-seq-len",
        "2048",
        "--save-every",
        str(num_iterations),
        "--model-step",
        str(model_step),
    ]
    if init_checkpoint_dir:
        cmd += ["--init-vlm-checkpoint-dir", init_checkpoint_dir, "--init-vlm-checkpoint-step", str(init_checkpoint_step)]
    if max_examples > 0:
        cmd += ["--max-examples", str(max_examples)]
    if profile_timing:
        cmd += ["--profile-timing"]
    return cmd


def build_eval_cmd(
    checkpoint_dir: str = "/vol/checkpoints/stage2",
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
        "stage1_preview": build_stage1_cmd(num_iterations=1, batch_size=1, max_examples=1),
        "stage2_preview": build_stage2_cmd(num_iterations=1, batch_size=1, max_examples=1),
        "eval_preview": build_eval_cmd(limit=1, max_scan=2, benchmarks="mmstar"),
    }


@app.function(volumes={"/vol": VOL}, timeout=10 * 60)
def doctor():
    summary = build_doctor_summary()
    print("Modal doctor")
    print("APP_NAME", summary["app"])
    print("GPU_TYPE", summary["gpu"])
    print("VOLUME_DIRS", ",".join(summary["volume_dirs"]))
    for key in ["stage1_preview", "stage2_preview", "eval_preview"]:
        print(key, " ".join(shlex.quote(arg) for arg in summary[key]))
    for module in ["scripts.vlm_train", "scripts.vlm_eval"]:
        subprocess.run(["python", "-m", module, "--help"], cwd="/root/nanochat-llava", check=True, stdout=subprocess.DEVNULL)
        print("help_ok", module)
    VOL.commit()


@app.function(gpu=GPU_TYPE, volumes={"/vol": VOL}, secrets=SECRETS, timeout=30 * 60)
def smoke():
    _run(["python", "-m", "pytest", "tests/test_vlm_smoke.py", "-q"])


@app.function(gpu=GPU_TYPE, volumes={"/vol": VOL}, secrets=SECRETS, timeout=24 * 60 * 60)
def stage1(
    out_dir: str = "/vol/checkpoints/stage1",
    num_iterations: int = 1000,
    batch_size: int = 32,
    max_batch_tokens: int = 0,
    max_examples: int = -1,
    run: str = "dummy",
    model_step: int = 650,
):
    _run(build_stage1_cmd(out_dir, num_iterations, batch_size, max_batch_tokens, max_examples, run, model_step))


@app.function(gpu=GPU_TYPE, volumes={"/vol": VOL}, secrets=SECRETS, timeout=24 * 60 * 60)
def stage2(
    init_checkpoint_dir: str = "",
    init_checkpoint_step: int = 0,
    out_dir: str = "/vol/checkpoints/stage2",
    num_iterations: int = 1000,
    batch_size: int = 24,
    max_batch_tokens: int = 12000,
    max_examples: int = -1,
    run: str = "dummy",
    model_step: int = 650,
    profile_timing: bool = False,
    hf_repo: str = "HuggingFaceM4/FineVision",
    hf_config: str = "LLaVA_Instruct_150K",
):
    _run(build_stage2_cmd(
        init_checkpoint_dir,
        init_checkpoint_step,
        out_dir,
        num_iterations,
        batch_size,
        max_batch_tokens,
        max_examples,
        run,
        model_step,
        profile_timing,
        hf_repo,
        hf_config,
    ))


@app.function(gpu=GPU_TYPE, volumes={"/vol": VOL}, secrets=SECRETS, timeout=24 * 60 * 60)
def eval(
    checkpoint_dir: str = "/vol/checkpoints/stage2",
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
