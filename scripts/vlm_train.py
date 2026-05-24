"""
Minimal two-stage nanochat-llava trainer.

Stage 1: freeze nanochat + SigLIP, train only the projector on image captions.
Stage 2: freeze SigLIP, train projector + nanochat on visual instructions.

The code intentionally stays close to nanoVLM/LLaVA: images are encoded on the
fly, the vision encoder is frozen, and the projector uses a higher LR than the
language model.
"""

import argparse
import io
import json
import math
import os
import random
import time
import zipfile
from pathlib import Path

import torch
import wandb

from nanochat.checkpoint_manager import load_model
from nanochat.common import DummyWandb, autodetect_device_type, compute_cleanup, compute_init, get_base_dir, get_peak_flops, print0
from nanochat.tokenizer import get_token_bytes
from nanochat.vision import (
    IMAGE_MARKER,
    IMAGE_TOKEN_ID,
    SIGLIP_MODEL_ID,
    VISION_GRID,
    VISION_TOKENS,
    SigLIPPooledFeatureExtractor,
    VisionProjector,
    build_multimodal_batch,
    count_image_tokens,
    ensure_hf_nanochat_checkpoint,
    load_vlm_checkpoint,
    render_caption_example,
    render_vision_conversation,
    save_vlm_checkpoint,
)
from scripts.vlm_eval import evaluate_vlm


IMAGE_KEYS = ("image", "image_path", "filename", "path")
DEFAULT_STAGE_DATA = {
    1: ("liuhaotian/LLaVA-Pretrain", "blip_laion_cc_sbu_558k.json"),
    2: ("HuggingFaceM4/FineVision", None),
}
DEFAULT_STAGE_CONFIG = {2: "LLaVA_Instruct_150K"}
MODEL_SOURCE = "sft"
UNTRUNCATED_MAX_TOKENS = 1_000_000_000
INIT_LR_FRAC = 0.05
WARMDOWN_RATIO = 0.5
_ZIP_CACHE = {}
PROFILE_KEYS = (
    "data",
    "image_open",
    "image_processor",
    "image_transfer",
    "siglip_forward",
    "siglip_pool",
    "image_siglip",
    "batch",
    "fwdbwd",
    "optim",
)
IMAGE_PROFILE_KEYS = ("image_open", "image_processor", "image_transfer", "siglip_forward", "siglip_pool")


def add_profile(profile, key, elapsed):
    if profile is not None:
        profile[key] += elapsed


def _first_assistant_text(example):
    conv = example.get("conversations") or example.get("messages") or []
    for msg in conv:
        role = msg.get("from", msg.get("role"))
        if role in {"gpt", "assistant"}:
            return msg.get("value", msg.get("content", ""))
    return example.get("caption", example.get("answer", ""))


def _ensure_image_marker_in_conversation(example):
    example = dict(example)
    conv = example.get("conversations") or example.get("messages")
    if conv is None and example.get("texts") is not None:
        conv = []
        for text in example["texts"]:
            user = text.get("user", "")
            assistant = text.get("assistant", "")
            if user or assistant:
                conv.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
        if conv:
            example["messages"] = conv
        else:
            conv = None
    if conv is None:
        question = example.get("question", "Describe the image.")
        answer = example.get("answer", example.get("caption", ""))
        example["conversations"] = [
            {"from": "human", "value": f"{IMAGE_MARKER}\n{question}"},
            {"from": "gpt", "value": answer},
        ]
        return example

    conv = [dict(msg) for msg in conv]
    key_role = "from" if "from" in conv[0] else "role"
    key_text = "value" if "value" in conv[0] else "content"
    for msg in conv:
        if msg[key_role] in {"human", "user"}:
            if IMAGE_MARKER not in msg[key_text]:
                msg[key_text] = f"{IMAGE_MARKER}\n{msg[key_text]}"
            break
    if "conversations" in example:
        example["conversations"] = conv
    else:
        example["messages"] = conv
    return example


def _load_json(path: Path):
    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _iter_hf_json_records(hf_repo: str, hf_file: str):
    from datasets import load_dataset

    data_files = f"hf://datasets/{hf_repo}/{hf_file}"
    yield from load_dataset("json", data_files=data_files, split="train", streaming=True)


def _iter_hf_dataset_records(hf_repo: str, hf_config: str):
    from datasets import load_dataset

    ds = load_dataset(hf_repo, hf_config, split="train", streaming=True)
    try:
        from datasets import Image as HFImage
        from datasets import Sequence

        if "images" in (getattr(ds, "features", None) or {}):
            ds = ds.cast_column("images", Sequence(HFImage(decode=False)))
    except ImportError:
        pass
    yield from ds


def _stream_record_limit(args):
    if args.max_examples > 0:
        return args.max_examples
    return max(args.device_batch_size * args.grad_accum_steps * (args.num_iterations + 1) * 2, args.device_batch_size * 64)


def load_records(args):
    if args.data_json:
        path = Path(args.data_json)
        records = _load_json(path)
        assert isinstance(records, list), f"expected a JSON list in {path}"
        if args.max_examples > 0:
            records = records[: args.max_examples]
        return records, str(path)

    hf_repo, hf_file, hf_config = args.hf_repo, args.hf_file, getattr(args, "hf_config", None)
    if not hf_repo and not hf_file and not hf_config:
        hf_repo, hf_file = DEFAULT_STAGE_DATA[args.stage]
        hf_config = DEFAULT_STAGE_CONFIG.get(args.stage)
    assert hf_repo and (hf_file or hf_config), "provide --data-json, --hf-repo plus --hf-file, or --hf-repo plus --hf-config"

    limit = _stream_record_limit(args)
    records = []
    source = f"stream:{hf_repo}/{hf_file or hf_config}"
    iterator = _iter_hf_dataset_records(hf_repo, hf_config) if hf_config else _iter_hf_json_records(hf_repo, hf_file)
    for rec in iterator:
        records.append(rec)
        if len(records) >= limit:
            break
    assert records, f"streamed no records from {hf_repo}/{hf_file or hf_config}"
    return records, f"{source} first {len(records):,} rows"


def maybe_use_hf_image_zip(args):
    if not args.hf_image_zip:
        return
    hf_repo = args.hf_repo or DEFAULT_STAGE_DATA[args.stage][0]
    from huggingface_hub import hf_hub_download

    zip_path = hf_hub_download(repo_id=hf_repo, filename=args.hf_image_zip, repo_type="dataset")
    if not args.image_zip:
        args.image_zip = zip_path
        print0(f"Using image zip {args.hf_image_zip} directly from {zip_path}")


def _image_value(record):
    images = record.get("images")
    if images:
        if isinstance(images, list) and len(images) == 1:
            return images[0]
        return None
    for key in IMAGE_KEYS:
        value = record.get(key)
        if value is not None:
            return value
    return None


def _unique_setdefault(mapping, key, value):
    existing = mapping.get(key)
    if existing is None and key in mapping:
        return
    if existing is not None and existing != value:
        mapping[key] = None
    else:
        mapping[key] = value


def get_zip_file(image_zip):
    zip_path = str(Path(image_zip).expanduser())
    entry = _ZIP_CACHE.get(zip_path)
    if entry is not None:
        return entry

    zf = zipfile.ZipFile(zip_path)
    names = [info.filename for info in zf.infolist() if not info.is_dir()]
    suffix_map = {}
    basename_map = {}
    for name in names:
        parts = name.split("/")
        if not parts:
            continue
        _unique_setdefault(basename_map, parts[-1], name)
        for i in range(1, len(parts)):
            _unique_setdefault(suffix_map, "/".join(parts[i:]), name)
    entry = {"zip": zf, "names": set(names), "suffix_map": suffix_map, "basename_map": basename_map}
    _ZIP_CACHE[zip_path] = entry
    return entry


def _zip_image_candidates(value):
    raw = str(value).replace("\\", "/").lstrip("/")
    while raw.startswith("./"):
        raw = raw[2:]
    basename = raw.rsplit("/", 1)[-1]
    candidates = [raw, f"images/{raw}", basename, f"images/{basename}"]
    seen = set()
    return [candidate for candidate in candidates if candidate and not (candidate in seen or seen.add(candidate))]


def resolve_image_in_zip(value, image_zip):
    entry = get_zip_file(image_zip)
    for candidate in _zip_image_candidates(value):
        if candidate in entry["names"]:
            return candidate
        suffix_match = entry["suffix_map"].get(candidate)
        if suffix_match is not None:
            return suffix_match
        basename_match = entry["basename_map"].get(candidate)
        if basename_match is not None:
            return basename_match
    raise FileNotFoundError(f"could not find image {value!r} in zip {image_zip!r}")


def resolve_image_path(value, image_root):
    path = Path(str(value))
    candidates = [path] if path.is_absolute() else []
    if not path.is_absolute() and image_root:
        root = Path(image_root)
        candidates.extend([root / path, root / "images" / path, root / path.name])
    if not candidates:
        candidates.append(path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"could not find image {value!r} under {image_root!r}")


def open_image(record, image_root, image_zip=None, profile=None):
    value = _image_value(record)
    if value is None:
        raise KeyError(f"record has no image field among {IMAGE_KEYS}")
    if hasattr(value, "convert"):
        t = time.perf_counter()
        image = value.convert("RGB")
        add_profile(profile, "image_open", time.perf_counter() - t)
        return image

    from PIL import Image

    if isinstance(value, dict):
        t = time.perf_counter()
        if value.get("bytes") is not None:
            with Image.open(io.BytesIO(value["bytes"])) as img:
                image = img.convert("RGB")
        elif value.get("path"):
            with Image.open(value["path"]) as img:
                image = img.convert("RGB")
        else:
            raise ValueError("HF image dict has neither bytes nor path")
        add_profile(profile, "image_open", time.perf_counter() - t)
        return image

    try:
        path = resolve_image_path(value, image_root)
        t = time.perf_counter()
        with Image.open(path) as img:
            image = img.convert("RGB")
        add_profile(profile, "image_open", time.perf_counter() - t)
        return image
    except FileNotFoundError as local_error:
        if image_zip:
            t = time.perf_counter()
            entry = get_zip_file(image_zip)
            member = resolve_image_in_zip(value, image_zip)
            with entry["zip"].open(member) as f:
                with Image.open(f) as img:
                    image = img.convert("RGB")
            add_profile(profile, "image_open", time.perf_counter() - t)
            return image
        raise local_error


def expanded_input_len(tokens):
    image_count = count_image_tokens(tokens[:-1])
    return len(tokens) - 1 + image_count * (VISION_TOKENS - 1)


def supervised_target_count(tokens, mask, image_token_id=IMAGE_TOKEN_ID):
    count = 0
    for i, tok in enumerate(tokens[:-1]):
        next_tok = int(tokens[i + 1])
        if int(tok) == image_token_id or next_tok == image_token_id:
            continue
        count += int(mask[i + 1]) == 1
    return count


def render_records(records, tokenizer, stage, max_seq_len):
    rendered = []
    for rec in records:
        if _image_value(rec) is None:
            continue
        if stage == 1:
            caption = rec.get("caption") or rec.get("blip_caption") or _first_assistant_text(rec)
            tokens, mask = render_caption_example(tokenizer, caption, max_tokens=UNTRUNCATED_MAX_TOKENS)
        else:
            tokens, mask = render_vision_conversation(tokenizer, _ensure_image_marker_in_conversation(rec), max_tokens=UNTRUNCATED_MAX_TOKENS)
        if count_image_tokens(tokens) != 1 or count_image_tokens(tokens[:-1]) != 1:
            continue
        length = expanded_input_len(tokens)
        if length > max_seq_len:
            continue
        if supervised_target_count(tokens, mask) > 0:
            rendered.append({"tokens": tokens, "mask": mask, "record": rec, "expanded_len": length})
    assert rendered, "no usable image-text examples loaded"
    return rendered


def next_batch(examples, batch_size, cursor, rng, max_batch_tokens=0):
    if cursor == 0:
        rng.shuffle(examples)
    batch = []
    max_len = 0
    while len(batch) < batch_size:
        if cursor >= len(examples):
            cursor = 0
            rng.shuffle(examples)
            if batch:
                break
        candidate = examples[cursor]
        next_max_len = max(max_len, candidate["expanded_len"])
        if batch and max_batch_tokens > 0 and next_max_len * (len(batch) + 1) > max_batch_tokens:
            break
        batch.append(candidate)
        cursor += 1
        max_len = next_max_len
        if cursor >= len(examples):
            cursor = 0
            break
    return batch, cursor


def batch_features_and_examples(extractor, examples, image_root, image_zip=None, skip_bad_images=False, profile=None, synchronize=None):
    images = []
    kept_examples = []
    for i, example in enumerate(examples):
        record = example["record"]
        try:
            images.append(open_image(record, image_root, image_zip=image_zip, profile=profile))
            kept_examples.append(example)
        except Exception as exc:
            if not skip_bad_images:
                raise
            print0(f"skipping image {record.get('image', record.get('id', i))}: {type(exc).__name__}: {exc}")
    if not images:
        return None, kept_examples
    if profile is None:
        return extractor(images), kept_examples
    return extractor(images, profile=profile, synchronize=synchronize), kept_examples


def count_params(parameters):
    return sum(p.numel() for p in parameters)


def get_lr_multiplier(step: int, num_iterations: int, warmup_ratio: float, warmdown_ratio: float, final_lr_frac: float) -> float:
    progress = 0.0 if num_iterations <= 1 else (step - 1) / (num_iterations - 1)
    if warmup_ratio > 0 and progress < warmup_ratio:
        return (progress + 1e-8) / warmup_ratio
    if warmdown_ratio <= 0 or progress <= 1.0 - warmdown_ratio:
        return 1.0
    decay = (progress - (1.0 - warmdown_ratio)) / warmdown_ratio
    return (1.0 - decay) + decay * final_lr_frac


def get_muon_momentum(step: int) -> float:
    frac = min(step / 300, 1.0)
    return (1 - frac) * 0.85 + frac * 0.95


def setup_llm_optimizer(model, args):
    optimizer = model.setup_optimizer(
        unembedding_lr=args.unembedding_lr,
        embedding_lr=args.embedding_lr,
        matrix_lr=args.matrix_lr,
        scalar_lr=args.scalar_lr,
        weight_decay=args.weight_decay,
    )
    for group in optimizer.param_groups:
        group["lr"] = group["lr"] * args.init_lr_frac
        group["initial_lr"] = group["lr"]
    return optimizer


def apply_llm_schedule(optimizer, lrm: float, step: int):
    muon_momentum = get_muon_momentum(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group["kind"] == "muon":
            group["momentum"] = muon_momentum


def freeze_value_embedding_path(model):
    for p in model.value_embeds.parameters():
        p.requires_grad = False
    for block in model.transformer.h:
        if block.attn.ve_gate is not None:
            for p in block.attn.ve_gate.parameters():
                p.requires_grad = False


def control_losses(model, projector, extractor, examples, image_root, image_zip, device, batch_size=4, step=1, skip_bad_images=False):
    count = min(max(batch_size, 0), len(examples))
    start = ((max(step, 1) - 1) * count) % len(examples) if count > 0 else 0
    batch = [examples[(start + i) % len(examples)] for i in range(count)]
    if len(batch) < 2:
        return None
    feats, batch = batch_features_and_examples(
        extractor,
        batch,
        image_root,
        image_zip=image_zip,
        skip_bad_images=skip_bad_images,
    )
    if feats is None or len(batch) < 2:
        return None
    feats = feats.to(device)
    rows = [ex["tokens"] for ex in batch]
    masks = [ex["mask"] for ex in batch]
    with torch.no_grad():
        aligned_batch = build_multimodal_batch(model, projector, rows, feats, loss_mask_rows=masks, value_fallback_token_id=rows[0][0])
        aligned = model(aligned_batch.value_token_ids, aligned_batch.targets, input_embeds=aligned_batch.input_embeds)
        shuffled_batch = build_multimodal_batch(model, projector, rows, feats.roll(shifts=1, dims=0), loss_mask_rows=masks, value_fallback_token_id=rows[0][0])
        shuffled = model(shuffled_batch.value_token_ids, shuffled_batch.targets, input_embeds=shuffled_batch.input_embeds)
        no_image_batch = build_multimodal_batch(model, projector, rows, torch.zeros_like(feats), loss_mask_rows=masks, value_fallback_token_id=rows[0][0])
        no_image = model(no_image_batch.value_token_ids, no_image_batch.targets, input_embeds=no_image_batch.input_embeds)
    return float(aligned), float(shuffled), float(no_image)


def evaluate_vlm_bpb(model, projector, extractor, examples, image_root, image_zip, device, token_bytes, batch_size=4, max_examples=-1, max_seq_len=2048, skip_bad_images=False):
    limit = len(examples) if max_examples <= 0 else min(max_examples, len(examples))
    model_was_training = model.training
    projector_was_training = projector.training
    model.eval()
    projector.eval()
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=device)
    total_bytes = torch.tensor(0, dtype=torch.int64, device=device)
    total_targets = torch.tensor(0, dtype=torch.int64, device=device)
    seen = 0
    try:
        with torch.no_grad():
            for start in range(0, limit, batch_size):
                batch_examples = examples[start : start + batch_size]
                feats, batch_examples = batch_features_and_examples(
                    extractor,
                    batch_examples,
                    image_root,
                    image_zip=image_zip,
                    skip_bad_images=skip_bad_images,
                )
                if feats is None or not batch_examples:
                    continue
                feats = feats.to(device=device, non_blocking=True)
                rows = [ex["tokens"] for ex in batch_examples]
                masks = [ex["mask"] for ex in batch_examples]
                batch = build_multimodal_batch(model, projector, rows, feats, loss_mask_rows=masks, max_seq_len=max_seq_len, value_fallback_token_id=rows[0][0])
                loss = model(batch.value_token_ids, batch.targets, input_embeds=batch.input_embeds, loss_reduction="none").view(-1)
                targets = batch.targets.view(-1)
                valid = targets >= 0
                safe_targets = torch.where(valid, targets, torch.zeros_like(targets))
                num_bytes = torch.where(valid, token_bytes[safe_targets], torch.zeros_like(targets, dtype=token_bytes.dtype))
                counted = num_bytes > 0
                total_nats += (loss * counted).sum()
                total_bytes += num_bytes.sum()
                total_targets += counted.sum()
                seen += len(batch_examples)
    finally:
        model.train(model_was_training)
        projector.train(projector_was_training)
    bytes_f = int(total_bytes.item())
    nats_f = float(total_nats.item())
    targets_i = int(total_targets.item())
    return {
        "bpb": nats_f / (math.log(2) * bytes_f) if bytes_f > 0 else float("inf"),
        "loss": nats_f / max(targets_i, 1),
        "n": seen,
        "bytes": bytes_f,
        "target_tokens": targets_i,
    }


def save_training_checkpoint(out_dir, step, model, projector, args, model_meta, data_path, rank=0):
    meta = {
        "step": step,
        "stage": args.stage,
        "model_config": model_meta["model_config"],
        "user_config": vars(args),
        "data_path": data_path,
        "data_config": {
            "data_path": data_path,
            "data_json": getattr(args, "data_json", None),
            "hf_repo": getattr(args, "hf_repo", None),
            "hf_file": getattr(args, "hf_file", None),
            "hf_config": getattr(args, "hf_config", None),
            "stream_hf_data": True,
            "max_examples": getattr(args, "max_examples", None),
            "image_root": getattr(args, "image_root", None),
            "image_zip": getattr(args, "image_zip", None),
            "hf_image_zip": getattr(args, "hf_image_zip", None),
            "skip_bad_images": getattr(args, "skip_bad_images", None),
        },
        "vision_config": {
            "siglip_model_id": args.siglip_model_id,
            "pooling": "nanovlm_pixel_shuffle",
            "vision_grid": VISION_GRID,
            "vision_tokens": VISION_TOKENS,
            "projector_vision_dim": projector.vision_dim,
            "projector_n_embd": projector.n_embd,
        },
        "init_vlm_checkpoint": {
            "dir": args.init_vlm_checkpoint_dir,
            "step": args.init_vlm_checkpoint_step,
        },
    }
    save_vlm_checkpoint(out_dir, step, model, projector, optimizer_data=None, meta_data=meta, rank=rank)


def main():
    parser = argparse.ArgumentParser(description="Train minimal nanochat-llava vision path")
    parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2], help="1=projector only, 2=projector+LLM")
    parser.add_argument("--data-json", default=None, help="local LLaVA-style JSON/JSONL")
    parser.add_argument("--hf-repo", default=None, help="Hugging Face dataset repo containing --hf-file")
    parser.add_argument("--hf-file", default=None, help="JSON/JSONL file inside --hf-repo")
    parser.add_argument("--hf-config", default=None, help="Hugging Face dataset config to stream directly, e.g. FineVision LLaVA_Instruct_150K")
    parser.add_argument("--hf-image-zip", default=None, help="optional image zip inside --hf-repo")
    parser.add_argument("--image-zip", default=None, help="local zip containing referenced images")
    parser.add_argument("--image-root", default=None, help="directory containing referenced images")
    parser.add_argument("--skip-bad-images", action=argparse.BooleanOptionalAction, default=False, help="skip records whose image cannot be opened")
    parser.add_argument("--max-examples", type=int, default=-1)
    parser.add_argument("--siglip-model-id", default=SIGLIP_MODEL_ID)
    parser.add_argument("--siglip-cache-dir", default=None, help="optional HF cache dir for SigLIP weights")
    parser.add_argument("--hf-checkpoint", default="karpathy/nanochat-d32", help="HF nanochat checkpoint repo to link into NANOCHAT_BASE_DIR")
    parser.add_argument("--model-tag", default="d32")
    parser.add_argument("--model-step", type=int, default=None)
    parser.add_argument("--device-type", default="", choices=["", "cuda", "cpu", "mps"])
    parser.add_argument("--device-batch-size", type=int, default=4)
    parser.add_argument("--max-batch-tokens", type=int, default=0, help="optional cap on padded tokens per device batch")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--num-iterations", type=int, default=1000)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--projector-lr", type=float, default=2e-3)
    parser.add_argument("--val-json", default=None, help="local held-out VLM JSON for validation BPB")
    parser.add_argument("--val-image-root", default=None, help="image root for --val-json; defaults to the JSON parent directory")
    parser.add_argument("--val-image-zip", default=None, help="optional image zip for --val-json")
    parser.add_argument("--eval-every", type=int, default=-1, help="evaluate held-out VLM BPB every N steps (-1 = disable)")
    parser.add_argument("--eval-examples", type=int, default=-1, help="held-out examples to score for BPB (-1 = all)")
    parser.add_argument("--eval-batch-size", type=int, default=4, help="batch size for held-out VLM BPB")
    parser.add_argument("--vlmcore-every", type=int, default=-1, help="run VLMCORE benchmark subset every N steps (-1 = disable)")
    parser.add_argument("--vlmcore-benchmarks", default="mmstar,scienceqa,chartqa,mmmu,textvqa")
    parser.add_argument("--vlmcore-mmmu-configs", default="Accounting")
    parser.add_argument("--vlmcore-max-per-benchmark", type=int, default=100)
    parser.add_argument("--vlmcore-max-scan", type=int, default=2000)
    parser.add_argument("--vlmcore-max-new-tokens", type=int, default=16)
    parser.add_argument("--vlmcore-print-samples", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=-1)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--profile-timing", action="store_true", help="log per-step image decode/processor/SigLIP and LLM timing")
    parser.add_argument("--control-batch-size", type=int, default=4)
    parser.add_argument("--control-margin", type=float, default=0.01)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--init-vlm-checkpoint-dir", default=None, help="optional stage-1 VLM checkpoint dir to initialize projector/model")
    parser.add_argument("--init-vlm-checkpoint-step", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.eval_every > 0 and not args.val_json:
        parser.error("--eval-every requires --val-json")

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    _, ddp_rank, _, ddp_world_size, device = compute_init(device_type)
    assert ddp_world_size == 1, "v0 VLM trainer is single-GPU; launch one process"
    synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
    wandb_run = DummyWandb() if args.run == "dummy" or ddp_rank != 0 else wandb.init(project="nanochat-vlm", name=args.run, config=vars(args))

    base_dir = get_base_dir()
    out_dir = args.out_dir or os.path.join(base_dir, "vlm_checkpoints", f"{args.model_tag}_stage{args.stage}")
    if args.hf_checkpoint:
        ensure_hf_nanochat_checkpoint(args.hf_checkpoint, base_dir, model_tag=args.model_tag, source=MODEL_SOURCE)

    model, tokenizer, meta = load_model(MODEL_SOURCE, device, phase="train", model_tag=args.model_tag, step=args.model_step)
    args.embedding_lr = meta.get("user_config", {}).get("embedding_lr", 0.3)
    args.unembedding_lr = meta.get("user_config", {}).get("unembedding_lr", 0.004)
    args.matrix_lr = meta.get("user_config", {}).get("matrix_lr", 0.02)
    args.scalar_lr = meta.get("user_config", {}).get("scalar_lr", 0.5)
    args.init_lr_frac = INIT_LR_FRAC
    args.warmup_ratio = 0.0
    args.warmdown_ratio = WARMDOWN_RATIO
    args.final_lr_frac = 0.0
    args.weight_decay = 0.0
    t = time.perf_counter()
    records, data_path = load_records(args)
    print0(f"Loaded {len(records):,} records in {time.perf_counter() - t:.2f}s")
    maybe_use_hf_image_zip(args)
    t = time.perf_counter()
    examples = render_records(records, tokenizer, args.stage, args.max_seq_len)
    print0(f"Rendered {len(examples):,} usable examples in {time.perf_counter() - t:.2f}s")
    val_examples = None
    val_image_root = None
    if args.val_json:
        val_path = Path(args.val_json)
        val_image_root = args.val_image_root or str(val_path.parent)
        val_records = _load_json(val_path)
        assert isinstance(val_records, list), f"expected a JSON list in {val_path}"
        val_examples = render_records(val_records, tokenizer, args.stage, args.max_seq_len)
        if args.eval_examples > 0:
            val_examples = val_examples[: args.eval_examples]
        print0(f"Rendered {len(val_examples):,} held-out validation examples from {val_path}")

    siglip_cache_dir = args.siglip_cache_dir or os.environ.get("NANOCHAT_SIGLIP_CACHE_DIR")
    extractor = SigLIPPooledFeatureExtractor(args.siglip_model_id, device=device, cache_dir=siglip_cache_dir, verbose=ddp_rank == 0)
    projector = VisionProjector(extractor.vision_dim, model.config.n_embd).to(device=device)
    if args.init_vlm_checkpoint_dir:
        assert args.init_vlm_checkpoint_step is not None, "--init-vlm-checkpoint-step is required with --init-vlm-checkpoint-dir"
        model_state, projector, _, init_meta = load_vlm_checkpoint(args.init_vlm_checkpoint_dir, args.init_vlm_checkpoint_step, device, load_optimizer=False, checkpoint_device=torch.device("cpu"))
        model.load_state_dict(model_state, strict=True)
        assert projector.vision_dim == extractor.vision_dim, "stage checkpoint projector vision dim does not match SigLIP"
        print0(f"Initialized VLM state from {args.init_vlm_checkpoint_dir} step {args.init_vlm_checkpoint_step} (stage {init_meta.get('stage')})", flush=True)

    if args.stage == 1:
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        llm_optimizer = None
    else:
        for p in model.parameters():
            p.requires_grad = True
        freeze_value_embedding_path(model)
        model.train()
        llm_optimizer = setup_llm_optimizer(model, args)
    projector.train()
    projector_optimizer = torch.optim.AdamW(projector.parameters(), lr=args.projector_lr, weight_decay=0.0)

    rng = random.Random(args.seed)
    cursor = 0
    smooth_loss = 0.0
    smooth_count = 0
    t_start = time.time()
    gpu_name = torch.cuda.get_device_name(0) if device_type == "cuda" else device_type
    gpu_peak_flops = get_peak_flops(gpu_name) if device_type == "cuda" else float("inf")
    num_flops_per_token = model.estimate_flops()
    total_params = count_params(model.parameters()) + count_params(projector.parameters())
    total_trainable = count_params(p for p in list(model.parameters()) + list(projector.parameters()) if p.requires_grad)
    token_bytes = get_token_bytes(device=device) if val_examples is not None else None
    print0(f"VLM stage {args.stage} | GPU: {gpu_name} | examples: {len(examples):,} | data: {data_path} | out: {out_dir}", flush=True)
    print0(f"Params total/trainable: {total_params:,}/{total_trainable:,}")
    print0(f"Estimated LLM FLOPs/token: {num_flops_per_token:e} | Peak BF16 FLOPS: {gpu_peak_flops:.2e}")
    llm_lr_text = "frozen" if llm_optimizer is None else (
        f"nanochat MuonAdamW init_frac={args.init_lr_frac:g} "
        f"unembed={args.unembedding_lr:g} embed={args.embedding_lr:g} matrix={args.matrix_lr:g} scalar={args.scalar_lr:g}"
    )
    print0(f"LRs: projector={args.projector_lr:g} llm={llm_lr_text}")
    setup_mem_mib = torch.cuda.memory_allocated() / 1024 / 1024 if device_type == "cuda" else 0.0
    print0(f"Allocated memory after setup: {setup_mem_mib:.2f}MiB")
    if device_type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    min_val_bpb = float("inf")

    def run_val_bpb(step):
        nonlocal min_val_bpb
        val_stats = evaluate_vlm_bpb(
            model,
            projector,
            extractor,
            val_examples,
            val_image_root,
            args.val_image_zip,
            device,
            token_bytes,
            batch_size=args.eval_batch_size,
            max_examples=args.eval_examples,
            max_seq_len=args.max_seq_len,
            skip_bad_images=args.skip_bad_images,
        )
        min_val_bpb = min(min_val_bpb, val_stats["bpb"])
        print0(
            f"step {step:05d}/{args.num_iterations:05d} | val_bpb {val_stats['bpb']:.4f} | "
            f"min_val_bpb {min_val_bpb:.4f} | val_loss {val_stats['loss']:.4f} | val_examples {val_stats['n']}",
            flush=True,
        )
        wandb_run.log({
            "step": step,
            "val/bpb": val_stats["bpb"],
            "val/min_bpb": min_val_bpb,
            "val/loss": val_stats["loss"],
            "val/examples": val_stats["n"],
            "val/bytes": val_stats["bytes"],
            "val/target_tokens": val_stats["target_tokens"],
        })

    def run_vlmcore(step):
        results = evaluate_vlm(
            model,
            projector,
            tokenizer,
            extractor,
            benchmarks=args.vlmcore_benchmarks,
            mmmu_configs=args.vlmcore_mmmu_configs,
            limit=args.vlmcore_max_per_benchmark,
            max_scan=args.vlmcore_max_scan,
            max_new_tokens=args.vlmcore_max_new_tokens,
            print_samples=args.vlmcore_print_samples,
        )
        scores = {key: row["score"] for key, row in results["benchmarks"].items()}
        mean_score = sum(scores.values()) / max(len(scores), 1)
        score_str = " ".join(f"{key}={score:.3f}" for key, score in scores.items())
        print0(f"step {step:05d}/{args.num_iterations:05d} | VLMCORE {mean_score:.4f} | {score_str}", flush=True)
        wandb_run.log({
            "step": step,
            "vlmcore/mean_score": mean_score,
            **{f"vlmcore/{key}_score": score for key, score in scores.items()},
        })

    if args.eval_every > 0:
        run_val_bpb(0)

    for step in range(1, args.num_iterations + 1):
        synchronize()
        t0 = time.perf_counter()
        profile = {key: 0.0 for key in PROFILE_KEYS}
        train_loss = None
        projector_optimizer.zero_grad(set_to_none=True)
        if llm_optimizer is not None:
            llm_optimizer.zero_grad(set_to_none=True)
        tokens_this_step = 0
        samples_this_step = 0
        for _ in range(args.grad_accum_steps):
            t = time.perf_counter()
            batch_examples, cursor = next_batch(examples, args.device_batch_size, cursor, rng, max_batch_tokens=args.max_batch_tokens)
            rows = [ex["tokens"] for ex in batch_examples]
            masks = [ex["mask"] for ex in batch_examples]
            profile["data"] += time.perf_counter() - t

            if args.profile_timing:
                synchronize()
            t = time.perf_counter()
            feats, batch_examples = batch_features_and_examples(
                extractor,
                batch_examples,
                args.image_root,
                image_zip=args.image_zip,
                skip_bad_images=args.skip_bad_images,
                profile=profile if args.profile_timing else None,
                synchronize=synchronize if args.profile_timing else None,
            )
            if feats is not None:
                feats = feats.to(device=device, non_blocking=True)
            if feats is None or not batch_examples:
                continue
            rows = [ex["tokens"] for ex in batch_examples]
            masks = [ex["mask"] for ex in batch_examples]
            if args.profile_timing:
                synchronize()
            image_elapsed = time.perf_counter() - t
            profile["image_siglip"] += image_elapsed if not args.profile_timing else 0.0

            t = time.perf_counter()
            batch = build_multimodal_batch(model, projector, rows, feats, loss_mask_rows=masks, max_seq_len=args.max_seq_len, value_fallback_token_id=rows[0][0])
            profile["batch"] += time.perf_counter() - t

            if args.profile_timing:
                synchronize()
            t = time.perf_counter()
            loss = model(batch.value_token_ids, batch.targets, input_embeds=batch.input_embeds) / args.grad_accum_steps
            loss.backward()
            if args.profile_timing:
                synchronize()
            profile["fwdbwd"] += time.perf_counter() - t
            train_loss = loss.detach() * args.grad_accum_steps
            tokens_this_step += int(batch.lengths.sum())
            samples_this_step += len(batch_examples)
        if train_loss is None:
            raise RuntimeError("no usable images loaded for this optimizer step; check image URLs or disable --skip-bad-images")
        lrm = get_lr_multiplier(step, args.num_iterations, args.warmup_ratio, args.warmdown_ratio, args.final_lr_frac)
        if llm_optimizer is not None:
            apply_llm_schedule(llm_optimizer, lrm, step)
        if args.profile_timing:
            synchronize()
        t = time.perf_counter()
        projector_optimizer.step()
        if llm_optimizer is not None:
            llm_optimizer.step()
        if args.profile_timing:
            synchronize()
        profile["optim"] += time.perf_counter() - t

        synchronize()
        dt = time.perf_counter() - t0
        loss_f = float(train_loss)
        smooth_loss = 0.9 * smooth_loss + 0.1 * loss_f
        smooth_count += 1
        debiased = smooth_loss / (1 - 0.9**smooth_count)
        samples_per_sec = samples_this_step / max(dt, 1e-9)
        tokens_per_sec = tokens_this_step / max(dt, 1e-9)
        flops_per_sec = num_flops_per_token * tokens_this_step / max(dt, 1e-9)
        mfu = 100 * flops_per_sec / gpu_peak_flops
        if args.profile_timing:
            profile["image_siglip"] = sum(profile[key] for key in IMAGE_PROFILE_KEYS)
        if step == 1 or step % args.log_every == 0 or step == args.num_iterations:
            controls = control_losses(
                model,
                projector,
                extractor,
                examples,
                args.image_root,
                args.image_zip,
                device,
                batch_size=args.control_batch_size,
                step=step,
                skip_bad_images=args.skip_bad_images,
            )
            controls_str = ""
            profile_str = ""
            log_data = {
                "step": step,
                "train/loss": debiased,
                "train/raw_loss": loss_f,
                "train/samples_per_sec": samples_per_sec,
                "train/tokens_per_sec": tokens_per_sec,
                "train/mfu": mfu,
                "train/lrm": lrm if llm_optimizer is not None else 0.0,
            }
            if args.profile_timing:
                profile_str = (
                    " | timing data/image_total/open/processor/h2d/siglip/pool/batch/fwdbwd/optim "
                    f"{profile['data']:.3f}/{profile['image_siglip']:.3f}/{profile['image_open']:.3f}/"
                    f"{profile['image_processor']:.3f}/{profile['image_transfer']:.3f}/"
                    f"{profile['siglip_forward']:.3f}/{profile['siglip_pool']:.3f}/"
                    f"{profile['batch']:.3f}/{profile['fwdbwd']:.3f}/{profile['optim']:.3f}s"
                )
                log_data.update({f"timing/{k}_sec": v for k, v in profile.items()})
            if controls is not None:
                aligned, shuffled, no_image = controls
                controls_pass = aligned + args.control_margin < shuffled and aligned + args.control_margin < no_image
                controls_str = f" | controls aligned/shuffled/no_image {aligned:.4f}/{shuffled:.4f}/{no_image:.4f} pass={controls_pass}"
                log_data.update({
                    "controls/aligned_loss": aligned,
                    "controls/shuffled_loss": shuffled,
                    "controls/no_image_loss": no_image,
                    "controls/pass": int(controls_pass),
                })
            print0(
                f"step {step:05d}/{args.num_iterations:05d} | loss {debiased:.6f} | "
                f"samples/sec {samples_per_sec:.2f} | tokens/sec {tokens_per_sec:.0f} | bf16_mfu {mfu:.2f}"
                f"{'' if llm_optimizer is None else f' | lrm {lrm:.3f}'}{profile_str}{controls_str}",
                flush=True,
            )
            wandb_run.log(log_data)
        if args.eval_every > 0 and (step % args.eval_every == 0 or step == args.num_iterations):
            run_val_bpb(step)
        if args.vlmcore_every > 0 and (step % args.vlmcore_every == 0 or step == args.num_iterations):
            run_vlmcore(step)
        if args.save_every > 0 and step % args.save_every == 0:
            save_training_checkpoint(out_dir, step, model, projector, args, meta, data_path, rank=ddp_rank)

    if args.save_every <= 0 or args.num_iterations % args.save_every != 0:
        save_training_checkpoint(out_dir, args.num_iterations, model, projector, args, meta, data_path, rank=ddp_rank)

    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024 if device_type == "cuda" else 0.0
    total_time_min = (time.time() - t_start) / 60
    print0(f"Peak memory usage: {peak_mem:.2f}MiB", flush=True)
    print0(f"Total training time: {total_time_min:.2f}m", flush=True)
    wandb_run.log({"gpu/peak_mem_mib": peak_mem, "train/total_time_min": total_time_min})
    wandb_run.finish()
    compute_cleanup()


if __name__ == "__main__":
    main()
