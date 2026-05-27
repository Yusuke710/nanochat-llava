"""
Minimal nanochat-llava trainer.

This starts from an already-trained nanochat checkpoint, freezes SigLIP, and
trains a linear vision projector plus nanochat on visual-instruction examples.
Packed examples use FA3 varlen attention so examples in the same batch cannot
attend across boundaries.
"""

import argparse
import itertools
import io
import json
import os
import random
import time
import zipfile
from pathlib import Path

import torch
import wandb
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info

from nanochat.checkpoint_manager import load_model
from nanochat.common import DummyWandb, autodetect_device_type, compute_cleanup, compute_init, get_base_dir, get_peak_flops, print0
from nanochat.flash_attention import require_fa3_varlen
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
    render_vision_conversation,
    save_vlm_checkpoint,
)


IMAGE_KEYS = ("image", "image_path", "filename", "path")
DEFAULT_HF_REPO = "HuggingFaceM4/FineVisionMax"
MODEL_SOURCE = "sft"
UNTRUNCATED_MAX_TOKENS = 1_000_000_000
INIT_LR_FRAC = 0.05
WARMDOWN_RATIO = 0.5
_ZIP_CACHE = {}
DEFAULT_VAL_EXAMPLES = 2048


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
    if conv is None:
        question = example.get("question", "Describe the image.")
        answer = example.get("answer", example.get("caption", ""))
        example["messages"] = [
            {"role": "user", "content": f"{IMAGE_MARKER}\n{question}"},
            {"role": "assistant", "content": answer},
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


def _open_hf_record_stream(hf_repo: str):
    from datasets import load_dataset

    ds = load_dataset(hf_repo, split="train", streaming=True)
    try:
        from datasets import Image as HFImage
        from datasets import Sequence

        if "images" in (getattr(ds, "features", None) or {}):
            ds = ds.cast_column("images", Sequence(HFImage(decode=False)))
    except ImportError:
        pass
    return ds


def hf_source(args):
    hf_repo = args.hf_repo or DEFAULT_HF_REPO
    return f"stream:{hf_repo}"


def load_local_records(path_arg):
    path = Path(path_arg)
    records = _load_json(path)
    assert isinstance(records, list), f"expected a JSON list in {path}"
    return records, str(path)


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


def get_zip_file(image_zip):
    zip_path = str(Path(image_zip).expanduser())
    entry = _ZIP_CACHE.get(zip_path)
    if entry is not None:
        return entry
    zf = zipfile.ZipFile(zip_path)
    names = [info.filename for info in zf.infolist() if not info.is_dir()]
    entry = {"zip": zf, "names": set(names), "basename": {name.rsplit("/", 1)[-1]: name for name in names}}
    _ZIP_CACHE[zip_path] = entry
    return entry


def open_image(record, image_root=None, image_zip=None):
    value = _image_value(record)
    if value is None:
        raise KeyError(f"record has no image field among {IMAGE_KEYS}")
    if hasattr(value, "convert"):
        return value.convert("RGB")

    from PIL import Image

    if isinstance(value, dict):
        if value.get("bytes") is not None:
            with Image.open(io.BytesIO(value["bytes"])) as img:
                return img.convert("RGB")
        if value.get("path"):
            with Image.open(value["path"]) as img:
                return img.convert("RGB")
        raise ValueError("HF image dict has neither bytes nor path")

    path = Path(str(value))
    candidates = [path] if path.is_absolute() else []
    if image_root and not path.is_absolute():
        root = Path(image_root)
        candidates.extend([root / path, root / "images" / path, root / path.name])
    for candidate in candidates or [path]:
        if candidate.exists():
            with Image.open(candidate) as img:
                return img.convert("RGB")
    if image_zip:
        entry = get_zip_file(image_zip)
        member = str(value).replace("\\", "/").lstrip("/")
        member = member if member in entry["names"] else entry["basename"].get(Path(member).name)
        if member:
            with entry["zip"].open(member) as f:
                with Image.open(f) as img:
                    return img.convert("RGB")
    raise FileNotFoundError(f"could not find image {value!r}")


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


def render_record(rec, tokenizer, max_seq_len=2048):
    if _image_value(rec) is None:
        return None
    tokens, mask = render_vision_conversation(tokenizer, _ensure_image_marker_in_conversation(rec), max_tokens=UNTRUNCATED_MAX_TOKENS)
    if count_image_tokens(tokens) != 1 or count_image_tokens(tokens[:-1]) != 1:
        return None
    expanded_len = expanded_input_len(tokens)
    if expanded_len > max_seq_len:
        return None
    if supervised_target_count(tokens, mask) <= 0:
        return None
    return {"tokens": tokens, "mask": mask, "record": rec, "expanded_len": expanded_len}


def render_records(records, tokenizer, max_seq_len=2048):
    rendered = []
    for rec in records:
        example = render_record(rec, tokenizer, max_seq_len=max_seq_len)
        if example is not None:
            rendered.append(example)
    assert rendered, "no usable image-text examples loaded"
    return rendered


def collect_rendered_examples(records, tokenizer, max_seq_len=2048, limit=DEFAULT_VAL_EXAMPLES):
    if limit <= 0:
        return [], 0
    rendered = []
    seen = 0
    for rec in records:
        seen += 1
        example = render_record(rec, tokenizer, max_seq_len=max_seq_len)
        if example is not None:
            rendered.append(example)
            if len(rendered) >= limit:
                break
    return rendered, seen


def split_train_val_examples(examples, val_examples=DEFAULT_VAL_EXAMPLES, use_val=True):
    if not use_val or val_examples <= 0 or len(examples) < 2:
        return examples, []
    val_count = min(val_examples, max(1, len(examples) // 10), len(examples) - 1)
    return examples[:-val_count], examples[-val_count:]


def next_batch(examples, batch_size, cursor, rng, max_batch_tokens=0):
    if cursor == 0:
        rng.shuffle(examples)
    batch = []
    total_len = 0
    while len(batch) < batch_size:
        if cursor >= len(examples):
            cursor = 0
            rng.shuffle(examples)
            if batch:
                break
        candidate = examples[cursor]
        candidate_len = int(candidate["expanded_len"])
        if batch and max_batch_tokens > 0 and total_len + candidate_len > max_batch_tokens:
            break
        batch.append(candidate)
        total_len += candidate_len
        cursor += 1
        if cursor >= len(examples):
            cursor = 0
            break
    return batch, cursor


def open_images(examples, image_root=None, image_zip=None, skip_bad_images=False):
    images = []
    kept = []
    for i, example in enumerate(examples):
        try:
            images.append(open_image(example["record"], image_root=image_root, image_zip=image_zip))
            kept.append(example)
        except Exception as exc:
            if not skip_bad_images:
                raise
            print0(f"skipping image {example['record'].get('id', i)}: {type(exc).__name__}: {exc}")
    return images, kept


def pack_examples(examples, image_features):
    row = []
    mask = []
    segment_lengths = []
    for example in examples:
        row.extend(example["tokens"])
        mask.extend(example["mask"])
        segment_lengths.append(len(example["tokens"]))
    return [row], [mask], image_features[:len(examples)], [len(examples)], [segment_lengths]


class PackedVisionBatchDataset(Dataset):
    """Prepared VLM microbatches with image decode/CPU processing in DataLoader workers."""

    def __init__(
        self,
        batches,
        siglip_model_id,
        siglip_cache_dir=None,
        image_root=None,
        image_zip=None,
        skip_bad_images=True,
    ):
        self.batches = batches
        self.siglip_model_id = siglip_model_id
        self.siglip_cache_dir = siglip_cache_dir
        self.image_root = image_root
        self.image_zip = image_zip
        self.skip_bad_images = skip_bad_images
        self.processor = None

    def __len__(self):
        return len(self.batches)

    def _processor(self):
        if self.processor is None:
            from transformers import AutoImageProcessor

            kwargs = {"cache_dir": self.siglip_cache_dir} if self.siglip_cache_dir else {}
            self.processor = AutoImageProcessor.from_pretrained(self.siglip_model_id, **kwargs)
        return self.processor

    def __getitem__(self, idx):
        examples = self.batches[idx]
        images, kept = open_images(examples, self.image_root, self.image_zip, self.skip_bad_images)
        if not images:
            return None
        pixel_values = self._processor()(images=images, return_tensors="pt")["pixel_values"]
        rows, masks, pixel_values, image_counts, segment_lengths = pack_examples(kept, pixel_values)
        return {
            "rows": rows,
            "masks": masks,
            "pixel_values": pixel_values,
            "image_counts": image_counts,
            "segment_lengths": segment_lengths,
            "num_examples": len(kept),
        }


class StreamingVisionBatchDataset(IterableDataset):
    """FineVisionMax streaming microbatches with online token packing."""

    def __init__(
        self,
        args,
        tokenizer,
        siglip_cache_dir=None,
        skip_records=0,
        max_batches=None,
    ):
        self.hf_repo = args.hf_repo or DEFAULT_HF_REPO
        self.batch_size = args.device_batch_size
        self.max_batch_tokens = args.max_batch_tokens
        self.max_seq_len = args.max_seq_len
        self.siglip_model_id = args.siglip_model_id
        self.siglip_cache_dir = siglip_cache_dir
        self.image_root = args.image_root
        self.image_zip = args.image_zip
        self.skip_bad_images = args.skip_bad_images
        self.tokenizer = tokenizer
        self.skip_records = skip_records
        self.max_batches = max_batches
        self.processor = None

    def _processor(self):
        if self.processor is None:
            from transformers import AutoImageProcessor

            kwargs = {"cache_dir": self.siglip_cache_dir} if self.siglip_cache_dir else {}
            self.processor = AutoImageProcessor.from_pretrained(self.siglip_model_id, **kwargs)
        return self.processor

    def _record_stream(self, worker_id, num_workers):
        ds = _open_hf_record_stream(self.hf_repo)
        if self.skip_records > 0:
            ds = ds.skip(self.skip_records) if hasattr(ds, "skip") else itertools.islice(ds, self.skip_records, None)
        return itertools.islice(ds, worker_id, None, num_workers)

    def _pack_batch(self, examples):
        images, kept = open_images(examples, self.image_root, self.image_zip, self.skip_bad_images)
        if not images:
            return None
        pixel_values = self._processor()(images=images, return_tensors="pt")["pixel_values"]
        rows, masks, pixel_values, image_counts, segment_lengths = pack_examples(kept, pixel_values)
        return {
            "rows": rows,
            "masks": masks,
            "pixel_values": pixel_values,
            "image_counts": image_counts,
            "segment_lengths": segment_lengths,
            "num_examples": len(kept),
        }

    def _batch_limit(self, num_workers):
        if self.max_batches is None:
            return None
        return max(1, (self.max_batches + num_workers - 1) // num_workers)

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        batch_limit = self._batch_limit(num_workers)
        yielded = 0
        batch = []
        total_len = 0
        for rec in self._record_stream(worker_id, num_workers):
            example = render_record(rec, self.tokenizer, max_seq_len=self.max_seq_len)
            if example is None:
                continue
            candidate_len = int(example["expanded_len"])
            if batch and self.max_batch_tokens > 0 and total_len + candidate_len > self.max_batch_tokens:
                packed = self._pack_batch(batch)
                if packed is not None:
                    yield packed
                    yielded += 1
                    if batch_limit is not None and yielded >= batch_limit:
                        return
                batch = []
                total_len = 0
            batch.append(example)
            total_len += candidate_len
            if len(batch) >= self.batch_size:
                packed = self._pack_batch(batch)
                if packed is not None:
                    yield packed
                    yielded += 1
                    if batch_limit is not None and yielded >= batch_limit:
                        return
                batch = []
                total_len = 0
        if batch:
            packed = self._pack_batch(batch)
            if packed is not None:
                yield packed


def build_training_batches(examples, num_batches, batch_size, max_batch_tokens, seed):
    examples = list(examples)
    rng = random.Random(seed)
    cursor = 0
    batches = []
    for _ in range(num_batches):
        batch, cursor = next_batch(examples, batch_size, cursor, rng, max_batch_tokens=max_batch_tokens)
        if not batch:
            raise RuntimeError("could not build a non-empty VLM microbatch")
        batches.append(batch)
    return batches


def num_packed_batches(examples, batch_size, max_batch_tokens):
    if not examples:
        return 0
    count = 0
    batch_count = 0
    total_len = 0
    for example in examples:
        candidate_len = int(example["expanded_len"])
        if batch_count and (
            batch_count >= batch_size or (max_batch_tokens > 0 and total_len + candidate_len > max_batch_tokens)
        ):
            count += 1
            batch_count = 0
            total_len = 0
        batch_count += 1
        total_len += candidate_len
    return count + int(batch_count > 0)


def build_batch_loader(args, tokenizer, siglip_cache_dir, device_type, num_batches=None, seed=0, examples=None, skip_records=0):
    workers = max(0, args.num_workers)
    kwargs = {
        "batch_size": None,
        "num_workers": workers,
        "pin_memory": device_type == "cuda",
    }
    if workers > 0:
        kwargs.update({
            "prefetch_factor": 2,
        })
        if device_type == "cuda":
            kwargs["multiprocessing_context"] = "spawn"
    if examples is None:
        dataset = StreamingVisionBatchDataset(
            args,
            tokenizer,
            siglip_cache_dir=siglip_cache_dir,
            skip_records=skip_records,
            max_batches=num_batches,
        )
    else:
        assert num_batches is not None
        batches = build_training_batches(
            examples,
            num_batches=num_batches,
            batch_size=args.device_batch_size,
            max_batch_tokens=args.max_batch_tokens,
            seed=seed,
        )
        dataset = PackedVisionBatchDataset(
            batches,
            args.siglip_model_id,
            siglip_cache_dir=siglip_cache_dir,
            image_root=args.image_root,
            image_zip=args.image_zip,
            skip_bad_images=args.skip_bad_images,
        )
    return DataLoader(dataset, **kwargs)


def compute_vlm_loss(model, projector, extractor, packed):
    feats = extractor.encode_pixel_values(packed["pixel_values"])
    rows = packed["rows"]
    batch = build_multimodal_batch(
        model,
        projector,
        rows,
        feats,
        loss_mask_rows=packed["masks"],
        image_counts_per_row=packed["image_counts"],
        segment_token_lengths_per_row=packed["segment_lengths"],
        max_seq_len=None,
        value_fallback_token_id=rows[0][0],
    )
    loss = model(
        batch.value_token_ids,
        batch.targets,
        input_embeds=batch.input_embeds,
        position_ids=batch.position_ids,
        segment_starts=batch.segment_starts,
        cu_seqlens=batch.cu_seqlens,
        max_seqlen=batch.max_seqlen,
        loss_indices=batch.loss_indices,
        loss_targets=batch.loss_targets,
    )
    return loss, int(batch.token_count or batch.lengths.sum()), int(batch.loss_targets.numel()), int(packed["num_examples"])


@torch.no_grad()
def evaluate_vlm_loss(model, projector, extractor, loader):
    model_was_training = model.training
    projector_was_training = projector.training
    model.eval()
    projector.eval()
    total_loss = 0.0
    total_targets = 0
    total_tokens = 0
    total_samples = 0
    for packed in loader:
        if packed is None or int(packed["num_examples"]) == 0:
            continue
        loss, token_count, target_count, sample_count = compute_vlm_loss(model, projector, extractor, packed)
        total_loss += float(loss) * target_count
        total_targets += target_count
        total_tokens += token_count
        total_samples += sample_count
    if model_was_training:
        model.train()
    if projector_was_training:
        projector.train()
    if total_targets == 0:
        return None
    return {
        "loss": total_loss / total_targets,
        "target_tokens": total_targets,
        "tokens": total_tokens,
        "samples": total_samples,
    }


def count_params(parameters):
    return sum(p.numel() for p in parameters)


def estimate_lm_head_flops_per_token(model):
    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    return 6.0 * count_params(base_model.lm_head.parameters())


def get_lr_multiplier(step: int, num_iterations: int, warmup_ratio: float, warmdown_ratio: float, final_lr_frac: float) -> float:
    progress = 0.0 if num_iterations <= 1 else (step - 1) / (num_iterations - 1)
    if warmup_ratio > 0 and progress < warmup_ratio:
        return (progress + 1e-8) / warmup_ratio
    if warmdown_ratio <= 0 or progress <= 1.0 - warmdown_ratio:
        return 1.0
    decay = (progress - (1.0 - warmdown_ratio)) / warmdown_ratio
    return (1.0 - decay) + decay * final_lr_frac


def freeze_value_embedding_path(model):
    for p in model.value_embeds.parameters():
        p.requires_grad = False
    for block in model.transformer.h:
        if block.attn.ve_gate is not None:
            for p in block.attn.ve_gate.parameters():
                p.requires_grad = False


def setup_llm_optimizer(model, args):
    optimizer = model.setup_optimizer(
        unembedding_lr=args.unembedding_lr,
        embedding_lr=args.embedding_lr,
        matrix_lr=args.matrix_lr,
        scalar_lr=args.scalar_lr,
        weight_decay=0.0,
    )
    for group in optimizer.param_groups:
        group["lr"] = group["lr"] * INIT_LR_FRAC
        group["initial_lr"] = group["lr"]
    return optimizer


def apply_llm_schedule(optimizer, lrm: float, step: int):
    muon_momentum = (1 - min(step / 300, 1.0)) * 0.85 + min(step / 300, 1.0) * 0.95
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group["kind"] == "muon":
            group["momentum"] = muon_momentum


def save_training_checkpoint(out_dir, step, model, projector, args, model_meta, data_path, rank=0):
    meta = {
        "step": step,
        "model_config": model_meta["model_config"],
        "user_config": vars(args),
        "data_path": data_path,
        "vision_config": {
            "siglip_model_id": args.siglip_model_id,
            "pooling": "nanovlm_pixel_shuffle",
            "vision_grid": VISION_GRID,
            "vision_tokens": VISION_TOKENS,
            "projector_vision_dim": projector.vision_dim,
            "projector_n_embd": projector.n_embd,
        },
    }
    save_vlm_checkpoint(out_dir, step, model, projector, optimizer_data=None, meta_data=meta, rank=rank)


def main():
    parser = argparse.ArgumentParser(description="Train minimal nanochat-llava")
    parser.add_argument("--run", type=str, default="dummy")
    parser.add_argument("--data-json", default=None)
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    parser.add_argument("--image-zip", default=None)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--skip-bad-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--siglip-model-id", default=SIGLIP_MODEL_ID)
    parser.add_argument("--siglip-cache-dir", default=None)
    parser.add_argument("--hf-checkpoint", default="karpathy/nanochat-d32")
    parser.add_argument("--model-tag", default="d32")
    parser.add_argument("--model-step", type=int, default=650)
    parser.add_argument("--device-type", default="", choices=["", "cuda", "cpu", "mps"])
    parser.add_argument("--device-batch-size", type=int, default=128)
    parser.add_argument("--max-batch-tokens", type=int, default=12000)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-iterations", type=int, default=1000)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--projector-lr", type=float, default=2e-3)
    parser.add_argument("--save-every", type=int, default=-1)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--val-examples", type=int, default=DEFAULT_VAL_EXAMPLES)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--init-vlm-checkpoint-dir", default=None)
    parser.add_argument("--init-vlm-checkpoint-step", type=int, default=None)
    parser.add_argument("--require-fa3-varlen", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.require_fa3_varlen:
        require_fa3_varlen()

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    _, ddp_rank, _, ddp_world_size, device = compute_init(device_type)
    assert ddp_world_size == 1, "v0 VLM trainer is single-GPU; launch one process"
    wandb_run = DummyWandb() if args.run == "dummy" or ddp_rank != 0 else wandb.init(project="nanochat-vlm", name=args.run, config=vars(args))

    base_dir = get_base_dir()
    out_dir = args.out_dir or os.path.join(base_dir, "vlm_checkpoints", args.model_tag)
    if args.hf_checkpoint:
        ensure_hf_nanochat_checkpoint(args.hf_checkpoint, base_dir, model_tag=args.model_tag, source=MODEL_SOURCE)

    model, tokenizer, meta = load_model(MODEL_SOURCE, device, phase="train", model_tag=args.model_tag, step=args.model_step)
    user_config = meta.get("user_config", {})
    args.embedding_lr = user_config.get("embedding_lr", 0.3)
    args.unembedding_lr = user_config.get("unembedding_lr", 0.004)
    args.matrix_lr = user_config.get("matrix_lr", 0.02)
    args.scalar_lr = user_config.get("scalar_lr", 0.5)

    use_val = args.eval_every > 0 and args.val_examples > 0
    t = time.perf_counter()
    val_records_seen = 0
    if args.data_json:
        records, data_path = load_local_records(args.data_json)
        examples = render_records(records, tokenizer, max_seq_len=args.max_seq_len)
        train_examples, val_examples = split_train_val_examples(
            examples,
            val_examples=args.val_examples,
            use_val=use_val,
        )
        train_desc = f"{len(train_examples):,}"
        print0(
            f"Loaded {len(records):,} records and rendered {len(train_examples):,} train / "
            f"{len(val_examples):,} val examples in {time.perf_counter() - t:.2f}s"
        )
    else:
        data_path = hf_source(args)
        train_examples = None
        val_examples = []
        if use_val and args.val_examples > 0:
            val_stream = _open_hf_record_stream(args.hf_repo or DEFAULT_HF_REPO)
            val_examples, val_records_seen = collect_rendered_examples(
                val_stream,
                tokenizer,
                max_seq_len=args.max_seq_len,
                limit=args.val_examples,
            )
            assert val_examples, f"streamed no usable validation examples from {data_path}"
        train_desc = f"stream after {val_records_seen:,} validation-scan records"
        print0(
            f"Prepared {data_path} with {train_desc} / {len(val_examples):,} val examples "
            f"in {time.perf_counter() - t:.2f}s"
        )

    siglip_cache_dir = args.siglip_cache_dir or os.environ.get("NANOCHAT_SIGLIP_CACHE_DIR")
    extractor = SigLIPPooledFeatureExtractor(args.siglip_model_id, device=device, cache_dir=siglip_cache_dir, verbose=ddp_rank == 0)
    projector = VisionProjector(extractor.vision_dim, model.config.n_embd).to(device=device)
    if args.init_vlm_checkpoint_dir:
        assert args.init_vlm_checkpoint_step is not None
        model_state, projector, _, init_meta = load_vlm_checkpoint(args.init_vlm_checkpoint_dir, args.init_vlm_checkpoint_step, device, load_optimizer=False, checkpoint_device=torch.device("cpu"))
        model.load_state_dict(model_state, strict=True)
        assert projector.vision_dim == extractor.vision_dim
        print0(f"Initialized VLM checkpoint from {args.init_vlm_checkpoint_dir} step {args.init_vlm_checkpoint_step} ({init_meta.get('step')})")

    for p in model.parameters():
        p.requires_grad = True
    freeze_value_embedding_path(model)
    model.train()
    projector.train()
    llm_optimizer = setup_llm_optimizer(model, args)
    projector_optimizer = torch.optim.AdamW(projector.parameters(), lr=args.projector_lr, weight_decay=0.0)
    if train_examples is None:
        batch_loader = build_batch_loader(
            args,
            tokenizer,
            siglip_cache_dir,
            device_type,
            num_batches=args.num_iterations * args.grad_accum_steps,
            seed=args.seed,
            skip_records=val_records_seen,
        )
    else:
        batch_loader = build_batch_loader(
            args,
            tokenizer,
            siglip_cache_dir,
            device_type,
            num_batches=args.num_iterations * args.grad_accum_steps,
            seed=args.seed,
            examples=train_examples,
        )
    batch_iter = iter(batch_loader)
    val_loader = None
    if val_examples:
        eval_batches = num_packed_batches(val_examples, args.device_batch_size, args.max_batch_tokens)
        val_loader = build_batch_loader(
            args,
            tokenizer,
            siglip_cache_dir,
            device_type,
            num_batches=eval_batches,
            seed=args.seed + 1,
            examples=val_examples,
        )

    smooth_loss = 0.0
    smooth_count = 0
    t_start = time.time()
    gpu_name = torch.cuda.get_device_name(0) if device_type == "cuda" else device_type
    gpu_peak_flops = get_peak_flops(gpu_name) if device_type == "cuda" else float("inf")
    flops_per_token = model.estimate_flops()
    lm_head_flops_per_token = estimate_lm_head_flops_per_token(model)
    trunk_flops_per_token = max(0.0, flops_per_token - lm_head_flops_per_token)
    total_params = count_params(model.parameters()) + count_params(projector.parameters())
    total_trainable = count_params(p for p in list(model.parameters()) + list(projector.parameters()) if p.requires_grad)
    print0(f"VLM | GPU: {gpu_name} | train: {train_desc} | val: {len(val_examples):,} | data: {data_path} | out: {out_dir}")
    print0(f"Params total/trainable: {total_params:,}/{total_trainable:,}")
    print0(f"FLOPs/token: {flops_per_token:e} | Peak BF16 FLOPS: {gpu_peak_flops:.2e}")
    if device_type == "cuda":
        print0(f"Allocated memory after setup: {torch.cuda.memory_allocated() / 1024 / 1024:.2f}MiB")
        torch.cuda.reset_peak_memory_stats()

    for step in range(1, args.num_iterations + 1):
        if device_type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        projector_optimizer.zero_grad(set_to_none=True)
        llm_optimizer.zero_grad(set_to_none=True)
        train_loss = None
        tokens_this_step = 0
        target_tokens_this_step = 0
        samples_this_step = 0
        for _ in range(args.grad_accum_steps):
            packed = next(batch_iter)
            if packed is None or int(packed["num_examples"]) == 0:
                continue
            loss, token_count, target_count, sample_count = compute_vlm_loss(model, projector, extractor, packed)
            loss = loss / args.grad_accum_steps
            loss.backward()
            train_loss = loss.detach() * args.grad_accum_steps
            tokens_this_step += token_count
            target_tokens_this_step += target_count
            samples_this_step += sample_count
        if train_loss is None:
            raise RuntimeError("no usable images loaded for this optimizer step")

        lrm = get_lr_multiplier(step, args.num_iterations, 0.0, WARMDOWN_RATIO, 0.0)
        apply_llm_schedule(llm_optimizer, lrm, step)
        projector_optimizer.step()
        llm_optimizer.step()

        if device_type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        loss_f = float(train_loss)
        smooth_loss = 0.9 * smooth_loss + 0.1 * loss_f
        smooth_count += 1
        debiased = smooth_loss / (1 - 0.9**smooth_count)
        tokens_per_sec = tokens_this_step / max(dt, 1e-9)
        samples_per_sec = samples_this_step / max(dt, 1e-9)
        step_flops = trunk_flops_per_token * tokens_this_step + lm_head_flops_per_token * target_tokens_this_step
        mfu = 100 * step_flops / max(dt, 1e-9) / gpu_peak_flops
        if step == 1 or step % args.log_every == 0 or step == args.num_iterations:
            print0(
                f"step {step:05d}/{args.num_iterations:05d} | loss {debiased:.6f} | "
                f"samples/sec {samples_per_sec:.2f} | tokens/sec {tokens_per_sec:.0f} | "
                f"target_tokens {target_tokens_this_step:,} | bf16_mfu {mfu:.2f} | lrm {lrm:.3f}",
                flush=True,
            )
            wandb_run.log({
                "step": step,
                "train/loss": debiased,
                "train/raw_loss": loss_f,
                "train/samples_per_sec": samples_per_sec,
                "train/tokens_per_sec": tokens_per_sec,
                "train/target_tokens": target_tokens_this_step,
                "train/mfu": mfu,
                "train/lrm": lrm,
            })
        if val_loader is not None and (step % args.eval_every == 0 or step == args.num_iterations):
            val_stats = evaluate_vlm_loss(model, projector, extractor, val_loader)
            if val_stats is not None:
                print0(
                    f"step {step:05d}/{args.num_iterations:05d} | val_loss {val_stats['loss']:.6f} | "
                    f"val_target_tokens {val_stats['target_tokens']:,}",
                    flush=True,
                )
                wandb_run.log({
                    "step": step,
                    "val/loss": val_stats["loss"],
                    "val/target_tokens": val_stats["target_tokens"],
                    "val/tokens": val_stats["tokens"],
                    "val/samples": val_stats["samples"],
                })
        if not args.no_save and args.save_every > 0 and step % args.save_every == 0:
            save_training_checkpoint(out_dir, step, model, projector, args, meta, data_path, rank=ddp_rank)

    if not args.no_save and (args.save_every <= 0 or args.num_iterations % args.save_every != 0):
        save_training_checkpoint(out_dir, args.num_iterations, model, projector, args, meta, data_path, rank=ddp_rank)

    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024 if device_type == "cuda" else 0.0
    total_time_min = (time.time() - t_start) / 60
    print0(f"Peak memory usage: {peak_mem:.2f}MiB")
    print0(f"Total training time: {total_time_min:.2f}m")
    wandb_run.log({"gpu/peak_mem_mib": peak_mem, "train/total_time_min": total_time_min})
    wandb_run.finish()
    compute_cleanup()


if __name__ == "__main__":
    main()
