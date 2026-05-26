"""
Minimal nanochat-llava visual instruction trainer.

The trainer freezes SigLIP and trains the projector plus nanochat on visual
instructions.

The code intentionally stays close to nanoVLM/LLaVA: images are encoded on the
fly, the vision encoder is frozen, and the projector uses a higher LR than the
language model.
"""

import argparse
import concurrent.futures
import gc
import io
import json
import math
import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import queue
import random
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import wandb

from nanochat.checkpoint_manager import load_model
from nanochat.common import DummyWandb, autodetect_device_type, compute_cleanup, compute_init, get_base_dir, get_peak_flops, print0
from nanochat.flash_attention import attention_backend_info, require_fa3_varlen
from nanochat.tokenizer import get_token_bytes, get_tokenizer
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
    ensure_hf_nanochat_tokenizer,
    load_vlm_checkpoint,
    render_vision_conversation,
    save_vlm_checkpoint,
)
from scripts.vlm_eval import evaluate_vlm


IMAGE_KEYS = ("image", "image_path", "filename", "path")
DEFAULT_HF_REPO = "HuggingFaceM4/the_cauldron"
DEFAULT_HF_CONFIG = "all"
MODEL_SOURCE = "sft"
UNTRUNCATED_MAX_TOKENS = 1_000_000_000
PROFILE_KEYS = (
    "data_wait",
    "data",
    "image_open",
    "image_processor",
    "image_transfer",
    "siglip_forward",
    "siglip_pool",
    "image_siglip",
    "pack",
    "batch",
    "batch_projector",
    "fwdbwd",
    "optim",
    "optim_projector",
    "optim_llm",
)
IMAGE_PROFILE_KEYS = ("image_open", "image_processor", "image_transfer", "siglip_forward", "siglip_pool")
PROFILE_SUMMARY_KEYS = (
    "data_wait",
    "data",
    "image_siglip",
    "image_open",
    "image_processor",
    "image_transfer",
    "siglip_forward",
    "siglip_pool",
    "pack",
    "batch",
    "batch_projector",
    "fwdbwd",
    "optim",
    "optim_projector",
    "optim_llm",
)
PROFILE_ACCOUNTED_KEYS = ("data_wait", "data", "image_siglip", "pack", "batch", "fwdbwd", "optim")
PACKED_SELECTION_TOKEN_QUANTUM = 512


def add_profile(profile, key, elapsed):
    if profile is not None:
        profile[key] += elapsed


def new_profile():
    return {key: 0.0 for key in PROFILE_KEYS}


def merge_profile(dst, src):
    if src is None:
        return
    for key, value in src.items():
        if key in dst:
            dst[key] += value


def profile_accounted_seconds(profile):
    return sum(float(profile.get(key, 0.0)) for key in PROFILE_ACCOUNTED_KEYS)


def profile_other_seconds(profile, total_seconds):
    return max(0.0, float(total_seconds) - profile_accounted_seconds(profile))


def format_profile_summary(title, profile, total_seconds):
    denom = max(float(total_seconds), 1e-9)
    parts = [f"{title} wall={float(total_seconds):.3f}s"]
    for key in PROFILE_SUMMARY_KEYS:
        value = float(profile.get(key, 0.0))
        parts.append(f"{key}={value:.3f}s/{100 * value / denom:.1f}%")
    other = profile_other_seconds(profile, total_seconds)
    parts.append(f"other={other:.3f}s/{100 * other / denom:.1f}%")
    return " ".join(parts)


def cuda_memory_stats_mib(device_type):
    if device_type != "cuda":
        return {
            "allocated": 0.0,
            "reserved": 0.0,
            "max_allocated": 0.0,
            "max_reserved": 0.0,
        }
    scale = 1024 * 1024
    return {
        "allocated": torch.cuda.memory_allocated() / scale,
        "reserved": torch.cuda.memory_reserved() / scale,
        "max_allocated": torch.cuda.max_memory_allocated() / scale,
        "max_reserved": torch.cuda.max_memory_reserved() / scale,
    }


@dataclass
class PreparedBatch:
    examples: list
    images: list | None = None
    pixel_values: torch.Tensor | None = None
    profile: dict | None = None
    selected_examples: int | None = None


class _PrefetchError:
    def __init__(self, exc):
        self.exc = exc


def unpack_selected_examples(selected):
    if isinstance(selected, tuple) and len(selected) == 2:
        examples, selected_count = selected
        return examples, int(selected_count)
    return selected, len(selected)


class PrefetchIterator:
    def __init__(self, iterable, maxsize):
        self._queue = queue.Queue(maxsize=max(1, int(maxsize)))
        self._sentinel = object()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, args=(iter(iterable),), daemon=True)
        self._thread.start()

    def _put(self, item):
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                pass
        return False

    def _worker(self, iterator):
        try:
            for item in iterator:
                if not self._put(item):
                    break
        except BaseException as exc:
            self._put(_PrefetchError(exc))
        finally:
            self._put(self._sentinel)

    def __iter__(self):
        return self

    def __next__(self):
        item = self._queue.get()
        if item is self._sentinel:
            raise StopIteration
        if isinstance(item, _PrefetchError):
            raise item.exc
        return item

    def close(self):
        self._stop.set()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._thread.join(timeout=2.0)


class PreparedBatchPrefetcher:
    def __init__(
        self,
        select_examples,
        extractor,
        image_root=None,
        skip_bad_images=False,
        profile_timing=False,
        prefetch_processor=True,
        maxsize=2,
        num_workers=2,
    ):
        self._select_examples = select_examples
        self._extractor = extractor
        self._image_root = image_root
        self._skip_bad_images = skip_bad_images
        self._profile_timing = profile_timing
        self._prefetch_processor = prefetch_processor
        self._queue = queue.Queue(maxsize=max(1, int(maxsize)))
        self._sentinel = object()
        self._stop = threading.Event()
        self._inflight = threading.Semaphore(max(1, int(maxsize)))
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(num_workers)))
        self._producer = threading.Thread(target=self._run_producer, daemon=True)
        self._producer.start()

    def _put(self, item):
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                pass
        return False

    def _prepare_selected(self, examples, profile):
        examples, selected_count = unpack_selected_examples(examples)
        images, kept_examples = open_images_for_examples(
            examples,
            image_root=self._image_root,
            skip_bad_images=self._skip_bad_images,
            profile=profile,
        )
        if not images:
            return PreparedBatch(kept_examples, profile=profile, selected_examples=selected_count)
        if self._prefetch_processor:
            pixel_values = self._extractor.preprocess(images, profile=profile)
            return PreparedBatch(kept_examples, pixel_values=pixel_values, profile=profile, selected_examples=selected_count)
        return PreparedBatch(kept_examples, images=images, profile=profile, selected_examples=selected_count)

    def _on_done(self, fut):
        self._inflight.release()
        self._put(fut)

    def _run_producer(self):
        try:
            while not self._stop.is_set():
                self._inflight.acquire()
                if self._stop.is_set():
                    self._inflight.release()
                    break
                profile = new_profile() if self._profile_timing else None
                t = time.perf_counter()
                examples = self._select_examples()
                add_profile(profile, "data", time.perf_counter() - t)
                fut = self._executor.submit(self._prepare_selected, examples, profile)
                fut.add_done_callback(self._on_done)
        except BaseException as exc:
            self._put(_PrefetchError(exc))
        finally:
            self._put(self._sentinel)

    def __iter__(self):
        return self

    def __next__(self):
        item = self._queue.get()
        if item is self._sentinel:
            raise StopIteration
        if isinstance(item, _PrefetchError):
            raise item.exc
        return item.result()

    def close(self):
        self._stop.set()
        self._executor.shutdown(wait=False, cancel_futures=True)
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._producer.join(timeout=2.0)


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


def _resolve_hf_source(args):
    hf_repo = getattr(args, "hf_repo", None) or DEFAULT_HF_REPO
    hf_config = getattr(args, "hf_config", None) or DEFAULT_HF_CONFIG
    return hf_repo, hf_config


def _resolve_hf_configs(hf_repo, hf_config):
    if hf_config == "all":
        from datasets import get_dataset_config_names

        return get_dataset_config_names(hf_repo)
    return [config.strip() for config in hf_config.split(",") if config.strip()]


def _load_hf_stream(load_dataset, hf_repo, hf_config):
    ds = load_dataset(hf_repo, hf_config, split="train", streaming=True)
    try:
        from datasets import Image as HFImage
        from datasets import Sequence

        if "images" in (getattr(ds, "features", None) or {}):
            ds = ds.cast_column("images", Sequence(HFImage(decode=False)))
    except ImportError:
        pass
    return ds


def iter_hf_records(args, seed=None, buffer_size=0):
    hf_repo, hf_config = _resolve_hf_source(args)
    hf_configs = _resolve_hf_configs(hf_repo, hf_config)
    assert hf_configs, f"no HF dataset configs resolved for {hf_repo}/{hf_config}"
    from datasets import load_dataset

    epoch = 0
    while True:
        streams = [_load_hf_stream(load_dataset, hf_repo, config) for config in hf_configs]
        if len(streams) == 1:
            ds = streams[0]
        else:
            from datasets import interleave_datasets

            ds = interleave_datasets(streams, stopping_strategy="all_exhausted")
        if buffer_size > 0 and hasattr(ds, "shuffle"):
            ds = ds.shuffle(seed=None if seed is None else seed + epoch, buffer_size=buffer_size)
        emitted = False
        for rec in ds:
            emitted = True
            yield rec
        assert emitted, f"streamed no records from {data_source_name(args)}"
        epoch += 1


def data_source_name(args):
    hf_repo, hf_config = _resolve_hf_source(args)
    return f"stream:{hf_repo}/{hf_config}"


def load_records(args, val_count=0):
    if args.data_json:
        path = Path(args.data_json)
        records = _load_json(path)
        assert isinstance(records, list), f"expected a JSON list in {path}"
        train_limit = args.max_examples if args.max_examples > 0 else len(records)
        return records[:train_limit], records[train_limit : train_limit + val_count], str(path)

    assert args.max_examples > 0, "HF streaming path should not materialize without --max-examples"
    stream = iter_hf_records(
        args,
        seed=getattr(args, "seed", None),
        buffer_size=max(getattr(args, "stream_buffer_size", 0), 0),
    )
    records = [next(stream) for _ in range(args.max_examples + val_count)]
    train_limit = args.max_examples
    source = data_source_name(args)
    assert records, f"streamed no records from {source}"
    return records[:train_limit], records[train_limit:], f"{source} first {min(len(records), train_limit):,} train rows"


def _image_value(record):
    images = record.get("images")
    if isinstance(images, list):
        if len(images) == 1:
            return images[0]
        return None
    if images is not None:
        return images
    for key in IMAGE_KEYS:
        value = record.get(key)
        if value is not None:
            return value
    return None


def _resolve_local_image_path(path_value, image_root=None):
    path = Path(str(path_value))
    candidates = [path] if path.is_absolute() else []
    if not path.is_absolute() and image_root:
        root = Path(image_root)
        candidates.extend([root / path, root / "images" / path, root / path.name])
    if not candidates:
        candidates.append(path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def image_record_is_openable(record, image_root=None):
    value = _image_value(record)
    if value is None:
        return False
    if hasattr(value, "convert"):
        return True
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return True
        path = value.get("path")
        return path is not None and _resolve_local_image_path(path, image_root=image_root) is not None
    if isinstance(value, (str, os.PathLike)):
        return image_root is None or _resolve_local_image_path(value, image_root=image_root) is not None
    return True


def open_image(record, image_root=None, profile=None):
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
            path = _resolve_local_image_path(value["path"], image_root=image_root)
            if path is None:
                raise FileNotFoundError(value["path"])
            with Image.open(path) as img:
                image = img.convert("RGB")
        else:
            raise ValueError("HF image dict has neither bytes nor path")
        add_profile(profile, "image_open", time.perf_counter() - t)
        return image

    candidate = _resolve_local_image_path(value, image_root=image_root)
    if candidate is not None:
        t = time.perf_counter()
        with Image.open(candidate) as img:
            image = img.convert("RGB")
        add_profile(profile, "image_open", time.perf_counter() - t)
        return image
    raise FileNotFoundError(f"could not find image {value!r} under {image_root!r}")


def expanded_input_len(tokens):
    image_count = count_image_tokens(tokens[:-1])
    return len(tokens) - 1 + image_count * (VISION_TOKENS - 1)


def packed_expanded_len(examples, boundary_aware=False):
    assert examples, "cannot pack an empty example group"
    boundary_inputs = 0 if boundary_aware else len(examples) - 1
    return sum(ex["expanded_len"] for ex in examples) + boundary_inputs


def parse_bucket_lens(text: str, max_seq_len: int) -> list[int]:
    if not text:
        return []
    buckets = sorted({int(part) for part in text.split(",") if part.strip()})
    if not buckets:
        return []
    if buckets[0] <= 0:
        raise ValueError("bucket lengths must be positive")
    if buckets[-1] > max_seq_len:
        raise ValueError(f"largest bucket {buckets[-1]} exceeds --max-seq-len {max_seq_len}")
    if buckets[-1] < max_seq_len:
        buckets.append(max_seq_len)
    return buckets


def bucketed_len(length: int, bucket_lens: list[int] | None) -> int:
    if not bucket_lens:
        return length
    for bucket in bucket_lens:
        if length <= bucket:
            return bucket
    return bucket_lens[-1]


def one_image_text_capacity(bucket_len: int) -> int:
    return max(0, int(bucket_len) - VISION_TOKENS)


def causal_attention_pairs(rows: int, seq_len: int) -> int:
    seq_len = int(seq_len)
    return int(rows) * seq_len * (seq_len + 1) // 2


def causal_attention_pairs_for_lengths(lengths) -> int:
    return sum(int(length) * (int(length) + 1) // 2 for length in lengths)


def count_near_cap_segments(lengths, segment_cap_len: int, near_frac: float = 0.95) -> tuple[int, int]:
    cap = int(segment_cap_len or 0)
    if cap <= 0:
        return 0, 0
    threshold = math.ceil(cap * near_frac)
    near_cap = sum(1 for length in lengths if int(length) >= threshold)
    cap_hits = sum(1 for length in lengths if int(length) >= cap)
    return near_cap, cap_hits


def segment_length_percentile(lengths, q: float) -> int:
    values = sorted(int(length) for length in lengths)
    if not values:
        return 0
    return _percentile(values, q)


def add_segment_length_counts(counts: dict, lengths):
    for length in lengths or []:
        length = int(length)
        counts[length] = counts.get(length, 0) + 1


def segment_length_percentile_from_counts(counts: dict, q: float) -> int:
    total = sum(int(count) for count in counts.values())
    if total <= 0:
        return 0
    target = min(total - 1, max(0, round(q * (total - 1))))
    seen = 0
    for length, count in sorted(counts.items()):
        seen += int(count)
        if seen > target:
            return int(length)
    return int(max(counts))


def packed_segment_expanded_lengths(examples, group, boundary_aware=False) -> list[int]:
    lengths = []
    for pos, idx in enumerate(group):
        extra_boundary_input = 0 if boundary_aware else int(pos < len(group) - 1)
        lengths.append(int(examples[idx]["expanded_len"]) + extra_boundary_input)
    return lengths


def packed_attention_pairs(examples, groups, lengths, bucket_lens=None, boundary_aware=False) -> int:
    if boundary_aware:
        segment_lengths = [
            length
            for group in groups
            for length in packed_segment_expanded_lengths(examples, group, boundary_aware=True)
        ]
        return causal_attention_pairs_for_lengths(segment_lengths)
    bucket = bucketed_len(max(lengths), bucket_lens)
    return causal_attention_pairs(len(lengths), bucket)


def packed_budget_tokens(lengths, bucket_lens=None, compact_token_budget=False) -> int:
    if compact_token_budget:
        return sum(lengths)
    return bucketed_len(max(lengths), bucket_lens) * len(lengths)


def packed_selection_token_score(useful_tokens, compact_token_budget=False) -> int:
    if compact_token_budget:
        return (int(useful_tokens) + PACKED_SELECTION_TOKEN_QUANTUM - 1) // PACKED_SELECTION_TOKEN_QUANTUM
    return int(useful_tokens)


def format_count(value: float) -> str:
    value = float(value)
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.0f}"


def counted_iter(iterable, counter: dict, key: str):
    for item in iterable:
        counter[key] = counter.get(key, 0) + 1
        yield item


def _percentile(sorted_values: list[int], q: float) -> int:
    assert sorted_values, "cannot compute percentile of empty values"
    idx = min(len(sorted_values) - 1, max(0, round(q * (len(sorted_values) - 1))))
    return sorted_values[idx]


def summarize_length_stats(lengths: list[int], scanned: int, max_seq_len: int, bucket_lens=None, max_batch_tokens: int = 0):
    values = sorted(int(x) for x in lengths)
    if not values:
        return {
            "scanned": scanned,
            "usable": 0,
            "fit_count": 0,
            "fit_frac": 0.0,
            "buckets": [],
        }
    summary = {
        "scanned": scanned,
        "usable": len(values),
        "min": values[0],
        "mean": sum(values) / len(values),
        "p50": _percentile(values, 0.50),
        "p80": _percentile(values, 0.80),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": values[-1],
        "fit_count": sum(length <= max_seq_len for length in values),
        "fit_frac": sum(length <= max_seq_len for length in values) / len(values),
        "buckets": [],
    }
    if bucket_lens:
        prev = 0
        cumulative = 0
        for bucket in bucket_lens:
            in_bucket = [length for length in values if prev < length <= bucket]
            cumulative += len(in_bucket)
            avg_len = (sum(in_bucket) / len(in_bucket)) if in_bucket else 0.0
            avg_pad_frac = 1.0 - (sum(in_bucket) / max(len(in_bucket) * bucket, 1)) if in_bucket else 0.0
            summary["buckets"].append({
                "bucket": bucket,
                "count": len(in_bucket),
                "frac": len(in_bucket) / len(values),
                "cumulative": cumulative,
                "cumulative_frac": cumulative / len(values),
                "avg_len": avg_len,
                "avg_pad_frac": avg_pad_frac,
                "text_cap_1img": one_image_text_capacity(bucket),
                "rows_at_token_cap": (max_batch_tokens // bucket) if max_batch_tokens > 0 else 0,
            })
            prev = bucket
        overflow = [length for length in values if length > bucket_lens[-1]]
        if overflow:
            summary["overflow"] = len(overflow)
    return summary


def format_length_stats(summary, source: str, max_seq_len: int, bucket_lens=None, max_batch_tokens: int = 0) -> str:
    lines = [
        f"Length stats source={source}",
        f"records_scanned={summary['scanned']:,} usable_one_image_examples={summary['usable']:,}",
    ]
    if summary.get("elapsed"):
        rate = summary["usable"] / max(summary["elapsed"], 1e-9)
        lines.append(f"scan_elapsed={summary['elapsed']:.1f}s usable_examples/sec={rate:.1f}")
    if summary["usable"] == 0:
        lines.append("No usable one-image supervised examples were found.")
        return "\n".join(lines)
    lines.append(
        "expanded_len min/p50/p80/p90/p95/p99/max/mean "
        f"{summary['min']:,}/{summary['p50']:,}/{summary['p80']:,}/{summary['p90']:,}/"
        f"{summary['p95']:,}/{summary['p99']:,}/{summary['max']:,}/{summary['mean']:.1f}"
    )
    lines.append(
        f"fit_at_max_seq_len_{max_seq_len:,}={summary['fit_count']:,}/{summary['usable']:,} "
        f"({100 * summary['fit_frac']:.1f}%)"
    )
    if bucket_lens:
        rows_text = "rows@cap" if max_batch_tokens > 0 else "rows@cap(n/a)"
        lines.append(f"bucket | count | pct | cumulative | avg_len | avg_pad | text_cap_1img | {rows_text}")
        for row in summary["buckets"]:
            rows_at_cap = row["rows_at_token_cap"] if max_batch_tokens > 0 else "-"
            lines.append(
                f"{row['bucket']:>6} | {row['count']:>5} | {100 * row['frac']:>5.1f}% | "
                f"{row['cumulative']:>6} ({100 * row['cumulative_frac']:>5.1f}%) | "
                f"{row['avg_len']:>7.1f} | {100 * row['avg_pad_frac']:>6.1f}% | "
                f"{row['text_cap_1img']:>13} | {rows_at_cap}"
            )
        if summary.get("overflow", 0):
            lines.append(f"overflow_above_largest_bucket={summary['overflow']:,}")
    return "\n".join(lines)


def add_bucket_steady_step(
    bucket_stats: dict,
    bucket: int,
    elapsed: float,
    tokens: int,
    padded_tokens: int,
    samples: int,
    seq_flops: float = 0.0,
    seq_padded_flops: float = 0.0,
    attention_pairs: int = 0,
    segments: int = 0,
    segment_lengths=None,
    max_segment_len: int = 0,
    near_cap_segments: int = 0,
    cap_segments: int = 0,
):
    row = bucket_stats.setdefault(
        bucket,
        {
            "seconds": 0.0,
            "tokens": 0,
            "padded_tokens": 0,
            "samples": 0,
            "steps": 0,
            "seq_flops": 0.0,
            "seq_padded_flops": 0.0,
            "attention_pairs": 0,
            "segments": 0,
            "segment_length_counts": {},
            "max_segment_len": 0,
            "near_cap_segments": 0,
            "cap_segments": 0,
        },
    )
    row["seconds"] += elapsed
    row["tokens"] += int(tokens)
    row["padded_tokens"] += int(padded_tokens)
    row["samples"] += int(samples)
    row["seq_flops"] += float(seq_flops)
    row["seq_padded_flops"] += float(seq_padded_flops)
    row["attention_pairs"] += int(attention_pairs)
    row["segments"] += int(segments)
    add_segment_length_counts(row["segment_length_counts"], segment_lengths)
    row["max_segment_len"] = max(int(row["max_segment_len"]), int(max_segment_len))
    row["near_cap_segments"] += int(near_cap_segments)
    row["cap_segments"] += int(cap_segments)
    row["steps"] += 1


def bucket_steady_metrics(row: dict, num_flops_per_token: float, gpu_peak_flops: float):
    seconds = max(row["seconds"], 1e-9)
    tokens_per_sec = row["tokens"] / seconds
    padded_tokens_per_sec = row["padded_tokens"] / seconds
    seq_mfu = 100 * row.get("seq_flops", 0.0) / seconds / gpu_peak_flops
    seq_padded_mfu = 100 * row.get("seq_padded_flops", 0.0) / seconds / gpu_peak_flops
    return {
        "steps": row["steps"],
        "tokens_per_sec": tokens_per_sec,
        "padded_tokens_per_sec": padded_tokens_per_sec,
        "samples_per_sec": row["samples"] / seconds,
        "mfu": seq_mfu,
        "padded_mfu": seq_padded_mfu,
        "token_estimate_mfu": 100 * num_flops_per_token * tokens_per_sec / gpu_peak_flops,
        "token_estimate_padded_mfu": 100 * num_flops_per_token * padded_tokens_per_sec / gpu_peak_flops,
        "seq_mfu": seq_mfu,
        "seq_padded_mfu": seq_padded_mfu,
        "padding_frac": 1.0 - (row["tokens"] / max(row["padded_tokens"], 1)),
        "attention_pairs_per_step": row.get("attention_pairs", 0) / max(row["steps"], 1),
        "attention_pairs_per_token": row.get("attention_pairs", 0) / max(row["tokens"], 1),
        "segments_per_step": row.get("segments", 0) / max(row["steps"], 1),
        "avg_segment_len": row["tokens"] / max(row.get("segments", 0), 1),
        "p50_segment_len": segment_length_percentile_from_counts(row.get("segment_length_counts", {}), 0.50),
        "p90_segment_len": segment_length_percentile_from_counts(row.get("segment_length_counts", {}), 0.90),
        "max_segment_len": row.get("max_segment_len", 0),
        "near_cap_segments_per_step": row.get("near_cap_segments", 0) / max(row["steps"], 1),
        "cap_segments_per_step": row.get("cap_segments", 0) / max(row["steps"], 1),
    }


def format_bucket_steady_line(bucket: int, metrics: dict) -> str:
    return (
        f"  bucket {bucket:,} | steps {metrics['steps']} | tokens/sec {metrics['tokens_per_sec']:.0f} | "
        f"padded_tokens/sec {metrics['padded_tokens_per_sec']:.0f} | samples/sec {metrics['samples_per_sec']:.2f} | "
        f"steady_mfu {metrics['mfu']:.2f} | steady_padded_mfu {metrics['padded_mfu']:.2f} | "
        f"steady_seq_padded_mfu {metrics['seq_padded_mfu']:.2f} | "
        f"pad {100 * metrics['padding_frac']:.1f}% | "
        f"attn_pairs/token {metrics['attention_pairs_per_token']:.2f} | "
        f"avg_segment {metrics['avg_segment_len']:.1f} | "
        f"p50_segment {metrics['p50_segment_len']} | p90_segment {metrics['p90_segment_len']} | "
        f"max_segment {metrics['max_segment_len']} | "
        f"near_cap/step {metrics['near_cap_segments_per_step']:.1f} | "
        f"cap_hits/step {metrics['cap_segments_per_step']:.1f}"
    )


def should_count_mfu_step(step: int, global_warmup_steps: int, step_bucket: int, bucket_seen_before: int, bucket_warmup_steps: int) -> bool:
    if step <= global_warmup_steps:
        return False
    if step_bucket and bucket_seen_before < bucket_warmup_steps:
        return False
    return True


def static_mfu_step_bucket(seq_lens, compact_varlen: bool = False) -> int:
    if compact_varlen:
        return 0
    if seq_lens and len(set(seq_lens)) == 1:
        return int(seq_lens[0])
    return 0


def collect_length_stats(records, tokenizer, target_examples: int, max_records: int = 0):
    lengths = []
    scanned = 0
    for rec in records:
        scanned += 1
        ex = render_record(rec, tokenizer, UNTRUNCATED_MAX_TOKENS, require_openable_image=False)
        if ex is not None:
            lengths.append(ex["expanded_len"])
            if len(lengths) >= target_examples:
                break
        if max_records > 0 and scanned >= max_records:
            break
    return {"lengths": lengths, "scanned": scanned}


def run_length_stats(args, tokenizer, bucket_lens):
    t0 = time.perf_counter()
    if args.data_json:
        records = _load_json(Path(args.data_json))
        assert isinstance(records, list), f"expected a JSON list in {args.data_json}"
        source = str(args.data_json)
    else:
        records = iter_hf_records(
            args,
            seed=getattr(args, "seed", None),
            buffer_size=max(getattr(args, "stream_buffer_size", 0), 0),
        )
        source = data_source_name(args)
    collected = collect_length_stats(
        records,
        tokenizer,
        target_examples=args.length_stats_examples,
        max_records=args.length_stats_max_records,
    )
    summary = summarize_length_stats(
        collected["lengths"],
        scanned=collected["scanned"],
        max_seq_len=args.max_seq_len,
        bucket_lens=bucket_lens,
        max_batch_tokens=args.max_batch_tokens,
    )
    summary["elapsed"] = time.perf_counter() - t0
    print0(format_length_stats(summary, source, args.max_seq_len, bucket_lens, args.max_batch_tokens), flush=True)


def batch_plan_row(
    examples,
    bucket_lens,
    batch_size,
    max_batch_tokens=0,
    max_seq_len=None,
    max_images_per_row=1,
    fixed_rows=0,
    boundary_aware_pack=False,
    flatten_packed_batch=False,
):
    selected_examples = len(examples)
    if max_seq_len is None:
        groups = [[i] for i in range(len(examples))]
        lengths = [int(ex["expanded_len"]) for ex in examples]
    elif flatten_packed_batch and fixed_rows == 0:
        groups = []
        lengths = []
        total_len = 0
        for idx, example in enumerate(examples):
            length = int(example["expanded_len"])
            if length > max_seq_len:
                continue
            if max_batch_tokens > 0 and total_len + length > max_batch_tokens:
                continue
            groups.append([idx])
            lengths.append(length)
            total_len += length
        if not groups:
            return None
    else:
        groups, lengths = pack_example_groups(
            examples,
            max_seq_len=max_seq_len,
            max_batch_tokens=max_batch_tokens,
            max_images_per_row=max_images_per_row,
            fixed_rows=fixed_rows,
            bucket_lens=bucket_lens,
            boundary_aware=boundary_aware_pack,
            compact_token_budget=flatten_packed_batch,
        )
        if not groups or (fixed_rows > 0 and len(groups) != fixed_rows):
            return None
    packed_examples = sum(len(group) for group in groups)
    if boundary_aware_pack:
        segment_lengths = [
            length
            for group in groups
            for length in packed_segment_expanded_lengths(examples, group, boundary_aware=True)
        ]
    else:
        segment_lengths = lengths
    segment_count = len(segment_lengths)
    max_segment_len = max(segment_lengths) if segment_lengths else 0
    segment_cap_len = int(max_seq_len or 0)
    near_cap_segments, cap_segments = count_near_cap_segments(segment_lengths, segment_cap_len)
    if flatten_packed_batch:
        useful_tokens = sum(lengths)
        attention_pairs = packed_attention_pairs(
            examples,
            groups,
            lengths,
            bucket_lens=bucket_lens,
            boundary_aware=boundary_aware_pack,
        )
        return {
            "bucket": useful_tokens,
            "rows": 1,
            "examples": packed_examples,
            "selected_examples": selected_examples,
            "dropped_examples": selected_examples - packed_examples,
            "target_rows": 1,
            "useful_tokens": useful_tokens,
            "padded_tokens": useful_tokens,
            "attention_pairs": attention_pairs,
            "segments": segment_count,
            "segment_lengths": segment_lengths,
            "max_segment_len": max_segment_len,
            "near_cap_segments": near_cap_segments,
            "cap_segments": cap_segments,
            "segment_cap_len": segment_cap_len,
            "max_len": useful_tokens,
            "min_len": useful_tokens,
            "fill_frac": 1.0,
            "pad_frac": 0.0,
        }
    max_len = max(lengths)
    bucket = bucketed_len(max_len, bucket_lens)
    useful_tokens = sum(lengths)
    padded_tokens = bucket * len(lengths)
    attention_pairs = packed_attention_pairs(
        examples,
        groups,
        lengths,
        bucket_lens=bucket_lens,
        boundary_aware=boundary_aware_pack,
    )
    target_rows = _target_bucket_rows(bucket, batch_size, max_batch_tokens)
    return {
        "bucket": bucket,
        "rows": len(lengths),
        "examples": packed_examples,
        "selected_examples": selected_examples,
        "dropped_examples": selected_examples - packed_examples,
        "target_rows": target_rows,
        "useful_tokens": useful_tokens,
        "padded_tokens": padded_tokens,
        "attention_pairs": attention_pairs,
        "segments": segment_count,
        "segment_lengths": segment_lengths,
        "max_segment_len": max_segment_len,
        "near_cap_segments": near_cap_segments,
        "cap_segments": cap_segments,
        "segment_cap_len": segment_cap_len,
        "max_len": max_len,
        "min_len": min(lengths),
        "fill_frac": len(lengths) / max(target_rows, 1),
        "pad_frac": 1.0 - (useful_tokens / max(padded_tokens, 1)),
    }


def summarize_optimizer_step_groups(rows, grad_accum_steps=1):
    group_size = max(1, int(grad_accum_steps))
    if group_size <= 1:
        return None
    groups = [rows[i:i + group_size] for i in range(0, len(rows), group_size)]
    total = {
        "grad_accum_steps": group_size,
        "groups": len(groups),
        "complete_steps": 0,
        "incomplete_steps": 0,
        "same_bucket_steps": 0,
        "mixed_bucket_steps": 0,
        "rows": 0,
        "examples": 0,
        "dropped_examples": 0,
        "useful_tokens": 0,
        "padded_tokens": 0,
        "attention_pairs": 0,
        "segments": 0,
        "max_segment_len": 0,
    }
    by_bucket = {}
    for group in groups:
        if len(group) != group_size:
            total["incomplete_steps"] += 1
            continue
        total["complete_steps"] += 1
        group_buckets = {row["bucket"] for row in group}
        group_rows = sum(row["rows"] for row in group)
        group_examples = sum(row["examples"] for row in group)
        group_dropped = sum(row["dropped_examples"] for row in group)
        group_useful = sum(row["useful_tokens"] for row in group)
        group_padded = sum(row["padded_tokens"] for row in group)
        group_attention_pairs = sum(row["attention_pairs"] for row in group)
        total["rows"] += group_rows
        total["examples"] += group_examples
        total["dropped_examples"] += group_dropped
        total["useful_tokens"] += group_useful
        total["padded_tokens"] += group_padded
        total["attention_pairs"] += group_attention_pairs
        if len(group_buckets) == 1:
            total["same_bucket_steps"] += 1
            bucket = next(iter(group_buckets))
            bucket_row = by_bucket.setdefault(bucket, {
                "steps": 0,
                "rows": 0,
                "examples": 0,
                "dropped_examples": 0,
                "useful_tokens": 0,
                "padded_tokens": 0,
                "attention_pairs": 0,
            })
            bucket_row["steps"] += 1
            bucket_row["rows"] += group_rows
            bucket_row["examples"] += group_examples
            bucket_row["dropped_examples"] += group_dropped
            bucket_row["useful_tokens"] += group_useful
            bucket_row["padded_tokens"] += group_padded
            bucket_row["attention_pairs"] += group_attention_pairs
        else:
            total["mixed_bucket_steps"] += 1
    return {"total": total, "buckets": by_bucket}


def summarize_batch_plan(rows, grad_accum_steps=1):
    by_bucket = {}
    total = {
        "steps": 0,
        "rows": 0,
        "examples": 0,
        "selected_examples": 0,
        "dropped_examples": 0,
        "useful_tokens": 0,
        "padded_tokens": 0,
        "attention_pairs": 0,
        "segments": 0,
        "segment_lengths": [],
        "max_segment_len": 0,
        "near_cap_segments": 0,
        "cap_segments": 0,
    }
    for row in rows:
        total["steps"] += 1
        total["rows"] += row["rows"]
        total["examples"] += row["examples"]
        total["selected_examples"] += row["selected_examples"]
        total["dropped_examples"] += row["dropped_examples"]
        total["useful_tokens"] += row["useful_tokens"]
        total["padded_tokens"] += row["padded_tokens"]
        total["attention_pairs"] += row["attention_pairs"]
        total["segments"] += row.get("segments", 0)
        total["segment_lengths"].extend(row.get("segment_lengths", []))
        total["max_segment_len"] = max(total["max_segment_len"], row.get("max_segment_len", 0))
        total["near_cap_segments"] += row.get("near_cap_segments", 0)
        total["cap_segments"] += row.get("cap_segments", 0)
        bucket = by_bucket.setdefault(row["bucket"], {
            "steps": 0,
            "rows": 0,
            "examples": 0,
            "selected_examples": 0,
            "dropped_examples": 0,
            "target_rows": row["target_rows"],
            "useful_tokens": 0,
            "padded_tokens": 0,
            "attention_pairs": 0,
            "segments": 0,
            "segment_lengths": [],
            "max_segment_len": 0,
            "near_cap_segments": 0,
            "cap_segments": 0,
            "segment_cap_len": row.get("segment_cap_len", 0),
            "min_rows": None,
            "max_rows": 0,
        })
        bucket["steps"] += 1
        bucket["rows"] += row["rows"]
        bucket["examples"] += row["examples"]
        bucket["selected_examples"] += row["selected_examples"]
        bucket["dropped_examples"] += row["dropped_examples"]
        bucket["target_rows"] = row["target_rows"]
        bucket["useful_tokens"] += row["useful_tokens"]
        bucket["padded_tokens"] += row["padded_tokens"]
        bucket["attention_pairs"] += row["attention_pairs"]
        bucket["segments"] += row.get("segments", 0)
        bucket["segment_lengths"].extend(row.get("segment_lengths", []))
        bucket["max_segment_len"] = max(bucket["max_segment_len"], row.get("max_segment_len", 0))
        bucket["near_cap_segments"] += row.get("near_cap_segments", 0)
        bucket["cap_segments"] += row.get("cap_segments", 0)
        bucket["segment_cap_len"] = max(bucket.get("segment_cap_len", 0), row.get("segment_cap_len", 0))
        bucket["min_rows"] = row["rows"] if bucket["min_rows"] is None else min(bucket["min_rows"], row["rows"])
        bucket["max_rows"] = max(bucket["max_rows"], row["rows"])
    return {
        "total": total,
        "buckets": by_bucket,
        "optimizer_steps": summarize_optimizer_step_groups(rows, grad_accum_steps=grad_accum_steps),
    }


def format_batch_plan(summary, source, args, bucket_lens, elapsed=None, records_scanned=None, rendered_examples=None):
    total = summary["total"]
    lines = [
        f"Batch plan source={source}",
        (
            f"steps={total['steps']:,} batch_size={args.device_batch_size:,} max_batch_tokens={args.max_batch_tokens:,} "
            f"grad_accum_steps={getattr(args, 'grad_accum_steps', 1)} "
            f"bucket_lens={bucket_lens or 'none'} bucket_selection={args.bucket_selection} "
            f"bucket_min_fill_frac={args.bucket_min_fill_frac:g} bucket_cycle_repeat={args.bucket_cycle_repeat} "
            f"pack_examples={args.pack_examples} boundary_aware_pack={getattr(args, 'boundary_aware_pack', False)} "
            f"flatten_packed_batch={getattr(args, 'flatten_packed_batch', False)}"
        ),
    ]
    if elapsed is not None:
        timing = f"planning_elapsed={elapsed:.1f}s"
        if records_scanned is not None:
            timing += f" records_scanned={records_scanned:,}"
        if rendered_examples is not None:
            rate = rendered_examples / max(elapsed, 1e-9)
            timing += f" rendered_examples={rendered_examples:,} rendered_examples/sec={rate:.1f}"
        lines.append(timing)
    if total["steps"] == 0:
        lines.append("No batches were planned.")
        return "\n".join(lines)
    total_pad = 1.0 - (total["useful_tokens"] / max(total["padded_tokens"], 1))
    total_segment_p50 = segment_length_percentile(total.get("segment_lengths", []), 0.50)
    total_segment_p90 = segment_length_percentile(total.get("segment_lengths", []), 0.90)
    lines.append(
        f"overall rows/step={total['rows'] / total['steps']:.1f} "
        f"examples/step={total['examples'] / total['steps']:.1f} "
        f"dropped/step={total['dropped_examples'] / total['steps']:.1f} "
        f"tokens/step={total['useful_tokens'] / total['steps']:.0f}/{total['padded_tokens'] / total['steps']:.0f} "
        f"pad={100 * total_pad:.1f}% "
        f"attn_pairs/step={format_count(total['attention_pairs'] / total['steps'])} "
        f"attn_pairs/token={total['attention_pairs'] / max(total['useful_tokens'], 1):.2f} "
        f"segments/step={total['segments'] / total['steps']:.1f} "
        f"avg_segment={total['useful_tokens'] / max(total['segments'], 1):.1f} "
        f"p50_segment={total_segment_p50} "
        f"p90_segment={total_segment_p90} "
        f"max_segment={total['max_segment_len']} "
        f"near_cap/step={total['near_cap_segments'] / total['steps']:.1f} "
        f"cap_hits/step={total['cap_segments'] / total['steps']:.1f}"
    )
    optimizer_steps = summary.get("optimizer_steps")
    if optimizer_steps is not None:
        opt_total = optimizer_steps["total"]
        denom = max(opt_total["complete_steps"], 1)
        opt_pad = 1.0 - (opt_total["useful_tokens"] / max(opt_total["padded_tokens"], 1))
        lines.append(
            f"optimizer_steps complete={opt_total['complete_steps']}/{opt_total['groups']} "
            f"incomplete={opt_total['incomplete_steps']} same_bucket={opt_total['same_bucket_steps']} "
            f"mixed_bucket={opt_total['mixed_bucket_steps']} "
            f"tokens/optimizer_step={opt_total['useful_tokens'] / denom:.0f}/{opt_total['padded_tokens'] / denom:.0f} "
            f"pad={100 * opt_pad:.1f}% "
            f"attn_pairs/optimizer_step={format_count(opt_total['attention_pairs'] / denom)} "
            f"attn_pairs/token={opt_total['attention_pairs'] / max(opt_total['useful_tokens'], 1):.2f}"
        )
        if optimizer_steps["buckets"]:
            lines.append("optimizer bucket | steps | rows/step | examples/step | tokens/optimizer_step useful/padded | pad | attn_pairs/optimizer_step | attn_pairs/token")
            for bucket, row in sorted(optimizer_steps["buckets"].items()):
                denom = max(row["steps"], 1)
                pad = 1.0 - (row["useful_tokens"] / max(row["padded_tokens"], 1))
                lines.append(
                    f"{bucket:>16} | {row['steps']:>5} | {row['rows'] / denom:>9.1f} | "
                    f"{row['examples'] / denom:>13.1f} | {row['useful_tokens'] / denom:>7.0f}/{row['padded_tokens'] / denom:<7.0f} | "
                    f"{100 * pad:>5.1f}% | {format_count(row['attention_pairs'] / denom):>24} | "
                    f"{row['attention_pairs'] / max(row['useful_tokens'], 1):>16.2f}"
                )
    lines.append("bucket | steps | rows avg/min/max | examples avg/dropped | segments avg | segment_len avg/p50/p90/max | near_cap/cap avg | target_rows | avg_fill | tokens/step useful/padded | pad | attn_pairs/step | attn_pairs/token")
    for bucket, row in sorted(summary["buckets"].items()):
        avg_rows = row["rows"] / row["steps"]
        avg_examples = row["examples"] / row["steps"]
        avg_dropped = row["dropped_examples"] / row["steps"]
        avg_segments = row["segments"] / row["steps"]
        avg_segment_len = row["useful_tokens"] / max(row["segments"], 1)
        segment_p50 = segment_length_percentile(row.get("segment_lengths", []), 0.50)
        segment_p90 = segment_length_percentile(row.get("segment_lengths", []), 0.90)
        avg_near_cap = row["near_cap_segments"] / row["steps"]
        avg_cap_hits = row["cap_segments"] / row["steps"]
        avg_useful = row["useful_tokens"] / row["steps"]
        avg_padded = row["padded_tokens"] / row["steps"]
        avg_attention_pairs = row["attention_pairs"] / row["steps"]
        pad = 1.0 - (row["useful_tokens"] / max(row["padded_tokens"], 1))
        fill = avg_rows / max(row["target_rows"], 1)
        lines.append(
            f"{bucket:>6} | {row['steps']:>5} | {avg_rows:>5.1f}/{row['min_rows']:>3}/{row['max_rows']:<3} | "
            f"{avg_examples:>6.1f}/{avg_dropped:<7.1f} | "
            f"{avg_segments:>12.1f} | {avg_segment_len:>9.1f}/{segment_p50:<3}/{segment_p90:<3}/{row['max_segment_len']:<7} | "
            f"{avg_near_cap:>7.1f}/{avg_cap_hits:<7.1f} | "
            f"{row['target_rows']:>11} | {100 * fill:>7.1f}% | {avg_useful:>7.0f}/{avg_padded:<7.0f} | "
            f"{100 * pad:>5.1f}% | {format_count(avg_attention_pairs):>15} | "
            f"{row['attention_pairs'] / max(row['useful_tokens'], 1):>16.2f}"
        )
    return "\n".join(lines)


def run_batch_plan(args, tokenizer, bucket_lens):
    t0 = time.perf_counter()
    rng = random.Random(args.seed)
    records_scanned = None
    rendered_examples = None
    if args.data_json:
        records = _load_json(Path(args.data_json))
        assert isinstance(records, list), f"expected a JSON list in {args.data_json}"
        examples = render_records(records, tokenizer, args.max_seq_len, image_root=args.image_root, require_openable_image=False)
        cursor = 0
        source = str(args.data_json)
        records_scanned = len(records)
        rendered_examples = len(examples)
        materialized_batch_buffer = []
        materialized_batch_buffer_size = (
            args.batch_buffer_size
            if args.batch_buffer_size > 0
            else max(args.device_batch_size * args.grad_accum_steps * 4, args.device_batch_size)
        )

        def select_batch():
            nonlocal cursor
            if args.pack_examples > 1:
                batch, cursor = next_materialized_packed_batch(
                    examples,
                    args.device_batch_size,
                    cursor,
                    rng,
                    max_batch_tokens=args.max_batch_tokens,
                    batch_buffer=materialized_batch_buffer,
                    batch_buffer_size=materialized_batch_buffer_size,
                    bucket_lens=bucket_lens,
                    bucket_selection=args.bucket_selection,
                    pack_max_seq_len=pack_max_seq_len,
                    max_images_per_row=args.pack_examples,
                    pack_fixed_rows=args.pack_fixed_rows,
                    boundary_aware_pack=args.boundary_aware_pack,
                    flatten_packed_batch=args.flatten_packed_batch,
                )
            else:
                batch, cursor = next_batch(
                    examples,
                    args.device_batch_size,
                    cursor,
                    rng,
                    max_batch_tokens=args.max_batch_tokens,
                    bucket_lens=bucket_lens,
                )
            return batch

    else:
        counters = {"records": 0, "rendered": 0}
        record_iter = counted_iter(
            iter_hf_records(args, seed=args.seed, buffer_size=max(args.stream_buffer_size, args.device_batch_size)),
            counters,
            "records",
        )
        example_iter = counted_iter(
            iter_rendered_examples(record_iter, tokenizer, args.max_seq_len, image_root=args.image_root, require_openable_image=False),
            counters,
            "rendered",
        )
        pending = None
        buffer = []
        bucket_state = {}
        batch_buffer_size = args.batch_buffer_size if args.batch_buffer_size > 0 else max(args.device_batch_size * args.grad_accum_steps * 4, args.device_batch_size)
        source = data_source_name(args)

        def select_batch():
            nonlocal pending
            batch, pending = next_stream_batch(
                example_iter,
                args.device_batch_size,
                pending=pending,
                max_batch_tokens=args.max_batch_tokens,
                buffer=buffer,
                batch_buffer_size=batch_buffer_size,
                rng=rng,
                bucket_lens=bucket_lens,
                bucket_selection=args.bucket_selection,
                bucket_state=bucket_state,
                bucket_min_fill_frac=args.bucket_min_fill_frac,
                bucket_cycle_repeat=args.bucket_cycle_repeat,
                pack_max_seq_len=pack_max_seq_len,
                max_images_per_row=args.pack_examples,
                pack_fixed_rows=args.pack_fixed_rows,
                boundary_aware_pack=args.boundary_aware_pack,
                flatten_packed_batch=args.flatten_packed_batch,
            )
            return batch

    rows = []
    pack_max_seq_len = min(args.max_seq_len, args.pack_max_seq_len or args.max_seq_len)
    for _ in range(args.batch_plan_steps):
        batch = select_batch()
        if batch:
            row = batch_plan_row(
                batch,
                bucket_lens,
                args.device_batch_size,
                max_batch_tokens=args.max_batch_tokens,
                max_seq_len=pack_max_seq_len,
                max_images_per_row=args.pack_examples,
                fixed_rows=args.pack_fixed_rows,
                boundary_aware_pack=args.boundary_aware_pack,
                flatten_packed_batch=args.flatten_packed_batch,
            )
            if row is not None:
                rows.append(row)
    if not args.data_json:
        records_scanned = counters["records"]
        rendered_examples = counters["rendered"]
    print0(
        format_batch_plan(
            summarize_batch_plan(rows, grad_accum_steps=args.grad_accum_steps),
            source,
            args,
            bucket_lens,
            elapsed=time.perf_counter() - t0,
            records_scanned=records_scanned,
            rendered_examples=rendered_examples,
        ),
        flush=True,
    )


def exit_after_cpu_report():
    sys.stdout.flush()
    sys.stderr.flush()
    # HF streaming can leave non-daemon background state alive after the report.
    # These report modes run before model/GPU setup and have no cleanup to save.
    os._exit(0)


def _dense_bucket_indices(candidates, buffer, bucket, batch_size, max_batch_tokens=0):
    return candidates[-_target_bucket_rows(bucket, batch_size, max_batch_tokens):]


def _target_bucket_rows(bucket, batch_size, max_batch_tokens=0):
    if max_batch_tokens > 0:
        return max(1, min(batch_size, max_batch_tokens // bucket))
    return batch_size


def _bucket_meets_min_fill(candidates, bucket, batch_size, max_batch_tokens=0, min_fill_frac=0.0):
    if min_fill_frac <= 0:
        return True
    target_rows = _target_bucket_rows(bucket, batch_size, max_batch_tokens)
    return len(candidates) >= max(1, math.ceil(target_rows * min_fill_frac))


def _choose_stream_bucket(
    bucket_groups,
    ordered_indices,
    buffer,
    batch_size,
    max_batch_tokens=0,
    rng=None,
    bucket_lens=None,
    bucket_selection="sample",
    bucket_state=None,
    bucket_min_fill_frac=0.0,
    bucket_cycle_repeat=1,
):
    candidate_groups = bucket_groups
    if bucket_min_fill_frac > 0:
        filled_groups = {
            bucket: candidates
            for bucket, candidates in bucket_groups.items()
            if _bucket_meets_min_fill(
                candidates,
                bucket,
                batch_size,
                max_batch_tokens=max_batch_tokens,
                min_fill_frac=bucket_min_fill_frac,
            )
        }
        if filled_groups:
            candidate_groups = filled_groups
    if bucket_selection == "cycle":
        repeat = max(1, int(bucket_cycle_repeat))
        if bucket_state is not None:
            repeat_bucket = bucket_state.get("repeat_bucket")
            repeat_remaining = int(bucket_state.get("repeat_remaining", 0))
            if repeat_remaining > 0 and repeat_bucket in candidate_groups:
                bucket_state["repeat_remaining"] = repeat_remaining - 1
                return repeat_bucket
        cursor = bucket_state.get("cursor", 0) if bucket_state is not None else 0
        for offset in range(len(bucket_lens)):
            idx = (cursor + offset) % len(bucket_lens)
            bucket = bucket_lens[idx]
            if bucket in candidate_groups:
                if bucket_state is not None:
                    bucket_state["cursor"] = (idx + 1) % len(bucket_lens)
                    bucket_state["repeat_bucket"] = bucket
                    bucket_state["repeat_remaining"] = repeat - 1
                return bucket
    if bucket_selection in ("max-tokens", "max-compute"):
        best_bucket = None
        best_score = None
        for bucket, candidates in candidate_groups.items():
            selected = _dense_bucket_indices(candidates, buffer, bucket, batch_size, max_batch_tokens=max_batch_tokens)
            useful_tokens = sum(buffer[idx]["expanded_len"] for idx in selected)
            attention_pairs = causal_attention_pairs(len(selected), bucket)
            if bucket_selection == "max-compute":
                score = (attention_pairs, useful_tokens)
            else:
                score = (useful_tokens, -attention_pairs)
            if best_score is None or score > best_score:
                best_bucket = bucket
                best_score = score
        return best_bucket
    sample_indices = [idx for bucket in candidate_groups for idx in candidate_groups[bucket]]
    selected = sample_indices[rng.randrange(len(sample_indices))] if rng is not None and len(sample_indices) > 1 else sample_indices[0]
    return bucketed_len(buffer[selected]["expanded_len"], bucket_lens)


def _choose_stream_buffer_indices(
    buffer,
    ordered_indices,
    batch_size,
    max_batch_tokens=0,
    rng=None,
    bucket_lens=None,
    bucket_selection="sample",
    bucket_state=None,
    bucket_min_fill_frac=0.0,
    bucket_cycle_repeat=1,
):
    if bucket_lens:
        bucket_groups = {}
        for idx in ordered_indices:
            bucket = bucketed_len(buffer[idx]["expanded_len"], bucket_lens)
            bucket_groups.setdefault(bucket, []).append(idx)
        bucket = _choose_stream_bucket(
            bucket_groups,
            ordered_indices,
            buffer,
            batch_size,
            max_batch_tokens=max_batch_tokens,
            rng=rng,
            bucket_lens=bucket_lens,
            bucket_selection=bucket_selection,
            bucket_state=bucket_state,
            bucket_min_fill_frac=bucket_min_fill_frac,
            bucket_cycle_repeat=bucket_cycle_repeat,
        )
        candidates = bucket_groups[bucket]
        # candidates are sorted by length ascending; take the densest rows in the
        # selected static bucket to reduce useful-token padding waste.
        return _dense_bucket_indices(candidates, buffer, bucket, batch_size, max_batch_tokens=max_batch_tokens)

    num_windows = max(1, math.ceil(len(ordered_indices) / batch_size))
    start = (rng.randrange(num_windows) if rng is not None and num_windows > 1 else 0) * batch_size
    start = min(start, max(0, len(ordered_indices) - 1))
    batch_indices = []
    max_len = 0
    for idx in ordered_indices[start:]:
        candidate = buffer[idx]
        next_max_len = max(max_len, candidate["expanded_len"])
        next_padded_len = bucketed_len(next_max_len, bucket_lens)
        if batch_indices and max_batch_tokens > 0 and next_padded_len * (len(batch_indices) + 1) > max_batch_tokens:
            break
        batch_indices.append(idx)
        max_len = next_max_len
        if len(batch_indices) >= batch_size:
            break
    if not batch_indices:
        batch_indices.append(ordered_indices[start])
    return batch_indices


def _choose_stream_packed_indices(
    buffer,
    ordered_indices,
    batch_size,
    max_batch_tokens=0,
    max_seq_len=0,
    max_images_per_row=1,
    fixed_rows=0,
    bucket_lens=None,
    rng=None,
    bucket_selection="sample",
    boundary_aware=False,
    compact_token_budget=False,
    ):
    candidate_count = min(len(ordered_indices), max(1, int(batch_size)))
    num_windows = max(1, math.ceil(len(ordered_indices) / candidate_count))

    def shuffled_indices():
        candidates = list(ordered_indices)
        if rng is not None and len(candidates) > 1:
            rng.shuffle(candidates)
        return candidates

    if bucket_selection == "random" and compact_token_budget and fixed_rows == 0:
        selected = []
        total_len = 0
        effective_max_seq_len = max_seq_len
        if effective_max_seq_len <= 0:
            effective_max_seq_len = max(int(buffer[idx]["expanded_len"]) for idx in ordered_indices)
        for idx in shuffled_indices():
            candidate_len = int(buffer[idx]["expanded_len"])
            if candidate_len > effective_max_seq_len:
                continue
            if max_batch_tokens > 0 and total_len + candidate_len > max_batch_tokens:
                continue
            selected.append(idx)
            total_len += candidate_len
            if len(selected) >= candidate_count:
                break
        if selected:
            return selected

    def compact_window(window_indices):
        if not window_indices:
            return [], []
        effective_max_seq_len = max_seq_len
        if effective_max_seq_len <= 0:
            effective_max_seq_len = max(int(buffer[idx]["expanded_len"]) for idx in window_indices)
        candidate_indices = sorted(
            window_indices,
            key=lambda idx: int(buffer[idx]["expanded_len"]),
            reverse=True,
        )
        selected = []
        lengths = []
        total_len = 0
        for idx in candidate_indices:
            candidate_len = int(buffer[idx]["expanded_len"])
            if candidate_len > effective_max_seq_len:
                continue
            if max_batch_tokens > 0 and total_len + candidate_len > max_batch_tokens:
                continue
            selected.append(idx)
            lengths.append(candidate_len)
            total_len += candidate_len
            if len(selected) >= candidate_count:
                break
        return selected, lengths

    def pack_window(window_indices):
        if not window_indices:
            return [], [], []
        candidates = [buffer[idx] for idx in window_indices]
        effective_max_seq_len = max_seq_len
        if effective_max_seq_len <= 0:
            effective_max_seq_len = max(int(ex["expanded_len"]) for ex in candidates)
        groups, lengths = pack_example_groups(
            candidates,
            max_seq_len=effective_max_seq_len,
            max_batch_tokens=max_batch_tokens,
            max_images_per_row=max_images_per_row,
            fixed_rows=fixed_rows,
            bucket_lens=bucket_lens,
            boundary_aware=boundary_aware,
            compact_token_budget=compact_token_budget,
        )
        selected = [window_indices[idx] for group in groups for idx in group]
        return selected, groups, lengths

    def first_packable_selection(indices):
        for idx in indices:
            selected, _, _ = pack_window([idx])
            if selected:
                return selected
        return []

    if bucket_selection in ("max-tokens", "max-compute"):
        best_selected = []
        best_score = None
        for window_idx in range(num_windows):
            start = window_idx * candidate_count
            window_indices = ordered_indices[start:start + candidate_count]
            if compact_token_budget:
                selected, lengths = compact_window(window_indices)
                if not selected:
                    continue
                useful_tokens = sum(lengths)
                rows = len(lengths)
                attention_pairs = causal_attention_pairs_for_lengths(lengths)
                token_score = packed_selection_token_score(useful_tokens, compact_token_budget=True)
                attention_score = attention_pairs if bucket_selection == "max-compute" else -attention_pairs
                score = (token_score, attention_score, useful_tokens, -rows)
            else:
                selected, groups, lengths = pack_window(window_indices)
                if not selected:
                    continue
                useful_tokens = sum(lengths)
                rows = len(lengths)
                padded_tokens = packed_budget_tokens(
                    lengths,
                    bucket_lens=bucket_lens,
                    compact_token_budget=compact_token_budget,
                )
                attention_pairs = packed_attention_pairs(
                    [buffer[idx] for idx in window_indices],
                    groups,
                    lengths,
                    bucket_lens=bucket_lens,
                    boundary_aware=boundary_aware,
                )
                attention_score = attention_pairs if bucket_selection == "max-compute" else -attention_pairs
                score = (useful_tokens, -padded_tokens, attention_score, -rows)
            if best_score is None or score > best_score:
                best_score = score
                best_selected = selected
        if best_selected:
            return best_selected

    if bucket_selection == "random":
        selected, _, _ = pack_window(shuffled_indices()[:candidate_count])
        if selected:
            return selected

    start = (rng.randrange(num_windows) if rng is not None and num_windows > 1 else 0) * candidate_count
    start = min(start, max(0, len(ordered_indices) - 1))
    candidate_indices = ordered_indices[start:start + candidate_count]
    if not candidate_indices:
        candidate_indices = [ordered_indices[start]]
    selected, _, _ = pack_window(candidate_indices)
    if selected:
        return selected
    selected = first_packable_selection(ordered_indices)
    if selected:
        return selected
    return [ordered_indices[0]]


def pack_example_groups(
    examples,
    max_seq_len,
    max_batch_tokens=0,
    max_images_per_row=1,
    fixed_rows=0,
    bucket_lens=None,
    boundary_aware=False,
    compact_token_budget=False,
):
    if max_images_per_row <= 1:
        groups = []
        lengths = []
        current_max_len = 0
        current_total_len = 0
        for idx, example in enumerate(examples):
            candidate_len = int(example["expanded_len"])
            if candidate_len > max_seq_len:
                continue
            trial_rows = len(lengths) + 1
            trial_max_len = max(current_max_len, candidate_len)
            if max_batch_tokens > 0:
                trial_budget = (
                    current_total_len + candidate_len
                    if compact_token_budget
                    else bucketed_len(trial_max_len, bucket_lens) * trial_rows
                )
                if trial_budget > max_batch_tokens:
                    continue
            if fixed_rows > 0 and len(groups) >= fixed_rows:
                continue
            groups.append([idx])
            lengths.append(candidate_len)
            current_max_len = trial_max_len
            current_total_len += candidate_len
        return groups, lengths

    order = sorted(range(len(examples)), key=lambda i: examples[i]["expanded_len"], reverse=True)
    if fixed_rows > 0:
        packed_groups = [[] for _ in range(fixed_rows)]
        packed_lengths = [0] * fixed_rows
    else:
        packed_groups = []
        packed_lengths = []
    current_max_len = max(packed_lengths) if packed_lengths else 0
    current_total_len = sum(packed_lengths)

    for idx in order:
        candidate = examples[idx]
        if candidate["expanded_len"] > max_seq_len:
            continue
        candidate_len = int(candidate["expanded_len"])
        boundary_input = 0 if boundary_aware else 1
        best_row = None
        best_len = None
        for row_idx, group in enumerate(packed_groups):
            if len(group) >= max_images_per_row:
                continue
            next_len = packed_lengths[row_idx] + candidate_len + (boundary_input if group else 0)
            if next_len > max_seq_len:
                continue
            trial_max_len = max(current_max_len, next_len)
            if max_batch_tokens > 0:
                trial_budget = (
                    current_total_len - packed_lengths[row_idx] + next_len
                    if compact_token_budget
                    else bucketed_len(trial_max_len, bucket_lens) * len(packed_lengths)
                )
                if trial_budget > max_batch_tokens:
                    continue
            if best_len is None or next_len < best_len:
                best_row = row_idx
                best_len = next_len
        if best_row is not None:
            current_total_len += best_len - packed_lengths[best_row]
            packed_groups[best_row].append(idx)
            packed_lengths[best_row] = best_len
            current_max_len = max(current_max_len, best_len)
            continue

        if fixed_rows > 0:
            continue
        next_len = candidate_len
        trial_max_len = max(current_max_len, next_len)
        trial_rows = len(packed_lengths) + 1
        if max_batch_tokens > 0:
            trial_budget = (
                current_total_len + next_len
                if compact_token_budget
                else bucketed_len(trial_max_len, bucket_lens) * trial_rows
            )
            if trial_budget > max_batch_tokens:
                continue
        packed_groups.append([idx])
        packed_lengths.append(next_len)
        current_max_len = max(current_max_len, next_len)
        current_total_len += next_len

    if fixed_rows > 0:
        packed_pairs = [(group, length) for group, length in zip(packed_groups, packed_lengths) if group]
        packed_groups = [group for group, _ in packed_pairs]
        packed_lengths = [length for _, length in packed_pairs]
    if not packed_groups:
        return [], []
    return packed_groups, packed_lengths


def trim_examples_to_packable(
    examples,
    max_seq_len,
    max_batch_tokens=0,
    max_images_per_row=1,
    fixed_rows=0,
    bucket_lens=None,
    boundary_aware=False,
    compact_token_budget=False,
):
    if max_images_per_row <= 1:
        return list(examples)
    if compact_token_budget and fixed_rows == 0:
        keep = []
        total_len = 0
        for example in examples:
            length = int(example["expanded_len"])
            if length > max_seq_len:
                continue
            if max_batch_tokens > 0 and total_len + length > max_batch_tokens:
                continue
            keep.append(example)
            total_len += length
        return keep
    groups, _ = pack_example_groups(
        examples,
        max_seq_len=max_seq_len,
        max_batch_tokens=max_batch_tokens,
        max_images_per_row=max_images_per_row,
        fixed_rows=fixed_rows,
        bucket_lens=bucket_lens,
        boundary_aware=boundary_aware,
        compact_token_budget=compact_token_budget,
    )
    if fixed_rows > 0 and len(groups) != fixed_rows:
        return []
    keep_indices = [idx for group in groups for idx in group]
    return [examples[idx] for idx in keep_indices]


def _example_image_feature_spans(examples):
    starts = []
    counts = []
    cursor = 0
    for example in examples:
        count = count_image_tokens(example["tokens"][:-1])
        starts.append(cursor)
        counts.append(count)
        cursor += count
    return starts, counts, cursor


def _select_image_features(image_features, feature_indices):
    if not feature_indices:
        return image_features[:0]
    if all(idx == i for i, idx in enumerate(feature_indices)):
        if len(feature_indices) == image_features.size(0):
            return image_features
        return image_features[:len(feature_indices)]
    index_tensor = torch.tensor(feature_indices, dtype=torch.long, device=image_features.device)
    return image_features.index_select(0, index_tensor)


def pack_example_rows(
    examples,
    image_features,
    max_seq_len,
    max_batch_tokens=0,
    max_images_per_row=1,
    fixed_rows=0,
    bucket_lens=None,
    boundary_aware=False,
    return_segment_lengths=False,
    compact_token_budget=False,
):
    boundary_aware = bool(boundary_aware or return_segment_lengths)
    packed_groups, _ = pack_example_groups(
        examples,
        max_seq_len=max_seq_len,
        max_batch_tokens=max_batch_tokens,
        max_images_per_row=max_images_per_row,
        fixed_rows=fixed_rows,
        bucket_lens=bucket_lens,
        boundary_aware=boundary_aware,
        compact_token_budget=compact_token_budget,
    )
    if not packed_groups:
        if return_segment_lengths:
            return [], [], image_features[:0], [], []
        return [], [], image_features[:0], []

    feature_starts, feature_counts, expected_features = _example_image_feature_spans(examples)
    assert expected_features == image_features.size(0), (
        f"feature batch {image_features.size(0)} != image markers {expected_features}"
    )
    rows = []
    masks = []
    feature_indices = []
    image_counts = []
    segment_lengths = []
    for group in packed_groups:
        row = []
        mask = []
        row_image_count = 0
        row_segment_lengths = []
        for idx in group:
            row.extend(examples[idx]["tokens"])
            mask.extend(examples[idx]["mask"])
            feature_indices.extend(range(feature_starts[idx], feature_starts[idx] + feature_counts[idx]))
            row_segment_lengths.append(len(examples[idx]["tokens"]))
            row_image_count += feature_counts[idx]
        rows.append(row)
        masks.append(mask)
        image_counts.append(row_image_count)
        segment_lengths.append(row_segment_lengths)

    packed_features = _select_image_features(image_features, feature_indices)
    if return_segment_lengths:
        return rows, masks, packed_features, image_counts, segment_lengths
    return rows, masks, packed_features, image_counts


def flatten_examples_as_compact_batch(examples, image_features, max_seq_len, max_batch_tokens=0):
    row = []
    mask = []
    feature_indices = []
    segment_lengths = []
    image_count = 0
    total_len = 0
    feature_starts, feature_counts, expected_features = _example_image_feature_spans(examples)
    assert expected_features == image_features.size(0), (
        f"feature batch {image_features.size(0)} != image markers {expected_features}"
    )
    for idx, example in enumerate(examples):
        expanded_len = int(example["expanded_len"])
        if expanded_len > max_seq_len:
            continue
        if max_batch_tokens > 0 and total_len + expanded_len > max_batch_tokens:
            continue
        row.extend(example["tokens"])
        mask.extend(example["mask"])
        feature_indices.extend(range(feature_starts[idx], feature_starts[idx] + feature_counts[idx]))
        segment_lengths.append(len(example["tokens"]))
        image_count += feature_counts[idx]
        total_len += expanded_len
    if not segment_lengths:
        return [], [], image_features[:0], [], []
    packed_features = _select_image_features(image_features, feature_indices)
    return [row], [mask], packed_features, [image_count], [segment_lengths]


def flatten_packed_rows(rows, masks, image_counts, segment_lengths):
    flat_row = []
    flat_mask = []
    flat_segment_lengths = []
    for row, mask, lengths in zip(rows, masks, segment_lengths):
        flat_row.extend(row)
        flat_mask.extend(mask)
        flat_segment_lengths.extend(lengths)
    return [flat_row], [flat_mask], [sum(image_counts)], [flat_segment_lengths]


def packed_example_count(image_counts, segment_lengths=None):
    if segment_lengths is not None:
        return sum(len(row_lengths) for row_lengths in segment_lengths)
    return sum(image_counts)


def supervised_target_count(tokens, mask, image_token_id=IMAGE_TOKEN_ID):
    count = 0
    for i, tok in enumerate(tokens[:-1]):
        next_tok = int(tokens[i + 1])
        if int(tok) == image_token_id or next_tok == image_token_id:
            continue
        count += int(mask[i + 1]) == 1
    return count


def render_record(rec, tokenizer, max_seq_len, image_root=None, require_openable_image=False):
    if _image_value(rec) is None or (require_openable_image and not image_record_is_openable(rec, image_root=image_root)):
        return None
    tokens, mask = render_vision_conversation(tokenizer, _ensure_image_marker_in_conversation(rec), max_tokens=UNTRUNCATED_MAX_TOKENS)
    if count_image_tokens(tokens) != 1 or count_image_tokens(tokens[:-1]) != 1:
        return None
    length = expanded_input_len(tokens)
    if length > max_seq_len or supervised_target_count(tokens, mask) <= 0:
        return None
    return {"tokens": tokens, "mask": mask, "record": rec, "expanded_len": length}


def render_records(records, tokenizer, max_seq_len, image_root=None, require_openable_image=False):
    rendered = [
        ex for rec in records
        if (ex := render_record(rec, tokenizer, max_seq_len, image_root=image_root, require_openable_image=require_openable_image)) is not None
    ]
    assert rendered, "no usable image-text examples loaded"
    return rendered


def take_rendered_examples(record_iter, tokenizer, max_seq_len, count, image_root=None, require_openable_image=False):
    examples = []
    while len(examples) < count:
        ex = render_record(next(record_iter), tokenizer, max_seq_len, image_root=image_root, require_openable_image=require_openable_image)
        if ex is not None:
            examples.append(ex)
    return examples


def iter_rendered_examples(record_iter, tokenizer, max_seq_len, image_root=None, require_openable_image=False):
    for rec in record_iter:
        ex = render_record(rec, tokenizer, max_seq_len, image_root=image_root, require_openable_image=require_openable_image)
        if ex is not None:
            yield ex


def next_batch(examples, batch_size, cursor, rng, max_batch_tokens=0, bucket_lens=None):
    if cursor == 0:
        rng.shuffle(examples)
        if bucket_lens:
            examples.sort(key=lambda ex: (bucketed_len(ex["expanded_len"], bucket_lens), -ex["expanded_len"]))
    batch = []
    max_len = 0
    target_bucket = None
    while len(batch) < batch_size:
        if cursor >= len(examples):
            cursor = 0
            rng.shuffle(examples)
            if bucket_lens:
                examples.sort(key=lambda ex: (bucketed_len(ex["expanded_len"], bucket_lens), -ex["expanded_len"]))
            if batch:
                break
        candidate = examples[cursor]
        candidate_bucket = bucketed_len(candidate["expanded_len"], bucket_lens)
        if target_bucket is None:
            target_bucket = candidate_bucket
        elif bucket_lens and candidate_bucket != target_bucket:
            break
        next_max_len = max(max_len, candidate["expanded_len"])
        next_padded_len = bucketed_len(next_max_len, bucket_lens)
        if batch and max_batch_tokens > 0 and next_padded_len * (len(batch) + 1) > max_batch_tokens:
            break
        batch.append(candidate)
        cursor += 1
        max_len = next_max_len
        if cursor >= len(examples):
            cursor = 0
            break
    return batch, cursor


def next_materialized_packed_batch(
    examples,
    batch_size,
    cursor,
    rng,
    max_batch_tokens=0,
    batch_buffer=None,
    batch_buffer_size=0,
    bucket_lens=None,
    bucket_selection="sample",
    pack_max_seq_len=0,
    max_images_per_row=1,
    pack_fixed_rows=0,
    boundary_aware_pack=False,
    flatten_packed_batch=False,
):
    if max_images_per_row <= 1:
        return next_batch(examples, batch_size, cursor, rng, max_batch_tokens=max_batch_tokens, bucket_lens=bucket_lens)
    if batch_buffer is None:
        batch_buffer = []
    if not examples:
        return [], cursor
    target_buffer = min(len(examples), max(batch_size, int(batch_buffer_size or 0)))
    if target_buffer <= 0:
        target_buffer = min(len(examples), max(1, int(batch_size)))
    while len(batch_buffer) < target_buffer:
        if cursor == 0:
            if batch_buffer:
                break
            rng.shuffle(examples)
        batch_buffer.append(examples[cursor])
        cursor += 1
        if cursor >= len(examples):
            cursor = 0
            if batch_buffer:
                break
    if not batch_buffer:
        return [], cursor
    ordered = sorted(range(len(batch_buffer)), key=lambda i: batch_buffer[i]["expanded_len"])
    batch_indices = _choose_stream_packed_indices(
        batch_buffer,
        ordered,
        batch_size,
        max_batch_tokens=max_batch_tokens,
        max_seq_len=pack_max_seq_len,
        max_images_per_row=max_images_per_row,
        fixed_rows=pack_fixed_rows,
        bucket_lens=bucket_lens,
        rng=rng,
        bucket_selection=bucket_selection,
        boundary_aware=boundary_aware_pack,
        compact_token_budget=flatten_packed_batch,
    )
    batch = [batch_buffer[idx] for idx in batch_indices]
    for idx in sorted(batch_indices, reverse=True):
        del batch_buffer[idx]
    return batch, cursor


def next_stream_batch(
    example_iter,
    batch_size,
    pending=None,
    max_batch_tokens=0,
    buffer=None,
    batch_buffer_size=0,
    rng=None,
    bucket_lens=None,
    bucket_selection="sample",
    bucket_state=None,
    bucket_min_fill_frac=0.0,
    bucket_cycle_repeat=1,
    pack_max_seq_len=0,
    max_images_per_row=1,
    pack_fixed_rows=0,
    boundary_aware_pack=False,
    flatten_packed_batch=False,
):
    if buffer is not None and batch_buffer_size > batch_size:
        if pending is not None:
            buffer.append(pending)
            pending = None
        target_buffer = max(batch_size, int(batch_buffer_size))
        while len(buffer) < target_buffer:
            buffer.append(next(example_iter))
        ordered = sorted(range(len(buffer)), key=lambda i: buffer[i]["expanded_len"])
        if max_images_per_row > 1:
            batch_indices = _choose_stream_packed_indices(
                buffer,
                ordered,
                batch_size,
                max_batch_tokens=max_batch_tokens,
                max_seq_len=pack_max_seq_len,
                max_images_per_row=max_images_per_row,
                fixed_rows=pack_fixed_rows,
                bucket_lens=bucket_lens,
                rng=rng,
                bucket_selection=bucket_selection,
                boundary_aware=boundary_aware_pack,
                compact_token_budget=flatten_packed_batch,
            )
        else:
            batch_indices = _choose_stream_buffer_indices(
                buffer,
                ordered,
                batch_size,
                max_batch_tokens=max_batch_tokens,
                rng=rng,
                bucket_lens=bucket_lens,
                bucket_selection=bucket_selection,
                bucket_state=bucket_state,
                bucket_min_fill_frac=bucket_min_fill_frac,
                bucket_cycle_repeat=bucket_cycle_repeat,
            )
        batch = [buffer[idx] for idx in batch_indices]
        for idx in sorted(batch_indices, reverse=True):
            del buffer[idx]
        return batch, pending

    batch = []
    max_len = 0
    while len(batch) < batch_size:
        candidate = pending if pending is not None else next(example_iter)
        pending = None
        next_max_len = max(max_len, candidate["expanded_len"])
        next_padded_len = bucketed_len(next_max_len, bucket_lens)
        if batch and max_batch_tokens > 0 and next_padded_len * (len(batch) + 1) > max_batch_tokens:
            pending = candidate
            break
        batch.append(candidate)
        max_len = next_max_len
    return batch, pending


def open_images_for_examples(examples, image_root=None, skip_bad_images=False, profile=None):
    images = []
    kept_examples = []
    for i, example in enumerate(examples):
        record = example["record"]
        try:
            images.append(open_image(record, image_root, profile=profile))
            kept_examples.append(example)
        except Exception as exc:
            if not skip_bad_images:
                raise
            print0(f"skipping image {record.get('image', record.get('id', i))}: {type(exc).__name__}: {exc}")
    if not images:
        return None, kept_examples
    return images, kept_examples


def batch_features_and_examples(extractor, examples, image_root=None, skip_bad_images=False, profile=None, synchronize=None):
    images, kept_examples = open_images_for_examples(examples, image_root=image_root, skip_bad_images=skip_bad_images, profile=profile)
    if not images:
        return None, kept_examples
    if profile is None:
        return extractor(images), kept_examples
    return extractor(images, profile=profile, synchronize=synchronize), kept_examples


def prepare_training_batch(select_examples, extractor, image_root=None, skip_bad_images=False, profile_timing=False, prefetch_processor=True):
    profile = new_profile() if profile_timing else None
    t = time.perf_counter()
    examples, selected_count = unpack_selected_examples(select_examples())
    add_profile(profile, "data", time.perf_counter() - t)
    images, kept_examples = open_images_for_examples(examples, image_root=image_root, skip_bad_images=skip_bad_images, profile=profile)
    if not images:
        return PreparedBatch(kept_examples, profile=profile, selected_examples=selected_count)
    if prefetch_processor:
        pixel_values = extractor.preprocess(images, profile=profile)
        return PreparedBatch(kept_examples, pixel_values=pixel_values, profile=profile, selected_examples=selected_count)
    return PreparedBatch(kept_examples, images=images, profile=profile, selected_examples=selected_count)


def iter_prepared_batches(select_examples, extractor, image_root=None, skip_bad_images=False, profile_timing=False, prefetch_processor=True):
    while True:
        yield prepare_training_batch(
            select_examples,
            extractor,
            image_root=image_root,
            skip_bad_images=skip_bad_images,
            profile_timing=profile_timing,
            prefetch_processor=prefetch_processor,
        )


def count_params(parameters):
    return sum(p.numel() for p in parameters)


def estimate_model_flops(model, sequence_len=None, cache=None):
    key = None if sequence_len is None else int(sequence_len)
    if cache is not None and key in cache:
        return cache[key]
    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    value = base_model.estimate_flops(sequence_len=sequence_len)
    if cache is not None:
        cache[key] = value
    return value


def estimate_model_flops_components(model, sequence_len=None, cache=None):
    key = ("components", None if sequence_len is None else int(sequence_len))
    if cache is not None and key in cache:
        return cache[key]
    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    if hasattr(base_model, "estimate_flops_components"):
        value = base_model.estimate_flops_components(sequence_len=sequence_len)
    else:
        value = (base_model.estimate_flops(sequence_len=sequence_len), 0.0)
    if cache is not None:
        cache[key] = value
    return value


def estimate_lm_head_flops_per_token(model):
    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    lm_head = getattr(base_model, "lm_head", None)
    if lm_head is None:
        return 0.0
    return 6.0 * count_params(lm_head.parameters())


def estimate_model_flops_breakdown(model, sequence_len=None, cache=None):
    key = ("breakdown", None if sequence_len is None else int(sequence_len))
    if cache is not None and key in cache:
        return cache[key]
    non_attn_flops, attn_flops = estimate_model_flops_components(model, sequence_len=sequence_len, cache=cache)
    lm_head_flops = estimate_lm_head_flops_per_token(model)
    trunk_non_attn_flops = max(0.0, float(non_attn_flops) - float(lm_head_flops))
    value = (trunk_non_attn_flops, float(attn_flops), float(lm_head_flops))
    if cache is not None:
        cache[key] = value
    return value


def estimate_varlen_step_flops(
    model,
    segment_lengths,
    padded_tokens: int,
    useful_lm_head_tokens: int | None = None,
    padded_lm_head_tokens: int | None = None,
    cache=None,
):
    useful_flops = 0.0
    attention_flops = 0.0
    trunk_non_attn_flops = 0.0
    lm_head_flops = 0.0
    for length in segment_lengths:
        trunk, attn, lm_head = estimate_model_flops_breakdown(model, sequence_len=int(length), cache=cache)
        useful_flops += (trunk + attn) * int(length)
        attention_flops += attn * int(length)
        trunk_non_attn_flops = trunk
        lm_head_flops = lm_head
    useful_tokens = sum(int(length) for length in segment_lengths)
    if useful_lm_head_tokens is None:
        useful_lm_head_tokens = useful_tokens
    if padded_lm_head_tokens is None:
        padded_lm_head_tokens = int(padded_tokens)
    useful_flops += lm_head_flops * int(useful_lm_head_tokens)
    padded_flops = trunk_non_attn_flops * int(padded_tokens) + attention_flops + lm_head_flops * int(padded_lm_head_tokens)
    return useful_flops, padded_flops


def estimate_dense_step_flops(
    model,
    sequence_len: int,
    useful_tokens: int,
    padded_tokens: int,
    useful_lm_head_tokens: int | None = None,
    padded_lm_head_tokens: int | None = None,
    cache=None,
):
    trunk, attn, lm_head = estimate_model_flops_breakdown(model, sequence_len=sequence_len, cache=cache)
    useful_tokens = int(useful_tokens)
    padded_tokens = int(padded_tokens)
    if useful_lm_head_tokens is None:
        useful_lm_head_tokens = useful_tokens
    if padded_lm_head_tokens is None:
        padded_lm_head_tokens = padded_tokens
    useful_flops = (trunk + attn) * useful_tokens + lm_head * int(useful_lm_head_tokens)
    padded_flops = (trunk + attn) * padded_tokens + lm_head * int(padded_lm_head_tokens)
    return useful_flops, padded_flops


def evaluate_vlm_bpb(model, projector, extractor, examples, image_root, device, token_bytes, batch_size=4, max_examples=-1, max_seq_len=2048, skip_bad_images=False):
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
                    skip_bad_images=skip_bad_images,
                )
                if feats is None or not batch_examples:
                    continue
                feats = feats.to(device=device, non_blocking=True)
                rows = [ex["tokens"] for ex in batch_examples]
                masks = [ex["mask"] for ex in batch_examples]
                batch = build_multimodal_batch(model, projector, rows, feats, loss_mask_rows=masks, max_seq_len=max_seq_len, value_fallback_token_id=rows[0][0])
                loss = model(batch.value_token_ids, batch.targets, input_embeds=batch.input_embeds, loss_reduction="none", selective_loss=True).view(-1)
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
        "model_config": model_meta["model_config"],
        "user_config": vars(args),
        "data_path": data_path,
        "data_config": {
            "data_path": data_path,
            "data_json": getattr(args, "data_json", None),
            "hf_repo": getattr(args, "hf_repo", None),
            "hf_config": getattr(args, "hf_config", None),
            "stream_hf_data": True,
            "stream_buffer_size": getattr(args, "stream_buffer_size", None),
            "batch_buffer_size": getattr(args, "batch_buffer_size", None),
            "bucket_selection": getattr(args, "bucket_selection", None),
            "bucket_min_fill_frac": getattr(args, "bucket_min_fill_frac", None),
            "bucket_cycle_repeat": getattr(args, "bucket_cycle_repeat", None),
            "prefetch_batches": getattr(args, "prefetch_batches", None),
            "prefetch_workers": getattr(args, "prefetch_workers", None),
            "prefetch_processor": getattr(args, "prefetch_processor", None),
            "compile_model": getattr(args, "compile_model", None),
            "mfu_warmup_bucket_steps": getattr(args, "mfu_warmup_bucket_steps", None),
            "pack_examples": getattr(args, "pack_examples", None),
            "boundary_aware_pack": getattr(args, "boundary_aware_pack", None),
            "flatten_packed_batch": getattr(args, "flatten_packed_batch", None),
            "allow_leaky_pack": getattr(args, "allow_leaky_pack", None),
            "require_fa3_varlen": getattr(args, "require_fa3_varlen", None),
            "pack_max_seq_len": getattr(args, "pack_max_seq_len", None),
            "pad_to_max_seq_len": getattr(args, "pad_to_max_seq_len", None),
            "pad_to_bucket_lens": getattr(args, "pad_to_bucket_lens", None),
            "selective_loss": getattr(args, "selective_loss", None),
            "loss_chunk_size": getattr(args, "loss_chunk_size", None),
            "drop_zero_value_embeds": getattr(args, "drop_zero_value_embeds", None),
            "max_examples": getattr(args, "max_examples", None),
            "image_root": getattr(args, "image_root", None),
            "skip_bad_images": getattr(args, "skip_bad_images", None),
            "no_save": getattr(args, "no_save", None),
        },
        "vision_config": {
            "siglip_model_id": args.siglip_model_id,
            "siglip_use_fast_processor": getattr(args, "siglip_use_fast_processor", None),
            "siglip_forward_batch_size": getattr(args, "siglip_forward_batch_size", None),
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
    parser.add_argument("--data-json", default=None, help="local LLaVA-style JSON/JSONL")
    parser.add_argument("--hf-repo", default=None, help=f"Hugging Face dataset repo (default: {DEFAULT_HF_REPO})")
    parser.add_argument("--hf-config", default=None, help="Hugging Face dataset config to stream; use 'all' for all configs")
    parser.add_argument("--image-root", default=None, help="directory containing referenced images")
    parser.add_argument("--skip-bad-images", action=argparse.BooleanOptionalAction, default=True, help="skip records whose image cannot be opened")
    parser.add_argument("--max-examples", type=int, default=-1)
    parser.add_argument("--siglip-model-id", default=SIGLIP_MODEL_ID)
    parser.add_argument("--siglip-cache-dir", default=None, help="optional HF cache dir for SigLIP weights")
    parser.add_argument("--siglip-use-fast-processor", action=argparse.BooleanOptionalAction, default=True, help="use transformers fast image processor when available")
    parser.add_argument("--siglip-forward-batch-size", type=int, default=0, help="optional image microbatch size for frozen SigLIP forward (0 = all images at once)")
    parser.add_argument("--hf-checkpoint", default="karpathy/nanochat-d32", help="HF nanochat checkpoint repo to link into NANOCHAT_BASE_DIR")
    parser.add_argument("--model-tag", default="d32")
    parser.add_argument("--model-step", type=int, default=None)
    parser.add_argument("--device-type", default="", choices=["", "cuda", "cpu", "mps"])
    parser.add_argument("--fp8", action="store_true", help="enable nanochat FP8 Linear training for the LLM on H100+")
    parser.add_argument("--fp8-recipe", default="tensorwise", choices=["tensorwise"], help="FP8 scaling recipe")
    parser.add_argument("--drop-zero-value-embeds", action=argparse.BooleanOptionalAction, default=False, help="drop frozen zero value embeddings when the checkpoint only has compatibility-patched VE tables")
    parser.add_argument("--device-batch-size", type=int, default=4)
    parser.add_argument("--max-batch-tokens", type=int, default=0, help="optional cap on padded tokens per device batch")
    parser.add_argument("--stream-buffer-size", type=int, default=4096, help="raw HF records kept in the streaming shuffle buffer")
    parser.add_argument("--batch-buffer-size", type=int, default=0, help="rendered examples kept for length-aware streaming batches (0 = 4x device batch)")
    parser.add_argument("--bucket-selection", choices=["sample", "cycle", "max-tokens", "max-compute", "random"], default="sample", help="streaming batch choice policy for static buckets or packed candidate windows")
    parser.add_argument("--bucket-min-fill-frac", type=float, default=0.0, help="skip underfilled static buckets until they have this fraction of their token-capped rows (0 = disable)")
    parser.add_argument("--bucket-cycle-repeat", type=int, default=1, help="with --bucket-selection cycle, repeat each selected bucket for this many microbatches before advancing")
    parser.add_argument("--prefetch-batches", type=int, default=2, help="prepared CPU batches to keep ahead of the GPU (0 disables)")
    parser.add_argument("--prefetch-workers", type=int, default=1, help="CPU workers for prepared-batch prefetching")
    parser.add_argument("--prefetch-processor", action=argparse.BooleanOptionalAction, default=True, help="run image processor in the prefetch worker")
    parser.add_argument("--pack-examples", type=int, default=1, help="max image-text examples to pack into one VLM sequence row (1 disables)")
    parser.add_argument("--pack-max-seq-len", type=int, default=0, help="optional max packed row length before padding (0 = --max-seq-len)")
    parser.add_argument("--pack-fixed-rows", type=int, default=0, help="pack into exactly this many nonempty rows when possible (0 = dynamic rows)")
    parser.add_argument("--boundary-aware-pack", action="store_true", help="reset RoPE/smear and block attention across examples packed into one row")
    parser.add_argument("--flatten-packed-batch", action="store_true", help="with --boundary-aware-pack, flatten packed rows into one compact varlen batch to avoid per-layer gather/scatter")
    parser.add_argument("--allow-leaky-pack", action="store_true", help="diagnostic only: allow --pack-examples > 1 without boundary-aware attention")
    parser.add_argument("--require-fa3-varlen", action="store_true", help="fail if --boundary-aware-pack cannot use FA3 varlen attention")
    parser.add_argument("--pad-to-max-seq-len", action="store_true", help="pad every VLM batch to --max-seq-len for fixed-shape compile probes")
    parser.add_argument("--pad-to-bucket-lens", default="", help="comma-separated static padding buckets, e.g. 128,192,256,384,512")
    parser.add_argument("--selective-loss", action=argparse.BooleanOptionalAction, default=True, help="compute lm_head/loss only on supervised targets")
    parser.add_argument("--loss-chunk-size", type=int, default=0, help="optional chunk size for full ignore-index CE to avoid materializing all logits at once")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--num-iterations", type=int, default=1000)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--projector-lr", type=float, default=2e-3)
    parser.add_argument("--embedding-lr", type=float, default=None, help="LLM embedding LR (default: inherit from checkpoint)")
    parser.add_argument("--unembedding-lr", type=float, default=None, help="LLM unembedding LR (default: inherit from checkpoint)")
    parser.add_argument("--matrix-lr", type=float, default=None, help="LLM matrix LR (default: inherit from checkpoint)")
    parser.add_argument("--init-lr-frac", type=float, default=0.8)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--warmdown-ratio", type=float, default=0.5)
    parser.add_argument("--final-lr-frac", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=-1, help="evaluate held-out VLM BPB every N steps (-1 = disable)")
    parser.add_argument("--eval-examples", type=int, default=100, help="held-out HF stream examples to score for BPB")
    parser.add_argument("--eval-batch-size", type=int, default=4, help="batch size for held-out VLM BPB")
    parser.add_argument("--vlm-eval-every", type=int, default=-1, help="run VLM eval benchmark subset every N steps (-1 = disable)")
    parser.add_argument("--vlm-eval-benchmarks", default="mmstar,scienceqa,chartqa,mmmu,textvqa")
    parser.add_argument("--vlm-eval-mmmu-configs", default="Accounting")
    parser.add_argument("--vlm-eval-max-per-benchmark", type=int, default=100)
    parser.add_argument("--vlm-eval-max-scan", type=int, default=2000)
    parser.add_argument("--vlm-eval-max-new-tokens", type=int, default=16)
    parser.add_argument("--vlm-eval-print-samples", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=-1)
    parser.add_argument("--no-save", action="store_true", help="disable checkpoint writes, useful for throughput-only probes")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--mfu-warmup-steps", type=int, default=2, help="exclude the first N optimizer steps from steady-state MFU averages")
    parser.add_argument("--mfu-warmup-bucket-steps", type=int, default=0, help="also exclude the first N occurrences of each static bucket from steady-state MFU")
    parser.add_argument("--compile", dest="compile_model", action=argparse.BooleanOptionalAction, default=False, help="compile the LLM with torch.compile(dynamic=True)")
    parser.add_argument("--profile-timing", action="store_true", help="log per-step image decode/processor/SigLIP and LLM timing")
    parser.add_argument("--attention-backend-report", action="store_true", help="print attention backend info after device init and exit")
    parser.add_argument("--length-stats-examples", type=int, default=0, help="render this many usable examples, print expanded-length/bucket stats, and exit")
    parser.add_argument("--length-stats-max-records", type=int, default=0, help="optional raw-record scan cap for --length-stats-examples (0 = no cap)")
    parser.add_argument("--batch-plan-steps", type=int, default=0, help="render/select this many CPU-only train batches, print bucket fill stats, and exit")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--init-vlm-checkpoint-dir", default=None, help="optional VLM checkpoint dir to initialize projector/model")
    parser.add_argument("--init-vlm-checkpoint-step", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.hf_repo, args.hf_config = _resolve_hf_source(args)
    if args.pack_examples < 1:
        parser.error("--pack-examples must be >= 1")
    if args.pack_max_seq_len < 0:
        parser.error("--pack-max-seq-len must be >= 0")
    if args.pack_fixed_rows < 0:
        parser.error("--pack-fixed-rows must be >= 0")
    if args.pack_examples > 1 and not args.boundary_aware_pack and not args.allow_leaky_pack:
        parser.error("--pack-examples > 1 requires --boundary-aware-pack unless --allow-leaky-pack is set for a diagnostic ablation")
    if args.require_fa3_varlen and not args.boundary_aware_pack:
        parser.error("--require-fa3-varlen requires --boundary-aware-pack")
    if args.flatten_packed_batch and not args.boundary_aware_pack:
        parser.error("--flatten-packed-batch requires --boundary-aware-pack")
    if args.prefetch_workers < 1:
        parser.error("--prefetch-workers must be >= 1")
    if not 0.0 <= args.bucket_min_fill_frac <= 1.0:
        parser.error("--bucket-min-fill-frac must be between 0 and 1")
    if args.bucket_cycle_repeat < 1:
        parser.error("--bucket-cycle-repeat must be >= 1")
    if args.mfu_warmup_bucket_steps < 0:
        parser.error("--mfu-warmup-bucket-steps must be >= 0")
    if args.loss_chunk_size < 0:
        parser.error("--loss-chunk-size must be >= 0")
    if args.length_stats_examples < 0:
        parser.error("--length-stats-examples must be >= 0")
    if args.length_stats_max_records < 0:
        parser.error("--length-stats-max-records must be >= 0")
    if args.batch_plan_steps < 0:
        parser.error("--batch-plan-steps must be >= 0")
    if args.siglip_forward_batch_size < 0:
        parser.error("--siglip-forward-batch-size must be >= 0")
    try:
        pad_bucket_lens = parse_bucket_lens(args.pad_to_bucket_lens, args.max_seq_len)
    except ValueError as exc:
        parser.error(str(exc))
    if args.pad_to_max_seq_len and pad_bucket_lens:
        parser.error("--pad-to-max-seq-len and --pad-to-bucket-lens are mutually exclusive")
    if args.flatten_packed_batch and (args.pad_to_max_seq_len or pad_bucket_lens):
        parser.error("--flatten-packed-batch cannot be combined with static row padding")
    if args.eval_every > 0 and args.eval_examples <= 0:
        parser.error("--eval-every requires --eval-examples > 0")

    base_dir = get_base_dir()
    out_dir = args.out_dir or os.path.join(base_dir, "vlm_checkpoints", f"{args.model_tag}_vlm")

    if args.length_stats_examples > 0:
        if args.hf_checkpoint:
            ensure_hf_nanochat_tokenizer(args.hf_checkpoint, base_dir)
        run_length_stats(args, get_tokenizer(), pad_bucket_lens)
        exit_after_cpu_report()
    if args.batch_plan_steps > 0:
        if args.hf_checkpoint:
            ensure_hf_nanochat_tokenizer(args.hf_checkpoint, base_dir)
        run_batch_plan(args, get_tokenizer(), pad_bucket_lens)
        exit_after_cpu_report()

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    _, ddp_rank, _, ddp_world_size, device = compute_init(device_type)
    assert ddp_world_size == 1, "v0 VLM trainer is single-GPU; launch one process"
    if args.require_fa3_varlen:
        try:
            require_fa3_varlen()
        except RuntimeError as exc:
            parser.error(str(exc))
    print0(f"Attention backend: {attention_backend_info()}", flush=True)
    if args.attention_backend_report:
        compute_cleanup()
        return

    if args.hf_checkpoint:
        ensure_hf_nanochat_checkpoint(args.hf_checkpoint, base_dir, model_tag=args.model_tag, source=MODEL_SOURCE)

    synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
    wandb_run = DummyWandb() if args.run == "dummy" or ddp_rank != 0 else wandb.init(project="nanochat-vlm", name=args.run, config=vars(args))

    model, tokenizer, meta = load_model(MODEL_SOURCE, device, phase="train", model_tag=args.model_tag, step=args.model_step)
    pretrain_config = meta.get("user_config", {})
    for name, fallback in [("embedding_lr", 0.3), ("unembedding_lr", 0.004), ("matrix_lr", 0.02)]:
        if getattr(args, name) is None:
            setattr(args, name, pretrain_config.get(name, fallback))
    scalar_lr = pretrain_config.get("scalar_lr", 0.5)
    rng = random.Random(args.seed)
    t = time.perf_counter()
    val_examples = None
    stream_examples = None
    stream_pending = None
    stream_batch_buffer = []
    materialized_batch_buffer = []
    stream_bucket_state = {}
    stream_buffer_size = None
    batch_buffer_size = 0
    examples = None
    streaming_train = args.data_json is None and args.max_examples <= 0
    if streaming_train:
        stream_buffer_size = max(args.stream_buffer_size, args.device_batch_size)
        batch_buffer_size = args.batch_buffer_size if args.batch_buffer_size > 0 else max(args.device_batch_size * args.grad_accum_steps * 4, args.device_batch_size)
        args.batch_buffer_size = batch_buffer_size
        record_iter = iter_hf_records(args, seed=args.seed, buffer_size=stream_buffer_size)
        data_path = data_source_name(args)
        if args.eval_every > 0:
            val_examples = take_rendered_examples(
                record_iter,
                tokenizer,
                args.max_seq_len,
                args.eval_examples,
                image_root=args.image_root,
                require_openable_image=args.skip_bad_images,
            )
        stream_examples = iter_rendered_examples(
            record_iter,
            tokenizer,
            args.max_seq_len,
            image_root=args.image_root,
            require_openable_image=args.skip_bad_images,
        )
        print0(
            f"Initialized HF stream in {time.perf_counter() - t:.2f}s | "
            f"shuffle_buffer={stream_buffer_size:,} raw records | batch_buffer={batch_buffer_size:,} rendered examples | "
            f"val={0 if val_examples is None else len(val_examples):,}",
            flush=True,
        )
    else:
        records, val_records, data_path = load_records(args, val_count=args.eval_examples if args.eval_every > 0 else 0)
        print0(f"Loaded {len(records):,} records in {time.perf_counter() - t:.2f}s")
        t = time.perf_counter()
        examples = render_records(records, tokenizer, args.max_seq_len, image_root=args.image_root, require_openable_image=args.skip_bad_images)
        print0(f"Rendered {len(examples):,} usable examples in {time.perf_counter() - t:.2f}s")
        if args.pack_examples > 1:
            batch_buffer_size = args.batch_buffer_size if args.batch_buffer_size > 0 else max(args.device_batch_size * args.grad_accum_steps * 4, args.device_batch_size)
            args.batch_buffer_size = batch_buffer_size
        if args.eval_every > 0:
            val_examples = render_records(val_records, tokenizer, args.max_seq_len, image_root=args.image_root, require_openable_image=args.skip_bad_images)
    if val_examples is not None:
        print0(f"Rendered {len(val_examples):,} held-out validation examples")

    siglip_cache_dir = args.siglip_cache_dir or os.environ.get("NANOCHAT_SIGLIP_CACHE_DIR")
    extractor = SigLIPPooledFeatureExtractor(
        args.siglip_model_id,
        device=device,
        cache_dir=siglip_cache_dir,
        processor_use_fast=args.siglip_use_fast_processor,
        forward_batch_size=args.siglip_forward_batch_size,
        verbose=ddp_rank == 0,
    )
    projector = VisionProjector(extractor.vision_dim, model.config.n_embd).to(device=device)
    if args.init_vlm_checkpoint_dir:
        assert args.init_vlm_checkpoint_step is not None, "--init-vlm-checkpoint-step is required with --init-vlm-checkpoint-dir"
        model_state, projector, _, init_meta = load_vlm_checkpoint(args.init_vlm_checkpoint_dir, args.init_vlm_checkpoint_step, device, load_optimizer=False, checkpoint_device=torch.device("cpu"))
        model.load_state_dict(model_state, strict=True)
        assert projector.vision_dim == extractor.vision_dim, "checkpoint projector vision dim does not match SigLIP"
        print0(f"Initialized VLM state from {args.init_vlm_checkpoint_dir} step {args.init_vlm_checkpoint_step}", flush=True)

    dropped_value_embed_params = 0
    if args.drop_zero_value_embeds:
        compat = meta.get("compatibility_patches", {})
        expected_ve = compat.get("expected_value_embed_keys") or []
        missing_ve = compat.get("missing_value_embed_keys") or []
        if args.init_vlm_checkpoint_dir:
            print0("Skipping --drop-zero-value-embeds because an explicit VLM checkpoint was loaded", flush=True)
        elif expected_ve and set(missing_ve) == set(expected_ve):
            dropped = model.drop_value_embedding_path()
            dropped_value_embed_params = int(dropped["total_params"])
            gc.collect()
            if device_type == "cuda":
                torch.cuda.empty_cache()
            print0(
                "Dropped frozen zero value-embedding path "
                f"({dropped['value_embed_params']:,} VE params + {dropped['value_gate_params']:,} gate params)",
                flush=True,
            )
        else:
            print0(
                "--drop-zero-value-embeds requested but checkpoint has real value-embedding keys; keeping VE path",
                flush=True,
            )

    for p in model.parameters():
        p.requires_grad = True
    for p in model.value_embeds.parameters():
        p.requires_grad = False
    for block in model.transformer.h:
        if block.attn.ve_gate is not None:
            for p in block.attn.ve_gate.parameters():
                p.requires_grad = False
    model.train()
    if args.fp8:
        if device_type != "cuda":
            print0("Warning: FP8 training requires CUDA, ignoring --fp8 flag", flush=True)
        else:
            from nanochat.fp8 import Float8LinearConfig, convert_to_float8_training
            import torch.nn as nn

            def fp8_module_filter(mod: nn.Module, fqn: str) -> bool:
                if not isinstance(mod, nn.Linear):
                    return False
                if args.selective_loss and fqn == "lm_head":
                    return False
                if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                    return False
                if min(mod.in_features, mod.out_features) < 128:
                    return False
                return True

            fp8_config = Float8LinearConfig.from_recipe_name(args.fp8_recipe)
            num_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
            convert_to_float8_training(model, config=fp8_config, module_filter_fn=fp8_module_filter)
            num_fp8 = sum(1 for m in model.modules() if "Float8" in type(m).__name__)
            print0(
                f"FP8 LLM training enabled ({args.fp8_recipe}); converted {num_fp8}/{num_linear} Linear layers",
                flush=True,
            )
    if args.compile_model:
        compile_dynamic = not (args.pad_to_max_seq_len or pad_bucket_lens) or args.selective_loss
        model = torch.compile(model, dynamic=compile_dynamic)
        print0(f"Compiled LLM with torch.compile(dynamic={compile_dynamic})", flush=True)
    llm_optimizer = model.setup_optimizer(
        unembedding_lr=args.unembedding_lr,
        embedding_lr=args.embedding_lr,
        matrix_lr=args.matrix_lr,
        scalar_lr=scalar_lr,
        weight_decay=0.0,
    )
    for group in llm_optimizer.param_groups:
        group["lr"] = group["lr"] * args.init_lr_frac
        group["initial_lr"] = group["lr"]
    projector.train()
    projector_optimizer = torch.optim.AdamW(projector.parameters(), lr=args.projector_lr, weight_decay=0.0)

    cursor = 0
    smooth_loss = 0.0
    smooth_count = 0
    steady_seconds = 0.0
    steady_tokens = 0
    steady_padded_tokens = 0
    steady_seq_flops = 0.0
    steady_seq_padded_flops = 0.0
    steady_samples = 0
    steady_attention_pairs = 0
    steady_segments = 0
    steady_segment_length_counts = {}
    steady_max_segment_len = 0
    steady_near_cap_segments = 0
    steady_cap_segments = 0
    steady_steps = 0
    steady_profile = new_profile()
    steady_bucket_stats = {}
    bucket_seen_steps = {}
    t_start = time.time()
    gpu_name = torch.cuda.get_device_name(0) if device_type == "cuda" else device_type
    gpu_peak_flops = get_peak_flops(gpu_name) if device_type == "cuda" else float("inf")
    flops_per_token_cache = {}
    num_flops_per_token = estimate_model_flops(model, cache=flops_per_token_cache)
    total_params = count_params(model.parameters()) + count_params(projector.parameters())
    total_trainable = count_params(p for p in list(model.parameters()) + list(projector.parameters()) if p.requires_grad)
    token_bytes = get_token_bytes(device=device) if val_examples is not None else None
    example_count = f"hf stream shuffle_buffer={stream_buffer_size:,} batch_buffer={batch_buffer_size:,}" if stream_examples is not None else f"{len(examples):,}"
    print0(f"VLM train | GPU: {gpu_name} | examples: {example_count} | data: {data_path} | out: {out_dir}", flush=True)
    print0(f"Params total/trainable: {total_params:,}/{total_trainable:,}")
    print0(f"Estimated LLM FLOPs/token: {num_flops_per_token:e} | Peak BF16 FLOPS: {gpu_peak_flops:.2e}")
    llm_lr_text = (
        f"nanochat MuonAdamW init_frac={args.init_lr_frac:g} "
        f"unembed={args.unembedding_lr:g} embed={args.embedding_lr:g} matrix={args.matrix_lr:g} scalar={scalar_lr:g}"
    )
    print0(f"LRs: projector={args.projector_lr:g} llm={llm_lr_text}")
    print0(
        f"Pipeline: prefetch_batches={args.prefetch_batches} prefetch_workers={args.prefetch_workers} "
        f"prefetch_processor={args.prefetch_processor} "
        f"siglip_forward_batch_size={args.siglip_forward_batch_size} "
        f"batch_buffer_size={batch_buffer_size} bucket_selection={args.bucket_selection} "
        f"bucket_min_fill_frac={args.bucket_min_fill_frac:g} bucket_cycle_repeat={args.bucket_cycle_repeat} "
        f"pack_examples={args.pack_examples} "
        f"pack_max_seq_len={args.pack_max_seq_len or args.max_seq_len} pack_fixed_rows={args.pack_fixed_rows} "
        f"boundary_aware_pack={args.boundary_aware_pack} flatten_packed_batch={args.flatten_packed_batch} "
        f"drop_zero_value_embeds={args.drop_zero_value_embeds} dropped_value_embed_params={dropped_value_embed_params} "
        f"pad_buckets={pad_bucket_lens or 'none'} selective_loss={args.selective_loss} "
        f"loss_chunk_size={args.loss_chunk_size} fp8={args.fp8} fp8_recipe={args.fp8_recipe}"
    )
    setup_memory = cuda_memory_stats_mib(device_type)
    setup_mem_mib = setup_memory["allocated"]
    print0(f"Allocated memory after setup: {setup_mem_mib:.2f}MiB")
    wandb_run.log({
        "gpu/setup_allocated_mib": setup_memory["allocated"],
        "gpu/setup_reserved_mib": setup_memory["reserved"],
    })
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
            args.image_root,
            device,
            token_bytes,
            batch_size=args.eval_batch_size,
            max_examples=-1,
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

    def run_vlm_eval(step):
        results = evaluate_vlm(
            model,
            projector,
            tokenizer,
            extractor,
            benchmarks=args.vlm_eval_benchmarks,
            mmmu_configs=args.vlm_eval_mmmu_configs,
            limit=args.vlm_eval_max_per_benchmark,
            max_scan=args.vlm_eval_max_scan,
            max_new_tokens=args.vlm_eval_max_new_tokens,
            print_samples=args.vlm_eval_print_samples,
        )
        scores = {key: row["score"] for key, row in results["benchmarks"].items()}
        mean_score = sum(scores.values()) / max(len(scores), 1)
        score_str = " ".join(f"{key}={score:.3f}" for key, score in scores.items())
        print0(f"step {step:05d}/{args.num_iterations:05d} | VLM eval {mean_score:.4f} | {score_str}", flush=True)
        wandb_run.log({
            "step": step,
            "vlm_eval/mean_score": mean_score,
            **{f"vlm_eval/{key}_score": score for key, score in scores.items()},
        })

    if args.eval_every > 0:
        run_val_bpb(0)

    def select_train_examples():
        nonlocal cursor, stream_pending
        selection_max_tokens = args.max_batch_tokens if (args.pack_examples > 1 and stream_examples is not None) else (0 if args.pack_examples > 1 else args.max_batch_tokens)
        if stream_examples is None:
            if args.pack_examples > 1:
                batch_examples, cursor = next_materialized_packed_batch(
                    examples,
                    args.device_batch_size,
                    cursor,
                    rng,
                    max_batch_tokens=args.max_batch_tokens,
                    batch_buffer=materialized_batch_buffer,
                    batch_buffer_size=batch_buffer_size,
                    bucket_lens=pad_bucket_lens,
                    bucket_selection=args.bucket_selection,
                    pack_max_seq_len=min(args.max_seq_len, args.pack_max_seq_len or args.max_seq_len),
                    max_images_per_row=args.pack_examples,
                    pack_fixed_rows=args.pack_fixed_rows,
                    boundary_aware_pack=args.boundary_aware_pack,
                    flatten_packed_batch=args.flatten_packed_batch,
                )
            else:
                batch_examples, cursor = next_batch(
                    examples,
                    args.device_batch_size,
                    cursor,
                    rng,
                    max_batch_tokens=selection_max_tokens,
                    bucket_lens=pad_bucket_lens,
                )
        else:
            batch_examples, stream_pending = next_stream_batch(
                stream_examples,
                args.device_batch_size,
                pending=stream_pending,
                max_batch_tokens=selection_max_tokens,
                buffer=stream_batch_buffer,
                batch_buffer_size=batch_buffer_size,
                rng=rng,
                bucket_lens=pad_bucket_lens,
                bucket_selection=args.bucket_selection,
                bucket_state=stream_bucket_state,
                bucket_min_fill_frac=args.bucket_min_fill_frac,
                bucket_cycle_repeat=args.bucket_cycle_repeat,
                pack_max_seq_len=min(args.max_seq_len, args.pack_max_seq_len or args.max_seq_len),
                max_images_per_row=args.pack_examples,
                pack_fixed_rows=args.pack_fixed_rows,
                boundary_aware_pack=args.boundary_aware_pack,
                flatten_packed_batch=args.flatten_packed_batch,
            )
        selected_count = len(batch_examples)
        if args.pack_examples > 1:
            batch_examples = trim_examples_to_packable(
                batch_examples,
                max_seq_len=min(args.max_seq_len, args.pack_max_seq_len or args.max_seq_len),
                max_batch_tokens=args.max_batch_tokens,
                max_images_per_row=args.pack_examples,
                fixed_rows=args.pack_fixed_rows,
                bucket_lens=pad_bucket_lens,
                boundary_aware=args.boundary_aware_pack,
                compact_token_budget=args.flatten_packed_batch,
            )
        return batch_examples, selected_count

    prepared_batches = iter_prepared_batches(
        select_train_examples,
        extractor,
        image_root=args.image_root,
        skip_bad_images=args.skip_bad_images,
        profile_timing=args.profile_timing,
        prefetch_processor=args.prefetch_processor,
    )
    using_prefetch = args.prefetch_batches > 0
    if using_prefetch:
        if args.prefetch_workers > 1:
            prepared_batches = PreparedBatchPrefetcher(
                select_train_examples,
                extractor,
                image_root=args.image_root,
                skip_bad_images=args.skip_bad_images,
                profile_timing=args.profile_timing,
                prefetch_processor=args.prefetch_processor,
                maxsize=args.prefetch_batches,
                num_workers=args.prefetch_workers,
            )
        else:
            prepared_batches = PrefetchIterator(prepared_batches, maxsize=args.prefetch_batches)

    synchronize()
    for step in range(1, args.num_iterations + 1):
        t0 = time.perf_counter()
        profile = new_profile()
        train_loss = None
        projector_optimizer.zero_grad(set_to_none=True)
        llm_optimizer.zero_grad(set_to_none=True)
        tokens_this_step = 0
        padded_tokens_this_step = 0
        supervised_targets_this_step = 0
        lm_head_tokens_this_step = 0
        padded_lm_head_tokens_this_step = 0
        seq_flops_this_step = 0.0
        seq_padded_flops_this_step = 0.0
        samples_this_step = 0
        selected_samples_this_step = 0
        dropped_samples_this_step = 0
        rows_this_step = 0
        max_seq_this_step = 0
        segments_this_step = 0
        max_segment_this_step = 0
        near_cap_segments_this_step = 0
        cap_segments_this_step = 0
        segment_lengths_this_step = []
        seq_lens_this_step = []
        attention_pairs_this_step = 0
        for _ in range(args.grad_accum_steps):
            if using_prefetch:
                t = time.perf_counter()
                prepared = next(prepared_batches)
                profile["data_wait"] += time.perf_counter() - t
            else:
                prepared = next(prepared_batches)
            merge_profile(profile, prepared.profile)
            batch_examples = prepared.examples
            if not batch_examples:
                selected_samples_this_step += prepared.selected_examples or 0
                dropped_samples_this_step += prepared.selected_examples or 0
                continue
            selected_count = prepared.selected_examples if prepared.selected_examples is not None else len(batch_examples)
            selected_samples_this_step += selected_count
            if args.profile_timing:
                synchronize()
            t_image = time.perf_counter()
            if prepared.pixel_values is not None:
                feats = extractor.encode_pixel_values(
                    prepared.pixel_values,
                    profile=profile if args.profile_timing else None,
                    synchronize=synchronize if args.profile_timing else None,
                )
            elif prepared.images is not None:
                feats = extractor(
                    prepared.images,
                    profile=profile if args.profile_timing else None,
                    synchronize=synchronize if args.profile_timing else None,
                )
            else:
                feats = None
            if feats is None:
                continue
            if not args.profile_timing:
                profile["image_siglip"] += time.perf_counter() - t_image
            t_pack = time.perf_counter()
            pack_seq_len = min(args.max_seq_len, args.pack_max_seq_len or args.max_seq_len)
            if args.flatten_packed_batch and args.boundary_aware_pack and args.pack_fixed_rows == 0:
                rows, masks, feats, image_counts, segment_lengths = flatten_examples_as_compact_batch(
                    batch_examples,
                    feats,
                    max_seq_len=pack_seq_len,
                    max_batch_tokens=args.max_batch_tokens,
                )
            else:
                packed = pack_example_rows(
                    batch_examples,
                    feats,
                    max_seq_len=pack_seq_len,
                    max_batch_tokens=args.max_batch_tokens,
                    max_images_per_row=args.pack_examples,
                    fixed_rows=args.pack_fixed_rows,
                    bucket_lens=pad_bucket_lens,
                    boundary_aware=args.boundary_aware_pack,
                    return_segment_lengths=args.boundary_aware_pack,
                    compact_token_budget=args.flatten_packed_batch,
                )
                if args.boundary_aware_pack:
                    rows, masks, feats, image_counts, segment_lengths = packed
                else:
                    rows, masks, feats, image_counts = packed
                    segment_lengths = None
            if not rows:
                dropped_samples_this_step += selected_count
                continue
            if args.pack_fixed_rows > 0 and len(rows) != args.pack_fixed_rows:
                dropped_samples_this_step += selected_count
                continue
            if args.flatten_packed_batch and len(rows) > 1:
                rows, masks, image_counts, segment_lengths = flatten_packed_rows(
                    rows,
                    masks,
                    image_counts,
                    segment_lengths,
                )
            packed_samples = packed_example_count(image_counts, segment_lengths)
            dropped_samples_this_step += max(0, selected_count - packed_samples)
            if args.profile_timing:
                synchronize()
            profile["pack"] += time.perf_counter() - t_pack

            pad_to_len = None
            if args.pad_to_max_seq_len:
                pad_to_len = args.max_seq_len

            t = time.perf_counter()
            batch = build_multimodal_batch(
                model,
                projector,
                rows,
                feats,
                loss_mask_rows=masks,
                image_counts_per_row=image_counts,
                max_seq_len=None if args.flatten_packed_batch else args.max_seq_len,
                pad_to_len=pad_to_len,
                pad_to_bucket_lens=pad_bucket_lens,
                value_fallback_token_id=rows[0][0],
                segment_token_lengths_per_row=segment_lengths,
                compact_varlen_indices=args.flatten_packed_batch,
                return_segment_ids=not args.flatten_packed_batch,
                return_segment_starts=False,
                return_boundary_metadata=args.boundary_aware_pack,
                return_loss_indices=args.selective_loss,
                return_targets=not args.selective_loss,
                return_lengths=not args.boundary_aware_pack,
                profile=profile if args.profile_timing else None,
                synchronize=synchronize if args.profile_timing else None,
            )
            if args.profile_timing:
                synchronize()
            profile["batch"] += time.perf_counter() - t

            t = time.perf_counter()
            boundary_kwargs = {}
            if args.boundary_aware_pack:
                boundary_kwargs = {
                    "position_ids": batch.position_ids,
                    "segment_ids": batch.segment_ids,
                    "segment_start_indices": batch.segment_start_indices,
                    "cu_seqlens": batch.cu_seqlens,
                    "max_seqlen": batch.max_segment_len,
                    "varlen_indices": batch.varlen_indices,
                }
            loss = model(
                batch.value_token_ids,
                batch.targets,
                input_embeds=batch.input_embeds,
                selective_loss=args.selective_loss,
                loss_chunk_size=args.loss_chunk_size,
                loss_indices=batch.loss_indices if args.selective_loss else None,
                loss_targets=batch.loss_targets if args.selective_loss else None,
                **boundary_kwargs,
            ) / args.grad_accum_steps
            loss.backward()
            if args.profile_timing:
                synchronize()
            profile["fwdbwd"] += time.perf_counter() - t
            train_loss = loss.detach() * args.grad_accum_steps
            micro_tokens = batch.token_count if batch.token_count is not None else int(batch.lengths.sum())
            micro_padded_tokens = (
                batch.padded_token_count
                if batch.padded_token_count is not None
                else int(batch.input_embeds.shape[0] * batch.input_embeds.shape[1])
            )
            micro_supervised_targets = (
                batch.supervised_target_count
                if batch.supervised_target_count is not None
                else int((batch.targets != -1).sum())
            )
            if args.selective_loss:
                useful_lm_head_tokens = micro_supervised_targets
                padded_lm_head_tokens = micro_supervised_targets
            else:
                useful_lm_head_tokens = micro_tokens
                padded_lm_head_tokens = micro_padded_tokens
            tokens_this_step += micro_tokens
            padded_tokens_this_step += micro_padded_tokens
            supervised_targets_this_step += micro_supervised_targets
            lm_head_tokens_this_step += useful_lm_head_tokens
            padded_lm_head_tokens_this_step += padded_lm_head_tokens
            batch_seq_len = int(batch.input_embeds.shape[1])
            if args.boundary_aware_pack and batch.segment_lengths:
                micro_segment_lengths = [int(length) for length in batch.segment_lengths]
                useful_seq_flops, padded_seq_flops = estimate_varlen_step_flops(
                    model,
                    micro_segment_lengths,
                    micro_padded_tokens,
                    useful_lm_head_tokens=useful_lm_head_tokens,
                    padded_lm_head_tokens=padded_lm_head_tokens,
                    cache=flops_per_token_cache,
                )
                seq_flops_this_step += useful_seq_flops
                seq_padded_flops_this_step += padded_seq_flops
                attention_pairs_this_step += int(batch.attention_pairs or 0)
                segments_this_step += len(micro_segment_lengths)
                segment_lengths_this_step.extend(micro_segment_lengths)
                max_segment_this_step = max(max_segment_this_step, max(micro_segment_lengths))
                near_cap, cap_hits = count_near_cap_segments(micro_segment_lengths, pack_seq_len)
                near_cap_segments_this_step += near_cap
                cap_segments_this_step += cap_hits
            else:
                row_lengths = [int(length) for length in batch.lengths.tolist()]
                useful_seq_flops, padded_seq_flops = estimate_dense_step_flops(
                    model,
                    sequence_len=batch_seq_len,
                    useful_tokens=micro_tokens,
                    padded_tokens=micro_padded_tokens,
                    useful_lm_head_tokens=useful_lm_head_tokens,
                    padded_lm_head_tokens=padded_lm_head_tokens,
                    cache=flops_per_token_cache,
                )
                seq_flops_this_step += useful_seq_flops
                seq_padded_flops_this_step += padded_seq_flops
                attention_pairs_this_step += causal_attention_pairs(batch.input_embeds.shape[0], batch_seq_len)
                segments_this_step += len(rows)
                max_segment_this_step = max(max_segment_this_step, batch_seq_len)
                segment_lengths_this_step.extend(row_lengths)
                segment_cap_len = pack_seq_len if args.pack_examples > 1 else args.max_seq_len
                near_cap, cap_hits = count_near_cap_segments(row_lengths, segment_cap_len)
                near_cap_segments_this_step += near_cap
                cap_segments_this_step += cap_hits
            seq_lens_this_step.append(batch_seq_len)
            max_seq_this_step = max(max_seq_this_step, batch_seq_len)
            samples_this_step += packed_samples
            rows_this_step += len(rows)
        if train_loss is None:
            raise RuntimeError("no usable images loaded for this optimizer step; check image URLs or disable --skip-bad-images")
        progress = 0.0 if args.num_iterations <= 1 else (step - 1) / (args.num_iterations - 1)
        if args.warmup_ratio > 0 and progress < args.warmup_ratio:
            lrm = (progress + 1e-8) / args.warmup_ratio
        elif args.warmdown_ratio <= 0 or progress <= 1.0 - args.warmdown_ratio:
            lrm = 1.0
        else:
            decay = (progress - (1.0 - args.warmdown_ratio)) / args.warmdown_ratio
            lrm = (1.0 - decay) + decay * args.final_lr_frac
        muon_momentum = (1 - min(step / 300, 1.0)) * 0.85 + min(step / 300, 1.0) * 0.95
        for group in llm_optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group["kind"] == "muon":
                group["momentum"] = muon_momentum
        if args.profile_timing:
            synchronize()
        t = time.perf_counter()
        projector_optimizer.step()
        if args.profile_timing:
            synchronize()
        projector_optim_elapsed = time.perf_counter() - t
        profile["optim_projector"] += projector_optim_elapsed
        t = time.perf_counter()
        llm_optimizer.step()
        if args.profile_timing:
            synchronize()
        llm_optim_elapsed = time.perf_counter() - t
        profile["optim_llm"] += llm_optim_elapsed
        profile["optim"] += projector_optim_elapsed + llm_optim_elapsed

        loss_f = float(train_loss)
        dt = time.perf_counter() - t0
        smooth_loss = 0.9 * smooth_loss + 0.1 * loss_f
        smooth_count += 1
        debiased = smooth_loss / (1 - 0.9**smooth_count)
        samples_per_sec = samples_this_step / max(dt, 1e-9)
        tokens_per_sec = tokens_this_step / max(dt, 1e-9)
        padded_tokens_per_sec = padded_tokens_this_step / max(dt, 1e-9)
        flops_per_sec = seq_flops_this_step / max(dt, 1e-9)
        mfu = 100 * flops_per_sec / gpu_peak_flops
        padded_flops_per_sec = seq_padded_flops_this_step / max(dt, 1e-9)
        padded_mfu = 100 * padded_flops_per_sec / gpu_peak_flops
        seq_mfu = 100 * seq_flops_this_step / max(dt, 1e-9) / gpu_peak_flops
        seq_padded_mfu = 100 * seq_padded_flops_this_step / max(dt, 1e-9) / gpu_peak_flops
        token_estimate_mfu = 100 * num_flops_per_token * tokens_this_step / max(dt, 1e-9) / gpu_peak_flops
        token_estimate_padded_mfu = 100 * num_flops_per_token * padded_tokens_this_step / max(dt, 1e-9) / gpu_peak_flops
        padding_frac = 1.0 - (tokens_this_step / max(padded_tokens_this_step, 1))
        attention_pairs_per_token = attention_pairs_this_step / max(tokens_this_step, 1)
        avg_segment_this_step = tokens_this_step / max(segments_this_step, 1)
        p50_segment_this_step = segment_length_percentile(segment_lengths_this_step, 0.50)
        p90_segment_this_step = segment_length_percentile(segment_lengths_this_step, 0.90)
        step_bucket = static_mfu_step_bucket(seq_lens_this_step, compact_varlen=args.flatten_packed_batch)
        bucket_seen_before = bucket_seen_steps.get(step_bucket, 0) if step_bucket else 0
        if step_bucket:
            bucket_seen_steps[step_bucket] = bucket_seen_before + 1
        count_for_mfu = should_count_mfu_step(
            step,
            args.mfu_warmup_steps,
            step_bucket,
            bucket_seen_before,
            args.mfu_warmup_bucket_steps,
        )
        if count_for_mfu:
            steady_seconds += dt
            steady_tokens += tokens_this_step
            steady_padded_tokens += padded_tokens_this_step
            steady_seq_flops += seq_flops_this_step
            steady_seq_padded_flops += seq_padded_flops_this_step
            steady_samples += samples_this_step
            steady_attention_pairs += attention_pairs_this_step
            steady_segments += segments_this_step
            add_segment_length_counts(steady_segment_length_counts, segment_lengths_this_step)
            steady_max_segment_len = max(steady_max_segment_len, max_segment_this_step)
            steady_near_cap_segments += near_cap_segments_this_step
            steady_cap_segments += cap_segments_this_step
            steady_steps += 1
            if step_bucket:
                add_bucket_steady_step(
                    steady_bucket_stats,
                    step_bucket,
                    dt,
                    tokens_this_step,
                    padded_tokens_this_step,
                    samples_this_step,
                    seq_flops=seq_flops_this_step,
                    seq_padded_flops=seq_padded_flops_this_step,
                    attention_pairs=attention_pairs_this_step,
                    segments=segments_this_step,
                    segment_lengths=segment_lengths_this_step,
                    max_segment_len=max_segment_this_step,
                    near_cap_segments=near_cap_segments_this_step,
                    cap_segments=cap_segments_this_step,
                )
        steady_tokens_per_sec = steady_tokens / max(steady_seconds, 1e-9)
        steady_samples_per_sec = steady_samples / max(steady_seconds, 1e-9)
        steady_mfu = 100 * steady_seq_flops / max(steady_seconds, 1e-9) / gpu_peak_flops
        steady_padded_mfu = 100 * steady_seq_padded_flops / max(steady_seconds, 1e-9) / gpu_peak_flops
        steady_seq_mfu = 100 * steady_seq_flops / max(steady_seconds, 1e-9) / gpu_peak_flops
        steady_seq_padded_mfu = 100 * steady_seq_padded_flops / max(steady_seconds, 1e-9) / gpu_peak_flops
        steady_token_estimate_mfu = 100 * num_flops_per_token * steady_tokens / max(steady_seconds, 1e-9) / gpu_peak_flops
        steady_token_estimate_padded_mfu = 100 * num_flops_per_token * steady_padded_tokens / max(steady_seconds, 1e-9) / gpu_peak_flops
        steady_padding_frac = 1.0 - (steady_tokens / max(steady_padded_tokens, 1))
        steady_attention_pairs_per_step = steady_attention_pairs / max(steady_steps, 1)
        steady_attention_pairs_per_token = steady_attention_pairs / max(steady_tokens, 1)
        steady_segments_per_step = steady_segments / max(steady_steps, 1)
        steady_avg_segment_len = steady_tokens / max(steady_segments, 1)
        steady_p50_segment_len = segment_length_percentile_from_counts(steady_segment_length_counts, 0.50)
        steady_p90_segment_len = segment_length_percentile_from_counts(steady_segment_length_counts, 0.90)
        steady_near_cap_segments_per_step = steady_near_cap_segments / max(steady_steps, 1)
        steady_cap_segments_per_step = steady_cap_segments / max(steady_steps, 1)
        bucket_metrics = bucket_steady_metrics(steady_bucket_stats[step_bucket], num_flops_per_token, gpu_peak_flops) if step_bucket in steady_bucket_stats else None
        if args.profile_timing:
            profile["image_siglip"] = sum(profile[key] for key in IMAGE_PROFILE_KEYS)
            if count_for_mfu:
                merge_profile(steady_profile, profile)
        if step == 1 or step % args.log_every == 0 or step == args.num_iterations:
            profile_str = ""
            bucket_str = f" | bucket {step_bucket:,}" if step_bucket else ""
            memory = cuda_memory_stats_mib(device_type)
            if bucket_metrics is not None:
                bucket_str += (
                    f" bucket_steady_mfu {bucket_metrics['mfu']:.2f} "
                    f"bucket_steady_seq_padded_mfu {bucket_metrics['seq_padded_mfu']:.2f} "
                    f"bucket_steady_attn_pairs/token {bucket_metrics['attention_pairs_per_token']:.2f} "
                    f"bucket_steady_p50_segment {bucket_metrics['p50_segment_len']:,} "
                    f"bucket_steady_p90_segment {bucket_metrics['p90_segment_len']:,} "
                    f"({bucket_metrics['steps']} steps)"
                )
            elif step_bucket and step > args.mfu_warmup_steps and bucket_seen_before < args.mfu_warmup_bucket_steps:
                bucket_str += f" bucket_warmup {bucket_seen_before + 1}/{args.mfu_warmup_bucket_steps}"
            log_data = {
                "step": step,
                "train/loss": debiased,
                "train/raw_loss": loss_f,
                "train/samples": samples_this_step,
                "train/selected_samples": selected_samples_this_step,
                "train/dropped_samples": dropped_samples_this_step,
                "train/samples_per_sec": samples_per_sec,
                "train/rows": rows_this_step,
                "train/tokens_per_sec": tokens_per_sec,
                "train/mfu": mfu,
                "train/eff_llm_mfu": mfu,
                "train/padded_tokens_per_sec": padded_tokens_per_sec,
                "train/padded_mfu": padded_mfu,
                "train/padded_llm_mfu": padded_mfu,
                "train/token_estimate_mfu": token_estimate_mfu,
                "train/token_estimate_padded_mfu": token_estimate_padded_mfu,
                "train/seq_mfu": seq_mfu,
                "train/seq_padded_mfu": seq_padded_mfu,
                "train/padding_frac": padding_frac,
                "train/max_seq_len": max_seq_this_step,
                "train/bucket_seq_len": step_bucket,
                "train/tokens": tokens_this_step,
                "train/padded_tokens": padded_tokens_this_step,
                "train/supervised_targets": supervised_targets_this_step,
                "train/lm_head_tokens": lm_head_tokens_this_step,
                "train/padded_lm_head_tokens": padded_lm_head_tokens_this_step,
                "train/attention_pairs": attention_pairs_this_step,
                "train/attention_pairs_per_token": attention_pairs_per_token,
                "train/segments": segments_this_step,
                "train/avg_segment_len": avg_segment_this_step,
                "train/p50_segment_len": p50_segment_this_step,
                "train/p90_segment_len": p90_segment_this_step,
                "train/max_segment_len": max_segment_this_step,
                "train/near_cap_segments": near_cap_segments_this_step,
                "train/cap_segments": cap_segments_this_step,
                "train/seq_flops": seq_flops_this_step,
                "train/seq_padded_flops": seq_padded_flops_this_step,
                "train/lrm": lrm,
                "train/mfu_counted": float(count_for_mfu),
                "train/bucket_seen_steps": bucket_seen_steps.get(step_bucket, 0) if step_bucket else 0,
                "train/mfu_warmup_bucket_steps": args.mfu_warmup_bucket_steps,
                "train/steady_steps": steady_steps,
                "train/steady_samples_per_sec": steady_samples_per_sec,
                "train/steady_tokens_per_sec": steady_tokens_per_sec,
                "train/steady_mfu": steady_mfu,
                "train/steady_eff_llm_mfu": steady_mfu,
                "train/steady_padded_mfu": steady_padded_mfu,
                "train/steady_padded_llm_mfu": steady_padded_mfu,
                "train/steady_token_estimate_mfu": steady_token_estimate_mfu,
                "train/steady_token_estimate_padded_mfu": steady_token_estimate_padded_mfu,
                "train/steady_seq_mfu": steady_seq_mfu,
                "train/steady_seq_padded_mfu": steady_seq_padded_mfu,
                "train/steady_padding_frac": steady_padding_frac,
                "train/steady_attention_pairs_per_step": steady_attention_pairs_per_step,
                "train/steady_attention_pairs_per_token": steady_attention_pairs_per_token,
                "train/steady_segments_per_step": steady_segments_per_step,
                "train/steady_avg_segment_len": steady_avg_segment_len,
                "train/steady_p50_segment_len": steady_p50_segment_len,
                "train/steady_p90_segment_len": steady_p90_segment_len,
                "train/steady_max_segment_len": steady_max_segment_len,
                "train/steady_near_cap_segments_per_step": steady_near_cap_segments_per_step,
                "train/steady_cap_segments_per_step": steady_cap_segments_per_step,
                "gpu/allocated_mib": memory["allocated"],
                "gpu/reserved_mib": memory["reserved"],
                "gpu/max_allocated_mib": memory["max_allocated"],
                "gpu/max_reserved_mib": memory["max_reserved"],
            }
            if bucket_metrics is not None:
                log_data.update({
                    "train/bucket_steady_steps": bucket_metrics["steps"],
                    "train/bucket_steady_tokens_per_sec": bucket_metrics["tokens_per_sec"],
                    "train/bucket_steady_padded_tokens_per_sec": bucket_metrics["padded_tokens_per_sec"],
                    "train/bucket_steady_samples_per_sec": bucket_metrics["samples_per_sec"],
                    "train/bucket_steady_mfu": bucket_metrics["mfu"],
                    "train/bucket_steady_eff_llm_mfu": bucket_metrics["mfu"],
                    "train/bucket_steady_padded_mfu": bucket_metrics["padded_mfu"],
                    "train/bucket_steady_padded_llm_mfu": bucket_metrics["padded_mfu"],
                    "train/bucket_steady_token_estimate_mfu": bucket_metrics["token_estimate_mfu"],
                    "train/bucket_steady_token_estimate_padded_mfu": bucket_metrics["token_estimate_padded_mfu"],
                    "train/bucket_steady_seq_mfu": bucket_metrics["seq_mfu"],
                    "train/bucket_steady_seq_padded_mfu": bucket_metrics["seq_padded_mfu"],
                    "train/bucket_steady_padding_frac": bucket_metrics["padding_frac"],
                    "train/bucket_steady_attention_pairs_per_step": bucket_metrics["attention_pairs_per_step"],
                    "train/bucket_steady_attention_pairs_per_token": bucket_metrics["attention_pairs_per_token"],
                    "train/bucket_steady_segments_per_step": bucket_metrics["segments_per_step"],
                    "train/bucket_steady_avg_segment_len": bucket_metrics["avg_segment_len"],
                    "train/bucket_steady_p50_segment_len": bucket_metrics["p50_segment_len"],
                    "train/bucket_steady_p90_segment_len": bucket_metrics["p90_segment_len"],
                    "train/bucket_steady_max_segment_len": bucket_metrics["max_segment_len"],
                    "train/bucket_steady_near_cap_segments_per_step": bucket_metrics["near_cap_segments_per_step"],
                    "train/bucket_steady_cap_segments_per_step": bucket_metrics["cap_segments_per_step"],
                })
            if args.profile_timing:
                other_seconds = profile_other_seconds(profile, dt)
                steady_other_seconds = profile_other_seconds(steady_profile, steady_seconds)
                profile_str = (
                    " | timing wait/data/image_total/open/processor/h2d/siglip/pool/pack/batch/batch_projector/fwdbwd/optim/projector_optim/llm_optim "
                    f"{profile['data_wait']:.3f}/{profile['data']:.3f}/{profile['image_siglip']:.3f}/{profile['image_open']:.3f}/"
                    f"{profile['image_processor']:.3f}/{profile['image_transfer']:.3f}/"
                    f"{profile['siglip_forward']:.3f}/{profile['siglip_pool']:.3f}/"
                    f"{profile['pack']:.3f}/{profile['batch']:.3f}/{profile['batch_projector']:.3f}/"
                    f"{profile['fwdbwd']:.3f}/{profile['optim']:.3f}/"
                    f"{profile['optim_projector']:.3f}/{profile['optim_llm']:.3f}s"
                )
                log_data.update({f"timing/{k}_sec": v for k, v in profile.items()})
                log_data.update({
                    "timing/other_sec": other_seconds,
                    "timing/other_frac": other_seconds / max(dt, 1e-9),
                    "timing/steady_other_sec": steady_other_seconds,
                    "timing/steady_other_frac": steady_other_seconds / max(steady_seconds, 1e-9),
                })
            print0(
                f"step {step:05d}/{args.num_iterations:05d} | loss {debiased:.6f} | "
                f"samples/sec {samples_per_sec:.2f} | samples {samples_this_step}/{selected_samples_this_step} "
                f"dropped {dropped_samples_this_step} rows {rows_this_step} | tokens/sec {tokens_per_sec:.0f} | "
                f"tokens {tokens_this_step:,}/{padded_tokens_this_step:,} max_seq {max_seq_this_step:,} "
                f"loss_tokens {supervised_targets_this_step:,} lm_head {lm_head_tokens_this_step:,}/{padded_lm_head_tokens_this_step:,} "
                f"segments {segments_this_step} avg_segment {avg_segment_this_step:.1f} "
                f"p50_segment {p50_segment_this_step:,} p90_segment {p90_segment_this_step:,} max_segment {max_segment_this_step:,} "
                f"near_cap {near_cap_segments_this_step} cap_hits {cap_segments_this_step} | "
                f"attn_pairs {format_count(attention_pairs_this_step)} attn_pairs/token {attention_pairs_per_token:.2f} | "
                f"eff_llm_mfu {mfu:.2f} | padded_llm_mfu {padded_mfu:.2f} | seq_padded_mfu {seq_padded_mfu:.2f} | "
                f"pad {100 * padding_frac:.1f}%"
                f" | steady_mfu {steady_mfu:.2f} ({steady_steps} steps after warmup)"
                f" steady_attn_pairs/token {steady_attention_pairs_per_token:.2f}"
                f" steady_avg_segment {steady_avg_segment_len:.1f} steady_p50_segment {steady_p50_segment_len:,}"
                f" steady_p90_segment {steady_p90_segment_len:,} steady_max_segment {steady_max_segment_len:,}"
                f" steady_near_cap/step {steady_near_cap_segments_per_step:.1f} steady_cap_hits/step {steady_cap_segments_per_step:.1f}"
                f" | mem alloc/peak/reserved {memory['allocated']:.0f}/{memory['max_allocated']:.0f}/{memory['reserved']:.0f}MiB"
                f"{bucket_str} | lrm {lrm:.3f}{profile_str}",
                flush=True,
            )
            wandb_run.log(log_data)
        if args.eval_every > 0 and (step % args.eval_every == 0 or step == args.num_iterations):
            run_val_bpb(step)
        if args.vlm_eval_every > 0 and (step % args.vlm_eval_every == 0 or step == args.num_iterations):
            run_vlm_eval(step)
        if not args.no_save and args.save_every > 0 and step % args.save_every == 0:
            save_training_checkpoint(out_dir, step, model, projector, args, meta, data_path, rank=ddp_rank)

    if using_prefetch and hasattr(prepared_batches, "close"):
        prepared_batches.close()

    if not args.no_save and (args.save_every <= 0 or args.num_iterations % args.save_every != 0):
        save_training_checkpoint(out_dir, args.num_iterations, model, projector, args, meta, data_path, rank=ddp_rank)

    final_memory = cuda_memory_stats_mib(device_type)
    peak_mem = final_memory["max_allocated"]
    total_time_min = (time.time() - t_start) / 60
    if steady_bucket_stats:
        print0("Bucket steady stats:", flush=True)
        for bucket, row in sorted(steady_bucket_stats.items()):
            metrics = bucket_steady_metrics(row, num_flops_per_token, gpu_peak_flops)
            print0(format_bucket_steady_line(bucket, metrics), flush=True)
    if args.profile_timing and steady_steps > 0:
        print0(format_profile_summary("Steady timing totals", steady_profile, steady_seconds), flush=True)
    print0(f"Peak memory usage: {peak_mem:.2f}MiB", flush=True)
    print0(f"Total training time: {total_time_min:.2f}m", flush=True)
    wandb_run.log({
        "gpu/peak_mem_mib": peak_mem,
        "gpu/final_allocated_mib": final_memory["allocated"],
        "gpu/final_reserved_mib": final_memory["reserved"],
        "gpu/final_max_reserved_mib": final_memory["max_reserved"],
        "train/total_time_min": total_time_min,
    })
    wandb_run.finish()
    compute_cleanup()


if __name__ == "__main__":
    main()
