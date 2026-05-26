"""
Minimal LLaVA-style vision helpers for nanochat.

The main idea is intentionally small:
- keep the language model unchanged for text-only callers
- represent a literal "<image>" marker as an out-of-vocab sentinel while rendering
- replace that sentinel with 64 projected SigLIP visual-token features before GPT.forward()
"""

from __future__ import annotations

import copy
import math
import os
import time
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn

from nanochat.checkpoint_manager import load_checkpoint, save_checkpoint
from nanochat.gpt import Linear

IMAGE_MARKER = "<image>"
IMAGE_TOKEN_ID = -200
IGNORE_INDEX = -1
SIGLIP_MODEL_ID = "google/siglip-base-patch16-512"
VISION_GRID = 8
VISION_TOKENS = VISION_GRID * VISION_GRID
CHECKPOINT_SOURCE_DIRS = {
    "base": "base_checkpoints",
    "sft": "chatsft_checkpoints",
    "rl": "chatrl_checkpoints",
}


@dataclass
class MultimodalBatch:
    input_embeds: torch.Tensor
    value_token_ids: torch.Tensor
    targets: torch.Tensor | None
    lengths: torch.Tensor | None
    position_ids: torch.Tensor | None = None
    segment_ids: torch.Tensor | None = None
    segment_starts: torch.Tensor | None = None
    segment_start_indices: torch.Tensor | None = None
    cu_seqlens: torch.Tensor | None = None
    max_segment_len: int | None = None
    varlen_indices: torch.Tensor | None = None
    segment_lengths: list[int] | None = None
    attention_pairs: int | None = None
    token_count: int | None = None
    padded_token_count: int | None = None
    supervised_target_count: int | None = None
    loss_indices: torch.Tensor | None = None
    loss_targets: torch.Tensor | None = None


class VisionProjector(nn.Module):
    """Single linear map from pooled vision feature dim to GPT embedding dim."""

    def __init__(self, vision_dim: int, n_embd: int):
        super().__init__()
        self.vision_dim = vision_dim
        self.n_embd = n_embd
        self.proj = Linear(vision_dim, n_embd, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size(-1) == self.vision_dim, f"expected vision dim {self.vision_dim}, got {x.size(-1)}"
        return self.proj(x)


def freeze_module(module: nn.Module) -> nn.Module:
    module.eval()
    for p in module.parameters():
        p.requires_grad = False
    return module


def pool_siglip_features(features: torch.Tensor, output_grid: int = VISION_GRID) -> torch.Tensor:
    """
    Compress patch features to an output_grid x output_grid grid using nanoVLM-style pixel shuffle.

    SigLIP base patch-16/512 produces a 32x32 patch grid. Instead of averaging
    each 4x4 patch block, concatenate the 16 patch vectors into the channel
    dimension so the projector still sees fine visual detail while the LLM sees
    exactly 64 visual tokens per image. Preserve the encoder dtype; the projector
    path already casts features to the GPT embedding dtype when needed.
    """
    if features.ndim == 3:
        B, N, C = features.shape
        grid = int(math.isqrt(N))
        if grid * grid != N:
            grid = int(math.isqrt(N - 1))
            assert grid * grid == N - 1, f"cannot infer square patch grid from {N} tokens"
            features = features[:, 1:, :]  # tolerate encoders that prepend a CLS token
        H = W = grid
        x = features.view(B, grid, grid, C)
    elif features.ndim == 4:
        B, H, W, C = features.shape
        assert H == W, f"expected square patch grid, got {H}x{W}"
        x = features
    else:
        raise ValueError(f"expected features with 3 or 4 dims, got {features.shape}")

    assert H % output_grid == 0, f"patch grid {H}x{W} is not divisible by output grid {output_grid}"
    scale = H // output_grid
    x = x.reshape(B, output_grid, scale, output_grid, scale, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.reshape(B, output_grid * output_grid, C * scale * scale)


class SigLIPPooledFeatureExtractor:
    """Frozen SigLIP base patch-16/512 encoder that returns 8x8 pooled features."""

    def __init__(
        self,
        model_id: str = SIGLIP_MODEL_ID,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        output_grid: int = VISION_GRID,
        cache_dir: str | None = None,
        processor_use_fast: bool = True,
        forward_batch_size: int = 0,
        verbose: bool = False,
    ):
        from transformers import AutoImageProcessor, SiglipVisionModel

        self.model_id = model_id
        self.cache_dir = cache_dir
        self.output_grid = output_grid
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.bfloat16 if self.device.type == "cuda" else torch.float32)
        self.forward_batch_size = max(0, int(forward_batch_size))
        kwargs = {"cache_dir": cache_dir} if cache_dir else {}
        if verbose:
            print(f"Loading SigLIP processor {model_id} cache={cache_dir or '<default>'}", flush=True)
        self.processor = AutoImageProcessor.from_pretrained(model_id, use_fast=processor_use_fast, **kwargs)
        if verbose:
            print(f"Loading SigLIP vision model {model_id} cache={cache_dir or '<default>'}", flush=True)
        self.model = SiglipVisionModel.from_pretrained(model_id, **kwargs).to(device=self.device, dtype=self.dtype)
        freeze_module(self.model)
        patch_grid = int(self.model.config.image_size) // int(self.model.config.patch_size)
        assert patch_grid % output_grid == 0, f"SigLIP patch grid {patch_grid} is not divisible by output grid {output_grid}"
        self.patch_dim = self.model.config.hidden_size
        self.vision_dim = self.patch_dim * (patch_grid // output_grid) ** 2
        if verbose:
            print(f"Loaded SigLIP vision model patch_dim={self.patch_dim} projected_feature_dim={self.vision_dim}", flush=True)

    @torch.no_grad()
    def preprocess(self, images, profile=None) -> torch.Tensor:
        t = time.perf_counter()
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"]
        if self.device.type == "cuda":
            pixel_values = pixel_values.pin_memory()
        if profile is not None:
            profile["image_processor"] += time.perf_counter() - t
        return pixel_values

    def _encode_pixel_values_chunk(self, pixel_values: torch.Tensor, profile=None, synchronize=None) -> torch.Tensor:
        sync = synchronize or (lambda: None)
        if profile is not None:
            sync()

        t = time.perf_counter()
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype, non_blocking=True)
        if profile is not None:
            sync()
            profile["image_transfer"] += time.perf_counter() - t

        t = time.perf_counter()
        out = self.model(pixel_values=pixel_values)
        if profile is not None:
            sync()
            profile["siglip_forward"] += time.perf_counter() - t

        t = time.perf_counter()
        pooled = pool_siglip_features(out.last_hidden_state, output_grid=self.output_grid)
        if profile is not None:
            sync()
            profile["siglip_pool"] += time.perf_counter() - t
        return pooled

    @torch.no_grad()
    def encode_pixel_values(self, pixel_values: torch.Tensor, profile=None, synchronize=None) -> torch.Tensor:
        batch_size = self.forward_batch_size
        if batch_size <= 0 or pixel_values.size(0) <= batch_size:
            return self._encode_pixel_values_chunk(pixel_values, profile=profile, synchronize=synchronize)
        chunks = []
        for start in range(0, pixel_values.size(0), batch_size):
            chunks.append(
                self._encode_pixel_values_chunk(
                    pixel_values[start : start + batch_size],
                    profile=profile,
                    synchronize=synchronize,
                )
            )
        return torch.cat(chunks, dim=0)

    @torch.no_grad()
    def __call__(self, images, profile=None, synchronize=None) -> torch.Tensor:
        pixel_values = self.preprocess(images, profile=profile)
        return self.encode_pixel_values(pixel_values, profile=profile, synchronize=synchronize)


def encode_with_image_markers(tokenizer, text: str, image_token_id: int = IMAGE_TOKEN_ID) -> list[int]:
    parts = text.split(IMAGE_MARKER)
    ids: list[int] = []
    for i, part in enumerate(parts):
        if part:
            ids.extend(tokenizer.encode(part))
        if i != len(parts) - 1:
            ids.append(image_token_id)
    return ids


def count_image_tokens(token_ids: Iterable[int], image_token_id: int = IMAGE_TOKEN_ID) -> int:
    return sum(1 for tok in token_ids if int(tok) == image_token_id)


def _add_tokens(ids: list[int], mask: list[int], token_ids, mask_val: int):
    if isinstance(token_ids, int):
        token_ids = [token_ids]
    ids.extend(int(t) for t in token_ids)
    mask.extend([mask_val] * len(token_ids))


def _normalize_messages(conversation: dict) -> list[dict]:
    if "messages" in conversation:
        messages = copy.deepcopy(conversation["messages"])
    elif "conversations" in conversation:
        role_map = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant"}
        messages = [
            {"role": role_map[m.get("from", m.get("role"))], "content": m.get("value", m.get("content", ""))}
            for m in conversation["conversations"]
        ]
    else:
        raise KeyError("conversation must contain 'messages' or 'conversations'")
    if messages and messages[0]["role"] == "system":
        assert len(messages) > 1 and messages[1]["role"] == "user", "system message must be followed by user"
        messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
        messages = messages[1:]
    return messages


def render_vision_conversation(tokenizer, conversation: dict, max_tokens: int = 2048) -> tuple[list[int], list[int]]:
    """Render visual instruction data with assistant-only loss."""
    messages = _normalize_messages(conversation)
    assert len(messages) >= 1, "empty conversation"
    bos = tokenizer.get_bos_token_id()
    user_start, user_end = tokenizer.encode_special("<|user_start|>"), tokenizer.encode_special("<|user_end|>")
    assistant_start, assistant_end = tokenizer.encode_special("<|assistant_start|>"), tokenizer.encode_special("<|assistant_end|>")
    ids: list[int] = []
    mask: list[int] = []
    _add_tokens(ids, mask, bos, 0)
    for i, message in enumerate(messages):
        expected = "user" if i % 2 == 0 else "assistant"
        assert message["role"] == expected, f"message {i} is {message['role']}, expected {expected}"
        content = message["content"]
        assert isinstance(content, str), "v0 vision conversations expect string content"
        if message["role"] == "user":
            _add_tokens(ids, mask, user_start, 0)
            _add_tokens(ids, mask, encode_with_image_markers(tokenizer, content), 0)
            _add_tokens(ids, mask, user_end, 0)
        else:
            _add_tokens(ids, mask, assistant_start, 0)
            _add_tokens(ids, mask, tokenizer.encode(content), 1)
            _add_tokens(ids, mask, assistant_end, 1)
    return ids[:max_tokens], mask[:max_tokens]


def _unwrap_model(model):
    if hasattr(model, "module"):
        model = model.module
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


def build_multimodal_batch(
    model,
    projector: VisionProjector,
    token_rows: list[list[int]],
    image_features: torch.Tensor,
    loss_mask_rows: list[list[int]] | None = None,
    image_counts_per_row: list[int] | None = None,
    image_token_id: int = IMAGE_TOKEN_ID,
    ignore_index: int = IGNORE_INDEX,
    max_seq_len: int | None = None,
    pad_to_len: int | None = None,
    pad_to_bucket_lens: list[int] | None = None,
    value_fallback_token_id: int | None = None,
    segment_token_lengths_per_row: list[list[int]] | None = None,
    compact_varlen_indices: bool = False,
    return_segment_ids: bool = True,
    return_segment_starts: bool = True,
    return_boundary_metadata: bool = True,
    return_loss_indices: bool = True,
    return_targets: bool = True,
    return_lengths: bool = True,
    profile: dict | None = None,
    synchronize=None,
) -> MultimodalBatch:
    """
    Build GPT inputs by expanding each image sentinel in token_rows[:-1] to 64 projected tokens.

    Targets are shifted by one token, matching nanochat's text loaders. Image
    sentinel targets and all expanded visual-token positions are ignored.
    """
    base_model = _unwrap_model(model)
    wte = base_model.transformer.wte
    device = wte.weight.device
    batch_size = len(token_rows)
    assert batch_size > 0
    assert not (pad_to_len is not None and pad_to_bucket_lens), "pad_to_len and pad_to_bucket_lens are mutually exclusive"
    if compact_varlen_indices:
        assert pad_to_len is None and not pad_to_bucket_lens, "compact varlen batches cannot add row padding"
    if not return_boundary_metadata:
        return_segment_ids = False
        return_segment_starts = False
    if not return_segment_ids and return_boundary_metadata:
        assert compact_varlen_indices and max_seq_len is None, "omitting segment ids is only supported for untruncated compact varlen batches"
    if loss_mask_rows is None:
        loss_mask_rows = [[1] * len(row) for row in token_rows]
    assert len(loss_mask_rows) == batch_size
    if segment_token_lengths_per_row is None:
        segment_token_lengths_per_row = [[len(row)] for row in token_rows]
    assert len(segment_token_lengths_per_row) == batch_size
    if value_fallback_token_id is None:
        value_fallback_token_id = int(token_rows[0][0])
        assert value_fallback_token_id >= 0, "first token must be a real token when no fallback is provided"
    assert image_features.ndim == 3, f"expected image_features (B, 64, C), got {image_features.shape}"
    assert image_features.size(1) == VISION_TOKENS, f"expected {VISION_TOKENS} visual tokens, got {image_features.size(1)}"
    if image_counts_per_row is None:
        image_counts_per_row = [1] * batch_size
        assert image_features.size(0) == batch_size, f"feature batch {image_features.size(0)} != token batch {batch_size}"
    assert len(image_counts_per_row) == batch_size
    assert sum(image_counts_per_row) == image_features.size(0), (
        f"feature batch {image_features.size(0)} != image markers {sum(image_counts_per_row)}"
    )
    derive_position_ids_from_segments = return_boundary_metadata and compact_varlen_indices and max_seq_len is None
    derive_segment_start_indices_from_segments = return_boundary_metadata and compact_varlen_indices and max_seq_len is None

    value_rows: list[list[int]] = []
    target_rows: list[list[int]] | None = [] if return_targets else None
    position_rows: list[list[int]] | None = None if not return_boundary_metadata or derive_position_ids_from_segments else []
    segment_rows: list[list[int]] | None = [] if return_segment_ids else None
    segment_start_rows: list[list[bool]] | None = [] if return_segment_starts else None
    segment_start_index_rows: list[list[int]] | None = None if derive_segment_start_indices_from_segments else []
    expanded_segment_length_rows: list[list[int]] | None = [] if return_boundary_metadata else None
    loss_index_rows: list[list[int]] | None = [] if return_loss_indices else None
    loss_target_rows: list[list[int]] | None = [] if return_loss_indices else None
    visual_spans: list[tuple[int, int, int]] = []
    lengths: list[int] = []
    supervised_target_count = 0
    embed_dtype = wte.weight.dtype
    feature_cursor = 0

    for row_idx, (row, mask_row) in enumerate(zip(token_rows, loss_mask_rows)):
        expected_images = int(image_counts_per_row[row_idx])
        segment_token_lengths = [int(length) for length in segment_token_lengths_per_row[row_idx]]
        assert len(row) == len(mask_row), "token and mask rows must have the same length"
        assert len(row) >= 2, "need at least two tokens to build shifted targets"
        assert segment_token_lengths and all(length > 0 for length in segment_token_lengths), "segment lengths must be positive"
        assert sum(segment_token_lengths) == len(row), "segment lengths must sum to the token row length"
        assert expected_images > 0, "each multimodal row must contain at least one image"
        value_row: list[int] = []
        target_row: list[int] | None = [] if return_targets else None
        position_row: list[int] | None = [] if position_rows is not None else None
        segment_row: list[int] | None = [] if return_segment_ids else None
        segment_start_row: list[bool] | None = [] if return_segment_starts else None
        segment_start_index_row: list[int] | None = [] if segment_start_index_rows is not None else None
        loss_index_row: list[int] | None = [] if return_loss_indices else None
        loss_target_row: list[int] | None = [] if return_loss_indices else None
        expanded_segment_lengths: list[int] = []
        row_images = 0
        segment_idx = 0
        segment_end = segment_token_lengths[0]
        segment_pos = 0
        for i, tok in enumerate(row[:-1]):
            while i >= segment_end and segment_idx + 1 < len(segment_token_lengths):
                segment_idx += 1
                segment_end += segment_token_lengths[segment_idx]
                segment_pos = 0
            tok = int(tok)
            if i + 1 == segment_end and segment_idx + 1 < len(segment_token_lengths):
                assert tok != image_token_id, "packed segment boundary token cannot be an image marker"
                expanded_segment_lengths.append(segment_pos)
                continue
            next_tok = int(row[i + 1])
            supervise_next = int(mask_row[i + 1]) == 1
            if tok == image_token_id:
                value_pos = len(value_row)
                assert row_images < expected_images, f"row {row_idx} has more <image> markers than expected {expected_images}"
                assert feature_cursor < image_features.size(0), (
                    f"image feature cursor {feature_cursor} exceeds feature batch {image_features.size(0)}"
                )
                row_images += 1
                visual_spans.append((row_idx, value_pos, feature_cursor))
                feature_cursor += 1
                value_row.extend([value_fallback_token_id] * VISION_TOKENS)
                if target_row is not None:
                    target_row.extend([ignore_index] * VISION_TOKENS)
                if position_row is not None:
                    position_row.extend(range(segment_pos, segment_pos + VISION_TOKENS))
                if segment_row is not None:
                    segment_row.extend([segment_idx] * VISION_TOKENS)
                if segment_pos == 0 and segment_start_index_row is not None:
                    segment_start_index_row.append(value_pos)
                if segment_start_row is not None:
                    segment_start_row.extend([segment_pos == 0] + [False] * (VISION_TOKENS - 1))
                segment_pos += VISION_TOKENS
            else:
                value_pos = len(value_row)
                value_row.append(tok)
                target = next_tok if next_tok != image_token_id and supervise_next else ignore_index
                if target_row is not None:
                    target_row.append(target)
                if target != ignore_index:
                    if max_seq_len is None or value_pos < max_seq_len:
                        supervised_target_count += 1
                        if loss_index_row is not None:
                            assert loss_target_row is not None
                            loss_index_row.append(value_pos)
                            loss_target_row.append(target)
                if position_row is not None:
                    position_row.append(segment_pos)
                if segment_row is not None:
                    segment_row.append(segment_idx)
                if segment_pos == 0 and segment_start_index_row is not None:
                    segment_start_index_row.append(value_pos)
                if segment_start_row is not None:
                    segment_start_row.append(segment_pos == 0)
                segment_pos += 1
        assert row_images == expected_images, f"consumed {row_images} image features, expected {expected_images}"
        expanded_segment_lengths.append(segment_pos)

        if max_seq_len is not None and len(value_row) > max_seq_len:
            value_row = value_row[:max_seq_len]
            if target_row is not None:
                target_row = target_row[:max_seq_len]
            if position_row is not None:
                position_row = position_row[:max_seq_len]
            if segment_row is not None:
                segment_row = segment_row[:max_seq_len]
            if segment_start_row is not None:
                segment_start_row = segment_start_row[:max_seq_len]
            if segment_start_index_row is not None:
                segment_start_index_row = [idx for idx in segment_start_index_row if idx < max_seq_len]
            if loss_index_row is not None:
                assert loss_target_row is not None
                kept_losses = [(idx, target) for idx, target in zip(loss_index_row, loss_target_row) if idx < max_seq_len]
                loss_index_row = [idx for idx, _ in kept_losses]
                loss_target_row = [target for _, target in kept_losses]
        value_rows.append(value_row)
        if target_rows is not None:
            assert target_row is not None
            target_rows.append(target_row)
        if position_rows is not None:
            assert position_row is not None
            position_rows.append(position_row)
        if loss_index_rows is not None:
            assert loss_target_rows is not None
            assert loss_index_row is not None and loss_target_row is not None
            loss_index_rows.append(loss_index_row)
            loss_target_rows.append(loss_target_row)
        if segment_rows is not None:
            assert segment_row is not None
            segment_rows.append(segment_row)
        if segment_start_rows is not None:
            assert segment_start_row is not None
            segment_start_rows.append(segment_start_row)
        if segment_start_index_rows is not None:
            assert segment_start_index_row is not None
            segment_start_index_rows.append(segment_start_index_row)
        if expanded_segment_length_rows is not None:
            expanded_segment_length_rows.append(expanded_segment_lengths)
        lengths.append(len(value_row))
    assert feature_cursor == image_features.size(0), f"consumed {feature_cursor} image features, had {image_features.size(0)}"

    max_len = max(lengths)
    if compact_varlen_indices:
        assert sum(lengths) == batch_size * max_len, "compact varlen batches require no implicit row padding"
    if pad_to_bucket_lens:
        bucket_len = next((int(bucket) for bucket in pad_to_bucket_lens if max_len <= int(bucket)), None)
        assert bucket_len is not None, f"longest row {max_len} exceeds largest bucket {max(pad_to_bucket_lens)}"
        max_len = bucket_len
    elif pad_to_len is not None:
        assert pad_to_len >= max_len, f"pad_to_len {pad_to_len} < longest row {max_len}"
        max_len = pad_to_len
    padded_value_rows = []
    padded_target_rows = [] if return_targets else None
    padded_position_rows = [] if position_rows is not None else None
    padded_segment_rows = [] if return_segment_ids else None
    padded_segment_start_rows = [] if return_segment_starts else None
    for row_idx, value_row in enumerate(value_rows):
        target_row = target_rows[row_idx] if target_rows is not None else None
        position_row = position_rows[row_idx] if position_rows is not None else None
        segment_row = segment_rows[row_idx] if segment_rows is not None else None
        segment_start_row = segment_start_rows[row_idx] if segment_start_rows is not None else None
        pad = max_len - len(value_row)
        if pad > 0:
            padded_value_rows.append(value_row + [value_fallback_token_id] * pad)
            if padded_target_rows is not None:
                assert target_row is not None
                padded_target_rows.append(target_row + [ignore_index] * pad)
            if padded_position_rows is not None:
                assert position_row is not None
                padded_position_rows.append(position_row + [0] * pad)
            if padded_segment_rows is not None:
                padded_segment_rows.append(segment_row + [segment_row[-1] if segment_row else 0] * pad)
            if padded_segment_start_rows is not None:
                assert segment_start_row is not None
                padded_segment_start_rows.append(segment_start_row + [False] * pad)
        else:
            padded_value_rows.append(value_row)
            if padded_target_rows is not None:
                assert target_row is not None
                padded_target_rows.append(target_row)
            if padded_position_rows is not None:
                assert position_row is not None
                padded_position_rows.append(position_row)
            if padded_segment_rows is not None:
                padded_segment_rows.append(segment_row)
            if padded_segment_start_rows is not None:
                assert segment_start_row is not None
                padded_segment_start_rows.append(segment_start_row)
    value_token_ids = torch.tensor(padded_value_rows, dtype=torch.long, device=device)
    targets = torch.tensor(padded_target_rows, dtype=torch.long, device=device) if padded_target_rows is not None else None
    token_count = sum(lengths)
    padded_token_count = batch_size * max_len
    if return_loss_indices:
        assert loss_index_rows is not None and loss_target_rows is not None
        loss_indices_list = []
        loss_targets_list = []
        for row_idx, (loss_index_row, loss_target_row) in enumerate(zip(loss_index_rows, loss_target_rows)):
            row_offset = row_idx * max_len
            loss_indices_list.extend(row_offset + pos for pos in loss_index_row)
            loss_targets_list.extend(loss_target_row)
        assert len(loss_targets_list) == supervised_target_count
        loss_indices = torch.tensor(loss_indices_list, dtype=torch.long, device=device)
        loss_targets = torch.tensor(loss_targets_list, dtype=torch.long, device=device)
    else:
        loss_indices = None
        loss_targets = None
    lengths_tensor = torch.tensor(lengths, dtype=torch.long, device=device) if return_lengths else None
    position_ids = None
    segment_ids = None
    segment_starts = None
    segment_start_indices = None
    cu_seqlens = None
    varlen_indices = None
    flat_segment_lengths = None
    max_segment_len = None
    attention_pairs = None
    if return_boundary_metadata:
        segment_ids = None if padded_segment_rows is None else torch.tensor(padded_segment_rows, dtype=torch.long, device=device)
        segment_starts = None if padded_segment_start_rows is None else torch.tensor(padded_segment_start_rows, dtype=torch.bool, device=device)
        smear_row_len = max(max_len - 1, 0)
        derive_compact_segment_start_indices = (
            segment_start_index_rows is None
            and compact_varlen_indices
            and segment_rows is None
            and batch_size == 1
        )
        segment_start_indices_list = None if derive_compact_segment_start_indices else []
        if segment_start_index_rows is None and segment_start_indices_list is not None:
            assert expanded_segment_length_rows is not None
            for row_idx, segment_lengths in enumerate(expanded_segment_length_rows):
                start = 0
                for seg_len in segment_lengths:
                    if start > 0:
                        segment_start_indices_list.append(row_idx * smear_row_len + start - 1)
                    start += seg_len
        else:
            if segment_start_indices_list is not None:
                assert segment_start_index_rows is not None
                for row_idx, starts in enumerate(segment_start_index_rows):
                    segment_start_indices_list.extend(row_idx * smear_row_len + start - 1 for start in starts if start > 0)
        cu_values = [0]
        flat_indices = None if compact_varlen_indices else []
        flat_segment_lengths = []
        max_segment_len = 0
        if segment_rows is None:
            assert flat_indices is None
            assert expanded_segment_length_rows is not None
            for segment_lengths in expanded_segment_length_rows:
                for seg_len in segment_lengths:
                    max_segment_len = max(max_segment_len, seg_len)
                    flat_segment_lengths.append(seg_len)
                    cu_values.append(cu_values[-1] + seg_len)
        else:
            for row_idx, (segment_row, row_len) in enumerate(zip(segment_rows, lengths)):
                if row_len <= 0:
                    continue
                start = 0
                while start < row_len:
                    seg = segment_row[start]
                    end = start + 1
                    while end < row_len and segment_row[end] == seg:
                        end += 1
                    seg_len = end - start
                    max_segment_len = max(max_segment_len, seg_len)
                    flat_segment_lengths.append(seg_len)
                    cu_values.append(cu_values[-1] + seg_len)
                    if flat_indices is not None:
                        flat_indices.extend(row_idx * max_len + pos for pos in range(start, end))
                    start = end
        cu_seqlens = torch.tensor(cu_values, dtype=torch.int32, device=device)
        if derive_compact_segment_start_indices:
            segment_start_indices = torch.tensor([start - 1 for start in cu_values[1:-1]], dtype=torch.long, device=device)
        else:
            assert segment_start_indices_list is not None
            segment_start_indices = torch.tensor(segment_start_indices_list, dtype=torch.long, device=device)
        varlen_indices = None if flat_indices is None else torch.tensor(flat_indices, dtype=torch.long, device=device)
        attention_pairs = sum(length * (length + 1) // 2 for length in flat_segment_lengths)
        if padded_position_rows is None:
            assert compact_varlen_indices, "segment-derived positions are only used for compact varlen batches"
            segment_lengths_tensor = torch.tensor(flat_segment_lengths, dtype=torch.long, device=device)
            segment_offsets = torch.tensor(cu_values[:-1], dtype=torch.long, device=device)
            position_ids = torch.arange(cu_values[-1], dtype=torch.long, device=device)
            position_ids = position_ids - segment_offsets.repeat_interleave(segment_lengths_tensor)
            position_ids = position_ids.view(batch_size, max_len)
        else:
            position_ids = torch.tensor(padded_position_rows, dtype=torch.long, device=device)

    input_embeds = wte(value_token_ids)
    if visual_spans:
        feats = image_features
        if hasattr(feats, "is_inference") and feats.is_inference():
            feats = feats.clone()
        feats = feats.to(device=device, dtype=embed_dtype if embed_dtype != torch.float16 else torch.float32)
        if synchronize is not None:
            synchronize()
        t_projector = time.perf_counter() if profile is not None else None
        projected = projector(feats.reshape(-1, feats.size(-1))).to(device=device, dtype=input_embeds.dtype)
        if t_projector is not None:
            if synchronize is not None:
                synchronize()
            profile["batch_projector"] = profile.get("batch_projector", 0.0) + (time.perf_counter() - t_projector)
        projected = projected.view(feats.size(0), VISION_TOKENS, -1)
        full_spans = max_seq_len is None or all(start + VISION_TOKENS <= lengths[row_idx] for row_idx, start, _ in visual_spans)
        if full_spans:
            starts = torch.tensor([start for _, start, _ in visual_spans], dtype=torch.long, device=device).unsqueeze(1)
            offsets = torch.arange(VISION_TOKENS, dtype=torch.long, device=device).unsqueeze(0)
            if input_embeds.size(0) == 1:
                positions = (starts + offsets).reshape(-1)
                input_embeds[0].index_copy_(0, positions, projected.reshape(-1, projected.size(-1)))
            else:
                row_idx = torch.tensor([row_idx for row_idx, _, _ in visual_spans], dtype=torch.long, device=device).unsqueeze(1)
                positions = starts + offsets
                input_embeds[row_idx, positions] = projected
        else:
            span_meta = torch.tensor(visual_spans, dtype=torch.long, device=device)
            span_lengths = lengths_tensor
            if span_lengths is None:
                span_lengths = torch.tensor(lengths, dtype=torch.long, device=device)
            offsets = torch.arange(VISION_TOKENS, dtype=torch.long, device=device).unsqueeze(0)
            span_lens = (span_lengths[span_meta[:, 0]] - span_meta[:, 1]).clamp(min=0, max=VISION_TOKENS)
            valid = offsets < span_lens.unsqueeze(1)
            rows = span_meta[:, 0:1].expand(-1, VISION_TOKENS)[valid]
            positions = (span_meta[:, 1:2] + offsets)[valid]
            features = span_meta[:, 2:3].expand(-1, VISION_TOKENS)[valid]
            feature_offsets = offsets.expand(span_meta.size(0), -1)[valid]
            input_embeds[rows, positions] = projected[features, feature_offsets]
    return MultimodalBatch(
        input_embeds=input_embeds,
        value_token_ids=value_token_ids,
        targets=targets,
        lengths=lengths_tensor,
        position_ids=position_ids,
        segment_ids=segment_ids,
        segment_starts=segment_starts,
        segment_start_indices=segment_start_indices,
        cu_seqlens=cu_seqlens,
        max_segment_len=max_segment_len,
        varlen_indices=varlen_indices,
        segment_lengths=flat_segment_lengths,
        attention_pairs=attention_pairs,
        token_count=token_count,
        padded_token_count=padded_token_count,
        supervised_target_count=supervised_target_count,
        loss_indices=loss_indices,
        loss_targets=loss_targets,
    )


@torch.inference_mode()
def generate_vision(
    model,
    projector: VisionProjector,
    tokenizer,
    prompt_tokens: list[int],
    image_features: torch.Tensor,
    max_tokens: int = 32,
    temperature: float = 0.0,
    top_k: int | None = None,
    seed: int = 42,
) -> list[int]:
    """Naive no-KV-cache multimodal generation for small verifier subsets."""
    device = _unwrap_model(model).get_device()
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    tokens = list(prompt_tokens)
    for _ in range(max_tokens):
        row = tokens + [tokens[-1]]
        mask = [0] * len(row)
        batch = build_multimodal_batch(model, projector, [row], image_features, loss_mask_rows=[mask], value_fallback_token_id=tokenizer.get_bos_token_id())
        batch.targets = None
        logits = model(batch.value_token_ids, input_embeds=batch.input_embeds)
        logits = logits[:, batch.lengths[0].item() - 1, :]
        if top_k is not None and top_k > 0:
            vals, idx = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
            logits = torch.full_like(logits, -float("inf")).scatter(1, idx, vals)
        if temperature == 0.0:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1, generator=rng)
        tok = int(next_id.item())
        tokens.append(tok)
        try:
            if tok in {tokenizer.encode_special("<|assistant_end|>"), tokenizer.get_bos_token_id()}:
                break
        except Exception:
            pass
    return tokens[len(prompt_tokens):]


def save_vlm_checkpoint(checkpoint_dir, step, model, projector, optimizer_data, meta_data, rank=0):
    model_data = {
        "model": _unwrap_model(model).state_dict(),
        "projector": projector.state_dict(),
        "projector_config": {"vision_dim": projector.vision_dim, "n_embd": projector.n_embd},
    }
    save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=rank)


def load_vlm_checkpoint(checkpoint_dir, step, device, load_optimizer=False, rank=0, checkpoint_device=None):
    load_device = device if checkpoint_device is None else checkpoint_device
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, step, load_device, load_optimizer=load_optimizer, rank=rank)
    config = model_data["projector_config"]
    projector = VisionProjector(**config).to(device=device)
    projector.load_state_dict(model_data["projector"], strict=True)
    return model_data["model"], projector, optimizer_data, meta_data


def ensure_hf_nanochat_checkpoint(repo_id: str, base_dir: str, model_tag: str = "d32", source: str = "sft") -> str:
    """
    Link karpathy/nanochat-d32 files from the Hugging Face cache into nanochat's
    local checkpoint layout.
    """
    from huggingface_hub import snapshot_download

    if source not in CHECKPOINT_SOURCE_DIRS:
        raise ValueError(f"unknown checkpoint source {source!r}; expected one of {sorted(CHECKPOINT_SOURCE_DIRS)}")
    snapshot = snapshot_download(repo_id=repo_id)
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    checkpoint_dir = os.path.join(base_dir, CHECKPOINT_SOURCE_DIRS[source], model_tag)
    os.makedirs(tokenizer_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    for name in ["tokenizer.pkl", "token_bytes.pt"]:
        src = os.path.join(snapshot, name)
        dst = os.path.join(tokenizer_dir, name)
        if os.path.exists(src) and not os.path.exists(dst):
            os.symlink(src, dst)
    for name in os.listdir(snapshot):
        if name.startswith(("model_", "meta_", "optim_")) and name.endswith((".pt", ".json")):
            src = os.path.join(snapshot, name)
            dst = os.path.join(checkpoint_dir, name)
            if not os.path.exists(dst):
                os.symlink(src, dst)
    return checkpoint_dir


def ensure_hf_nanochat_tokenizer(repo_id: str, base_dir: str) -> str:
    """Link only tokenizer files needed by CPU-only data diagnostics."""
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(repo_id=repo_id, allow_patterns=["tokenizer.pkl", "token_bytes.pt"])
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    os.makedirs(tokenizer_dir, exist_ok=True)
    for name in ["tokenizer.pkl", "token_bytes.pt"]:
        src = os.path.join(snapshot, name)
        dst = os.path.join(tokenizer_dir, name)
        if os.path.exists(src) and not os.path.exists(dst):
            os.symlink(src, dst)
    return tokenizer_dir
