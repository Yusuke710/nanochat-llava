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
    lengths: torch.Tensor
    position_ids: torch.Tensor | None = None
    segment_starts: torch.Tensor | None = None
    cu_seqlens: torch.Tensor | None = None
    max_seqlen: int | None = None
    segment_lengths: list[int] | None = None
    token_count: int | None = None
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
    exactly 64 visual tokens per image.
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
    return x.reshape(B, output_grid * output_grid, C * scale * scale).float()


class SigLIPPooledFeatureExtractor:
    """Frozen SigLIP base patch-16/512 encoder that returns 8x8 pooled features."""

    def __init__(
        self,
        model_id: str = SIGLIP_MODEL_ID,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        output_grid: int = VISION_GRID,
        cache_dir: str | None = None,
        verbose: bool = False,
    ):
        from transformers import AutoImageProcessor, SiglipVisionModel

        self.model_id = model_id
        self.cache_dir = cache_dir
        self.output_grid = output_grid
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.bfloat16 if self.device.type == "cuda" else torch.float32)
        kwargs = {"cache_dir": cache_dir} if cache_dir else {}
        if verbose:
            print(f"Loading SigLIP processor {model_id} cache={cache_dir or '<default>'}", flush=True)
        self.processor = AutoImageProcessor.from_pretrained(model_id, **kwargs)
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
    def preprocess(self, images) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt")
        return inputs["pixel_values"]

    @torch.no_grad()
    def encode_pixel_values(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype, non_blocking=self.device.type == "cuda")
        out = self.model(pixel_values=pixel_values)
        return pool_siglip_features(out.last_hidden_state, output_grid=self.output_grid)

    @torch.no_grad()
    def __call__(self, images) -> torch.Tensor:
        return self.encode_pixel_values(self.preprocess(images))


def count_text_image_markers(text: str) -> int:
    return str(text).count(IMAGE_MARKER)


def format_image_markers(image_count: int) -> str:
    image_count = int(image_count)
    if image_count <= 0:
        return ""
    if image_count == 1:
        return IMAGE_MARKER
    return "\n".join(f"Image {idx}: {IMAGE_MARKER}" for idx in range(1, image_count + 1))


def encode_with_image_markers(tokenizer, text: str, image_token_id: int = IMAGE_TOKEN_ID) -> list[int]:
    text = str(text)
    ids: list[int] = []
    parts = text.split(IMAGE_MARKER)
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


def render_caption_example(
    tokenizer,
    caption: str,
    prompt: str = f"{IMAGE_MARKER}\nDescribe the image.",
    max_tokens: int = 2048,
) -> tuple[list[int], list[int]]:
    """Render stage-1 image-caption data with caption-only loss."""
    bos = tokenizer.get_bos_token_id()
    user_start, user_end = tokenizer.encode_special("<|user_start|>"), tokenizer.encode_special("<|user_end|>")
    assistant_start, assistant_end = tokenizer.encode_special("<|assistant_start|>"), tokenizer.encode_special("<|assistant_end|>")
    ids: list[int] = []
    mask: list[int] = []
    _add_tokens(ids, mask, bos, 0)
    _add_tokens(ids, mask, user_start, 0)
    _add_tokens(ids, mask, encode_with_image_markers(tokenizer, prompt), 0)
    _add_tokens(ids, mask, user_end, 0)
    _add_tokens(ids, mask, assistant_start, 0)
    _add_tokens(ids, mask, tokenizer.encode(caption), 1)
    _add_tokens(ids, mask, assistant_end, 1)
    return ids[:max_tokens], mask[:max_tokens]


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
    """Render stage-2 visual instruction data with assistant-only loss."""
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
    image_token_id: int = IMAGE_TOKEN_ID,
    ignore_index: int = IGNORE_INDEX,
    max_seq_len: int | None = None,
    value_fallback_token_id: int | None = None,
    image_counts_per_row: list[int] | None = None,
    segment_token_lengths_per_row: list[list[int]] | None = None,
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
    if loss_mask_rows is None:
        loss_mask_rows = [[1] * len(row) for row in token_rows]
    assert len(loss_mask_rows) == batch_size
    if image_counts_per_row is None:
        image_counts_per_row = [1] * batch_size
    assert len(image_counts_per_row) == batch_size
    assert sum(image_counts_per_row) == image_features.size(0), (
        f"feature batch {image_features.size(0)} != image markers {sum(image_counts_per_row)}"
    )
    if segment_token_lengths_per_row is None:
        segment_token_lengths_per_row = [[len(row)] for row in token_rows]
    assert len(segment_token_lengths_per_row) == batch_size
    if value_fallback_token_id is None:
        value_fallback_token_id = int(token_rows[0][0])
        assert value_fallback_token_id >= 0, "first token must be a real token when no fallback is provided"
    assert image_features.ndim == 3, f"expected image_features (B, 64, C), got {image_features.shape}"

    value_rows: list[list[int]] = []
    target_rows: list[list[int]] = []
    position_rows: list[list[int]] = []
    segment_start_rows: list[list[bool]] = []
    visual_spans: list[tuple[int, int, int]] = []
    flat_segment_lengths: list[int] = []
    lengths: list[int] = []
    feature_cursor = 0

    for row_idx, (row, mask_row) in enumerate(zip(token_rows, loss_mask_rows)):
        expected_images = int(image_counts_per_row[row_idx])
        segment_token_lengths = [int(length) for length in segment_token_lengths_per_row[row_idx]]
        assert len(row) == len(mask_row), "token and mask rows must have the same length"
        assert len(row) >= 2, "need at least two tokens to build shifted targets"
        assert segment_token_lengths and all(length > 1 for length in segment_token_lengths), "segments must have at least two tokens"
        assert sum(segment_token_lengths) == len(row), "segment lengths must sum to row length"
        values: list[int] = []
        targets_row: list[int] = []
        positions: list[int] = []
        segment_starts_row: list[bool] = []
        segment_idx = 0
        segment_end = segment_token_lengths[0]
        segment_pos = 0
        row_images = 0

        for i, tok in enumerate(row[:-1]):
            while i >= segment_end and segment_idx + 1 < len(segment_token_lengths):
                segment_idx += 1
                segment_end += segment_token_lengths[segment_idx]
                segment_pos = 0
            tok = int(tok)
            if i + 1 == segment_end and segment_idx + 1 < len(segment_token_lengths):
                assert tok != image_token_id, "packed segment boundary token cannot be an image marker"
                flat_segment_lengths.append(segment_pos)
                continue
            next_tok = int(row[i + 1])
            supervise_next = int(mask_row[i + 1]) == 1
            if tok == image_token_id:
                assert row_images < expected_images, f"row {row_idx} has more <image> markers than expected"
                visual_spans.append((row_idx, len(values), feature_cursor))
                feature_cursor += 1
                row_images += 1
                values.extend([value_fallback_token_id] * VISION_TOKENS)
                targets_row.extend([ignore_index] * VISION_TOKENS)
                positions.extend(range(segment_pos, segment_pos + VISION_TOKENS))
                segment_starts_row.extend([segment_pos == 0] + [False] * (VISION_TOKENS - 1))
                segment_pos += VISION_TOKENS
            else:
                target = next_tok if next_tok != image_token_id and supervise_next else ignore_index
                values.append(tok)
                targets_row.append(target)
                positions.append(segment_pos)
                segment_starts_row.append(segment_pos == 0)
                segment_pos += 1
        assert row_images == expected_images, f"consumed {row_images} image features, expected {expected_images}"
        flat_segment_lengths.append(segment_pos)

        if max_seq_len is not None and len(values) > max_seq_len:
            values = values[:max_seq_len]
            targets_row = targets_row[:max_seq_len]
            positions = positions[:max_seq_len]
            segment_starts_row = segment_starts_row[:max_seq_len]
            flat_segment_lengths[-1] = min(flat_segment_lengths[-1], max_seq_len)
        value_rows.append(values)
        target_rows.append(targets_row)
        position_rows.append(positions)
        segment_start_rows.append(segment_starts_row)
        lengths.append(len(values))
    assert feature_cursor == image_features.size(0), f"consumed {feature_cursor} image features, had {image_features.size(0)}"

    max_len = max(lengths)
    value_token_ids = torch.tensor(
        [row + [value_fallback_token_id] * (max_len - len(row)) for row in value_rows],
        dtype=torch.long,
        device=device,
    )
    targets = torch.tensor(
        [row + [ignore_index] * (max_len - len(row)) for row in target_rows],
        dtype=torch.long,
        device=device,
    )
    position_ids = torch.tensor(
        [row + [0] * (max_len - len(row)) for row in position_rows],
        dtype=torch.long,
        device=device,
    )
    segment_starts = torch.tensor(
        [row + [False] * (max_len - len(row)) for row in segment_start_rows],
        dtype=torch.bool,
        device=device,
    )
    input_embeds = wte(value_token_ids)
    if visual_spans:
        assert image_features.size(1) == VISION_TOKENS, f"expected {VISION_TOKENS} visual tokens, got {image_features.size(1)}"
        feats = image_features
        if hasattr(feats, "is_inference") and feats.is_inference():
            feats = feats.clone()
        feats = feats.to(device=device, dtype=wte.weight.dtype if wte.weight.dtype != torch.float16 else torch.float32)
        projected = projector(feats.reshape(-1, feats.size(-1))).to(device=device, dtype=input_embeds.dtype)
        projected = projected.view(image_features.size(0), VISION_TOKENS, -1)
        for row_idx, start, feature_idx in visual_spans:
            end = min(start + VISION_TOKENS, max_len)
            if end > start:
                input_embeds[row_idx, start:end] = projected[feature_idx, : end - start]
    cu_values = [0]
    for length in flat_segment_lengths:
        cu_values.append(cu_values[-1] + int(length))
    cu_seqlens = torch.tensor(cu_values, dtype=torch.int32, device=device)
    flat_targets = targets.view(-1)
    loss_indices = (flat_targets != ignore_index).nonzero(as_tuple=False).flatten()
    loss_targets = flat_targets.index_select(0, loss_indices)
    return MultimodalBatch(
        input_embeds=input_embeds,
        value_token_ids=value_token_ids,
        targets=targets,
        lengths=torch.tensor(lengths, dtype=torch.long, device=device),
        position_ids=position_ids,
        segment_starts=segment_starts,
        cu_seqlens=cu_seqlens,
        max_seqlen=max(flat_segment_lengths),
        segment_lengths=flat_segment_lengths,
        token_count=sum(lengths),
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
