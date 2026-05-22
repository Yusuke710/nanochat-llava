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
import torch.nn.functional as F

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
    def __call__(self, images) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device=self.device, dtype=self.dtype)
        out = self.model(pixel_values=pixel_values)
        return pool_siglip_features(out.last_hidden_state, output_grid=self.output_grid)


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
    if value_fallback_token_id is None:
        value_fallback_token_id = int(token_rows[0][0])
        assert value_fallback_token_id >= 0, "first token must be a real token when no fallback is provided"
    assert image_features.ndim == 3, f"expected image_features (B, 64, C), got {image_features.shape}"
    assert image_features.size(0) == batch_size, f"feature batch {image_features.size(0)} != token batch {batch_size}"

    embed_rows: list[torch.Tensor] = []
    value_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    lengths: list[int] = []
    embed_dtype = wte.weight.dtype

    for row_idx, (row, mask_row) in enumerate(zip(token_rows, loss_mask_rows)):
        assert len(row) == len(mask_row), "token and mask rows must have the same length"
        assert len(row) >= 2, "need at least two tokens to build shifted targets"
        assert count_image_tokens(row[:-1], image_token_id) == 1, "v0 expects exactly one <image> marker per row"
        chunks: list[torch.Tensor] = []
        value_chunks: list[torch.Tensor] = []
        target_chunks: list[torch.Tensor] = []

        for i, tok in enumerate(row[:-1]):
            tok = int(tok)
            next_tok = int(row[i + 1])
            supervise_next = int(mask_row[i + 1]) == 1
            if tok == image_token_id:
                feats = image_features[row_idx]
                assert feats.shape[0] == VISION_TOKENS, f"expected {VISION_TOKENS} visual tokens, got {feats.shape[0]}"
                if hasattr(feats, "is_inference") and feats.is_inference():
                    feats = feats.clone()
                feats = feats.to(device=device, dtype=embed_dtype if embed_dtype != torch.float16 else torch.float32)
                visual_embeds = projector(feats).to(device=device)
                chunks.append(visual_embeds)
                value_chunks.append(torch.full((VISION_TOKENS,), value_fallback_token_id, dtype=torch.long, device=device))
                visual_targets = torch.full((VISION_TOKENS,), ignore_index, dtype=torch.long, device=device)
                target_chunks.append(visual_targets)
            else:
                token = torch.tensor([tok], dtype=torch.long, device=device)
                chunks.append(wte(token))
                value_chunks.append(token)
                target = next_tok if next_tok != image_token_id and supervise_next else ignore_index
                target_chunks.append(torch.tensor([target], dtype=torch.long, device=device))

        embeds = torch.cat(chunks, dim=0)
        values = torch.cat(value_chunks, dim=0)
        targets = torch.cat(target_chunks, dim=0)
        if max_seq_len is not None and embeds.size(0) > max_seq_len:
            embeds = embeds[:max_seq_len]
            values = values[:max_seq_len]
            targets = targets[:max_seq_len]
        embed_rows.append(embeds)
        value_rows.append(values)
        target_rows.append(targets)
        lengths.append(embeds.size(0))

    max_len = max(lengths)
    pad_embed = wte(torch.tensor([value_fallback_token_id], dtype=torch.long, device=device))[0]
    input_embeds = torch.stack([
        torch.cat([row, pad_embed.expand(max_len - row.size(0), -1)], dim=0) if row.size(0) < max_len else row
        for row in embed_rows
    ])
    value_token_ids = torch.stack([
        F.pad(row, (0, max_len - row.size(0)), value=value_fallback_token_id) if row.size(0) < max_len else row
        for row in value_rows
    ])
    targets = torch.stack([
        F.pad(row, (0, max_len - row.size(0)), value=ignore_index) if row.size(0) < max_len else row
        for row in target_rows
    ])
    return MultimodalBatch(
        input_embeds=input_embeds,
        value_token_ids=value_token_ids,
        targets=targets,
        lengths=torch.tensor(lengths, dtype=torch.long, device=device),
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
