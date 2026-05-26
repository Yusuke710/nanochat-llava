"""
Unified Flash Attention interface with automatic FA3/SDPA switching.

Exports `flash_attn` module that matches the FA3 API exactly, but falls back
to PyTorch SDPA on non-Hopper GPUs (including Blackwell), MPS, and CPU.

Usage (drop-in replacement for FA3):
    from nanochat.flash_attention import flash_attn

    # Training (no KV cache)
    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)

    # Inference (with KV cache)
    y = flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, ...)
"""
import torch
import torch.nn.functional as F


# =============================================================================
# Detection: Try to load FA3 on Hopper+ GPUs
# =============================================================================
def _load_flash_attention_3():
    """Try to load Flash Attention 3 (requires Hopper GPU, sm90)."""
    if not torch.cuda.is_available():
        return None
    try:
        major, _ = torch.cuda.get_device_capability()
        # FA3 kernels are compiled for Hopper (sm90) only
        # Ada (sm89), Blackwell (sm100) need SDPA fallback until FA3 is recompiled
        if major != 9:
            return None
        import os
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from kernels import get_kernel
        return get_kernel('varunneal/flash-attention-3').flash_attn_interface
    except Exception:
        return None


_fa3 = _load_flash_attention_3()
HAS_FA3 = _fa3 is not None

# Override for testing: set to 'fa3', 'sdpa', or None (auto)
_override_impl = None


def _resolve_use_fa3():
    """Decide once whether to use FA3, based on availability, override, and dtype."""
    if _override_impl == 'fa3':
        assert HAS_FA3, "Cannot override to FA3: not available on this hardware"
        return True
    if _override_impl == 'sdpa':
        return False
    if HAS_FA3:
        # FA3 Hopper kernels only support bf16 and fp8; fp16/fp32 must use SDPA fallback
        from nanochat.common import COMPUTE_DTYPE
        if COMPUTE_DTYPE == torch.bfloat16:
            return True
        return False
    return False

USE_FA3 = _resolve_use_fa3()


def has_fa3_varlen():
    return bool(USE_FA3 and _fa3 is not None and hasattr(_fa3, "flash_attn_varlen_func"))


def attention_backend_info():
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        cuda_capability = f"sm{capability[0]}{capability[1]}"
    else:
        cuda_capability = "none"
    return {
        "has_fa3": HAS_FA3,
        "use_fa3": USE_FA3,
        "has_fa3_varlen": has_fa3_varlen(),
        "dense_backend": "fa3" if USE_FA3 else "sdpa",
        "varlen_backend": "fa3" if has_fa3_varlen() else "sdpa",
        "cuda_capability": cuda_capability,
    }


def require_fa3_varlen():
    if not has_fa3_varlen():
        raise RuntimeError(f"FA3 varlen attention is required but unavailable: {attention_backend_info()}")


# =============================================================================
# SDPA helpers
# =============================================================================
def _sdpa_segment_attention(q, k, v, window_size, enable_gqa, segment_ids):
    """
    Dense SDPA fallback for packed training rows.

    q, k, v are (B, H, T, D). segment_ids is (B, T), where equal ids denote
    tokens from the same packed segment. The mask is causal and block-diagonal
    by segment. Padding ids are harmless: padded queries may attend within their
    padding segment, but real tokens cannot attend forward into padding.
    """
    B, _, Tq, _ = q.shape
    Tk = k.size(2)
    assert Tq == Tk, "packed segment attention is training-only and expects square q/k lengths"
    assert segment_ids.shape == (B, Tq), f"segment ids shape {segment_ids.shape} != {(B, Tq)}"
    device = q.device
    row_idx = torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx

    window = window_size[0]
    if window >= 0 and window < Tk:
        mask = mask & ((row_idx - col_idx) <= window)

    same_segment = segment_ids[:, :, None] == segment_ids[:, None, :]
    mask = same_segment & mask.unsqueeze(0)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, :, :], enable_gqa=enable_gqa)


def _sdpa_varlen_attention(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, window_size):
    """
    SDPA fallback for flattened varlen training attention.

    q is (total_q, Hq, D), k/v are (total_k, Hkv, D). cu_seqlens tensors split
    them into independent causal sequences. This is for correctness tests and
    non-Hopper paths; H100 should use FA3's varlen kernel.
    """
    del max_seqlen_q, max_seqlen_k
    q_offsets = [int(x) for x in cu_seqlens_q.detach().cpu()]
    k_offsets = [int(x) for x in cu_seqlens_k.detach().cpu()]
    assert len(q_offsets) == len(k_offsets), "q/k varlen batches must have the same number of segments"
    outputs = []
    for qs, qe, ks, ke in zip(q_offsets[:-1], q_offsets[1:], k_offsets[:-1], k_offsets[1:]):
        if qe == qs:
            continue
        qi = q[qs:qe].unsqueeze(0).transpose(1, 2)
        ki = k[ks:ke].unsqueeze(0).transpose(1, 2)
        vi = v[ks:ke].unsqueeze(0).transpose(1, 2)
        enable_gqa = qi.size(1) != ki.size(1)
        yi = _sdpa_attention(qi, ki, vi, window_size, enable_gqa)
        outputs.append(yi.transpose(1, 2).squeeze(0))
    if not outputs:
        return q.new_empty((0, q.size(1), q.size(2)))
    return torch.cat(outputs, dim=0)


def _sdpa_attention(q, k, v, window_size, enable_gqa):
    """
    SDPA attention with sliding window support.
    q, k, v are (B, H, T, D) format.
    """
    Tq = q.size(2)
    Tk = k.size(2)
    window = window_size[0]

    # Full context, same length
    if (window < 0 or window >= Tq) and Tq == Tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)

    # Single token generation
    if Tq == 1:
        if window >= 0 and window < Tk:
            # window is "left" tokens we need to include (window + 1) keys total
            start = max(0, Tk - (window + 1))
            k = k[:, :, start:, :]
            v = v[:, :, start:, :]
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)

    # Need explicit mask for sliding window/chunk inference
    device = q.device
    # For chunk inference (Tq != Tk), is_causal is not aligned to cache position => build an explicit bool mask
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx

    # sliding window (left)
    if window >= 0 and window < Tk:
        mask = mask & ((row_idx - col_idx) <= window)

    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)

# =============================================================================
# Public API: Same interface as FA3
# =============================================================================
def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1), segment_ids=None):
    """
    Flash Attention for training (no KV cache).

    Args:
        q, k, v: Tensors of shape (B, T, H, D)
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T, H, D)
    """
    if USE_FA3 and segment_ids is None:
        return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)

    # SDPA fallback: transpose (B, T, H, D) -> (B, H, T, D)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    enable_gqa = q.size(1) != k.size(1)
    if segment_ids is not None:
        y = _sdpa_segment_attention(q, k, v, window_size, enable_gqa, segment_ids)
    else:
        y = _sdpa_attention(q, k, v, window_size, enable_gqa)
    return y.transpose(1, 2)  # back to (B, T, H, D)


def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    causal=False,
    window_size=(-1, -1),
):
    """
    Flash Attention varlen training wrapper.

    q/k/v are flattened as (total_tokens, heads, head_dim). cu_seqlens split
    the flattened tokens into independent sequences. The signature follows the
    FA3 Hopper interface used by `varunneal/flash-attention-3`.
    """
    if USE_FA3 and hasattr(_fa3, "flash_attn_varlen_func"):
        return _fa3.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            causal=causal,
            window_size=window_size,
        )
    return _sdpa_varlen_attention(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, window_size)


def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
                            causal=False, window_size=(-1, -1)):
    """
    Flash Attention with KV cache for inference.

    FA3 updates k_cache/v_cache in-place. Our SDPA fallback does the same.

    Args:
        q: Queries, shape (B, T_new, H, D)
        k_cache, v_cache: Pre-allocated cache tensors, shape (B, T_max, H_kv, D)
        k, v: New keys/values to insert, shape (B, T_new, H_kv, D)
        cache_seqlens: Current position in cache, shape (B,) int32
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T_new, H, D)
    """
    if USE_FA3:
        return _fa3.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
            causal=causal, window_size=window_size
        )

    # SDPA fallback: manually manage KV cache
    B, T_new, H, D = q.shape
    pos = cache_seqlens[0].item()  # assume uniform position across batch

    # Insert new k, v into cache (in-place, matching FA3 behavior)
    if k is not None and v is not None:
        k_cache[:, pos:pos+T_new, :, :] = k
        v_cache[:, pos:pos+T_new, :, :] = v

    # Get full cache up to current position + new tokens
    end_pos = pos + T_new
    k_full = k_cache[:, :end_pos, :, :]
    v_full = v_cache[:, :end_pos, :, :]

    # Transpose to SDPA layout: (B, T, H, D) -> (B, H, T, D)
    q_sdpa = q.transpose(1, 2)
    k_sdpa = k_full.transpose(1, 2)
    v_sdpa = v_full.transpose(1, 2)

    enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)
    y_sdpa = _sdpa_attention(q_sdpa, k_sdpa, v_sdpa, window_size, enable_gqa)

    return y_sdpa.transpose(1, 2)  # back to (B, T, H, D)


# =============================================================================
# Export: flash_attn module interface (drop-in replacement for FA3)
# =============================================================================
from types import SimpleNamespace
flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_varlen_func=flash_attn_varlen_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
    attention_backend_info=attention_backend_info,
    has_fa3_varlen=has_fa3_varlen,
    require_fa3_varlen=require_fa3_varlen,
)
