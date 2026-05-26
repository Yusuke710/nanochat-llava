import json
import inspect
import subprocess
import sys
import types
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from nanochat.checkpoint_manager import _patch_missing_keys
from nanochat.gpt import GPT, GPTConfig
from nanochat.vision import (
    IMAGE_MARKER,
    IMAGE_TOKEN_ID,
    VISION_TOKENS,
    SigLIPPooledFeatureExtractor,
    VisionProjector,
    build_multimodal_batch,
    count_image_tokens,
    encode_with_image_markers,
    ensure_hf_nanochat_checkpoint,
    ensure_hf_nanochat_tokenizer,
    load_vlm_checkpoint,
    pool_siglip_features,
    render_vision_conversation,
    save_vlm_checkpoint,
)
from scripts import vlm_train
from scripts.vlm_eval import (
    benchmark_specs,
    coerce_options,
    evaluate_vlm,
    exact_or_choice_match,
    get_answers,
    make_prompt,
    make_result_sample,
    parse_inline_options,
)
from scripts.vlm_train import (
    batch_features_and_examples,
    evaluate_vlm_bpb,
    iter_hf_records,
    iter_rendered_examples,
    image_record_is_openable,
    load_records,
    next_batch,
    next_stream_batch,
    open_image,
    pack_example_rows,
    prepare_training_batch,
    render_records,
    save_training_checkpoint,
    supervised_target_count,
)


class TinyTokenizer:
    def __init__(self):
        self.special = {
            "<|bos|>": 1,
            "<|user_start|>": 2,
            "<|user_end|>": 3,
            "<|assistant_start|>": 4,
            "<|assistant_end|>": 5,
        }

    def get_bos_token_id(self):
        return self.special["<|bos|>"]

    def encode_special(self, text):
        return self.special[text]

    def encode(self, text):
        return [20 + (b % 80) for b in text.encode("utf-8")]

    def decode(self, ids):
        return "".join(chr((i - 20) % 80) for i in ids if i >= 20)

    def get_vocab_size(self):
        return 128


def tiny_model():
    torch.manual_seed(123)
    config = GPTConfig(sequence_len=128, vocab_size=128, n_layer=2, n_head=2, n_kv_head=2, n_embd=32, window_pattern="L")
    model = GPT(config)
    model.init_weights()
    model.eval()
    projector = VisionProjector(vision_dim=8, n_embd=config.n_embd)
    projector.eval()
    return model, projector


def test_old_nanochat_checkpoint_keys_are_neutral_patched():
    config = GPTConfig(sequence_len=32, vocab_size=128, n_layer=3, n_head=2, n_kv_head=2, n_embd=32, window_pattern="L")
    model = GPT(config)
    model.init_weights()
    model_data = {key: value.clone() for key, value in model.state_dict().items()}
    for key in list(model_data):
        if (
            key in {"resid_lambdas", "x0_lambdas", "smear_lambda", "backout_lambda", "smear_gate.weight"}
            or key.startswith("value_embeds.")
            or key.endswith(".attn.ve_gate.weight")
        ):
            del model_data[key]

    _patch_missing_keys(model_data, config)
    model.load_state_dict(model_data, strict=True, assign=True)
    assert torch.all(model.resid_lambdas == 1)
    assert torch.all(model.x0_lambdas == 0)
    assert torch.all(model.smear_lambda == 0)
    assert torch.all(model.backout_lambda == 0)
    for weight in model.value_embeds.state_dict().values():
        assert torch.all(weight == 0)


def test_drop_zero_value_embedding_path_preserves_outputs():
    config = GPTConfig(sequence_len=32, vocab_size=128, n_layer=3, n_head=2, n_kv_head=2, n_embd=32, window_pattern="L")
    model = GPT(config)
    model.init_weights()
    model.eval()
    for weight in model.value_embeds.parameters():
        weight.data.zero_()

    idx = torch.randint(0, config.vocab_size, (2, 12))
    with torch.no_grad():
        before = model(idx)
        dropped = model.drop_value_embedding_path()
        after = model(idx)

    assert dropped["value_embed_params"] > 0
    assert dropped["total_params"] >= dropped["value_embed_params"]
    assert len(model.value_embeds) == 0
    for block in model.transformer.h:
        assert block.attn.ve_gate is None
    torch.testing.assert_close(after, before)


def test_pool_siglip_features_uses_nanovlm_pixel_shuffle():
    feats = torch.randn(2, 32 * 32, 8)
    pooled = pool_siglip_features(feats)
    assert pooled.shape == (2, VISION_TOKENS, 8 * 16)

    feats_with_cls = torch.randn(2, 1 + 32 * 32, 8)
    assert pool_siglip_features(feats_with_cls).shape == (2, VISION_TOKENS, 8 * 16)

    ordered = torch.arange(32 * 32, dtype=torch.float32).view(1, 32 * 32, 1)
    shuffled = pool_siglip_features(ordered)
    expected = torch.tensor([0, 1, 2, 3, 32, 33, 34, 35, 64, 65, 66, 67, 96, 97, 98, 99], dtype=torch.float32)
    torch.testing.assert_close(shuffled[0, 0], expected)


def test_pool_siglip_features_preserves_encoder_dtype():
    feats = torch.randn(1, 32 * 32, 8, dtype=torch.bfloat16)
    pooled = pool_siglip_features(feats)
    assert pooled.dtype == torch.bfloat16


def test_siglip_forward_batch_size_chunks_encoder_forward():
    class FakeSigLIP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.batch_sizes = []

        def forward(self, pixel_values):
            self.batch_sizes.append(pixel_values.size(0))
            base = pixel_values.flatten(1).sum(dim=1).view(-1, 1, 1)
            patch_offsets = torch.arange(16, dtype=pixel_values.dtype).view(1, 16, 1)
            channels = torch.arange(2, dtype=pixel_values.dtype).view(1, 1, 2)
            return SimpleNamespace(last_hidden_state=base + patch_offsets + channels)

    extractor = object.__new__(SigLIPPooledFeatureExtractor)
    extractor.device = torch.device("cpu")
    extractor.dtype = torch.float32
    extractor.output_grid = 2
    extractor.forward_batch_size = 2
    extractor.model = FakeSigLIP()

    pixels = torch.arange(5 * 3, dtype=torch.float32).view(5, 3, 1, 1)
    chunked = extractor.encode_pixel_values(pixels)
    assert extractor.model.batch_sizes == [2, 2, 1]

    extractor.forward_batch_size = 0
    extractor.model.batch_sizes.clear()
    unchunked = extractor.encode_pixel_values(pixels)
    assert extractor.model.batch_sizes == [5]
    torch.testing.assert_close(chunked, unchunked)


def test_projector_forward_shape_and_dtype():
    projector = VisionProjector(vision_dim=8, n_embd=32)
    features = torch.randn(3, VISION_TOKENS, 8, requires_grad=True)
    out = projector(features)
    assert out.shape == (3, VISION_TOKENS, 32)
    out.square().mean().backward()
    assert projector.proj.weight.grad is not None

    bf16_out = projector(torch.randn(1, VISION_TOKENS, 8, dtype=torch.bfloat16))
    assert bf16_out.dtype == torch.bfloat16
    assert projector.proj.weight.dtype == torch.float32


def test_image_marker_encoding_and_rendering():
    tokenizer = TinyTokenizer()
    ids = encode_with_image_markers(tokenizer, f"look {IMAGE_MARKER} now")
    assert count_image_tokens(ids) == 1

    conv_ids, conv_mask = render_vision_conversation(
        tokenizer,
        {"messages": [{"role": "user", "content": f"{IMAGE_MARKER}\nWhat?"}, {"role": "assistant", "content": "Answer."}]},
    )
    assert count_image_tokens(conv_ids) == 1
    assert conv_mask[conv_ids.index(tokenizer.encode_special("<|assistant_start|>"))] == 0
    assert sum(conv_mask) > 0


def test_visual_token_insertion_and_target_masking():
    model, projector = tiny_model()
    row = [1, 10, IMAGE_TOKEN_ID, 11, 12]
    mask = [1] * len(row)
    features = torch.randn(1, VISION_TOKENS, 8)
    profile = vlm_train.new_profile()
    batch = build_multimodal_batch(
        model,
        projector,
        [row],
        features,
        loss_mask_rows=[mask],
        value_fallback_token_id=1,
        profile=profile,
    )
    assert profile["batch_projector"] > 0
    assert batch.input_embeds.shape == (1, len(row) - 1 + VISION_TOKENS - 1, model.config.n_embd)
    assert batch.token_count == len(row) - 1 + VISION_TOKENS - 1
    assert batch.padded_token_count == batch.token_count
    assert batch.supervised_target_count == 2
    assert batch.loss_indices.tolist() == [0, 66]
    assert batch.loss_targets.tolist() == [10, 12]
    targets = batch.targets[0]
    assert targets[0].item() == 10
    assert targets[1].item() == -1
    assert torch.all(targets[2 : 2 + VISION_TOKENS] == -1)
    assert targets[-1].item() == 12
    assert torch.isfinite(model(batch.value_token_ids, batch.targets, input_embeds=batch.input_embeds))

    with pytest.raises(AssertionError, match="expected 1"):
        build_multimodal_batch(model, projector, [[1, 10, 11]], features, loss_mask_rows=[[1, 1, 1]], value_fallback_token_id=1)


def test_multimodal_batch_can_pad_to_fixed_len():
    model, projector = tiny_model()
    row = [1, 10, IMAGE_TOKEN_ID, 11, 12]
    features = torch.randn(1, VISION_TOKENS, 8)
    pad_to_len = len(row) - 1 + VISION_TOKENS - 1 + 8
    batch = build_multimodal_batch(model, projector, [row], features, pad_to_len=pad_to_len, value_fallback_token_id=1)
    assert batch.input_embeds.shape == (1, pad_to_len, model.config.n_embd)
    assert batch.value_token_ids.shape == (1, pad_to_len)
    assert batch.targets.shape == (1, pad_to_len)
    assert batch.lengths.item() == len(row) - 1 + VISION_TOKENS - 1
    assert batch.token_count == len(row) - 1 + VISION_TOKENS - 1
    assert batch.padded_token_count == pad_to_len
    assert torch.all(batch.targets[0, batch.lengths.item():] == -1)


def test_multimodal_batch_can_skip_selective_loss_indices():
    model, projector = tiny_model()
    row = [1, 10, IMAGE_TOKEN_ID, 11, 12]
    mask = [1] * len(row)
    features = torch.randn(1, VISION_TOKENS, 8)
    batch = build_multimodal_batch(
        model,
        projector,
        [row],
        features,
        loss_mask_rows=[mask],
        value_fallback_token_id=1,
        return_loss_indices=False,
    )
    assert batch.supervised_target_count == 2
    assert batch.loss_indices is None
    assert batch.loss_targets is None
    assert torch.isfinite(model(batch.value_token_ids, batch.targets, input_embeds=batch.input_embeds, selective_loss=False))

    clipped = build_multimodal_batch(
        model,
        projector,
        [row],
        features,
        loss_mask_rows=[mask],
        max_seq_len=3,
        value_fallback_token_id=1,
        return_loss_indices=False,
    )
    assert clipped.supervised_target_count == 1
    assert clipped.loss_indices is None
    assert clipped.loss_targets is None
    assert clipped.targets.tolist() == [[10, -1, -1]]


def test_multimodal_batch_can_skip_dense_targets_for_selective_loss():
    model, projector = tiny_model()
    row = [1, 10, IMAGE_TOKEN_ID, 11, 12]
    mask = [1] * len(row)
    features = torch.randn(1, VISION_TOKENS, 8)
    dense_batch = build_multimodal_batch(
        model,
        projector,
        [row],
        features,
        loss_mask_rows=[mask],
        value_fallback_token_id=1,
    )
    sparse_batch = build_multimodal_batch(
        model,
        projector,
        [row],
        features,
        loss_mask_rows=[mask],
        value_fallback_token_id=1,
        return_targets=False,
    )
    assert sparse_batch.targets is None
    assert sparse_batch.supervised_target_count == dense_batch.supervised_target_count
    torch.testing.assert_close(sparse_batch.loss_indices, dense_batch.loss_indices)
    torch.testing.assert_close(sparse_batch.loss_targets, dense_batch.loss_targets)
    sparse_loss = model(
        sparse_batch.value_token_ids,
        sparse_batch.targets,
        input_embeds=sparse_batch.input_embeds,
        selective_loss=True,
        loss_indices=sparse_batch.loss_indices,
        loss_targets=sparse_batch.loss_targets,
    )
    dense_loss = model(
        dense_batch.value_token_ids,
        dense_batch.targets,
        input_embeds=dense_batch.input_embeds,
        selective_loss=True,
        loss_indices=dense_batch.loss_indices,
        loss_targets=dense_batch.loss_targets,
    )
    torch.testing.assert_close(sparse_loss, dense_loss, rtol=1e-6, atol=1e-6)


def test_multimodal_batch_can_skip_lengths_tensor_for_training_counts():
    model, projector = tiny_model()
    row = [1, 10, IMAGE_TOKEN_ID, 11, 12]
    mask = [1] * len(row)
    features = torch.randn(1, VISION_TOKENS, 8)
    batch = build_multimodal_batch(
        model,
        projector,
        [row],
        features,
        loss_mask_rows=[mask],
        value_fallback_token_id=1,
        return_lengths=False,
    )
    assert batch.lengths is None
    assert batch.token_count == len(row) - 1 + VISION_TOKENS - 1
    assert batch.padded_token_count == batch.token_count
    assert batch.supervised_target_count == 2
    assert torch.isfinite(model(batch.value_token_ids, batch.targets, input_embeds=batch.input_embeds))


def test_multimodal_batch_can_skip_boundary_metadata_for_dense_training():
    model, projector = tiny_model()
    row = [1, 10, IMAGE_TOKEN_ID, 11, 12]
    mask = [1] * len(row)
    features = torch.randn(1, VISION_TOKENS, 8)
    full_batch = build_multimodal_batch(
        model,
        projector,
        [row],
        features,
        loss_mask_rows=[mask],
        value_fallback_token_id=1,
    )
    lean_batch = build_multimodal_batch(
        model,
        projector,
        [row],
        features,
        loss_mask_rows=[mask],
        value_fallback_token_id=1,
        return_boundary_metadata=False,
    )
    assert lean_batch.position_ids is None
    assert lean_batch.segment_ids is None
    assert lean_batch.segment_starts is None
    assert lean_batch.segment_start_indices is None
    assert lean_batch.cu_seqlens is None
    assert lean_batch.max_segment_len is None
    assert lean_batch.varlen_indices is None
    assert lean_batch.segment_lengths is None
    assert lean_batch.attention_pairs is None
    torch.testing.assert_close(lean_batch.input_embeds, full_batch.input_embeds)
    torch.testing.assert_close(lean_batch.value_token_ids, full_batch.value_token_ids)
    torch.testing.assert_close(lean_batch.targets, full_batch.targets)
    torch.testing.assert_close(lean_batch.lengths, full_batch.lengths)
    lean_loss = model(lean_batch.value_token_ids, lean_batch.targets, input_embeds=lean_batch.input_embeds)
    full_loss = model(full_batch.value_token_ids, full_batch.targets, input_embeds=full_batch.input_embeds)
    torch.testing.assert_close(lean_loss, full_loss, rtol=1e-6, atol=1e-6)


def test_multimodal_batch_can_skip_lengths_tensor_with_truncated_image_span():
    model, projector = tiny_model()
    row = [1, 10, IMAGE_TOKEN_ID, 11, 12]
    features = torch.randn(1, VISION_TOKENS, 8)
    batch = build_multimodal_batch(
        model,
        projector,
        [row],
        features,
        max_seq_len=3,
        value_fallback_token_id=1,
        return_lengths=False,
    )
    assert batch.lengths is None
    assert batch.input_embeds.shape == (1, 3, model.config.n_embd)
    assert batch.token_count == 3
    assert batch.padded_token_count == 3


def test_multimodal_batch_can_pad_to_static_bucket():
    model, projector = tiny_model()
    row = [1, 10, IMAGE_TOKEN_ID, 11, 12]
    features = torch.randn(1, VISION_TOKENS, 8)
    batch = build_multimodal_batch(
        model,
        projector,
        [row],
        features,
        pad_to_bucket_lens=[96, 128],
        value_fallback_token_id=1,
    )
    assert batch.input_embeds.shape == (1, 96, model.config.n_embd)
    assert batch.value_token_ids.shape == (1, 96)
    assert batch.targets.shape == (1, 96)
    assert batch.lengths.item() == len(row) - 1 + VISION_TOKENS - 1
    assert batch.token_count == len(row) - 1 + VISION_TOKENS - 1
    assert batch.padded_token_count == 96
    assert torch.all(batch.targets[0, batch.lengths.item():] == -1)


def test_multimodal_batch_counts_survive_truncated_image_span():
    model, projector = tiny_model()
    row = [1, 10, 11, IMAGE_TOKEN_ID, 12]
    features = torch.randn(1, VISION_TOKENS, 8)
    batch = build_multimodal_batch(
        model,
        projector,
        [row],
        features,
        max_seq_len=2,
        value_fallback_token_id=1,
    )
    assert batch.input_embeds.shape == (1, 2, model.config.n_embd)
    assert batch.token_count == 2
    assert batch.padded_token_count == 2
    assert batch.supervised_target_count == 2
    assert batch.loss_indices.tolist() == [0, 1]
    assert batch.loss_targets.tolist() == [10, 11]


def test_multimodal_batch_allows_packed_image_rows():
    model, projector = tiny_model()
    row = [1, 10, IMAGE_TOKEN_ID, 11, 12, 1, 13, IMAGE_TOKEN_ID, 14, 15]
    mask = [1, 1, 0, 1, 1, 0, 0, 0, 1, 1]
    features = torch.randn(2, VISION_TOKENS, 8)
    batch = build_multimodal_batch(
        model,
        projector,
        [row],
        features,
        loss_mask_rows=[mask],
        image_counts_per_row=[2],
        value_fallback_token_id=1,
    )
    assert batch.input_embeds.shape[0] == 1
    assert batch.lengths.item() == len(row) - 1 + 2 * (VISION_TOKENS - 1)
    assert batch.token_count == len(row) - 1 + 2 * (VISION_TOKENS - 1)
    assert batch.padded_token_count == batch.token_count
    assert batch.supervised_target_count == 3
    projected = projector(features.reshape(-1, features.size(-1))).view(2, VISION_TOKENS, -1)
    torch.testing.assert_close(batch.input_embeds[0, 2:2 + VISION_TOKENS], projected[0])
    torch.testing.assert_close(batch.input_embeds[0, 70:70 + VISION_TOKENS], projected[1])
    assert torch.isfinite(model(batch.value_token_ids, batch.targets, input_embeds=batch.input_embeds))
    with pytest.raises(AssertionError, match="expected 1"):
        build_multimodal_batch(model, projector, [row], features[:1], loss_mask_rows=[mask], image_counts_per_row=[1], value_fallback_token_id=1)


def test_multimodal_batch_backward_reaches_projector_after_image_insert():
    model, projector = tiny_model()
    projector.train()
    row = [1, 10, IMAGE_TOKEN_ID, 11, 12]
    mask = [1] * len(row)
    features = torch.randn(1, VISION_TOKENS, 8)
    batch = build_multimodal_batch(model, projector, [row], features, loss_mask_rows=[mask], value_fallback_token_id=1)
    loss = batch.input_embeds[0, 2:2 + VISION_TOKENS].square().mean()
    loss.backward()
    assert projector.proj.weight.grad is not None
    assert torch.isfinite(projector.proj.weight.grad).all()
    assert projector.proj.weight.grad.abs().sum() > 0
    if model.transformer.wte.weight.grad is not None:
        assert model.transformer.wte.weight.grad.abs().sum() == 0


def test_text_only_gpt_path_and_embed_hook_match():
    model, _ = tiny_model()
    row = [1, 10, 11, 12, 13]
    ids = torch.tensor([row[:-1]], dtype=torch.long)
    targets = torch.tensor([row[1:]], dtype=torch.long)
    text_loss = model(ids, targets)
    embeds = model.transformer.wte(ids)
    embed_loss = model(ids, targets, input_embeds=embeds)
    torch.testing.assert_close(embed_loss, text_loss, rtol=0, atol=0)
    torch.testing.assert_close(model(ids, input_embeds=embeds), model(ids), rtol=0, atol=0)
    with pytest.raises(AssertionError, match="input_embeds last dim"):
        model(ids, input_embeds=embeds[..., :-1])


def test_gpt_selective_loss_matches_ignore_index_path():
    model, _ = tiny_model()
    ids = torch.tensor([[1, 10, 11, 12], [1, 20, 21, 22]], dtype=torch.long)
    targets = torch.tensor([[10, -1, 12, -1], [20, 21, -1, 23]], dtype=torch.long)
    embeds = model.transformer.wte(ids)

    for reduction in ("none", "sum", "mean"):
        full_loss = model(ids, targets, loss_reduction=reduction)
        selective_loss = model(ids, targets, input_embeds=embeds, loss_reduction=reduction, selective_loss=True)
        target_indices = (targets.view(-1) != -1).nonzero(as_tuple=False).view(-1)
        indexed_loss = model(
            ids,
            targets,
            input_embeds=embeds,
            loss_reduction=reduction,
            selective_loss=True,
            loss_indices=target_indices,
            loss_targets=targets.view(-1).index_select(0, target_indices),
        )
        targetless_indexed_loss = model(
            ids,
            None,
            input_embeds=embeds,
            loss_reduction=reduction,
            selective_loss=True,
            loss_indices=target_indices,
            loss_targets=targets.view(-1).index_select(0, target_indices),
        )
        chunked_selective_loss = model(
            ids,
            targets,
            input_embeds=embeds,
            loss_reduction=reduction,
            selective_loss=True,
            loss_chunk_size=2,
        )
        chunked_indexed_loss = model(
            ids,
            None,
            input_embeds=embeds,
            loss_reduction=reduction,
            selective_loss=True,
            loss_indices=target_indices,
            loss_targets=targets.view(-1).index_select(0, target_indices),
            loss_chunk_size=2,
        )
        chunked_loss = model(ids, targets, loss_reduction=reduction, loss_chunk_size=3)
        torch.testing.assert_close(selective_loss, full_loss, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(indexed_loss, full_loss, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(targetless_indexed_loss, full_loss, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(chunked_selective_loss, full_loss, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(chunked_indexed_loss, full_loss, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(chunked_loss, full_loss, rtol=1e-6, atol=1e-6)


def test_gpt_flop_estimate_can_use_actual_sequence_len():
    model, _ = tiny_model()
    assert model.estimate_flops(sequence_len=64) < model.estimate_flops(sequence_len=128)
    base, attn = model.estimate_flops_components(sequence_len=64)
    assert base > attn > 0
    assert base + attn == model.estimate_flops(sequence_len=64)


def test_model_flop_estimate_cache_reuses_sequence_lengths():
    class FakeModel:
        def __init__(self):
            self.calls = []

        def estimate_flops(self, sequence_len=None):
            self.calls.append(sequence_len)
            return 10 if sequence_len is None else sequence_len

    fake = FakeModel()
    cache = {}
    assert vlm_train.estimate_model_flops(fake, cache=cache) == 10
    assert vlm_train.estimate_model_flops(fake, cache=cache) == 10
    assert vlm_train.estimate_model_flops(fake, sequence_len=128, cache=cache) == 128
    assert vlm_train.estimate_model_flops(fake, sequence_len=128, cache=cache) == 128
    assert fake.calls == [None, 128]


def test_varlen_step_flop_estimate_counts_segment_attention_and_padded_matmuls():
    class FakeModel:
        def estimate_flops_components(self, sequence_len=None):
            return 100, int(sequence_len)

    useful, padded = vlm_train.estimate_varlen_step_flops(FakeModel(), [2, 3], padded_tokens=10, cache={})
    assert useful == (100 + 2) * 2 + (100 + 3) * 3
    assert padded == 100 * 10 + 2 * 2 + 3 * 3


def test_step_flop_estimate_charges_selective_lm_head_only_on_loss_tokens():
    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = torch.nn.Linear(2, 3, bias=False)

        def estimate_flops_components(self, sequence_len=None):
            return 100 + 6 * self.lm_head.weight.numel(), int(sequence_len)

    fake = FakeModel()
    lm_head_flops = 6 * fake.lm_head.weight.numel()
    useful, padded = vlm_train.estimate_varlen_step_flops(
        fake,
        [2, 3],
        padded_tokens=10,
        useful_lm_head_tokens=2,
        padded_lm_head_tokens=2,
        cache={},
    )
    assert useful == (100 + 2) * 2 + (100 + 3) * 3 + lm_head_flops * 2
    assert padded == 100 * 10 + 2 * 2 + 3 * 3 + lm_head_flops * 2

    dense_useful, dense_padded = vlm_train.estimate_dense_step_flops(
        fake,
        sequence_len=5,
        useful_tokens=5,
        padded_tokens=8,
        useful_lm_head_tokens=2,
        padded_lm_head_tokens=2,
        cache={},
    )
    assert dense_useful == (100 + 5) * 5 + lm_head_flops * 2
    assert dense_padded == (100 + 5) * 8 + lm_head_flops * 2


def test_vlm_checkpoint_save_load(tmp_path):
    model, projector = tiny_model()
    save_vlm_checkpoint(tmp_path, 7, model, projector, {"ok": True}, {"mode": "vlm"}, rank=0)
    model_state, loaded_projector, optimizer_data, meta = load_vlm_checkpoint(tmp_path, 7, torch.device("cpu"), load_optimizer=True, rank=0)
    assert meta["mode"] == "vlm"
    assert optimizer_data["ok"] is True
    assert set(model_state) == set(model.state_dict())
    for key, value in projector.state_dict().items():
        torch.testing.assert_close(value, loaded_projector.state_dict()[key])


def test_hf_nanochat_d32_links_to_sft_layout(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    for name in ["tokenizer.pkl", "token_bytes.pt", "model_000650.pt", "meta_000650.json"]:
        (snapshot / name).write_text("x", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=lambda repo_id: str(snapshot)))

    checkpoint_dir = ensure_hf_nanochat_checkpoint("karpathy/nanochat-d32", str(tmp_path / "base"), model_tag="d32", source="sft")
    assert checkpoint_dir.endswith("chatsft_checkpoints/d32")
    assert (tmp_path / "base" / "tokenizer" / "tokenizer.pkl").exists()
    assert (tmp_path / "base" / "chatsft_checkpoints" / "d32" / "model_000650.pt").exists()


def test_hf_tokenizer_only_link_skips_model_layout(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    for name in ["tokenizer.pkl", "token_bytes.pt", "model_000650.pt", "meta_000650.json"]:
        (snapshot / name).write_text("x", encoding="utf-8")
    calls = []

    def fake_snapshot_download(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return str(snapshot)

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=fake_snapshot_download))
    tokenizer_dir = ensure_hf_nanochat_tokenizer("karpathy/nanochat-d32", str(tmp_path / "base"))

    assert tokenizer_dir.endswith("tokenizer")
    assert calls == [("karpathy/nanochat-d32", {"allow_patterns": ["tokenizer.pkl", "token_bytes.pt"]})]
    assert (tmp_path / "base" / "tokenizer" / "tokenizer.pkl").exists()
    assert not (tmp_path / "base" / "chatsft_checkpoints").exists()


def test_training_checkpoint_metadata_records_vision_config(tmp_path):
    model, projector = tiny_model()
    args = SimpleNamespace(
        data_json=None,
        hf_repo="repo",
        hf_config="cfg",
        max_examples=10,
        image_root="/images",
        skip_bad_images=True,
        siglip_model_id="google/siglip-base-patch16-512",
        siglip_use_fast_processor=True,
        init_vlm_checkpoint_dir="/tmp/checkpoint",
        init_vlm_checkpoint_step=1,
    )
    model_meta = {"model_config": {"n_embd": 32}}
    save_training_checkpoint(tmp_path, 3, model, projector, args, model_meta, "stream:repo/file.json", rank=0)
    _, _, _, meta = load_vlm_checkpoint(tmp_path, 3, torch.device("cpu"))
    assert meta["vision_config"]["vision_tokens"] == VISION_TOKENS
    assert meta["vision_config"]["projector_vision_dim"] == projector.vision_dim


def test_training_rendering_filters_bad_rows_and_counts_targets():
    tokenizer = TinyTokenizer()
    records = [{
        "image": "tiny.jpg",
        "texts": [{"user": "What is shown?", "assistant": "A small image."}],
    }]
    rendered = render_records(records, tokenizer, max_seq_len=256)
    assert len(rendered) == 1
    assert supervised_target_count(rendered[0]["tokens"], rendered[0]["mask"]) > 0

    direct_image_answer_tokens = [1, IMAGE_TOKEN_ID, 65]
    assert supervised_target_count(direct_image_answer_tokens, [0, 0, 1]) == 0
    with pytest.raises(AssertionError, match="no usable"):
        render_records([{"image": "tiny.jpg", "texts": [{"user": "Q?", "assistant": "x" * 300}]}], tokenizer, max_seq_len=128)


def test_next_batch_respects_padded_token_budget():
    examples = [
        {"expanded_len": 100, "tokens": [1, IMAGE_TOKEN_ID, 2], "mask": [0, 0, 1], "record": {"image": "a"}},
        {"expanded_len": 80, "tokens": [1, IMAGE_TOKEN_ID, 3], "mask": [0, 0, 1], "record": {"image": "b"}},
        {"expanded_len": 20, "tokens": [1, IMAGE_TOKEN_ID, 4], "mask": [0, 0, 1], "record": {"image": "c"}},
    ]
    batch, cursor = next_batch(examples, batch_size=3, cursor=0, rng=__import__("random").Random(0), max_batch_tokens=180)
    assert len(batch) <= 2
    assert cursor > 0


def test_bucketed_padding_respects_token_budget():
    buckets = vlm_train.parse_bucket_lens("96,128", max_seq_len=256)
    assert buckets == [96, 128, 256]
    assert vlm_train.bucketed_len(97, buckets) == 128
    examples = [
        {"expanded_len": 90, "tokens": [1, IMAGE_TOKEN_ID, 2], "mask": [0, 0, 1], "record": {"image": "a"}},
        {"expanded_len": 97, "tokens": [1, IMAGE_TOKEN_ID, 3], "mask": [0, 0, 1], "record": {"image": "b"}},
        {"expanded_len": 120, "tokens": [1, IMAGE_TOKEN_ID, 4], "mask": [0, 0, 1], "record": {"image": "c"}},
    ]
    batch, cursor = next_batch(
        examples,
        batch_size=3,
        cursor=0,
        rng=__import__("random").Random(0),
        max_batch_tokens=256,
        bucket_lens=buckets,
    )
    assert {vlm_train.bucketed_len(ex["expanded_len"], buckets) for ex in batch} == {96}
    assert vlm_train.bucketed_len(max(ex["expanded_len"] for ex in batch), buckets) * len(batch) <= 256
    assert cursor > 0


def test_length_stats_summary_reports_realistic_bucket_budget():
    buckets = vlm_train.parse_bucket_lens("96,128", max_seq_len=256)
    summary = vlm_train.summarize_length_stats(
        [90, 97, 120, 200, 300],
        scanned=7,
        max_seq_len=256,
        bucket_lens=buckets,
        max_batch_tokens=256,
    )
    assert summary["scanned"] == 7
    assert summary["usable"] == 5
    assert summary["fit_count"] == 4
    assert summary["buckets"][0]["bucket"] == 96
    assert summary["buckets"][0]["text_cap_1img"] == 32
    assert summary["buckets"][0]["rows_at_token_cap"] == 2
    assert summary["buckets"][1]["count"] == 2
    assert summary["buckets"][2]["count"] == 1
    assert summary["overflow"] == 1
    summary["elapsed"] = 2.0
    text = vlm_train.format_length_stats(summary, "test-source", 256, buckets, max_batch_tokens=256)
    assert "expanded_len min/p50/p80/p90/p95/p99/max/mean" in text
    assert "fit_at_max_seq_len_256=4/5" in text
    assert "usable_examples/sec=2.5" in text
    assert "text_cap_1img" in text
    assert "rows@cap" in text


def test_batch_plan_summary_reports_bucket_fill_and_padding():
    buckets = vlm_train.parse_bucket_lens("96,128", max_seq_len=256)
    rows = [
        vlm_train.batch_plan_row([{"expanded_len": 90}, {"expanded_len": 80}], buckets, batch_size=4, max_batch_tokens=256),
        vlm_train.batch_plan_row([{"expanded_len": 127}, {"expanded_len": 126}], buckets, batch_size=4, max_batch_tokens=256),
    ]
    summary = vlm_train.summarize_batch_plan(rows)
    assert summary["total"]["steps"] == 2
    assert summary["total"]["rows"] == 4
    assert summary["total"]["examples"] == 4
    assert summary["total"]["dropped_examples"] == 0
    assert summary["total"]["attention_pairs"] == vlm_train.causal_attention_pairs(2, 96) + vlm_train.causal_attention_pairs(2, 128)
    assert summary["total"]["segments"] == 4
    assert vlm_train.segment_length_percentile(summary["total"]["segment_lengths"], 0.50) == 126
    assert vlm_train.segment_length_percentile(summary["total"]["segment_lengths"], 0.90) == 127
    assert summary["total"]["max_segment_len"] == 127
    assert summary["buckets"][96]["target_rows"] == 2
    assert summary["buckets"][128]["target_rows"] == 2
    assert summary["buckets"][128]["examples"] == 2
    assert summary["buckets"][128]["useful_tokens"] == 253
    assert vlm_train.format_count(summary["buckets"][128]["attention_pairs"]) == "16.5K"
    text = vlm_train.format_batch_plan(
        summary,
        "test-source",
        SimpleNamespace(
            device_batch_size=4,
            max_batch_tokens=256,
            grad_accum_steps=1,
            bucket_selection="cycle",
            bucket_min_fill_frac=0.75,
            bucket_cycle_repeat=1,
            pack_examples=1,
        ),
        buckets,
        elapsed=3.0,
        records_scanned=10,
        rendered_examples=6,
    )
    assert "Batch plan source=test-source" in text
    assert "bucket_selection=cycle" in text
    assert "planning_elapsed=3.0s records_scanned=10 rendered_examples=6 rendered_examples/sec=2.0" in text
    assert "bucket | steps | rows avg/min/max" in text
    assert "examples/step=2.0 dropped/step=0.0" in text
    assert "attn_pairs/step" in text
    assert "segments/step=2.0 avg_segment=105.8 p50_segment=126 p90_segment=127 max_segment=127 near_cap/step=0.0 cap_hits/step=0.0" in text
    assert "segment_len avg/p50/p90/max" in text
    assert "near_cap/cap avg" in text


def test_batch_plan_summary_reports_optimizer_step_groups():
    buckets = vlm_train.parse_bucket_lens("96,128", max_seq_len=256)
    rows = [
        vlm_train.batch_plan_row([{"expanded_len": 90}, {"expanded_len": 80}], buckets, batch_size=4, max_batch_tokens=256),
        vlm_train.batch_plan_row([{"expanded_len": 91}, {"expanded_len": 89}], buckets, batch_size=4, max_batch_tokens=256),
        vlm_train.batch_plan_row([{"expanded_len": 127}, {"expanded_len": 126}], buckets, batch_size=4, max_batch_tokens=256),
        vlm_train.batch_plan_row([{"expanded_len": 120}, {"expanded_len": 121}], buckets, batch_size=4, max_batch_tokens=256),
        vlm_train.batch_plan_row([{"expanded_len": 200}], buckets, batch_size=4, max_batch_tokens=256),
    ]
    summary = vlm_train.summarize_batch_plan(rows, grad_accum_steps=2)
    optimizer = summary["optimizer_steps"]["total"]
    assert optimizer["groups"] == 3
    assert optimizer["complete_steps"] == 2
    assert optimizer["incomplete_steps"] == 1
    assert optimizer["same_bucket_steps"] == 2
    assert optimizer["mixed_bucket_steps"] == 0
    assert summary["optimizer_steps"]["buckets"][96]["steps"] == 1
    assert summary["optimizer_steps"]["buckets"][128]["steps"] == 1
    text = vlm_train.format_batch_plan(
        summary,
        "test-source",
        SimpleNamespace(
            device_batch_size=4,
            max_batch_tokens=256,
            grad_accum_steps=2,
            bucket_selection="cycle",
            bucket_min_fill_frac=0.75,
            bucket_cycle_repeat=2,
            pack_examples=1,
        ),
        buckets,
    )
    assert "optimizer_steps complete=2/3 incomplete=1 same_bucket=2 mixed_bucket=0" in text
    assert "optimizer bucket | steps | rows/step" in text


def test_batch_plan_row_models_packed_examples():
    buckets = vlm_train.parse_bucket_lens("128", max_seq_len=256)
    examples = [
        {"expanded_len": 70, "tokens": [1, 10, IMAGE_TOKEN_ID, 11, 12], "mask": [0, 0, 0, 1, 1]},
        {"expanded_len": 72, "tokens": [1, 20, IMAGE_TOKEN_ID, 21, 22], "mask": [0, 0, 0, 1, 1]},
        {"expanded_len": 180, "tokens": [1, 30, IMAGE_TOKEN_ID, 31, 32], "mask": [0, 0, 0, 1, 1]},
    ]
    row = vlm_train.batch_plan_row(
        examples,
        buckets,
        batch_size=4,
        max_batch_tokens=512,
        max_seq_len=256,
        max_images_per_row=2,
        boundary_aware_pack=True,
    )
    assert row["bucket"] == 256
    assert row["rows"] == 2
    assert row["examples"] == 3
    assert row["dropped_examples"] == 0
    assert row["target_rows"] == 2
    assert row["useful_tokens"] == vlm_train.packed_expanded_len(examples[1:], boundary_aware=True) + examples[0]["expanded_len"]
    assert row["padded_tokens"] == 512
    expected_segment_pairs = vlm_train.causal_attention_pairs_for_lengths([180, 72, 70])
    assert row["attention_pairs"] == expected_segment_pairs
    assert row["attention_pairs"] < vlm_train.causal_attention_pairs(row["rows"], row["bucket"])
    assert row["segments"] == 3
    assert row["max_segment_len"] == 180


def test_batch_plan_row_reports_near_cap_segments():
    buckets = vlm_train.parse_bucket_lens("256", max_seq_len=256)
    examples = [
        {"expanded_len": 256, "tokens": [1, IMAGE_TOKEN_ID, 10], "mask": [0, 0, 1]},
        {"expanded_len": 244, "tokens": [1, IMAGE_TOKEN_ID, 11], "mask": [0, 0, 1]},
        {"expanded_len": 200, "tokens": [1, IMAGE_TOKEN_ID, 12], "mask": [0, 0, 1]},
    ]
    row = vlm_train.batch_plan_row(
        examples,
        buckets,
        batch_size=4,
        max_batch_tokens=1024,
        max_seq_len=256,
        max_images_per_row=2,
        boundary_aware_pack=True,
    )
    assert row["segment_cap_len"] == 256
    assert row["near_cap_segments"] == 2
    assert row["cap_segments"] == 1
    summary = vlm_train.summarize_batch_plan([row])
    assert summary["total"]["near_cap_segments"] == 2
    assert summary["total"]["cap_segments"] == 1
    text = vlm_train.format_batch_plan(
        summary,
        "test-source",
        SimpleNamespace(
            device_batch_size=4,
            max_batch_tokens=1024,
            grad_accum_steps=1,
            bucket_selection="max-tokens",
            bucket_min_fill_frac=0.0,
            bucket_cycle_repeat=1,
            pack_examples=2,
        ),
        buckets,
    )
    assert "near_cap/step=2.0 cap_hits/step=1.0" in text
    assert "near_cap/cap avg" in text


def test_count_near_cap_segments_matches_batch_plan_threshold():
    near_cap, cap_hits = vlm_train.count_near_cap_segments([972, 973, 1024, 120], 1024)
    assert near_cap == 2
    assert cap_hits == 1


def test_profile_summary_reports_pack_timing():
    profile = vlm_train.new_profile()
    profile["pack"] = 0.25
    profile["batch_projector"] = 0.125
    text = vlm_train.format_profile_summary("Steady timing totals", profile, total_seconds=1.0)
    assert "pack=0.250s/25.0%" in text
    assert "batch_projector=0.125s/12.5%" in text
    assert "other=0.750s/75.0%" in text
    assert vlm_train.profile_other_seconds(profile, total_seconds=1.0) == pytest.approx(0.75)


def test_batch_plan_row_loose_packing_uses_dense_attention():
    buckets = vlm_train.parse_bucket_lens("128", max_seq_len=256)
    examples = [
        {"expanded_len": 70, "tokens": [1, 10, IMAGE_TOKEN_ID, 11, 12], "mask": [0, 0, 0, 1, 1]},
        {"expanded_len": 72, "tokens": [1, 20, IMAGE_TOKEN_ID, 21, 22], "mask": [0, 0, 0, 1, 1]},
        {"expanded_len": 180, "tokens": [1, 30, IMAGE_TOKEN_ID, 31, 32], "mask": [0, 0, 0, 1, 1]},
    ]
    row = vlm_train.batch_plan_row(
        examples,
        buckets,
        batch_size=4,
        max_batch_tokens=512,
        max_seq_len=256,
        max_images_per_row=2,
        boundary_aware_pack=False,
    )
    assert row["attention_pairs"] == vlm_train.causal_attention_pairs(row["rows"], row["bucket"])


def test_pack_trim_filters_examples_before_vision_work():
    buckets = vlm_train.parse_bucket_lens("256", max_seq_len=512)
    examples = [
        {"expanded_len": 200, "tokens": [1, IMAGE_TOKEN_ID, 10], "mask": [0, 0, 1]},
        {"expanded_len": 200, "tokens": [1, IMAGE_TOKEN_ID, 11], "mask": [0, 0, 1]},
        {"expanded_len": 200, "tokens": [1, IMAGE_TOKEN_ID, 12], "mask": [0, 0, 1]},
        {"expanded_len": 200, "tokens": [1, IMAGE_TOKEN_ID, 13], "mask": [0, 0, 1]},
    ]
    kept = vlm_train.trim_examples_to_packable(
        examples,
        max_seq_len=512,
        max_batch_tokens=512,
        max_images_per_row=2,
        bucket_lens=buckets,
    )
    assert len(kept) == 2

    kept = vlm_train.trim_examples_to_packable(
        examples[:1],
        max_seq_len=128,
        max_batch_tokens=512,
        max_images_per_row=2,
        bucket_lens=buckets,
    )
    assert kept == []


def test_bucket_steady_metrics_accumulate_by_static_shape():
    stats = {}
    vlm_train.add_bucket_steady_step(
        stats,
        bucket=96,
        elapsed=2.0,
        tokens=180,
        padded_tokens=192,
        samples=2,
        seq_flops=1800,
        seq_padded_flops=1920,
        attention_pairs=1000,
        segments=2,
        segment_lengths=[100, 80],
        max_segment_len=100,
        near_cap_segments=1,
    )
    vlm_train.add_bucket_steady_step(
        stats,
        bucket=96,
        elapsed=1.0,
        tokens=90,
        padded_tokens=96,
        samples=1,
        seq_flops=900,
        seq_padded_flops=960,
        attention_pairs=500,
        segments=1,
        segment_lengths=[90],
        max_segment_len=90,
    )
    vlm_train.add_bucket_steady_step(stats, bucket=128, elapsed=1.0, tokens=100, padded_tokens=128, samples=1)

    metrics = vlm_train.bucket_steady_metrics(stats[96], num_flops_per_token=10.0, gpu_peak_flops=1000.0)
    assert metrics["steps"] == 2
    assert metrics["tokens_per_sec"] == 90
    assert metrics["padded_tokens_per_sec"] == 96
    assert metrics["samples_per_sec"] == 1
    assert metrics["mfu"] == pytest.approx(90.0)
    assert metrics["padded_mfu"] == pytest.approx(96.0)
    assert metrics["token_estimate_mfu"] == pytest.approx(90.0)
    assert metrics["token_estimate_padded_mfu"] == pytest.approx(96.0)
    assert metrics["seq_mfu"] == pytest.approx(90.0)
    assert metrics["seq_padded_mfu"] == pytest.approx(96.0)
    assert metrics["padding_frac"] == pytest.approx(1 - 270 / 288)
    assert metrics["attention_pairs_per_step"] == pytest.approx(750.0)
    assert metrics["attention_pairs_per_token"] == pytest.approx(1500 / 270)
    assert metrics["segments_per_step"] == pytest.approx(1.5)
    assert metrics["avg_segment_len"] == pytest.approx(90.0)
    assert metrics["p50_segment_len"] == 90
    assert metrics["p90_segment_len"] == 100
    assert metrics["max_segment_len"] == 100
    assert metrics["near_cap_segments_per_step"] == pytest.approx(0.5)
    assert metrics["cap_segments_per_step"] == 0
    line = vlm_train.format_bucket_steady_line(96, metrics)
    assert "bucket 96 | steps 2" in line
    assert "attn_pairs/token 5.56" in line
    assert "avg_segment 90.0 | p50_segment 90 | p90_segment 100 | max_segment 100" in line


def test_mfu_step_counting_respects_global_and_bucket_warmups():
    assert not vlm_train.should_count_mfu_step(
        step=2,
        global_warmup_steps=2,
        step_bucket=96,
        bucket_seen_before=10,
        bucket_warmup_steps=1,
    )
    assert not vlm_train.should_count_mfu_step(
        step=3,
        global_warmup_steps=2,
        step_bucket=128,
        bucket_seen_before=0,
        bucket_warmup_steps=1,
    )
    assert vlm_train.should_count_mfu_step(
        step=3,
        global_warmup_steps=2,
        step_bucket=96,
        bucket_seen_before=1,
        bucket_warmup_steps=1,
    )


def test_compact_varlen_steps_do_not_create_static_mfu_buckets():
    assert vlm_train.static_mfu_step_bucket([128, 128], compact_varlen=False) == 128
    assert vlm_train.static_mfu_step_bucket([128, 192], compact_varlen=False) == 0
    assert vlm_train.static_mfu_step_bucket([65519], compact_varlen=True) == 0
    assert vlm_train.should_count_mfu_step(
        step=3,
        global_warmup_steps=2,
        step_bucket=0,
        bucket_seen_before=0,
        bucket_warmup_steps=1,
    )


def test_materialized_bucket_batches_stop_at_bucket_boundaries():
    buckets = vlm_train.parse_bucket_lens("96,128", max_seq_len=256)
    examples = [
        {"expanded_len": 120, "tokens": [1, IMAGE_TOKEN_ID, 2], "mask": [0, 0, 1], "record": {"image": "a"}},
        {"expanded_len": 80, "tokens": [1, IMAGE_TOKEN_ID, 3], "mask": [0, 0, 1], "record": {"image": "b"}},
        {"expanded_len": 127, "tokens": [1, IMAGE_TOKEN_ID, 4], "mask": [0, 0, 1], "record": {"image": "c"}},
        {"expanded_len": 90, "tokens": [1, IMAGE_TOKEN_ID, 5], "mask": [0, 0, 1], "record": {"image": "d"}},
        {"expanded_len": 200, "tokens": [1, IMAGE_TOKEN_ID, 6], "mask": [0, 0, 1], "record": {"image": "e"}},
    ]
    rng = __import__("random").Random(0)
    first, cursor = next_batch(examples, batch_size=4, cursor=0, rng=rng, max_batch_tokens=512, bucket_lens=buckets)
    second, cursor = next_batch(examples, batch_size=4, cursor=cursor, rng=rng, max_batch_tokens=512, bucket_lens=buckets)
    assert {vlm_train.bucketed_len(ex["expanded_len"], buckets) for ex in first} == {96}
    assert {vlm_train.bucketed_len(ex["expanded_len"], buckets) for ex in second} == {128}
    assert [ex["expanded_len"] for ex in first] == [90, 80]
    assert [ex["expanded_len"] for ex in second] == [127, 120]


def test_materialized_packed_batch_uses_packable_buffer_before_vision_work():
    class LastWindowRng:
        def shuffle(self, values):
            pass

        def randrange(self, upper):
            return upper - 1

    examples = [
        {"expanded_len": 500, "tokens": [1, IMAGE_TOKEN_ID, 10], "mask": [0, 0, 1], "record": {"image": "a"}},
        {"expanded_len": 80, "tokens": [1, IMAGE_TOKEN_ID, 11], "mask": [0, 0, 1], "record": {"image": "b"}},
        {"expanded_len": 90, "tokens": [1, IMAGE_TOKEN_ID, 12], "mask": [0, 0, 1], "record": {"image": "c"}},
        {"expanded_len": 501, "tokens": [1, IMAGE_TOKEN_ID, 13], "mask": [0, 0, 1], "record": {"image": "d"}},
    ]
    buffer = []
    batch, cursor = vlm_train.next_materialized_packed_batch(
        examples,
        batch_size=2,
        cursor=0,
        rng=LastWindowRng(),
        max_batch_tokens=160,
        batch_buffer=buffer,
        batch_buffer_size=4,
        pack_max_seq_len=100,
        max_images_per_row=2,
        boundary_aware_pack=True,
    )
    assert [example["expanded_len"] for example in batch] == [80]
    assert cursor == 0
    assert [example["expanded_len"] for example in buffer] == [500, 90, 501]

    batch, cursor = vlm_train.next_materialized_packed_batch(
        examples,
        batch_size=2,
        cursor=cursor,
        rng=LastWindowRng(),
        max_batch_tokens=160,
        batch_buffer=buffer,
        batch_buffer_size=4,
        pack_max_seq_len=100,
        max_images_per_row=2,
        boundary_aware_pack=True,
    )
    assert [example["expanded_len"] for example in batch] == [90]
    assert [example["expanded_len"] for example in buffer] == [500, 501]


def test_load_records_and_render_finevision_schema(monkeypatch):
    streamed = [
        {
            "images": [Image.new("RGB", (4, 4), color=(1, 2, 3))],
            "texts": [{"user": "What color?", "assistant": "Red."}],
        }
        for _ in range(2)
    ]

    def fake_load_dataset(repo, config, **kwargs):
        assert repo == "repo"
        assert config == "cfg"
        assert kwargs["streaming"] is True
        return iter(streamed)

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=fake_load_dataset))
    args = SimpleNamespace(data_json=None, hf_repo="repo", hf_config="cfg", max_examples=1, device_batch_size=2, grad_accum_steps=1, num_iterations=1)
    records, val_records, source = load_records(args, val_count=1)
    rendered = render_records(records, TinyTokenizer(), max_seq_len=256)
    assert source == "stream:repo/cfg first 1 train rows"
    assert len(rendered) == 1
    assert len(val_records) == 1
    assert count_image_tokens(rendered[0]["tokens"]) == 1


def test_hf_stream_shuffle_and_lazy_stream_batch(monkeypatch):
    streamed = [
        {
            "images": [Image.new("RGB", (4, 4), color=(i, 2, 3))],
            "texts": [{"user": "What color?", "assistant": f"Color {i}."}],
        }
        for i in range(6)
    ]
    shuffle_calls = []

    class FakeStream:
        def __iter__(self):
            return iter(streamed)

        def shuffle(self, **kwargs):
            shuffle_calls.append(kwargs)
            return self

    def fake_load_dataset(repo, config, **kwargs):
        assert kwargs["streaming"] is True
        return FakeStream()

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=fake_load_dataset))
    args = SimpleNamespace(hf_repo="repo", hf_config="cfg")
    source = iter_rendered_examples(iter_hf_records(args, seed=123, buffer_size=2), TinyTokenizer(), max_seq_len=256)
    batch, pending = next_stream_batch(source, batch_size=2)
    assert len(batch) == 2
    assert pending is None
    assert all(example["expanded_len"] <= 256 for example in batch)
    assert shuffle_calls == [{"seed": 123, "buffer_size": 2}]


def test_stream_batch_buffer_groups_similar_lengths():
    examples = iter([
        {"expanded_len": 20, "tokens": [1, IMAGE_TOKEN_ID, 2], "mask": [0, 0, 1], "record": {"image": "a"}},
        {"expanded_len": 200, "tokens": [1, IMAGE_TOKEN_ID, 3], "mask": [0, 0, 1], "record": {"image": "b"}},
        {"expanded_len": 22, "tokens": [1, IMAGE_TOKEN_ID, 4], "mask": [0, 0, 1], "record": {"image": "c"}},
        {"expanded_len": 210, "tokens": [1, IMAGE_TOKEN_ID, 5], "mask": [0, 0, 1], "record": {"image": "d"}},
    ])
    buffer = []
    batch, pending = next_stream_batch(examples, batch_size=2, buffer=buffer, batch_buffer_size=4, rng=__import__("random").Random(0))
    lengths = sorted(ex["expanded_len"] for ex in batch)
    assert pending is None
    assert lengths in ([20, 22], [200, 210])
    assert len(buffer) == 2


def test_stream_batch_buffer_selects_one_static_bucket():
    buckets = vlm_train.parse_bucket_lens("96,128", max_seq_len=256)
    examples = iter([
        {"expanded_len": 80, "tokens": [1, IMAGE_TOKEN_ID, 2], "mask": [0, 0, 1], "record": {"image": "a"}},
        {"expanded_len": 90, "tokens": [1, IMAGE_TOKEN_ID, 3], "mask": [0, 0, 1], "record": {"image": "b"}},
        {"expanded_len": 120, "tokens": [1, IMAGE_TOKEN_ID, 4], "mask": [0, 0, 1], "record": {"image": "c"}},
        {"expanded_len": 127, "tokens": [1, IMAGE_TOKEN_ID, 5], "mask": [0, 0, 1], "record": {"image": "d"}},
        {"expanded_len": 200, "tokens": [1, IMAGE_TOKEN_ID, 6], "mask": [0, 0, 1], "record": {"image": "e"}},
        {"expanded_len": 220, "tokens": [1, IMAGE_TOKEN_ID, 7], "mask": [0, 0, 1], "record": {"image": "f"}},
    ])
    buffer = []
    batch, pending = next_stream_batch(
        examples,
        batch_size=4,
        max_batch_tokens=512,
        buffer=buffer,
        batch_buffer_size=6,
        rng=__import__("random").Random(0),
        bucket_lens=buckets,
    )
    batch_buckets = {vlm_train.bucketed_len(ex["expanded_len"], buckets) for ex in batch}
    assert pending is None
    assert len(batch_buckets) == 1
    assert next(iter(batch_buckets)) * len(batch) <= 512


def test_stream_bucket_selection_prefers_dense_rows_within_bucket():
    buckets = vlm_train.parse_bucket_lens("128", max_seq_len=256)
    buffer = [
        {"expanded_len": 98},
        {"expanded_len": 101},
        {"expanded_len": 126},
        {"expanded_len": 127},
    ]
    ordered = sorted(range(len(buffer)), key=lambda i: buffer[i]["expanded_len"])
    indices = vlm_train._choose_stream_buffer_indices(
        buffer,
        ordered,
        batch_size=4,
        max_batch_tokens=256,
        rng=__import__("random").Random(0),
        bucket_lens=buckets,
    )
    assert [buffer[i]["expanded_len"] for i in indices] == [126, 127]


def test_stream_bucket_selection_can_cycle_static_buckets():
    buckets = vlm_train.parse_bucket_lens("96,128", max_seq_len=256)
    buffer = [
        {"expanded_len": 90},
        {"expanded_len": 120},
        {"expanded_len": 200},
    ]
    ordered = sorted(range(len(buffer)), key=lambda i: buffer[i]["expanded_len"])
    state = {}
    selected_buckets = []
    for _ in range(4):
        indices = vlm_train._choose_stream_buffer_indices(
            buffer,
            ordered,
            batch_size=1,
            max_batch_tokens=256,
            bucket_lens=buckets,
            bucket_selection="cycle",
            bucket_state=state,
        )
        selected_buckets.append(vlm_train.bucketed_len(buffer[indices[0]]["expanded_len"], buckets))
    assert selected_buckets == [96, 128, 256, 96]


def test_stream_bucket_cycle_can_repeat_for_grad_accum_microbatches():
    buckets = vlm_train.parse_bucket_lens("96,128", max_seq_len=256)
    buffer = [
        {"expanded_len": 88},
        {"expanded_len": 90},
        {"expanded_len": 118},
        {"expanded_len": 120},
        {"expanded_len": 200},
        {"expanded_len": 210},
    ]
    ordered = sorted(range(len(buffer)), key=lambda i: buffer[i]["expanded_len"])
    state = {}
    selected_buckets = []
    for _ in range(6):
        indices = vlm_train._choose_stream_buffer_indices(
            buffer,
            ordered,
            batch_size=1,
            max_batch_tokens=256,
            bucket_lens=buckets,
            bucket_selection="cycle",
            bucket_state=state,
            bucket_cycle_repeat=2,
        )
        selected_buckets.append(vlm_train.bucketed_len(buffer[indices[0]]["expanded_len"], buckets))
    assert selected_buckets == [96, 96, 128, 128, 256, 256]


def test_stream_bucket_cycle_skips_underfilled_buckets_when_requested():
    buckets = vlm_train.parse_bucket_lens("96,128", max_seq_len=256)
    buffer = [
        {"expanded_len": 90},
        {"expanded_len": 120},
        {"expanded_len": 121},
        {"expanded_len": 127},
        {"expanded_len": 200},
    ]
    ordered = sorted(range(len(buffer)), key=lambda i: buffer[i]["expanded_len"])
    state = {}
    indices = vlm_train._choose_stream_buffer_indices(
        buffer,
        ordered,
        batch_size=4,
        max_batch_tokens=512,
        bucket_lens=buckets,
        bucket_selection="cycle",
        bucket_state=state,
        bucket_min_fill_frac=0.75,
    )
    assert [buffer[i]["expanded_len"] for i in indices] == [120, 121, 127]
    assert state["cursor"] == 2


def test_stream_batch_can_select_only_packable_examples():
    examples = iter([
        {"expanded_len": 120, "tokens": [1, IMAGE_TOKEN_ID, 10], "mask": [0, 0, 1], "record": {"image": "a"}},
        {"expanded_len": 120, "tokens": [1, IMAGE_TOKEN_ID, 11], "mask": [0, 0, 1], "record": {"image": "b"}},
        {"expanded_len": 120, "tokens": [1, IMAGE_TOKEN_ID, 12], "mask": [0, 0, 1], "record": {"image": "c"}},
        {"expanded_len": 120, "tokens": [1, IMAGE_TOKEN_ID, 13], "mask": [0, 0, 1], "record": {"image": "d"}},
        {"expanded_len": 80, "tokens": [1, IMAGE_TOKEN_ID, 14], "mask": [0, 0, 1], "record": {"image": "e"}},
        {"expanded_len": 80, "tokens": [1, IMAGE_TOKEN_ID, 15], "mask": [0, 0, 1], "record": {"image": "f"}},
    ])
    buffer = []
    batch, pending = next_stream_batch(
        examples,
        batch_size=4,
        max_batch_tokens=256,
        buffer=buffer,
        batch_buffer_size=6,
        rng=__import__("random").Random(0),
        pack_max_seq_len=256,
        max_images_per_row=2,
        boundary_aware_pack=True,
    )
    assert pending is None
    assert len(batch) == 2
    assert len(buffer) == 4
    row = vlm_train.batch_plan_row(
        batch,
        bucket_lens=None,
        batch_size=4,
        max_batch_tokens=256,
        max_seq_len=256,
        max_images_per_row=2,
        boundary_aware_pack=True,
    )
    assert row["dropped_examples"] == 0
    assert row["padded_tokens"] <= 256


def test_stream_packed_selection_can_choose_max_token_window():
    buffer = [
        {"expanded_len": 20},
        {"expanded_len": 21},
        {"expanded_len": 22},
        {"expanded_len": 23},
        {"expanded_len": 70},
        {"expanded_len": 71},
        {"expanded_len": 72},
        {"expanded_len": 73},
    ]
    ordered = sorted(range(len(buffer)), key=lambda i: buffer[i]["expanded_len"])
    sample = vlm_train._choose_stream_packed_indices(
        buffer,
        ordered,
        batch_size=4,
        max_batch_tokens=256,
        max_seq_len=256,
        max_images_per_row=2,
        boundary_aware=True,
    )
    max_tokens = vlm_train._choose_stream_packed_indices(
        buffer,
        ordered,
        batch_size=4,
        max_batch_tokens=256,
        max_seq_len=256,
        max_images_per_row=2,
        bucket_selection="max-tokens",
        boundary_aware=True,
    )
    assert sum(buffer[idx]["expanded_len"] for idx in max_tokens) > sum(buffer[idx]["expanded_len"] for idx in sample)
    assert [buffer[idx]["expanded_len"] for idx in max_tokens] == [73, 72]


def test_stream_packed_random_selection_samples_across_length_buffer():
    buffer = [
        {"expanded_len": 20},
        {"expanded_len": 21},
        {"expanded_len": 22},
        {"expanded_len": 23},
        {"expanded_len": 70},
        {"expanded_len": 71},
        {"expanded_len": 72},
        {"expanded_len": 73},
    ]
    ordered = sorted(range(len(buffer)), key=lambda i: buffer[i]["expanded_len"])
    selected = vlm_train._choose_stream_packed_indices(
        buffer,
        ordered,
        batch_size=4,
        max_batch_tokens=160,
        max_seq_len=256,
        max_images_per_row=2,
        bucket_selection="random",
        boundary_aware=True,
        compact_token_budget=True,
        rng=__import__("random").Random(0),
    )
    assert [buffer[idx]["expanded_len"] for idx in selected] == [70, 21, 22, 20]
    assert sum(buffer[idx]["expanded_len"] for idx in selected) <= 160


def test_stream_packed_selection_skips_unfit_fallback_windows():
    class LastWindowRng:
        def randrange(self, upper):
            return upper - 1

    buffer = [{"expanded_len": 80}, {"expanded_len": 90}, {"expanded_len": 500}, {"expanded_len": 501}]
    selected = vlm_train._choose_stream_packed_indices(
        buffer,
        list(range(len(buffer))),
        batch_size=2,
        max_batch_tokens=160,
        max_seq_len=100,
        max_images_per_row=2,
        bucket_selection="sample",
        boundary_aware=True,
        rng=LastWindowRng(),
    )
    assert selected == [0]

    selected = vlm_train._choose_stream_packed_indices(
        buffer[2:],
        [0, 1],
        batch_size=1,
        max_batch_tokens=160,
        max_seq_len=100,
        max_images_per_row=2,
        bucket_selection="sample",
        boundary_aware=True,
        rng=LastWindowRng(),
    )
    assert selected == [0]


def test_stream_packed_random_compact_selection_respects_first_token_cap():
    class ReverseRng:
        def shuffle(self, values):
            values.reverse()

    buffer = [{"expanded_len": 80}, {"expanded_len": 90}, {"expanded_len": 500}, {"expanded_len": 501}]
    selected = vlm_train._choose_stream_packed_indices(
        buffer,
        list(range(len(buffer))),
        batch_size=2,
        max_batch_tokens=200,
        max_seq_len=1000,
        max_images_per_row=2,
        bucket_selection="random",
        boundary_aware=True,
        compact_token_budget=True,
        rng=ReverseRng(),
    )
    assert selected == [1, 0]
    assert sum(buffer[idx]["expanded_len"] for idx in selected) <= 200


def test_stream_packed_max_token_selection_tiebreaks_by_segment_attention():
    buffer = [
        {"expanded_len": 90},
        {"expanded_len": 90},
        {"expanded_len": 10},
        {"expanded_len": 10},
        {"expanded_len": 50},
        {"expanded_len": 50},
        {"expanded_len": 50},
        {"expanded_len": 50},
    ]
    ordered = list(range(len(buffer)))
    selected = vlm_train._choose_stream_packed_indices(
        buffer,
        ordered,
        batch_size=4,
        max_batch_tokens=200,
        max_seq_len=100,
        max_images_per_row=2,
        bucket_selection="max-tokens",
        boundary_aware=True,
    )
    first_groups, first_lengths = vlm_train.pack_example_groups(
        buffer[:4],
        max_seq_len=100,
        max_batch_tokens=200,
        max_images_per_row=2,
        boundary_aware=True,
    )
    second_groups, second_lengths = vlm_train.pack_example_groups(
        buffer[4:],
        max_seq_len=100,
        max_batch_tokens=200,
        max_images_per_row=2,
        boundary_aware=True,
    )
    assert first_lengths == second_lengths == [100, 100]
    first_pairs = vlm_train.packed_attention_pairs(
        buffer[:4],
        first_groups,
        first_lengths,
        boundary_aware=True,
    )
    second_pairs = vlm_train.packed_attention_pairs(
        buffer[4:],
        second_groups,
        second_lengths,
        boundary_aware=True,
    )
    assert first_pairs > second_pairs
    assert [buffer[idx]["expanded_len"] for idx in selected] == [50, 50, 50, 50]


def test_stream_packed_max_compute_selection_prefers_more_attention_near_full():
    buffer = [
        {"expanded_len": 90},
        {"expanded_len": 90},
        {"expanded_len": 10},
        {"expanded_len": 10},
        {"expanded_len": 50},
        {"expanded_len": 50},
        {"expanded_len": 50},
        {"expanded_len": 50},
    ]
    ordered = list(range(len(buffer)))
    selected = vlm_train._choose_stream_packed_indices(
        buffer,
        ordered,
        batch_size=4,
        max_batch_tokens=200,
        max_seq_len=100,
        max_images_per_row=2,
        bucket_selection="max-compute",
        boundary_aware=True,
    )
    assert sorted(buffer[idx]["expanded_len"] for idx in selected) == [10, 10, 90, 90]

    selected_compact = vlm_train._choose_stream_packed_indices(
        buffer,
        ordered,
        batch_size=4,
        max_batch_tokens=200,
        max_seq_len=100,
        max_images_per_row=2,
        bucket_selection="max-compute",
        boundary_aware=True,
        compact_token_budget=True,
    )
    assert sorted(buffer[idx]["expanded_len"] for idx in selected_compact) == [10, 10, 90, 90]


def test_flattened_packed_batch_plan_reports_compact_tokens():
    examples = [{"expanded_len": 70}, {"expanded_len": 72}, {"expanded_len": 30}, {"expanded_len": 32}]
    row = vlm_train.batch_plan_row(
        examples,
        bucket_lens=None,
        batch_size=4,
        max_batch_tokens=256,
        max_seq_len=128,
        max_images_per_row=2,
        boundary_aware_pack=True,
        flatten_packed_batch=True,
    )
    assert row["rows"] == 1
    assert row["examples"] == 4
    assert row["useful_tokens"] == row["padded_tokens"] == 204
    assert row["attention_pairs"] == vlm_train.causal_attention_pairs_for_lengths([72, 70, 32, 30])
    text = vlm_train.format_batch_plan(
        vlm_train.summarize_batch_plan([row]),
        "unit",
        SimpleNamespace(
            device_batch_size=4,
            max_batch_tokens=256,
            grad_accum_steps=1,
            bucket_selection="max-tokens",
            bucket_min_fill_frac=0.0,
            bucket_cycle_repeat=1,
            pack_examples=2,
            boundary_aware_pack=True,
            flatten_packed_batch=True,
        ),
        [],
    )
    assert "attn_pairs/token=29.93" in text
    assert "attn_pairs/token" in text.splitlines()[-2]


def test_packed_large_batch_plan_fills_realistic_token_budget():
    examples = [{"expanded_len": 512} for _ in range(160)]
    row = vlm_train.batch_plan_row(
        examples,
        bucket_lens=None,
        batch_size=1024,
        max_batch_tokens=65_536,
        max_seq_len=1024,
        max_images_per_row=16,
        boundary_aware_pack=True,
        flatten_packed_batch=True,
    )
    assert row["rows"] == 1
    assert row["examples"] == 128
    assert row["dropped_examples"] == 32
    assert row["segments"] == 128
    assert row["max_segment_len"] == 512
    assert row["useful_tokens"] == row["padded_tokens"] == 65_536
    assert row["attention_pairs"] == vlm_train.causal_attention_pairs_for_lengths([512] * 128)
    assert row["attention_pairs"] < vlm_train.causal_attention_pairs(1, 65_536)


def test_compact_packed_budget_uses_total_tokens_not_dense_rows():
    examples = [{"expanded_len": 90} for _ in range(4)]
    dense_groups, dense_lengths = vlm_train.pack_example_groups(
        examples,
        max_seq_len=180,
        max_batch_tokens=300,
        max_images_per_row=2,
        boundary_aware=True,
    )
    compact_groups, compact_lengths = vlm_train.pack_example_groups(
        examples,
        max_seq_len=180,
        max_batch_tokens=300,
        max_images_per_row=2,
        boundary_aware=True,
        compact_token_budget=True,
    )
    assert sum(len(group) for group in dense_groups) == 2
    assert dense_lengths == [180]
    assert sum(len(group) for group in compact_groups) == 3
    assert sum(compact_lengths) == 270
    row = vlm_train.batch_plan_row(
        examples,
        bucket_lens=None,
        batch_size=4,
        max_batch_tokens=300,
        max_seq_len=180,
        max_images_per_row=2,
        boundary_aware_pack=True,
        flatten_packed_batch=True,
    )
    assert row["examples"] == 3
    assert row["useful_tokens"] == row["padded_tokens"] == 270


def test_direct_compact_batch_matches_compact_packer_budget():
    examples = [
        {"expanded_len": 90, "tokens": [1, IMAGE_TOKEN_ID, 10], "mask": [0, 0, 1]},
        {"expanded_len": 90, "tokens": [1, IMAGE_TOKEN_ID, 11], "mask": [0, 0, 1]},
        {"expanded_len": 90, "tokens": [1, IMAGE_TOKEN_ID, 12], "mask": [0, 0, 1]},
        {"expanded_len": 90, "tokens": [1, IMAGE_TOKEN_ID, 13], "mask": [0, 0, 1]},
    ]
    groups, lengths = vlm_train.pack_example_groups(
        examples,
        max_seq_len=180,
        max_batch_tokens=270,
        max_images_per_row=2,
        boundary_aware=True,
        compact_token_budget=True,
    )
    selected = [examples[idx] for group in groups for idx in group]
    assert sum(lengths) == 270
    assert len(selected) == 3
    trimmed = vlm_train.trim_examples_to_packable(
        examples,
        max_seq_len=180,
        max_batch_tokens=270,
        max_images_per_row=2,
        boundary_aware=True,
        compact_token_budget=True,
    )
    assert trimmed == selected

    feats = torch.randn(len(selected), VISION_TOKENS, 8)
    rows, masks, packed_feats, image_counts, segment_lengths = vlm_train.flatten_examples_as_compact_batch(
        selected,
        feats,
        max_seq_len=180,
        max_batch_tokens=270,
    )
    assert rows == [[tok for example in selected for tok in example["tokens"]]]
    assert masks == [[item for example in selected for item in example["mask"]]]
    assert packed_feats is feats
    assert image_counts == [3]
    assert segment_lengths == [[len(example["tokens"]) for example in selected]]


def test_compact_trim_preserves_selected_order_without_virtual_repack():
    examples = [
        {"expanded_len": 90, "tokens": [1, IMAGE_TOKEN_ID, 10], "mask": [0, 0, 1]},
        {"expanded_len": 10, "tokens": [1, IMAGE_TOKEN_ID, 11], "mask": [0, 0, 1]},
        {"expanded_len": 90, "tokens": [1, IMAGE_TOKEN_ID, 12], "mask": [0, 0, 1]},
    ]
    trimmed = vlm_train.trim_examples_to_packable(
        examples,
        max_seq_len=100,
        max_batch_tokens=190,
        max_images_per_row=2,
        boundary_aware=True,
        compact_token_budget=True,
    )
    assert [example["tokens"][-1] for example in trimmed] == [10, 11, 12]

    row = vlm_train.batch_plan_row(
        trimmed,
        bucket_lens=None,
        batch_size=8,
        max_batch_tokens=190,
        max_seq_len=100,
        max_images_per_row=2,
        boundary_aware_pack=True,
        flatten_packed_batch=True,
    )
    assert row["examples"] == 3
    assert row["useful_tokens"] == 190
    assert row["attention_pairs"] == vlm_train.causal_attention_pairs_for_lengths([90, 10, 90])


def test_direct_compact_plan_matches_flattened_runtime_after_skips():
    def example(tokens):
        return {
            "expanded_len": vlm_train.expanded_input_len(tokens),
            "tokens": tokens,
            "mask": [0] * (len(tokens) - 1) + [1],
        }

    examples = [
        example([1, IMAGE_TOKEN_ID, 10, 11, 12]),
        example([1, IMAGE_TOKEN_ID, 20, IMAGE_TOKEN_ID, 21, 22]),
        example([1, 30, IMAGE_TOKEN_ID, 31, 32, 33]),
        example([1, IMAGE_TOKEN_ID, 40, 41, 42, 43, 44]),
    ]
    max_seq_len = 100
    max_batch_tokens = examples[0]["expanded_len"] + examples[2]["expanded_len"]
    assert examples[1]["expanded_len"] > max_seq_len

    row = vlm_train.batch_plan_row(
        examples,
        bucket_lens=None,
        batch_size=8,
        max_batch_tokens=max_batch_tokens,
        max_seq_len=max_seq_len,
        max_images_per_row=4,
        boundary_aware_pack=True,
        flatten_packed_batch=True,
    )
    expected_lengths = [examples[0]["expanded_len"], examples[2]["expanded_len"]]
    assert row["examples"] == 2
    assert row["dropped_examples"] == 2
    assert row["useful_tokens"] == sum(expected_lengths)
    assert row["attention_pairs"] == vlm_train.causal_attention_pairs_for_lengths(expected_lengths)

    trimmed = vlm_train.trim_examples_to_packable(
        examples,
        max_seq_len=max_seq_len,
        max_batch_tokens=max_batch_tokens,
        max_images_per_row=4,
        boundary_aware=True,
        compact_token_budget=True,
    )
    assert trimmed == [examples[0], examples[2]]

    image_features = torch.arange(5 * VISION_TOKENS * 8, dtype=torch.float32).view(5, VISION_TOKENS, 8)
    rows, masks, packed_feats, image_counts, segment_lengths = vlm_train.flatten_examples_as_compact_batch(
        examples,
        image_features,
        max_seq_len=max_seq_len,
        max_batch_tokens=max_batch_tokens,
    )
    assert rows == [[*examples[0]["tokens"], *examples[2]["tokens"]]]
    assert masks == [[*examples[0]["mask"], *examples[2]["mask"]]]
    assert image_counts == [2]
    assert segment_lengths == [[len(examples[0]["tokens"]), len(examples[2]["tokens"])]]
    torch.testing.assert_close(packed_feats[:, 0, 0], image_features[[0, 3], 0, 0])

    model, projector = tiny_model()
    batch = build_multimodal_batch(
        model,
        projector,
        rows,
        packed_feats,
        loss_mask_rows=masks,
        image_counts_per_row=image_counts,
        value_fallback_token_id=1,
        segment_token_lengths_per_row=segment_lengths,
        compact_varlen_indices=True,
        return_segment_ids=False,
        return_segment_starts=False,
        return_targets=False,
        return_lengths=False,
    )
    assert batch.segment_lengths == expected_lengths
    assert batch.token_count == sum(expected_lengths)
    assert batch.padded_token_count == sum(expected_lengths)
    assert batch.attention_pairs == row["attention_pairs"]


def test_compact_packed_max_token_selection_prefers_lower_attention_near_full():
    buffer = [{"expanded_len": 500} for _ in range(4)] + [{"expanded_len": 450} for _ in range(4)]
    selected = vlm_train._choose_stream_packed_indices(
        buffer,
        list(range(len(buffer))),
        batch_size=4,
        max_batch_tokens=2048,
        max_seq_len=1000,
        max_images_per_row=2,
        bucket_selection="max-tokens",
        boundary_aware=True,
        compact_token_budget=True,
    )
    assert [buffer[idx]["expanded_len"] for idx in selected] == [450, 450, 450, 450]


def test_pack_example_rows_concatenates_short_vlm_examples():
    examples = [
        {"expanded_len": 70, "tokens": [1, 10, IMAGE_TOKEN_ID, 11, 12], "mask": [0, 0, 0, 1, 1]},
        {"expanded_len": 72, "tokens": [1, 20, IMAGE_TOKEN_ID, 21, 22], "mask": [0, 0, 0, 1, 1]},
        {"expanded_len": 180, "tokens": [1, 30, IMAGE_TOKEN_ID, 31, 32], "mask": [0, 0, 0, 1, 1]},
    ]
    assert vlm_train.packed_expanded_len(examples[:2]) == examples[0]["expanded_len"] + examples[1]["expanded_len"] + 1
    assert vlm_train.packed_expanded_len(examples[:2], boundary_aware=True) == examples[0]["expanded_len"] + examples[1]["expanded_len"]
    feats = torch.arange(3 * VISION_TOKENS * 2, dtype=torch.float32).view(3, VISION_TOKENS, 2)
    rows, masks, packed_feats, image_counts = pack_example_rows(
        examples,
        feats,
        max_seq_len=256,
        max_batch_tokens=512,
        max_images_per_row=2,
    )
    assert sum(image_counts) == 3
    assert max(image_counts) == 2
    assert any(count_image_tokens(row) == 2 for row in rows)
    assert len(rows) == len(masks) == len(image_counts)
    assert packed_feats.shape == feats.shape

    rows, masks, packed_feats, image_counts = pack_example_rows(
        examples * 4,
        feats.repeat(4, 1, 1),
        max_seq_len=256,
        max_batch_tokens=0,
        max_images_per_row=4,
        fixed_rows=3,
    )
    assert len(rows) == 3
    assert len(masks) == 3
    assert len(image_counts) == 3
    assert all(count > 0 for count in image_counts)
    assert packed_feats.shape[0] == sum(image_counts)

    rows, masks, packed_feats, image_counts = pack_example_rows(
        examples,
        feats,
        max_seq_len=64,
        max_batch_tokens=512,
        max_images_per_row=2,
    )
    assert rows == []
    assert masks == []
    assert image_counts == []
    assert packed_feats.shape[0] == 0


def test_pack_example_rows_preserves_multi_image_feature_spans():
    examples = [
        {"expanded_len": 70, "tokens": [1, 10, IMAGE_TOKEN_ID, 11, 12], "mask": [0, 0, 0, 1, 1]},
        {"expanded_len": 80, "tokens": [1, 20, IMAGE_TOKEN_ID, 21, IMAGE_TOKEN_ID, 22], "mask": [0, 0, 0, 1, 0, 1]},
    ]
    feats = torch.arange(3 * VISION_TOKENS * 2, dtype=torch.float32).view(3, VISION_TOKENS, 2)
    rows, masks, packed_feats, image_counts, segment_lengths = pack_example_rows(
        examples,
        feats,
        max_seq_len=256,
        max_batch_tokens=0,
        max_images_per_row=3,
        boundary_aware=True,
        return_segment_lengths=True,
    )
    assert rows == [[*examples[1]["tokens"], *examples[0]["tokens"]]]
    assert masks == [[*examples[1]["mask"], *examples[0]["mask"]]]
    assert image_counts == [3]
    assert segment_lengths == [[len(examples[1]["tokens"]), len(examples[0]["tokens"])]]
    torch.testing.assert_close(packed_feats, torch.cat([feats[1:3], feats[0:1]], dim=0))


def test_pack_example_groups_fixed_rows_do_not_add_empty_row_boundary():
    examples = [
        {"expanded_len": 120, "tokens": [1, IMAGE_TOKEN_ID, 10], "mask": [0, 0, 1]},
        {"expanded_len": 120, "tokens": [1, IMAGE_TOKEN_ID, 11], "mask": [0, 0, 1]},
    ]
    groups, lengths = vlm_train.pack_example_groups(
        examples,
        max_seq_len=120,
        max_batch_tokens=240,
        max_images_per_row=1,
        fixed_rows=2,
        boundary_aware=False,
    )
    assert groups == [[0], [1]]
    assert lengths == [120, 120]


def test_pack_example_groups_single_rows_respect_caps():
    examples = [
        {"expanded_len": 120, "tokens": [1, IMAGE_TOKEN_ID, 10], "mask": [0, 0, 1]},
        {"expanded_len": 80, "tokens": [1, IMAGE_TOKEN_ID, 11], "mask": [0, 0, 1]},
        {"expanded_len": 50, "tokens": [1, IMAGE_TOKEN_ID, 12], "mask": [0, 0, 1]},
    ]
    groups, lengths = vlm_train.pack_example_groups(
        examples,
        max_seq_len=100,
        max_batch_tokens=160,
        max_images_per_row=1,
    )
    assert groups == [[1], [2]]
    assert lengths == [80, 50]

    groups, lengths = vlm_train.pack_example_groups(
        examples,
        max_seq_len=100,
        max_batch_tokens=100,
        max_images_per_row=1,
    )
    assert groups == [[1]]
    assert lengths == [80]

    groups, lengths = vlm_train.pack_example_groups(
        examples[:1],
        max_seq_len=100,
        max_batch_tokens=0,
        max_images_per_row=2,
    )
    assert groups == []
    assert lengths == []


def test_pack_example_groups_preserves_token_cap_with_incremental_max():
    examples = [
        {"expanded_len": 120, "tokens": [1, IMAGE_TOKEN_ID, 10], "mask": [0, 0, 1]},
        {"expanded_len": 120, "tokens": [1, IMAGE_TOKEN_ID, 11], "mask": [0, 0, 1]},
        {"expanded_len": 80, "tokens": [1, IMAGE_TOKEN_ID, 12], "mask": [0, 0, 1]},
    ]
    groups, lengths = vlm_train.pack_example_groups(
        examples,
        max_seq_len=240,
        max_batch_tokens=240,
        max_images_per_row=2,
        boundary_aware=True,
    )
    assert len(groups) == 1
    assert lengths == [240]
    assert max(lengths) * len(lengths) <= 240


def test_boundary_aware_packed_multimodal_loss_matches_separate_examples(monkeypatch):
    import nanochat.gpt as gpt_module

    model, projector = tiny_model()
    with torch.no_grad():
        model.smear_lambda.fill_(1.0)
    examples = [
        {"expanded_len": 70, "tokens": [1, 10, IMAGE_TOKEN_ID, 11, 12, 13], "mask": [0, 0, 0, 1, 1, 1]},
        {"expanded_len": 72, "tokens": [1, 20, IMAGE_TOKEN_ID, 21, 22, 23], "mask": [0, 0, 0, 1, 1, 1]},
    ]
    feats = torch.randn(2, VISION_TOKENS, 8)
    packed = pack_example_rows(
        examples,
        feats,
        max_seq_len=160,
        max_batch_tokens=0,
        max_images_per_row=2,
        return_segment_lengths=True,
    )
    rows, masks, packed_feats, image_counts, segment_lengths = packed
    assert len(rows) == 1
    assert segment_lengths == [[len(examples[0]["tokens"]), len(examples[1]["tokens"])]]

    packed_batch = build_multimodal_batch(
        model,
        projector,
        rows,
        packed_feats,
        loss_mask_rows=masks,
        image_counts_per_row=image_counts,
        value_fallback_token_id=1,
        segment_token_lengths_per_row=segment_lengths,
    )
    first_segment_len = vlm_train.expanded_input_len(examples[0]["tokens"])
    second_start = first_segment_len
    assert packed_batch.segment_starts[0, 0]
    assert packed_batch.segment_starts[0, second_start]
    assert int(packed_batch.position_ids[0, second_start]) == 0
    assert int(packed_batch.segment_ids[0, second_start - 1]) == 0
    assert int(packed_batch.segment_ids[0, second_start]) == 1
    assert packed_batch.cu_seqlens.tolist() == [0, first_segment_len, packed_batch.lengths[0].item()]
    assert packed_batch.max_segment_len == first_segment_len
    assert packed_batch.varlen_indices.numel() == int(packed_batch.lengths.sum())
    assert packed_batch.segment_lengths == [first_segment_len, vlm_train.expanded_input_len(examples[1]["tokens"])]
    assert int(packed_batch.lengths[0]) == sum(vlm_train.expanded_input_len(example["tokens"]) for example in examples)
    assert packed_batch.attention_pairs == vlm_train.causal_attention_pairs_for_lengths(packed_batch.segment_lengths)

    varlen_calls = []
    original_varlen = gpt_module.flash_attn.flash_attn_varlen_func

    def record_varlen(*args, **kwargs):
        varlen_calls.append(args[3].detach().cpu().tolist())
        return original_varlen(*args, **kwargs)

    monkeypatch.setattr(gpt_module.flash_attn, "flash_attn_varlen_func", record_varlen)
    packed_loss = model(
        packed_batch.value_token_ids,
        packed_batch.targets,
        input_embeds=packed_batch.input_embeds,
        loss_reduction="none",
        selective_loss=True,
        position_ids=packed_batch.position_ids,
        segment_ids=packed_batch.segment_ids,
        segment_starts=packed_batch.segment_starts,
        cu_seqlens=packed_batch.cu_seqlens,
        max_seqlen=packed_batch.max_segment_len,
        varlen_indices=packed_batch.varlen_indices,
        loss_indices=packed_batch.loss_indices,
        loss_targets=packed_batch.loss_targets,
    )
    assert varlen_calls == [packed_batch.cu_seqlens.tolist()] * model.config.n_layer
    packed_valid = packed_loss[packed_batch.targets.view(-1) != -1]

    separate_valid = []
    for idx, example in enumerate(examples):
        batch = build_multimodal_batch(
            model,
            projector,
            [example["tokens"]],
            feats[idx:idx + 1],
            loss_mask_rows=[example["mask"]],
            value_fallback_token_id=1,
        )
        loss = model(
            batch.value_token_ids,
            batch.targets,
            input_embeds=batch.input_embeds,
            loss_reduction="none",
            selective_loss=True,
        )
        separate_valid.append(loss[batch.targets.view(-1) != -1])
    torch.testing.assert_close(packed_valid, torch.cat(separate_valid), rtol=2e-3, atol=1e-2)


def test_sparse_segment_start_indices_match_dense_smear_mask():
    model, projector = tiny_model()
    with torch.no_grad():
        model.smear_lambda.fill_(1.0)
    examples = [
        {"expanded_len": 70, "tokens": [1, 10, IMAGE_TOKEN_ID, 11, 12, 13], "mask": [0, 0, 0, 1, 1, 1]},
        {"expanded_len": 72, "tokens": [1, 20, IMAGE_TOKEN_ID, 21, 22, 23], "mask": [0, 0, 0, 1, 1, 1]},
    ]
    feats = torch.randn(2, VISION_TOKENS, 8)
    rows, masks, packed_feats, image_counts, segment_lengths = pack_example_rows(
        examples,
        feats,
        max_seq_len=160,
        max_batch_tokens=0,
        max_images_per_row=2,
        boundary_aware=True,
        return_segment_lengths=True,
    )
    dense_batch = build_multimodal_batch(
        model,
        projector,
        rows,
        packed_feats,
        loss_mask_rows=masks,
        image_counts_per_row=image_counts,
        value_fallback_token_id=1,
        segment_token_lengths_per_row=segment_lengths,
    )
    sparse_batch = build_multimodal_batch(
        model,
        projector,
        rows,
        packed_feats,
        loss_mask_rows=masks,
        image_counts_per_row=image_counts,
        value_fallback_token_id=1,
        segment_token_lengths_per_row=segment_lengths,
        return_segment_starts=False,
    )
    second_start = vlm_train.expanded_input_len(examples[0]["tokens"])
    assert sparse_batch.segment_starts is None
    assert sparse_batch.segment_start_indices.tolist() == [second_start - 1]
    dense_loss = model(
        dense_batch.value_token_ids,
        dense_batch.targets,
        input_embeds=dense_batch.input_embeds,
        loss_reduction="none",
        selective_loss=True,
        position_ids=dense_batch.position_ids,
        segment_ids=dense_batch.segment_ids,
        segment_starts=dense_batch.segment_starts,
        cu_seqlens=dense_batch.cu_seqlens,
        max_seqlen=dense_batch.max_segment_len,
        varlen_indices=dense_batch.varlen_indices,
        loss_indices=dense_batch.loss_indices,
        loss_targets=dense_batch.loss_targets,
    )
    sparse_loss = model(
        sparse_batch.value_token_ids,
        sparse_batch.targets,
        input_embeds=sparse_batch.input_embeds,
        loss_reduction="none",
        selective_loss=True,
        position_ids=sparse_batch.position_ids,
        segment_ids=sparse_batch.segment_ids,
        segment_start_indices=sparse_batch.segment_start_indices,
        cu_seqlens=sparse_batch.cu_seqlens,
        max_seqlen=sparse_batch.max_segment_len,
        varlen_indices=sparse_batch.varlen_indices,
        loss_indices=sparse_batch.loss_indices,
        loss_targets=sparse_batch.loss_targets,
    )
    torch.testing.assert_close(sparse_loss, dense_loss, rtol=0, atol=0)


def test_compact_segment_start_indices_derive_from_segment_lengths():
    model, projector = tiny_model()
    with torch.no_grad():
        model.smear_lambda.fill_(1.0)
    examples = [
        {"tokens": [1, 10, IMAGE_TOKEN_ID, 11, 12, 13], "mask": [0, 0, 0, 1, 1, 1]},
        {"tokens": [1, 20, IMAGE_TOKEN_ID, 21, 22], "mask": [0, 0, 0, 1, 1]},
        {"tokens": [1, 30, IMAGE_TOKEN_ID, 31, 32, 33], "mask": [0, 0, 0, 1, 1, 1]},
    ]
    for example in examples:
        example["expanded_len"] = vlm_train.expanded_input_len(example["tokens"])
    feats = torch.randn(len(examples), VISION_TOKENS, 8)
    rows, masks, packed_feats, image_counts, segment_lengths = vlm_train.flatten_examples_as_compact_batch(
        examples,
        feats,
        max_seq_len=140,
        max_batch_tokens=0,
    )
    dense_batch = build_multimodal_batch(
        model,
        projector,
        rows,
        packed_feats,
        loss_mask_rows=masks,
        image_counts_per_row=image_counts,
        value_fallback_token_id=1,
        segment_token_lengths_per_row=segment_lengths,
        compact_varlen_indices=True,
        return_segment_ids=False,
    )
    sparse_batch = build_multimodal_batch(
        model,
        projector,
        rows,
        packed_feats,
        loss_mask_rows=masks,
        image_counts_per_row=image_counts,
        value_fallback_token_id=1,
        segment_token_lengths_per_row=segment_lengths,
        compact_varlen_indices=True,
        return_segment_ids=False,
        return_segment_starts=False,
    )
    expected = []
    start = 0
    for length in dense_batch.segment_lengths:
        if start > 0:
            expected.append(start - 1)
        start += length
    assert sparse_batch.segment_starts is None
    assert sparse_batch.segment_start_indices.tolist() == expected
    torch.testing.assert_close(sparse_batch.segment_start_indices, sparse_batch.cu_seqlens[1:-1].long() - 1)
    dense_loss = model(
        dense_batch.value_token_ids,
        dense_batch.targets,
        input_embeds=dense_batch.input_embeds,
        selective_loss=True,
        position_ids=dense_batch.position_ids,
        segment_starts=dense_batch.segment_starts,
        cu_seqlens=dense_batch.cu_seqlens,
        max_seqlen=dense_batch.max_segment_len,
        varlen_indices=dense_batch.varlen_indices,
        loss_indices=dense_batch.loss_indices,
        loss_targets=dense_batch.loss_targets,
    )
    sparse_loss = model(
        sparse_batch.value_token_ids,
        sparse_batch.targets,
        input_embeds=sparse_batch.input_embeds,
        selective_loss=True,
        position_ids=sparse_batch.position_ids,
        segment_start_indices=sparse_batch.segment_start_indices,
        cu_seqlens=sparse_batch.cu_seqlens,
        max_seqlen=sparse_batch.max_segment_len,
        varlen_indices=sparse_batch.varlen_indices,
        loss_indices=sparse_batch.loss_indices,
        loss_targets=sparse_batch.loss_targets,
    )
    torch.testing.assert_close(sparse_loss, dense_loss, rtol=0, atol=0)


def test_compact_boundary_aware_packed_batch_matches_padded_rows():
    model, projector = tiny_model()
    examples = [
        {"tokens": [1, 10, IMAGE_TOKEN_ID, 11, 12, 13], "mask": [0, 0, 0, 1, 1, 1]},
        {"tokens": [1, 20, IMAGE_TOKEN_ID, 21, 22], "mask": [0, 0, 0, 1, 1]},
        {"tokens": [1, 30, IMAGE_TOKEN_ID, 31, 32, 33, 34], "mask": [0, 0, 0, 1, 1, 1, 1]},
        {"tokens": [1, 40, IMAGE_TOKEN_ID, 41], "mask": [0, 0, 0, 1]},
    ]
    for example in examples:
        example["expanded_len"] = vlm_train.expanded_input_len(example["tokens"])
    feats = torch.randn(len(examples), VISION_TOKENS, 8)
    rows, masks, packed_feats, image_counts, segment_lengths = pack_example_rows(
        examples,
        feats,
        max_seq_len=140,
        max_batch_tokens=0,
        max_images_per_row=2,
        boundary_aware=True,
        return_segment_lengths=True,
    )
    assert len(rows) == 2

    padded_batch = build_multimodal_batch(
        model,
        projector,
        rows,
        packed_feats,
        loss_mask_rows=masks,
        image_counts_per_row=image_counts,
        value_fallback_token_id=1,
        segment_token_lengths_per_row=segment_lengths,
    )
    assert padded_batch.varlen_indices is not None
    assert padded_batch.varlen_indices.numel() < padded_batch.input_embeds.shape[0] * padded_batch.input_embeds.shape[1]

    flat_rows, flat_masks, flat_image_counts, flat_segment_lengths = vlm_train.flatten_packed_rows(
        rows,
        masks,
        image_counts,
        segment_lengths,
    )
    compact_batch = build_multimodal_batch(
        model,
        projector,
        flat_rows,
        packed_feats,
        loss_mask_rows=flat_masks,
        image_counts_per_row=flat_image_counts,
        value_fallback_token_id=1,
        segment_token_lengths_per_row=flat_segment_lengths,
        compact_varlen_indices=True,
        return_segment_ids=False,
    )
    assert compact_batch.varlen_indices is None
    assert compact_batch.segment_ids is None
    assert compact_batch.input_embeds.shape[0] == 1
    assert int(compact_batch.lengths.sum()) == int(padded_batch.lengths.sum())
    assert compact_batch.cu_seqlens.tolist() == padded_batch.cu_seqlens.tolist()
    assert compact_batch.segment_lengths == padded_batch.segment_lengths
    assert compact_batch.attention_pairs == padded_batch.attention_pairs

    boundary_kwargs = {
        "position_ids": padded_batch.position_ids,
        "segment_ids": padded_batch.segment_ids,
        "segment_starts": padded_batch.segment_starts,
        "cu_seqlens": padded_batch.cu_seqlens,
        "max_seqlen": padded_batch.max_segment_len,
        "varlen_indices": padded_batch.varlen_indices,
    }
    padded_loss = model(
        padded_batch.value_token_ids,
        padded_batch.targets,
        input_embeds=padded_batch.input_embeds,
        loss_reduction="none",
        selective_loss=True,
        **boundary_kwargs,
    )
    compact_loss = model(
        compact_batch.value_token_ids,
        compact_batch.targets,
        input_embeds=compact_batch.input_embeds,
        loss_reduction="none",
        selective_loss=True,
        position_ids=compact_batch.position_ids,
        segment_starts=compact_batch.segment_starts,
        cu_seqlens=compact_batch.cu_seqlens,
        max_seqlen=compact_batch.max_segment_len,
        varlen_indices=compact_batch.varlen_indices,
    )
    padded_valid = padded_loss[padded_batch.targets.view(-1) != -1]
    compact_valid = compact_loss[compact_batch.targets.view(-1) != -1]
    torch.testing.assert_close(compact_valid, padded_valid, rtol=2e-3, atol=1e-2)


def test_flatten_examples_as_compact_batch_keeps_segments_and_feature_order():
    examples = [
        {"expanded_len": 70, "tokens": [1, 10, IMAGE_TOKEN_ID, 11, 12], "mask": [0, 0, 0, 1, 1]},
        {"expanded_len": 72, "tokens": [1, 20, IMAGE_TOKEN_ID, 21, 22], "mask": [0, 0, 0, 1, 1]},
        {"expanded_len": 90, "tokens": [1, 30, IMAGE_TOKEN_ID, 31, 32], "mask": [0, 0, 0, 1, 1]},
    ]
    feats = torch.arange(3 * VISION_TOKENS * 2, dtype=torch.float32).view(3, VISION_TOKENS, 2)
    rows, masks, packed_feats, image_counts, segment_lengths = vlm_train.flatten_examples_as_compact_batch(
        examples,
        feats,
        max_seq_len=128,
        max_batch_tokens=142,
    )
    assert rows == [[*examples[0]["tokens"], *examples[1]["tokens"]]]
    assert masks == [[*examples[0]["mask"], *examples[1]["mask"]]]
    assert image_counts == [2]
    assert segment_lengths == [[len(examples[0]["tokens"]), len(examples[1]["tokens"])]]
    torch.testing.assert_close(packed_feats, feats[:2])
    assert packed_feats.data_ptr() == feats.data_ptr()

    rows, masks, packed_feats, image_counts, segment_lengths = vlm_train.flatten_examples_as_compact_batch(
        examples[:1],
        feats[:1],
        max_seq_len=128,
        max_batch_tokens=32,
    )
    assert rows == []
    assert image_counts == []
    assert segment_lengths == []
    assert packed_feats.shape[0] == 0

    reordered_examples = [examples[2], examples[0]]
    reordered_feats = torch.cat([feats[2:3], feats[0:1]], dim=0)
    rows, masks, packed_feats, image_counts, segment_lengths = vlm_train.flatten_examples_as_compact_batch(
        reordered_examples,
        reordered_feats,
        max_seq_len=128,
        max_batch_tokens=80,
    )
    assert rows == [examples[0]["tokens"]]
    assert image_counts == [1]
    assert segment_lengths == [[len(examples[0]["tokens"])]]
    torch.testing.assert_close(packed_feats, reordered_feats[1:2])
    assert packed_feats.data_ptr() != reordered_feats.data_ptr()

    multi_image_examples = [
        {"expanded_len": 70, "tokens": [1, 10, IMAGE_TOKEN_ID, 11, 12], "mask": [0, 0, 0, 1, 1]},
        {"expanded_len": 80, "tokens": [1, 20, IMAGE_TOKEN_ID, 21, IMAGE_TOKEN_ID, 22], "mask": [0, 0, 0, 1, 0, 1]},
    ]
    multi_feats = torch.arange(3 * VISION_TOKENS * 2, dtype=torch.float32).view(3, VISION_TOKENS, 2)
    rows, masks, packed_feats, image_counts, segment_lengths = vlm_train.flatten_examples_as_compact_batch(
        multi_image_examples,
        multi_feats,
        max_seq_len=128,
        max_batch_tokens=0,
    )
    assert rows == [[*multi_image_examples[0]["tokens"], *multi_image_examples[1]["tokens"]]]
    assert image_counts == [3]
    assert segment_lengths == [[len(example["tokens"]) for example in multi_image_examples]]
    torch.testing.assert_close(packed_feats, multi_feats)


def test_packed_example_count_uses_segments_not_image_markers():
    assert vlm_train.packed_example_count([1, 2]) == 3
    assert vlm_train.packed_example_count([3], [[5, 6]]) == 2
    assert vlm_train.packed_example_count([1, 2], [[5], [6]]) == 2


def test_direct_compact_batch_loss_matches_repack_then_flatten_path():
    model, projector = tiny_model()
    examples = [
        {"tokens": [1, 10, IMAGE_TOKEN_ID, 11, 12, 13], "mask": [0, 0, 0, 1, 1, 1]},
        {"tokens": [1, 20, IMAGE_TOKEN_ID, 21, 22], "mask": [0, 0, 0, 1, 1]},
        {"tokens": [1, 30, IMAGE_TOKEN_ID, 31, 32, 33, 34], "mask": [0, 0, 0, 1, 1, 1, 1]},
        {"tokens": [1, 40, IMAGE_TOKEN_ID, 41], "mask": [0, 0, 0, 1]},
    ]
    for example in examples:
        example["expanded_len"] = vlm_train.expanded_input_len(example["tokens"])
    feats = torch.randn(len(examples), VISION_TOKENS, 8)

    packed_rows, packed_masks, packed_feats, image_counts, segment_lengths = pack_example_rows(
        examples,
        feats,
        max_seq_len=140,
        max_batch_tokens=0,
        max_images_per_row=2,
        boundary_aware=True,
        return_segment_lengths=True,
        compact_token_budget=True,
    )
    flat_rows, flat_masks, flat_image_counts, flat_segment_lengths = vlm_train.flatten_packed_rows(
        packed_rows,
        packed_masks,
        image_counts,
        segment_lengths,
    )
    repacked_batch = build_multimodal_batch(
        model,
        projector,
        flat_rows,
        packed_feats,
        loss_mask_rows=flat_masks,
        image_counts_per_row=flat_image_counts,
        value_fallback_token_id=1,
        segment_token_lengths_per_row=flat_segment_lengths,
        compact_varlen_indices=True,
        return_segment_ids=False,
    )

    direct_rows, direct_masks, direct_feats, direct_image_counts, direct_segment_lengths = vlm_train.flatten_examples_as_compact_batch(
        examples,
        feats,
        max_seq_len=140,
        max_batch_tokens=0,
    )
    direct_batch = build_multimodal_batch(
        model,
        projector,
        direct_rows,
        direct_feats,
        loss_mask_rows=direct_masks,
        image_counts_per_row=direct_image_counts,
        value_fallback_token_id=1,
        segment_token_lengths_per_row=direct_segment_lengths,
        compact_varlen_indices=True,
        return_segment_ids=False,
    )
    assert sum(direct_batch.segment_lengths) == int(direct_batch.lengths.sum())
    assert direct_batch.attention_pairs == repacked_batch.attention_pairs
    assert direct_batch.attention_pairs == vlm_train.causal_attention_pairs_for_lengths(direct_batch.segment_lengths)
    assert sorted(direct_batch.segment_lengths) == sorted(repacked_batch.segment_lengths)

    def compact_loss(batch):
        return model(
            batch.value_token_ids,
            batch.targets,
            input_embeds=batch.input_embeds,
            selective_loss=True,
            position_ids=batch.position_ids,
            segment_starts=batch.segment_starts,
            cu_seqlens=batch.cu_seqlens,
            max_seqlen=batch.max_segment_len,
            varlen_indices=batch.varlen_indices,
            loss_indices=batch.loss_indices,
            loss_targets=batch.loss_targets,
        )

    torch.testing.assert_close(compact_loss(direct_batch), compact_loss(repacked_batch), rtol=2e-3, atol=1e-2)


def test_direct_compact_batch_loss_matches_separate_examples():
    model, projector = tiny_model()
    with torch.no_grad():
        model.smear_lambda.fill_(1.0)
    examples = [
        {"tokens": [1, 10, IMAGE_TOKEN_ID, 11, 12, 13], "mask": [0, 0, 0, 1, 1, 1]},
        {"tokens": [1, 20, IMAGE_TOKEN_ID, 21, 22], "mask": [0, 0, 0, 1, 1]},
        {"tokens": [1, 30, IMAGE_TOKEN_ID, 31, 32, 33], "mask": [0, 0, 0, 1, 1, 1]},
    ]
    for example in examples:
        example["expanded_len"] = vlm_train.expanded_input_len(example["tokens"])
    feats = torch.randn(len(examples), VISION_TOKENS, 8)

    rows, masks, packed_feats, image_counts, segment_lengths = vlm_train.flatten_examples_as_compact_batch(
        examples,
        feats,
        max_seq_len=140,
        max_batch_tokens=0,
    )
    compact_batch = build_multimodal_batch(
        model,
        projector,
        rows,
        packed_feats,
        loss_mask_rows=masks,
        image_counts_per_row=image_counts,
        value_fallback_token_id=1,
        segment_token_lengths_per_row=segment_lengths,
        compact_varlen_indices=True,
        return_segment_ids=False,
    )
    assert compact_batch.segment_ids is None
    assert compact_batch.varlen_indices is None
    assert compact_batch.cu_seqlens.tolist() == [0, *torch.tensor(compact_batch.segment_lengths).cumsum(0).tolist()]
    assert compact_batch.segment_starts[0, 0]
    assert compact_batch.segment_starts[0, compact_batch.segment_lengths[0]]
    assert compact_batch.position_ids.tolist() == [[pos for length in compact_batch.segment_lengths for pos in range(length)]]
    assert int(compact_batch.position_ids[0, compact_batch.segment_lengths[0]]) == 0
    projected = projector(packed_feats.reshape(-1, packed_feats.size(-1))).view(len(examples), VISION_TOKENS, -1)
    cursor = 0
    for idx, example in enumerate(examples):
        image_start = cursor + example["tokens"].index(IMAGE_TOKEN_ID)
        torch.testing.assert_close(compact_batch.input_embeds[0, image_start:image_start + VISION_TOKENS], projected[idx])
        cursor += vlm_train.expanded_input_len(example["tokens"])

    compact_loss = model(
        compact_batch.value_token_ids,
        compact_batch.targets,
        input_embeds=compact_batch.input_embeds,
        loss_reduction="none",
        selective_loss=True,
        position_ids=compact_batch.position_ids,
        segment_starts=compact_batch.segment_starts,
        cu_seqlens=compact_batch.cu_seqlens,
        max_seqlen=compact_batch.max_segment_len,
        varlen_indices=compact_batch.varlen_indices,
    )
    compact_valid = compact_loss[compact_batch.targets.view(-1) != -1]

    separate_valid = []
    for idx, example in enumerate(examples):
        batch = build_multimodal_batch(
            model,
            projector,
            [example["tokens"]],
            feats[idx:idx + 1],
            loss_mask_rows=[example["mask"]],
            value_fallback_token_id=1,
        )
        loss = model(
            batch.value_token_ids,
            batch.targets,
            input_embeds=batch.input_embeds,
            loss_reduction="none",
            selective_loss=True,
        )
        separate_valid.append(loss[batch.targets.view(-1) != -1])
    torch.testing.assert_close(compact_valid, torch.cat(separate_valid), rtol=2e-3, atol=1e-2)


def test_iter_hf_records_can_stream_all_configs(monkeypatch):
    calls = []

    def fake_get_dataset_config_names(repo):
        assert repo == "repo"
        return ["a", "b"]

    def fake_load_dataset(repo, config, **kwargs):
        calls.append((repo, config, kwargs["streaming"]))
        return iter([
            {
                "images": [Image.new("RGB", (4, 4), color=(1, 2, 3))],
                "texts": [{"user": f"{config}?", "assistant": config}],
            }
        ])

    def fake_interleave_datasets(streams, **kwargs):
        assert kwargs["stopping_strategy"] == "all_exhausted"
        for stream in streams:
            yield from stream

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(
            get_dataset_config_names=fake_get_dataset_config_names,
            interleave_datasets=fake_interleave_datasets,
            load_dataset=fake_load_dataset,
        ),
    )
    args = SimpleNamespace(hf_repo="repo", hf_config="all")
    stream = iter_hf_records(args)
    assert next(stream)["texts"][0]["assistant"] == "a"
    assert next(stream)["texts"][0]["assistant"] == "b"
    assert calls == [("repo", "a", True), ("repo", "b", True)]


def test_local_and_finevision_image_loading(tmp_path):
    img = Image.new("RGB", (4, 4), color=(1, 2, 3))
    img_path = tmp_path / "tiny.jpg"
    img.save(img_path)
    loaded = open_image({"image": "tiny.jpg"}, tmp_path)
    assert loaded.size == (4, 4)

    direct = open_image({"images": [Image.new("RGB", (3, 2), color=(1, 1, 1))]}, tmp_path)
    assert direct.size == (3, 2)

    import io

    encoded = io.BytesIO()
    Image.new("RGB", (2, 2), color=(9, 8, 7)).save(encoded, format="PNG")
    from_bytes = open_image({"images": [{"bytes": encoded.getvalue(), "path": None}]}, tmp_path)
    assert from_bytes.size == (2, 2)
    assert image_record_is_openable({"images": [{"bytes": encoded.getvalue(), "path": None}]})
    assert not image_record_is_openable({"images": [{"bytes": None, "path": "/definitely/missing.jpg"}]})
    assert render_records([{"images": [{"bytes": None, "path": "/definitely/missing.jpg"}], "texts": [{"user": "Q?", "assistant": "A."}]}], TinyTokenizer(), max_seq_len=256, require_openable_image=False)
    with pytest.raises(AssertionError, match="no usable"):
        render_records(
            [{"images": [{"bytes": None, "path": "/definitely/missing.jpg"}], "texts": [{"user": "Q?", "assistant": "A."}]}],
            TinyTokenizer(),
            max_seq_len=256,
            require_openable_image=True,
        )


def test_batch_features_can_skip_dead_images(tmp_path):
    Image.new("RGB", (4, 4), color=(1, 2, 3)).save(tmp_path / "ok.jpg")
    examples = [{"record": {"image": "ok.jpg"}}, {"record": {"image": "missing.jpg"}}]

    class Extractor:
        def __call__(self, images):
            return torch.stack([torch.full((VISION_TOKENS, 2), float(sum(image.getpixel((0, 0))))) for image in images])

    feats, kept = batch_features_and_examples(Extractor(), examples, tmp_path, skip_bad_images=True)
    assert len(kept) == 1
    assert feats.shape == (1, VISION_TOKENS, 2)


def test_prepare_training_batch_can_prefetch_processor(tmp_path):
    Image.new("RGB", (4, 4), color=(1, 2, 3)).save(tmp_path / "ok.jpg")
    examples = [{"record": {"image": "ok.jpg"}}]

    class Extractor:
        def preprocess(self, images, profile=None):
            assert len(images) == 1
            if profile is not None:
                profile["image_processor"] += 0.25
            return torch.ones(1, 3, 4, 4)

    prepared = prepare_training_batch(lambda: (examples, 3), Extractor(), tmp_path, profile_timing=True, prefetch_processor=True)
    assert prepared.images is None
    assert prepared.selected_examples == 3
    assert prepared.pixel_values.shape == (1, 3, 4, 4)
    assert prepared.profile["image_open"] > 0
    assert prepared.profile["image_processor"] == 0.25


def test_profile_includes_split_optimizer_timing_keys():
    profile = vlm_train.new_profile()
    assert "batch_projector" in profile
    assert "optim" in profile
    assert "optim_projector" in profile
    assert "optim_llm" in profile
    assert profile["batch_projector"] == 0.0
    assert profile["optim"] == 0.0
    assert profile["optim_projector"] == 0.0
    assert profile["optim_llm"] == 0.0


def test_cuda_memory_stats_cpu_reports_zeroes():
    stats = vlm_train.cuda_memory_stats_mib("cpu")
    assert stats == {
        "allocated": 0.0,
        "reserved": 0.0,
        "max_allocated": 0.0,
        "max_reserved": 0.0,
    }


def test_profile_summary_reports_warmup_excluded_timing_percentages():
    profile = vlm_train.new_profile()
    profile["fwdbwd"] = 3.0
    profile["optim"] = 1.0
    profile["image_siglip"] = 0.5
    text = vlm_train.format_profile_summary("Steady timing totals", profile, total_seconds=4.0)
    assert text.startswith("Steady timing totals wall=4.000s")
    assert "fwdbwd=3.000s/75.0%" in text
    assert "optim=1.000s/25.0%" in text
    assert "image_siglip=0.500s/12.5%" in text
    assert "other=0.000s/0.0%" in text
    assert vlm_train.profile_other_seconds(profile, total_seconds=4.0) == 0.0


def test_vlm_bpb_eval_scores_supervised_targets(tmp_path):
    Image.new("RGB", (4, 4), color=(1, 2, 3)).save(tmp_path / "tiny.jpg")
    records = [{
        "image": "tiny.jpg",
        "conversations": [
            {"from": "human", "value": f"{IMAGE_MARKER}\nWhat color?"},
            {"from": "gpt", "value": "red"},
        ],
    }]
    tokenizer = TinyTokenizer()
    examples = render_records(records, tokenizer, max_seq_len=128)
    model, projector = tiny_model()
    model.train()
    projector.train()

    class Extractor:
        def __call__(self, images):
            return torch.randn(len(images), VISION_TOKENS, 8)

    stats = evaluate_vlm_bpb(
        model,
        projector,
        Extractor(),
        examples,
        tmp_path,
        torch.device("cpu"),
        torch.ones(tokenizer.get_vocab_size(), dtype=torch.long),
        batch_size=1,
        max_seq_len=128,
    )
    assert stats["n"] == 1
    assert stats["bytes"] > 0
    assert stats["target_tokens"] > 0
    assert torch.isfinite(torch.tensor(stats["bpb"]))
    assert model.training
    assert projector.training


def test_eval_prompt_matching_and_samples():
    record = {"question": "What?", "options": ["red", "blue"], "answer": 1}
    prompt = make_prompt(record)
    assert IMAGE_MARKER in prompt
    assert get_answers(record) == ["B", "blue"]
    assert exact_or_choice_match("Answer: B", ["B"])
    assert not exact_or_choice_match("a chart", ["A"])
    assert coerce_options("['cat', 'dog']") == ["cat", "dog"]
    assert parse_inline_options("Options: A: cat, B: dog") == ["cat", "dog"]
    assert benchmark_specs(["mmmu"], mmmu_configs="Accounting,Basic_Medical_Science")[1]["key"] == "mmmu_Basic_Medical_Science"

    sample = make_result_sample(record, 3, "A", ["A"], True)
    assert sample["prediction_correct"] is True


def test_evaluate_vlm_small_loop_restores_train_mode(monkeypatch):
    image = Image.new("RGB", (4, 4), color=(255, 0, 0))

    def fake_load_benchmark(name, config=None):
        return [{"question": "What color?", "options": ["red", "blue"], "answer": "A", "image": image}]

    class Extractor:
        def __call__(self, images):
            return torch.randn(len(images), VISION_TOKENS, 8)

    monkeypatch.setattr("scripts.vlm_eval.load_benchmark", fake_load_benchmark)
    model, projector = tiny_model()
    model.train()
    projector.train()
    results = evaluate_vlm(
        model,
        projector,
        TinyTokenizer(),
        Extractor(),
        benchmarks="mmstar",
        limit=1,
        max_scan=1,
        max_new_tokens=1,
    )
    assert results["benchmarks"]["mmstar"]["n"] == 1
    assert model.training
    assert projector.training


def test_modal_command_builders():
    import modal_vlm

    train = modal_vlm.build_train_cmd(init_checkpoint_step=250, max_examples=4, profile_timing=True)
    assert train[:3] == ["python", "-m", "scripts.vlm_train"]
    assert "--stage" not in train
    assert train[train.index("--hf-repo") + 1] == "HuggingFaceM4/the_cauldron"
    assert train[train.index("--hf-config") + 1] == "all"
    assert train[train.index("--grad-accum-steps") + 1] == "1"
    assert train[train.index("--max-seq-len") + 1] == "2048"
    assert train[train.index("--stream-buffer-size") + 1] == "4096"
    assert train[train.index("--bucket-selection") + 1] == "sample"
    assert train[train.index("--bucket-min-fill-frac") + 1] == "0.0"
    assert train[train.index("--bucket-cycle-repeat") + 1] == "1"
    assert train[train.index("--prefetch-batches") + 1] == "2"
    assert train[train.index("--prefetch-workers") + 1] == "1"
    assert train[train.index("--mfu-warmup-steps") + 1] == "2"
    assert train[train.index("--mfu-warmup-bucket-steps") + 1] == "0"
    assert train[train.index("--log-every") + 1] == "10"
    assert "--skip-bad-images" in train
    assert "--no-save" not in train
    assert "--image-url-template" not in train
    assert "--init-vlm-checkpoint-dir" not in train
    assert "--profile-timing" in train
    assert "--pack-examples" not in train
    assert "--pad-to-max-seq-len" not in train
    assert "--no-selective-loss" not in train

    train_no_processor_prefetch = modal_vlm.build_train_cmd(prefetch_processor=False, batch_buffer_size=128)
    assert "--no-prefetch-processor" in train_no_processor_prefetch
    assert train_no_processor_prefetch[train_no_processor_prefetch.index("--batch-buffer-size") + 1] == "128"
    assert "--no-skip-bad-images" in modal_vlm.build_train_cmd(skip_bad_images=False)
    assert "--compile" in modal_vlm.build_train_cmd(compile_model=True)
    fp8_train = modal_vlm.build_train_cmd(fp8=True, fp8_recipe="tensorwise")
    assert "--fp8" in fp8_train
    assert fp8_train[fp8_train.index("--fp8-recipe") + 1] == "tensorwise"
    siglip_chunked_train = modal_vlm.build_train_cmd(siglip_forward_batch_size=32)
    assert siglip_chunked_train[siglip_chunked_train.index("--siglip-forward-batch-size") + 1] == "32"
    assert "--drop-zero-value-embeds" in modal_vlm.build_train_cmd(drop_zero_value_embeds=True)
    fixed_train = modal_vlm.build_train_cmd(max_seq_len=128, pad_to_max_seq_len=True, selective_loss=False)
    assert fixed_train[fixed_train.index("--max-seq-len") + 1] == "128"
    assert "--pad-to-max-seq-len" in fixed_train
    assert "--no-selective-loss" in fixed_train
    chunked_train = modal_vlm.build_train_cmd(selective_loss=False, loss_chunk_size=2048)
    assert chunked_train[chunked_train.index("--loss-chunk-size") + 1] == "2048"
    bucket_train = modal_vlm.build_train_cmd(max_seq_len=512, pad_to_bucket_lens="96,128,256", selective_loss=False)
    assert bucket_train[bucket_train.index("--pad-to-bucket-lens") + 1] == "96,128,256"
    assert "--no-save" in modal_vlm.build_train_cmd(no_save=True)
    packed_train = modal_vlm.build_train_cmd(pack_examples=4, pack_max_seq_len=1024, pack_fixed_rows=16, boundary_aware_pack=True)
    assert packed_train[packed_train.index("--pack-examples") + 1] == "4"
    assert packed_train[packed_train.index("--pack-max-seq-len") + 1] == "1024"
    assert packed_train[packed_train.index("--pack-fixed-rows") + 1] == "16"
    assert "--boundary-aware-pack" in packed_train
    assert "--flatten-packed-batch" in modal_vlm.build_train_cmd(boundary_aware_pack=True, flatten_packed_batch=True)
    leaky_pack_train = modal_vlm.build_train_cmd(pack_examples=2, allow_leaky_pack=True)
    assert "--allow-leaky-pack" in leaky_pack_train

    mfu_probe = modal_vlm.build_mfu_probe_cmd(
        grad_accum_steps=4,
        pack_examples=3,
        pack_max_seq_len=1024,
        pack_fixed_rows=16,
        boundary_aware_pack=True,
        siglip_forward_batch_size=32,
        drop_zero_value_embeds=True,
        max_seq_len=128,
        pad_to_max_seq_len=True,
        selective_loss=False,
    )
    assert mfu_probe[mfu_probe.index("--num-iterations") + 1] == "6"
    assert mfu_probe[mfu_probe.index("--device-batch-size") + 1] == "256"
    assert mfu_probe[mfu_probe.index("--hf-config") + 1] == "vqav2"
    assert mfu_probe[mfu_probe.index("--grad-accum-steps") + 1] == "4"
    assert mfu_probe[mfu_probe.index("--siglip-forward-batch-size") + 1] == "32"
    assert "--drop-zero-value-embeds" in mfu_probe
    assert mfu_probe[mfu_probe.index("--stream-buffer-size") + 1] == "256"
    assert mfu_probe[mfu_probe.index("--batch-buffer-size") + 1] == "512"
    assert mfu_probe[mfu_probe.index("--prefetch-batches") + 1] == "2"
    assert mfu_probe[mfu_probe.index("--prefetch-workers") + 1] == "1"
    assert mfu_probe[mfu_probe.index("--log-every") + 1] == "1"
    assert mfu_probe[mfu_probe.index("--max-seq-len") + 1] == "128"
    assert mfu_probe[mfu_probe.index("--pack-examples") + 1] == "3"
    assert mfu_probe[mfu_probe.index("--pack-max-seq-len") + 1] == "1024"
    assert mfu_probe[mfu_probe.index("--pack-fixed-rows") + 1] == "16"
    assert "--boundary-aware-pack" in mfu_probe
    assert "--pad-to-max-seq-len" in mfu_probe
    assert "--no-selective-loss" in mfu_probe
    assert "--profile-timing" in mfu_probe
    assert "--no-save" in mfu_probe
    assert "--profile-timing" not in modal_vlm.build_mfu_probe_cmd(profile_timing=False)
    fp8_mfu_probe = modal_vlm.build_mfu_probe_cmd(fp8=True)
    assert "--fp8" in fp8_mfu_probe
    assert fp8_mfu_probe[fp8_mfu_probe.index("--fp8-recipe") + 1] == "tensorwise"

    bucket_probe = modal_vlm.build_mfu_probe_cmd(
        prefetch_batches=8,
        prefetch_workers=4,
        pad_to_bucket_lens="96,128",
        bucket_selection="cycle",
        bucket_min_fill_frac=0.75,
    )
    assert bucket_probe[bucket_probe.index("--prefetch-batches") + 1] == "8"
    assert bucket_probe[bucket_probe.index("--prefetch-workers") + 1] == "4"
    assert bucket_probe[bucket_probe.index("--bucket-selection") + 1] == "cycle"
    assert bucket_probe[bucket_probe.index("--bucket-min-fill-frac") + 1] == "0.75"
    assert bucket_probe[bucket_probe.index("--bucket-cycle-repeat") + 1] == "1"
    assert bucket_probe[bucket_probe.index("--pad-to-bucket-lens") + 1] == "96,128"
    bucket_warmup_probe = modal_vlm.build_mfu_probe_cmd(pad_to_bucket_lens="96,128", mfu_warmup_bucket_steps=1)
    assert bucket_warmup_probe[bucket_warmup_probe.index("--mfu-warmup-bucket-steps") + 1] == "1"

    bucketed_probe = modal_vlm.build_bucketed_mfu_probe_cmd()
    assert bucketed_probe[bucketed_probe.index("--num-iterations") + 1] == "14"
    assert bucketed_probe[bucketed_probe.index("--device-batch-size") + 1] == "512"
    assert bucketed_probe[bucketed_probe.index("--max-batch-tokens") + 1] == "21504"
    assert bucketed_probe[bucketed_probe.index("--max-seq-len") + 1] == "512"
    assert bucketed_probe[bucketed_probe.index("--batch-buffer-size") + 1] == "4096"
    assert bucketed_probe[bucketed_probe.index("--bucket-selection") + 1] == "cycle"
    assert bucketed_probe[bucketed_probe.index("--bucket-min-fill-frac") + 1] == "0.75"
    assert bucketed_probe[bucketed_probe.index("--bucket-cycle-repeat") + 1] == "1"
    assert bucketed_probe[bucketed_probe.index("--prefetch-batches") + 1] == "8"
    assert bucketed_probe[bucketed_probe.index("--prefetch-workers") + 1] == "4"
    assert bucketed_probe[bucketed_probe.index("--mfu-warmup-bucket-steps") + 1] == "1"
    assert bucketed_probe[bucketed_probe.index("--pad-to-bucket-lens") + 1] == "128,192,256,384,512"
    assert "96,128" not in bucketed_probe
    assert "--compile" in bucketed_probe
    assert "--no-selective-loss" in bucketed_probe
    assert "--profile-timing" not in bucketed_probe
    assert "--profile-timing" in modal_vlm.build_bucketed_mfu_probe_cmd(profile_timing=True)
    bucketed_grad2 = modal_vlm.build_bucketed_mfu_probe_cmd(grad_accum_steps=2)
    assert bucketed_grad2[bucketed_grad2.index("--grad-accum-steps") + 1] == "2"
    assert bucketed_grad2[bucketed_grad2.index("--bucket-cycle-repeat") + 1] == "2"

    packed_probe = modal_vlm.build_packed_mfu_probe_cmd()
    assert packed_probe[packed_probe.index("--num-iterations") + 1] == "10"
    assert packed_probe[packed_probe.index("--device-batch-size") + 1] == "512"
    assert packed_probe[packed_probe.index("--max-batch-tokens") + 1] == "32768"
    assert packed_probe[packed_probe.index("--max-seq-len") + 1] == "1024"
    assert packed_probe[packed_probe.index("--batch-buffer-size") + 1] == "4096"
    assert packed_probe[packed_probe.index("--bucket-selection") + 1] == "max-tokens"
    assert packed_probe[packed_probe.index("--prefetch-batches") + 1] == "8"
    assert packed_probe[packed_probe.index("--prefetch-workers") + 1] == "4"
    assert packed_probe[packed_probe.index("--pack-examples") + 1] == "8"
    assert packed_probe[packed_probe.index("--pack-max-seq-len") + 1] == "1024"
    assert "--boundary-aware-pack" in packed_probe
    assert "--flatten-packed-batch" in packed_probe
    assert "--require-fa3-varlen" in packed_probe
    assert "--no-selective-loss" not in packed_probe
    assert "--profile-timing" not in packed_probe
    assert "--compile" not in packed_probe
    assert "--compile" in modal_vlm.build_packed_mfu_probe_cmd(compile_model=True)
    assert "--profile-timing" in modal_vlm.build_packed_mfu_probe_cmd(profile_timing=True)
    assert "--require-fa3-varlen" not in modal_vlm.build_packed_mfu_probe_cmd(require_fa3_varlen=False)
    assert "--flatten-packed-batch" not in modal_vlm.build_packed_mfu_probe_cmd(flatten_packed_batch=False)
    packed_fp8 = modal_vlm.build_packed_mfu_probe_cmd(fp8=True, fp8_recipe="tensorwise")
    assert "--fp8" in packed_fp8
    assert packed_fp8[packed_fp8.index("--fp8-recipe") + 1] == "tensorwise"
    packed_random = modal_vlm.build_packed_mfu_probe_cmd(bucket_selection="random")
    assert packed_random[packed_random.index("--bucket-selection") + 1] == "random"
    packed_random_named = modal_vlm.build_packed_random_mfu_probe_cmd()
    assert packed_random_named[packed_random_named.index("--bucket-selection") + 1] == "random"
    assert packed_random_named[packed_random_named.index("--out-dir") + 1].endswith("packed_random_mfu_probe")
    packed_random_named_override = modal_vlm.build_packed_random_mfu_probe_cmd(bucket_selection="max-tokens")
    assert packed_random_named_override[packed_random_named_override.index("--bucket-selection") + 1] == "random"
    packed_large = modal_vlm.build_packed_large_mfu_probe_cmd()
    assert packed_large[packed_large.index("--device-batch-size") + 1] == "1024"
    assert packed_large[packed_large.index("--max-batch-tokens") + 1] == "65536"
    assert packed_large[packed_large.index("--max-seq-len") + 1] == "1024"
    assert packed_large[packed_large.index("--batch-buffer-size") + 1] == "8192"
    assert packed_large[packed_large.index("--pack-examples") + 1] == "16"
    assert packed_large[packed_large.index("--pack-max-seq-len") + 1] == "1024"
    assert packed_large[packed_large.index("--out-dir") + 1].endswith("packed_large_mfu_probe")
    assert "--boundary-aware-pack" in packed_large
    assert "--flatten-packed-batch" in packed_large
    assert "--require-fa3-varlen" in packed_large
    leaky_large = modal_vlm.build_leaky_packed_large_mfu_probe_cmd()
    assert leaky_large[leaky_large.index("--device-batch-size") + 1] == "1024"
    assert leaky_large[leaky_large.index("--max-batch-tokens") + 1] == "65536"
    assert leaky_large[leaky_large.index("--max-seq-len") + 1] == "1024"
    assert leaky_large[leaky_large.index("--batch-buffer-size") + 1] == "8192"
    assert leaky_large[leaky_large.index("--pack-examples") + 1] == "16"
    assert leaky_large[leaky_large.index("--pack-max-seq-len") + 1] == "1024"
    assert leaky_large[leaky_large.index("--pad-to-bucket-lens") + 1] == "1024"
    assert leaky_large[leaky_large.index("--out-dir") + 1].endswith("leaky_packed_large_mfu_probe")
    assert "--allow-leaky-pack" in leaky_large
    assert "--boundary-aware-pack" not in leaky_large
    assert "--flatten-packed-batch" not in leaky_large
    assert "--require-fa3-varlen" not in leaky_large
    packed_large_random = modal_vlm.build_packed_large_random_mfu_probe_cmd(bucket_selection="max-tokens")
    assert packed_large_random[packed_large_random.index("--bucket-selection") + 1] == "random"
    assert packed_large_random[packed_large_random.index("--device-batch-size") + 1] == "1024"
    assert packed_large_random[packed_large_random.index("--max-batch-tokens") + 1] == "65536"
    assert packed_large_random[packed_large_random.index("--max-seq-len") + 1] == "1024"
    assert packed_large_random[packed_large_random.index("--batch-buffer-size") + 1] == "8192"
    assert packed_large_random[packed_large_random.index("--pack-examples") + 1] == "16"
    assert packed_large_random[packed_large_random.index("--pack-max-seq-len") + 1] == "1024"
    assert packed_large_random[packed_large_random.index("--out-dir") + 1].endswith("packed_large_random_mfu_probe")
    assert "--boundary-aware-pack" in packed_large_random
    assert "--flatten-packed-batch" in packed_large_random
    assert "--require-fa3-varlen" in packed_large_random
    packed_large_compute = modal_vlm.build_packed_large_mfu_probe_cmd(bucket_selection="max-compute")
    assert packed_large_compute[packed_large_compute.index("--bucket-selection") + 1] == "max-compute"
    assert packed_large_compute[packed_large_compute.index("--max-batch-tokens") + 1] == "65536"
    packed_large_compute_named = modal_vlm.build_packed_large_compute_mfu_probe_cmd(bucket_selection="max-tokens")
    assert packed_large_compute_named[packed_large_compute_named.index("--bucket-selection") + 1] == "max-compute"
    assert packed_large_compute_named[packed_large_compute_named.index("--device-batch-size") + 1] == "1024"
    assert packed_large_compute_named[packed_large_compute_named.index("--max-batch-tokens") + 1] == "65536"
    assert packed_large_compute_named[packed_large_compute_named.index("--pack-examples") + 1] == "16"
    assert packed_large_compute_named[packed_large_compute_named.index("--out-dir") + 1].endswith("packed_large_compute_mfu_probe")
    assert "--boundary-aware-pack" in packed_large_compute_named
    assert "--flatten-packed-batch" in packed_large_compute_named
    assert "--require-fa3-varlen" in packed_large_compute_named
    packed_profile = modal_vlm.build_packed_profile_mfu_probe_cmd(profile_timing=False)
    assert packed_profile[packed_profile.index("--out-dir") + 1].endswith("packed_profile_mfu_probe")
    assert "--profile-timing" in packed_profile
    assert "--boundary-aware-pack" in packed_profile
    assert "--flatten-packed-batch" in packed_profile
    assert "--require-fa3-varlen" in packed_profile
    packed_large_profile = modal_vlm.build_packed_large_profile_mfu_probe_cmd(profile_timing=False)
    assert packed_large_profile[packed_large_profile.index("--out-dir") + 1].endswith("packed_large_profile_mfu_probe")
    assert packed_large_profile[packed_large_profile.index("--device-batch-size") + 1] == "1024"
    assert packed_large_profile[packed_large_profile.index("--max-batch-tokens") + 1] == "65536"
    assert packed_large_profile[packed_large_profile.index("--max-seq-len") + 1] == "1024"
    assert packed_large_profile[packed_large_profile.index("--batch-buffer-size") + 1] == "8192"
    assert packed_large_profile[packed_large_profile.index("--pack-examples") + 1] == "16"
    assert packed_large_profile[packed_large_profile.index("--pack-max-seq-len") + 1] == "1024"
    assert "--profile-timing" in packed_large_profile
    assert "--boundary-aware-pack" in packed_large_profile
    assert "--flatten-packed-batch" in packed_large_profile
    assert "--require-fa3-varlen" in packed_large_profile
    packed_large_random_profile = modal_vlm.build_packed_large_random_profile_mfu_probe_cmd(
        bucket_selection="max-tokens",
        profile_timing=False,
    )
    assert packed_large_random_profile[packed_large_random_profile.index("--out-dir") + 1].endswith(
        "packed_large_random_profile_mfu_probe"
    )
    assert packed_large_random_profile[packed_large_random_profile.index("--bucket-selection") + 1] == "random"
    assert packed_large_random_profile[packed_large_random_profile.index("--device-batch-size") + 1] == "1024"
    assert packed_large_random_profile[packed_large_random_profile.index("--max-batch-tokens") + 1] == "65536"
    assert packed_large_random_profile[packed_large_random_profile.index("--max-seq-len") + 1] == "1024"
    assert packed_large_random_profile[packed_large_random_profile.index("--batch-buffer-size") + 1] == "8192"
    assert packed_large_random_profile[packed_large_random_profile.index("--pack-examples") + 1] == "16"
    assert packed_large_random_profile[packed_large_random_profile.index("--pack-max-seq-len") + 1] == "1024"
    assert "--profile-timing" in packed_large_random_profile
    assert "--boundary-aware-pack" in packed_large_random_profile
    assert "--flatten-packed-batch" in packed_large_random_profile
    assert "--require-fa3-varlen" in packed_large_random_profile
    packed_large_compute_profile = modal_vlm.build_packed_large_compute_profile_mfu_probe_cmd(
        bucket_selection="max-tokens",
        profile_timing=False,
    )
    assert packed_large_compute_profile[packed_large_compute_profile.index("--out-dir") + 1].endswith(
        "packed_large_compute_profile_mfu_probe"
    )
    assert packed_large_compute_profile[packed_large_compute_profile.index("--bucket-selection") + 1] == "max-compute"
    assert packed_large_compute_profile[packed_large_compute_profile.index("--device-batch-size") + 1] == "1024"
    assert packed_large_compute_profile[packed_large_compute_profile.index("--max-batch-tokens") + 1] == "65536"
    assert "--profile-timing" in packed_large_compute_profile
    assert "--boundary-aware-pack" in packed_large_compute_profile
    assert "--flatten-packed-batch" in packed_large_compute_profile
    assert "--require-fa3-varlen" in packed_large_compute_profile
    packed_plan = modal_vlm.build_packed_batch_plan_cmd()
    assert packed_plan[packed_plan.index("--device-type") + 1] == "cpu"
    assert packed_plan[packed_plan.index("--batch-plan-steps") + 1] == "2"
    assert packed_plan[packed_plan.index("--device-batch-size") + 1] == "512"
    assert packed_plan[packed_plan.index("--max-batch-tokens") + 1] == "32768"
    assert packed_plan[packed_plan.index("--pack-examples") + 1] == "8"
    assert "--boundary-aware-pack" in packed_plan
    assert "--flatten-packed-batch" in packed_plan
    assert "--require-fa3-varlen" not in packed_plan
    packed_random_plan = modal_vlm.build_packed_random_batch_plan_cmd(bucket_selection="max-tokens")
    assert packed_random_plan[packed_random_plan.index("--bucket-selection") + 1] == "random"
    assert packed_random_plan[packed_random_plan.index("--batch-plan-steps") + 1] == "2"
    assert "--boundary-aware-pack" in packed_random_plan
    assert "--flatten-packed-batch" in packed_random_plan
    packed_large_plan = modal_vlm.build_packed_large_batch_plan_cmd()
    assert packed_large_plan[packed_large_plan.index("--device-type") + 1] == "cpu"
    assert packed_large_plan[packed_large_plan.index("--batch-plan-steps") + 1] == "2"
    assert packed_large_plan[packed_large_plan.index("--device-batch-size") + 1] == "1024"
    assert packed_large_plan[packed_large_plan.index("--max-batch-tokens") + 1] == "65536"
    assert packed_large_plan[packed_large_plan.index("--max-seq-len") + 1] == "1024"
    assert packed_large_plan[packed_large_plan.index("--batch-buffer-size") + 1] == "8192"
    assert packed_large_plan[packed_large_plan.index("--pack-examples") + 1] == "16"
    assert packed_large_plan[packed_large_plan.index("--pack-max-seq-len") + 1] == "1024"
    assert "--boundary-aware-pack" in packed_large_plan
    assert "--flatten-packed-batch" in packed_large_plan
    assert "--require-fa3-varlen" not in packed_large_plan
    packed_large_random_plan = modal_vlm.build_packed_large_random_batch_plan_cmd(bucket_selection="max-tokens")
    assert packed_large_random_plan[packed_large_random_plan.index("--device-type") + 1] == "cpu"
    assert packed_large_random_plan[packed_large_random_plan.index("--bucket-selection") + 1] == "random"
    assert packed_large_random_plan[packed_large_random_plan.index("--device-batch-size") + 1] == "1024"
    assert packed_large_random_plan[packed_large_random_plan.index("--max-batch-tokens") + 1] == "65536"
    assert packed_large_random_plan[packed_large_random_plan.index("--max-seq-len") + 1] == "1024"
    assert packed_large_random_plan[packed_large_random_plan.index("--batch-buffer-size") + 1] == "8192"
    assert packed_large_random_plan[packed_large_random_plan.index("--pack-examples") + 1] == "16"
    assert packed_large_random_plan[packed_large_random_plan.index("--pack-max-seq-len") + 1] == "1024"
    assert "--boundary-aware-pack" in packed_large_random_plan
    assert "--flatten-packed-batch" in packed_large_random_plan
    assert "--require-fa3-varlen" not in packed_large_random_plan
    packed_large_compute_plan = modal_vlm.build_packed_large_batch_plan_cmd(bucket_selection="max-compute")
    assert packed_large_compute_plan[packed_large_compute_plan.index("--bucket-selection") + 1] == "max-compute"
    assert packed_large_compute_plan[packed_large_compute_plan.index("--max-batch-tokens") + 1] == "65536"
    packed_large_compute_named_plan = modal_vlm.build_packed_large_compute_batch_plan_cmd(bucket_selection="max-tokens")
    assert packed_large_compute_named_plan[packed_large_compute_named_plan.index("--device-type") + 1] == "cpu"
    assert packed_large_compute_named_plan[packed_large_compute_named_plan.index("--bucket-selection") + 1] == "max-compute"
    assert packed_large_compute_named_plan[packed_large_compute_named_plan.index("--device-batch-size") + 1] == "1024"
    assert packed_large_compute_named_plan[packed_large_compute_named_plan.index("--max-batch-tokens") + 1] == "65536"
    assert packed_large_compute_named_plan[packed_large_compute_named_plan.index("--pack-examples") + 1] == "16"
    assert "--boundary-aware-pack" in packed_large_compute_named_plan
    assert "--flatten-packed-batch" in packed_large_compute_named_plan
    leaky_large_plan = modal_vlm.build_leaky_packed_large_batch_plan_cmd()
    assert leaky_large_plan[leaky_large_plan.index("--device-type") + 1] == "cpu"
    assert leaky_large_plan[leaky_large_plan.index("--device-batch-size") + 1] == "1024"
    assert leaky_large_plan[leaky_large_plan.index("--max-batch-tokens") + 1] == "65536"
    assert leaky_large_plan[leaky_large_plan.index("--batch-buffer-size") + 1] == "8192"
    assert leaky_large_plan[leaky_large_plan.index("--pack-examples") + 1] == "16"
    assert leaky_large_plan[leaky_large_plan.index("--pad-to-bucket-lens") + 1] == "1024"
    assert "--allow-leaky-pack" in leaky_large_plan
    assert "--boundary-aware-pack" not in leaky_large_plan
    assert "--flatten-packed-batch" not in leaky_large_plan
    chunked_bucketed_probe = modal_vlm.build_bucketed_mfu_probe_cmd(loss_chunk_size=4096)
    assert chunked_bucketed_probe[chunked_bucketed_probe.index("--loss-chunk-size") + 1] == "4096"
    assert "--loss-chunk-size" not in modal_vlm.build_packed_mfu_probe_cmd()
    assert inspect.signature(modal_vlm.train.get_raw_f()).parameters["fp8"].default is False
    assert inspect.signature(modal_vlm.train.get_raw_f()).parameters["fp8_recipe"].default == "tensorwise"
    assert inspect.signature(modal_vlm.mfu_probe.get_raw_f()).parameters["fp8"].default is False
    assert inspect.signature(modal_vlm.mfu_probe.get_raw_f()).parameters["fp8_recipe"].default == "tensorwise"
    assert inspect.signature(modal_vlm.train.get_raw_f()).parameters["siglip_forward_batch_size"].default == 0
    assert inspect.signature(modal_vlm.mfu_probe.get_raw_f()).parameters["siglip_forward_batch_size"].default == 0
    assert inspect.signature(modal_vlm.train.get_raw_f()).parameters["drop_zero_value_embeds"].default is False
    assert inspect.signature(modal_vlm.mfu_probe.get_raw_f()).parameters["drop_zero_value_embeds"].default is False
    assert inspect.signature(modal_vlm.packed_mfu_probe.get_raw_f()).parameters["fp8"].default is False
    assert inspect.signature(modal_vlm.packed_mfu_probe.get_raw_f()).parameters["fp8_recipe"].default == "tensorwise"
    assert inspect.signature(modal_vlm.train.get_raw_f()).parameters["loss_chunk_size"].default == 0
    assert inspect.signature(modal_vlm.mfu_probe.get_raw_f()).parameters["loss_chunk_size"].default == 0
    assert inspect.signature(modal_vlm.bucketed_mfu_probe.get_raw_f()).parameters["loss_chunk_size"].default == 0
    assert inspect.signature(modal_vlm.packed_mfu_probe.get_raw_f()).parameters["loss_chunk_size"].default == 0
    assert inspect.signature(modal_vlm.train.get_raw_f()).parameters["flatten_packed_batch"].default is False
    assert inspect.signature(modal_vlm.mfu_probe.get_raw_f()).parameters["flatten_packed_batch"].default is False
    assert inspect.signature(modal_vlm.train.get_raw_f()).parameters["allow_leaky_pack"].default is False
    assert inspect.signature(modal_vlm.mfu_probe.get_raw_f()).parameters["allow_leaky_pack"].default is False
    assert inspect.signature(modal_vlm.packed_mfu_probe.get_raw_f()).parameters["flatten_packed_batch"].default is True
    assert inspect.signature(modal_vlm.packed_random_mfu_probe.get_raw_f()).parameters["flatten_packed_batch"].default is True
    assert inspect.signature(modal_vlm.packed_large_mfu_probe.get_raw_f()).parameters["flatten_packed_batch"].default is True
    assert inspect.signature(modal_vlm.packed_large_random_mfu_probe.get_raw_f()).parameters["flatten_packed_batch"].default is True
    assert inspect.signature(modal_vlm.packed_large_compute_mfu_probe.get_raw_f()).parameters["flatten_packed_batch"].default is True
    assert inspect.signature(modal_vlm.packed_profile_mfu_probe.get_raw_f()).parameters["flatten_packed_batch"].default is True
    assert inspect.signature(modal_vlm.packed_large_profile_mfu_probe.get_raw_f()).parameters["flatten_packed_batch"].default is True
    assert inspect.signature(modal_vlm.packed_large_random_profile_mfu_probe.get_raw_f()).parameters["flatten_packed_batch"].default is True
    assert inspect.signature(modal_vlm.packed_large_compute_profile_mfu_probe.get_raw_f()).parameters["flatten_packed_batch"].default is True
    assert inspect.signature(modal_vlm.leaky_packed_large_mfu_probe.get_raw_f()).parameters["pack_examples"].default == 16
    assert inspect.signature(modal_vlm.packed_batch_plan.get_raw_f()).parameters["batch_plan_steps"].default == 2
    assert inspect.signature(modal_vlm.packed_random_batch_plan.get_raw_f()).parameters["batch_plan_steps"].default == 2
    assert inspect.signature(modal_vlm.packed_large_batch_plan.get_raw_f()).parameters["batch_plan_steps"].default == 2
    assert inspect.signature(modal_vlm.packed_large_random_batch_plan.get_raw_f()).parameters["batch_plan_steps"].default == 2
    assert inspect.signature(modal_vlm.packed_large_compute_batch_plan.get_raw_f()).parameters["batch_plan_steps"].default == 2
    assert inspect.signature(modal_vlm.leaky_packed_large_batch_plan.get_raw_f()).parameters["batch_plan_steps"].default == 2
    backend_cmd = modal_vlm.build_attention_backend_cmd()
    assert backend_cmd == [
        "python",
        "-m",
        "scripts.vlm_train",
        "--device-type",
        "cuda",
        "--attention-backend-report",
        "--boundary-aware-pack",
        "--require-fa3-varlen",
    ]
    assert "--require-fa3-varlen" not in modal_vlm.build_attention_backend_cmd(require_fa3_varlen=False)
    doctor = modal_vlm.build_doctor_summary()
    assert "bucketed_mfu_probe_preview" in doctor
    assert "packed_mfu_probe_preview" in doctor
    assert "packed_random_mfu_probe_preview" in doctor
    assert "packed_large_mfu_probe_preview" in doctor
    assert "packed_large_random_mfu_probe_preview" in doctor
    assert "packed_large_compute_mfu_probe_preview" in doctor
    assert "packed_profile_mfu_probe_preview" in doctor
    assert "packed_large_profile_mfu_probe_preview" in doctor
    assert "packed_large_random_profile_mfu_probe_preview" in doctor
    assert "packed_large_compute_profile_mfu_probe_preview" in doctor
    assert "leaky_packed_large_mfu_probe_preview" in doctor
    assert "packed_batch_plan_preview" in doctor
    assert "packed_random_batch_plan_preview" in doctor
    assert "packed_large_batch_plan_preview" in doctor
    assert "packed_large_random_batch_plan_preview" in doctor
    assert "packed_large_compute_batch_plan_preview" in doctor
    assert "leaky_packed_large_batch_plan_preview" in doctor
    assert doctor["packed_random_mfu_probe_preview"][doctor["packed_random_mfu_probe_preview"].index("--bucket-selection") + 1] == "random"
    assert doctor["packed_large_mfu_probe_preview"][doctor["packed_large_mfu_probe_preview"].index("--max-batch-tokens") + 1] == "65536"
    assert doctor["packed_large_random_mfu_probe_preview"][doctor["packed_large_random_mfu_probe_preview"].index("--bucket-selection") + 1] == "random"
    assert doctor["packed_large_compute_mfu_probe_preview"][doctor["packed_large_compute_mfu_probe_preview"].index("--bucket-selection") + 1] == "max-compute"
    assert "--profile-timing" in doctor["packed_profile_mfu_probe_preview"]
    assert "--profile-timing" in doctor["packed_large_profile_mfu_probe_preview"]
    assert doctor["packed_large_profile_mfu_probe_preview"][doctor["packed_large_profile_mfu_probe_preview"].index("--max-batch-tokens") + 1] == "65536"
    assert "--profile-timing" in doctor["packed_large_random_profile_mfu_probe_preview"]
    assert doctor["packed_large_random_profile_mfu_probe_preview"][doctor["packed_large_random_profile_mfu_probe_preview"].index("--bucket-selection") + 1] == "random"
    assert doctor["packed_large_random_profile_mfu_probe_preview"][doctor["packed_large_random_profile_mfu_probe_preview"].index("--max-batch-tokens") + 1] == "65536"
    assert "--profile-timing" in doctor["packed_large_compute_profile_mfu_probe_preview"]
    assert doctor["packed_large_compute_profile_mfu_probe_preview"][doctor["packed_large_compute_profile_mfu_probe_preview"].index("--bucket-selection") + 1] == "max-compute"
    assert "--allow-leaky-pack" in doctor["leaky_packed_large_mfu_probe_preview"]
    assert "--boundary-aware-pack" not in doctor["leaky_packed_large_mfu_probe_preview"]
    assert doctor["packed_batch_plan_preview"][doctor["packed_batch_plan_preview"].index("--batch-plan-steps") + 1] == "2"
    assert doctor["packed_random_batch_plan_preview"][doctor["packed_random_batch_plan_preview"].index("--bucket-selection") + 1] == "random"
    assert doctor["packed_large_batch_plan_preview"][doctor["packed_large_batch_plan_preview"].index("--max-batch-tokens") + 1] == "65536"
    assert doctor["packed_large_random_batch_plan_preview"][doctor["packed_large_random_batch_plan_preview"].index("--bucket-selection") + 1] == "random"
    assert doctor["packed_large_compute_batch_plan_preview"][doctor["packed_large_compute_batch_plan_preview"].index("--bucket-selection") + 1] == "max-compute"
    assert "--allow-leaky-pack" in doctor["leaky_packed_large_batch_plan_preview"]
    assert "attention_backend_preview" in doctor

    train_from_checkpoint = modal_vlm.build_train_cmd(init_checkpoint_dir="/checkpoint", init_checkpoint_step=250)
    assert train_from_checkpoint[train_from_checkpoint.index("--init-vlm-checkpoint-dir") + 1] == "/checkpoint"
    assert train_from_checkpoint[train_from_checkpoint.index("--init-vlm-checkpoint-step") + 1] == "250"

    eval_cmd = modal_vlm.build_eval_cmd(limit=3, max_scan=9, benchmarks="mmstar,chartqa", print_samples=2)
    assert eval_cmd[:3] == ["python", "-m", "scripts.vlm_eval"]
    assert eval_cmd[eval_cmd.index("--benchmarks") + 1] == "mmstar,chartqa"


def test_attention_backend_report_exits_before_model_setup():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.vlm_train",
            "--device-type",
            "cpu",
            "--attention-backend-report",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "Attention backend:" in result.stdout
    assert "Loading SigLIP" not in result.stdout
    assert "FA3 varlen attention is required" not in result.stderr


def test_attention_backend_report_require_fa3_errors_cleanly_on_cpu():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.vlm_train",
            "--device-type",
            "cpu",
            "--attention-backend-report",
            "--boundary-aware-pack",
            "--require-fa3-varlen",
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert "FA3 varlen attention is required but unavailable" in result.stderr
    assert "Traceback" not in result.stderr


def test_pack_examples_requires_boundary_aware_or_explicit_leaky_opt_in():
    blocked = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.vlm_train",
            "--device-type",
            "cpu",
            "--attention-backend-report",
            "--pack-examples",
            "2",
        ],
        text=True,
        capture_output=True,
    )
    assert blocked.returncode == 2
    assert "--pack-examples > 1 requires --boundary-aware-pack" in blocked.stderr
    assert "Traceback" not in blocked.stderr

    allowed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.vlm_train",
            "--device-type",
            "cpu",
            "--attention-backend-report",
            "--pack-examples",
            "2",
            "--allow-leaky-pack",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "Attention backend:" in allowed.stdout
    assert "Loading SigLIP" not in allowed.stdout
