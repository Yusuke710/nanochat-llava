import json
import sys
import types
import zipfile
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
    VisionProjector,
    build_multimodal_batch,
    count_image_tokens,
    count_text_image_markers,
    encode_with_image_markers,
    ensure_hf_nanochat_checkpoint,
    format_image_markers,
    load_vlm_checkpoint,
    pool_siglip_features,
    render_caption_example,
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
    build_training_batches,
    choose_fixed_pack,
    collect_rendered_examples,
    compute_vlm_loss,
    evaluate_vlm_loss,
    get_lr_multiplier,
    image_values,
    num_packed_batches,
    open_image,
    _open_hf_record_stream,
    open_images_by_rows,
    pack_fixed_rows,
    render_record,
    render_records,
    save_training_checkpoint,
    split_train_val_examples,
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
    assert count_text_image_markers("Image 1: <image>\nImage 2: <image>") == 2
    assert format_image_markers(1) == IMAGE_MARKER
    assert format_image_markers(2) == f"Image 1: {IMAGE_MARKER}\nImage 2: {IMAGE_MARKER}"

    caption_ids, caption_mask = render_caption_example(tokenizer, "a caption")
    assert count_image_tokens(caption_ids) == 1
    assert sum(caption_mask) > 0

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
    batch = build_multimodal_batch(model, projector, [row], features, loss_mask_rows=[mask], value_fallback_token_id=1)
    assert batch.input_embeds.shape == (1, len(row) - 1 + VISION_TOKENS - 1, model.config.n_embd)
    targets = batch.targets[0]
    assert targets[0].item() == 10
    assert targets[1].item() == -1
    assert torch.all(targets[2 : 2 + VISION_TOKENS] == -1)
    assert targets[-1].item() == 12
    valid = (batch.targets.view(-1) != -1).nonzero(as_tuple=False).flatten()
    torch.testing.assert_close(batch.loss_indices.cpu(), valid.cpu())
    torch.testing.assert_close(batch.loss_targets.cpu(), batch.targets.view(-1).index_select(0, valid).cpu())
    full_loss = model(batch.value_token_ids, batch.targets, input_embeds=batch.input_embeds)
    target_only_loss = model(
        batch.value_token_ids,
        batch.targets,
        input_embeds=batch.input_embeds,
        loss_indices=batch.loss_indices,
        loss_targets=batch.loss_targets,
    )
    assert torch.isfinite(target_only_loss)
    torch.testing.assert_close(target_only_loss, full_loss, rtol=1e-5, atol=1e-5)

    with pytest.raises(AssertionError, match="consumed 0 image features"):
        build_multimodal_batch(model, projector, [[1, 10, 11]], features, loss_mask_rows=[[1, 1, 1]], value_fallback_token_id=1)


def test_boundary_aware_varlen_packed_loss_matches_separate_examples():
    model, projector = tiny_model()
    with torch.no_grad():
        model.smear_lambda.fill_(1.0)
    row1 = [1, 10, IMAGE_TOKEN_ID, 11, 12]
    row2 = [1, 20, IMAGE_TOKEN_ID, 21, 22]
    mask1 = [1] * len(row1)
    mask2 = [1] * len(row2)
    features = torch.randn(2, VISION_TOKENS, 8)

    separate = build_multimodal_batch(
        model,
        projector,
        [row1, row2],
        features,
        loss_mask_rows=[mask1, mask2],
        value_fallback_token_id=1,
    )
    separate_loss = model(
        separate.value_token_ids,
        separate.targets,
        input_embeds=separate.input_embeds,
        loss_reduction="sum",
    )

    packed = build_multimodal_batch(
        model,
        projector,
        [row1 + row2],
        features,
        loss_mask_rows=[mask1 + mask2],
        image_counts_per_row=[2],
        segment_token_lengths_per_row=[[len(row1), len(row2)]],
        max_seq_len=None,
        value_fallback_token_id=1,
    )
    assert packed.cu_seqlens.tolist() == [0, int(separate.lengths[0]), int(separate.lengths.sum())]
    assert packed.segment_starts[0, 0]
    assert packed.segment_starts[0, int(separate.lengths[0])]
    assert int(packed.position_ids[0, int(separate.lengths[0])]) == 0
    packed_loss = model(
        packed.value_token_ids,
        packed.targets,
        input_embeds=packed.input_embeds,
        position_ids=packed.position_ids,
        segment_starts=packed.segment_starts,
        cu_seqlens=packed.cu_seqlens,
        max_seqlen=packed.max_seqlen,
        loss_reduction="sum",
    )
    torch.testing.assert_close(packed_loss, separate_loss, rtol=1e-5, atol=1e-5)


def test_fixed_pack_rows_pad_with_ignored_boundary_segments():
    model, projector = tiny_model()
    examples = [
        {"tokens": [1, IMAGE_TOKEN_ID, 10, 11], "mask": [0, 0, 1, 1], "expanded_len": VISION_TOKENS + 2, "record": {"id": i}}
        for i in range(2)
    ]
    row_examples, selected = choose_fixed_pack(examples, 2, 80)
    assert sorted(selected) == [0, 1]
    features = torch.randn(2, VISION_TOKENS, 8)
    rows, masks, image_counts, segment_lengths = None, None, None, None
    rows, masks, features, image_counts, segment_lengths = pack_fixed_rows(row_examples, features, max_seq_len=80)
    assert len(rows) == 2
    assert image_counts == [1, 1]
    assert all(lengths[-1] == 15 for lengths in segment_lengths)  # 14 dummy shifted positions + 1 target token

    batch = build_multimodal_batch(
        model,
        projector,
        rows,
        features,
        loss_mask_rows=masks,
        image_counts_per_row=image_counts,
        segment_token_lengths_per_row=segment_lengths,
        value_fallback_token_id=1,
    )
    assert batch.value_token_ids.shape == (2, 80)
    assert int(batch.cu_seqlens[-1]) == 160
    assert int(batch.lengths.sum()) == 160
    assert int(batch.loss_targets.numel()) == 2


def test_mixed_text_and_multi_image_rows_share_one_batch():
    model, projector = tiny_model()
    text = {
        "tokens": [1, 10, 11, 12],
        "mask": [0, 1, 1, 1],
        "expanded_len": 3,
        "image_values": [],
        "image_count": 0,
        "record": {"id": "text"},
    }
    multi = {
        "tokens": [1, IMAGE_TOKEN_ID, 20, IMAGE_TOKEN_ID, 21, 22],
        "mask": [0, 0, 1, 0, 1, 1],
        "expanded_len": 5 + 2 * (VISION_TOKENS - 1),
        "image_values": ["a.jpg", "b.jpg"],
        "image_count": 2,
        "record": {"id": "multi"},
    }
    rows, masks, features, image_counts, segment_lengths = pack_fixed_rows(
        [[text, multi], []],
        torch.randn(2, VISION_TOKENS, 8),
        max_seq_len=192,
    )
    assert image_counts == [2, 0]
    assert features.shape == (2, VISION_TOKENS, 8)

    batch = build_multimodal_batch(
        model,
        projector,
        rows,
        features,
        loss_mask_rows=masks,
        image_counts_per_row=image_counts,
        segment_token_lengths_per_row=segment_lengths,
        value_fallback_token_id=1,
    )
    assert batch.value_token_ids.shape == (2, 192)
    assert int(batch.cu_seqlens[-1]) == 384
    assert int(batch.loss_targets.numel()) >= 3


def test_text_only_batch_skips_siglip():
    model, projector = tiny_model()
    packed = {
        "rows": [[1, 10, 11, 12]],
        "masks": [[0, 1, 1, 1]],
        "pixel_values": None,
        "image_counts": [0],
        "segment_lengths": [[4]],
        "num_examples": 1,
    }

    class Extractor:
        vision_dim = 8

        def encode_pixel_values(self, pixel_values):  # pragma: no cover - must not be called
            raise AssertionError("text-only batches should not run SigLIP")

    loss, token_count, target_count, sample_count = compute_vlm_loss(model, projector, Extractor(), packed)
    assert torch.isfinite(loss)
    assert token_count == 3
    assert target_count == 3
    assert sample_count == 1


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


def test_vlm_checkpoint_save_load(tmp_path):
    model, projector = tiny_model()
    save_vlm_checkpoint(tmp_path, 7, model, projector, {"ok": True}, {"stage": 1}, rank=0)
    model_state, loaded_projector, optimizer_data, meta = load_vlm_checkpoint(tmp_path, 7, torch.device("cpu"), load_optimizer=True, rank=0)
    assert meta["stage"] == 1
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


def test_training_checkpoint_metadata_records_vision_config(tmp_path):
    model, projector = tiny_model()
    args = SimpleNamespace(
        data_json=None,
        hf_repo="repo",
        image_root="/images",
        image_zip=None,
        skip_bad_images=True,
        siglip_model_id="google/siglip-base-patch16-512",
    )
    model_meta = {"model_config": {"n_embd": 32}}
    save_training_checkpoint(tmp_path, 3, model, projector, args, model_meta, "stream:repo", rank=0)
    _, _, _, meta = load_vlm_checkpoint(tmp_path, 3, torch.device("cpu"))
    assert meta["vision_config"]["vision_tokens"] == VISION_TOKENS
    assert meta["vision_config"]["projector_vision_dim"] == projector.vision_dim


def test_training_rendering_filters_bad_rows_and_counts_targets():
    tokenizer = TinyTokenizer()
    records = [{"image": "tiny.jpg", "messages": [{"role": "user", "content": "Describe."}, {"role": "assistant", "content": "caption"}]}]
    rendered = render_records(records, tokenizer, max_seq_len=256)
    assert len(rendered) == 1
    assert supervised_target_count(rendered[0]["tokens"], rendered[0]["mask"]) > 0

    direct_image_answer_tokens = [1, IMAGE_TOKEN_ID, 65]
    assert supervised_target_count(direct_image_answer_tokens, [0, 0, 1]) == 0
    with pytest.raises(AssertionError, match="no usable"):
        render_records(
            [{"image": "tiny.jpg", "messages": [{"role": "user", "content": "Describe."}, {"role": "assistant", "content": "x" * 300}]}],
            tokenizer,
            max_seq_len=128,
        )


def test_render_record_supports_text_only_and_multi_image():
    tokenizer = TinyTokenizer()
    text_only = render_record(
        {"messages": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}]},
        tokenizer,
        max_seq_len=128,
    )
    assert text_only is not None
    assert text_only["image_count"] == 0
    assert count_image_tokens(text_only["tokens"]) == 0

    multi = render_record(
        {
            "images": ["a.jpg", "b.jpg"],
            "messages": [{"role": "user", "content": "Compare them."}, {"role": "assistant", "content": "The second is brighter."}],
        },
        tokenizer,
        max_seq_len=256,
    )
    assert multi is not None
    assert multi["image_count"] == 2
    assert image_values(multi["record"]) == ["a.jpg", "b.jpg"]
    assert count_image_tokens(multi["tokens"]) == 2
    normalized = vlm_train._ensure_image_markers_in_conversation(
        {
            "images": ["a.jpg", "b.jpg"],
            "messages": [{"role": "user", "content": "Compare them."}, {"role": "assistant", "content": "The second is brighter."}],
        },
        2,
    )
    assert normalized["messages"][0]["content"].startswith(f"Image 1: {IMAGE_MARKER}\nImage 2: {IMAGE_MARKER}\n")

    explicit_per_turn = render_record(
        {
            "images": ["a.jpg", "b.jpg"],
            "messages": [
                {"role": "user", "content": "Image 1: <image>\nWhat is the first?"},
                {"role": "assistant", "content": "A."},
                {"role": "user", "content": "Image 2: <image>\nWhat is the second?"},
                {"role": "assistant", "content": "B."},
            ],
        },
        tokenizer,
        max_seq_len=512,
    )
    assert explicit_per_turn is not None
    assert explicit_per_turn["image_values"] == ["a.jpg", "b.jpg"]
    assert count_image_tokens(explicit_per_turn["tokens"]) == 2
    marker_positions = [i for i, tok in enumerate(explicit_per_turn["tokens"]) if tok == IMAGE_TOKEN_ID]
    assert marker_positions[1] > marker_positions[0] + 10

    implicit_per_turn = render_record(
        {
            "messages": [
                {"role": "user", "content": "What is the first?", "image": "a.jpg"},
                {"role": "assistant", "content": "A."},
                {"role": "user", "content": "What is the second?", "images": ["b.jpg"]},
                {"role": "assistant", "content": "B."},
            ],
        },
        tokenizer,
        max_seq_len=512,
    )
    assert implicit_per_turn is not None
    assert implicit_per_turn["image_values"] == ["a.jpg", "b.jpg"]
    assert count_image_tokens(implicit_per_turn["tokens"]) == 2
    marker_positions = [i for i, tok in enumerate(implicit_per_turn["tokens"]) if tok == IMAGE_TOKEN_ID]
    assert marker_positions[1] > marker_positions[0] + 10

    implicit_text_pairs = render_record(
        {
            "texts": [
                {"user": "What is the first?", "assistant": "A.", "image": "a.jpg"},
                {"user": "What is the second?", "assistant": "B.", "images": ["b.jpg"]},
            ],
        },
        tokenizer,
        max_seq_len=512,
    )
    assert implicit_text_pairs is not None
    assert implicit_text_pairs["image_values"] == ["a.jpg", "b.jpg"]
    assert count_image_tokens(implicit_text_pairs["tokens"]) == 2

    mismatched = render_record(
        {
            "images": ["a.jpg", "b.jpg"],
            "messages": [{"role": "user", "content": f"{IMAGE_MARKER}\nCompare."}, {"role": "assistant", "content": "A."}],
        },
        tokenizer,
        max_seq_len=256,
    )
    assert mismatched is None


def test_build_training_batches_uses_fixed_shape_best_fit():
    examples = [
        {"expanded_len": 70, "tokens": [1, IMAGE_TOKEN_ID, 2], "mask": [0, 0, 1], "record": {"image": "a"}},
        {"expanded_len": 70, "tokens": [1, IMAGE_TOKEN_ID, 3], "mask": [0, 0, 1], "record": {"image": "b"}},
        {"expanded_len": 20, "tokens": [1, IMAGE_TOKEN_ID, 4], "mask": [0, 0, 1], "record": {"image": "c"}},
        {"expanded_len": 20, "tokens": [1, IMAGE_TOKEN_ID, 5], "mask": [0, 0, 1], "record": {"image": "d"}},
    ]
    batches = build_training_batches(examples, num_batches=1, batch_size=2, max_seq_len=100, max_images=0, seed=0)
    assert len(batches) == 1
    assert len(batches[0]) == 2
    assert all(sum(example["expanded_len"] for example in row) <= 100 for row in batches[0])
    assert sorted(sum(example["expanded_len"] for example in row) for row in batches[0]) == [90, 90]


def test_split_train_val_examples_is_small_and_optional():
    examples = [{"i": i} for i in range(20)]
    train, val = split_train_val_examples(examples, val_examples=8, use_val=True)
    assert len(train) == 18
    assert len(val) == 2
    assert val == examples[-2:]

    train, val = split_train_val_examples(examples, val_examples=8, use_val=False)
    assert train == examples
    assert val == []


def test_num_packed_batches_uses_fixed_shape():
    examples = [{"expanded_len": n} for n in (60, 50, 80, 20, 120)]
    assert num_packed_batches([], batch_size=2, max_seq_len=100) == 0
    assert num_packed_batches(examples, batch_size=2, max_seq_len=100) == 2


def test_evaluate_vlm_loss_restores_train_mode():
    model, projector = tiny_model()
    model.train()
    projector.train()

    class Extractor:
        def encode_pixel_values(self, pixel_values):
            return torch.randn(pixel_values.size(0), VISION_TOKENS, 8)

    packed = {
        "rows": [[1, IMAGE_TOKEN_ID, 10, 11]],
        "masks": [[0, 0, 1, 1]],
        "pixel_values": torch.randn(1, 3, 4, 4),
        "image_counts": [1],
        "segment_lengths": [[4]],
        "num_examples": 1,
    }
    stats = evaluate_vlm_loss(model, projector, Extractor(), [packed])
    assert stats["target_tokens"] > 0
    assert stats["loss"] > 0
    assert model.training
    assert projector.training


def test_hf_record_stream_uses_native_streaming(monkeypatch):
    streamed = [{"image": f"{i}.jpg", "caption": f"caption {i}"} for i in range(5)]

    def fake_load_dataset(*args, **kwargs):
        assert kwargs["streaming"] is True
        assert kwargs["split"] == "train"
        return iter(streamed)

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=fake_load_dataset))
    records = list(_open_hf_record_stream("repo"))
    assert records == streamed


def test_streaming_finevision_schema_renders_validation_slice(monkeypatch):
    streamed = [
        {
            "images": [Image.new("RGB", (4, 4), color=(1, 2, 3))],
            "texts": [{"user": "What color?", "assistant": "Red."}],
        }
        for _ in range(2)
    ]

    def fake_load_dataset(repo, **kwargs):
        assert repo == "repo"
        assert kwargs["streaming"] is True
        assert kwargs["split"] == "train"
        return iter(streamed)

    class HFImage:
        def __init__(self, decode=False):
            self.decode = decode

    class Sequence:
        def __init__(self, feature):
            self.feature = feature

    datasets = types.SimpleNamespace(load_dataset=fake_load_dataset, Image=HFImage, Sequence=Sequence)
    monkeypatch.setitem(sys.modules, "datasets", datasets)
    records = _open_hf_record_stream("repo")
    rendered, seen = collect_rendered_examples(records, TinyTokenizer(), max_seq_len=256, limit=1)
    assert seen == 1
    assert len(rendered) == 1
    assert count_image_tokens(rendered[0]["tokens"]) == 1


def test_image_zip_and_finevision_loading(tmp_path):
    img = Image.new("RGB", (4, 4), color=(1, 2, 3))
    img_path = tmp_path / "tiny.jpg"
    img.save(img_path)
    zip_path = tmp_path / "images.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(img_path, "images/tiny.jpg")
    loaded = open_image({"image": "tiny.jpg"}, tmp_path / "missing", image_zip=zip_path)
    assert loaded.size == (4, 4)

    direct = open_image({"images": [Image.new("RGB", (3, 2), color=(1, 1, 1))]}, tmp_path)
    assert direct.size == (3, 2)

    import io

    encoded = io.BytesIO()
    Image.new("RGB", (2, 2), color=(9, 8, 7)).save(encoded, format="PNG")
    from_bytes = open_image({"images": [{"bytes": encoded.getvalue(), "path": None}]}, tmp_path)
    assert from_bytes.size == (2, 2)


def test_open_images_by_rows_can_skip_dead_images(tmp_path):
    Image.new("RGB", (4, 4), color=(1, 2, 3)).save(tmp_path / "ok.jpg")
    examples = [{"record": {"image": "ok.jpg"}}, {"record": {"image": "missing.jpg"}}]

    images, kept_rows = open_images_by_rows([[examples[0]], [examples[1]]], tmp_path, skip_bad_images=True)
    assert kept_rows == [[examples[0]], []]
    assert len(images) == 1
    assert images[0].size == (4, 4)


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


def test_lr_schedule_and_modal_command_builders():
    assert [round(get_lr_multiplier(i, 5, 0.0, 0.5, 0.0), 2) for i in range(1, 6)] == [1.0, 1.0, 1.0, 0.5, 0.0]

    import modal_vlm

    train_cmd = modal_vlm.build_train_cmd(no_save=True, require_fa3_varlen=True)
    assert train_cmd[:3] == ["python", "-m", "scripts.vlm_train"]
    assert train_cmd[train_cmd.index("--hf-repo") + 1] == "HuggingFaceM4/FineVisionMax"
    assert train_cmd[train_cmd.index("--device-batch-size") + 1] == "32"
    assert train_cmd[train_cmd.index("--max-seq-len") + 1] == "512"
    assert train_cmd[train_cmd.index("--max-batch-images") + 1] == "96"
    assert "--hf-config" not in train_cmd
    assert "--require-fa3-varlen" in train_cmd
    assert "--no-save" in train_cmd
    assert "--max-batch-tokens" not in train_cmd
    assert "--pack-batch-size" not in train_cmd
    assert "--pack-seq-len" not in train_cmd
    assert "--max-" + "examples" not in train_cmd
    assert "--eval-tokens" not in train_cmd
    assert "--eval-steps" not in train_cmd
    assert "--profile-timing" not in train_cmd
    assert "--fp8" not in train_cmd

    eval_cmd = modal_vlm.build_eval_cmd(limit=3, max_scan=9, benchmarks="mmstar,chartqa")
    assert eval_cmd[:3] == ["python", "-m", "scripts.vlm_eval"]
    assert eval_cmd[eval_cmd.index("--benchmarks") + 1] == "mmstar,chartqa"
    assert "--control" not in eval_cmd
