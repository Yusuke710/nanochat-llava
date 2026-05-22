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
    encode_with_image_markers,
    ensure_hf_nanochat_checkpoint,
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
    exact_or_choice_match,
    get_answers,
    make_prompt,
    make_result_sample,
    parse_inline_options,
    visual_control_passes,
)
from scripts.vlm_train import (
    batch_features_and_examples,
    get_lr_multiplier,
    load_records,
    next_batch,
    open_image,
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
    assert torch.isfinite(model(batch.value_token_ids, batch.targets, input_embeds=batch.input_embeds))

    with pytest.raises(AssertionError, match="exactly one"):
        build_multimodal_batch(model, projector, [[1, 10, 11]], features, loss_mask_rows=[[1, 1, 1]], value_fallback_token_id=1)


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
        stage=2,
        data_json=None,
        hf_repo="repo",
        hf_file="file.json",
        stream_hf_data=True,
        max_examples=10,
        image_root="/images",
        image_zip=None,
        hf_image_zip=None,
        image_url_template="http://x/{basename}",
        skip_bad_images=True,
        siglip_model_id="google/siglip-base-patch16-512",
        init_vlm_checkpoint_dir="/tmp/stage1",
        init_vlm_checkpoint_step=1,
    )
    model_meta = {"model_config": {"n_embd": 32}}
    save_training_checkpoint(tmp_path, 3, model, projector, args, model_meta, "stream:repo/file.json", rank=0)
    _, _, _, meta = load_vlm_checkpoint(tmp_path, 3, torch.device("cpu"))
    assert meta["vision_config"]["vision_tokens"] == VISION_TOKENS
    assert meta["vision_config"]["projector_vision_dim"] == projector.vision_dim
    assert meta["data_config"]["image_download_timeout"] == vlm_train.IMAGE_DOWNLOAD_TIMEOUT


def test_training_rendering_filters_bad_rows_and_counts_targets():
    tokenizer = TinyTokenizer()
    records = [{"image": "tiny.jpg", "blip_caption": "caption"}]
    rendered = render_records(records, tokenizer, stage=1, max_seq_len=256)
    assert len(rendered) == 1
    assert supervised_target_count(rendered[0]["tokens"], rendered[0]["mask"]) > 0

    direct_image_answer_tokens = [1, IMAGE_TOKEN_ID, 65]
    assert supervised_target_count(direct_image_answer_tokens, [0, 0, 1]) == 0
    with pytest.raises(AssertionError, match="no usable"):
        render_records([{"image": "tiny.jpg", "caption": "x" * 300}], tokenizer, stage=1, max_seq_len=128)


def test_next_batch_respects_padded_token_budget():
    examples = [
        {"expanded_len": 100, "tokens": [1, IMAGE_TOKEN_ID, 2], "mask": [0, 0, 1], "record": {"image": "a"}},
        {"expanded_len": 80, "tokens": [1, IMAGE_TOKEN_ID, 3], "mask": [0, 0, 1], "record": {"image": "b"}},
        {"expanded_len": 20, "tokens": [1, IMAGE_TOKEN_ID, 4], "mask": [0, 0, 1], "record": {"image": "c"}},
    ]
    batch, cursor = next_batch(examples, batch_size=3, cursor=0, rng=__import__("random").Random(0), max_batch_tokens=180)
    assert len(batch) <= 2
    assert cursor > 0


def test_load_records_streams_hf_json(monkeypatch):
    streamed = [{"image": f"{i}.jpg", "caption": f"caption {i}"} for i in range(5)]

    def fake_load_dataset(*args, **kwargs):
        assert kwargs["streaming"] is True
        return iter(streamed)

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=fake_load_dataset))
    args = SimpleNamespace(data_json=None, hf_repo="repo", hf_file="file.json", stage=1, stream_hf_data=True, max_examples=3, device_batch_size=2, grad_accum_steps=1, num_iterations=1)
    records, source = load_records(args)
    assert len(records) == 3
    assert source == "stream:repo/file.json first 3 rows"


def test_image_zip_and_on_demand_loading(tmp_path, monkeypatch):
    img = Image.new("RGB", (4, 4), color=(1, 2, 3))
    img_path = tmp_path / "tiny.jpg"
    img.save(img_path)
    zip_path = tmp_path / "images.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(img_path, "images/tiny.jpg")
    loaded = open_image({"image": "tiny.jpg"}, tmp_path / "missing", image_zip=zip_path)
    assert loaded.size == (4, 4)

    source = tmp_path / "source.jpg"
    Image.new("RGB", (2, 2), color=(9, 8, 7)).save(source)

    def fake_urlretrieve(url, filename):
        Path = __import__("pathlib").Path
        Path(filename).write_bytes(source.read_bytes())

    monkeypatch.setattr(vlm_train.urllib.request, "urlretrieve", fake_urlretrieve)
    fetched = open_image({"image": "remote/a.jpg", "url": "http://example/a.jpg"}, tmp_path / "cache")
    assert fetched.size == (2, 2)
    assert (tmp_path / "cache" / "remote" / "a.jpg").exists()


def test_batch_features_can_skip_dead_images(tmp_path):
    Image.new("RGB", (4, 4), color=(1, 2, 3)).save(tmp_path / "ok.jpg")
    examples = [{"record": {"image": "ok.jpg"}}, {"record": {"image": "missing.jpg"}}]

    class Extractor:
        def __call__(self, images):
            return torch.stack([torch.full((VISION_TOKENS, 2), float(sum(image.getpixel((0, 0))))) for image in images])

    feats, kept = batch_features_and_examples(Extractor(), examples, tmp_path, skip_bad_images=True)
    assert len(kept) == 1
    assert feats.shape == (1, VISION_TOKENS, 2)


def test_eval_prompt_matching_and_samples():
    record = {"question": "What?", "options": ["red", "blue"], "answer": 1}
    prompt = make_prompt(record)
    assert IMAGE_MARKER in prompt
    assert get_answers(record) == ["B", "blue"]
    assert exact_or_choice_match("Answer: B", ["B"])
    assert not exact_or_choice_match("a chart", ["A"])
    assert coerce_options("['cat', 'dog']") == ["cat", "dog"]
    assert parse_inline_options("Options: A: cat, B: dog") == ["cat", "dog"]
    assert visual_control_passes(0.4, 0.3, margin=0.05)
    assert benchmark_specs(["mmmu"], mmmu_configs="Accounting,Basic_Medical_Science")[1]["key"] == "mmmu_Basic_Medical_Science"

    sample = make_result_sample(record, 3, "A", ["A"], True, control_pred="B", control_is_correct=False)
    assert sample["prediction_correct"] is True
    assert sample["zero_image_correct"] is False
    assert sample["prediction_changed"] is True


def test_lr_schedule_and_modal_command_builders():
    assert [round(get_lr_multiplier(i, 5, 0.0, 0.5, 0.0), 2) for i in range(1, 6)] == [1.0, 1.0, 1.0, 0.5, 0.0]

    import modal_vlm

    stage1 = modal_vlm.build_stage1_cmd(max_examples=8)
    assert stage1[:3] == ["python", "-m", "scripts.vlm_train"]
    assert stage1[stage1.index("--stage") + 1] == "1"
    assert stage1[stage1.index("--hf-file") + 1] == "blip_laion_cc_sbu_558k_meta.json"
    assert "--hf-image-zip" not in stage1
    assert "--skip-bad-images" in stage1
    assert "--feature-cache-dir" not in stage1

    stage2 = modal_vlm.build_stage2_cmd(init_checkpoint_step=250, max_examples=4, profile_timing=True)
    assert stage2[stage2.index("--stage") + 1] == "2"
    assert stage2[stage2.index("--hf-file") + 1] == "llava_instruct_150k.json"
    assert stage2[stage2.index("--image-url-template") + 1] == modal_vlm.COCO_TRAIN2017_IMAGE_URL
    assert "--profile-timing" in stage2
    assert "--pack-examples" not in stage2

    eval_cmd = modal_vlm.build_eval_cmd(limit=3, max_scan=9, benchmarks="mmstar,chartqa", print_samples=2)
    assert eval_cmd[:3] == ["python", "-m", "scripts.vlm_eval"]
    assert eval_cmd[eval_cmd.index("--benchmarks") + 1] == "mmstar,chartqa"
    assert "--control" in eval_cmd
