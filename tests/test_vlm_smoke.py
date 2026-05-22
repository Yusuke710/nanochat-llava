import os
import time

import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.vision import IMAGE_TOKEN_ID, VISION_TOKENS, VisionProjector, build_multimodal_batch


def tiny_vlm(device):
    config = GPTConfig(sequence_len=96, vocab_size=128, n_layer=2, n_head=2, n_kv_head=2, n_embd=32, window_pattern="L")
    model = GPT(config).to(device=device)
    model.init_weights()
    projector = VisionProjector(vision_dim=16, n_embd=config.n_embd).to(device=device)
    return model, projector


def synthetic_batch(device):
    rows = [
        [1, IMAGE_TOKEN_ID, 30, 65],
        [1, IMAGE_TOKEN_ID, 30, 66],
    ]
    masks = [
        [0, 0, 0, 1],
        [0, 0, 0, 1],
    ]
    feats = torch.zeros(2, VISION_TOKENS, 16, device=device)
    feats[0, :, 0] = 2.0
    feats[1, :, 1] = 2.0
    return rows, masks, feats


def vlm_loss(model, projector, rows, masks, feats):
    batch = build_multimodal_batch(model, projector, rows, feats, loss_mask_rows=masks, value_fallback_token_id=1)
    return model(batch.value_token_ids, batch.targets, input_embeds=batch.input_embeds)


def test_tiny_vlm_learns_image_conditioned_answers():
    device_name = os.environ.get("VLM_SMOKE_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    torch.manual_seed(42)
    model, projector = tiny_vlm(device)
    rows, masks, feats = synthetic_batch(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(projector.parameters()), lr=3e-3, weight_decay=0.0)

    t0 = time.time()
    initial = float(vlm_loss(model, projector, rows, masks, feats).detach())
    for _ in range(80):
        loss = vlm_loss(model, projector, rows, masks, feats)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    samples_per_sec = 80 * len(rows) / max(time.time() - t0, 1e-9)

    model.eval()
    projector.eval()
    with torch.no_grad():
        aligned = float(vlm_loss(model, projector, rows, masks, feats))
        shuffled = float(vlm_loss(model, projector, rows, masks, feats.flip(0)))
        no_image = float(vlm_loss(model, projector, rows, masks, torch.zeros_like(feats)))

    print(
        f"vlm_smoke | device {device_name} | loss {initial:.4f}->{aligned:.4f} | "
        f"shuffled {shuffled:.4f} | no_image {no_image:.4f} | samples/sec {samples_per_sec:.1f}"
    )
    assert aligned < initial
    assert aligned + 0.05 < shuffled
    assert aligned + 0.05 < no_image
