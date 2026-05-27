# nanochat-llava Research Design

This file is human-owned. Codex may edit it only when the human explicitly asks.
Use `experiments.md` for agent-written notes, run logs, failed ideas, and research scratch work.

## Goal

Build a minimal LLaVA-style VLM on top of an already-trained nanochat
checkpoint on 1 GPU by adding:

- a frozen or lightly configurable vision encoder
- a small projector from vision features into nanochat token embedding space
- one visual-instruction training path: freeze SigLIP, train the projector plus nanochat

The end state should still feel like nanochat: small code surface, readable files, few knobs, and no framework-like configuration system.

## Hard Constraints

- Keep nanochat's README as the upstream project README.
- Keep upstream sync easy. Prefer additive files and thin hooks over broad edits to existing nanochat code, so future nanochat updates remain compatible with nanochat-llava.
- Preserve text-only pretraining, SFT, RL, evaluation, CLI, and web paths unless vision is explicitly enabled.
- Follow Karpathy-style minimalism: fewer abstractions, fewer options, clear defaults.
- Do not modify nanochat SFT or RL for v0. Start from a trained nanochat checkpoint instead.
- Git history is the progress tracker. This file records intent, not every step.

## V0 Design Decisions

- Start from the official `karpathy/nanochat-d32` checkpoint.
- Use `huggingface/nanoVLM` as a reference implementation because it is proven to work: https://github.com/huggingface/nanoVLM/tree/main. Borrow ideas from it, but keep nanochat-llava simpler and closer to nanochat style.
- Run nanochat benchmarks on that checkpoint to identify whether it behaves like SFT or RL; continue either way because the v0 priority is using Karpathy's original uploaded model.
- Training order is:

```text
karpathy/nanochat-d32 checkpoint
-> VLM train: freeze SigLIP, train projector and LLM on image data
```

- Use `SigLIP base patch-16/512` as the first vision encoder target, about 93M parameters.
- Keep the vision encoder frozen for v0.
- Use simple pooling for visual-token compression.
- Start with an `8x8` pooled grid, i.e. 64 visual tokens per image.
- Use a LLaVA-style literal `<image>` marker in text and replace it with projected visual tokens in the model input stream.
- Start with a linear projector from SigLIP feature dimension to nanochat embedding dimension.
- Do not mix text-only data into VLM training for v0. Keep it image-only until the simple LLaVA path works.
- Mask image-token targets out of the loss. Use assistant-only masking for conversation data.
- Preserve all text-only nanochat paths unless vision is explicitly enabled.

## V0 Data

- Use `HuggingFaceM4/FineVisionMax` as the default visual-instruction data source for v0.
- Treat FineVisionMax as the VLM-data analogue of Karpathy's switch to `karpathy/climbmix-400b-shuffle` / NVIDIA ClimbMix-400B for nanochat text pretraining: the larger cleaned mixed-data source, not a one-off LLaVA-only corpus.
- Keep the first implementation simple: convert FineVisionMax image-text/VQA examples into the LLaVA-style `<image>` conversation format and train only on image-conditioned examples.
- Scale reference: `nanoVLM-222M` reports about 1.7M unique `HuggingFaceM4/the_cauldron` samples, about 6 H100-hours, and 35.3% MMStar. With the older 5-epoch, max-sequence-128 setup, that is roughly 8.5M sample presentations and about 1B multimodal token-equivalents through the decoder.
- Interpret nanoVLM as multimodal alignment / finetuning from pretrained SigLIP and language backbones, not from-scratch capability training.
- References: https://huggingface.co/lusxvr/nanoVLM-222M, https://huggingface.co/blog/nanovlm, https://github.com/huggingface/nanoVLM.

## V0 Benchmarks

Use the benchmark suite selected by the local BenchPress-style benchmark study
in `benchpress_vision/results/recommendation.md`. This is mainly a verifier for
whether the VLM training loop is working.

V0 verifier run:

- MMStar: vision-indispensable multimodal reasoning.
- ScienceQA: science and diagram-style question answering.
- ChartQA: chart and structured visual reasoning.
- MMMU: multidiscipline multimodal reasoning.
- TextVQA: scene-text/OCR question answering.

## Open Decisions

- Whether 64 visual tokens is sufficient; start small and check benchmark performance.
- Whether the linear projector is enough; try an MLP only after the baseline runs.
