# BenchPress Vision Sources

This file records the official technical reports, official model cards, and official project pages used to build `scores_raw.csv`.

Source priority follows `benchpress_vision.md`: vendor/project technical reports and model cards first, comparison rows from official reports second, and no third-party leaderboard rows in this v0 matrix.

Important normalization notes:

- Percent-like accuracy and score values are kept on a 0-100 scale.
- OCRBench values reported as points out of 1000 are divided by 1000 and scaled to 0-100.
- MME sum values are divided by 2800 and scaled to 0-100.
- LLaVA's `1519/332` style MME entries are summed before MME normalization.
- The raw CSV preserves each report's split or prompt in `benchmark_variant`; the matrix groups by canonical benchmark name because many reports mix `test`, `val`, and toolkit variants for the same task.

## Source Table

| Source | Type | Rows | Benchmarks | Notes |
|---|---:|---:|---|---|
| [InternVL2.5 official technical report blog](https://internvl.github.io/blog/2024-12-05-InternVL-2.5/) | official_project_report | 228 | AI2D, ChartQA, DocVQA, HallusionBench, InfoVQA, MMBench, MMBench-CN, MME, MMMU, MMStar, MMVet, MathVista, OCRBench, POPE, RealWorldQA, TextVQA | Primary for InternVL2.5 rows. Non-InternVL comparison rows are marked as comparison evidence. |
| [InternVL2.5-MPO official technical report blog](https://internvl.github.io/blog/2024-12-20-InternVL-2.5-MPO/) | official_project_report | 56 | AI2D, HallusionBench, MMBench, MMMU, MMStar, MMVet, MathVista, OCRBench | Primary for InternVL2.5-MPO rows on the OpenCompass benchmark subset. |
| [LLaVA official model zoo](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md) | official_model_card | 104 | GQA, LLaVA-Bench-Wild, MMBench, MMBench-CN, MME, MMMU, MMVet, MathVista, POPE, SEED-Image, ScienceQA, TextVQA, VQAv2, VizWiz | Primary for LLaVA v1.5 and v1.6 model rows. |
| [SmolVLM-500M-Instruct official model card](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct/blob/401eccf6f9c3555db6e85d77ee76772fff15c382/README.md) | official_model_card | 27 | AI2D, ChartQA, DocVQA, MMMU, MMStar, MathVista, OCRBench, ScienceQA, TextVQA | Primary for SmolVLM 256M, 500M, and 2.2B small VLM rows. |
| [SmolVLM2-2.2B-Instruct official model card](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct/blob/482adb537c021c86670beed01cd58990d01e72e4/README.md) | official_model_card | 12 | AI2D, ChartQA, DocVQA, MLVU, MMMU, MMStar, MVBench, MathVista, OCRBench, ScienceQA, TextVQA, Video-MME | Primary for SmolVLM2 2.2B image and video rows. |
| [Qwen2.5-VL-7B-Instruct official model card](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/main/README.md) | official_model_card | 43 | ChartQA, DocVQA, HallusionBench, InfoVQA, MMBench, MMMU, MMMU-Pro, MMStar, MMVet, MathVision, MathVista, OCRBench, TextVQA | Primary for Qwen2.5-VL-7B. Competitor rows are marked as comparison evidence. |
| [Gemini 1.5 official technical report](https://storage.googleapis.com/deepmind-media/gemini/gemini_v1_5_report.pdf) | official_vendor_report | 48 | AI2D, BetterChartQA, ChartQA, DUDE, DocVQA, EgoSchema, InfographicVQA, MMMU, MathVista, RealWorldQA, TextVQA, VQAv2 | Primary vendor source for Gemini 1.0 and 1.5 vision benchmark rows. |
| [Gemma 3 official model card](https://ai.google.dev/gemma/docs/core/model_card_3) | official_model_card | 48 | AI2D, ChartQA, DocVQA, InfoVQA, MMMU, MathVista, RealWorldQA, TextVQA, VQAv2 | Primary source for Gemma 3 multimodal rows. |
| [MiniCPM-V official repository](https://github.com/OpenBMB/MiniCPM-V) | official_project_report | 22 | AI2D, DocVQA, HallusionBench, MMBench, MMBench-CN, MMMU, MMStar, MMVet, MathVista, OCRBench, TextVQA | Primary for MiniCPM-o 4.5 and related current MiniCPM rows. |

## Frontier-Model Handling

Gemini rows come from Google's official Gemini report. OpenAI and Anthropic rows are kept only where official VLM-family reports include exact public scores for the target matrix and are marked as comparison evidence, so they calibrate the high end without dominating the recommendation. The selected suites are evaluated primarily on all rows and reported with coverage diagnostics; they are not chosen by closed-model preference.
