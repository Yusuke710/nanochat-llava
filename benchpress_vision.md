# BenchPress Vision Benchmark Plan

This is a planning artifact for choosing a minimal vision benchmark set for
nanochat-llava. It follows the BenchPress idea from "You Don't Need to Run
Every Eval": build a model-by-benchmark score matrix, test whether it is
low-rank/redundant, and use matrix completion plus greedy selection to decide
which few benchmarks are worth running.

Source method:

- Article: https://dimitrisp.substack.com/p/you-dont-need-to-run-every-eval
- Code/data reference: https://github.com/anadim/llm-benchmark-matrix

## Goal

Find a minimal vision benchmark set for nanochat-llava using evidence, not
prior preference. The current four-benchmark set is only a vibe-based strawman:

```text
MMStar
OCRBench
ChartQA
ScienceQA
```

Do not trust this set. Test it against report-mined alternatives and replace
any or all of it if the matrix says another suite gives better coverage.

The final output should be a recommendation like:

```text
Best k=4: <benchmark>, <benchmark>, <benchmark>, <benchmark>
Best k=5: <benchmark>, <benchmark>, <benchmark>, <benchmark>, <benchmark>
Vibe baseline rank/error: <numbers for MMStar/OCRBench/ChartQA/ScienceQA>
Reason to include each selected benchmark: <non-redundant capability signal>
Expected coverage/predictive error: <numbers>
```

## Principle

Do not choose benchmarks only by vibes or popularity. Choose them by measuring
which benchmarks add non-redundant information about VLM capability.

The experiment should answer:

- Which VLM benchmarks are mostly redundant?
- Which benchmarks are hard to predict from the others?
- Which 4 or 5 benchmarks best predict the rest?
- Does any small set cover general vision reasoning, OCR/document, chart/structured reasoning, hallucination/grounding, and science/diagram-style VQA?

## Candidate Benchmarks

Start with benchmarks that have many public VLM scores and deterministic or
mostly deterministic scoring. Expand this list while reading technical reports:
if an official VLM report repeatedly uses a benchmark, add it as a candidate
column rather than forcing the initial list.

Initial vibe baseline only:

- MMStar
- OCRBench
- ChartQA
- ScienceQA

Core candidate pool from LLaVA, nanoVLM, SmolVLM, Claude 3, Gemini, DeepSeek-VL,
Kimi-VL, and Qwen-VL style reports:

- MMBench
- MMBench-CN
- MME
- MME-RealWorld
- MMMU
- MMMU-Pro
- MMStar
- MMVet
- MathVista
- MathVision
- AI2D
- RealWorldQA
- SEED-Bench
- SEED-Bench-2
- VQAv2
- GQA
- VizWiz
- LLaVA-Bench
- LLaVA-Bench-in-the-Wild
- HallusionBench
- Vibe-Eval
- ZeroBench
- POPE
- TextVQA
- ST-VQA
- OCR-VQA
- OCRBench
- OCRBench v2
- DocVQA
- InfoVQA
- InfographicVQA
- DUDE
- ChartQA
- BetterChartQA
- ChartBench
- ChartX
- CharXiv
- ScienceQA
- RefCOCO
- RefCOCO+
- RefCOCOg
- ScreenSpot
- ScreenSpot-V2
- ScreenSpot-Pro
- OSWorld
- WindowsAgentArena
- Video-MME
- MMBench-Video
- Video-MMMU
- MVBench
- MLVU
- LongVideoBench
- EgoSchema
- TempCompass
- WorldSense
- MMLongBench-Doc
- TOMATO

Prefer deterministic metrics first. Keep GPT-judge benchmarks such as MMVet or
LLaVA-Bench as optional columns, not mandatory columns.

Keep benchmark variants separate unless they are clearly the same benchmark.
Examples: `ChartQA` and `BetterChartQA` are separate; `OCRBench` and
`OCRBench v2` are separate; `MMMU` and `MMMU-Pro` are separate.

## Candidate Models

Collect scores for a broad range of VLMs, including both small/open VLMs and
frontier reference models. nanochat-llava is small, so small/open models should
be prioritized during analysis, but frontier models are important anchors for
the high end of the score matrix.

Small/open models to prioritize:

- nanoVLM variants
- SmolVLM-256M, SmolVLM-500M, SmolVLM/SmolVLM2 larger variants
- Moondream variants
- PaliGemma variants
- LLaVA-1.5 / LLaVA-NeXT variants
- TinyLLaVA variants
- Qwen2-VL / Qwen2.5-VL small variants
- InternVL small variants
- MiniCPM-V variants
- Idefics2 / Idefics3 variants
- Phi vision variants
- Gemma 3 vision variants

Frontier closed and open-weight models to include when official technical
reports or model cards provide relevant scores:

- OpenAI: GPT-4V, GPT-4o, GPT-4.1, GPT-5 family if available.
- Anthropic: Claude 3 Opus/Sonnet/Haiku, Claude 3.5/3.7 Sonnet, Claude 4 family if available.
- Google DeepMind: Gemini 1.5, Gemini 2.0, Gemini 2.5, Gemini 3 family if available.
- Moonshot AI: Kimi-VL, Kimi K2.5, Kimi K2.6 if an official multimodal report or model card is available.
- DeepSeek: DeepSeek-VL, DeepSeek-VL2, Janus-Pro, and later DeepSeek multimodal models if available.
- Alibaba/Qwen: Qwen-VL, Qwen2-VL, Qwen2.5-VL, Qwen3-VL if available.
- Other frontier/open VLM families: InternVL, MiniCPM-V, Idefics, Llama Vision, Molmo, Pixtral, GLM/GLM-V, Phi Vision, Gemma/PaliGemma.

Do not let closed frontier rows dominate the matrix. They should calibrate the
top end, while small/open rows should remain well represented.

## Frontier Source Requirements

For frontier models, collect scores from official technical reports, model
cards, or launch/evaluation reports whenever possible. Do not run these
benchmarks yourself for GPT, Claude, Gemini, Kimi, DeepSeek, Qwen, or other
frontier systems; that is redundant and may not match the vendor's evaluation
setup. These rows are report-mining rows, not compute jobs.

Start source discovery from:

- OpenAI GPT-4V/GPT-4o/GPT-5 system cards and official launch benchmark tables.
- Anthropic Claude model cards and system cards.
- Google Gemini technical reports and official model cards.
- Moonshot/Kimi technical reports and official Hugging Face model cards, especially Kimi-VL, Kimi K2.5, and Kimi K2.6.
- DeepSeek multimodal technical reports, especially DeepSeek-VL2 and Janus-Pro.
- Qwen-VL/Qwen2-VL/Qwen2.5-VL/Qwen3-VL technical reports.
- Official model cards or technical reports for InternVL, MiniCPM-V, Idefics, Llama Vision, Molmo, Pixtral, GLM-V, Phi Vision, Gemma, and PaliGemma.

If an official report does not contain one of our target benchmark scores, leave
that matrix cell missing. Do not infer it from a different benchmark. Third-party
leaderboards may be recorded as secondary evidence only when official sources
are unavailable, and they must be marked as lower confidence.

Seed official sources to check first:

- OpenAI GPT-4V system card: https://cdn.openai.com/papers/GPTV_System_Card.pdf
- OpenAI GPT-4o system card: https://cdn.openai.com/gpt-4o-system-card.pdf
- OpenAI GPT-4.1 launch benchmarks: https://openai.com/index/gpt-4-1/
- Anthropic system card index: https://www.anthropic.com/system-cards
- Anthropic Claude 3 model card: https://www-cdn.anthropic.com/f2986af8d052f26236f6251da62d16172cfabd6e/claude-3-model-card.pdf
- Google Gemini 1.5 report: https://storage.googleapis.com/deepmind-media/gemini/gemini_v1_5_report.pdf
- Google Gemini 2.5 report: https://storage.googleapis.com/deepmind-media/gemini/gemini_v2_5_report.pdf
- LLaVA-1.5 paper: https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_Improved_Baselines_with_Visual_Instruction_Tuning_CVPR_2024_paper.pdf
- LLaVA model zoo benchmark table: https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md
- SmolVLM paper: https://openreview.net/pdf/ff0790564f57d670a9033629dfbdaa6328752eca.pdf
- nanoVLM repository/eval list: https://github.com/huggingface/nanoVLM
- DeepSeek-VL2 technical report: https://arxiv.org/abs/2412.10302
- Kimi-VL technical report: https://arxiv.org/abs/2504.07491
- Kimi-VL official repository: https://github.com/MoonshotAI/Kimi-VL
- Qwen2.5-VL technical report: https://arxiv.org/abs/2502.13923
- Qwen2.5-VL official model card: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
- Qwen3-VL technical report, if current and official enough for the run date: https://arxiv.org/abs/2511.21631

Initial source-check notes:

- LLaVA-style reports commonly use VQAv2, GQA, VizWiz, ScienceQA-IMG, TextVQA, POPE, MME, MMBench, MMBench-CN, SEED-Bench, LLaVA-Bench-in-the-Wild, and MMVet.
- SmolVLM-style reports emphasize OCRBench, AI2D, ChartQA, TextVQA, DocVQA, ScienceQA, MMMU, MathVista, MMStar, and video benchmarks such as Video-MME, MLVU, MVBench, TempCompass, and WorldSense.
- Claude 3's official model card reports multimodal rows including MMMU, MathVista, AI2D, ChartQA, and DocVQA.
- Gemini 2.5's official report includes newer/adjacent vision rows such as Vibe-Eval, ZeroBench, and BetterChartQA; keep BetterChartQA separate from ChartQA.
- GPT-4V/GPT-4o official system cards may not contain our exact four benchmark scores. Keep GPT rows sparse unless another official OpenAI report gives the exact score.
- DeepSeek-VL2, Kimi-VL, Qwen2.5-VL, and Qwen3-VL are likely higher-yield sources for MMStar/OCRBench/ChartQA-style public VLM rows.
- Kimi-VL-style reports also include long-context/video/screen-agent benchmarks such as LongVideoBench, MMLongBench-Doc, ScreenSpot, OSWorld, and WindowsAgentArena. These should enter the candidate pool but may be excluded from the final v0 suite if they are too expensive or irrelevant to single-image nanochat-llava.
- Do not use LLM Reference, AI wiki pages, blogs, or benchmark aggregator pages as primary frontier sources. They are useful only for discovering the official report to cite.

## Data Collection Rules

Create a citation-backed table where each row is one observed score:

```text
model_id
model_family
model_size
benchmark
benchmark_variant
score_raw
score_normalized_0_100
metric_name
source_url
source_type
notes
```

Source priority:

1. official technical reports
2. official model cards
3. official benchmark leaderboards
4. lmms-eval or VLMEvalKit results
5. reputable third-party tables

Every score must have a source URL. For frontier models, prefer official
technical reports/model cards even if that leaves more missing cells. If the
exact benchmark split, prompt, or metric is unclear, record it but mark the cell
as low confidence.

Normalize all scores to "higher is better" on a 0-100 scale. For benchmarks
with unusual scales, record both the raw score and the normalization rule.
Examples:

- accuracy percentages stay as-is
- OCRBench-style point totals are divided by the official maximum score
- MME-style totals are divided by the official maximum score for the selected subset

Do not merge benchmark variants unless they are genuinely the same task. For
example, `mmmu_val` and `mmmu_test` should remain separate unless there is not
enough data.

## Matrix Construction

Build:

```text
M[model, benchmark] = normalized score in [0, 100]
```

Missing scores remain missing. Do not fill them manually.
For the public-score matrix, do not run new evaluations to fill missing cells.
The only model that should later be evaluated by us is nanochat-llava itself.

Then compute:

- number of models
- number of benchmarks
- observed cells
- fill rate
- per-benchmark coverage
- per-model coverage

Minimum useful target:

- at least 30 models
- at least 10 benchmarks
- at least 25% fill rate
- each benchmark used in the final recommendation has at least 10 observed scores, unless it is intentionally included as a low-coverage but unique signal

If this target is not met, reduce the benchmark candidate pool to the best
covered columns and try again.

## Low-Rank Check

Before trusting BenchPress, verify that the vision benchmark matrix has the
same kind of redundancy as the article found for LLM text benchmarks.

1. Find the largest reasonably sized fully observed submatrix.
2. Run SVD on normalized scores.
3. Plot singular values and cumulative spectrum.
4. Record how much variance rank 1, rank 2, and rank 3 explain.
5. Also compute benchmark-to-benchmark correlation heatmaps.

If rank 2 or rank 3 does not explain much, report that BenchPress-style
prediction is weak for vision and do not over-trust the selected set.

## BenchPress Predictor

Implement the article's two-part predictor.

### Transform

Work in logit space:

```text
p = clamp(score / 100, eps, 1 - eps)
logit(p) = log(p / (1 - p))
```

Use a small epsilon such as `1e-4` or `1e-3` to avoid infinities.

### Ingredient 1: Logit Benchmark Regression

For each target benchmark:

1. For every other benchmark, find models with scores on both.
2. Fit a simple linear or ridge regression in logit space.
3. Score the fit quality.
4. Keep the top `k=5` predictor benchmarks.
5. Predict the target from those 5 benchmarks.
6. Average predictions weighted by fit quality.

If a target/model pair does not have enough observed predictor scores, return
no regression prediction for that cell.

### Ingredient 2: Rank-2 SVD Logit Imputation

1. Initialize missing entries with benchmark column means in logit space.
2. Run rank-2 SVD.
3. Reconstruct the matrix.
4. Replace only missing entries with reconstructed values.
5. Keep observed entries pinned to their real values.
6. Iterate until convergence.

Also sweep ranks 1 through 8 to verify that rank 2 is actually best for VLM
data. Do not assume the LLM result transfers.

### Blend

Use the article's default blend first:

```text
BenchPress = 0.6 * LogitBenchReg + 0.4 * SVD_Logit_rank2
```

If regression has no coverage for a cell, fall back to SVD.

Run a blend sweep from 0.0 to 1.0 in steps of 0.1 and record whether 0.6/0.4
is still near-optimal for vision.

## Validation Protocol

Follow the original evaluation shape:

- per-model leave-50%-out holdout
- 3 random folds
- seed 42
- all other models keep their observed data
- predict held-out cells

Primary metrics:

- Median absolute error in raw score points
- Median absolute percentage error
- percentage within +/-3 points
- percentage within +/-5 points
- prediction coverage

Also run a "few revealed scores" experiment:

1. For each model, hide all scores.
2. Reveal 1, 2, 3, 4, 5, ... scores.
3. Predict the remaining scores after each reveal count.
4. Plot error versus number of revealed benchmarks.

This tells us how many vision benchmarks we need to run for nanochat-llava
before predictions for the rest become useful.

## Benchmark Selection

Run greedy forward selection for `k=1..8`.

At each step:

1. Try adding each remaining candidate benchmark to the selected set.
2. Treat selected benchmarks as the revealed scores for each model.
3. Use BenchPress to predict all unselected benchmarks.
4. Choose the benchmark that gives the lowest held-out error.

Report:

- best k=4 set
- best k=5 set
- error curve from k=1 to k=8
- principal-component coverage of each selected benchmark
- redundancy/correlation among selected benchmarks

Then compare:

```text
vibe_baseline_4 = {MMStar, OCRBench, ChartQA, ScienceQA}
greedy_best_4 = {...}
greedy_best_5 = {...}
nanoVLM_default_subset = {MMStar, MMMU, OCRBench, TextVQA, DocVQA, ScienceQA, MME, InfoVQA, ChartQA}
```

Treat the vibe baseline as a control, not as a default. It is fine if the final
recommendation keeps zero, one, two, three, or four of those benchmarks.

Recommend the evidence-selected set by default. If the vibe baseline is close,
say so, but do not keep it unless it is genuinely competitive:

- within 1.5 raw score points of greedy_best_4, or within 10% relative MedAPE
  of greedy_best_4
- covers at least three distinct benchmark clusters
- is cheaper or easier enough to justify any small loss

If the evidence-selected set differs from the vibe baseline, explain the
replacement:

```text
replace <weak/redundant benchmark> with <benchmark that improves coverage>
```

## Benchmark Predictability Report

For every benchmark, report how predictable it is from the rest:

```text
benchmark
coverage
median_abs_error_when_held_out
median_abs_percentage_error_when_held_out
top_5_predictor_benchmarks
correlation_cluster
notes
```

Benchmarks with low predictability are important because they may measure
capabilities not covered by the rest. These should be considered for inclusion
even if they are expensive or annoying.

## nanochat-llava Adaptation

The final v0 benchmark suite should stay small because every run should be
usable during development. The first suite is allowed to be imperfect if it is:

- deterministic
- cheap enough to run repeatedly
- covers obvious failure modes of a 64-token pooled visual encoder
- compatible with lmms-eval or easy to implement minimally

For nanochat-llava specifically, include no-image or shuffled-image controls
for a small subset of each selected benchmark. This checks whether the model is
actually using the image rather than answering from language priors.

## Deliverables

The `/goal` run should produce:

```text
benchpress_vision/
  scores_raw.csv
  score_matrix.csv
  sources.md
  benchpress_vision.py
  results/
    matrix_stats.json
    svd_spectrum.png
    benchmark_correlations.png
    rank_sweep.csv
    blend_sweep.csv
    holdout_results.csv
    greedy_selection.csv
    benchmark_predictability.csv
    recommendation.md
```

Do not edit `research_design.md` from the `/goal` run unless explicitly asked.
The output should be a recommendation first, not a codebase change.

## Suggested `/goal` Prompt

```text
Use benchpress_vision.md as the experiment spec. Build a cited VLM
model-by-benchmark score matrix from official technical reports/model cards
first, expanding the benchmark candidate pool with benchmarks commonly used in
those VLM reports. Treat {MMStar, OCRBench, ChartQA, ScienceQA} only as a
vibe-based baseline, not as a trusted target. Implement the BenchPress-style
logit regression plus rank-2 SVD predictor, validate it with per-model
leave-50%-out holdout, and use greedy forward selection to find the best k=4
and k=5 minimal benchmark suites for nanochat-llava. Produce the deliverables
under benchpress_vision/ and summarize the best suite, how it compares to the
vibe baseline, and why each selected benchmark adds non-redundant signal. Do
not edit research_design.md unless I explicitly ask.
```
