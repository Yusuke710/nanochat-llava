# BenchPress Vision Recommendation

## Recommendation

Best k=4: MMStar, ScienceQA, ChartQA, MMMU

Best k=5: MMStar, ScienceQA, ChartQA, MMMU, TextVQA

Primary holdout validation: MedAE 2.95, MedAPE 4.47%, within +/-5 67.0%, coverage 100.0%.

Best k=4 suite error: MedAE 1.88, MedAPE 2.69%, within +/-5 81.8%, coverage 100.0%.
Best k=5 suite error: MedAE 1.59, MedAPE 2.36%, within +/-5 89.2%, coverage 100.0%.
Vibe baseline `MMStar, OCRBench, ChartQA, ScienceQA` error: MedAE 2.02, MedAPE 3.08%, within +/-5 80.3%, coverage 100.0%.
nanoVLM default subset error: MedAE 1.59, MedAPE 2.45%, within +/-5 89.4%, coverage 100.0%.
Judge-style optional columns excluded from the default nanochat run-set greedy search: LLaVA-Bench-Wild, MMVet. They remain in the matrix as diagnostics.
The k=4 suite is error-minimal under the deterministic search, but k=5 is the practical default when one more run is affordable because it restores an explicit OCR/text-reading signal.

## Vibe Baseline Comparison

The vibe baseline is close enough to be a reasonable control, but it is still not the evidence-selected default.
It is 0.14 median absolute score points worse than greedy_best_4.

Recommended replacements relative to the vibe baseline:
- replace OCRBench with MMMU to improve non-redundant coverage.

## Why Each Selected Benchmark Adds Signal

- MMStar: vision-indispensable multimodal reasoning; cluster=general_reasoning; mean abs corr to selected peers 0.89. PC1 0.29, PC2 0.20, rank-2 norm 0.12.
- ScienceQA: science and diagram-style QA; cluster=math_science; mean abs corr to selected peers 0.61. PC1 0.01, PC2 -0.17, rank-2 norm 0.03.
- ChartQA: chart and plot reasoning; cluster=chart_document; mean abs corr to selected peers 0.87. PC1 0.20, PC2 0.21, rank-2 norm 0.08.
- MMMU: multidiscipline multimodal reasoning; cluster=general_reasoning; mean abs corr to selected peers 0.74. PC1 0.39, PC2 0.18, rank-2 norm 0.19.
- TextVQA: scene text OCR QA; cluster=ocr_text; mean abs corr to selected peers 0.75. PC1 0.27, PC2 -0.10, rank-2 norm 0.08.

## Low-Rank Check

Largest fully observed submatrix used for the spectrum: 11 models x 15 benchmarks.
Rank-1 explains 87.6% of variance; rank-2 explains 95.4%; rank-3 explains 98.0%.
Rank sweep best median absolute error is rank 1 at 2.97; rank-2 gives 3.53.
Blend sweep best regression weight is 0.7; the default 0.6/0.4 blend is retained for comparability with BenchPress.

## Redundancy

k=4 selected suite mean abs pairwise correlation 0.81, max 0.93.
k=5 selected suite mean abs pairwise correlation 0.78, max 0.93.

Hard-to-predict benchmarks, sorted by held-out median absolute error:
- Video-MME: MedAE 19.34, coverage 1, predictors .
- MLVU: MedAE 16.24, coverage 1, predictors .
- BetterChartQA: MedAE 16.00, coverage 4, predictors .
- MathVision: MedAE 8.77, coverage 2, predictors .
- EgoSchema: MedAE 7.90, coverage 4, predictors .
- VizWiz: MedAE 7.26, coverage 8, predictors MMVet;VQAv2;TextVQA;GQA;SEED-Image.
- DUDE: MedAE 6.00, coverage 4, predictors .
- VQAv2: MedAE 5.77, coverage 18, predictors LLaVA-Bench-Wild;SEED-Image;MME;GQA;ChartQA.

## nanochat-llava Adaptation

For v0 development runs, use the k=4 suite as the default and keep the k=5 suite as the higher-confidence periodic run. For each selected benchmark, add a small no-image or shuffled-image control slice so improvements require visual grounding rather than language priors.
