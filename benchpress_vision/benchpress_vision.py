#!/usr/bin/env python3
"""BenchPress-style VLM benchmark selection for nanochat-llava.

Run from the repository root:

    python benchpress_vision/benchpress_vision.py

Requires numpy, pandas, and matplotlib. The script is self-contained: it builds
the cited raw score table from report-mined records, normalizes scores, runs
holdout validation, and writes all deliverables under benchpress_vision/.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RAW_CSV = ROOT / "scores_raw.csv"
MATRIX_CSV = ROOT / "score_matrix.csv"
SOURCES_MD = ROOT / "sources.md"

EPS = 1e-3
SEED = 42
HOLDOUT_FOLDS = 3
REG_TOP_K = 5
DEFAULT_BLEND_REG_WEIGHT = 0.6
DEFAULT_SVD_RANK = 2

SOURCE_PRIORITY = {
    "official_vendor_report": 1,
    "official_model_card": 1,
    "official_project_report": 2,
    "official_model_card_comparison": 3,
    "official_project_report_comparison": 3,
}

SOURCES = {
    "internvl25": {
        "title": "InternVL2.5 official technical report blog",
        "url": "https://internvl.github.io/blog/2024-12-05-InternVL-2.5/",
        "type": "official_project_report",
        "notes": "Primary for InternVL2.5 rows. Non-InternVL comparison rows are marked as comparison evidence.",
    },
    "internvl25_mpo": {
        "title": "InternVL2.5-MPO official technical report blog",
        "url": "https://internvl.github.io/blog/2024-12-20-InternVL-2.5-MPO/",
        "type": "official_project_report",
        "notes": "Primary for InternVL2.5-MPO rows on the OpenCompass benchmark subset.",
    },
    "llava_model_zoo": {
        "title": "LLaVA official model zoo",
        "url": "https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md",
        "type": "official_model_card",
        "notes": "Primary for LLaVA v1.5 and v1.6 model rows.",
    },
    "smolvlm_500m": {
        "title": "SmolVLM-500M-Instruct official model card",
        "url": "https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct/blob/401eccf6f9c3555db6e85d77ee76772fff15c382/README.md",
        "type": "official_model_card",
        "notes": "Primary for SmolVLM 256M, 500M, and 2.2B small VLM rows.",
    },
    "smolvlm2": {
        "title": "SmolVLM2-2.2B-Instruct official model card",
        "url": "https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct/blob/482adb537c021c86670beed01cd58990d01e72e4/README.md",
        "type": "official_model_card",
        "notes": "Primary for SmolVLM2 2.2B image and video rows.",
    },
    "qwen25vl": {
        "title": "Qwen2.5-VL-7B-Instruct official model card",
        "url": "https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/main/README.md",
        "type": "official_model_card",
        "notes": "Primary for Qwen2.5-VL-7B. Competitor rows are marked as comparison evidence.",
    },
    "gemini15": {
        "title": "Gemini 1.5 official technical report",
        "url": "https://storage.googleapis.com/deepmind-media/gemini/gemini_v1_5_report.pdf",
        "type": "official_vendor_report",
        "notes": "Primary vendor source for Gemini 1.0 and 1.5 vision benchmark rows.",
    },
    "gemma3": {
        "title": "Gemma 3 official model card",
        "url": "https://ai.google.dev/gemma/docs/core/model_card_3",
        "type": "official_model_card",
        "notes": "Primary source for Gemma 3 multimodal rows.",
    },
    "minicpm_repo": {
        "title": "MiniCPM-V official repository",
        "url": "https://github.com/OpenBMB/MiniCPM-V",
        "type": "official_project_report",
        "notes": "Primary for MiniCPM-o 4.5 and related current MiniCPM rows.",
    },
}

MODEL_META: dict[str, tuple[str, str]] = {
    "GPT-4V": ("OpenAI", "closed"),
    "GPT-4o-20240513": ("OpenAI", "closed"),
    "GPT-4o-mini": ("OpenAI", "closed"),
    "Claude-3-Opus": ("Anthropic", "closed"),
    "Claude-3.5-Sonnet": ("Anthropic", "closed"),
    "Gemini-1.0-Pro": ("Google DeepMind", "closed"),
    "Gemini-1.0-Ultra": ("Google DeepMind", "closed"),
    "Gemini-1.5-Flash": ("Google DeepMind", "closed"),
    "Gemini-1.5-Pro": ("Google DeepMind", "closed"),
    "Gemini2.5-Flash-Nonthinking": ("Google DeepMind", "closed"),
    "Qwen2-VL-2B": ("Qwen", "2B"),
    "Qwen2-VL-7B": ("Qwen", "7B"),
    "Qwen2-VL-72B": ("Qwen", "72B"),
    "Qwen2.5-VL-7B": ("Qwen", "7B"),
    "Qwen3-VL-8B-Instruct": ("Qwen", "8B"),
    "InternVL2.5-1B": ("InternVL", "1B"),
    "InternVL2.5-2B": ("InternVL", "2B"),
    "InternVL2.5-4B": ("InternVL", "4B"),
    "InternVL2.5-8B": ("InternVL", "8B"),
    "InternVL2.5-26B": ("InternVL", "26B"),
    "InternVL2.5-38B": ("InternVL", "38B"),
    "InternVL2.5-78B": ("InternVL", "78B"),
    "InternVL2.5-1B-MPO": ("InternVL", "1B"),
    "InternVL2.5-2B-MPO": ("InternVL", "2B"),
    "InternVL2.5-4B-MPO": ("InternVL", "4B"),
    "InternVL2.5-8B-MPO": ("InternVL", "8B"),
    "InternVL2.5-26B-MPO": ("InternVL", "26B"),
    "InternVL2.5-38B-MPO": ("InternVL", "38B"),
    "InternVL2.5-78B-MPO": ("InternVL", "78B"),
    "LLaVA-1.6-Vicuna-7B": ("LLaVA", "7B"),
    "LLaVA-1.6-Vicuna-13B": ("LLaVA", "13B"),
    "LLaVA-1.6-Mistral-7B": ("LLaVA", "7B"),
    "LLaVA-1.6-Hermes-Yi-34B": ("LLaVA", "34B"),
    "LLaVA-1.5-7B": ("LLaVA", "7B"),
    "LLaVA-1.5-13B": ("LLaVA", "13B"),
    "LLaVA-1.5-7B-LoRA": ("LLaVA", "7B"),
    "LLaVA-1.5-13B-LoRA": ("LLaVA", "13B"),
    "SmolVLM-256M": ("SmolVLM", "256M"),
    "SmolVLM-500M": ("SmolVLM", "500M"),
    "SmolVLM-2.2B": ("SmolVLM", "2.2B"),
    "SmolVLM2-2.2B": ("SmolVLM", "2.2B"),
    "MiniCPM-o-2.6": ("MiniCPM", "8B"),
    "MiniCPM-o-4.5-Instruct": ("MiniCPM", "9B"),
    "MiniCPM-o-4.5-Thinking": ("MiniCPM", "9B"),
    "Gemma-3-IT-4B": ("Gemma", "4B"),
    "Gemma-3-IT-12B": ("Gemma", "12B"),
    "Gemma-3-IT-27B": ("Gemma", "27B"),
    "Gemma-3-PT-4B": ("Gemma", "4B"),
    "Gemma-3-PT-12B": ("Gemma", "12B"),
    "Gemma-3-PT-27B": ("Gemma", "27B"),
}

BENCHMARK_NOTES = {
    "MMMU": "multidiscipline multimodal reasoning",
    "MMMU-Pro": "harder multimodal reasoning",
    "MathVista": "visual math reasoning",
    "MathVision": "hard visual math reasoning",
    "AI2D": "science diagrams",
    "ChartQA": "chart and plot reasoning",
    "BetterChartQA": "harder chart reasoning",
    "DocVQA": "document OCR and layout QA",
    "InfoVQA": "infographic document QA",
    "InfographicVQA": "infographic document QA",
    "TextVQA": "scene text OCR QA",
    "OCRBench": "OCR aggregate benchmark",
    "MME": "general perception and cognition aggregate",
    "MMBench": "general multimodal reasoning",
    "MMBench-CN": "Chinese MMBench variant",
    "MMStar": "vision-indispensable multimodal reasoning",
    "MMVet": "LLM-judge multi-skill benchmark",
    "HallusionBench": "hallucination and visual grounding",
    "POPE": "object hallucination probe",
    "RealWorldQA": "real-world visual perception",
    "ScienceQA": "science and diagram-style QA",
    "VQAv2": "natural image VQA",
    "GQA": "compositional image QA",
    "VizWiz": "assistive image QA",
    "SEED-Image": "general image understanding",
    "LLaVA-Bench-Wild": "in-the-wild instruction following",
    "DUDE": "multi-page document understanding",
    "Video-MME": "video understanding",
    "MVBench": "video understanding",
    "MLVU": "multi-task long video understanding",
    "EgoSchema": "long video understanding",
}

BENCHMARK_CLUSTER = {
    "MMMU": "general_reasoning",
    "MMMU-Pro": "general_reasoning",
    "MMBench": "general_reasoning",
    "MMBench-CN": "general_reasoning",
    "MMStar": "general_reasoning",
    "MMVet": "general_reasoning_judge",
    "MME": "general_perception",
    "SEED-Image": "general_perception",
    "MathVista": "math_science",
    "MathVision": "math_science",
    "AI2D": "math_science",
    "ScienceQA": "math_science",
    "ChartQA": "chart_document",
    "BetterChartQA": "chart_document",
    "DocVQA": "ocr_document",
    "InfoVQA": "ocr_document",
    "InfographicVQA": "ocr_document",
    "DUDE": "ocr_document",
    "TextVQA": "ocr_text",
    "OCRBench": "ocr_text",
    "RealWorldQA": "real_world",
    "VQAv2": "natural_image",
    "GQA": "natural_image",
    "VizWiz": "natural_image",
    "POPE": "hallucination_grounding",
    "HallusionBench": "hallucination_grounding",
    "LLaVA-Bench-Wild": "instruction_judge",
    "Video-MME": "video",
    "MVBench": "video",
    "MLVU": "video",
    "EgoSchema": "video",
}

VIBE_BASELINE = ["MMStar", "OCRBench", "ChartQA", "ScienceQA"]
NANOVLM_DEFAULT = [
    "MMStar",
    "MMMU",
    "OCRBench",
    "TextVQA",
    "DocVQA",
    "ScienceQA",
    "MME",
    "InfoVQA",
    "ChartQA",
]

JUDGE_OPTIONAL_BENCHMARKS = {"MMVet", "LLaVA-Bench-Wild"}


def parse_numeric(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        if math.isnan(float(raw)):
            return None
        return float(raw)
    s = str(raw).strip()
    if not s or s == "-":
        return None
    s = s.replace("%", "")
    if "/" in s:
        parts = [p.strip() for p in s.split("/") if p.strip()]
        try:
            return sum(float(p) for p in parts)
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        return None


def normalize_score(benchmark: str, raw: Any) -> tuple[float | None, str, str]:
    value = parse_numeric(raw)
    if value is None:
        return None, "", ""
    if benchmark == "OCRBench":
        if value > 100:
            return value / 1000.0 * 100.0, "points/1000", "divided by 1000 and scaled to 0-100"
        return value, "percent_or_points/100", "already reported on a 0-100 scale"
    if benchmark == "MME":
        return value / 2800.0 * 100.0, "points/2800", "MME sum divided by 2800 and scaled to 0-100"
    if benchmark == "MMBench-Video":
        return value / 4.0 * 100.0, "score/4", "MMBench-Video score divided by 4 and scaled to 0-100"
    return value, "accuracy_or_score_percent", "kept as reported percentage/score"


def source_type_for(source_key: str, model_id: str, primary_families: set[str]) -> str:
    src_type = SOURCES[source_key]["type"]
    family = MODEL_META.get(model_id, ("unknown", ""))[0]
    if family in primary_families:
        return src_type
    if src_type == "official_model_card":
        return "official_model_card_comparison"
    return "official_project_report_comparison"


def source_priority(source_type: str) -> int:
    return SOURCE_PRIORITY.get(source_type, 5)


def build_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    def add_scores(
        source_key: str,
        model_id: str,
        scores: dict[str, Any],
        *,
        primary_families: set[str],
        variants: dict[str, str] | None = None,
        notes: str = "",
    ) -> None:
        family, size = MODEL_META.get(model_id, ("unknown", "unknown"))
        src_type = source_type_for(source_key, model_id, primary_families)
        for benchmark, raw in scores.items():
            norm, metric, norm_rule = normalize_score(benchmark, raw)
            if norm is None:
                continue
            records.append(
                {
                    "model_id": model_id,
                    "model_family": family,
                    "model_size": size,
                    "benchmark": benchmark,
                    "benchmark_variant": (variants or {}).get(benchmark, "reported"),
                    "score_raw": raw,
                    "score_normalized_0_100": round(float(norm), 4),
                    "metric_name": metric,
                    "source_url": SOURCES[source_key]["url"],
                    "source_type": src_type,
                    "source_priority": source_priority(src_type),
                    "notes": f"{notes} Normalization: {norm_rule}".strip(),
                }
            )

    # InternVL2.5 official report tables. InternVL rows are primary; comparison
    # rows are kept for coverage but have lower source priority.
    internvl_variants = {
        "MMMU": "val",
        "MathVista": "testmini",
        "AI2D": "test_masked",
        "ChartQA": "test_avg",
        "TextVQA": "val",
        "DocVQA": "test",
        "InfoVQA": "test",
        "OCRBench": "v1",
        "RealWorldQA": "test",
        "MME": "sum",
        "MMBench": "v1.1_en",
        "MMBench-CN": "test",
        "MMVet": "gpt4_turbo",
        "MMStar": "val",
        "HallusionBench": "avg",
        "POPE": "avg",
    }
    internvl_rows = {
        "GPT-4V": {
            "MMMU": 63.1,
            "MathVista": 58.1,
            "AI2D": 78.2,
            "ChartQA": 78.5,
            "TextVQA": 78.0,
            "DocVQA": 88.4,
            "InfoVQA": 75.1,
            "OCRBench": 645,
            "RealWorldQA": 61.4,
            "MME": 1926.6,
            "MMBench": 80.0,
            "MMBench-CN": 80.2,
            "MMVet": 67.5,
            "MMStar": 56.0,
            "HallusionBench": 46.5,
        },
        "GPT-4o-20240513": {
            "MMMU": 69.1,
            "MathVista": 63.8,
            "AI2D": 84.6,
            "ChartQA": 85.7,
            "TextVQA": 77.4,
            "DocVQA": 92.8,
            "InfoVQA": 79.2,
            "OCRBench": 736,
            "RealWorldQA": 75.4,
            "MMBench": 83.1,
            "MMBench-CN": 82.1,
            "MMVet": 69.1,
            "MMStar": 64.7,
            "HallusionBench": 55.0,
            "POPE": 86.9,
        },
        "Claude-3-Opus": {
            "AI2D": 70.6,
            "ChartQA": 80.8,
            "TextVQA": 67.5,
            "DocVQA": 89.3,
            "InfoVQA": 55.6,
            "OCRBench": 694,
            "MME": 1586.8,
            "MMBench": 60.1,
            "MMBench-CN": 59.2,
            "MMVet": 51.7,
            "MMStar": 45.7,
            "HallusionBench": 37.8,
        },
        "Claude-3.5-Sonnet": {
            "MMMU": 68.3,
            "MathVista": 67.7,
            "AI2D": 81.2,
            "ChartQA": 90.8,
            "TextVQA": 74.1,
            "DocVQA": 95.2,
            "InfoVQA": 74.3,
            "OCRBench": 788,
            "RealWorldQA": 60.1,
            "MMBench": 80.9,
            "MMBench-CN": 83.5,
            "MMVet": 70.1,
            "MMStar": 65.1,
            "HallusionBench": 55.5,
        },
        "Gemini-1.5-Pro": {
            "MMMU": 62.2,
            "MathVista": 63.9,
            "AI2D": 79.1,
            "ChartQA": 87.2,
            "TextVQA": 78.8,
            "DocVQA": 93.1,
            "InfoVQA": 81.0,
            "OCRBench": 754,
            "RealWorldQA": 67.5,
            "MMBench": 74.6,
            "MMBench-CN": 73.8,
            "MMVet": 64.0,
            "MMStar": 59.1,
            "HallusionBench": 45.6,
        },
        "Qwen2-VL-2B": {
            "MMMU": 41.1,
            "MathVista": 43.0,
            "AI2D": 74.7,
            "ChartQA": 73.5,
            "TextVQA": 79.7,
            "DocVQA": 90.1,
            "InfoVQA": 65.5,
            "OCRBench": 809,
            "RealWorldQA": 62.6,
            "MME": 1872.0,
            "MMBench": 72.2,
            "MMBench-CN": 73.5,
            "MMVet": 49.5,
            "MMStar": 48.0,
            "HallusionBench": 41.7,
        },
        "Qwen2-VL-7B": {
            "MMMU": 54.1,
            "MathVista": 58.2,
            "AI2D": 83.0,
            "ChartQA": 83.0,
            "TextVQA": 84.3,
            "DocVQA": 94.5,
            "InfoVQA": 76.5,
            "OCRBench": 866,
            "RealWorldQA": 70.1,
            "MME": 2326.8,
            "MMBench": 80.7,
            "MMBench-CN": 80.5,
            "MMVet": 62.0,
            "MMStar": 60.7,
            "HallusionBench": 50.6,
            "POPE": 88.1,
        },
        "Qwen2-VL-72B": {
            "MMMU": 64.5,
            "MathVista": 70.5,
            "AI2D": 88.1,
            "ChartQA": 88.3,
            "TextVQA": 85.5,
            "DocVQA": 96.5,
            "InfoVQA": 84.5,
            "OCRBench": 877,
            "RealWorldQA": 77.8,
            "MME": 2482.7,
            "MMBench": 85.9,
            "MMBench-CN": 86.6,
            "MMVet": 74.0,
            "MMStar": 68.3,
            "HallusionBench": 58.1,
        },
        "InternVL2.5-1B": {
            "MMMU": 40.9,
            "MathVista": 43.2,
            "AI2D": 69.3,
            "ChartQA": 75.9,
            "TextVQA": 72.0,
            "DocVQA": 84.8,
            "InfoVQA": 56.0,
            "OCRBench": 785,
            "RealWorldQA": 57.5,
            "MME": 1950.5,
            "MMBench": 68.4,
            "MMBench-CN": 66.3,
            "MMVet": 48.8,
            "MMStar": 50.1,
            "HallusionBench": 39.0,
            "POPE": 89.9,
        },
        "InternVL2.5-2B": {
            "MMMU": 43.6,
            "MathVista": 51.3,
            "AI2D": 74.9,
            "ChartQA": 79.2,
            "TextVQA": 74.3,
            "DocVQA": 88.7,
            "InfoVQA": 60.9,
            "OCRBench": 804,
            "RealWorldQA": 60.1,
            "MME": 2138.2,
            "MMBench": 72.2,
            "MMBench-CN": 71.9,
            "MMVet": 60.8,
            "MMStar": 53.7,
            "HallusionBench": 42.6,
            "POPE": 90.6,
        },
        "InternVL2.5-4B": {
            "MMMU": 52.3,
            "MathVista": 60.5,
            "AI2D": 81.4,
            "ChartQA": 84.0,
            "TextVQA": 76.8,
            "DocVQA": 91.6,
            "InfoVQA": 72.1,
            "OCRBench": 828,
            "RealWorldQA": 64.3,
            "MME": 2337.5,
            "MMBench": 79.3,
            "MMBench-CN": 79.3,
            "MMVet": 60.6,
            "MMStar": 58.3,
            "HallusionBench": 46.3,
            "POPE": 90.9,
        },
        "InternVL2.5-8B": {
            "MMMU": 56.0,
            "MathVista": 64.4,
            "AI2D": 84.5,
            "ChartQA": 84.8,
            "TextVQA": 79.1,
            "DocVQA": 93.0,
            "InfoVQA": 77.6,
            "OCRBench": 822,
            "RealWorldQA": 70.1,
            "MME": 2344.1,
            "MMBench": 83.2,
            "MMBench-CN": 82.6,
            "MMVet": 62.8,
            "MMStar": 62.8,
            "HallusionBench": 50.1,
            "POPE": 90.6,
        },
        "InternVL2.5-26B": {
            "MMMU": 60.0,
            "MathVista": 67.7,
            "AI2D": 86.4,
            "ChartQA": 87.2,
            "TextVQA": 82.4,
            "DocVQA": 94.0,
            "InfoVQA": 79.8,
            "OCRBench": 852,
            "RealWorldQA": 74.5,
            "MME": 2373.3,
            "MMBench": 84.2,
            "MMBench-CN": 85.5,
            "MMVet": 65.0,
            "MMStar": 66.5,
            "HallusionBench": 55.0,
            "POPE": 90.6,
        },
        "InternVL2.5-38B": {
            "MMMU": 63.9,
            "MathVista": 71.9,
            "AI2D": 87.6,
            "ChartQA": 88.2,
            "TextVQA": 82.7,
            "DocVQA": 95.3,
            "InfoVQA": 83.6,
            "OCRBench": 842,
            "RealWorldQA": 73.5,
            "MME": 2455.8,
            "MMBench": 85.5,
            "MMBench-CN": 86.3,
            "MMVet": 68.8,
            "MMStar": 67.9,
            "HallusionBench": 56.8,
            "POPE": 90.7,
        },
        "InternVL2.5-78B": {
            "MMMU": 70.1,
            "MathVista": 72.3,
            "AI2D": 89.1,
            "ChartQA": 88.3,
            "TextVQA": 83.4,
            "DocVQA": 95.1,
            "InfoVQA": 84.1,
            "OCRBench": 854,
            "RealWorldQA": 78.7,
            "MME": 2494.5,
            "MMBench": 87.4,
            "MMBench-CN": 88.5,
            "MMVet": 72.3,
            "MMStar": 69.5,
            "HallusionBench": 57.4,
            "POPE": 90.8,
        },
    }
    for model, scores in internvl_rows.items():
        add_scores(
            "internvl25",
            model,
            scores,
            primary_families={"InternVL"},
            variants=internvl_variants,
            notes="InternVL2.5 report table; AI2D uses the first w./wo-mask value when both are reported.",
        )

    mpo_variants = {
        "MMBench": "v1.1",
        "MMStar": "val",
        "MMMU": "val",
        "MathVista": "testmini",
        "HallusionBench": "avg",
        "AI2D": "test",
        "OCRBench": "v1",
        "MMVet": "reported",
    }
    mpo_rows = {
        "InternVL2.5-1B-MPO": {
            "MMBench": 67.2,
            "MMStar": 49.7,
            "MMMU": 40.8,
            "MathVista": 53.0,
            "HallusionBench": 40.0,
            "AI2D": 69.4,
            "OCRBench": 836,
            "MMVet": 47.2,
        },
        "InternVL2.5-2B-MPO": {
            "MMBench": 71.6,
            "MMStar": 55.0,
            "MMMU": 45.0,
            "MathVista": 56.4,
            "HallusionBench": 43.0,
            "AI2D": 75.3,
            "OCRBench": 842,
            "MMVet": 65.4,
        },
        "InternVL2.5-4B-MPO": {
            "MMBench": 78.6,
            "MMStar": 60.2,
            "MMMU": 51.6,
            "MathVista": 65.3,
            "HallusionBench": 47.8,
            "AI2D": 82.0,
            "OCRBench": 880,
            "MMVet": 67.1,
        },
        "InternVL2.5-8B-MPO": {
            "MMBench": 82.4,
            "MMStar": 65.7,
            "MMMU": 54.9,
            "MathVista": 68.9,
            "HallusionBench": 51.4,
            "AI2D": 84.5,
            "OCRBench": 883,
            "MMVet": 66.9,
        },
        "InternVL2.5-26B-MPO": {
            "MMBench": 84.2,
            "MMStar": 67.9,
            "MMMU": 57.3,
            "MathVista": 72.2,
            "HallusionBench": 55.4,
            "AI2D": 86.7,
            "OCRBench": 907,
            "MMVet": 68.8,
        },
        "InternVL2.5-38B-MPO": {
            "MMBench": 85.6,
            "MMStar": 69.8,
            "MMMU": 64.1,
            "MathVista": 73.8,
            "HallusionBench": 61.5,
            "AI2D": 88.1,
            "OCRBench": 885,
            "MMVet": 72.5,
        },
        "InternVL2.5-78B-MPO": {
            "MMBench": 87.8,
            "MMStar": 71.9,
            "MMMU": 68.7,
            "MathVista": 76.5,
            "HallusionBench": 58.9,
            "AI2D": 89.3,
            "OCRBench": 907,
            "MMVet": 73.6,
        },
    }
    for model, scores in mpo_rows.items():
        add_scores(
            "internvl25_mpo",
            model,
            scores,
            primary_families={"InternVL"},
            variants=mpo_variants,
            notes="InternVL2.5-MPO OpenCompass comparison table.",
        )

    llava_v16_variants = {
        "MMMU": "reported",
        "MathVista": "reported",
        "VQAv2": "reported",
        "GQA": "reported",
        "VizWiz": "reported",
        "ScienceQA": "img",
        "TextVQA": "reported",
        "POPE": "reported",
        "MME": "perception_plus_cognition",
        "MMBench": "reported",
        "MMBench-CN": "reported",
        "SEED-Image": "reported",
        "LLaVA-Bench-Wild": "reported",
        "MMVet": "reported",
    }
    llava_rows = {
        "LLaVA-1.6-Vicuna-7B": {
            "MMMU": 35.8,
            "MathVista": 34.6,
            "VQAv2": 81.8,
            "GQA": 64.2,
            "VizWiz": 57.6,
            "ScienceQA": 70.1,
            "TextVQA": 64.9,
            "POPE": 86.5,
            "MME": "1519/332",
            "MMBench": 67.4,
            "MMBench-CN": 60.6,
            "SEED-Image": 70.2,
            "LLaVA-Bench-Wild": 81.6,
            "MMVet": 43.9,
        },
        "LLaVA-1.6-Vicuna-13B": {
            "MMMU": 36.2,
            "MathVista": 35.3,
            "VQAv2": 82.8,
            "GQA": 65.4,
            "VizWiz": 60.5,
            "ScienceQA": 73.6,
            "TextVQA": 67.1,
            "POPE": 86.2,
            "MME": "1575/326",
            "MMBench": 70.0,
            "MMBench-CN": 64.4,
            "SEED-Image": 71.9,
            "LLaVA-Bench-Wild": 87.3,
            "MMVet": 48.4,
        },
        "LLaVA-1.6-Mistral-7B": {
            "MMMU": 35.3,
            "MathVista": 37.7,
            "VQAv2": 82.2,
            "GQA": 64.8,
            "VizWiz": 60.0,
            "ScienceQA": 72.8,
            "TextVQA": 65.7,
            "POPE": 86.7,
            "MME": "1498/321",
            "MMBench": 68.7,
            "MMBench-CN": 61.2,
            "SEED-Image": 72.2,
            "LLaVA-Bench-Wild": 83.2,
            "MMVet": 47.3,
        },
        "LLaVA-1.6-Hermes-Yi-34B": {
            "MMMU": 51.1,
            "MathVista": 46.5,
            "VQAv2": 83.7,
            "GQA": 67.1,
            "VizWiz": 63.8,
            "ScienceQA": 81.8,
            "TextVQA": 69.5,
            "POPE": 87.7,
            "MME": "1631/397",
            "MMBench": 79.3,
            "MMBench-CN": 79.0,
            "SEED-Image": 75.9,
            "LLaVA-Bench-Wild": 89.6,
            "MMVet": 57.4,
        },
        "LLaVA-1.5-7B": {
            "VQAv2": 78.5,
            "GQA": 62.0,
            "VizWiz": 50.0,
            "ScienceQA": 66.8,
            "TextVQA": 58.2,
            "POPE": 85.9,
            "MME": 1510.7,
            "MMBench": 64.3,
            "MMBench-CN": 58.3,
            "SEED-Image": 58.6,
            "LLaVA-Bench-Wild": 65.4,
            "MMVet": 31.1,
        },
        "LLaVA-1.5-13B": {
            "VQAv2": 80.0,
            "GQA": 63.3,
            "VizWiz": 53.6,
            "ScienceQA": 71.6,
            "TextVQA": 61.3,
            "POPE": 85.9,
            "MME": 1531.3,
            "MMBench": 67.7,
            "MMBench-CN": 63.6,
            "SEED-Image": 61.6,
            "LLaVA-Bench-Wild": 72.5,
            "MMVet": 36.1,
        },
        "LLaVA-1.5-7B-LoRA": {
            "VQAv2": 79.1,
            "GQA": 63.0,
            "VizWiz": 47.8,
            "ScienceQA": 68.4,
            "TextVQA": 58.2,
            "POPE": 86.4,
            "MME": 1476.9,
            "MMBench": 66.1,
            "MMBench-CN": 58.9,
            "SEED-Image": 60.1,
            "LLaVA-Bench-Wild": 67.9,
            "MMVet": 30.2,
        },
        "LLaVA-1.5-13B-LoRA": {
            "VQAv2": 80.0,
            "GQA": 63.3,
            "VizWiz": 58.9,
            "ScienceQA": 71.2,
            "TextVQA": 60.2,
            "POPE": 86.7,
            "MME": 1541.7,
            "MMBench": 68.5,
            "MMBench-CN": 61.5,
            "SEED-Image": 61.3,
            "LLaVA-Bench-Wild": 69.5,
            "MMVet": 38.3,
        },
    }
    for model, scores in llava_rows.items():
        add_scores(
            "llava_model_zoo",
            model,
            scores,
            primary_families={"LLaVA"},
            variants=llava_v16_variants,
            notes="Official LLaVA model zoo benchmark table.",
        )

    smol_variants = {
        "MathVista": "testmini",
        "MMMU": "val",
        "OCRBench": "normalized_0_100",
        "MMStar": "val",
        "AI2D": "reported",
        "ChartQA": "test",
        "ScienceQA": "reported",
        "TextVQA": "val",
        "DocVQA": "val",
    }
    smol_rows = {
        "SmolVLM-256M": {
            "MathVista": 35.9,
            "MMMU": 28.3,
            "OCRBench": 52.6,
            "MMStar": 34.6,
            "AI2D": 47.0,
            "ChartQA": 55.8,
            "ScienceQA": 73.6,
            "TextVQA": 49.9,
            "DocVQA": 58.3,
        },
        "SmolVLM-500M": {
            "MathVista": 40.1,
            "MMMU": 33.7,
            "OCRBench": 61.0,
            "MMStar": 38.3,
            "AI2D": 59.5,
            "ChartQA": 63.2,
            "ScienceQA": 79.7,
            "TextVQA": 60.5,
            "DocVQA": 70.5,
        },
        "SmolVLM-2.2B": {
            "MathVista": 43.9,
            "MMMU": 38.3,
            "OCRBench": 65.5,
            "MMStar": 41.8,
            "AI2D": 64.0,
            "ChartQA": 71.6,
            "ScienceQA": 84.5,
            "TextVQA": 72.1,
            "DocVQA": 79.7,
        },
    }
    for model, scores in smol_rows.items():
        add_scores(
            "smolvlm_500m",
            model,
            scores,
            primary_families={"SmolVLM"},
            variants=smol_variants,
            notes="Official SmolVLM 256M/500M model card evaluation table.",
        )

    smolvlm2_scores = {
        "MathVista": 51.5,
        "MMMU": 42.0,
        "OCRBench": 72.9,
        "MMStar": 46.0,
        "AI2D": 70.0,
        "ChartQA": 68.84,
        "ScienceQA": 90.0,
        "TextVQA": 73.21,
        "DocVQA": 79.98,
        "Video-MME": 52.1,
        "MLVU": 55.2,
        "MVBench": 46.27,
    }
    add_scores(
        "smolvlm2",
        "SmolVLM2-2.2B",
        smolvlm2_scores,
        primary_families={"SmolVLM"},
        variants={
            **smol_variants,
            "Video-MME": "reported",
            "MLVU": "m_avg",
            "MVBench": "reported",
        },
        notes="Official SmolVLM2 2.2B model card image and video evaluation tables.",
    )

    qwen_variants = {
        "MMMU": "val",
        "MMMU-Pro": "val",
        "DocVQA": "test",
        "InfoVQA": "test",
        "ChartQA": "test",
        "TextVQA": "val",
        "OCRBench": "v1",
        "MMBench": "v1.1_en_test",
        "MMStar": "val",
        "MMVet": "gpt4_turbo",
        "HallusionBench": "avg",
        "MathVista": "testmini",
        "MathVision": "reported",
    }
    qwen_rows = {
        "MiniCPM-o-2.6": {
            "MMMU": 50.4,
            "DocVQA": 93.0,
            "TextVQA": 80.1,
            "OCRBench": 852,
            "MMBench": 78.0,
            "MMStar": 57.5,
            "MMVet": 60.0,
            "HallusionBench": 48.1,
            "MathVista": 60.6,
        },
        "GPT-4o-mini": {
            "MMMU": 60.0,
            "MMMU-Pro": 37.6,
            "OCRBench": 785,
            "MMBench": 76.0,
            "MMStar": 54.8,
            "MMVet": 66.9,
            "HallusionBench": 46.1,
            "MathVista": 52.4,
        },
        "Qwen2-VL-7B": {
            "MMMU": 54.1,
            "MMMU-Pro": 30.5,
            "DocVQA": 94.5,
            "InfoVQA": 76.5,
            "ChartQA": 83.0,
            "TextVQA": 84.3,
            "OCRBench": 845,
            "MMBench": 80.7,
            "MMStar": 60.7,
            "MMVet": 62.0,
            "HallusionBench": 50.6,
            "MathVista": 58.2,
            "MathVision": 16.3,
        },
        "Qwen2.5-VL-7B": {
            "MMMU": 58.6,
            "MMMU-Pro": 41.0,
            "DocVQA": 95.7,
            "InfoVQA": 82.6,
            "ChartQA": 87.3,
            "TextVQA": 84.9,
            "OCRBench": 864,
            "MMBench": 82.6,
            "MMStar": 63.9,
            "MMVet": 67.1,
            "HallusionBench": 52.9,
            "MathVista": 68.2,
            "MathVision": 25.07,
        },
    }
    for model, scores in qwen_rows.items():
        add_scores(
            "qwen25vl",
            model,
            scores,
            primary_families={"Qwen"},
            variants=qwen_variants,
            notes="Official Qwen2.5-VL-7B model card image benchmark table.",
        )

    gemini_variants = {
        "MMMU": "val",
        "MathVista": "testmini",
        "AI2D": "test",
        "ChartQA": "test",
        "BetterChartQA": "internal",
        "DocVQA": "test",
        "DUDE": "test",
        "InfographicVQA": "test",
        "TextVQA": "val",
        "VQAv2": "test_dev",
        "RealWorldQA": "test",
        "EgoSchema": "test",
    }
    gemini_rows = {
        "Gemini-1.0-Pro": {
            "MMMU": 47.9,
            "MathVista": 46.6,
            "AI2D": 81.9,
            "ChartQA": 74.1,
            "BetterChartQA": 43.0,
            "DocVQA": 88.3,
            "DUDE": 39.0,
            "InfographicVQA": 75.2,
            "TextVQA": 74.6,
            "VQAv2": 71.2,
            "RealWorldQA": 61.6,
            "EgoSchema": 55.7,
        },
        "Gemini-1.0-Ultra": {
            "MMMU": 59.4,
            "MathVista": 53.0,
            "AI2D": 87.7,
            "ChartQA": 80.8,
            "BetterChartQA": 47.9,
            "DocVQA": 92.4,
            "DUDE": 44.0,
            "InfographicVQA": 80.3,
            "TextVQA": 82.3,
            "VQAv2": 77.8,
            "RealWorldQA": 64.7,
            "EgoSchema": 61.5,
        },
        "Gemini-1.5-Flash": {
            "MMMU": 56.1,
            "MathVista": 58.4,
            "AI2D": 91.7,
            "ChartQA": 85.4,
            "BetterChartQA": 59.0,
            "DocVQA": 89.9,
            "DUDE": 48.0,
            "InfographicVQA": 75.3,
            "TextVQA": 78.7,
            "VQAv2": 80.1,
            "RealWorldQA": 67.5,
            "EgoSchema": 65.7,
        },
        "Gemini-1.5-Pro": {
            "MMMU": 62.2,
            "MathVista": 63.9,
            "AI2D": 94.4,
            "ChartQA": 87.2,
            "BetterChartQA": 65.8,
            "DocVQA": 93.1,
            "DUDE": 46.0,
            "InfographicVQA": 81.0,
            "TextVQA": 78.7,
            "VQAv2": 80.2,
            "RealWorldQA": 70.4,
            "EgoSchema": 72.2,
        },
    }
    for model, scores in gemini_rows.items():
        add_scores(
            "gemini15",
            model,
            scores,
            primary_families={"Google DeepMind"},
            variants=gemini_variants,
            notes="Official Gemini 1.5 report Table 18/19 image and video results.",
        )

    gemma_variants = {
        "MMMU": "val_or_pt",
        "DocVQA": "reported",
        "InfoVQA": "reported",
        "TextVQA": "reported",
        "AI2D": "reported",
        "ChartQA": "reported",
        "VQAv2": "val",
        "MathVista": "testmini",
        "RealWorldQA": "reported",
    }
    gemma_rows = {
        "Gemma-3-IT-4B": {
            "MMMU": 48.8,
            "DocVQA": 75.8,
            "InfoVQA": 50.0,
            "TextVQA": 57.8,
            "AI2D": 74.8,
            "ChartQA": 68.8,
            "VQAv2": 62.4,
            "MathVista": 50.0,
        },
        "Gemma-3-IT-12B": {
            "MMMU": 59.6,
            "DocVQA": 87.1,
            "InfoVQA": 64.9,
            "TextVQA": 67.7,
            "AI2D": 84.2,
            "ChartQA": 75.7,
            "VQAv2": 71.6,
            "MathVista": 62.9,
        },
        "Gemma-3-IT-27B": {
            "MMMU": 64.9,
            "DocVQA": 86.6,
            "InfoVQA": 70.6,
            "TextVQA": 65.1,
            "AI2D": 84.5,
            "ChartQA": 78.0,
            "VQAv2": 71.0,
            "MathVista": 67.6,
        },
        "Gemma-3-PT-4B": {
            "MMMU": 39.2,
            "DocVQA": 72.8,
            "InfoVQA": 44.1,
            "TextVQA": 58.9,
            "RealWorldQA": 45.5,
            "AI2D": 63.2,
            "ChartQA": 63.6,
            "VQAv2": 63.9,
        },
        "Gemma-3-PT-12B": {
            "MMMU": 50.3,
            "DocVQA": 82.3,
            "InfoVQA": 54.8,
            "TextVQA": 66.5,
            "RealWorldQA": 52.2,
            "AI2D": 75.2,
            "ChartQA": 74.7,
            "VQAv2": 71.2,
        },
        "Gemma-3-PT-27B": {
            "MMMU": 56.1,
            "DocVQA": 85.6,
            "InfoVQA": 59.4,
            "TextVQA": 68.6,
            "RealWorldQA": 53.9,
            "AI2D": 79.0,
            "ChartQA": 76.3,
            "VQAv2": 72.9,
        },
    }
    for model, scores in gemma_rows.items():
        add_scores(
            "gemma3",
            model,
            scores,
            primary_families={"Gemma"},
            variants=gemma_variants,
            notes="Official Gemma 3 model card multimodal benchmark table.",
        )

    minicpm45_variants = {
        "MMBench": "en_v1.1",
        "MMBench-CN": "cn_v1.1",
        "MathVista": "reported",
        "MMVet": "reported",
        "MMMU": "reported",
        "MMStar": "reported",
        "HallusionBench": "reported",
        "AI2D": "reported",
        "OCRBench": "v1",
        "TextVQA": "val",
        "DocVQA": "val",
    }
    minicpm45_rows = {
        "MiniCPM-o-4.5-Instruct": {
            "MMBench": 87.6,
            "MMBench-CN": 87.2,
            "MathVista": 80.1,
            "MMVet": 74.4,
            "MMMU": 67.6,
            "MMStar": 73.1,
            "HallusionBench": 63.2,
            "AI2D": 87.6,
            "OCRBench": 876,
            "TextVQA": 83.8,
            "DocVQA": 94.7,
        },
        "MiniCPM-o-4.5-Thinking": {
            "MMBench": 89.0,
            "MMBench-CN": 87.6,
            "MathVista": 81.0,
            "MMVet": 73.6,
            "MMMU": 70.2,
            "MMStar": 73.6,
            "HallusionBench": 62.6,
            "AI2D": 88.5,
            "OCRBench": 879,
            "TextVQA": 79.8,
            "DocVQA": 92.3,
        },
    }
    for model, scores in minicpm45_rows.items():
        add_scores(
            "minicpm_repo",
            model,
            scores,
            primary_families={"MiniCPM"},
            variants=minicpm45_variants,
            notes="Official MiniCPM-V repository MiniCPM-o 4.5 visual understanding table.",
        )

    return records


def write_sources_md(raw: pd.DataFrame) -> None:
    lines = [
        "# BenchPress Vision Sources",
        "",
        "This file records the official technical reports, official model cards, and official project pages used to build `scores_raw.csv`.",
        "",
        "Source priority follows `benchpress_vision.md`: vendor/project technical reports and model cards first, comparison rows from official reports second, and no third-party leaderboard rows in this v0 matrix.",
        "",
        "Important normalization notes:",
        "",
        "- Percent-like accuracy and score values are kept on a 0-100 scale.",
        "- OCRBench values reported as points out of 1000 are divided by 1000 and scaled to 0-100.",
        "- MME sum values are divided by 2800 and scaled to 0-100.",
        "- LLaVA's `1519/332` style MME entries are summed before MME normalization.",
        "- The raw CSV preserves each report's split or prompt in `benchmark_variant`; the matrix groups by canonical benchmark name because many reports mix `test`, `val`, and toolkit variants for the same task.",
        "",
        "## Source Table",
        "",
        "| Source | Type | Rows | Benchmarks | Notes |",
        "|---|---:|---:|---|---|",
    ]
    for key, source in SOURCES.items():
        src_rows = raw[raw["source_url"] == source["url"]]
        benches = ", ".join(sorted(src_rows["benchmark"].unique())) if not src_rows.empty else ""
        lines.append(
            f"| [{source['title']}]({source['url']}) | {source['type']} | {len(src_rows)} | {benches} | {source['notes']} |"
        )
    lines.extend(
        [
            "",
            "## Frontier-Model Handling",
            "",
            "Gemini rows come from Google's official Gemini report. OpenAI and Anthropic rows are kept only where official VLM-family reports include exact public scores for the target matrix and are marked as comparison evidence, so they calibrate the high end without dominating the recommendation. The selected suites are evaluated primarily on all rows and reported with coverage diagnostics; they are not chosen by closed-model preference.",
        ]
    )
    SOURCES_MD.write_text("\n".join(lines) + "\n")


def make_raw_and_matrix() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records = build_records()
    raw = pd.DataFrame(records)
    raw = raw.sort_values(
        ["source_priority", "model_family", "model_id", "benchmark", "source_url"]
    ).reset_index(drop=True)
    raw.to_csv(RAW_CSV, index=False, quoting=csv.QUOTE_MINIMAL)

    # Matrix construction keeps the highest-priority source for duplicate
    # model/benchmark cells, then averages ties if two same-priority official
    # rows remain.
    grouped = (
        raw.sort_values(["source_priority"])
        .groupby(["model_id", "benchmark"], as_index=False)
        .agg(
            {
                "score_normalized_0_100": "first",
                "model_family": "first",
                "model_size": "first",
                "source_url": "first",
                "source_type": "first",
                "source_priority": "first",
            }
        )
    )
    matrix = grouped.pivot(
        index="model_id", columns="benchmark", values="score_normalized_0_100"
    ).sort_index(axis=0).sort_index(axis=1)
    matrix.to_csv(MATRIX_CSV)
    write_sources_md(raw)
    return raw, grouped, matrix


def to_logit(scores: np.ndarray) -> np.ndarray:
    p = np.clip(scores / 100.0, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def from_logit(logits: np.ndarray) -> np.ndarray:
    logits = np.clip(logits, -20.0, 20.0)
    p = 1.0 / (1.0 + np.exp(-logits))
    return np.clip(p * 100.0, 0.0, 100.0)


def impute_logit_svd(
    matrix: pd.DataFrame, rank: int = DEFAULT_SVD_RANK, max_iter: int = 50, tol: float = 1e-5
) -> pd.DataFrame:
    values = matrix.to_numpy(dtype=float)
    observed = ~np.isnan(values)
    logit_values = np.full_like(values, np.nan, dtype=float)
    logit_values[observed] = to_logit(values[observed])

    finite = np.isfinite(logit_values)
    total_count = int(finite.sum())
    overall = float(np.nansum(logit_values) / total_count) if total_count else 0.0
    col_counts = finite.sum(axis=0)
    col_sums = np.nansum(np.where(finite, logit_values, 0.0), axis=0)
    col_means = np.divide(
        col_sums,
        col_counts,
        out=np.full(logit_values.shape[1], overall, dtype=float),
        where=col_counts > 0,
    )
    filled = np.where(observed, logit_values, col_means)
    filled = np.nan_to_num(filled, nan=overall, posinf=8.0, neginf=-8.0)

    rank = max(1, min(rank, min(filled.shape) - 1 if min(filled.shape) > 1 else 1))
    for _ in range(max_iter):
        previous_missing = filled[~observed].copy()
        col_center = filled.mean(axis=0)
        centered = filled - col_center
        u, s, vt = np.linalg.svd(centered, full_matrices=False)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            recon = (u[:, :rank] * s[:rank]) @ vt[:rank, :] + col_center
        new_missing = np.nan_to_num(recon[~observed], nan=overall, posinf=8.0, neginf=-8.0)
        filled[~observed] = np.clip(0.5 * filled[~observed] + 0.5 * new_missing, -8.0, 8.0)
        filled[observed] = logit_values[observed]
        if previous_missing.size == 0:
            break
        delta = np.nanmax(np.abs(filled[~observed] - previous_missing))
        if delta < tol:
            break
    return pd.DataFrame(from_logit(filled), index=matrix.index, columns=matrix.columns)


@dataclass
class RegModel:
    source: str
    target: str
    intercept: float
    slope: float
    quality: float
    n: int


def fit_pairwise_regressions(matrix: pd.DataFrame, min_pairs: int = 6) -> dict[str, list[RegModel]]:
    logits = pd.DataFrame(to_logit(matrix.to_numpy(dtype=float)), index=matrix.index, columns=matrix.columns)
    fits: dict[str, list[RegModel]] = {}
    for target in matrix.columns:
        target_fits: list[RegModel] = []
        y = logits[target]
        for source in matrix.columns:
            if source == target:
                continue
            x = logits[source]
            mask = x.notna() & y.notna()
            n = int(mask.sum())
            if n < min_pairs:
                continue
            xv = x[mask].to_numpy(dtype=float)
            yv = y[mask].to_numpy(dtype=float)
            x_mean = float(xv.mean())
            y_mean = float(yv.mean())
            denom = float(np.sum((xv - x_mean) ** 2)) + 1e-6
            slope = float(np.sum((xv - x_mean) * (yv - y_mean)) / denom)
            intercept = y_mean - slope * x_mean
            yhat = intercept + slope * xv
            ss_res = float(np.sum((yv - yhat) ** 2))
            ss_tot = float(np.sum((yv - y_mean) ** 2)) + 1e-9
            r2 = max(0.0, 1.0 - ss_res / ss_tot)
            corr = float(np.corrcoef(xv, yv)[0, 1]) if n >= 3 else 0.0
            quality = max(r2, abs(corr) ** 2, 1e-4)
            target_fits.append(RegModel(source, target, intercept, slope, quality, n))
        target_fits.sort(key=lambda f: (f.quality, f.n), reverse=True)
        fits[target] = target_fits[:REG_TOP_K]
    return fits


def regression_prediction(
    matrix: pd.DataFrame,
    model_id: str,
    benchmark: str,
    fits: dict[str, list[RegModel]],
    allowed_sources: set[str] | None = None,
) -> float | None:
    preds = []
    weights = []
    for fit in fits.get(benchmark, []):
        if allowed_sources is not None and fit.source not in allowed_sources:
            continue
        if model_id not in matrix.index or fit.source not in matrix.columns:
            continue
        observed = matrix.loc[model_id, fit.source]
        if pd.isna(observed):
            continue
        pred_logit = fit.intercept + fit.slope * float(to_logit(np.array([observed]))[0])
        preds.append(float(from_logit(np.array([pred_logit]))[0]))
        weights.append(fit.quality)
    if not preds:
        return None
    return float(np.average(preds, weights=np.array(weights)))


def build_predictor(
    train: pd.DataFrame,
    *,
    svd_rank: int = DEFAULT_SVD_RANK,
    blend_reg_weight: float = DEFAULT_BLEND_REG_WEIGHT,
) -> tuple[pd.DataFrame, dict[str, list[RegModel]]]:
    svd_pred = impute_logit_svd(train, rank=svd_rank)
    fits = fit_pairwise_regressions(train)
    return svd_pred, fits


def predict_cell(
    train: pd.DataFrame,
    svd_pred: pd.DataFrame,
    fits: dict[str, list[RegModel]],
    model_id: str,
    benchmark: str,
    *,
    blend_reg_weight: float = DEFAULT_BLEND_REG_WEIGHT,
) -> tuple[float | None, str]:
    svd_value = float(svd_pred.loc[model_id, benchmark])
    reg_value = regression_prediction(train, model_id, benchmark, fits)
    if reg_value is None:
        return svd_value, "svd_fallback"
    alpha = blend_reg_weight
    return float(alpha * reg_value + (1.0 - alpha) * svd_value), "blend"


_SVD_BASIS_CACHE: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray, list[str], list[str]]] = {}


def full_matrix_svd_basis(
    matrix: pd.DataFrame, rank: int = DEFAULT_SVD_RANK
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Return column logit means and benchmark loadings learned from the public matrix."""
    key = (id(matrix), matrix.shape[0], matrix.shape[1], rank)
    cached = _SVD_BASIS_CACHE.get(key)
    if cached is not None:
        return cached
    imputed_scores = impute_logit_svd(matrix, rank=rank)
    logits = to_logit(imputed_scores.to_numpy(dtype=float))
    col_means = logits.mean(axis=0)
    centered = logits - col_means
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:rank, :]
    result = (col_means, basis, list(matrix.index), list(matrix.columns))
    _SVD_BASIS_CACHE[key] = result
    return result


def svd_latent_predictions_from_selected(
    matrix: pd.DataFrame, selected: list[str], rank: int = DEFAULT_SVD_RANK
) -> pd.DataFrame:
    col_means, basis, model_ids, benchmarks = full_matrix_svd_basis(matrix, rank=rank)
    bench_to_idx = {b: i for i, b in enumerate(benchmarks)}
    selected_idxs = [bench_to_idx[b] for b in selected if b in bench_to_idx]
    out = pd.DataFrame(index=model_ids, columns=benchmarks, dtype=float)
    for model in model_ids:
        observed_selected = [
            idx
            for idx in selected_idxs
            if not pd.isna(matrix.loc[model, benchmarks[idx]])
        ]
        if observed_selected:
            x = basis[:, observed_selected].T
            y_scores = np.array([matrix.loc[model, benchmarks[idx]] for idx in observed_selected], dtype=float)
            y = to_logit(y_scores) - col_means[observed_selected]
            latent, *_ = np.linalg.lstsq(x, y, rcond=None)
            pred_logits = col_means + latent @ basis
        else:
            pred_logits = col_means
        out.loc[model, :] = from_logit(pred_logits)
    return out


def metric_summary(df: pd.DataFrame) -> dict[str, float]:
    valid = df.dropna(subset=["prediction"]).copy()
    if valid.empty:
        return {
            "n": 0,
            "coverage": 0.0,
            "median_abs_error": math.nan,
            "median_abs_percentage_error": math.nan,
            "within_3_points": 0.0,
            "within_5_points": 0.0,
        }
    valid["abs_error"] = (valid["prediction"] - valid["true_score"]).abs()
    valid["abs_pct_error"] = valid["abs_error"] / valid["true_score"].clip(lower=1e-6) * 100.0
    return {
        "n": int(len(valid)),
        "coverage": float(len(valid) / len(df)) if len(df) else 0.0,
        "median_abs_error": float(valid["abs_error"].median()),
        "median_abs_percentage_error": float(valid["abs_pct_error"].median()),
        "within_3_points": float((valid["abs_error"] <= 3.0).mean()),
        "within_5_points": float((valid["abs_error"] <= 5.0).mean()),
    }


def make_holdout_predictions(
    matrix: pd.DataFrame,
    *,
    svd_rank: int = DEFAULT_SVD_RANK,
    blend_reg_weight: float = DEFAULT_BLEND_REG_WEIGHT,
    folds: int = HOLDOUT_FOLDS,
) -> pd.DataFrame:
    rng = random.Random(SEED)
    predictions: list[dict[str, Any]] = []
    observed_by_model = {
        model: [bench for bench in matrix.columns if not pd.isna(matrix.loc[model, bench])]
        for model in matrix.index
    }
    for fold in range(folds):
        train = matrix.copy()
        held_cells: list[tuple[str, str]] = []
        for model, benches in observed_by_model.items():
            if len(benches) < 2:
                continue
            shuffled = benches[:]
            rng.shuffle(shuffled)
            holdout_n = max(1, len(shuffled) // 2)
            for bench in shuffled[:holdout_n]:
                train.loc[model, bench] = np.nan
                held_cells.append((model, bench))
        svd_pred, fits = build_predictor(train, svd_rank=svd_rank, blend_reg_weight=blend_reg_weight)
        for model, bench in held_cells:
            pred, method = predict_cell(
                train,
                svd_pred,
                fits,
                model,
                bench,
                blend_reg_weight=blend_reg_weight,
            )
            true_score = float(matrix.loc[model, bench])
            predictions.append(
                {
                    "fold": fold,
                    "model_id": model,
                    "benchmark": bench,
                    "true_score": true_score,
                    "prediction": pred,
                    "method": method,
                    "abs_error": abs(pred - true_score) if pred is not None else math.nan,
                    "abs_percentage_error": abs(pred - true_score) / max(true_score, 1e-6) * 100.0
                    if pred is not None
                    else math.nan,
                }
            )
    return pd.DataFrame(predictions)


def rank_sweep(matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rank in range(1, min(8, min(matrix.shape) - 1) + 1):
        preds = make_holdout_predictions(matrix, svd_rank=rank, blend_reg_weight=0.0)
        summary = metric_summary(preds)
        rows.append({"rank": rank, **summary})
    return pd.DataFrame(rows)


def blend_sweep(matrix: pd.DataFrame, svd_rank: int = DEFAULT_SVD_RANK) -> pd.DataFrame:
    rows = []
    for i in range(11):
        alpha = i / 10.0
        preds = make_holdout_predictions(matrix, svd_rank=svd_rank, blend_reg_weight=alpha)
        summary = metric_summary(preds)
        rows.append({"regression_weight": alpha, "svd_weight": 1.0 - alpha, **summary})
    return pd.DataFrame(rows)


def suite_predictions(
    matrix: pd.DataFrame,
    selected: list[str],
    *,
    svd_rank: int = DEFAULT_SVD_RANK,
    blend_reg_weight: float = DEFAULT_BLEND_REG_WEIGHT,
) -> pd.DataFrame:
    selected = [b for b in selected if b in matrix.columns]
    selected_set = set(selected)
    svd_rows = svd_latent_predictions_from_selected(matrix, selected, rank=svd_rank)
    fits = fit_pairwise_regressions(matrix)
    rows: list[dict[str, Any]] = []
    for model in matrix.index:
        observed = [b for b in matrix.columns if not pd.isna(matrix.loc[model, b])]
        hidden = [b for b in observed if b not in selected_set]
        if not hidden:
            continue
        for bench in hidden:
            svd_value = float(svd_rows.loc[model, bench])
            reg_value = regression_prediction(
                matrix,
                model,
                bench,
                fits,
                allowed_sources=selected_set,
            )
            if reg_value is None:
                pred = svd_value
                method = "svd_latent_fallback"
            else:
                pred = blend_reg_weight * reg_value + (1.0 - blend_reg_weight) * svd_value
                method = "blend_selected"
            true_score = float(matrix.loc[model, bench])
            rows.append(
                {
                    "model_id": model,
                    "benchmark": bench,
                    "true_score": true_score,
                    "prediction": pred,
                    "method": method,
                    "abs_error": abs(pred - true_score) if pred is not None else math.nan,
                }
            )
    return pd.DataFrame(rows)


def suite_score(matrix: pd.DataFrame, selected: list[str]) -> dict[str, float]:
    preds = suite_predictions(matrix, selected)
    return metric_summary(preds)


def greedy_selection(matrix: pd.DataFrame, max_k: int = 8) -> pd.DataFrame:
    coverage = matrix.notna().sum()
    eligible = [
        b
        for b in matrix.columns
        if coverage[b] >= 10 or b in VIBE_BASELINE or b in NANOVLM_DEFAULT
    ]
    eligible = [b for b in eligible if b not in JUDGE_OPTIONAL_BENCHMARKS]
    selected: list[str] = []
    rows = []
    for k in range(1, max_k + 1):
        candidate_rows = []
        for candidate in eligible:
            if candidate in selected:
                continue
            suite = selected + [candidate]
            metrics = suite_score(matrix, suite)
            candidate_rows.append((candidate, metrics))
        candidate_rows.sort(
            key=lambda item: (
                item[1]["median_abs_error"],
                item[1]["median_abs_percentage_error"],
                -coverage[item[0]],
            )
        )
        best_candidate, best_metrics = candidate_rows[0]
        selected.append(best_candidate)
        rows.append(
            {
                "k": k,
                "added_benchmark": best_candidate,
                "selected_suite": ";".join(selected),
                **best_metrics,
                "selected_coverage_counts": json.dumps(
                    {b: int(coverage[b]) for b in selected}, sort_keys=True
                ),
            }
        )
    return pd.DataFrame(rows)


def full_submatrix(matrix: pd.DataFrame) -> pd.DataFrame:
    cols_by_coverage = list(matrix.notna().sum().sort_values(ascending=False).index)
    best: pd.DataFrame | None = None
    best_score = -1
    max_cols = min(15, len(cols_by_coverage))
    for ncols in range(3, max_cols + 1):
        cols = cols_by_coverage[:ncols]
        complete = matrix[cols].dropna(axis=0, how="any")
        if complete.shape[0] < 8:
            continue
        # Prefer broad benchmark coverage, then total observed area.
        score = complete.shape[1] * 1000 + complete.shape[0] * complete.shape[1]
        if score > best_score:
            best = complete
            best_score = score
    if best is not None:
        return best
    return matrix[cols_by_coverage[: min(8, len(cols_by_coverage))]].dropna(axis=0, how="any")


def compute_svd_spectrum(matrix: pd.DataFrame) -> dict[str, Any]:
    sub = full_submatrix(matrix)
    if min(sub.shape) < 2:
        sub = matrix.dropna(axis=1, thresh=max(6, int(0.4 * len(matrix))))
        sub = sub.fillna(sub.mean(axis=0))
    centered = sub - sub.mean(axis=0)
    u, s, vt = np.linalg.svd(centered.to_numpy(dtype=float), full_matrices=False)
    var = s**2
    explained = var / var.sum() if var.sum() > 0 else np.zeros_like(var)
    cumulative = np.cumsum(explained)
    return {
        "submatrix_models": int(sub.shape[0]),
        "submatrix_benchmarks": int(sub.shape[1]),
        "benchmarks": list(sub.columns),
        "singular_values": [float(x) for x in s],
        "explained_variance": [float(x) for x in explained],
        "cumulative_explained_variance": [float(x) for x in cumulative],
    }


def plot_svd_spectrum(spectrum: dict[str, Any]) -> None:
    s = np.array(spectrum["singular_values"], dtype=float)
    cum = np.array(spectrum["cumulative_explained_variance"], dtype=float)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = np.arange(1, len(s) + 1)
    ax1.bar(x, s, color="#4c78a8", label="singular value")
    ax1.set_xlabel("Component")
    ax1.set_ylabel("Singular value")
    ax2 = ax1.twinx()
    ax2.plot(x, cum, color="#f58518", marker="o", label="cumulative variance")
    ax2.set_ylabel("Cumulative explained variance")
    ax2.set_ylim(0, 1.05)
    ax1.set_title("VLM Benchmark Matrix Spectrum")
    fig.tight_layout()
    fig.savefig(RESULTS / "svd_spectrum.png", dpi=160)
    plt.close(fig)


def plot_correlations(matrix: pd.DataFrame) -> pd.DataFrame:
    corr = matrix.corr(min_periods=5)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr.fillna(0.0).to_numpy(dtype=float), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.index, fontsize=7)
    ax.set_title("Benchmark Correlations (pairwise, min 5 models)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(RESULTS / "benchmark_correlations.png", dpi=180)
    plt.close(fig)
    return corr


def benchmark_predictability(
    holdout: pd.DataFrame, matrix: pd.DataFrame, fits: dict[str, list[RegModel]]
) -> pd.DataFrame:
    coverage = matrix.notna().sum().to_dict()
    rows = []
    for bench in matrix.columns:
        bench_preds = holdout[holdout["benchmark"] == bench]
        metrics = metric_summary(bench_preds) if not bench_preds.empty else {}
        top = [fit.source for fit in fits.get(bench, [])[:REG_TOP_K]]
        rows.append(
            {
                "benchmark": bench,
                "coverage": int(coverage.get(bench, 0)),
                "median_abs_error_when_held_out": metrics.get("median_abs_error", math.nan),
                "median_abs_percentage_error_when_held_out": metrics.get(
                    "median_abs_percentage_error", math.nan
                ),
                "prediction_coverage": metrics.get("coverage", math.nan),
                "top_5_predictor_benchmarks": ";".join(top),
                "correlation_cluster": BENCHMARK_CLUSTER.get(bench, "other"),
                "notes": BENCHMARK_NOTES.get(bench, ""),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["median_abs_error_when_held_out", "coverage"], ascending=[False, False]
    )


def few_revealed_experiment(matrix: pd.DataFrame, max_revealed: int = 8) -> pd.DataFrame:
    rng = random.Random(SEED)
    rows = []
    observed_by_model = {
        model: [b for b in matrix.columns if not pd.isna(matrix.loc[model, b])]
        for model in matrix.index
    }
    for reveal_count in range(1, max_revealed + 1):
        all_preds = []
        for model, observed in observed_by_model.items():
            if len(observed) <= reveal_count:
                continue
            shuffled = observed[:]
            rng.shuffle(shuffled)
            revealed = set(shuffled[:reveal_count])
            train = matrix.copy()
            for bench in observed:
                if bench not in revealed:
                    train.loc[model, bench] = np.nan
            svd_pred, fits = build_predictor(train)
            for bench in observed:
                if bench in revealed:
                    continue
                pred, method = predict_cell(train, svd_pred, fits, model, bench)
                true_score = float(matrix.loc[model, bench])
                all_preds.append(
                    {
                        "model_id": model,
                        "benchmark": bench,
                        "true_score": true_score,
                        "prediction": pred,
                        "method": method,
                    }
                )
        summary = metric_summary(pd.DataFrame(all_preds))
        rows.append({"revealed_scores": reveal_count, **summary})
    return pd.DataFrame(rows)


def plot_few_revealed(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["revealed_scores"], df["median_abs_error"], marker="o", color="#4c78a8")
    ax.set_xlabel("Revealed benchmarks per model")
    ax.set_ylabel("Median absolute error")
    ax.set_title("Few-Revealed-Scores Prediction Error")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS / "few_revealed_error.png", dpi=160)
    plt.close(fig)


def pc_loadings(matrix: pd.DataFrame) -> pd.DataFrame:
    filled = matrix.fillna(matrix.mean(axis=0))
    centered = filled - filled.mean(axis=0)
    u, s, vt = np.linalg.svd(centered.to_numpy(dtype=float), full_matrices=False)
    rows = []
    for idx, bench in enumerate(matrix.columns):
        rows.append(
            {
                "benchmark": bench,
                "pc1_loading": float(vt[0, idx]) if vt.shape[0] > 0 else math.nan,
                "pc2_loading": float(vt[1, idx]) if vt.shape[0] > 1 else math.nan,
                "rank2_loading_norm": float(np.sum(vt[:2, idx] ** 2)) if vt.shape[0] > 1 else math.nan,
            }
        )
    return pd.DataFrame(rows)


def suite_redundancy(corr: pd.DataFrame, suite: list[str]) -> dict[str, float]:
    pairs = []
    for a, b in itertools.combinations(suite, 2):
        if a in corr.index and b in corr.columns and not pd.isna(corr.loc[a, b]):
            pairs.append(abs(float(corr.loc[a, b])))
    return {
        "mean_abs_pairwise_corr": float(np.mean(pairs)) if pairs else math.nan,
        "max_abs_pairwise_corr": float(np.max(pairs)) if pairs else math.nan,
    }


def explain_benchmark(bench: str, selected: list[str], corr: pd.DataFrame) -> str:
    cluster = BENCHMARK_CLUSTER.get(bench, "other")
    base = BENCHMARK_NOTES.get(bench, "benchmark signal")
    others = [b for b in selected if b != bench and b in corr.columns and bench in corr.index]
    if others:
        corrs = [
            abs(float(corr.loc[bench, other]))
            for other in others
            if not pd.isna(corr.loc[bench, other])
        ]
        corr_note = f"mean abs corr to selected peers {np.mean(corrs):.2f}" if corrs else "limited pairwise overlap"
    else:
        corr_note = "first selected benchmark"
    return f"{base}; cluster={cluster}; {corr_note}."


def format_metric(metrics: dict[str, float]) -> str:
    return (
        f"MedAE {metrics['median_abs_error']:.2f}, "
        f"MedAPE {metrics['median_abs_percentage_error']:.2f}%, "
        f"within +/-5 {metrics['within_5_points'] * 100:.1f}%, "
        f"coverage {metrics['coverage'] * 100:.1f}%"
    )


def write_recommendation(
    matrix: pd.DataFrame,
    stats: dict[str, Any],
    greedy: pd.DataFrame,
    holdout_summary: dict[str, float],
    rank_df: pd.DataFrame,
    blend_df: pd.DataFrame,
    corr: pd.DataFrame,
    predictability: pd.DataFrame,
) -> None:
    best4 = greedy.loc[greedy["k"] == 4, "selected_suite"].iloc[0].split(";")
    best5 = greedy.loc[greedy["k"] == 5, "selected_suite"].iloc[0].split(";")
    best4_metrics = suite_score(matrix, best4)
    best5_metrics = suite_score(matrix, best5)
    vibe_metrics = suite_score(matrix, VIBE_BASELINE)
    nanovlm_metrics = suite_score(matrix, NANOVLM_DEFAULT)
    rank2 = rank_df.loc[rank_df["rank"] == 2].iloc[0].to_dict()
    best_rank = rank_df.sort_values("median_abs_error").iloc[0].to_dict()
    best_blend = blend_df.sort_values("median_abs_error").iloc[0].to_dict()
    pc = pc_loadings(matrix).set_index("benchmark")
    red4 = suite_redundancy(corr, best4)
    red5 = suite_redundancy(corr, best5)
    vibe_delta = vibe_metrics["median_abs_error"] - best4_metrics["median_abs_error"]
    vibe_close = (
        vibe_delta <= 1.5
        or vibe_metrics["median_abs_percentage_error"]
        <= 1.1 * best4_metrics["median_abs_percentage_error"]
    )

    lines = [
        "# BenchPress Vision Recommendation",
        "",
        "## Recommendation",
        "",
        f"Best k=4: {', '.join(best4)}",
        "",
        f"Best k=5: {', '.join(best5)}",
        "",
        f"Primary holdout validation: {format_metric(holdout_summary)}.",
        "",
        f"Best k=4 suite error: {format_metric(best4_metrics)}.",
        f"Best k=5 suite error: {format_metric(best5_metrics)}.",
        f"Vibe baseline `{', '.join(VIBE_BASELINE)}` error: {format_metric(vibe_metrics)}.",
        f"nanoVLM default subset error: {format_metric(nanovlm_metrics)}.",
        f"Judge-style optional columns excluded from the default nanochat run-set greedy search: {', '.join(sorted(JUDGE_OPTIONAL_BENCHMARKS))}. They remain in the matrix as diagnostics.",
    ]
    best4_clusters = {BENCHMARK_CLUSTER.get(b, "other") for b in best4}
    if not ({"ocr_text", "ocr_document"} & best4_clusters):
        lines.append(
            "The k=4 suite is error-minimal under the deterministic search, but k=5 is the practical default when one more run is affordable because it restores an explicit OCR/text-reading signal."
        )
    lines.extend(["", "## Vibe Baseline Comparison", ""])
    if vibe_close:
        lines.append(
            "The vibe baseline is close enough to be a reasonable control, but it is still not the evidence-selected default."
        )
    else:
        lines.append(
            "The vibe baseline is not competitive with the evidence-selected k=4 suite under the suite simulation."
        )
    lines.extend(
        [
            f"It is {vibe_delta:.2f} median absolute score points worse than greedy_best_4.",
            "",
            "Recommended replacements relative to the vibe baseline:",
        ]
    )
    old_missing = [b for b in VIBE_BASELINE if b not in best4]
    new_added = [b for b in best4 if b not in VIBE_BASELINE]
    for old, replacement in itertools.zip_longest(old_missing, new_added):
        if old and replacement:
            lines.append(f"- replace {old} with {replacement} to improve non-redundant coverage.")
        elif replacement:
            lines.append(f"- add {replacement} to improve non-redundant coverage.")
    if all(b in best4 for b in VIBE_BASELINE):
        lines.append("- no replacement: greedy retained the full vibe baseline.")

    lines.extend(
        [
            "",
            "## Why Each Selected Benchmark Adds Signal",
            "",
        ]
    )
    for bench in best5:
        load = pc.loc[bench] if bench in pc.index else None
        pc_text = (
            f"PC1 {load['pc1_loading']:.2f}, PC2 {load['pc2_loading']:.2f}, rank-2 norm {load['rank2_loading_norm']:.2f}"
            if load is not None
            else "PC loading unavailable"
        )
        lines.append(f"- {bench}: {explain_benchmark(bench, best5, corr)} {pc_text}.")

    lines.extend(
        [
            "",
            "## Low-Rank Check",
            "",
            f"Largest fully observed submatrix used for the spectrum: {stats['svd_spectrum']['submatrix_models']} models x {stats['svd_spectrum']['submatrix_benchmarks']} benchmarks.",
            f"Rank-1 explains {stats['svd_spectrum']['rank1_explained'] * 100:.1f}% of variance; rank-2 explains {stats['svd_spectrum']['rank2_cumulative'] * 100:.1f}%; rank-3 explains {stats['svd_spectrum']['rank3_cumulative'] * 100:.1f}%.",
            f"Rank sweep best median absolute error is rank {int(best_rank['rank'])} at {best_rank['median_abs_error']:.2f}; rank-2 gives {rank2['median_abs_error']:.2f}.",
            f"Blend sweep best regression weight is {best_blend['regression_weight']:.1f}; the default 0.6/0.4 blend is retained for comparability with BenchPress.",
            "",
            "## Redundancy",
            "",
            f"k=4 selected suite mean abs pairwise correlation {red4['mean_abs_pairwise_corr']:.2f}, max {red4['max_abs_pairwise_corr']:.2f}.",
            f"k=5 selected suite mean abs pairwise correlation {red5['mean_abs_pairwise_corr']:.2f}, max {red5['max_abs_pairwise_corr']:.2f}.",
            "",
            "Hard-to-predict benchmarks, sorted by held-out median absolute error:",
        ]
    )
    for _, row in predictability.head(8).iterrows():
        lines.append(
            f"- {row['benchmark']}: MedAE {row['median_abs_error_when_held_out']:.2f}, coverage {int(row['coverage'])}, predictors {row['top_5_predictor_benchmarks']}."
        )

    lines.extend(
        [
            "",
            "## nanochat-llava Adaptation",
            "",
            "For v0 development runs, use the k=4 suite as the default and keep the k=5 suite as the higher-confidence periodic run. For each selected benchmark, add a small no-image or shuffled-image control slice so improvements require visual grounding rather than language priors.",
        ]
    )
    (RESULTS / "recommendation.md").write_text("\n".join(lines) + "\n")


def matrix_stats(raw: pd.DataFrame, matrix: pd.DataFrame, spectrum: dict[str, Any]) -> dict[str, Any]:
    observed = int(matrix.notna().sum().sum())
    rank1 = spectrum["explained_variance"][0] if spectrum["explained_variance"] else math.nan
    rank2 = spectrum["cumulative_explained_variance"][1] if len(spectrum["cumulative_explained_variance"]) > 1 else math.nan
    rank3 = spectrum["cumulative_explained_variance"][2] if len(spectrum["cumulative_explained_variance"]) > 2 else math.nan
    return {
        "raw_observations": int(len(raw)),
        "models": int(matrix.shape[0]),
        "benchmarks": int(matrix.shape[1]),
        "observed_cells": observed,
        "fill_rate": float(observed / (matrix.shape[0] * matrix.shape[1])),
        "per_benchmark_coverage": {k: int(v) for k, v in matrix.notna().sum().sort_values(ascending=False).items()},
        "per_model_coverage": {k: int(v) for k, v in matrix.notna().sum(axis=1).sort_values(ascending=False).items()},
        "target_check": {
            "at_least_30_models": bool(matrix.shape[0] >= 30),
            "at_least_10_benchmarks": bool(matrix.shape[1] >= 10),
            "at_least_25_percent_fill": bool(observed / (matrix.shape[0] * matrix.shape[1]) >= 0.25),
        },
        "svd_spectrum": {
            **spectrum,
            "rank1_explained": float(rank1),
            "rank2_cumulative": float(rank2),
            "rank3_cumulative": float(rank3),
        },
    }


def main() -> None:
    ROOT.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    raw, grouped, matrix = make_raw_and_matrix()
    spectrum = compute_svd_spectrum(matrix)
    stats = matrix_stats(raw, matrix, spectrum)
    plot_svd_spectrum(spectrum)
    corr = plot_correlations(matrix)

    rank_df = rank_sweep(matrix)
    rank_df.to_csv(RESULTS / "rank_sweep.csv", index=False)
    blend_df = blend_sweep(matrix)
    blend_df.to_csv(RESULTS / "blend_sweep.csv", index=False)

    holdout = make_holdout_predictions(matrix)
    holdout.to_csv(RESULTS / "holdout_results.csv", index=False)
    holdout_summary = metric_summary(holdout)
    stats["holdout_summary_default"] = holdout_summary

    svd_pred, fits = build_predictor(matrix)
    predictability = benchmark_predictability(holdout, matrix, fits)
    predictability.to_csv(RESULTS / "benchmark_predictability.csv", index=False)

    greedy = greedy_selection(matrix, max_k=8)
    greedy.to_csv(RESULTS / "greedy_selection.csv", index=False)

    few = few_revealed_experiment(matrix)
    few.to_csv(RESULTS / "few_revealed.csv", index=False)
    plot_few_revealed(few)

    with (RESULTS / "matrix_stats.json").open("w") as f:
        json.dump(stats, f, indent=2)

    write_recommendation(
        matrix,
        stats,
        greedy,
        holdout_summary,
        rank_df,
        blend_df,
        corr,
        predictability,
    )
    print(f"Wrote {RAW_CSV}")
    print(f"Wrote {MATRIX_CSV}")
    print(f"Wrote results to {RESULTS}")
    print(f"Models: {matrix.shape[0]}, benchmarks: {matrix.shape[1]}, fill: {stats['fill_rate']:.1%}")
    best4 = greedy.loc[greedy["k"] == 4, "selected_suite"].iloc[0]
    best5 = greedy.loc[greedy["k"] == 5, "selected_suite"].iloc[0]
    print(f"Best k=4: {best4}")
    print(f"Best k=5: {best5}")
    print(f"Holdout: {format_metric(holdout_summary)}")


if __name__ == "__main__":
    main()
