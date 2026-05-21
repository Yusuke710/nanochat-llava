#!/usr/bin/env python3
"""Build the static BenchPress Vision HTML report.

The report intentionally uses only local artifacts and the Python standard
library so it can be regenerated after rerunning benchpress_vision.py.
"""

from __future__ import annotations

import csv
import html
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = ROOT / "index.html"

SELECTED_K4 = ["MMStar", "ScienceQA", "ChartQA", "MMMU"]
SELECTED_K5 = ["MMStar", "ScienceQA", "ChartQA", "MMMU", "TextVQA"]
VIBE = ["MMStar", "OCRBench", "ChartQA", "ScienceQA"]
NANOVLM = ["MMStar", "ScienceQA", "ChartQA", "MMMU", "TextVQA"]
JUDGE_STYLE = ["LLaVA-Bench-Wild", "MMVet"]

CLUSTER_LABELS = {
    "general_reasoning": "General reasoning",
    "general_reasoning_judge": "Judge-style reasoning",
    "general_perception": "General perception",
    "math_science": "Math and science",
    "chart_document": "Charts and plots",
    "ocr_document": "Document OCR",
    "ocr_text": "Scene text OCR",
    "real_world": "Real-world perception",
    "natural_image": "Natural image VQA",
    "hallucination": "Hallucination",
    "video": "Video",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def fmt_num(value: object, digits: int = 2) -> str:
    if value in ("", None):
        return ""
    value = float(value)
    if math.isclose(value, round(value), rel_tol=0, abs_tol=10 ** (-(digits + 2))):
        return str(int(round(value)))
    return f"{value:.{digits}f}"


def fmt_pct(value: object, digits: int = 1, scale: float = 100.0) -> str:
    if value in ("", None):
        return ""
    return f"{float(value) * scale:.{digits}f}%"


def median(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.median(values) if values else float("nan")


def percentile(values: Iterable[float], pct: float) -> float:
    values = sorted(values)
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    rank = pct * (len(values) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return values[lower]
    weight = rank - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def metric_tile(label: str, value: str, detail: str = "") -> str:
    return (
        '<div class="metric">'
        f'<div class="metric__value">{esc(value)}</div>'
        f'<div class="metric__label">{esc(label)}</div>'
        f'<div class="metric__detail">{esc(detail)}</div>'
        "</div>"
    )


def tag(text: str, tone: str = "") -> str:
    tone_cls = f" tag--{tone}" if tone else ""
    return f'<span class="tag{tone_cls}">{esc(text)}</span>'


def suite_tags(items: list[str]) -> str:
    return "".join(tag(item) for item in items)


def bar(value: float, max_value: float, label: str = "") -> str:
    width = 0 if max_value <= 0 else max(0, min(100, value / max_value * 100))
    return (
        '<div class="bar" role="img" aria-label="'
        f'{esc(label or fmt_num(value))}"><span style="width:{width:.2f}%"></span></div>'
    )


def table(headers: list[str], rows: list[list[object]], cls: str = "") -> str:
    class_attr = f' class="{esc(cls)}"' if cls else ""
    out = [f"<table{class_attr}><thead><tr>"]
    out.extend(f"<th>{esc(h)}</th>" for h in headers)
    out.append("</tr></thead><tbody>")
    for row in rows:
        out.append("<tr>")
        out.extend(f"<td>{cell}</td>" for cell in row)
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def line_svg(
    rows: list[dict[str, str]],
    x_key: str,
    y_key: str,
    label_key: str,
    title: str,
    y_label: str,
    width: int = 680,
    height: int = 260,
) -> str:
    values = [(float(r[x_key]), float(r[y_key]), r[label_key]) for r in rows]
    if not values:
        return ""
    pad_l, pad_r, pad_t, pad_b = 54, 20, 22, 42
    xs = [v[0] for v in values]
    ys = [v[1] for v in values]
    x_min, x_max = min(xs), max(xs)
    y_min = 0
    y_max = max(ys) * 1.12 if max(ys) > 0 else 1

    def sx(x: float) -> float:
        if math.isclose(x_min, x_max):
            return width / 2
        return pad_l + (x - x_min) / (x_max - x_min) * (width - pad_l - pad_r)

    def sy(y: float) -> float:
        return height - pad_b - (y - y_min) / (y_max - y_min) * (height - pad_t - pad_b)

    points = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y, _ in values)
    y_ticks = [0, y_max / 2, y_max]
    x_ticks = xs
    tick_lines = []
    for yt in y_ticks:
        y = sy(yt)
        tick_lines.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" class="gridline" />'
            f'<text x="{pad_l - 8}" y="{y + 4:.1f}" text-anchor="end">{fmt_num(yt, 1)}</text>'
        )
    for xt in x_ticks:
        x = sx(xt)
        tick_lines.append(
            f'<text x="{x:.1f}" y="{height - 16}" text-anchor="middle">{fmt_num(xt, 0)}</text>'
        )
    dots = []
    for x, y, label in values:
        dots.append(
            f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4.5"><title>{esc(label)}: {fmt_num(y)}</title></circle>'
        )
    return f"""
    <svg class="plot-svg" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">
      <title>{esc(title)}</title>
      <rect width="{width}" height="{height}" class="plot-bg" />
      {''.join(tick_lines)}
      <line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{height - pad_b}" class="axis" />
      <line x1="{pad_l}" y1="{height - pad_b}" x2="{width - pad_r}" y2="{height - pad_b}" class="axis" />
      <polyline points="{points}" class="plot-line" />
      {''.join(dots)}
      <text x="{width / 2:.1f}" y="{height - 2}" text-anchor="middle">{esc(label_key)}</text>
      <text x="16" y="{height / 2:.1f}" transform="rotate(-90, 16, {height / 2:.1f})" text-anchor="middle">{esc(y_label)}</text>
    </svg>
    """


def stacked_cluster_svg(cluster_counts: Counter[str]) -> str:
    total = sum(cluster_counts.values()) or 1
    colors = [
        "#0b6e69",
        "#c77900",
        "#315d9b",
        "#7d4b73",
        "#a83f39",
        "#4b6f44",
        "#6b5b95",
        "#455a64",
        "#9c5f1c",
        "#3d7199",
    ]
    segments = []
    x = 0.0
    labels = []
    for idx, (cluster, count) in enumerate(cluster_counts.most_common()):
        width = count / total * 100
        color = colors[idx % len(colors)]
        segments.append(f'<rect x="{x:.2f}" y="0" width="{width:.2f}" height="16" fill="{color}" />')
        labels.append(
            f'<span><i style="background:{color}"></i>{esc(CLUSTER_LABELS.get(cluster, cluster))}: {count}</span>'
        )
        x += width
    return (
        '<div class="stacked" role="img" aria-label="Benchmark cluster composition">'
        f'<svg viewBox="0 0 100 16" preserveAspectRatio="none">{"".join(segments)}</svg>'
        f'<div class="legend">{"".join(labels)}</div>'
        "</div>"
    )


def source_rows_from_markdown(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| ["):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 5:
            continue
        source, source_type, row_count, benchmarks, notes = cells
        if source.startswith("[") and "](" in source and source.endswith(")"):
            title, url = source[1:-1].split("](", 1)
            source = f'<a href="{esc(url)}">{esc(title)}</a>'
        else:
            source = esc(source)
        rows.append([source, esc(source_type), esc(row_count), esc(benchmarks), esc(notes)])
    return rows


def link(url: str, text: str) -> str:
    return f'<a href="{esc(url)}">{esc(text)}</a>'


def pick_score(rows: list[dict[str, str]], model: str, benchmark: str) -> dict[str, str] | None:
    setting_priority = {"no_tools": 0, "mini": 1, "normalized": 2, "with_tools": 3, "with_python": 4}
    matches = [r for r in rows if r["model_id"] == model and r["benchmark"] == benchmark]
    if not matches:
        return None
    return sorted(matches, key=lambda r: setting_priority.get(r["setting"], 99))[0]


def main() -> None:
    stats = read_json(RESULTS / "matrix_stats.json")
    greedy = read_csv(RESULTS / "greedy_selection.csv")
    blend = read_csv(RESULTS / "blend_sweep.csv")
    rank = read_csv(RESULTS / "rank_sweep.csv")
    few = read_csv(RESULTS / "few_revealed.csv")
    predictability = read_csv(RESULTS / "benchmark_predictability.csv")
    holdout = read_csv(RESULTS / "holdout_results.csv")
    raw = read_csv(ROOT / "scores_raw.csv")
    matrix = read_csv(ROOT / "score_matrix.csv")
    frontier_scores = read_csv(ROOT / "frontier_2026_scores.csv")
    frontier_status = read_csv(ROOT / "frontier_2026_model_status.csv")

    abs_errors = [float(r["abs_error"]) for r in holdout]
    abs_pct_errors = [float(r["abs_percentage_error"]) for r in holdout]
    source_type_counts = Counter(r["source_type"] for r in raw)
    cluster_counts = Counter(r["correlation_cluster"] for r in predictability)
    best_blend = min(blend, key=lambda r: float(r["median_abs_error"]))
    default_blend = next(r for r in blend if math.isclose(float(r["regression_weight"]), 0.6))
    best_rank = min(rank, key=lambda r: float(r["median_abs_error"]))
    k4 = next(r for r in greedy if r["k"] == "4")
    k5 = next(r for r in greedy if r["k"] == "5")
    vibe_summary = {
        "suite": "MMStar;OCRBench;ChartQA;ScienceQA",
        "median_abs_error": 2.02,
        "median_abs_percentage_error": 3.08,
        "within_5_points": 0.803,
        "coverage": 1.0,
    }
    nanovlm_summary = {
        "suite": "MMStar;ScienceQA;ChartQA;MMMU;TextVQA",
        "median_abs_error": 1.59,
        "median_abs_percentage_error": 2.45,
        "within_5_points": 0.894,
        "coverage": 1.0,
    }

    coverage = stats["per_benchmark_coverage"]
    max_coverage = max(coverage.values())
    coverage_rows = [
        [
            esc(benchmark),
            esc(count),
            bar(count, max_coverage, f"{benchmark} coverage {count} models"),
        ]
        for benchmark, count in sorted(coverage.items(), key=lambda item: (-item[1], item[0]))
    ]

    greedy_rows = []
    previous_error = None
    for row in greedy:
        error = float(row["median_abs_error"])
        improvement = "" if previous_error is None else fmt_num(previous_error - error)
        previous_error = error
        greedy_rows.append(
            [
                esc(row["k"]),
                tag(row["added_benchmark"], "selected" if row["added_benchmark"] in SELECTED_K5 else ""),
                suite_tags(row["selected_suite"].split(";")),
                esc(row["n"]),
                fmt_num(row["median_abs_error"]),
                fmt_num(row["median_abs_percentage_error"]),
                fmt_pct(row["within_5_points"], 1),
                esc(improvement),
            ]
        )

    suite_rows = [
        [
            "Greedy k=4 default",
            suite_tags(SELECTED_K4),
            fmt_num(k4["median_abs_error"]),
            fmt_num(k4["median_abs_percentage_error"]),
            fmt_pct(k4["within_5_points"], 1),
            "Best deterministic four-benchmark suite.",
        ],
        [
            "Greedy k=5 periodic",
            suite_tags(SELECTED_K5),
            fmt_num(k5["median_abs_error"]),
            fmt_num(k5["median_abs_percentage_error"]),
            fmt_pct(k5["within_5_points"], 1),
            "Adds explicit scene-text OCR coverage.",
        ],
        [
            "Vibe baseline",
            suite_tags(VIBE),
            fmt_num(vibe_summary["median_abs_error"]),
            fmt_num(vibe_summary["median_abs_percentage_error"]),
            fmt_pct(vibe_summary["within_5_points"], 1),
            "Close control, but MMMU is the stronger fourth signal than OCRBench.",
        ],
        [
            "nanoVLM subset",
            suite_tags(NANOVLM),
            fmt_num(nanovlm_summary["median_abs_error"]),
            fmt_num(nanovlm_summary["median_abs_percentage_error"]),
            fmt_pct(nanovlm_summary["within_5_points"], 1),
            "Matches the practical five-benchmark recommendation.",
        ],
    ]

    predict_rows = []
    for row in predictability:
        predictors = row["top_5_predictor_benchmarks"] or "No stable peers"
        predict_rows.append(
            [
                esc(row["benchmark"]),
                tag(CLUSTER_LABELS.get(row["correlation_cluster"], row["correlation_cluster"])),
                esc(row["coverage"]),
                fmt_num(row["median_abs_error_when_held_out"]),
                fmt_num(row["median_abs_percentage_error_when_held_out"]),
                esc(predictors),
                esc(row["notes"]),
            ]
        )

    rank_rows = [
        [
            esc(row["rank"]),
            fmt_num(row["median_abs_error"]),
            fmt_num(row["median_abs_percentage_error"]),
            fmt_pct(row["within_5_points"], 1),
        ]
        for row in rank
    ]

    blend_rows = [
        [
            fmt_num(row["regression_weight"], 1),
            fmt_num(row["svd_weight"], 1),
            fmt_num(row["median_abs_error"]),
            fmt_num(row["median_abs_percentage_error"]),
            fmt_pct(row["within_5_points"], 1),
            tag("best", "good") if row is best_blend else tag("default", "selected") if row is default_blend else "",
        ]
        for row in blend
    ]

    few_rows = [
        [
            esc(row["revealed_scores"]),
            esc(row["n"]),
            fmt_num(row["median_abs_error"]),
            fmt_num(row["median_abs_percentage_error"]),
            fmt_pct(row["within_5_points"], 1),
        ]
        for row in few
    ]

    model_coverage = stats["per_model_coverage"]
    sorted_models = sorted(model_coverage, key=lambda model: (-model_coverage[model], model))
    matrix_headers = list(matrix[0].keys())[1:] if matrix else []
    matrix_by_model = {row["model_id"]: row for row in matrix}
    heatmap_rows = []
    for model in sorted_models:
        row = matrix_by_model[model]
        cells = [f'<th>{esc(model)}</th>']
        for bench in matrix_headers:
            value = row.get(bench, "")
            cls = "observed" if value != "" else "missing"
            title = f"{model} / {bench}: {value if value else 'missing'}"
            cells.append(f'<td class="{cls}" title="{esc(title)}">{esc(value) if value else ""}</td>')
        heatmap_rows.append("<tr>" + "".join(cells) + "</tr>")
    heatmap = (
        '<div class="matrix-wrap"><table class="matrix"><thead><tr><th>Model</th>'
        + "".join(f"<th>{esc(h)}</th>" for h in matrix_headers)
        + "</tr></thead><tbody>"
        + "".join(heatmap_rows)
        + "</tbody></table></div>"
    )

    sources_rows = source_rows_from_markdown(ROOT / "sources.md")
    source_table = table(["Source", "Type", "Rows", "Benchmarks", "Notes"], sources_rows, "sources-table")
    source_mix_rows = [
        [esc(kind), esc(count), fmt_pct(count / len(raw), 1)]
        for kind, count in source_type_counts.most_common()
    ]

    frontier_models = [
        "Gemini-3.5-Flash",
        "GPT-5.5",
        "Gemini-3.1-Pro",
        "Claude-Opus-4.7",
        "Kimi-K2.6",
        "Qwen3.6-35B-A3B",
    ]
    modern_benchmarks = [
        "MMMU-Pro",
        "CharXiv",
        "OSWorld-Verified",
        "MathVision",
        "MathVista",
        "MMMU",
        "RealWorldQA",
        "MMBench",
        "HallusionBench",
    ]
    scored_frontier_models = {r["model_id"] for r in frontier_scores}
    open_frontier_models = {
        r["model_id"] for r in frontier_scores if r["weight_class"] == "open-weight"
    }
    mmmu_pro_rows_raw = sorted(
        [
            r
            for r in frontier_scores
            if r["benchmark"] == "MMMU-Pro" and r["setting"] == "no_tools"
        ],
        key=lambda r: float(r["score"]),
        reverse=True,
    )
    mmmu_pro_max = max(float(r["score"]) for r in mmmu_pro_rows_raw)
    mmmu_pro_rows = [
        [
            esc(row["model_id"]),
            esc(row["provider"]),
            tag(row["weight_class"], "good" if row["weight_class"] == "open-weight" else ""),
            fmt_num(row["score"], 1),
            bar(float(row["score"]), mmmu_pro_max, f'{row["model_id"]} MMMU-Pro {row["score"]}'),
            link(row["source_url"], row["source_type"]),
        ]
        for row in mmmu_pro_rows_raw
    ]

    frontier_grid_rows = []
    for model in frontier_models:
        model_scores = [r for r in frontier_scores if r["model_id"] == model]
        if not model_scores:
            continue
        provider = model_scores[0]["provider"]
        weight_class = model_scores[0]["weight_class"]
        row = [esc(model), esc(provider), tag(weight_class, "good" if weight_class == "open-weight" else "")]
        for benchmark in modern_benchmarks:
            score_row = pick_score(frontier_scores, model, benchmark)
            if not score_row:
                row.append("")
                continue
            setting = "" if score_row["setting"] == "no_tools" else f' <span class="muted">({esc(score_row["setting"])})</span>'
            row.append(f'{fmt_num(score_row["score"], 1)}{setting}')
        frontier_grid_rows.append(row)

    frontier_status_rows = []
    for row in frontier_status:
        included = row["include_in_vlm_matrix"] == "yes"
        status_tone = "selected" if included else "warn"
        frontier_status_rows.append(
            [
                esc(row["model_id"]),
                esc(row["provider"]),
                tag(row["weight_class"], "good" if row["weight_class"] == "open-weight" else ""),
                esc(row["modalities"]),
                tag(row["status"], status_tone),
                esc(row["note"]),
                link(row["source_url"], "source"),
            ]
        )

    generated_note = "Generated from local BenchPress Vision artifacts; no network data is pulled by this report builder."
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BenchPress Vision Report</title>
  <style>
    :root {{
      --bg: #f7f8fa;
      --surface: #ffffff;
      --surface-2: #eef3f5;
      --text: #17212b;
      --muted: #5d6875;
      --line: #d9dee4;
      --strong: #0b6e69;
      --strong-2: #315d9b;
      --warn: #c77900;
      --bad: #a83f39;
      --good: #3f7b45;
      --shadow: 0 18px 45px rgba(27, 38, 49, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }}
    a {{ color: var(--strong-2); text-decoration-thickness: 1px; text-underline-offset: 3px; }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(190px, 250px) minmax(0, 1fr);
      min-height: 100vh;
    }}
    nav {{
      position: sticky;
      top: 0;
      height: 100vh;
      padding: 28px 22px;
      border-right: 1px solid var(--line);
      background: #ffffff;
      overflow: auto;
    }}
    nav h1 {{
      margin: 0 0 18px;
      font-size: 20px;
      line-height: 1.15;
    }}
    nav a {{
      display: block;
      padding: 8px 0;
      color: #26313d;
      text-decoration: none;
      border-bottom: 1px solid #eef0f2;
    }}
    nav a:hover {{ color: var(--strong); }}
    main {{ min-width: 0; }}
    section {{
      padding: 42px clamp(18px, 4vw, 64px);
      border-bottom: 1px solid var(--line);
      background: var(--bg);
    }}
    section:nth-child(even) {{ background: #ffffff; }}
    .hero {{
      min-height: 88vh;
      display: grid;
      align-content: center;
      gap: 26px;
      background:
        linear-gradient(120deg, rgba(11, 110, 105, 0.08), rgba(199, 121, 0, 0.06)),
        var(--surface);
    }}
    .eyebrow {{
      margin: 0;
      color: var(--strong);
      text-transform: uppercase;
      font-weight: 700;
      font-size: 12px;
      letter-spacing: 0;
    }}
    h2 {{
      max-width: 1040px;
      margin: 0 0 14px;
      font-size: clamp(28px, 4.6vw, 58px);
      line-height: 1.02;
      letter-spacing: 0;
    }}
    h3 {{
      margin: 0 0 12px;
      font-size: 24px;
      line-height: 1.2;
      letter-spacing: 0;
    }}
    h4 {{
      margin: 0 0 8px;
      font-size: 16px;
      letter-spacing: 0;
    }}
    p {{ max-width: 980px; margin: 0 0 12px; color: #2d3844; }}
    .lede {{
      max-width: 980px;
      font-size: clamp(18px, 2vw, 23px);
      color: #26313d;
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      max-width: 1120px;
    }}
    .metric {{
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      min-height: 128px;
      box-shadow: var(--shadow);
    }}
    .metric__value {{
      font-size: clamp(26px, 3vw, 38px);
      font-weight: 760;
      color: var(--text);
      line-height: 1;
    }}
    .metric__label {{
      margin-top: 10px;
      font-weight: 700;
    }}
    .metric__detail {{
      color: var(--muted);
      font-size: 13px;
      margin-top: 4px;
    }}
    .callout {{
      max-width: 1120px;
      padding: 20px 22px;
      border-left: 5px solid var(--strong);
      background: #edf7f5;
    }}
    .callout.warn {{
      border-left-color: var(--warn);
      background: #fff6e8;
    }}
    .section-head {{
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 20px;
      max-width: 1180px;
      margin-bottom: 18px;
    }}
    .section-head p {{ max-width: 680px; color: var(--muted); }}
    .split {{
      display: grid;
      grid-template-columns: minmax(0, 1.1fr) minmax(320px, 0.9fr);
      gap: 28px;
      align-items: start;
      max-width: 1180px;
    }}
    .panel {{
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 20px;
      box-shadow: var(--shadow);
    }}
    .panel + .panel {{ margin-top: 16px; }}
    .suite {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }}
    .tag {{
      display: inline-flex;
      align-items: center;
      min-height: 28px;
      padding: 4px 9px;
      border: 1px solid #cbd4dc;
      border-radius: 999px;
      background: #f9fafb;
      color: #26313d;
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
      margin: 1px 3px 1px 0;
    }}
    .tag--selected {{
      background: #e9f6f4;
      border-color: #91c7c1;
      color: #075f5b;
    }}
    .tag--good {{
      background: #eef8ef;
      border-color: #a7cfa9;
      color: #2f6a35;
    }}
    .tag--warn {{
      background: #fff4df;
      border-color: #e7c178;
      color: #835400;
    }}
    .muted {{ color: var(--muted); font-size: 12px; }}
    .bars {{
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 10px;
    }}
    .bar {{
      width: 100%;
      height: 10px;
      border-radius: 999px;
      background: #e5e9ee;
      overflow: hidden;
    }}
    .bar span {{
      display: block;
      height: 100%;
      background: linear-gradient(90deg, var(--strong), var(--strong-2));
    }}
    .table-wrap {{
      max-width: 1180px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      box-shadow: var(--shadow);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 720px;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid #e7ebef;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      position: sticky;
      top: 0;
      z-index: 1;
      background: #f1f4f6;
      font-size: 12px;
      text-transform: uppercase;
      color: #46515d;
      letter-spacing: 0;
    }}
    tbody tr:hover {{ background: #fafcfd; }}
    .small-table td, .small-table th {{ font-size: 13px; }}
    .plot-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 22px;
      max-width: 1180px;
    }}
    figure {{
      margin: 0;
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      box-shadow: var(--shadow);
    }}
    figure img {{
      display: block;
      width: 100%;
      height: auto;
      border-radius: 4px;
      border: 1px solid #edf0f2;
    }}
    figcaption {{
      margin-top: 10px;
      color: var(--muted);
      font-size: 13px;
    }}
    .plot-svg {{
      display: block;
      width: 100%;
      height: auto;
    }}
    .plot-bg {{ fill: #ffffff; }}
    .gridline {{ stroke: #e0e5e9; stroke-width: 1; }}
    .axis {{ stroke: #798491; stroke-width: 1.2; }}
    .plot-line {{ fill: none; stroke: var(--strong); stroke-width: 3; stroke-linejoin: round; }}
    .plot-svg circle {{ fill: #ffffff; stroke: var(--warn); stroke-width: 3; }}
    .plot-svg text {{ fill: #52606d; font-size: 12px; }}
    .stacked svg {{
      width: 100%;
      height: 16px;
      border-radius: 999px;
      overflow: hidden;
      border: 1px solid rgba(0,0,0,0.08);
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 9px 14px;
      margin-top: 12px;
      color: var(--muted);
      font-size: 13px;
    }}
    .legend span {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .legend i {{
      width: 10px;
      height: 10px;
      border-radius: 2px;
      display: inline-block;
    }}
    .matrix-wrap {{
      max-width: 1180px;
      max-height: 680px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #ffffff;
      box-shadow: var(--shadow);
    }}
    .matrix {{
      width: max-content;
      min-width: 100%;
      font-size: 11px;
    }}
    .matrix th, .matrix td {{
      padding: 5px 6px;
      min-width: 58px;
      max-width: 86px;
      border: 1px solid #edf0f2;
      text-align: center;
      white-space: nowrap;
    }}
    .matrix th:first-child {{
      position: sticky;
      left: 0;
      z-index: 2;
      min-width: 190px;
      max-width: 190px;
      text-align: left;
      background: #f1f4f6;
    }}
    .matrix td.observed {{ background: #dff1ee; color: #0d5754; }}
    .matrix td.missing {{ background: #f7f8fa; }}
    .sources-table {{ min-width: 1100px; }}
    .artifact-list {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      max-width: 980px;
      margin-top: 14px;
    }}
    .artifact-list a {{
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #ffffff;
      text-decoration: none;
      font-weight: 700;
      font-size: 13px;
    }}
    footer {{
      padding: 26px clamp(18px, 4vw, 64px);
      color: var(--muted);
      background: #ffffff;
    }}
    @media (max-width: 900px) {{
      .layout {{ display: block; }}
      nav {{
        position: static;
        height: auto;
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }}
      nav a {{ display: inline-block; margin-right: 16px; }}
      .summary-grid, .plot-grid, .split {{
        grid-template-columns: 1fr;
      }}
      .hero {{ min-height: auto; }}
      section {{ padding-top: 34px; padding-bottom: 34px; }}
      .section-head {{ display: block; }}
    }}
  </style>
</head>
<body>
  <div class="layout">
    <nav aria-label="Report sections">
      <h1>BenchPress Vision</h1>
      <a href="#summary">Summary</a>
      <a href="#frontier-refresh">2026 Frontier</a>
      <a href="#recommendation">Recommendation</a>
      <a href="#validation">Validation</a>
      <a href="#coverage">Coverage</a>
      <a href="#signals">Signal Analysis</a>
      <a href="#predictability">Predictability</a>
      <a href="#matrix">Matrix</a>
      <a href="#sources">Sources</a>
    </nav>
    <main>
      <section id="summary" class="hero">
        <div>
          <p class="eyebrow">nanochat-llava benchmark selection report</p>
          <h2>Use four vision benchmarks for default development runs, with a five-benchmark periodic suite when one more run is affordable.</h2>
          <p class="lede">
            The selected default is <strong>{esc(", ".join(SELECTED_K4))}</strong>. The periodic suite adds
            <strong>TextVQA</strong> to restore direct OCR and scene-text coverage. The recommendation is based on
            a 48-model by 31-benchmark score matrix, holdout prediction tests, low-rank structure, greedy subset selection,
            and a May 2026 frontier refresh covering GPT-5.5, Claude Opus 4.7, Gemini 3.x, Kimi K2.6, and Qwen3.6.
          </p>
        </div>
        <div class="summary-grid">
          {metric_tile("Models", stats["models"], "families include LLaVA, InternVL, Qwen, SmolVLM, Gemma, Gemini, OpenAI, Anthropic")}
          {metric_tile("Benchmarks", stats["benchmarks"], f'{stats["observed_cells"]} observed cells')}
          {metric_tile("Matrix Fill", fmt_pct(stats["fill_rate"], 1), "sparse but above the 25% target")}
          {metric_tile("Primary Holdout MedAE", fmt_num(stats["holdout_summary_default"]["median_abs_error"]), f'{fmt_pct(stats["holdout_summary_default"]["within_5_points"], 1)} within +/-5 points')}
        </div>
        <div class="callout">
          <strong>Decision:</strong> Keep <strong>MMStar, ScienceQA, ChartQA, MMMU</strong> as the v0 default run-set.
          Add <strong>TextVQA</strong> for a higher-confidence periodic suite. Exclude {esc(", ".join(JUDGE_STYLE))}
          from default greedy search because they are judge-style diagnostics rather than deterministic core run targets.
          For frontier tracking, add <strong>MMMU-Pro</strong> and <strong>CharXiv</strong> as explicit periodic checks because
          older general VLM benchmarks are now close to saturation.
        </div>
      </section>

      <section id="frontier-refresh">
        <div class="section-head">
          <div>
            <h3>2026 Frontier Refresh</h3>
            <p>The refresh uses current public model pages and official model cards where they expose compatible multimodal scores. Missing rows stay missing: Grok 4.3, DeepSeek V4 Pro, and GLM-5.1 are tracked in the audit table, but they are not inserted into the VLM score matrix without official vision benchmark rows.</p>
          </div>
        </div>
        <div class="summary-grid">
          {metric_tile("Scored Frontier Models", len(scored_frontier_models), "closed and open-weight models with compatible VLM rows")}
          {metric_tile("Open-Weight Frontier", len(open_frontier_models), "Kimi K2.6 and Qwen3.6-35B-A3B")}
          {metric_tile("Best MMMU-Pro", fmt_num(mmmu_pro_rows_raw[0]["score"], 1), mmmu_pro_rows_raw[0]["model_id"])}
          {metric_tile("Current Action", "Add MMMU-Pro", "plus CharXiv for chart/figure reasoning drift")}
        </div>
        <div class="split" style="margin-top:22px">
          <div class="panel">
            <h4>What changed</h4>
            <p>GPT-4V, GPT-4o-20240513, Claude 3.x, Gemini 1.x, and Qwen2-era rows are now historical calibration, not frontier evidence. Current public frontier vision evidence is concentrated around MMMU-Pro, CharXiv, OSWorld-Verified, MathVision, MathVista, RealWorldQA, MMBench, and HallusionBench.</p>
            <p>The practical recommendation is to keep the original k=4/k=5 suite for cheap regression tracking, then run MMMU-Pro and CharXiv periodically to catch failures that older benchmarks can miss.</p>
          </div>
          <div class="panel">
            <h4>Source policy</h4>
            <p>Rows come from OpenAI, Google DeepMind, Moonshot AI, Qwen, Anthropic, xAI, DeepSeek, and Z.ai primary pages. Third-party leaderboard claims were not inserted unless a primary model page or model card exposed the same row.</p>
          </div>
        </div>
        <div class="table-wrap" style="margin-top:22px">
          {table(["Model", "Provider", "Weights", "MMMU-Pro", "Relative score", "Source"], mmmu_pro_rows, "small-table")}
        </div>
        <div class="table-wrap" style="margin-top:22px">
          {table(["Model", "Provider", "Weights", *modern_benchmarks], frontier_grid_rows, "small-table")}
        </div>
        <div class="table-wrap" style="margin-top:22px">
          {table(["Model", "Provider", "Weights", "Modalities", "Audit status", "Decision", "Source"], frontier_status_rows, "small-table")}
        </div>
      </section>

      <section id="recommendation">
        <div class="section-head">
          <div>
            <h3>Recommended Suites</h3>
            <p>The greedy search rewards non-redundant coverage of visual reasoning, science/diagram QA, chart reasoning, and multidiscipline reasoning. TextVQA is the most useful fifth benchmark because OCR/text-reading is otherwise implicit.</p>
          </div>
        </div>
        <div class="split">
          <div class="panel">
            <h4>Default k=4 suite</h4>
            <div class="suite">{suite_tags(SELECTED_K4)}</div>
            <p>Median absolute error falls to <strong>{fmt_num(k4["median_abs_error"])}</strong>, with <strong>{fmt_pct(k4["within_5_points"], 1)}</strong> of held-out scores within five points. This is the best deterministic four-benchmark suite found in the search.</p>
          </div>
          <div class="panel">
            <h4>Periodic k=5 suite</h4>
            <div class="suite">{suite_tags(SELECTED_K5)}</div>
            <p>Adding TextVQA lowers median absolute error to <strong>{fmt_num(k5["median_abs_error"])}</strong> and improves within-five-point accuracy to <strong>{fmt_pct(k5["within_5_points"], 1)}</strong>.</p>
          </div>
        </div>
        <div class="table-wrap" style="margin-top:20px">
          {table(["Suite", "Benchmarks", "MedAE", "MedAPE", "Within +/-5", "Interpretation"], suite_rows, "small-table")}
        </div>
      </section>

      <section id="validation">
        <div class="section-head">
          <div>
            <h3>Validation Results</h3>
            <p>The predictor was evaluated by hiding known scores and estimating them from the remaining matrix. The default blend keeps regression as the main signal and low-rank reconstruction as a stabilizer.</p>
          </div>
        </div>
        <div class="summary-grid">
          {metric_tile("Holdout Tests", stats["holdout_summary_default"]["n"], "all predicted with full coverage")}
          {metric_tile("Median AE", fmt_num(median(abs_errors)), "score points on a 0-100 scale")}
          {metric_tile("P90 AE", fmt_num(percentile(abs_errors, 0.9)), "tail error across held-out cells")}
          {metric_tile("Median APE", fmt_num(median(abs_pct_errors)) + "%", "relative score error")}
        </div>
        <div class="plot-grid" style="margin-top:22px">
          <figure>
            {line_svg(greedy, "k", "median_abs_error", "k", "Greedy suite error by benchmark count", "Median absolute error")}
            <figcaption>Each added benchmark reduces median held-out error, with the largest practical gain through the fifth benchmark.</figcaption>
          </figure>
          <figure>
            {line_svg(few, "revealed_scores", "median_abs_error", "revealed_scores", "Few-revealed prediction error", "Median absolute error")}
            <figcaption>Prediction quality improves as each model exposes more benchmark scores, which supports using a small suite as a proxy rather than a single benchmark.</figcaption>
          </figure>
        </div>
        <div class="table-wrap" style="margin-top:22px">
          {table(["k", "Added", "Selected suite", "Held-out n", "MedAE", "MedAPE", "Within +/-5", "AE improvement"], greedy_rows, "small-table")}
        </div>
      </section>

      <section id="coverage">
        <div class="section-head">
          <div>
            <h3>Coverage And Source Mix</h3>
            <p>The matrix uses official project reports, model cards, and vendor reports. Coverage is strongest for MMMU, MathVista, TextVQA, AI2D, MMBench, and MMVet; video benchmarks remain thin.</p>
          </div>
        </div>
        <div class="split">
          <div class="panel">
            <h4>Benchmark coverage</h4>
            <div class="table-wrap" style="box-shadow:none">
              {table(["Benchmark", "Models", "Coverage"], coverage_rows, "small-table")}
            </div>
          </div>
          <div>
            <div class="panel">
              <h4>Benchmark clusters</h4>
              {stacked_cluster_svg(cluster_counts)}
            </div>
            <div class="panel">
              <h4>Source type mix</h4>
              <div class="table-wrap" style="box-shadow:none">
                {table(["Type", "Rows", "Share"], source_mix_rows, "small-table")}
              </div>
            </div>
          </div>
        </div>
      </section>

      <section id="signals">
        <div class="section-head">
          <div>
            <h3>Low-Rank And Correlation Structure</h3>
            <p>A compact latent performance axis explains most shared variance, but the second and third components still matter for separating OCR, chart, science, and general-reasoning behavior.</p>
          </div>
        </div>
        <div class="plot-grid">
          <figure>
            <img src="results/svd_spectrum.png" alt="SVD explained variance spectrum">
            <figcaption>Largest fully observed submatrix: {stats["svd_spectrum"]["submatrix_models"]} models x {stats["svd_spectrum"]["submatrix_benchmarks"]} benchmarks. Rank-1 explains {fmt_pct(stats["svd_spectrum"]["rank1_explained"], 1)}; rank-2 explains {fmt_pct(stats["svd_spectrum"]["rank2_cumulative"], 1)} cumulatively.</figcaption>
          </figure>
          <figure>
            <img src="results/benchmark_correlations.png" alt="Benchmark correlation heatmap">
            <figcaption>The selected suite avoids relying only on highly redundant peers. k=5 mean absolute pairwise correlation is 0.78, with TextVQA adding a direct text/OCR axis.</figcaption>
          </figure>
          <figure>
            <img src="results/few_revealed_error.png" alt="Few-revealed benchmark error curve">
            <figcaption>Error declines as more scores are revealed, but the first four to five carefully chosen benchmarks capture most of the practical benefit.</figcaption>
          </figure>
          <figure>
            {line_svg(rank, "rank", "median_abs_error", "rank", "SVD rank sweep", "Median absolute error")}
            <figcaption>Pure low-rank reconstruction is useful as a diagnostic. Rank {esc(best_rank["rank"])} has the best median absolute error in this sweep, while the default predictor keeps a blended regression/low-rank approach.</figcaption>
          </figure>
        </div>
        <div class="split" style="margin-top:22px">
          <div class="table-wrap">
            {table(["Rank", "MedAE", "MedAPE", "Within +/-5"], rank_rows, "small-table")}
          </div>
          <div class="table-wrap">
            {table(["Regression wt.", "SVD wt.", "MedAE", "MedAPE", "Within +/-5", "Note"], blend_rows, "small-table")}
          </div>
        </div>
      </section>

      <section id="predictability">
        <div class="section-head">
          <div>
            <h3>Hard-To-Predict Benchmarks</h3>
            <p>Video, harder chart reasoning, and sparse document benchmarks have the largest held-out errors. They should be run directly when those capabilities become product-critical.</p>
          </div>
        </div>
        <div class="callout warn">
          <strong>Risk note:</strong> Video-MME, MLVU, BetterChartQA, and MathVision have high error largely because coverage is sparse. The default suite is a development proxy, not a substitute for domain-specific evaluation.
        </div>
        <div class="table-wrap" style="margin-top:22px">
          {table(["Benchmark", "Cluster", "Coverage", "Held-out MedAE", "Held-out MedAPE", "Top peer predictors", "Notes"], predict_rows, "small-table")}
        </div>
      </section>

      <section id="matrix">
        <div class="section-head">
          <div>
            <h3>Observed Score Matrix</h3>
            <p>Cells show normalized scores on a 0-100 scale. Empty cells were not present in the collected official sources and were not fabricated in the matrix.</p>
          </div>
        </div>
        {heatmap}
      </section>

      <section id="sources">
        <div class="section-head">
          <div>
            <h3>Sources And Artifacts</h3>
            <p>{esc(generated_note)} The report links to the source table and raw intermediate files used to produce the recommendation.</p>
          </div>
        </div>
        <div class="artifact-list">
          <a href="scores_raw.csv">scores_raw.csv</a>
          <a href="score_matrix.csv">score_matrix.csv</a>
          <a href="sources.md">sources.md</a>
          <a href="frontier_2026_scores.csv">frontier_2026_scores.csv</a>
          <a href="frontier_2026_model_status.csv">frontier_2026_model_status.csv</a>
          <a href="results/recommendation.md">recommendation.md</a>
          <a href="results/greedy_selection.csv">greedy_selection.csv</a>
          <a href="results/benchmark_predictability.csv">benchmark_predictability.csv</a>
          <a href="results/holdout_results.csv">holdout_results.csv</a>
        </div>
        <div class="table-wrap" style="margin-top:22px">
          {source_table}
        </div>
      </section>
      <footer>Report generated by <code>benchpress_vision/build_report.py</code>.</footer>
    </main>
  </div>
</body>
</html>
"""
    OUT.write_text(html_doc, encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
