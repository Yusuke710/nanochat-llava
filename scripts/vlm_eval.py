"""
Small verifier eval for nanochat-llava v0.

This is intentionally a subset runner, not a full leaderboard harness. It loads
MMStar, ScienceQA, ChartQA, MMMU, and TextVQA through Hugging Face datasets,
runs naive image-conditioned generation, and writes JSON scores so stage-to-stage
progress or failed visual controls are visible in logs.
"""

import argparse
import ast
import json
import os
import re
import string

import torch

from nanochat.checkpoint_manager import load_model
from nanochat.common import compute_cleanup, compute_init, get_base_dir, print0, autodetect_device_type
from nanochat.vision import (
    IMAGE_MARKER,
    SIGLIP_MODEL_ID,
    SigLIPPooledFeatureExtractor,
    encode_with_image_markers,
    ensure_hf_nanochat_checkpoint,
    generate_vision,
    load_vlm_checkpoint,
)


BENCHMARKS = {
    "mmstar": {"dataset": "Lin-Chen/MMStar", "split": "val"},
    "scienceqa": {"dataset": "derek-thomas/ScienceQA", "split": "validation"},
    "chartqa": {"dataset": "HuggingFaceM4/ChartQA", "split": "val"},
    "mmmu": {"dataset": "MMMU/MMMU", "split": "validation", "config": "Accounting"},
    "textvqa": {"dataset": "lmms-lab/textvqa", "split": "validation"},
}
MODEL_SOURCE = "sft"


def result_key(name, config=None, default_config=None):
    if config is None or config == default_config:
        return name
    suffix = re.sub(r"[^0-9A-Za-z]+", "_", str(config)).strip("_")
    return f"{name}_{suffix}" if suffix else name


def benchmark_specs(names, mmmu_configs="Accounting"):
    specs = []
    for raw_name in names:
        name, sep, inline_config = raw_name.partition(":")
        name = name.strip().lower()
        config = inline_config.strip() if sep else None
        if name not in BENCHMARKS:
            raise ValueError(f"unknown benchmark {raw_name!r}; expected one of {sorted(BENCHMARKS)}")
        if name == "mmmu":
            configs = [config] if config else [c.strip() for c in str(mmmu_configs).split(",") if c.strip()]
            if not configs:
                configs = [BENCHMARKS["mmmu"]["config"]]
            for cfg in configs:
                specs.append({
                    "name": name,
                    "config": cfg,
                    "key": result_key(name, cfg, default_config=BENCHMARKS["mmmu"]["config"] if not sep and len(configs) == 1 else None),
                })
        else:
            specs.append({"name": name, "config": config, "key": result_key(name, config)})
    return specs


SPECIAL_TOKEN_RE = re.compile(r"<\|[^>]*?\|>")


def strip_special_tokens(text):
    return SPECIAL_TOKEN_RE.sub("", str(text))


def normalize_answer(text):
    if isinstance(text, list):
        text = text[0] if text else ""
    text = strip_special_tokens(text).lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def exact_or_choice_match(pred, answers):
    pred_text = strip_special_tokens(pred)
    pred_norm = normalize_answer(pred)
    if not isinstance(answers, list):
        answers = [answers]
    answer_norms = [normalize_answer(a) for a in answers]
    pred_tokens = pred_norm.split()
    for answer in answer_norms:
        if not answer:
            continue
        if pred_norm == answer:
            return True
        if len(answer) == 1 and answer.isalpha():
            # A bare "a" appears constantly as an article, so require either
            # letter-only output or an explicit option/answer pattern.
            if re.search(rf"^\s*{re.escape(answer)}\s*(?:[).:,-]|$)", pred_text, flags=re.IGNORECASE):
                return True
            if re.search(
                rf"\b(?:answer|option|choice)(?:\s+letter)?\s*(?:is|:)?\s*{re.escape(answer)}\b",
                pred_text,
                flags=re.IGNORECASE,
            ):
                return True
            continue
        if len(answer) == 1 and answer.isdigit():
            if re.search(rf"^\s*{re.escape(answer)}\s*(?:[).:,-]|$)", pred_text):
                return True
            if re.search(
                rf"\b(?:answer|option|choice)(?:\s+letter)?\s*(?:is|:)?\s*{re.escape(answer)}\b",
                pred_text,
                flags=re.IGNORECASE,
            ):
                return True
            continue
        if len(answer.split()) == 1 and answer in pred_tokens:
            return True
        if answer in pred_norm:
            return True
    return False


def visual_control_passes(score, zero_image_score, margin=0.0):
    return score > zero_image_score + margin


def predictions_differ(left, right):
    return normalize_answer(left) != normalize_answer(right)


def as_list(value):
    if isinstance(value, list):
        return value
    return [value]


def get_first(record, keys, default=""):
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return default


def coerce_options(options):
    if options in (None, "", []):
        return []
    if isinstance(options, (list, tuple)):
        return list(options)
    if isinstance(options, str):
        try:
            parsed = ast.literal_eval(options)
        except (SyntaxError, ValueError):
            return options
        if isinstance(parsed, (list, tuple)):
            return list(parsed)
    return options


def get_image(record):
    for _, value in record.items():
        if hasattr(value, "convert"):
            return value.convert("RGB")
    raise KeyError("no PIL image field found")


def format_options(options):
    options = coerce_options(options)
    if options in (None, "", []):
        return ""
    if isinstance(options, str):
        return f"\nOptions: {options}"
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return "\n" + "\n".join(f"{letters[i]}. {opt}" for i, opt in enumerate(options))


def parse_inline_options(text):
    match = re.search(r"Options:\s*(.+)$", str(text), flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return []
    raw = " ".join(match.group(1).split())
    pairs = re.findall(r"(?:^|,\s*)([A-Z])\s*[:.]\s*(.*?)(?=,\s*[A-Z]\s*[:.]|$)", raw)
    if not pairs:
        return []
    pairs = sorted(pairs, key=lambda item: ord(item[0]) - ord("A"))
    return [text.strip().rstrip(",") for _, text in pairs]


def make_prompt(record):
    question = get_first(record, ["question", "query", "text", "hint", "Question"])
    question = re.sub(r"<image\s*\d*>", "", str(question)).strip()
    choices = get_first(record, ["choices", "options", "multiple_choices"], default=None)
    structured_choices = coerce_options(choices)
    has_choices = isinstance(structured_choices, list) and len(structured_choices) > 0
    if not has_choices:
        has_choices = len(parse_inline_options(question)) > 0
    answer_prompt = "Answer with the option letter only:" if has_choices else "Answer:"
    return f"{IMAGE_MARKER}\n{question}{format_options(choices)}\n{answer_prompt}"


def get_answers(record):
    answer = get_first(record, ["answer", "label", "answers", "gt_answer", "response"], default="")
    if isinstance(answer, dict):
        answer = list(answer.values())
    answers = as_list(answer)
    choices = coerce_options(get_first(record, ["choices", "options", "multiple_choices"], default=None))
    if not isinstance(choices, list) or not choices:
        choices = parse_inline_options(get_first(record, ["question", "query", "text", "hint", "Question"], default=""))
    if isinstance(choices, list):
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        expanded = []
        for item in answers:
            idx = None
            if isinstance(item, int):
                idx = item
            elif isinstance(item, str) and item.strip().isdigit():
                idx = int(item.strip())
            elif isinstance(item, str) and len(item.strip()) == 1 and item.strip().upper() in letters:
                idx = letters.index(item.strip().upper())
            if idx is not None and 0 <= idx < len(choices):
                expanded.extend([letters[idx], choices[idx]])
            elif isinstance(item, str) and item in choices:
                expanded.extend([item, letters[choices.index(item)]])
            else:
                expanded.append(item)
        answers = expanded
    return answers


def render_prompt_tokens(tokenizer, prompt):
    bos = tokenizer.get_bos_token_id()
    user_start, user_end = tokenizer.encode_special("<|user_start|>"), tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")
    return [bos, user_start] + encode_with_image_markers(tokenizer, prompt) + [user_end, assistant_start]


def preview_text(text, max_chars=220):
    text = " ".join(str(text).split())
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def make_result_sample(
    record,
    index,
    pred,
    answers,
    is_correct,
    control_pred=None,
    control_is_correct=None,
):
    sample = {
        "index": index,
        "prompt": preview_text(make_prompt(record)),
        "prediction": preview_text(pred),
        "prediction_correct": bool(is_correct),
        "answers": [preview_text(a, 80) for a in as_list(answers)],
    }
    if control_pred is not None:
        sample["zero_image_prediction"] = preview_text(control_pred)
        sample["zero_image_correct"] = bool(control_is_correct)
        sample["prediction_changed"] = predictions_differ(pred, control_pred)
    return sample


def load_benchmark(name, config=None):
    from datasets import load_dataset

    cfg = BENCHMARKS[name]
    config = config if config is not None else cfg.get("config")
    if config is not None:
        return load_dataset(cfg["dataset"], config, split=cfg["split"])
    return load_dataset(cfg["dataset"], split=cfg["split"])


def load_vlm(args, device):
    base_dir = get_base_dir()
    if args.hf_checkpoint:
        ensure_hf_nanochat_checkpoint(args.hf_checkpoint, base_dir, model_tag=args.model_tag, source=MODEL_SOURCE)
    model, tokenizer, _ = load_model(MODEL_SOURCE, device, phase="eval", model_tag=args.model_tag, step=args.model_step)
    model_state, projector, _, vlm_meta = load_vlm_checkpoint(args.checkpoint_dir, args.checkpoint_step, device, load_optimizer=False)
    model.load_state_dict(model_state, strict=True, assign=True)
    model.eval()
    projector.eval()
    return model, projector, tokenizer, vlm_meta


def main():
    parser = argparse.ArgumentParser(description="Run nanochat-llava verifier eval subsets")
    parser.add_argument("--benchmarks", default="mmstar,scienceqa,chartqa,mmmu,textvqa")
    parser.add_argument("--mmmu-configs", default="Accounting", help="comma-separated MMMU configs used when --benchmarks includes bare mmmu")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--max-scan", type=int, default=0, help="max records to scan per benchmark; 0 means limit*20")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--hf-checkpoint", default="karpathy/nanochat-d32")
    parser.add_argument("--model-tag", default="d32")
    parser.add_argument("--model-step", type=int, default=None)
    parser.add_argument("--device-type", default="", choices=["", "cuda", "cpu", "mps"])
    parser.add_argument("--siglip-model-id", default=SIGLIP_MODEL_ID)
    parser.add_argument("--siglip-cache-dir", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--print-samples", type=int, default=0, help="print and store up to N generations per benchmark")
    parser.add_argument("--control", action="store_true", help="also score zero-image control")
    parser.add_argument("--control-margin", type=float, default=0.0, help="minimum score margin for image-conditioned run to pass zero-image control")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    _, _, _, _, device = compute_init(device_type)
    model, projector, tokenizer, vlm_meta = load_vlm(args, device)
    siglip_cache_dir = args.siglip_cache_dir or os.environ.get("NANOCHAT_SIGLIP_CACHE_DIR")
    extractor = SigLIPPooledFeatureExtractor(args.siglip_model_id, device=device, cache_dir=siglip_cache_dir, verbose=True)
    gpu_name = torch.cuda.get_device_name(0) if device_type == "cuda" else device_type
    benchmark_names = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    specs = benchmark_specs(benchmark_names, args.mmmu_configs)
    print0(
        f"VLM eval | GPU: {gpu_name} | checkpoint: {args.checkpoint_dir}@{args.checkpoint_step} | "
        f"benchmarks: {','.join(s['key'] for s in specs)} | limit: {args.limit} | max_scan: {args.max_scan} | "
        f"control: {args.control} margin: {args.control_margin}"
    )

    results = {
        "checkpoint_dir": args.checkpoint_dir,
        "checkpoint_step": args.checkpoint_step,
        "gpu": gpu_name,
        "requested_benchmarks": benchmark_names,
        "resolved_benchmarks": specs,
        "mmmu_configs": args.mmmu_configs,
        "limit": args.limit,
        "max_scan": args.max_scan,
        "control": args.control,
        "control_margin": args.control_margin,
        "max_new_tokens": args.max_new_tokens,
        "print_samples": args.print_samples,
        "siglip_model_id": args.siglip_model_id,
        "checkpoint_meta": vlm_meta,
        "benchmarks": {},
    }
    for spec in specs:
        name = spec["name"]
        row_key = spec["key"]
        ds = load_benchmark(name, spec.get("config"))
        correct = 0
        control_correct = 0
        image_only_correct = 0
        zero_only_correct = 0
        both_correct = 0
        both_wrong = 0
        prediction_changed = 0
        total = 0
        skipped = 0
        scanned = 0
        samples = []
        max_scan = args.max_scan if args.max_scan > 0 else (args.limit * 20 if args.limit > 0 else 0)
        for rec in ds:
            if args.limit > 0 and total >= args.limit:
                break
            if max_scan > 0 and scanned >= max_scan:
                break
            scanned += 1
            try:
                image = get_image(rec)
                prompt_tokens = render_prompt_tokens(tokenizer, make_prompt(rec))
                feats = extractor([image])
                pred_tokens = generate_vision(model, projector, tokenizer, prompt_tokens, feats, max_tokens=args.max_new_tokens, temperature=0.0)
                pred = tokenizer.decode(pred_tokens)
                answers = get_answers(rec)
                is_correct = exact_or_choice_match(pred, answers)
                correct += int(is_correct)
                control_pred = None
                control_is_correct = None
                if args.control:
                    control_tokens = generate_vision(model, projector, tokenizer, prompt_tokens, torch.zeros_like(feats), max_tokens=args.max_new_tokens, temperature=0.0)
                    control_pred = tokenizer.decode(control_tokens)
                    control_is_correct = exact_or_choice_match(control_pred, answers)
                    control_correct += int(control_is_correct)
                    image_only_correct += int(is_correct and not control_is_correct)
                    zero_only_correct += int(control_is_correct and not is_correct)
                    both_correct += int(is_correct and control_is_correct)
                    both_wrong += int(not is_correct and not control_is_correct)
                    prediction_changed += int(predictions_differ(pred, control_pred))
                if len(samples) < args.print_samples:
                    sample = make_result_sample(
                        rec,
                        scanned - 1,
                        pred,
                        answers,
                        is_correct,
                        control_pred=control_pred,
                        control_is_correct=control_is_correct,
                    )
                    samples.append(sample)
                    print0(
                        f"{row_key} sample {len(samples)} | pred={sample['prediction']!r}"
                        + (f" | zero={sample['zero_image_prediction']!r}" if control_pred is not None else "")
                        + f" | answers={sample['answers']}"
                    )
                total += 1
            except Exception as e:
                skipped += 1
                print0(f"{row_key}: skipped one record: {type(e).__name__}: {e}")
        score = correct / max(total, 1)
        row = {"benchmark": name, "config": spec.get("config"), "n": total, "skipped": skipped, "scanned": scanned, "score": score}
        if args.control:
            row["zero_image_score"] = control_correct / max(total, 1)
            row["visual_control_pass"] = visual_control_passes(row["score"], row["zero_image_score"], args.control_margin)
            row["image_only_correct"] = image_only_correct
            row["zero_only_correct"] = zero_only_correct
            row["both_correct"] = both_correct
            row["both_wrong"] = both_wrong
            row["prediction_changed"] = prediction_changed
            row["prediction_changed_rate"] = prediction_changed / max(total, 1)
        if samples:
            row["samples"] = samples
        results["benchmarks"][row_key] = row
        print0(
            f"{row_key}: score={score:.4f} n={total} skipped={skipped} scanned={scanned}"
            + (
                f" zero_image={row['zero_image_score']:.4f} control_pass={row['visual_control_pass']}"
                f" image_only={row['image_only_correct']} zero_only={row['zero_only_correct']}"
                f" changed={row['prediction_changed']}/{total}"
                if args.control
                else ""
            )
        )

    if args.out:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print0(f"wrote {args.out}")
    compute_cleanup()


if __name__ == "__main__":
    main()
