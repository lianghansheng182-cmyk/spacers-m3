#!/usr/bin/env python3
"""Write the high-resolution tiling ablation report."""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

XLRS_SINGLE = REPO_ROOT / "results" / "xlrs_bench" / "partial_44shards_int4"
XLRS_TILING = REPO_ROOT / "results" / "highres" / "xlrs_tiling_44shards_int4"
MME_SINGLE = REPO_ROOT / "results" / "mme_realworld_rs"
MME_TILING = REPO_ROOT / "results" / "highres" / "mme_rs_tiling_full_int4"
OUTPUT = REPO_ROOT / "results" / "highres" / "highres_tiling_ablation.md"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def bucket_from_mp(mp: float) -> str:
    if mp < 4:
        return "<4MP"
    if mp <= 16:
        return "4-16MP"
    return ">16MP"


def pct(value: float) -> str:
    return f"{value:.6f}"


def flatten_mme_category(metrics: dict[str, Any]) -> dict[str, float]:
    rs = metrics["by_task"]["Perception"]["Remote Sensing"]
    return {
        category: values["true"] / (values["true"] + values["false"])
        for category, values in rs.items()
    }


def compare_xlrs() -> dict[str, Any]:
    single_summary = load_json(XLRS_SINGLE / "summary.json")
    tiling_summary = load_json(XLRS_TILING / "summary.json")
    single_rows = load_jsonl(XLRS_SINGLE / "predictions.jsonl")
    tiling_rows = load_jsonl(XLRS_TILING / "predictions.jsonl")
    if len(single_rows) != len(tiling_rows):
        raise ValueError("XLRS single and tiling row counts differ")

    flips = Counter()
    bucket_stats: dict[str, Counter[str]] = defaultdict(Counter)
    category_stats: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[dict[str, Any]]] = {"single_wrong_tiling_correct": [], "single_correct_tiling_wrong": []}
    for single, tiling in zip(single_rows, tiling_rows):
        if single["row_number"] != tiling["row_number"]:
            raise ValueError("XLRS row order mismatch")
        single_correct = bool(single["correct"])
        tiling_correct = bool(tiling["correct"])
        key = "same"
        if not single_correct and tiling_correct:
            key = "single_wrong_tiling_correct"
        elif single_correct and not tiling_correct:
            key = "single_correct_tiling_wrong"
        flips[key] += 1
        mp = tiling["image"]["original_pixels"] / 1_000_000
        bucket = bucket_from_mp(mp)
        bucket_stats[bucket]["count"] += 1
        bucket_stats[bucket]["single_correct"] += int(single_correct)
        bucket_stats[bucket]["tiling_correct"] += int(tiling_correct)
        category = str(tiling["category"])
        category_stats[category]["count"] += 1
        category_stats[category]["single_correct"] += int(single_correct)
        category_stats[category]["tiling_correct"] += int(tiling_correct)
        if key in examples and len(examples[key]) < 12:
            examples[key].append(
                {
                    "row_number": tiling["row_number"],
                    "category": category,
                    "size": f"{tiling['image']['original_width']}x{tiling['image']['original_height']}",
                    "question": str(tiling["question"])[:120],
                    "gt": tiling["target_answer"],
                    "single": single["parsed_answer"] or "NA",
                    "tiling": tiling["parsed_answer"] or "NA",
                }
            )
    return {
        "single_summary": single_summary,
        "tiling_summary": tiling_summary,
        "flips": flips,
        "bucket_stats": bucket_stats,
        "category_stats": category_stats,
        "examples": examples,
    }


def compare_mme() -> dict[str, Any]:
    single_summary = load_json(MME_SINGLE / "summary.json")
    tiling_summary = load_json(MME_TILING / "summary.json")
    single_rows = load_json(MME_SINGLE / "predictions.json")
    tiling_rows = load_json(MME_TILING / "predictions.json")
    single_by_id = {row["Question_id"]: row for row in single_rows}
    flips = Counter()
    bucket_stats: dict[str, Counter[str]] = defaultdict(Counter)
    category_stats: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[dict[str, Any]]] = {"single_wrong_tiling_correct": [], "single_correct_tiling_wrong": []}
    for tiling in tiling_rows:
        single = single_by_id[tiling["Question_id"]]
        single_correct = single["Parsed output"] == single["Ground truth"]
        tiling_correct = tiling["Parsed output"] == tiling["Ground truth"]
        key = "same"
        if not single_correct and tiling_correct:
            key = "single_wrong_tiling_correct"
        elif single_correct and not tiling_correct:
            key = "single_correct_tiling_wrong"
        flips[key] += 1
        pixels = tiling["Image metadata"]["original_pixels"]
        bucket = bucket_from_mp(pixels / 1_000_000)
        bucket_stats[bucket]["count"] += 1
        bucket_stats[bucket]["single_correct"] += int(single_correct)
        bucket_stats[bucket]["tiling_correct"] += int(tiling_correct)
        category = str(tiling["Category"])
        category_stats[category]["count"] += 1
        category_stats[category]["single_correct"] += int(single_correct)
        category_stats[category]["tiling_correct"] += int(tiling_correct)
        if key in examples and len(examples[key]) < 12:
            examples[key].append(
                {
                    "question_id": tiling["Question_id"],
                    "category": category,
                    "size": f"{tiling['Image metadata']['original_width']}x{tiling['Image metadata']['original_height']}",
                    "question": str(tiling["Text"])[:120],
                    "gt": tiling["Ground truth"],
                    "single": single["Parsed output"] or "NA",
                    "tiling": tiling["Parsed output"] or "NA",
                }
            )
    return {
        "single_summary": single_summary,
        "tiling_summary": tiling_summary,
        "flips": flips,
        "bucket_stats": bucket_stats,
        "category_stats": category_stats,
        "examples": examples,
    }


def add_stats_table(lines: list[str], title: str, stats: dict[str, Counter[str]]) -> None:
    lines.append(title)
    lines.append("")
    lines.append("| Group | Samples | Single acc | Tiling acc | Delta |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for name, counter in sorted(stats.items()):
        count = counter["count"]
        single_acc = counter["single_correct"] / count if count else 0.0
        tiling_acc = counter["tiling_correct"] / count if count else 0.0
        lines.append(f"| {name} | {count} | {single_acc:.6f} | {tiling_acc:.6f} | {tiling_acc - single_acc:+.6f} |")
    lines.append("")


def add_examples(lines: list[str], title: str, examples: list[dict[str, Any]]) -> None:
    lines.append(title)
    lines.append("")
    lines.append("| ID | Category | Size | GT | Single | Tiling | Question |")
    lines.append("| --- | --- | ---: | --- | --- | --- | --- |")
    for item in examples:
        ident = item.get("row_number") or item.get("question_id")
        question = str(item["question"]).replace("|", "/")
        lines.append(
            f"| {ident} | {item['category']} | {item['size']} | {item['gt']} | "
            f"{item['single']} | {item['tiling']} | {question} |"
        )
    lines.append("")


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    xlrs = compare_xlrs()
    mme = compare_mme()
    xs = xlrs["single_summary"]
    xt = xlrs["tiling_summary"]
    ms = mme["single_summary"]
    mt = mme["tiling_summary"]
    xlrs_acc_delta = xt["metrics"]["accuracy_micro"] - xs["metrics"]["accuracy_micro"]
    mme_acc_delta = mt["metrics"]["accuracy"] - ms["metrics"]["accuracy"]
    xlrs_latency_mult = xt["latency_mean_s"] / xs["latency_mean_s"]
    mme_latency_mult = mt["latency_mean_s"] / ms["latency_mean_s"]
    xlrs_vram_delta = xt["peak_vram_mb"] - xs["peak_vram_mb"]
    mme_vram_delta = mt["peak_vram_mb"] - ms["peak_vram_mb"]

    lines: list[str] = [
        "# High-Resolution Multi-View Tiling Ablation",
        "",
        "Created: 2026-08-25",
        "",
        "## Problem",
        "",
        "Ultra-high-resolution remote-sensing images are heavily downscaled when they are forced into a fixed visual-token budget. Small objects, counts, colors, and local spatial details can be lost in the global view.",
        "",
        "## Method",
        "",
        "The tested mechanism is fixed `global + 2x2 local tiles`: one complete global overview plus top-left, top-right, bottom-left, and bottom-right crops of the same source image. The model is unchanged: Qwen3-VL-2B-Instruct + LoRA v1 + INT4 NF4.",
        "",
        "The prompt explicitly states that the five images are multiple views of the same remote-sensing scene and that the local crops are only for recovering details. Ground truth and evaluators are unchanged.",
        "",
        "## XLRS A/B Result",
        "",
        "| Mode | Samples | Accuracy | Parse failures | Mean latency s | P95 latency s | Peak VRAM MB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| Single-view | {xs['sample_count']} | {xs['metrics']['accuracy_micro']:.6f} | {xs['metrics']['parse_failures']} | {xs['latency_mean_s']:.6f} | {xs['latency_p95_s']:.6f} | {xs['peak_vram_mb']:.2f} |",
        f"| Global+2x2 | {xt['sample_count']} | {xt['metrics']['accuracy_micro']:.6f} | {xt['metrics']['parse_failures']} | {xt['latency_mean_s']:.6f} | {xt['latency_p95_s']:.6f} | {xt['peak_vram_mb']:.2f} |",
        "",
        f"- Accuracy gain: {xlrs_acc_delta:+.6f}",
        f"- Latency multiplier: {xlrs_latency_mult:.2f}x",
        f"- Peak VRAM increase: {xlrs_vram_delta:+.2f} MB",
        f"- Single wrong -> tiling correct: {xlrs['flips']['single_wrong_tiling_correct']}",
        f"- Single correct -> tiling wrong: {xlrs['flips']['single_correct_tiling_wrong']}",
        "",
    ]
    add_stats_table(lines, "XLRS by category:", xlrs["category_stats"])
    add_stats_table(lines, "XLRS by resolution bucket:", xlrs["bucket_stats"])
    lines.extend(
        [
            "## MME-RS A/B Result",
            "",
            "| Mode | Samples | Accuracy | Parse failures | Mean latency s | P95 latency s | Peak VRAM MB |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            f"| Single-view | {ms['sample_count']} | {ms['metrics']['accuracy']:.6f} | {ms['metrics']['parse_failures']} | {ms['latency_mean_s']:.6f} | {ms['latency_p95_s']:.6f} | {ms['peak_vram_mb']:.2f} |",
            f"| Global+2x2 | {mt['sample_count']} | {mt['metrics']['accuracy']:.6f} | {mt['metrics']['parse_failures']} | {mt['latency_mean_s']:.6f} | {mt['latency_p95_s']:.6f} | {mt['peak_vram_mb']:.2f} |",
            "",
            f"- Accuracy gain: {mme_acc_delta:+.6f}",
            f"- Latency multiplier: {mme_latency_mult:.2f}x",
            f"- Peak VRAM increase: {mme_vram_delta:+.2f} MB",
            f"- Single wrong -> tiling correct: {mme['flips']['single_wrong_tiling_correct']}",
            f"- Single correct -> tiling wrong: {mme['flips']['single_correct_tiling_wrong']}",
            "",
        ]
    )
    add_stats_table(lines, "MME-RS by category:", mme["category_stats"])
    add_stats_table(lines, "MME-RS by resolution bucket:", mme["bucket_stats"])
    lines.extend(
        [
            "## Positive And Negative Samples",
            "",
        ]
    )
    add_examples(lines, "XLRS single wrong -> tiling correct:", xlrs["examples"]["single_wrong_tiling_correct"])
    add_examples(lines, "XLRS single correct -> tiling wrong:", xlrs["examples"]["single_correct_tiling_wrong"])
    add_examples(lines, "MME-RS single wrong -> tiling correct:", mme["examples"]["single_wrong_tiling_correct"])
    add_examples(lines, "MME-RS single correct -> tiling wrong:", mme["examples"]["single_correct_tiling_wrong"])
    lines.extend(
        [
            "## Deployment Decision",
            "",
            "- The mechanism is technically feasible on the RTX 5060 Laptop GPU 8GB with INT4 NF4. There were no inference failures in the full XLRS partial run or MME-RS full run.",
            "- The accuracy gains are small: XLRS +0.016234 and MME-RS +0.006666.",
            "- The resource cost is large: XLRS mean latency is 4.88x single-view and MME-RS mean latency is 4.54x single-view. Peak VRAM rises by about 1.0 GB.",
            "- The benefit is not cleanly concentrated in all high-resolution images because almost all XLRS local samples are already >16MP. Category-level gains are task-dependent.",
            "- A simple adaptive route based only on megapixels is not justified by this evidence. A more selective route would need a task-aware trigger, but that would be a new design beyond this fixed ablation.",
            "",
            "## Recommendation",
            "",
            "Do not make global+2x2 tiling the default deployment mode for the competition submission. Keep it as a reproducible innovation ablation and optional high-resolution fallback for selected tasks where latency is acceptable.",
            "",
            "It can be described as a tested resource-aware multi-scale inference mechanism, but not as the final default model path because the small accuracy gain does not justify the latency multiplier.",
            "",
            "## Limitations",
            "",
            "- XLRS is still a local 44/74-shard partial result, not the full official Lite benchmark.",
            "- The first version uses fixed 2x2 tiles with 0 overlap and no learned or detector-based routing.",
            "- The prompt is an added multi-view wrapper around official MCQ instructions; no evaluator or ground truth was changed.",
            "- No LoRA retraining or prompt search was performed.",
        ]
    )
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
