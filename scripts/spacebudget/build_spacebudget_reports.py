#!/usr/bin/env python3
"""Build SpaceBudget-RS audit reports from completed benchmark outputs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "results" / "spacebudget"
FIG_DIR = OUT_DIR / "figures"

PATCH_SIZE = 16
SPACEBUDGET_PROFILES = {
    "ECO": 512 * 512,
    "BALANCED": 1024 * 1024,
    "PRECISION": 2048 * 2048,
}

RUNS = {
    "MME-RS": {
        "ECO": OUT_DIR / "mme_eco",
        "BALANCED": OUT_DIR / "mme_balanced",
        "PRECISION": OUT_DIR / "mme_precision",
        "AUTO": OUT_DIR / "mme_auto",
    },
    "XLRS": {
        "ECO": OUT_DIR / "xlrs_eco_full",
        "BALANCED": REPO_ROOT / "results" / "xlrs_bench" / "partial_44shards_int4",
        "PRECISION": OUT_DIR / "xlrs_precision_full",
        "AUTO": OUT_DIR / "xlrs_auto_full",
    },
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return float(ordered[lower])
    weight = rank - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": round(sum(values) / len(values), 6),
        "p50": round(percentile(values, 0.50), 6),
        "p95": round(percentile(values, 0.95), 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def grid_tokens(grid: Any) -> tuple[int, int, list[dict[str, int]]]:
    if not grid:
        return 0, 0, []
    patch_tokens = 0
    processed = []
    for item in grid:
        t, h, w = [int(v) for v in item]
        patch_tokens += t * h * w
        processed.append({"frames": t, "height": h * PATCH_SIZE, "width": w * PATCH_SIZE})
    return patch_tokens, patch_tokens // 4, processed


def iter_predictions(benchmark: str, run_dir: Path) -> list[dict[str, Any]]:
    if benchmark == "MME-RS":
        path = run_dir / "predictions.json"
        if not path.exists():
            return []
        return load_json(path)
    path = run_dir / "predictions.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def accuracy_from_summary(benchmark: str, summary: dict[str, Any]) -> float | None:
    metrics = summary.get("metrics") or {}
    if benchmark == "MME-RS":
        return metrics.get("accuracy")
    return metrics.get("accuracy_micro")


def by_task_accuracy(benchmark: str, summary: dict[str, Any]) -> dict[str, float]:
    metrics = summary.get("metrics") or {}
    if benchmark == "MME-RS":
        out: dict[str, float] = {}
        by_task = (((metrics.get("by_task") or {}).get("Perception") or {}).get("Remote Sensing") or {})
        for name, item in by_task.items():
            total = int(item.get("true", 0)) + int(item.get("false", 0))
            out[name] = round(item.get("true", 0) / total, 6) if total else 0.0
        return out
    return {str(k): float(v) for k, v in (metrics.get("category_accuracy") or {}).items()}


def make_sample_audit(benchmark: str, budget: str, row: dict[str, Any]) -> dict[str, Any]:
    if benchmark == "MME-RS":
        meta = row.get("Image metadata") or {}
        task_type = row.get("Category")
        sample_id = row.get("Question_id")
        grid = row.get("Image grid thw")
        patch_tokens = row.get("Patch visual tokens")
        language_tokens = row.get("Language visual tokens")
        image_token_count = row.get("Image token count")
    else:
        meta = row.get("image") or {}
        task_type = row.get("category")
        sample_id = row.get("index", row.get("row_number"))
        grid = row.get("image_grid_thw")
        patch_tokens = row.get("patch_visual_tokens")
        language_tokens = row.get("language_visual_tokens")
        image_token_count = row.get("image_token_count")
    computed_patch, computed_language, processed = grid_tokens(grid)
    patch_tokens = int(patch_tokens or computed_patch)
    language_tokens = int(language_tokens or computed_language)
    width = int(meta.get("original_width") or meta.get("width") or 0)
    height = int(meta.get("original_height") or meta.get("height") or 0)
    return {
        "benchmark": benchmark,
        "budget": budget,
        "sample_id": sample_id,
        "task_type": task_type,
        "original_width": width,
        "original_height": height,
        "megapixels": round(width * height / 1_000_000, 6) if width and height else None,
        "processed_resolution": processed,
        "image_grid_thw": grid,
        "patch_visual_tokens": patch_tokens,
        "language_visual_tokens": language_tokens,
        "image_token_count": image_token_count,
    }


def run_record(benchmark: str, budget: str, run_dir: Path) -> dict[str, Any] | None:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return None
    summary = load_json(summary_path)
    predictions = iter_predictions(benchmark, run_dir)
    audits = [make_sample_audit(benchmark, budget, row) for row in predictions]
    lang_tokens = [row["language_visual_tokens"] for row in audits if row["language_visual_tokens"]]
    patch_tokens = [row["patch_visual_tokens"] for row in audits if row["patch_visual_tokens"]]
    token_summary = summary.get("visual_tokens") or {}
    if not token_summary:
        token_summary = {
            "patch": summarize(patch_tokens),
            "language": summarize(lang_tokens),
        }
    return {
        "benchmark": benchmark,
        "budget": budget,
        "run_dir": str(run_dir.relative_to(REPO_ROOT)),
        "sample_count": int(summary.get("sample_count", 0)),
        "failure_count": int(summary.get("failure_count", 0)),
        "parse_failures": int((summary.get("metrics") or {}).get("parse_failures", 0)),
        "accuracy": accuracy_from_summary(benchmark, summary),
        "latency_mean_s": summary.get("latency_mean_s"),
        "latency_p95_s": summary.get("latency_p95_s"),
        "peak_vram_mb": summary.get("peak_vram_mb"),
        "visual_tokens": token_summary,
        "metrics": summary.get("metrics") or {},
        "by_task_accuracy": by_task_accuracy(benchmark, summary),
        "sample_audit": audits,
    }


def pareto(records: list[dict[str, Any]], cost_key: str) -> list[str]:
    points = [r for r in records if r.get("accuracy") is not None and r.get(cost_key) is not None]
    frontier = []
    for candidate in points:
        dominated = False
        for other in points:
            if other is candidate:
                continue
            if other["accuracy"] >= candidate["accuracy"] and other[cost_key] <= candidate[cost_key]:
                if other["accuracy"] > candidate["accuracy"] or other[cost_key] < candidate[cost_key]:
                    dominated = True
                    break
        if not dominated:
            frontier.append(candidate["budget"])
    return frontier


def svg_scatter(path: Path, title: str, x_label: str, y_label: str, rows: list[dict[str, Any]], x_key: str) -> None:
    width, height = 760, 460
    pad = 70
    valid = [r for r in rows if r.get("accuracy") is not None and r.get(x_key) is not None]
    if not valid:
        path.write_text(f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'><text x='20' y='30'>{title}: unavailable</text></svg>\n")
        return
    xs = [float(r[x_key]) for r in valid]
    ys = [float(r["accuracy"]) for r in valid]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if math.isclose(xmin, xmax):
        xmax = xmin + 1
    if math.isclose(ymin, ymax):
        ymax = ymin + 0.01
    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        f"<text x='{width/2}' y='28' text-anchor='middle' font-size='20' font-family='Arial'>{title}</text>",
        f"<line x1='{pad}' y1='{height-pad}' x2='{width-pad}' y2='{height-pad}' stroke='#333'/>",
        f"<line x1='{pad}' y1='{height-pad}' x2='{pad}' y2='{pad}' stroke='#333'/>",
        f"<text x='{width/2}' y='{height-20}' text-anchor='middle' font-size='13' font-family='Arial'>{x_label}</text>",
        f"<text transform='translate(20 {height/2}) rotate(-90)' text-anchor='middle' font-size='13' font-family='Arial'>{y_label}</text>",
    ]
    colors = {"ECO": "#2a9d8f", "BALANCED": "#2878b5", "PRECISION": "#d1495b", "AUTO": "#7b2cbf"}
    for r in valid:
        x = pad + (float(r[x_key]) - xmin) / (xmax - xmin) * (width - 2 * pad)
        y = height - pad - (float(r["accuracy"]) - ymin) / (ymax - ymin) * (height - 2 * pad)
        color = colors.get(r["budget"], "#333")
        label = f"{r['benchmark']} {r['budget']} {r['accuracy']:.3f}"
        parts.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='7' fill='{color}'/>")
        parts.append(f"<text x='{x+10:.1f}' y='{y-10:.1f}' font-size='11' font-family='Arial'>{label}</text>")
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def svg_bar(path: Path, title: str, rows: list[dict[str, Any]], value_key: str) -> None:
    width, height = 900, 520
    pad = 90
    valid = [r for r in rows if r.get(value_key) is not None]
    values = [float(r[value_key]) for r in valid]
    maxv = max(values) if values else 1.0
    minv = min(values) if values else 0.0
    span = max(maxv - minv, 1e-9)
    zero_y = height - pad - (0 - minv) / span * (height - 2 * pad)
    bar_w = max((width - 2 * pad) / max(len(valid), 1) - 8, 10)
    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        f"<text x='{width/2}' y='30' text-anchor='middle' font-size='20' font-family='Arial'>{title}</text>",
        f"<line x1='{pad}' y1='{zero_y:.1f}' x2='{width-pad}' y2='{zero_y:.1f}' stroke='#999'/>",
    ]
    for i, r in enumerate(valid):
        value = float(r[value_key])
        y_value = height - pad - (value - minv) / span * (height - 2 * pad)
        h = abs(zero_y - y_value)
        x = pad + i * ((width - 2 * pad) / len(valid))
        y = min(y_value, zero_y)
        fill = "#2878b5" if value >= 0 else "#d1495b"
        parts.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{bar_w:.1f}' height='{h:.1f}' fill='{fill}'/>")
        text_y = y - 6 if value >= 0 else y + h + 14
        parts.append(f"<text x='{x+bar_w/2:.1f}' y='{text_y:.1f}' text-anchor='middle' font-size='10' font-family='Arial'>{value:.3f}</text>")
        parts.append(f"<text x='{x+bar_w/2:.1f}' y='{height-pad+16}' text-anchor='middle' font-size='9' font-family='Arial' transform='rotate(45 {x+bar_w/2:.1f},{height-pad+16})'>{r['benchmark']} {r['budget']}</text>")
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def svg_architecture(path: Path) -> None:
    parts = [
        "<svg xmlns='http://www.w3.org/2000/svg' width='920' height='540' viewBox='0 0 920 540'>",
        "<rect width='100%' height='100%' fill='white'/>",
        "<text x='460' y='32' text-anchor='middle' font-size='22' font-family='Arial'>SpaceRS-Agent with SpaceBudget-RS</text>",
    ]
    boxes = [
        ("Input", 360, 60),
        ("SpaceBudget-RS\nECO / BALANCED / PRECISION / HiRes fallback", 250, 125),
        ("RemoteLoRA\nQwen3-VL-2B + LoRA v1", 300, 220),
        ("OrbitQuant INT4 NF4", 315, 300),
        ("OrbitGuard-RS", 335, 380),
        ("Output", 360, 460),
    ]
    for text, x, y in boxes:
        parts.append(f"<rect x='{x}' y='{y}' width='220' height='54' rx='8' fill='#f4f7fb' stroke='#2f4b7c'/>")
        lines = text.split("\\n")
        for j, line in enumerate(lines):
            parts.append(f"<text x='{x+110}' y='{y+23+j*17}' text-anchor='middle' font-size='13' font-family='Arial'>{line}</text>")
    for y in [114, 179, 274, 354, 434]:
        parts.append(f"<line x1='470' y1='{y}' x2='470' y2='{y+26}' stroke='#333' marker-end='url(#arrow)'/>")
    parts.insert(2, "<defs><marker id='arrow' markerWidth='10' markerHeight='10' refX='6' refY='3' orient='auto'><path d='M0,0 L0,6 L6,3 z' fill='#333'/></marker></defs>")
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def flatten_summary(records: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for benchmark, by_budget in records.items():
        for budget, rec in by_budget.items():
            if rec is None:
                continue
            lang = ((rec.get("visual_tokens") or {}).get("language") or {})
            rows.append(
                {
                    "benchmark": benchmark,
                    "budget": budget,
                    "sample_count": rec["sample_count"],
                    "accuracy": rec["accuracy"],
                    "mean_language_visual_tokens": lang.get("mean"),
                    "p50_language_visual_tokens": lang.get("p50"),
                    "p95_language_visual_tokens": lang.get("p95"),
                    "latency_mean_s": rec["latency_mean_s"],
                    "latency_p95_s": rec["latency_p95_s"],
                    "peak_vram_mb": rec["peak_vram_mb"],
                    "failure_count": rec["failure_count"],
                    "parse_failures": rec.get("parse_failures", 0),
                }
            )
    return rows


def write_visual_audit(records: dict[str, dict[str, dict[str, Any]]], sample_path: Path) -> None:
    lines = [
        "# SpaceBudget-RS Visual Token Audit",
        "",
        "This audit is based on actual Qwen3-VL processor outputs recorded during inference.",
        "It does not estimate tokens from the original image resolution alone.",
        "",
        "Local processor/model configuration:",
        "",
        "- Qwen3-VL image budget is controlled through per-image `max_pixels`.",
        "- `image_grid_thw` is read from processor output.",
        "- Patch visual tokens = `sum(t * h * w)`.",
        "- Language-side visual tokens = patch tokens / 4 because `spatial_merge_size=2`.",
        "- Processed resolution is reconstructed as `grid_h/grid_w * patch_size`; local patch size is 16.",
        "",
        f"Per-sample audit rows are stored in `{sample_path.relative_to(REPO_ROOT)}`.",
        "",
        "| Benchmark | Budget | max_pixels | Samples | Mean language tokens | P50 | P95 | Peak VRAM MB | Mean latency s |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in flatten_summary(records):
        budget = row["budget"]
        lines.append(
            f"| {row['benchmark']} | {budget} | {SPACEBUDGET_PROFILES.get(budget, 'mixed')} | {row['sample_count']} | "
            f"{row['mean_language_visual_tokens']} | {row['p50_language_visual_tokens']} | {row['p95_language_visual_tokens']} | "
            f"{row['peak_vram_mb']} | {row['latency_mean_s']} |"
        )
    (OUT_DIR / "visual_token_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_ablation(records: dict[str, dict[str, dict[str, Any]]], summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# SpaceBudget-RS Ablation",
        "",
        "SpaceBudget-RS evaluates fixed visual-compute budgets and a deterministic task-aware AUTO policy for onboard remote-sensing MLLM inference.",
        "",
        "All runs use the frozen main model: Qwen3-VL-2B-Instruct + LoRA v1 + INT4 NF4. Prompts, datasets, and evaluators are unchanged.",
        "",
        "## Budget Profiles",
        "",
        "| Profile | max_pixels | Intended use |",
        "| --- | ---: | --- |",
        "| ECO | 262144 | Power-limited coarse recognition and low-sensitivity tasks |",
        "| BALANCED | 1048576 | Existing fixed single-view benchmark budget |",
        "| PRECISION | 4194304 | Detail-sensitive single-view runs; high latency cost on RTX 5060 Laptop 8GB |",
        "",
        "## Results",
        "",
        "| Benchmark | Budget | Samples | Accuracy | Mean visual tokens | P95 visual tokens | Mean latency s | P95 latency s | Peak VRAM MB | Run failures | Parse failures |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['benchmark']} | {row['budget']} | {row['sample_count']} | {row['accuracy']} | "
            f"{row['mean_language_visual_tokens']} | {row['p95_language_visual_tokens']} | "
            f"{row['latency_mean_s']} | {row['latency_p95_s']} | {row['peak_vram_mb']} | {row['failure_count']} | {row['parse_failures']} |"
        )
    lines += [
        "",
        "## AUTO vs Fixed BALANCED",
        "",
        "| Benchmark | BALANCED accuracy | AUTO accuracy | Accuracy delta | BALANCED mean tokens | AUTO mean tokens | Token saving |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for benchmark, by_budget in records.items():
        balanced = by_budget.get("BALANCED")
        auto = by_budget.get("AUTO")
        if not balanced or not auto:
            continue
        b_tokens = ((balanced.get("visual_tokens") or {}).get("language") or {}).get("mean")
        a_tokens = ((auto.get("visual_tokens") or {}).get("language") or {}).get("mean")
        b_acc = balanced.get("accuracy")
        a_acc = auto.get("accuracy")
        saving = None
        if b_tokens and a_tokens is not None:
            saving = round((float(b_tokens) - float(a_tokens)) / float(b_tokens), 6)
        lines.append(f"| {benchmark} | {b_acc} | {a_acc} | {round(float(a_acc) - float(b_acc), 6) if a_acc is not None and b_acc is not None else None} | {b_tokens} | {a_tokens} | {saving} |")
    lines += [
        "",
        "## AUTO Rule",
        "",
        "AUTO is deterministic and uses only pre-inference metadata: benchmark task/category text, question text, image dimensions, and resource mode.",
        "",
        "- MME count and XLRS overall-counting use ECO because higher budgets did not improve accuracy reliably.",
        "- Spatial, land-use, motion-state, and complex-reasoning categories use BALANCED because they show measurable ECO-to-BALANCED gains.",
        "- Color/small-object questions use BALANCED when category/question metadata indicates color detail; PRECISION did not improve MME color over BALANCED.",
        "- PRECISION+ HiRes tiling is available as an explicit fallback for selected regional-recognition tasks, but is not enabled globally.",
        "",
        "## Pareto Frontier",
        "",
    ]
    for benchmark, by_budget in records.items():
        rows = []
        for budget, rec in by_budget.items():
            if rec is None:
                continue
            lang = ((rec.get("visual_tokens") or {}).get("language") or {}).get("mean")
            rows.append({"budget": budget, "accuracy": rec.get("accuracy"), "tokens": lang, "latency": rec.get("latency_mean_s"), "vram": rec.get("peak_vram_mb")})
        lines.append(f"### {benchmark}")
        lines.append("")
        lines.append(f"- Accuracy vs visual tokens frontier: {', '.join(pareto(rows, 'tokens')) or 'unavailable'}")
        lines.append(f"- Accuracy vs latency frontier: {', '.join(pareto(rows, 'latency')) or 'unavailable'}")
        lines.append(f"- Accuracy vs peak VRAM frontier: {', '.join(pareto(rows, 'vram')) or 'unavailable'}")
        lines.append("")
    lines += [
        "## Resource Scenarios",
        "",
        "- Power-Limited: force ECO.",
        "- Normal: use AUTO or BALANCED when reproducibility requires a fixed budget.",
        "- Urgent Precision: use PRECISION, and enable HiRes fallback only for categories where tiling has demonstrated value.",
        "",
        "This is a compute-resource strategy simulation, not a real satellite power measurement.",
    ]
    (OUT_DIR / "spacebudget_ablation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, dict[str, Any]]] = {}
    samples = []
    for benchmark, by_budget in RUNS.items():
        records[benchmark] = {}
        for budget, run_dir in by_budget.items():
            rec = run_record(benchmark, budget, run_dir)
            if rec is None:
                continue
            samples.extend(rec.pop("sample_audit"))
            records[benchmark][budget] = rec

    sample_path = OUT_DIR / "visual_token_audit_samples.jsonl"
    with sample_path.open("w", encoding="utf-8") as handle:
        for row in samples:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_rows = flatten_summary(records)
    summary = {
        "profiles": SPACEBUDGET_PROFILES,
        "runs": records,
        "summary_rows": summary_rows,
        "figures": [
            "results/spacebudget/figures/accuracy_vs_visual_tokens.svg",
            "results/spacebudget/figures/accuracy_vs_latency.svg",
            "results/spacebudget/figures/task_budget_sensitivity.svg",
            "results/spacebudget/figures/spacebudget_architecture.svg",
            "results/spacebudget/figures/eco_balanced_precision_comparison.svg",
            "results/spacebudget/figures/auto_vs_fixed.svg",
        ],
    }
    (OUT_DIR / "spacebudget_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_visual_audit(records, sample_path)
    write_ablation(records, summary_rows)

    plot_rows = [
        {
            **row,
            "tokens": row["mean_language_visual_tokens"],
            "latency": row["latency_mean_s"],
            "vram": row["peak_vram_mb"],
        }
        for row in summary_rows
    ]
    svg_scatter(FIG_DIR / "accuracy_vs_visual_tokens.svg", "Accuracy vs Visual Tokens", "Mean language visual tokens", "Accuracy", plot_rows, "tokens")
    svg_scatter(FIG_DIR / "accuracy_vs_latency.svg", "Accuracy vs Latency", "Mean latency (s)", "Accuracy", plot_rows, "latency")
    svg_bar(FIG_DIR / "eco_balanced_precision_comparison.svg", "ECO / BALANCED / PRECISION Accuracy", [r for r in plot_rows if r["budget"] in {"ECO", "BALANCED", "PRECISION"}], "accuracy")
    svg_bar(FIG_DIR / "auto_vs_fixed.svg", "AUTO vs Fixed BALANCED Accuracy", [r for r in plot_rows if r["budget"] in {"BALANCED", "AUTO"}], "accuracy")
    svg_architecture(FIG_DIR / "spacebudget_architecture.svg")

    # Task sensitivity figure: average absolute gain from ECO to BALANCED per task when both are available.
    sensitivity_rows = []
    for benchmark, by_budget in records.items():
        eco = by_budget.get("ECO")
        balanced = by_budget.get("BALANCED")
        if not eco or not balanced:
            continue
        for task, bal_acc in balanced.get("by_task_accuracy", {}).items():
            if task in eco.get("by_task_accuracy", {}):
                sensitivity_rows.append(
                    {
                        "benchmark": benchmark,
                        "budget": task,
                        "accuracy": round(float(bal_acc) - float(eco["by_task_accuracy"][task]), 6),
                    }
                )
    svg_bar(FIG_DIR / "task_budget_sensitivity.svg", "Task Sensitivity: BALANCED - ECO Accuracy", sensitivity_rows, "accuracy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
