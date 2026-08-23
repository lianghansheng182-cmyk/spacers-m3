#!/usr/bin/env python3
"""Write the LoRA v1 vs baseline comparison report from saved summaries."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def metric_bundle(summary: dict) -> dict[str, float]:
    metrics = summary["metrics"]
    return {
        "vqa_acc": metrics["vqa"]["accuracy"],
        "mean_iou": metrics["grounding"]["mean_iou"],
        "acc_05": metrics["grounding"]["acc_at_0_5"],
        "acc_07": metrics["grounding"]["acc_at_0_7"],
        "lat_mean": summary["latency_mean_s"],
        "lat_p95": summary["latency_p95_s"],
        "peak_vram": summary["peak_vram_mb"],
    }


def delta_row(name: str, base: float, lora: float) -> str:
    return f"| {name} | {base:.6g} | {lora:.6g} | {lora - base:+.6g} |"


def main() -> int:
    smoke = load_json("results/lora/qwen3_vl_2b_lora_v1_smoke/summary.json")
    full = load_json("results/lora/qwen3_vl_2b_lora_v1/summary.json")
    mini_base = load_json("results/vrsbench_mini/qwen3_vl_2b_zero_shot/summary.json")
    mini_lora = load_json("results/lora/qwen3_vl_2b_lora_v1_eval/mini_eval_prompt_v1/summary.json")
    eval_base = load_json("results/vrsbench_eval_v1/qwen3_vl_2b_prompt_v1/summary.json")
    eval_lora = load_json("results/lora/qwen3_vl_2b_lora_v1_eval/vrsbench_eval_v1_prompt_v1/summary.json")
    env = load_json("results/lora/qwen3_vl_2b_lora_v1/environment.json")
    config = Path("configs/lora/qwen3_vl_2b_lora_v1.yaml").read_text(encoding="utf-8")

    mini_b = metric_bundle(mini_base)
    mini_l = metric_bundle(mini_lora)
    eval_b = metric_bundle(eval_base)
    eval_l = metric_bundle(eval_lora)

    lines = [
        "# LoRA v1 vs Base Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Verdict",
        "",
        "LoRA v1 training completed and adapter evaluation finished on Prompt v1.",
        "",
        "- Dry-run: PASS",
        "- Smoke test: PASS",
        "- Full training: PASS",
        "- Mini eval inference: 30/30 success",
        "- VRSBench eval v1 inference: 90/90 success",
        "- Baseline outputs were not modified.",
        "",
        "## Training Artifact",
        "",
        "| Item | Value |",
        "|---|---|",
        "| Base model | `models/Qwen3-VL-2B-Instruct` |",
        "| Adapter | `results/lora/qwen3_vl_2b_lora_v1/adapter_final` |",
        f"| Adapter size | {mini_lora['adapter_size_mb']} MB |",
        f"| Git commit recorded at training start | `{env.get('git_commit', 'unknown')}` |",
        "| Prompt | v1 |",
        "",
        "## Dry-Run",
        "",
        "| Metric | Value |",
        "|---|---:|",
        "| Status | PASS |",
        "| Data samples collated | 10 |",
        f"| Total parameters | {full['total_parameters']} |",
        f"| Trainable parameters | {full['trainable_parameters']} |",
        f"| Trainable ratio | {full['trainable_ratio_percent']}% |",
        "| Dry-run report | `results/lora/lora_dry_run_report.md` |",
        "",
        "## Smoke Test",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Optimizer steps | {smoke['optimizer_steps']} |",
        f"| Micro steps | {smoke['micro_steps']} |",
        f"| Loss first window | {smoke['loss_first_window']:.6f} |",
        f"| Loss last window | {smoke['loss_last_window']:.6f} |",
        f"| NaN loss | {smoke['loss_nan']} |",
        f"| OOM count | {smoke['oom_count']} |",
        f"| Peak VRAM | {smoke['peak_vram_gib']} GiB |",
        f"| Checkpoints | {len(smoke['checkpoints'])} |",
        "",
        "## Full Training",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Optimizer steps | {full['optimizer_steps']} |",
        f"| Micro steps | {full['micro_steps']} |",
        f"| Loss first window | {full['loss_first_window']:.6f} |",
        f"| Loss last window | {full['loss_last_window']:.6f} |",
        f"| Min loss | {full['loss_min']:.6f} |",
        f"| Max loss | {full['loss_max']:.6f} |",
        f"| NaN loss | {full['loss_nan']} |",
        f"| OOM count | {full['oom_count']} |",
        f"| Elapsed seconds | {full['elapsed_seconds']} |",
        f"| Peak VRAM | {full['peak_vram_gib']} GiB |",
        f"| Checkpoints | {len(full['checkpoints'])} |",
        "",
        "## Mini Eval 30",
        "",
        "| Metric | Base | LoRA v1 | Delta |",
        "|---|---:|---:|---:|",
        delta_row("VQA accuracy", mini_b["vqa_acc"], mini_l["vqa_acc"]),
        delta_row("Grounding mean IoU", mini_b["mean_iou"], mini_l["mean_iou"]),
        delta_row("Grounding Acc@0.5", mini_b["acc_05"], mini_l["acc_05"]),
        delta_row("Grounding Acc@0.7", mini_b["acc_07"], mini_l["acc_07"]),
        delta_row("Latency mean seconds", mini_b["lat_mean"], mini_l["lat_mean"]),
        delta_row("Latency P95 seconds", mini_b["lat_p95"], mini_l["lat_p95"]),
        delta_row("Peak VRAM MB", mini_b["peak_vram"], mini_l["peak_vram"]),
        "",
        "## VRSBench Eval v1 90",
        "",
        "| Metric | Base | LoRA v1 | Delta |",
        "|---|---:|---:|---:|",
        delta_row("VQA accuracy", eval_b["vqa_acc"], eval_l["vqa_acc"]),
        delta_row("Grounding mean IoU", eval_b["mean_iou"], eval_l["mean_iou"]),
        delta_row("Grounding Acc@0.5", eval_b["acc_05"], eval_l["acc_05"]),
        delta_row("Grounding Acc@0.7", eval_b["acc_07"], eval_l["acc_07"]),
        delta_row("Latency mean seconds", eval_b["lat_mean"], eval_l["lat_mean"]),
        delta_row("Latency P95 seconds", eval_b["lat_p95"], eval_l["lat_p95"]),
        delta_row("Peak VRAM MB", eval_b["peak_vram"], eval_l["peak_vram"]),
        "",
        "## Issues Found",
        "",
        "- LoRA improves VQA accuracy on both eval sets.",
        "- Grounding mean IoU improves slightly on both eval sets, but Acc@0.7 drops on both sets.",
        "- One grounding prediction failed bbox parsing on the 90-sample eval.",
        "- Training used a constant learning rate; no scheduler or validation loop was added in this v1 run.",
        "- The training objective masks the prompt and trains only assistant tokens; caption/VQA/grounding are mixed by shuffled sample order.",
        "",
        "## Config",
        "",
        "```yaml",
        config.rstrip(),
        "```",
        "",
    ]
    Path("results/lora/lora_v1_vs_base_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote results/lora/lora_v1_vs_base_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
