#!/usr/bin/env python3
"""Write the current Base/v1/v2 LoRA model-selection conclusion."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def bundle(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary["metrics"]
    grounding = metrics["grounding"]
    return {
        "vqa_accuracy": metrics["vqa"]["accuracy"],
        "grounding_mean_iou": grounding["mean_iou"],
        "grounding_acc_at_0_5": grounding["acc_at_0_5"],
        "grounding_acc_at_0_7": grounding["acc_at_0_7"],
        "grounding_parse_failures": grounding["count"] - grounding["parsed_count"],
        "success_count": summary["success_count"],
        "failure_count": summary["failure_count"],
        "latency_mean_s": summary["latency_mean_s"],
        "latency_p95_s": summary["latency_p95_s"],
        "peak_vram_mb": summary["peak_vram_mb"],
        "adapter_size_mb": summary.get("adapter_size_mb"),
    }


def row(metric: str, base: Any, v1: Any, v2: Any) -> str:
    return f"| {metric} | {base} | {v1} | {v2} |"


def main() -> int:
    mini = {
        "base": bundle(load_json("results/vrsbench_mini/qwen3_vl_2b_zero_shot/summary.json")),
        "v1": bundle(load_json("results/lora/qwen3_vl_2b_lora_v1_eval/mini_eval_prompt_v1/summary.json")),
        "v2": bundle(load_json("results/lora/qwen3_vl_2b_lora_v2_eval/mini_eval_prompt_v1/summary.json")),
    }
    eval90 = {
        "base": bundle(load_json("results/vrsbench_eval_v1/qwen3_vl_2b_prompt_v1/summary.json")),
        "v1": bundle(load_json("results/lora/qwen3_vl_2b_lora_v1_eval/vrsbench_eval_v1_prompt_v1/summary.json")),
        "v2": bundle(load_json("results/lora/qwen3_vl_2b_lora_v2_eval/vrsbench_eval_v1_prompt_v1/summary.json")),
    }

    lines = [
        "# LoRA Model Selection Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Decision",
        "",
        "LoRA v1 is the current main model. Base remains the original zero-shot baseline. LoRA v2 is retained as a grounding-enhancement ablation, not promoted.",
        "",
        "## Mini Eval 30",
        "",
        "| Metric | Base | LoRA v1 | LoRA v2 |",
        "|---|---:|---:|---:|",
        row("VQA accuracy", mini["base"]["vqa_accuracy"], mini["v1"]["vqa_accuracy"], mini["v2"]["vqa_accuracy"]),
        row("Grounding mean IoU", mini["base"]["grounding_mean_iou"], mini["v1"]["grounding_mean_iou"], mini["v2"]["grounding_mean_iou"]),
        row("Grounding Acc@0.5", mini["base"]["grounding_acc_at_0_5"], mini["v1"]["grounding_acc_at_0_5"], mini["v2"]["grounding_acc_at_0_5"]),
        row("Grounding Acc@0.7", mini["base"]["grounding_acc_at_0_7"], mini["v1"]["grounding_acc_at_0_7"], mini["v2"]["grounding_acc_at_0_7"]),
        row("Grounding parse failures", mini["base"]["grounding_parse_failures"], mini["v1"]["grounding_parse_failures"], mini["v2"]["grounding_parse_failures"]),
        row("Mean latency s", mini["base"]["latency_mean_s"], mini["v1"]["latency_mean_s"], mini["v2"]["latency_mean_s"]),
        row("P95 latency s", mini["base"]["latency_p95_s"], mini["v1"]["latency_p95_s"], mini["v2"]["latency_p95_s"]),
        row("Peak VRAM MB", mini["base"]["peak_vram_mb"], mini["v1"]["peak_vram_mb"], mini["v2"]["peak_vram_mb"]),
        "",
        "## VRSBench Eval v1 90",
        "",
        "| Metric | Base | LoRA v1 | LoRA v2 |",
        "|---|---:|---:|---:|",
        row("VQA accuracy", eval90["base"]["vqa_accuracy"], eval90["v1"]["vqa_accuracy"], eval90["v2"]["vqa_accuracy"]),
        row("Grounding mean IoU", eval90["base"]["grounding_mean_iou"], eval90["v1"]["grounding_mean_iou"], eval90["v2"]["grounding_mean_iou"]),
        row("Grounding Acc@0.5", eval90["base"]["grounding_acc_at_0_5"], eval90["v1"]["grounding_acc_at_0_5"], eval90["v2"]["grounding_acc_at_0_5"]),
        row("Grounding Acc@0.7", eval90["base"]["grounding_acc_at_0_7"], eval90["v1"]["grounding_acc_at_0_7"], eval90["v2"]["grounding_acc_at_0_7"]),
        row("Grounding parse failures", eval90["base"]["grounding_parse_failures"], eval90["v1"]["grounding_parse_failures"], eval90["v2"]["grounding_parse_failures"]),
        row("Mean latency s", eval90["base"]["latency_mean_s"], eval90["v1"]["latency_mean_s"], eval90["v2"]["latency_mean_s"]),
        row("P95 latency s", eval90["base"]["latency_p95_s"], eval90["v1"]["latency_p95_s"], eval90["v2"]["latency_p95_s"]),
        row("Peak VRAM MB", eval90["base"]["peak_vram_mb"], eval90["v1"]["peak_vram_mb"], eval90["v2"]["peak_vram_mb"]),
        "",
        "## Why v1 Is Main",
        "",
        "- v1 gives the best 90-sample grounding mean IoU among the LoRA adapters: 0.443984.",
        "- v1 keeps VQA materially above Base: 0.60 -> 0.70 on the 90-sample eval.",
        "- v1 is the better balanced adapter for the current VQA + Grounding objective.",
        "",
        "## Why v2 Is Not Main",
        "",
        "- v2 improves 90-sample VQA to 0.733333 and removes the v1 grounding parse failure.",
        "- v2 regresses 90-sample Grounding mean IoU from 0.443984 to 0.382554.",
        "- v2 regresses 90-sample Grounding Acc@0.5 from 0.466667 to 0.40.",
        "- The regression is on localization quality, not parser compatibility.",
        "",
        "## v2 Positive Value",
        "",
        "- Grounding parse failures drop from 1 to 0 on the 90-sample eval.",
        "- VQA is maintained and improves from 0.70 to 0.733333 on the 90-sample eval.",
        "- The ablation confirms Prompt-v1-style grounding format augmentation can improve output-format reliability.",
        "",
        "## Current Conclusion",
        "",
        "Simply increasing Grounding samples and task weight does not stably improve precise localization. The next grounding work should target object selection and bbox boundary precision, not just more Grounding sampling.",
        "",
    ]
    output = Path("results/lora/lora_model_selection_report.md")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
