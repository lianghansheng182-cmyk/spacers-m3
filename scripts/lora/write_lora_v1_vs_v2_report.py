#!/usr/bin/env python3
"""Write Base vs LoRA v1 vs LoRA v2 metrics and grounding delta report."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def metric_bundle(summary: dict[str, Any]) -> dict[str, float]:
    metrics = summary["metrics"]
    grounding = metrics["grounding"]
    return {
        "vqa_acc": metrics["vqa"]["accuracy"],
        "mean_iou": grounding["mean_iou"],
        "acc_05": grounding["acc_at_0_5"],
        "acc_07": grounding["acc_at_0_7"],
        "parse_failure": grounding["count"] - grounding["parsed_count"],
        "lat_mean": summary["latency_mean_s"],
        "lat_p95": summary["latency_p95_s"],
        "peak_vram": summary["peak_vram_mb"],
        "adapter_size": summary.get("adapter_size_mb", 0.0),
    }


def table_row(name: str, base: float, v1: float, v2: float) -> str:
    return f"| {name} | {base:.6g} | {v1:.6g} | {v2:.6g} | {v2 - v1:+.6g} |"


def per_sample_by_id(metrics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in metrics["per_sample"] if row.get("task") == "grounding"}


def predictions_by_id(path: Path) -> dict[str, dict[str, Any]]:
    return {row.get("id"): row for row in load_jsonl(path)}


def classify(v2_metric: dict[str, Any], delta: float) -> str:
    if v2_metric.get("parsed_bbox") is None:
        return "parse failure"
    if delta > 0.02:
        return "improved"
    if delta < -0.02:
        return "declined"
    return "unchanged"


def box_area(box: list[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def center_distance(box_a: list[float], box_b: list[float]) -> float:
    ax = (float(box_a[0]) + float(box_a[2])) / 2
    ay = (float(box_a[1]) + float(box_a[3])) / 2
    bx = (float(box_b[0]) + float(box_b[2])) / 2
    by = (float(box_b[1]) + float(box_b[3])) / 2
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def error_category(metric: dict[str, Any]) -> str:
    pred = metric.get("parsed_bbox")
    gt = metric.get("target_bbox")
    if pred is None:
        return "parse failure"
    if not gt:
        return "unknown"
    iou = float(metric["iou"])
    if iou >= 0.5:
        return "mostly correct"
    pred_area = box_area(pred)
    gt_area = box_area(gt)
    area_ratio = pred_area / gt_area if gt_area else 0.0
    distance = center_distance(pred, gt)
    if iou == 0.0 and distance > 20:
        return "target mismatch or multi-target"
    if area_ratio >= 1.8:
        return "bbox too large"
    if area_ratio <= 0.55:
        return "bbox too small"
    if distance > 12:
        return "bbox shifted"
    return "bbox boundary inaccurate"


def write_report() -> None:
    mini_base = load_json(Path("results/vrsbench_mini/qwen3_vl_2b_zero_shot/summary.json"))
    mini_v1 = load_json(Path("results/lora/qwen3_vl_2b_lora_v1_eval/mini_eval_prompt_v1/summary.json"))
    mini_v2 = load_json(Path("results/lora/qwen3_vl_2b_lora_v2_eval/mini_eval_prompt_v1/summary.json"))
    eval_base = load_json(Path("results/vrsbench_eval_v1/qwen3_vl_2b_prompt_v1/summary.json"))
    eval_v1 = load_json(Path("results/lora/qwen3_vl_2b_lora_v1_eval/vrsbench_eval_v1_prompt_v1/summary.json"))
    eval_v2 = load_json(Path("results/lora/qwen3_vl_2b_lora_v2_eval/vrsbench_eval_v1_prompt_v1/summary.json"))
    smoke_v2 = load_json(Path("results/lora/qwen3_vl_2b_lora_v2_smoke/summary.json"))
    train_v2 = load_json(Path("results/lora/qwen3_vl_2b_lora_v2/summary.json"))
    build_v2 = load_json(Path("datasets/vrsbench_lora_v2/build_summary.json"))
    target_audit = load_json(Path("results/lora/qwen3_vl_2b_lora_v2_smoke/grounding_target_audit.json"))

    mini = {key: metric_bundle(value) for key, value in {"base": mini_base, "v1": mini_v1, "v2": mini_v2}.items()}
    eval90 = {key: metric_bundle(value) for key, value in {"base": eval_base, "v1": eval_v1, "v2": eval_v2}.items()}

    v1_metrics = per_sample_by_id(load_json(Path("results/lora/qwen3_vl_2b_lora_v1_eval/vrsbench_eval_v1_prompt_v1/metrics.json")))
    v2_metrics = per_sample_by_id(load_json(Path("results/lora/qwen3_vl_2b_lora_v2_eval/vrsbench_eval_v1_prompt_v1/metrics.json")))
    v1_preds = predictions_by_id(Path("results/lora/qwen3_vl_2b_lora_v1_eval/vrsbench_eval_v1_prompt_v1/predictions.jsonl"))
    v2_preds = predictions_by_id(Path("results/lora/qwen3_vl_2b_lora_v2_eval/vrsbench_eval_v1_prompt_v1/predictions.jsonl"))

    grounding_rows = []
    counts = {"improved": 0, "declined": 0, "unchanged": 0, "parse failure": 0}
    error_counts: dict[str, int] = {}
    for sample_id in sorted(v1_metrics):
        old = v1_metrics[sample_id]
        new = v2_metrics[sample_id]
        delta = float(new["iou"]) - float(old["iou"])
        category = classify(new, delta)
        geometry_category = error_category(new)
        counts[category] += 1
        error_counts[geometry_category] = error_counts.get(geometry_category, 0) + 1
        grounding_rows.append(
            "| {id} | <code>{v1_pred}</code> | <code>{v2_pred}</code> | {gt} | {v1_iou:.6f} | {v2_iou:.6f} | {delta:+.6f} | {parse_status} | {category} | {geometry_category} |".format(
                id=sample_id,
                v1_pred=str(v1_preds[sample_id].get("prediction", "")).replace("\n", " "),
                v2_pred=str(v2_preds[sample_id].get("prediction", "")).replace("\n", " "),
                gt=json.dumps(new.get("target_bbox"), ensure_ascii=False),
                v1_iou=float(old["iou"]),
                v2_iou=float(new["iou"]),
                delta=delta,
                parse_status=new.get("parse_status"),
                category=category,
                geometry_category=geometry_category,
            )
        )

    lines = [
        "# LoRA v1 vs LoRA v2 Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## V2 Data Change",
        "",
        "Caption and VQA source samples are kept from v1. Grounding train samples add one Prompt-v1-style Chinese wrapper per source sample, with unchanged GT bbox and canonical assistant output.",
        "",
        "```json",
        json.dumps(build_v2, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Smoke Test",
        "",
        "```json",
        json.dumps(smoke_v2, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Grounding Target Audit",
        "",
        "```json",
        json.dumps(
            {
                "status": target_audit["status"],
                "grounding_rows": target_audit["grounding_rows"],
                "sampled_rows": target_audit["sampled_rows"],
                "target_kind_counts": target_audit["target_kind_counts"],
                "example_target": target_audit["examples"][0]["supervised_assistant_target"] if target_audit["examples"] else None,
            },
            ensure_ascii=False,
            indent=2,
        ),
        "```",
        "",
        "## Full Training",
        "",
        "```json",
        json.dumps(train_v2, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Mini Eval 30",
        "",
        "| Metric | Base | LoRA v1 | LoRA v2 | v2-v1 |",
        "|---|---:|---:|---:|---:|",
        table_row("VQA accuracy", mini["base"]["vqa_acc"], mini["v1"]["vqa_acc"], mini["v2"]["vqa_acc"]),
        table_row("Grounding mean IoU", mini["base"]["mean_iou"], mini["v1"]["mean_iou"], mini["v2"]["mean_iou"]),
        table_row("Grounding Acc@0.5", mini["base"]["acc_05"], mini["v1"]["acc_05"], mini["v2"]["acc_05"]),
        table_row("Grounding Acc@0.7", mini["base"]["acc_07"], mini["v1"]["acc_07"], mini["v2"]["acc_07"]),
        table_row("Grounding parse failures", mini["base"]["parse_failure"], mini["v1"]["parse_failure"], mini["v2"]["parse_failure"]),
        table_row("Mean latency seconds", mini["base"]["lat_mean"], mini["v1"]["lat_mean"], mini["v2"]["lat_mean"]),
        table_row("P95 latency seconds", mini["base"]["lat_p95"], mini["v1"]["lat_p95"], mini["v2"]["lat_p95"]),
        table_row("Peak VRAM MB", mini["base"]["peak_vram"], mini["v1"]["peak_vram"], mini["v2"]["peak_vram"]),
        "",
        "## VRSBench Eval v1 90",
        "",
        "| Metric | Base | LoRA v1 | LoRA v2 | v2-v1 |",
        "|---|---:|---:|---:|---:|",
        table_row("VQA accuracy", eval90["base"]["vqa_acc"], eval90["v1"]["vqa_acc"], eval90["v2"]["vqa_acc"]),
        table_row("Grounding mean IoU", eval90["base"]["mean_iou"], eval90["v1"]["mean_iou"], eval90["v2"]["mean_iou"]),
        table_row("Grounding Acc@0.5", eval90["base"]["acc_05"], eval90["v1"]["acc_05"], eval90["v2"]["acc_05"]),
        table_row("Grounding Acc@0.7", eval90["base"]["acc_07"], eval90["v1"]["acc_07"], eval90["v2"]["acc_07"]),
        table_row("Grounding parse failures", eval90["base"]["parse_failure"], eval90["v1"]["parse_failure"], eval90["v2"]["parse_failure"]),
        table_row("Mean latency seconds", eval90["base"]["lat_mean"], eval90["v1"]["lat_mean"], eval90["v2"]["lat_mean"]),
        table_row("P95 latency seconds", eval90["base"]["lat_p95"], eval90["v1"]["lat_p95"], eval90["v2"]["lat_p95"]),
        table_row("Peak VRAM MB", eval90["base"]["peak_vram"], eval90["v1"]["peak_vram"], eval90["v2"]["peak_vram"]),
        "",
        "## 90-Eval Grounding V1 vs V2",
        "",
        "```json",
        json.dumps(counts, ensure_ascii=False, indent=2),
        "```",
        "",
        "V2 geometry/error categories:",
        "",
        "```json",
        json.dumps(dict(sorted(error_counts.items())), ensure_ascii=False, indent=2),
        "```",
        "",
        "| sample_id | v1 prediction | v2 prediction | GT | v1 IoU | v2 IoU | delta IoU | v2 parse | delta category | v2 error category |",
        "|---|---|---|---:|---:|---:|---:|---|---|---|",
        *grounding_rows,
        "",
    ]
    output = Path("results/lora/lora_v1_vs_v2_report.md")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    write_report()
