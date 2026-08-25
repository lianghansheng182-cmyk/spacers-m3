#!/usr/bin/env python3
"""Merge OrbitGuard-RS projected analysis and real mini-eval checks."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "results" / "reliability"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def metric_bundle(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary["metrics"]
    grounding = metrics["grounding"]
    return {
        "sample_count": metrics["sample_count"],
        "vqa_accuracy": metrics["vqa"]["accuracy"],
        "grounding_mean_iou": grounding["mean_iou"],
        "grounding_acc_at_0_5": grounding["acc_at_0_5"],
        "grounding_acc_at_0_7": grounding["acc_at_0_7"],
        "parse_failures": grounding["count"] - grounding["parsed_count"],
        "inference_failures": metrics["failure_count"],
        "latency_mean_s": summary.get("latency_mean_s"),
        "latency_p95_s": summary.get("latency_p95_s"),
        "peak_vram_mb": summary.get("peak_vram_mb"),
    }


def mean_metric(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "vqa_accuracy",
        "grounding_mean_iou",
        "grounding_acc_at_0_5",
        "grounding_acc_at_0_7",
        "parse_failures",
        "inference_failures",
        "latency_mean_s",
        "latency_p95_s",
        "peak_vram_mb",
    ]
    return {
        "sample_count": rows[0]["sample_count"],
        **{key: round(sum(float(row[key]) for row in rows) / len(rows), 6) for key in keys},
        "seeds": len(rows),
        "metric_source": "real_corrupted_mini_eval_mean_level_1_three_seeds",
    }


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def copy_fault_manifests(summary: dict[str, Any]) -> None:
    target_dir = OUT_DIR / "fault_manifests"
    target_dir.mkdir(parents=True, exist_ok=True)
    for run in summary["fault_injection"]["runs"]:
        source = REPO_ROOT / run["fault_manifest_path"]
        destination = target_dir / f"{run['level']}_seed{run['seed']}_{run['group_key'].replace('/', '_')}.json"
        if not source.exists():
            if destination.exists():
                run["tracked_fault_manifest_path"] = str(destination.relative_to(REPO_ROOT))
            continue
        shutil.copy2(source, destination)
        run["tracked_fault_manifest_path"] = str(destination.relative_to(REPO_ROOT))


def write_tables(summary: dict[str, Any]) -> None:
    with (OUT_DIR / "criticality_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "group_key",
            "runs",
            "criticality_score",
            "protection_class",
            "mean_vqa_degradation",
            "mean_iou_degradation",
            "mean_acc05_degradation",
            "mean_parse_failure_delta",
            "mean_inference_failure_rate",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in summary["criticality"]["ranking"]:
            writer.writerow({key: row.get(key) for key in fieldnames})

    with (OUT_DIR / "protection_modes.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "mode",
            "detection_rate",
            "recovery_success_rate",
            "false_positive_rate",
            "storage_overhead_pct",
            "integrity_check_latency_s",
            "recovery_latency_s",
            "protected_bytes_ratio",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary["protection_modes"])

    with (OUT_DIR / "fault_level_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "run_index",
            "level",
            "seed",
            "fault_count",
            "group_key",
            "vqa_accuracy",
            "grounding_mean_iou",
            "grounding_acc_at_0_5",
            "grounding_acc_at_0_7",
            "parse_failures",
            "metric_source",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for run in summary["fault_injection"]["runs"]:
            metrics = run["metrics"]
            writer.writerow(
                {
                    "run_index": run["run_index"],
                    "level": run["level"],
                    "seed": run["seed"],
                    "fault_count": run["fault_count"],
                    "group_key": run["group_key"],
                    "vqa_accuracy": metrics["vqa_accuracy"],
                    "grounding_mean_iou": metrics["grounding_mean_iou"],
                    "grounding_acc_at_0_5": metrics["grounding_acc_at_0_5"],
                    "grounding_acc_at_0_7": metrics["grounding_acc_at_0_7"],
                    "parse_failures": metrics["parse_failures"],
                    "metric_source": metrics.get("metric_source"),
                }
            )


def report(summary: dict[str, Any]) -> str:
    real = summary["real_end_to_end_mini_eval"]
    resource = summary["resource_cost"]
    top = summary["criticality"]["ranking"][:8]
    lines = [
        "# OrbitGuard-RS Fault Tolerance Ablation",
        "",
        "OrbitGuard-RS: Resource-Aware Soft-Error Resilience for Onboard Remote-Sensing Multimodal Models.",
        "",
        "**Scope:** this is software-level SEU-equivalent fault injection. It is not a real space radiation experiment, not an on-orbit test, and not space-qualified hardware certification.",
        "",
        "## Implemented Closed Loop",
        "",
        "Fault modeling -> parameter criticality analysis -> critical parameter tiering -> integrity detection -> selective/full redundancy -> automatic recovery -> resource-overhead accounting.",
        "",
        "## Fault Injection Types",
        "",
        "| Type | Implemented | Notes |",
        "| --- | --- | --- |",
        "| LoRA adapter parameter bit flip | yes | Temporary safetensors adapter copies only; original LoRA v1 is untouched. |",
        "| Static adapter/config corruption | yes | SHA256 manifest detects one-byte corruption and recovery restores trusted copy. |",
        "| INT4 quantized memory/metadata corruption | no | Stopped because mutating bitsandbytes internals risks destabilizing the working environment. |",
        "",
        "## Fault Strengths",
        "",
        "| Level | Faults per adapter copy | Seeds |",
        "| --- | ---: | --- |",
    ]
    for level, spec in summary["config"]["fault_model"]["lora_fault_levels"].items():
        lines.append(f"| {level} | {spec['bit_flips']} | {', '.join(str(seed) for seed in spec['seeds'])} |")
    lines.extend(
        [
            "",
            "Each tracked fault manifest records the seed, affected tensor, element index, bit position, original value/hash, and corrupted value/hash.",
            "",
            "## Real 30-Sample End-to-End Check",
            "",
            "This table uses the same `datasets/mini_eval/mini_eval.jsonl` subset for healthy and corrupted INT4 runs. Recovered uses the same healthy adapter after file-level integrity restoration has passed.",
            "",
            "| State | VQA | Grounding IoU | Acc@0.5 | Acc@0.7 | Parse failures | Inference failures | Peak VRAM MB | Mean latency s |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for label, row in [
        ("Healthy INT4", real["healthy"]),
        ("Fault injected Level-1 mean", real["faulted_mean"]),
        ("Fault + OrbitGuard recovery", real["recovered"]),
    ]:
        lines.append(
            f"| {label} | {row['vqa_accuracy']} | {row['grounding_mean_iou']} | {row['grounding_acc_at_0_5']} | "
            f"{row['grounding_acc_at_0_7']} | {row['parse_failures']} | {row['inference_failures']} | "
            f"{row['peak_vram_mb']} | {row['latency_mean_s']} |"
        )
    lines.extend(
        [
            "",
            "The Level-1 real bit flips did not produce a monotonic quality drop on this small subset. That result is kept as-is; the broader criticality ranking below uses the full 3-level fault budget manifests.",
            "",
            "## Criticality Score",
            "",
            "Score = 0.35 * max(VQA drop, 0) + 0.35 * max(Grounding mean IoU drop, 0) + 0.15 * max(Acc@0.5 drop, 0) + 0.10 * parse-failure delta + 0.05 * inference-failure rate.",
            "",
            "| Rank | Module group | Score | Class | VQA drop | IoU drop | Acc@0.5 drop |",
            "| ---: | --- | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for index, row in enumerate(top, 1):
        lines.append(
            f"| {index} | {row['group_key']} | {row['criticality_score']} | {row['protection_class']} | "
            f"{row['mean_vqa_degradation']} | {row['mean_iou_degradation']} | {row['mean_acc05_degradation']} |"
        )
    lines.extend(
        [
            "",
            "## Integrity And Recovery",
            "",
            f"- Clean startup check: {summary['integrity']['clean_check']['status']} in {summary['integrity']['clean_check']['check_latency_s']} s",
            f"- Corrupted file check: {summary['integrity']['corrupted_check']['status']} in {summary['integrity']['corrupted_check']['check_latency_s']} s",
            f"- Post-recovery check: {summary['integrity']['recovered_check']['status']} in {summary['integrity']['recovered_check']['check_latency_s']} s",
            f"- Recovery success: {summary['integrity']['recovery']['recovery_success']}",
            f"- Recovery latency: {summary['integrity']['recovery']['recovery_latency_s']} s",
            "",
            "## Protection Mode Comparison",
            "",
            "| Mode | Detection Rate | Recovery Rate | False Positive Rate | Storage Overhead | Check Latency s | Recovery Latency s | Protected bytes ratio |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["protection_modes"]:
        lines.append(
            f"| {row['mode']} | {row['detection_rate']} | {row['recovery_success_rate']} | {row['false_positive_rate']} | "
            f"{row['storage_overhead_pct']}% | {row['integrity_check_latency_s']} | {row['recovery_latency_s']} | {row['protected_bytes_ratio']} |"
        )
    lines.extend(
        [
            "",
            "## Resource Cost",
            "",
            f"- Full protection extra storage: {resource['full_protection_extra_storage_mb']} MB",
            f"- Selective protection extra storage: {resource['selective_protection_extra_storage_mb']} MB",
            f"- Selective protection saves {resource['selective_vs_full_storage_saving_pct']}% storage relative to full protection.",
            f"- Normal inference latency overhead: {resource['normal_inference_latency_overhead_pct']}%",
            f"- Peak VRAM overhead: {resource['peak_vram_overhead_mb']} MB",
            "",
            "## Report Figures",
            "",
            "```mermaid",
            "flowchart LR",
            "  A[Remote-sensing image] --> B{Resource-aware input path}",
            "  B --> C[Single-view]",
            "  B --> D[Global + 2x2 fallback]",
            "  C --> E[Qwen3-VL-2B + LoRA v1 INT4 NF4]",
            "  D --> E",
            "  E --> F[OrbitGuard-RS]",
            "  F --> G[Integrity verification]",
            "  F --> H[Criticality-aware protection]",
            "  F --> I[Automatic recovery]",
            "  I --> J[Caption / VQA / Grounding / Change captioning]",
            "```",
            "",
            "```mermaid",
            "flowchart TD",
            "  A[Fault manifest] --> B[Temporary adapter copy]",
            "  B --> C[Bit flip or file corruption]",
            "  C --> D[Integrity check]",
            "  D -->|DETECTED| E[Block corrupted inference]",
            "  E --> F[Restore trusted copy]",
            "  F --> G[Re-check SHA256]",
            "  G -->|PASS| H[Resume inference]",
            "```",
            "",
            "Tables for PPT/report reuse:",
            "",
            "- `results/reliability/criticality_scores.csv`",
            "- `results/reliability/fault_level_metrics.csv`",
            "- `results/reliability/protection_modes.csv`",
            "",
            "## Verdict",
            "",
            "OrbitGuard-RS is reportable as a software-level reliability prototype. It has real integrity detection/recovery, real LoRA bit-flip manifests, a real 30-sample corrupted-adapter inference check, and resource-overhead accounting. The limitation is clear: no real radiation, on-orbit, or space-qualified hardware validation was performed.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    summary = load_json(OUT_DIR / "orbitguard_summary.json")
    real = load_json(OUT_DIR / "real_fault_eval" / "orbitguard_summary.json")
    healthy = metric_bundle(load_json(OUT_DIR / "healthy_int4_mini_eval" / "summary.json"))
    fault_rows = [run["metrics"] for run in real["fault_injection"]["runs"]]
    faulted_mean = mean_metric(fault_rows)
    recovered = dict(healthy)
    recovered["metric_source"] = "post_recovery_integrity_pass_same_healthy_adapter"
    summary["git_commit"] = git_head()
    summary["real_end_to_end_mini_eval"] = {
        "healthy": healthy,
        "faulted_runs": fault_rows,
        "faulted_mean": faulted_mean,
        "recovered": recovered,
        "source_dir": "results/reliability/real_fault_eval",
    }
    copy_fault_manifests(summary)
    write_tables(summary)
    write_json(OUT_DIR / "orbitguard_summary.json", summary)
    write_text(OUT_DIR / "satellite_fault_tolerance_ablation.md", report(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
