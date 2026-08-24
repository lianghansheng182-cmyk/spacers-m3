#!/usr/bin/env python3
"""Audit LoRA v1 quantization readiness and write the quantization report."""

from __future__ import annotations

import importlib
import importlib.metadata as md
import json
import subprocess
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
BF16_EVAL_90 = REPO_ROOT / "results/lora/qwen3_vl_2b_lora_v1_eval/vrsbench_eval_v1_prompt_v1/summary.json"
BF16_EVAL_30 = REPO_ROOT / "results/lora/qwen3_vl_2b_lora_v1_eval/mini_eval_prompt_v1/summary.json"
REPORT_PATH = REPO_ROOT / "results/quantization/lora_v1_quantization_report.md"
AUDIT_JSON_PATH = REPO_ROOT / "results/quantization/lora_v1_quantization_audit.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def package_version(name: str) -> str:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return "MISSING"


def command_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, cwd=REPO_ROOT, text=True).strip()
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


def module_import_status(name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return {
            "status": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback_tail": traceback.format_exc().strip().splitlines()[-5:],
        }
    return {
        "status": "PASS",
        "module": getattr(module, "__name__", name),
        "version": getattr(module, "__version__", package_version(name)),
    }


def metric_bundle(summary: dict[str, Any]) -> dict[str, Any]:
    grounding = summary["metrics"]["grounding"]
    return {
        "vqa_accuracy": summary["metrics"]["vqa"]["accuracy"],
        "grounding_mean_iou": grounding["mean_iou"],
        "grounding_acc_at_0_5": grounding["acc_at_0_5"],
        "grounding_acc_at_0_7": grounding["acc_at_0_7"],
        "grounding_parse_failures": grounding["count"] - grounding["parsed_count"],
        "success_count": summary["success_count"],
        "failure_count": summary["failure_count"],
        "model_load_time_s": summary["model_load_time_s"],
        "latency_mean_s": summary["latency_mean_s"],
        "latency_p95_s": summary["latency_p95_s"],
        "peak_vram_mb": summary["peak_vram_mb"],
        "adapter_size_mb": summary.get("adapter_size_mb"),
    }


def path_size_mb(path: Path) -> float | None:
    if not path.exists():
        return None
    if path.is_file():
        return round(path.stat().st_size / 1024 / 1024, 3)
    total = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    return round(total / 1024 / 1024, 3)


def environment() -> dict[str, Any]:
    return {
        "python": command_output(["./.venv/bin/python", "--version"]),
        "packages": {
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "peft": package_version("peft"),
            "accelerate": package_version("accelerate"),
            "bitsandbytes": package_version("bitsandbytes"),
            "qwen-vl-utils": package_version("qwen-vl-utils"),
            "pillow": package_version("pillow"),
        },
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
    }


def code_audit() -> dict[str, Any]:
    tracked_files = command_output(["git", "ls-files"])
    return {
        "precision_backend_py": {
            "exists": (REPO_ROOT / "precision_backend.py").exists()
            or (REPO_ROOT / "src/utils/precision_backend.py").exists(),
            "tracked": "precision_backend.py" in tracked_files or "src/utils/precision_backend.py" in tracked_files,
            "note": "No tracked precision_backend.py was found; current precision handling is embedded in inference/evaluation scripts.",
        },
        "lora_adapter_loading": {
            "script": "scripts/lora/evaluate_lora_adapter.py",
            "method": "Load Qwen3VLForConditionalGeneration first, then PeftModel.from_pretrained(base_model, adapter_path).",
            "adapter": "results/lora/qwen3_vl_2b_lora_v1/adapter_final",
        },
        "benchmark_framework": {
            "summary_files": "summary.json and metrics.json from evaluate_lora_adapter/evaluate_mini_eval",
            "resource_monitor": "src/utils/resource_monitor.py records latency_s and peak_vram_mb.",
            "ttft": "not implemented",
            "true_tokens_per_second": "not implemented; benchmark_metrics.py intentionally does not fabricate token/s from word counts.",
        },
        "evaluation_protocol": {
            "prompt": "Prompt v1",
            "datasets": [
                "datasets/mini_eval/mini_eval.jsonl",
                "datasets/vrsbench_eval_v1/resolved_manifest.jsonl",
            ],
            "parser": "src/evaluation/evaluate_mini_eval.py parse_grounding_bbox",
        },
    }


def quant_smoke_status(precision: str, bnb_status: dict[str, Any]) -> dict[str, Any]:
    if bnb_status["status"] != "PASS":
        return {
            "precision": precision,
            "status": "NOT_RUN",
            "reason": "bitsandbytes backend is not importable in the active .venv, so real INT8/INT4 model loading cannot be attempted.",
            "backend_error": bnb_status["error"],
            "model_loaded": False,
            "adapter_loaded": False,
            "predictions_checked": 0,
        }
    return {
        "precision": precision,
        "status": "PENDING",
        "reason": "bitsandbytes imports successfully, but this audit script does not run the full quantized evaluator automatically.",
    }


def write_report(audit: dict[str, Any]) -> None:
    bf16_90 = audit["bf16_reference"]["vrsbench_eval_v1_90"]
    bf16_30 = audit["bf16_reference"]["mini_eval_30"]
    int8 = audit["smoke"]["int8"]
    int4 = audit["smoke"]["int4"]

    lines = [
        "# LoRA v1 Quantization Report",
        "",
        f"Generated: {audit['created']}",
        "",
        "## Verdict",
        "",
        "BF16 LoRA v1 is the only runnable precision in the current environment. INT8 and INT4 were not formally evaluated because the required bitsandbytes backend is missing from the active project virtual environment.",
        "",
        "This report does not claim quantization support from API availability alone.",
        "",
        "## Environment",
        "",
        "```json",
        json.dumps(audit["environment"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Code Audit",
        "",
        "```json",
        json.dumps(audit["code_audit"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## BF16 Reference",
        "",
        "LoRA v1 BF16 results are reused from the completed Prompt v1 evaluations with the same adapter and evaluation algorithm.",
        "",
        "| Dataset | VQA acc | Grounding mean IoU | Acc@0.5 | Acc@0.7 | Parse failures | Peak VRAM MB | Mean latency s | P95 latency s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| mini_eval 30 | {bf16_30['vqa_accuracy']} | {bf16_30['grounding_mean_iou']} | {bf16_30['grounding_acc_at_0_5']} | {bf16_30['grounding_acc_at_0_7']} | {bf16_30['grounding_parse_failures']} | {bf16_30['peak_vram_mb']} | {bf16_30['latency_mean_s']} | {bf16_30['latency_p95_s']} |",
        f"| vrsbench_eval_v1 90 | {bf16_90['vqa_accuracy']} | {bf16_90['grounding_mean_iou']} | {bf16_90['grounding_acc_at_0_5']} | {bf16_90['grounding_acc_at_0_7']} | {bf16_90['grounding_parse_failures']} | {bf16_90['peak_vram_mb']} | {bf16_90['latency_mean_s']} | {bf16_90['latency_p95_s']} |",
        "",
        "## Quantization Smoke",
        "",
        "| Precision | Status | Model loaded | Adapter loaded | Prediction checked | Reason/error |",
        "|---|---|---:|---:|---:|---|",
        f"| INT8 | {int8['status']} | {int8['model_loaded']} | {int8['adapter_loaded']} | {int8['predictions_checked']} | {int8['backend_error']} |",
        f"| INT4 | {int4['status']} | {int4['model_loaded']} | {int4['adapter_loaded']} | {int4['predictions_checked']} | {int4['backend_error']} |",
        "",
        "## Formal Quantized Eval",
        "",
        "| Precision | 90-run status | VQA acc | Grounding mean IoU | Acc@0.5 | Acc@0.7 | Parse failures | Peak VRAM MB | Mean latency s | P95 latency s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| BF16 | completed | {bf16_90['vqa_accuracy']} | {bf16_90['grounding_mean_iou']} | {bf16_90['grounding_acc_at_0_5']} | {bf16_90['grounding_acc_at_0_7']} | {bf16_90['grounding_parse_failures']} | {bf16_90['peak_vram_mb']} | {bf16_90['latency_mean_s']} | {bf16_90['latency_p95_s']} |",
        "| INT8 | not run | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |",
        "| INT4 | not run | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |",
        "",
        "## Storage",
        "",
        "| Artifact | Size MB |",
        "|---|---:|",
        f"| Base model directory | {audit['storage']['base_model_dir_mb']} |",
        f"| LoRA v1 adapter | {audit['storage']['lora_v1_adapter_mb']} |",
        "| INT8 quantized model artifact | not generated |",
        "| INT4 quantized model artifact | not generated |",
        "",
        "## Answers",
        "",
        "1. BF16 is the only completed precision. INT8/INT4 effects are unknown in this environment because quantized loading cannot be run without bitsandbytes.",
        "2. VRAM savings for INT8/INT4 were not measured.",
        "3. Storage savings for INT8/INT4 were not measured; no quantized model artifacts were generated.",
        "4. Latency changes for INT8/INT4 were not measured.",
        "5. The current deployment candidate remains LoRA v1 BF16.",
        "6. No quantized precision-loss conclusion can be made until INT8/INT4 smoke tests actually load and generate predictions.",
        "",
        "## Required Before Quantized Runs",
        "",
        "- Install a CUDA-compatible bitsandbytes build in `.venv`.",
        "- Add or enable an explicit precision backend for `BF16`, `INT8`, and `INT4` model loading.",
        "- Re-run smoke on Caption/VQA/Grounding samples before any 90-sample quantized eval.",
        "- Record real output token counts before reporting true token/s; TTFT is currently not implemented.",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    bnb_status = module_import_status("bitsandbytes")
    audit = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "environment": environment(),
        "code_audit": code_audit(),
        "backend_imports": {
            "bitsandbytes": bnb_status,
        },
        "bf16_reference": {
            "mini_eval_30": metric_bundle(load_json(BF16_EVAL_30)),
            "vrsbench_eval_v1_90": metric_bundle(load_json(BF16_EVAL_90)),
        },
        "storage": {
            "base_model_dir_mb": path_size_mb(REPO_ROOT / "models/Qwen3-VL-2B-Instruct"),
            "lora_v1_adapter_mb": path_size_mb(REPO_ROOT / "results/lora/qwen3_vl_2b_lora_v1/adapter_final"),
        },
        "smoke": {
            "int8": quant_smoke_status("int8", bnb_status),
            "int4": quant_smoke_status("int4", bnb_status),
        },
        "formal_quantized_eval": {
            "int8": "not_run_smoke_not_supported",
            "int4": "not_run_smoke_not_supported",
        },
    }
    AUDIT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_JSON_PATH.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(audit)
    print(json.dumps({"audit": str(AUDIT_JSON_PATH), "report": str(REPORT_PATH), "int8": audit["smoke"]["int8"], "int4": audit["smoke"]["int4"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
