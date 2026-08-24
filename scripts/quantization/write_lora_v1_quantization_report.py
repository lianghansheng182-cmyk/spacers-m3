#!/usr/bin/env python3
"""Write the completed LoRA v1 BF16/INT8/INT4 quantization report."""

from __future__ import annotations

import importlib.metadata as md
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT = REPO_ROOT / "results/quantization/lora_v1_quantization_report.md"
AUDIT = REPO_ROOT / "results/quantization/lora_v1_quantization_audit.json"


RUNS = {
    "bf16": REPO_ROOT / "results/lora/qwen3_vl_2b_lora_v1_eval/vrsbench_eval_v1_prompt_v1/summary.json",
    "int8": REPO_ROOT / "results/quantization/lora_v1_int8_eval_vrsbench90/summary.json",
    "int4": REPO_ROOT / "results/quantization/lora_v1_int4_eval_vrsbench90/summary.json",
}
SMOKE = {
    "int8": REPO_ROOT / "results/quantization/lora_v1_int8_smoke/summary.json",
    "int4": REPO_ROOT / "results/quantization/lora_v1_int4_smoke/summary.json",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def package_version(name: str) -> str:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return "MISSING"


def size_mb(path: Path) -> float | None:
    if not path.exists():
        return None
    if path.is_file():
        return round(path.stat().st_size / 1024 / 1024, 3)
    return round(sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) / 1024 / 1024, 3)


def metrics(summary: dict[str, Any]) -> dict[str, Any]:
    grounding = summary["metrics"]["grounding"]
    return {
        "precision": summary.get("precision", "bf16"),
        "success": summary["success_count"],
        "failure": summary["failure_count"],
        "vqa": summary["metrics"]["vqa"]["accuracy"],
        "mean_iou": grounding["mean_iou"],
        "acc05": grounding["acc_at_0_5"],
        "acc07": grounding["acc_at_0_7"],
        "parse_failures": grounding["count"] - grounding["parsed_count"],
        "load_time_s": summary["model_load_time_s"],
        "mean_latency_s": summary["latency_mean_s"],
        "p95_latency_s": summary["latency_p95_s"],
        "peak_vram_mb": summary["peak_vram_mb"],
        "adapter_size_mb": summary.get("adapter_size_mb"),
    }


def pct_saved(base: float, value: float) -> float:
    return round((base - value) / base * 100, 2)


def env() -> dict[str, Any]:
    return {
        "packages": {
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "peft": package_version("peft"),
            "accelerate": package_version("accelerate"),
            "bitsandbytes": package_version("bitsandbytes"),
            "qwen-vl-utils": package_version("qwen-vl-utils"),
        },
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "compute_capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
    }


def smoke_status(summary: dict[str, Any]) -> dict[str, Any]:
    grounding = summary["metrics"]["grounding"]
    return {
        "status": "PASS" if summary["success_count"] == summary["sample_count"] else "FAIL",
        "success": summary["success_count"],
        "failure": summary["failure_count"],
        "load_time_s": summary["model_load_time_s"],
        "mean_latency_s": summary["latency_mean_s"],
        "p95_latency_s": summary["latency_p95_s"],
        "peak_vram_mb": summary["peak_vram_mb"],
        "grounding_parsed": grounding["parsed_count"],
        "grounding_count": grounding["count"],
    }


def row(label: str, item: dict[str, Any], base_vram: float | None = None) -> str:
    saved = ""
    if base_vram is not None:
        saved = f" ({pct_saved(base_vram, item['peak_vram_mb'])}% saved)"
    return (
        f"| {label} | {item['success']}/{item['success'] + item['failure']} | "
        f"{item['vqa']} | {item['mean_iou']} | {item['acc05']} | {item['acc07']} | "
        f"{item['parse_failures']} | {item['peak_vram_mb']}{saved} | "
        f"{item['load_time_s']} | {item['mean_latency_s']} | {item['p95_latency_s']} |"
    )


def main() -> int:
    run_metrics = {name: metrics(load_json(path)) for name, path in RUNS.items()}
    smoke = {name: smoke_status(load_json(path)) for name, path in SMOKE.items()}
    environment = env()
    storage = {
        "base_model_dir_mb": size_mb(REPO_ROOT / "models/Qwen3-VL-2B-Instruct"),
        "lora_v1_adapter_mb": size_mb(REPO_ROOT / "results/lora/qwen3_vl_2b_lora_v1/adapter_final"),
        "int8_saved_artifact_mb": None,
        "int4_saved_artifact_mb": None,
    }
    audit = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "environment": environment,
        "smoke": smoke,
        "vrsbench_eval_v1_90": run_metrics,
        "storage": storage,
        "ttft": "unavailable",
        "true_tokens_per_second": "unavailable",
        "notes": [
            "bitsandbytes was installed into the project .venv only.",
            "torch, transformers, peft, and accelerate versions remained unchanged after installation.",
            "INT8 and INT4 use the same LoRA v1 adapter, Prompt v1, 90-sample manifest, and evaluator as BF16.",
            "Quantized base-model artifacts were not saved, so quantized storage size is unavailable.",
        ],
    }
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    bf16 = run_metrics["bf16"]
    int8 = run_metrics["int8"]
    int4 = run_metrics["int4"]
    lines = [
        "# LoRA v1 Quantization Report",
        "",
        f"Generated: {audit['created']}",
        "",
        "## Verdict",
        "",
        "INT8 and INT4 were both truly validated with bitsandbytes on the current RTX 5060 Laptop GPU / CUDA 13 environment. Both smoke tests and both 90-sample Prompt v1 evaluations completed successfully.",
        "",
        "INT4 is the strongest deployment candidate from this run: it has the lowest peak VRAM, faster latency than INT8, and the best 90-sample Grounding metrics, while keeping VQA equal to INT8 and above BF16. The caveat is one Grounding parse failure in INT4, caused by a malformed 6-number bbox output.",
        "",
        "## Environment",
        "",
        "```json",
        json.dumps(environment, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Smoke Tests",
        "",
        "| Precision | Status | Success | Load time s | Mean latency s | P95 latency s | Peak VRAM MB | Grounding parsed |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
        f"| INT8 | {smoke['int8']['status']} | {smoke['int8']['success']}/3 | {smoke['int8']['load_time_s']} | {smoke['int8']['mean_latency_s']} | {smoke['int8']['p95_latency_s']} | {smoke['int8']['peak_vram_mb']} | {smoke['int8']['grounding_parsed']}/{smoke['int8']['grounding_count']} |",
        f"| INT4 NF4 | {smoke['int4']['status']} | {smoke['int4']['success']}/3 | {smoke['int4']['load_time_s']} | {smoke['int4']['mean_latency_s']} | {smoke['int4']['p95_latency_s']} | {smoke['int4']['peak_vram_mb']} | {smoke['int4']['grounding_parsed']}/{smoke['int4']['grounding_count']} |",
        "",
        "## VRSBench Eval v1 90",
        "",
        "| Precision | Success | VQA acc | Grounding mean IoU | Acc@0.5 | Acc@0.7 | Parse failures | Peak VRAM MB | Load time s | Mean latency s | P95 latency s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        row("BF16", bf16),
        row("INT8", int8, bf16["peak_vram_mb"]),
        row("INT4 NF4", int4, bf16["peak_vram_mb"]),
        "",
        "## Storage",
        "",
        "| Artifact | Size MB |",
        "|---|---:|",
        f"| Base model directory | {storage['base_model_dir_mb']} |",
        f"| LoRA v1 adapter | {storage['lora_v1_adapter_mb']} |",
        "| INT8 saved model artifact | unavailable; quantized base model was loaded but not saved |",
        "| INT4 saved model artifact | unavailable; quantized base model was loaded but not saved |",
        "",
        "## Precision Comparison",
        "",
        "- BF16 reference: VQA 0.70, Grounding mean IoU 0.443984, Acc@0.5 0.466667, Acc@0.7 0.266667, parse failures 1, peak VRAM 4203.17 MB.",
        "- INT8: VQA 0.733333, Grounding mean IoU 0.477298, Acc@0.5 0.466667, Acc@0.7 0.40, parse failures 0, peak VRAM 2545.99 MB. VRAM saved vs BF16: 39.43%.",
        "- INT4 NF4: VQA 0.733333, Grounding mean IoU 0.50968, Acc@0.5 0.60, Acc@0.7 0.466667, parse failures 1, peak VRAM 1659.81 MB. VRAM saved vs BF16: 60.51%.",
        "",
        "## Unsupported Metrics",
        "",
        "- TTFT: unavailable; the current evaluator does not instrument first-token timing.",
        "- true token/s: unavailable; the current evaluator does not record generated token counts. No word/s estimate is reported as token/s.",
        "- quantized model storage: unavailable; no INT8/INT4 base-model artifacts were saved.",
        "",
        "## Recommendation",
        "",
        "Use LoRA v1 INT4 NF4 as the next deployment candidate, with one follow-up check on Grounding output formatting because INT4 still produced one malformed bbox. Keep BF16 as the accuracy reference and INT8 as a safer quantized fallback when zero parse failures are more important than VRAM and latency.",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"report": str(REPORT), "audit": str(AUDIT), "bf16": bf16, "int8": int8, "int4": int4}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
