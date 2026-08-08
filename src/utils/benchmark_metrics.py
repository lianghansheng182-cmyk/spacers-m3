"""Unified benchmark metrics aggregation for batch inference results."""

from __future__ import annotations

import json
import math
import platform
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = (
    PROJECT_ROOT / "results" / "vrsbench_mini" / "qwen3_vl_2b_zero_shot"
)
DEFAULT_REPORT_PATH = DEFAULT_RESULTS_DIR / "benchmark_report.json"
TTFT_NOT_IMPLEMENTED = "暂未实现"
TASK_TYPES = ("caption", "vqa", "grounding")


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    task: str
    status: str
    latency_s: float | None
    peak_vram_mb: float | None
    input_tokens: int | None
    output_tokens: int | None
    model_load_time_s: float | None
    vqa_accuracy: float | None
    grounding_iou: float | None
    error: str | None
    precision: str | None
    batch_size: int | None
    max_new_tokens: int | None
    image_resolution: str | None
    prompt_version: str | None
    raw: Mapping[str, Any]


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_present(record: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def _relative_to_project(path: Path) -> str:
    project_root = PROJECT_ROOT.resolve()
    candidate = path if path.is_absolute() else (Path.cwd() / path)
    resolved = candidate.resolve()
    try:
        return resolved.relative_to(project_root).as_posix()
    except ValueError:
        if not resolved.is_absolute():
            return path.as_posix()
        return Path(path).as_posix()


def _estimate_output_tokens(record: Mapping[str, Any]) -> int | None:
    explicit = _as_int(_first_present(record, "output_tokens", "generated_tokens"))
    if explicit is not None:
        return explicit
    prediction = _first_present(record, "prediction")
    if isinstance(prediction, str) and prediction.strip():
        return len(prediction.split())
    return None


def normalize_sample(record: Mapping[str, Any]) -> SampleRecord:
    success_value = _first_present(record, "success", "status")
    if isinstance(success_value, bool):
        status = "success" if success_value else "failed"
    else:
        status = str(success_value or "success").lower()
        if status in {"true", "ok"}:
            status = "success"
        elif status in {"false", "error"}:
            status = "failed"

    if _first_present(record, "error", "error_message"):
        status = "failed"

    task = str(_first_present(record, "task", "task_type", "type") or "unknown").lower()

    return SampleRecord(
        sample_id=str(_first_present(record, "sample_id", "id", "image_id") or ""),
        task=task,
        status=status,
        latency_s=_as_float(_first_present(record, "latency_s", "inference_time_s", "latency")),
        peak_vram_mb=_as_float(_first_present(record, "peak_vram_mb", "peak_vram", "vram_mb")),
        input_tokens=_as_int(_first_present(record, "input_tokens", "prompt_tokens")),
        output_tokens=_estimate_output_tokens(record),
        model_load_time_s=_as_float(
            _first_present(record, "model_load_time_s", "model_load_s", "load_time_s")
        ),
        vqa_accuracy=_as_float(_first_present(record, "vqa_accuracy", "accuracy")),
        grounding_iou=_as_float(_first_present(record, "grounding_iou", "iou", "mean_iou")),
        error=str(_first_present(record, "error", "error_message") or "") or None,
        precision=str(_first_present(record, "precision", "dtype") or "") or None,
        batch_size=_as_int(_first_present(record, "batch_size")),
        max_new_tokens=_as_int(_first_present(record, "max_new_tokens")),
        image_resolution=str(_first_present(record, "image_resolution", "resolution") or "")
        or None,
        prompt_version=str(_first_present(record, "prompt_version", "prompt") or "") or None,
        raw=record,
    )


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            item = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_no}")
        records.append(item)
    return records


def discover_prediction_files(results_dir: Path) -> list[Path]:
    if not results_dir.exists():
        return []

    candidates: list[Path] = []
    for pattern in (
        "predictions.jsonl",
        "sample_results.jsonl",
        "batch_results.jsonl",
        "inference_results.jsonl",
        "*.jsonl",
    ):
        for path in sorted(results_dir.glob(pattern)):
            if path.name in {"benchmark_report.json", "error_cases.jsonl"}:
                continue
            if path not in candidates:
                candidates.append(path)
    return candidates


def _metrics_lookup(metrics: Mapping[str, Any] | None) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    if not metrics:
        return lookup
    for item in metrics.get("per_sample", []):
        if not isinstance(item, dict):
            continue
        sample_id = str(item.get("id") or "")
        task = str(item.get("task") or "")
        if sample_id and task:
            lookup[(sample_id, task)] = item
    return lookup


def enrich_sample(
    record: Mapping[str, Any], metrics_item: Mapping[str, Any] | None
) -> dict[str, Any]:
    enriched = dict(record)
    if not metrics_item:
        return enriched

    if metrics_item.get("task") == "vqa" and "exact_match" in metrics_item:
        enriched["vqa_accuracy"] = 1.0 if metrics_item["exact_match"] else 0.0
    if metrics_item.get("task") == "grounding" and metrics_item.get("iou") is not None:
        enriched["grounding_iou"] = metrics_item["iou"]
    return enriched


def load_samples(results_dir: Path) -> tuple[list[SampleRecord], list[Path], dict[str, Any]]:
    files = discover_prediction_files(results_dir)
    metrics_payload: dict[str, Any] = {}
    metrics_path = results_dir / "metrics.json"
    if metrics_path.exists():
        metrics_payload = load_json(metrics_path)
    metrics_lookup = _metrics_lookup(metrics_payload)

    samples: list[SampleRecord] = []
    for file_path in files:
        for record in load_jsonl(file_path):
            sample_id = str(_first_present(record, "sample_id", "id") or "")
            task = str(_first_present(record, "task", "task_type") or "")
            metrics_item = metrics_lookup.get((sample_id, task))
            samples.append(normalize_sample(enrich_sample(record, metrics_item)))
    return samples, files, metrics_payload


def percentile(values: Sequence[float], pct: float = 95.0) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return round(values[0], 6)

    ordered = sorted(values)
    rank = (len(ordered) - 1) * (pct / 100.0)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return round(ordered[lower], 6)

    weight = rank - lower
    value = ordered[lower] * (1 - weight) + ordered[upper] * weight
    return round(value, 6)


def safe_rate(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 6)


def _collect_latencies(samples: Iterable[SampleRecord]) -> list[float]:
    latencies: list[float] = []
    for sample in samples:
        if sample.status == "success" and sample.latency_s is not None:
            latencies.append(sample.latency_s)
    return latencies


def _collect_peak_vram(samples: Iterable[SampleRecord]) -> float | None:
    values = [sample.peak_vram_mb for sample in samples if sample.peak_vram_mb is not None]
    if not values:
        return None
    return round(max(values), 2)


def _collect_output_tokens(samples: Iterable[SampleRecord]) -> list[int]:
    tokens: list[int] = []
    for sample in samples:
        if sample.status == "success" and sample.output_tokens is not None:
            tokens.append(sample.output_tokens)
    return tokens


def _summarize_group(
    samples: Sequence[SampleRecord],
    *,
    include_task_metrics: bool = False,
) -> dict[str, Any]:
    latencies = _collect_latencies(samples)
    output_tokens = _collect_output_tokens(samples)
    success_count = sum(1 for sample in samples if sample.status == "success")
    failure_count = sum(1 for sample in samples if sample.status != "success")
    total_count = len(samples)

    total_output_tokens = sum(output_tokens) if output_tokens else 0
    total_latency = sum(latencies) if latencies else 0.0
    token_speed = safe_rate(total_output_tokens, total_latency)

    summary: dict[str, Any] = {
        "sample_count": total_count,
        "success_count": success_count,
        "failure_count": failure_count,
        "failure_rate": safe_rate(failure_count, total_count) or 0.0,
        "mean_latency_s": round(statistics.mean(latencies), 6) if latencies else None,
        "p95_latency_s": percentile(latencies, 95.0),
        "per_sample_latency_s": [round(value, 6) for value in latencies],
        "peak_vram_mb": _collect_peak_vram(samples),
        "total_output_tokens": total_output_tokens,
        "mean_output_tokens": round(statistics.mean(output_tokens), 6) if output_tokens else None,
        "token_speed_tokens_per_s": token_speed,
        "ttft_s": TTFT_NOT_IMPLEMENTED,
    }

    if include_task_metrics:
        vqa_scores = [
            sample.vqa_accuracy
            for sample in samples
            if sample.vqa_accuracy is not None and sample.task == "vqa"
        ]
        if vqa_scores:
            summary["vqa_accuracy"] = round(statistics.mean(vqa_scores), 6)

        grounding_scores = [
            sample.grounding_iou
            for sample in samples
            if sample.grounding_iou is not None and sample.task == "grounding"
        ]
        if grounding_scores:
            summary["grounding_mean_iou"] = round(statistics.mean(grounding_scores), 6)

    return summary


def collect_environment(
    summary: Mapping[str, Any] | None = None,
    run_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    summary = summary or {}
    run_config = run_config or {}

    env: dict[str, Any] = {
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "cpu": platform.processor() or platform.machine(),
        "gpu_name": summary.get("gpu_name") or run_config.get("gpu_name") or "N/A",
        "driver_version": "N/A",
        "torch_version": summary.get("torch_version") or run_config.get("torch_version") or "N/A",
        "cuda_version": summary.get("cuda_version") or run_config.get("cuda_version") or "N/A",
        "cuda_available": False,
    }

    if torch is not None:
        env["torch_version"] = torch.__version__
        env["cuda_version"] = torch.version.cuda or env["cuda_version"]
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available() and env["gpu_name"] == "N/A":
            env["gpu_name"] = torch.cuda.get_device_name(0)
    return env


def _format_image_resolution(
    run_config: Mapping[str, Any] | None,
    samples: Sequence[SampleRecord],
) -> str | None:
    if run_config and run_config.get("max_pixels") is not None:
        return f"max_pixels={run_config['max_pixels']}"
    for sample in samples:
        if sample.image_resolution:
            return sample.image_resolution
    return None


def _pick_model_config(
    samples: Sequence[SampleRecord],
    summary: Mapping[str, Any] | None = None,
    run_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    summary = summary or {}
    run_config = run_config or {}
    return {
        "model_name": run_config.get("model_id")
        or _first_present(summary, "model_id")
        or "Qwen/Qwen3-VL-2B-Instruct",
        "precision": summary.get("precision") or run_config.get("precision"),
        "batch_size": summary.get("batch_size") or run_config.get("batch_size"),
        "max_new_tokens": summary.get("max_new_tokens") or run_config.get("max_new_tokens"),
        "image_resolution": _format_image_resolution(run_config, samples),
        "prompt_version": "batch-default",
    }


def build_benchmark_report(
    samples: Sequence[SampleRecord],
    *,
    results_dir: Path | None = None,
    source_files: Sequence[Path] | None = None,
    summary: Mapping[str, Any] | None = None,
    run_config: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    summary = summary or {}
    run_config = run_config or {}
    metrics = metrics or {}

    overall = _summarize_group(samples)
    model_load_values = [
        sample.model_load_time_s
        for sample in samples
        if sample.model_load_time_s is not None
    ]
    overall["model_load_time_s"] = (
        summary.get("model_load_time_s")
        or run_config.get("model_load_time_s")
        or (round(model_load_values[0], 6) if model_load_values else None)
    )

    latencies = _collect_latencies(samples)
    overall["total_runtime_s"] = summary.get("total_runtime_s") or (
        round(sum(latencies), 6) if latencies else 0.0
    )

    task_sections = {
        task: _summarize_group(
            [sample for sample in samples if sample.task == task],
            include_task_metrics=True,
        )
        for task in TASK_TYPES
    }

    if metrics.get("vqa"):
        task_sections["vqa"]["vqa_accuracy"] = metrics["vqa"].get("accuracy")
    if metrics.get("grounding"):
        task_sections["grounding"]["grounding_mean_iou"] = metrics["grounding"].get("mean_iou")
        task_sections["grounding"]["acc_at_0_5"] = metrics["grounding"].get("acc_at_0_5")
        task_sections["grounding"]["acc_at_0_7"] = metrics["grounding"].get("acc_at_0_7")

    return {
        "overall": overall,
        "caption": task_sections["caption"],
        "vqa": task_sections["vqa"],
        "grounding": task_sections["grounding"],
        "environment": collect_environment(summary, run_config),
        "model_config": _pick_model_config(samples, summary, run_config),
        "metadata": {
            "results_dir": _relative_to_project(results_dir or DEFAULT_RESULTS_DIR),
            "source_files": [_relative_to_project(path) for path in (source_files or [])],
            "sample_count": len(samples),
            "token_count_source": "estimated_from_prediction_words_when_missing",
        },
    }


def generate_benchmark_report(
    results_dir: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    resolved_results_dir = (results_dir or DEFAULT_RESULTS_DIR).resolve()
    resolved_output_path = (output_path or DEFAULT_REPORT_PATH).resolve()

    samples, source_files, metrics_payload = load_samples(resolved_results_dir)
    summary_payload = (
        load_json(resolved_results_dir / "summary.json")
        if (resolved_results_dir / "summary.json").exists()
        else {}
    )
    run_config_payload = (
        load_json(resolved_results_dir / "run_config.json")
        if (resolved_results_dir / "run_config.json").exists()
        else {}
    )

    report = build_benchmark_report(
        samples,
        results_dir=resolved_results_dir,
        source_files=source_files,
        summary=summary_payload,
        run_config=run_config_payload,
        metrics=metrics_payload,
    )

    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Generate unified benchmark report JSON.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing batch inference outputs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Path to write benchmark_report.json.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = generate_benchmark_report(args.results_dir, args.output)
    print(f"[PASS] Benchmark report saved to {_relative_to_project(args.output)}")
    print(f"[PASS] Aggregated {report['metadata']['sample_count']} samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
