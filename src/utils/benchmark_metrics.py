"""Aggregate benchmark metrics from already-produced inference outputs."""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "vrsbench_mini" / "qwen3_vl_2b_zero_shot"
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
    estimated_output_words: int | None
    model_load_time_s: float | None
    vqa_accuracy: float | None
    grounding_iou: float | None
    error: str | None
    image_resolution: str | None
    raw: Mapping[str, Any]


def _first(record: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def _float(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(value) or math.isinf(value) else value


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _relative(path: Path) -> str:
    resolved = (path if path.is_absolute() else Path.cwd() / path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def normalize_sample(record: Mapping[str, Any]) -> SampleRecord:
    success = _first(record, "success", "status")
    if isinstance(success, bool):
        status = "success" if success else "failed"
    else:
        status = str(success or "success").lower()
        status = {"true": "success", "ok": "success", "false": "failed", "error": "failed"}.get(status, status)
    error = str(_first(record, "error", "error_message") or "") or None
    if error:
        status = "failed"

    prediction = _first(record, "prediction")
    estimated_words = len(prediction.split()) if isinstance(prediction, str) and prediction.strip() else None

    # Do not fabricate tokenizer token counts from whitespace word counts.
    output_tokens = _int(_first(record, "output_tokens", "generated_tokens"))

    return SampleRecord(
        sample_id=str(_first(record, "sample_id", "id", "image_id") or ""),
        task=str(_first(record, "task", "task_type", "type") or "unknown").lower(),
        status=status,
        latency_s=_float(_first(record, "latency_s", "inference_time_s", "latency")),
        peak_vram_mb=_float(_first(record, "peak_vram_mb", "peak_vram", "vram_mb")),
        input_tokens=_int(_first(record, "input_tokens", "prompt_tokens")),
        output_tokens=output_tokens,
        estimated_output_words=estimated_words,
        model_load_time_s=_float(_first(record, "model_load_time_s", "model_load_s", "load_time_s")),
        vqa_accuracy=_float(_first(record, "vqa_accuracy", "accuracy")),
        grounding_iou=_float(_first(record, "grounding_iou", "iou", "mean_iou")),
        error=error,
        image_resolution=str(_first(record, "image_resolution", "resolution") or "") or None,
        raw=record,
    )


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_no}")
        rows.append(value)
    return rows


def discover_prediction_files(results_dir: Path) -> list[Path]:
    if not results_dir.exists():
        return []
    candidates: list[Path] = []
    for pattern in ("predictions.jsonl", "sample_results.jsonl", "batch_results.jsonl", "inference_results.jsonl", "*.jsonl"):
        for path in sorted(results_dir.glob(pattern)):
            if path.name in {"benchmark_report.json", "error_cases.jsonl"} or path in candidates:
                continue
            candidates.append(path)
    return candidates


def _metric_lookup(metrics: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in metrics.get("per_sample", []):
        if isinstance(row, dict) and row.get("id") and row.get("task"):
            lookup[(str(row["id"]), str(row["task"]))] = row
    return lookup


def load_samples(results_dir: Path) -> tuple[list[SampleRecord], list[Path], dict[str, Any]]:
    metrics_path = results_dir / "metrics.json"
    metrics = load_json(metrics_path) if metrics_path.exists() else {}
    lookup = _metric_lookup(metrics)
    files = discover_prediction_files(results_dir)
    samples: list[SampleRecord] = []
    for path in files:
        for row in load_jsonl(path):
            sample_id = str(_first(row, "sample_id", "id") or "")
            task = str(_first(row, "task", "task_type") or "")
            enriched = dict(row)
            metric = lookup.get((sample_id, task))
            if metric:
                if task == "vqa" and "exact_match" in metric:
                    enriched["vqa_accuracy"] = 1.0 if metric["exact_match"] else 0.0
                if task == "grounding" and metric.get("iou") is not None:
                    enriched["grounding_iou"] = metric["iou"]
            samples.append(normalize_sample(enriched))
    return samples, files, metrics


def percentile(values: Sequence[float], pct: float = 95.0) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return round(values[0], 6)
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100.0
    lo, hi = math.floor(rank), math.ceil(rank)
    if lo == hi:
        return round(ordered[lo], 6)
    weight = rank - lo
    return round(ordered[lo] * (1 - weight) + ordered[hi] * weight, 6)


def safe_rate(numerator: float, denominator: float) -> float | None:
    return None if denominator <= 0 else round(numerator / denominator, 6)


def _summarize(samples: Sequence[SampleRecord], task_metrics: bool = False) -> dict[str, Any]:
    ok = [s for s in samples if s.status == "success"]
    latencies = [s.latency_s for s in ok if s.latency_s is not None]
    vrams = [s.peak_vram_mb for s in samples if s.peak_vram_mb is not None]

    token_rows = [s for s in ok if s.output_tokens is not None and s.latency_s is not None]
    token_total = sum(s.output_tokens for s in token_rows) if token_rows else None
    token_latency = sum(s.latency_s for s in token_rows) if token_rows else 0.0

    word_rows = [s for s in ok if s.estimated_output_words is not None and s.latency_s is not None]
    word_total = sum(s.estimated_output_words for s in word_rows) if word_rows else None
    word_latency = sum(s.latency_s for s in word_rows) if word_rows else 0.0

    result: dict[str, Any] = {
        "sample_count": len(samples),
        "success_count": len(ok),
        "failure_count": len(samples) - len(ok),
        "failure_rate": safe_rate(len(samples) - len(ok), len(samples)) or 0.0,
        "mean_latency_s": round(statistics.mean(latencies), 6) if latencies else None,
        "p95_latency_s": percentile(latencies),
        "per_sample_latency_s": [round(v, 6) for v in latencies],
        "peak_vram_mb": round(max(vrams), 2) if vrams else None,
        "total_output_tokens": token_total,
        "mean_output_tokens": round(statistics.mean([s.output_tokens for s in token_rows]), 6) if token_rows else None,
        "token_count_coverage": safe_rate(len(token_rows), len(ok)) if ok else 0.0,
        "token_speed_tokens_per_s": safe_rate(token_total, token_latency) if token_total is not None else None,
        "estimated_output_words": word_total,
        "mean_estimated_output_words": round(statistics.mean([s.estimated_output_words for s in word_rows]), 6) if word_rows else None,
        "estimated_words_per_s": safe_rate(word_total, word_latency) if word_total is not None else None,
        "ttft_s": TTFT_NOT_IMPLEMENTED,
    }
    if task_metrics:
        vqa = [s.vqa_accuracy for s in samples if s.task == "vqa" and s.vqa_accuracy is not None]
        grounding = [s.grounding_iou for s in samples if s.task == "grounding" and s.grounding_iou is not None]
        if vqa:
            result["vqa_accuracy"] = round(statistics.mean(vqa), 6)
        if grounding:
            result["grounding_mean_iou"] = round(statistics.mean(grounding), 6)
    return result


def collect_environment(summary: Mapping[str, Any] | None = None, run_config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Use inference-time metadata only; never substitute the report machine."""
    summary, run_config = summary or {}, run_config or {}

    def pick(*keys: str) -> Any:
        for source in (summary, run_config):
            value = _first(source, *keys)
            if value not in (None, ""):
                return value
        return None

    gpu, cuda = pick("gpu_name"), pick("cuda_version")
    cuda_available = pick("cuda_available")
    if cuda_available is None:
        cuda_available = bool(gpu and cuda)
    return {
        "os": pick("os", "os_version"),
        "python_version": pick("python_version"),
        "cpu": pick("cpu", "cpu_name"),
        "gpu_name": gpu,
        "driver_version": pick("driver_version", "gpu_driver_version"),
        "torch_version": pick("torch_version"),
        "cuda_version": cuda,
        "cuda_available": bool(cuda_available),
        "source": "recorded_run_metadata",
    }


def _model_config(samples: Sequence[SampleRecord], summary: Mapping[str, Any], run_config: Mapping[str, Any]) -> dict[str, Any]:
    resolution = f"max_pixels={run_config['max_pixels']}" if run_config.get("max_pixels") is not None else next((s.image_resolution for s in samples if s.image_resolution), None)
    return {
        "model_name": run_config.get("model_id") or summary.get("model_id") or "Qwen/Qwen3-VL-2B-Instruct",
        "precision": summary.get("precision") or run_config.get("precision"),
        "batch_size": summary.get("batch_size") or run_config.get("batch_size"),
        "max_new_tokens": summary.get("max_new_tokens") or run_config.get("max_new_tokens"),
        "image_resolution": resolution,
        "prompt_version": run_config.get("prompt_version") or "batch-default",
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
    summary, run_config, metrics = summary or {}, run_config or {}, metrics or {}
    overall = _summarize(samples)
    loads = [s.model_load_time_s for s in samples if s.model_load_time_s is not None]
    overall["model_load_time_s"] = summary.get("model_load_time_s") or run_config.get("model_load_time_s") or (round(loads[0], 6) if loads else None)
    latencies = [s.latency_s for s in samples if s.status == "success" and s.latency_s is not None]
    overall["total_runtime_s"] = summary.get("total_runtime_s") or (round(sum(latencies), 6) if latencies else 0.0)

    sections = {task: _summarize([s for s in samples if s.task == task], True) for task in TASK_TYPES}
    if metrics.get("vqa"):
        sections["vqa"]["vqa_accuracy"] = metrics["vqa"].get("accuracy")
    if metrics.get("grounding"):
        sections["grounding"].update({
            "grounding_mean_iou": metrics["grounding"].get("mean_iou"),
            "acc_at_0_5": metrics["grounding"].get("acc_at_0_5"),
            "acc_at_0_7": metrics["grounding"].get("acc_at_0_7"),
        })

    return {
        "overall": overall,
        **sections,
        "environment": collect_environment(summary, run_config),
        "model_config": _model_config(samples, summary, run_config),
        "metadata": {
            "results_dir": _relative(results_dir or DEFAULT_RESULTS_DIR),
            "source_files": [_relative(p) for p in (source_files or [])],
            "sample_count": len(samples),
            "token_count_source": "explicit_inference_fields_only",
            "estimated_word_count_source": "whitespace_split_prediction",
            "note": "Estimated word counts are diagnostics only and must not be reported as tokenizer token counts or token/s.",
        },
    }


def generate_benchmark_report(results_dir: Path | None = None, output_path: Path | None = None) -> dict[str, Any]:
    results_dir = (results_dir or DEFAULT_RESULTS_DIR).resolve()
    output_path = (output_path or DEFAULT_REPORT_PATH).resolve()
    samples, files, metrics = load_samples(results_dir)
    summary = load_json(results_dir / "summary.json") if (results_dir / "summary.json").exists() else {}
    run_config = load_json(results_dir / "run_config.json") if (results_dir / "run_config.json").exists() else {}
    report = build_benchmark_report(samples, results_dir=results_dir, source_files=files, summary=summary, run_config=run_config, metrics=metrics)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Generate unified benchmark report JSON.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = generate_benchmark_report(args.results_dir, args.output)
    print(f"[PASS] Benchmark report saved to {_relative(args.output)}")
    print(f"[PASS] Aggregated {report['metadata']['sample_count']} samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
