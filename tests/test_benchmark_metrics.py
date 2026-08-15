import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.benchmark_metrics import (
    DEFAULT_RESULTS_DIR,
    PROJECT_ROOT,
    TTFT_NOT_IMPLEMENTED,
    build_benchmark_report,
    collect_environment,
    generate_benchmark_report,
    load_samples,
    normalize_sample,
    percentile,
    safe_rate,
)
from src.utils.benchmark_metrics import main as benchmark_main


FIXTURES_DIR = Path(__file__).parent / "fixtures"
PREDICTIONS_FIXTURE = FIXTURES_DIR / "benchmark_predictions.jsonl"
METRICS_FIXTURE = FIXTURES_DIR / "benchmark_metrics.json"


def _approx_equal(left: float, right: float, tol: float = 1e-6) -> bool:
    return abs(left - right) <= tol


def test_percentile_calculation():
    assert _approx_equal(percentile([1.0, 2.0, 3.0, 4.0, 20.0], 95.0), 16.8)
    assert percentile([7.5]) == 7.5
    assert percentile([]) is None
    assert safe_rate(10, 0) is None


def test_words_are_not_reported_as_tokens():
    sample = normalize_sample(
        {
            "id": "x1",
            "task": "caption",
            "success": True,
            "latency_s": 2.0,
            "prediction": "one two three",
        }
    )
    assert sample.output_tokens is None
    assert sample.estimated_output_words == 3

    report = build_benchmark_report([sample])
    assert report["overall"]["total_output_tokens"] is None
    assert report["overall"]["token_speed_tokens_per_s"] is None
    assert report["overall"]["token_count_coverage"] == 0.0
    assert report["overall"]["estimated_output_words"] == 3
    assert report["overall"]["estimated_words_per_s"] == 1.5


def test_explicit_tokens_are_used_when_available():
    sample = normalize_sample(
        {
            "id": "x1",
            "task": "vqa",
            "success": True,
            "latency_s": 2.0,
            "prediction": "yes",
            "output_tokens": 8,
        }
    )
    report = build_benchmark_report([sample])
    assert report["overall"]["total_output_tokens"] == 8
    assert report["overall"]["mean_output_tokens"] == 8
    assert report["overall"]["token_count_coverage"] == 1.0
    assert report["overall"]["token_speed_tokens_per_s"] == 4.0


def test_environment_uses_recorded_run_metadata():
    summary = {
        "gpu_name": "NVIDIA GeForce RTX 5060 Laptop GPU",
        "torch_version": "2.13.0+cu130",
        "cuda_version": "13.0",
    }
    run_config = {
        "torch_version": "should-not-win",
        "cuda_version": "should-not-win",
    }
    env = collect_environment(summary, run_config)
    assert env["gpu_name"] == "NVIDIA GeForce RTX 5060 Laptop GPU"
    assert env["torch_version"] == "2.13.0+cu130"
    assert env["cuda_version"] == "13.0"
    assert env["cuda_available"] is True
    assert env["source"] == "recorded_run_metadata"
    assert env["os"] is None
    assert env["cpu"] is None


def test_empty_results():
    report = build_benchmark_report([])
    assert report["overall"]["sample_count"] == 0
    assert report["overall"]["mean_latency_s"] is None
    assert report["overall"]["p95_latency_s"] is None
    assert report["overall"]["ttft_s"] == TTFT_NOT_IMPLEMENTED


def test_fixture_task_sections():
    with tempfile.TemporaryDirectory() as tmp:
        results_dir = Path(tmp) / "results"
        results_dir.mkdir()
        (results_dir / "predictions.jsonl").write_text(
            PREDICTIONS_FIXTURE.read_text(encoding="utf-8"), encoding="utf-8"
        )
        (results_dir / "metrics.json").write_text(
            METRICS_FIXTURE.read_text(encoding="utf-8"), encoding="utf-8"
        )
        samples, _, metrics = load_samples(results_dir)
        report = build_benchmark_report(samples, metrics=metrics)

    assert report["overall"]["sample_count"] == 6
    assert report["overall"]["success_count"] == 5
    assert report["caption"]["sample_count"] == 2
    assert report["vqa"]["vqa_accuracy"] == 0.5
    assert _approx_equal(report["grounding"]["grounding_mean_iou"], 0.675)
    assert report["overall"]["total_output_tokens"] is None
    assert report["overall"]["estimated_output_words"] == 14


def test_generate_report_keeps_recorded_environment():
    with tempfile.TemporaryDirectory() as tmp:
        results_dir = Path(tmp) / "results"
        results_dir.mkdir()
        (results_dir / "predictions.jsonl").write_text(
            PREDICTIONS_FIXTURE.read_text(encoding="utf-8"), encoding="utf-8"
        )
        (results_dir / "metrics.json").write_text(
            METRICS_FIXTURE.read_text(encoding="utf-8"), encoding="utf-8"
        )
        (results_dir / "summary.json").write_text(
            json.dumps(
                {
                    "gpu_name": "Recorded GPU",
                    "torch_version": "2.13.0+cu130",
                    "cuda_version": "13.0",
                }
            ),
            encoding="utf-8",
        )
        output = Path(tmp) / "report.json"
        report = generate_benchmark_report(results_dir, output)

    assert report["environment"]["gpu_name"] == "Recorded GPU"
    assert report["environment"]["torch_version"] == "2.13.0+cu130"
    assert report["environment"]["cuda_version"] == "13.0"


def test_project_paths_are_stable():
    assert PROJECT_ROOT == Path(__file__).resolve().parents[1]
    assert DEFAULT_RESULTS_DIR == (
        PROJECT_ROOT / "results" / "vrsbench_mini" / "qwen3_vl_2b_zero_shot"
    )


def test_main_cli_writes_report():
    with tempfile.TemporaryDirectory() as tmp:
        results_dir = Path(tmp) / "results"
        results_dir.mkdir()
        (results_dir / "predictions.jsonl").write_text(
            PREDICTIONS_FIXTURE.read_text(encoding="utf-8"), encoding="utf-8"
        )
        (results_dir / "metrics.json").write_text(
            METRICS_FIXTURE.read_text(encoding="utf-8"), encoding="utf-8"
        )
        output = Path(tmp) / "benchmark_report.json"
        exit_code = benchmark_main(
            ["--results-dir", str(results_dir), "--output", str(output)]
        )
        assert exit_code == 0
        assert json.loads(output.read_text(encoding="utf-8"))["overall"]["sample_count"] == 6


if __name__ == "__main__":
    test_percentile_calculation()
    test_words_are_not_reported_as_tokens()
    test_explicit_tokens_are_used_when_available()
    test_environment_uses_recorded_run_metadata()
    test_empty_results()
    test_fixture_task_sections()
    test_generate_report_keeps_recorded_environment()
    test_project_paths_are_stable()
    test_main_cli_writes_report()
    print("[PASS] All benchmark metric tests passed")
