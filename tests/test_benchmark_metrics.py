import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.benchmark_metrics import (
    DEFAULT_RESULTS_DIR,
    PROJECT_ROOT,
    TTFT_NOT_IMPLEMENTED,
    build_benchmark_report,
    collect_environment,
    discover_prediction_files,
    generate_benchmark_report,
    load_samples,
    normalize_sample,
    percentile,
    safe_rate,
)
from src.utils.benchmark_metrics import main as benchmark_main
from src.utils.resource_monitor import ResourceMonitor, monitor_run


FIXTURES_DIR = Path(__file__).parent / "fixtures"
PREDICTIONS_FIXTURE = FIXTURES_DIR / "benchmark_predictions.jsonl"
METRICS_FIXTURE = FIXTURES_DIR / "benchmark_metrics.json"
REAL_RESULTS_DIR = PROJECT_ROOT / "results" / "vrsbench_mini" / "qwen3_vl_2b_zero_shot"


def _approx_equal(left: float, right: float, tol: float = 1e-6) -> bool:
    return abs(left - right) <= tol


def test_percentile_calculation():
    values = [1.0, 2.0, 3.0, 4.0, 20.0]
    assert _approx_equal(percentile(values, 95.0), 16.8)
    assert percentile([7.5]) == 7.5
    assert percentile([]) is None


def test_token_speed_divide_by_zero():
    samples = [
        normalize_sample(
            {
                "id": "x1",
                "task": "caption",
                "success": True,
                "latency_s": 0.0,
                "prediction": "a b c",
            }
        )
    ]
    report = build_benchmark_report(samples)
    assert report["overall"]["token_speed_tokens_per_s"] is None
    assert safe_rate(10, 0) is None


def test_empty_results():
    report = build_benchmark_report([])
    assert report["overall"]["sample_count"] == 0
    assert report["overall"]["success_count"] == 0
    assert report["overall"]["failure_count"] == 0
    assert report["overall"]["mean_latency_s"] is None
    assert report["overall"]["p95_latency_s"] is None
    assert report["overall"]["ttft_s"] == TTFT_NOT_IMPLEMENTED


def test_real_batch_predictions():
    if not REAL_RESULTS_DIR.exists():
        return

    samples, _, metrics_payload = load_samples(REAL_RESULTS_DIR)
    report = build_benchmark_report(samples, metrics=metrics_payload)

    assert len(samples) == 30
    assert report["overall"]["success_count"] == 30
    assert report["overall"]["failure_count"] == 0
    assert report["overall"]["mean_latency_s"] is not None
    assert report["overall"]["p95_latency_s"] is not None
    assert report["overall"]["peak_vram_mb"] == 4184.17
    assert report["vqa"]["vqa_accuracy"] == 0.5
    assert _approx_equal(report["grounding"]["grounding_mean_iou"], 0.565188)
    assert report["overall"]["ttft_s"] == TTFT_NOT_IMPLEMENTED
    assert report["overall"]["total_output_tokens"] > 0


def test_fixture_task_sections():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        (results_dir / "predictions.jsonl").write_text(
            PREDICTIONS_FIXTURE.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        (results_dir / "metrics.json").write_text(
            METRICS_FIXTURE.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

        samples, _, metrics_payload = load_samples(results_dir)
        report = build_benchmark_report(samples, metrics=metrics_payload)

    assert report["overall"]["sample_count"] == 6
    assert report["caption"]["sample_count"] == 2
    assert report["vqa"]["sample_count"] == 2
    assert report["grounding"]["sample_count"] == 2
    assert report["vqa"]["vqa_accuracy"] == 0.5
    assert _approx_equal(report["grounding"]["grounding_mean_iou"], 0.675)


def test_normalize_sample_aliases():
    sample = normalize_sample(
        {
            "id": "001",
            "task_type": "vqa",
            "success": False,
            "inference_time_s": 1.25,
            "prediction": "one two",
            "peak_vram": 100.5,
            "error": "boom",
        }
    )
    assert sample.sample_id == "001"
    assert sample.task == "vqa"
    assert sample.latency_s == 1.25
    assert sample.output_tokens == 2
    assert sample.peak_vram_mb == 100.5
    assert sample.status == "failed"


def test_generate_report_from_other_directory():
    results_dir = PROJECT_ROOT / "tests" / "_tmp_benchmark_results"
    output_path = PROJECT_ROOT / "tests" / "_tmp_benchmark_report.json"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "predictions.jsonl").write_text(
        PREDICTIONS_FIXTURE.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (results_dir / "metrics.json").write_text(
        METRICS_FIXTURE.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    original_cwd = Path.cwd()
    os.chdir(PROJECT_ROOT.parent)
    try:
        report = generate_benchmark_report(results_dir, output_path)
    finally:
        os.chdir(original_cwd)
        if output_path.exists():
            output_path.unlink()
        for file_path in results_dir.glob("*"):
            file_path.unlink()
        results_dir.rmdir()

    assert report["metadata"]["sample_count"] == 6
    assert report["metadata"]["results_dir"] == "tests/_tmp_benchmark_results"


def test_project_root_paths_are_stable():
    assert PROJECT_ROOT == Path(__file__).resolve().parents[1]
    assert DEFAULT_RESULTS_DIR == PROJECT_ROOT / "results" / "vrsbench_mini" / "qwen3_vl_2b_zero_shot"


def test_no_gpu_environment_collection():
    with mock.patch("src.utils.benchmark_metrics.torch.cuda.is_available", return_value=False):
        env = collect_environment()
    assert env["cuda_available"] is False


def test_inference_exception_ends_monitor():
    original_stop = ResourceMonitor.stop
    stop_calls = []

    def tracked_stop(self):
        stop_calls.append(self)
        return original_stop(self)

    def failing():
        raise RuntimeError("simulated inference failure")

    with mock.patch("src.utils.resource_monitor.torch.cuda.is_available", return_value=False):
        with mock.patch.object(ResourceMonitor, "stop", tracked_stop):
            try:
                monitor_run(failing)
            except RuntimeError:
                pass
            assert len(stop_calls) == 1
            assert stop_calls[0].end_time is not None


def test_discover_prediction_files_ignores_error_cases():
    with tempfile.TemporaryDirectory() as tmp:
        results_dir = Path(tmp) / "run"
        results_dir.mkdir()
        (results_dir / "predictions.jsonl").write_text("{}\n", encoding="utf-8")
        (results_dir / "error_cases.jsonl").write_text("{}\n", encoding="utf-8")
        (results_dir / "benchmark_report.json").write_text("{}\n", encoding="utf-8")

        files = discover_prediction_files(results_dir)
        assert results_dir / "predictions.jsonl" in files
        assert results_dir / "error_cases.jsonl" not in files


def test_main_cli_writes_report():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        (results_dir / "predictions.jsonl").write_text(
            PREDICTIONS_FIXTURE.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        (results_dir / "metrics.json").write_text(
            METRICS_FIXTURE.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        output_path = tmp_path / "benchmark_report.json"

        exit_code = benchmark_main(
            [
                "--results-dir",
                str(results_dir),
                "--output",
                str(output_path),
            ]
        )

        assert exit_code == 0
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        assert payload["overall"]["sample_count"] == 6


if __name__ == "__main__":
    print("=" * 50)
    print("Running benchmark metrics tests")
    print("=" * 50)

    test_percentile_calculation()
    print("[PASS] P95 calculation")

    test_token_speed_divide_by_zero()
    print("[PASS] Token speed divide-by-zero handling")

    test_empty_results()
    print("[PASS] Empty results handling")

    test_real_batch_predictions()
    print("[PASS] Real batch predictions aggregation")

    test_fixture_task_sections()
    print("[PASS] Task sections and accuracy metrics")

    test_normalize_sample_aliases()
    print("[PASS] Sample field alias normalization")

    test_generate_report_from_other_directory()
    print("[PASS] Path resolution from other directory")

    test_project_root_paths_are_stable()
    print("[PASS] Project root path stability")

    test_no_gpu_environment_collection()
    print("[PASS] No-GPU environment branch")

    test_inference_exception_ends_monitor()
    print("[PASS] Inference exception ends monitor")

    test_discover_prediction_files_ignores_error_cases()
    print("[PASS] Prediction file discovery")

    test_main_cli_writes_report()
    print("[PASS] CLI report generation")

    print("=" * 50)
    print("[PASS] All tests passed")
    print("=" * 50)
