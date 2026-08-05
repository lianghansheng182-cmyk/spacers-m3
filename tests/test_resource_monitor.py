import os
import sys
import time
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from src.utils.resource_monitor import ResourceMonitor, monitor_inference, monitor_run


def test_import():
    print("[PASS] ResourceMonitor can be imported")
    assert ResourceMonitor is not None


def test_timing():
    monitor = ResourceMonitor()
    monitor.start()
    time.sleep(0.5)
    monitor.stop()
    assert monitor.latency_s >= 0.5
    print(f"[PASS] Timing works: {monitor.latency_s:.3f}s")


def test_peak_vram():
    monitor = ResourceMonitor()
    monitor.start()

    cuda_works = False
    if torch.cuda.is_available():
        try:
            tensor = torch.randn(1000, 1000, device="cuda")
            time.sleep(0.1)
            del tensor
            torch.cuda.empty_cache()
            cuda_works = True
        except Exception:
            cuda_works = False

    monitor.stop()

    if cuda_works:
        assert monitor.peak_vram_mb > 0.0
        print(f"[PASS] Peak VRAM recorded: {monitor.peak_vram_mb:.2f} MB")
    else:
        assert monitor.peak_vram_mb >= 0.0
        print("[PASS] Peak VRAM read completed without error")


def test_no_gpu_branch_with_mock():
    with mock.patch("src.utils.resource_monitor.torch.cuda.is_available", return_value=False):
        monitor = ResourceMonitor()
        assert monitor.device == "cpu"
        assert monitor.gpu_name == "N/A"

        monitor.start()
        time.sleep(0.05)
        monitor.stop()

        report = monitor.get_report()
        assert report["device"] == "cpu"
        assert report["gpu_name"] == "N/A"
        assert report["peak_vram_mb"] == 0.0
        assert report["latency_s"] >= 0.05

    print("[PASS] CPU-only branch works with mocked CUDA unavailable")


def test_context_manager():
    with monitor_inference() as monitor:
        time.sleep(0.2)

    assert monitor.end_time is not None
    assert monitor.latency_s >= 0.2
    print(f"[PASS] Context manager works: {monitor.latency_s:.3f}s")


def test_context_manager_stops_on_exception():
    monitor_ref = None
    with mock.patch("src.utils.resource_monitor.torch.cuda.is_available", return_value=False):
        try:
            with monitor_inference() as monitor:
                monitor_ref = monitor
                raise RuntimeError("simulated inference failure")
        except RuntimeError:
            pass

    assert monitor_ref is not None
    assert monitor_ref.end_time is not None
    assert monitor_ref.latency_s >= 0.0
    print("[PASS] Context manager stops monitor when an exception is raised")


def test_monitor_run():
    def dummy():
        time.sleep(0.1)
        return 42

    result, report = monitor_run(dummy)
    assert result == 42
    assert "latency_s" in report
    assert report["latency_s"] >= 0.1
    print(f"[PASS] monitor_run wrapper works: {report['latency_s']:.3f}s")


def test_monitor_run_stops_on_exception():
    def failing():
        time.sleep(0.05)
        raise ValueError("simulated failure")

    original_stop = ResourceMonitor.stop
    stop_calls: list[ResourceMonitor] = []

    def tracked_stop(self):
        stop_calls.append(self)
        return original_stop(self)

    with mock.patch("src.utils.resource_monitor.torch.cuda.is_available", return_value=False):
        with mock.patch.object(ResourceMonitor, "stop", tracked_stop):
            try:
                monitor_run(failing)
            except ValueError:
                pass

    assert len(stop_calls) == 1
    assert stop_calls[0].end_time is not None

    print("[PASS] monitor_run stops monitor when wrapped function raises")


if __name__ == "__main__":
    print("=" * 50)
    print("Running Resource Monitor tests")
    print("=" * 50)

    test_import()
    test_timing()
    test_peak_vram()
    test_no_gpu_branch_with_mock()
    test_context_manager()
    test_context_manager_stops_on_exception()
    test_monitor_run()
    test_monitor_run_stops_on_exception()

    print("=" * 50)
    print("[PASS] All tests passed")
    print("=" * 50)
