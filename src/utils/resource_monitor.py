import time
import torch
import gc
from contextlib import contextmanager
from typing import Dict, Any


class ResourceMonitor:
    def __init__(self):
        self.device = self._get_device()
        self.gpu_name = self._get_gpu_name()
        self.start_time = None
        self.end_time = None
        self.peak_vram_mb = 0.0
        self._reset_peak()

    def _get_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _get_gpu_name(self) -> str:
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return "N/A"

    def _reset_peak(self):
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        gc.collect()
        self.peak_vram_mb = 0.0

    def start(self):
        self._reset_peak()
        self.start_time = time.perf_counter()
        self.end_time = None
        return self

    def stop(self):
        self.end_time = time.perf_counter()
        if self.device == "cuda":
            self.peak_vram_mb = self._get_peak_vram_mb()
        return self

    def _get_peak_vram_mb(self) -> float:
        if self.device == "cuda":
            return torch.cuda.max_memory_allocated() / 1024**2
        return 0.0

    @property
    def latency_s(self) -> float:
        if self.start_time is None or self.end_time is None:
            return 0.0
        return round(self.end_time - self.start_time, 6)

    def get_report(self) -> Dict[str, Any]:
        return {
            "device": self.device,
            "gpu_name": self.gpu_name,
            "latency_s": self.latency_s,
            "peak_vram_mb": round(self.peak_vram_mb, 2),
        }


@contextmanager
def monitor_inference():
    monitor = ResourceMonitor()
    monitor.start()
    try:
        yield monitor
    finally:
        monitor.stop()


def monitor_run(func, *args, **kwargs):
    monitor = ResourceMonitor()
    monitor.start()
    result = func(*args, **kwargs)
    monitor.stop()
    return result, monitor.get_report()