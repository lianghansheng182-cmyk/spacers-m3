import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import torch
from src.utils.resource_monitor import ResourceMonitor, monitor_inference, monitor_run


def test_import():
    print("? 工具能够正常导入")
    from src.utils.resource_monitor import ResourceMonitor
    assert ResourceMonitor is not None


def test_timing():
    monitor = ResourceMonitor()
    monitor.start()
    time.sleep(0.5)
    monitor.stop()
    assert monitor.latency_s >= 0.5
    print(f"? 能正常开始和停止计时: {monitor.latency_s:.3f}s")


def test_peak_vram():
    monitor = ResourceMonitor()
    monitor.start()

    if torch.cuda.is_available():
        tensor = torch.randn(1000, 1000).cuda()
        time.sleep(0.1)
        del tensor
        torch.cuda.empty_cache()

    monitor.stop()
    print(f"? 峰值显存读取: {monitor.peak_vram_mb:.2f} MB")


def test_no_gpu_crash():
    try:
        monitor = ResourceMonitor()
        monitor.start()
        time.sleep(0.05)
        monitor.stop()
        report = monitor.get_report()
        assert "device" in report
        print("? 没有GPU时不会直接报错")
    except Exception as e:
        print(f"? 错误: {e}")
        raise


def test_context_manager():
    with monitor_inference() as m:
        time.sleep(0.2)
    print(f"? 上下文管理器正常: {m.latency_s:.3f}s")


def test_monitor_run():
    def dummy():
        time.sleep(0.1)
        return 42

    result, report = monitor_run(dummy)
    assert result == 42
    assert "latency_s" in report
    print(f"? monitor_run 包装函数正常: {report['latency_s']:.3f}s")


if __name__ == "__main__":
    print("=" * 50)
    print("运行 Resource Monitor 测试")
    print("=" * 50)

    test_import()
    test_timing()
    test_peak_vram()
    test_no_gpu_crash()
    test_context_manager()
    test_monitor_run()

    print("=" * 50)
    print("所有测试通过 ?")