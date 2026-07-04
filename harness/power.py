from __future__ import annotations
import threading, time
from dataclasses import dataclass

@dataclass
class PowerStats:
    mean_w: float
    peak_w: float
    energy_j: float
    n: int


def summarize_power(samples_w: list[float], dt_s: float) -> PowerStats:
    if not samples_w:
        return PowerStats(0.0, 0.0, 0.0, 0)
    mean_w = sum(samples_w) / len(samples_w)
    return PowerStats(mean_w, max(samples_w), mean_w * len(samples_w) * dt_s, len(samples_w))


class PowerSampler:
    """Background NVML power sampler. Use as a context manager around a workload."""

    def __init__(self, device_index: int = 0, interval_s: float = 0.01) -> None:
        self.device_index = device_index
        self.interval_s = interval_s
        self._samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.stats: PowerStats | None = None

    def _loop(self):
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        while not self._stop.is_set():
            self._samples.append(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0)  # mW -> W
            time.sleep(self.interval_s)
        pynvml.nvmlShutdown()

    def __enter__(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self.stats = summarize_power(self._samples, self.interval_s)
        return False